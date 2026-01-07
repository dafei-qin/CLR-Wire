"""
Checkpoint适配工具：处理模型结构变化后的权重加载
用于从旧的reshape方案(256, 1024)迁移到新的直接project方案(4096, 64)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from pathlib import Path


def load_checkpoint_skip_incompatible(
    model: nn.Module, 
    checkpoint_path: str,
    skip_keys: Optional[List[str]] = None,
    verbose: bool = True
) -> nn.Module:
    """
    加载checkpoint，自动跳过不兼容的层
    
    Args:
        model: 目标模型
        checkpoint_path: checkpoint文件路径
        skip_keys: 需要跳过的键列表（默认跳过linear和norm）
        verbose: 是否打印详细信息
    
    Returns:
        加载权重后的模型
    """
    if skip_keys is None:
        # 默认跳过 condition projection 相关的层
        skip_keys = ['linear.weight', 'linear.bias', 'norm.weight', 'norm.bias']
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 如果checkpoint是字典且包含'model'键
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 获取模型当前的 state_dict
    model_state = model.state_dict()
    
    # 过滤掉需要跳过的键和形状不匹配的键
    compatible_state = {}
    incompatible_keys = []
    shape_mismatch_keys = []
    
    for k, v in state_dict.items():
        # 检查是否在跳过列表中
        if any(skip_key in k for skip_key in skip_keys):
            incompatible_keys.append(k)
            continue
        
        # 检查键是否存在于模型中
        if k not in model_state:
            incompatible_keys.append(k)
            continue
        
        # 检查形状是否匹配
        if v.shape != model_state[k].shape:
            shape_mismatch_keys.append(f"{k}: {v.shape} -> {model_state[k].shape}")
            continue
        
        compatible_state[k] = v
    
    # 加载兼容的权重
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"📦 Checkpoint加载报告")
        print(f"{'='*60}")
        print(f"✅ 成功加载: {len(compatible_state)} 个参数")
        print(f"⚠️  跳过(在skip_keys中): {len(incompatible_keys)} 个")
        print(f"⚠️  形状不匹配: {len(shape_mismatch_keys)} 个")
        print(f"❓ 缺失的键: {len(missing_keys)} 个")
        print(f"❓ 多余的键: {len(unexpected_keys)} 个")
        
        if incompatible_keys and len(incompatible_keys) <= 10:
            print(f"\n跳过的键: {incompatible_keys}")
        
        if shape_mismatch_keys:
            print(f"\n形状不匹配的键:")
            for key_info in shape_mismatch_keys[:5]:
                print(f"  - {key_info}")
            if len(shape_mismatch_keys) > 5:
                print(f"  ... 还有 {len(shape_mismatch_keys) - 5} 个")
        
        print(f"\n🔄 未加载的层将保持随机初始化状态")
        print(f"💡 建议：先用大学习率训练这些层，再进行端到端微调")
        print(f"{'='*60}\n")
    
    return model


def create_staged_optimizer(
    model: nn.Module,
    stage: str = "warmup",
    warmup_lr: float = 1e-3,
    finetune_lr: float = 1e-5,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """
    创建分阶段的优化器
    
    Args:
        model: 模型
        stage: "warmup" 或 "finetune"
        warmup_lr: warmup阶段的学习率（只训练新初始化的层）
        finetune_lr: finetune阶段的学习率（训练所有层）
        weight_decay: 权重衰减
    
    Returns:
        优化器
    """
    if stage == "warmup":
        # 第一阶段：只训练 linear 和 norm 层
        print("\n🔥 Warmup阶段：只训练 condition projection 层")
        trainable_params = []
        frozen_params = 0
        
        for name, param in model.named_parameters():
            if 'linear' in name or 'norm' in name:
                param.requires_grad = True
                trainable_params.append(param)
                print(f"  ✓ {name}: 可训练")
            else:
                param.requires_grad = False
                frozen_params += 1
        
        print(f"\n📊 可训练参数: {len(trainable_params)}")
        print(f"📊 冻结参数: {frozen_params}")
        print(f"📊 学习率: {warmup_lr}\n")
        
        return torch.optim.AdamW(trainable_params, lr=warmup_lr, weight_decay=weight_decay)
    
    elif stage == "finetune":
        # 第二阶段：训练所有层
        print("\n🔥 Finetune阶段：训练所有层")
        
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
        
        # 可以对不同层使用不同的学习率
        param_groups = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if 'linear' in n or 'norm' in n],
                'lr': finetune_lr * 2,  # condition层用稍大的学习率
                'name': 'condition_projection'
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if 'linear' not in n and 'norm' not in n],
                'lr': finetune_lr,
                'name': 'backbone'
            }
        ]
        
        print(f"📊 Condition层学习率: {finetune_lr * 2}")
        print(f"📊 主干网络学习率: {finetune_lr}\n")
        
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'warmup' or 'finetune'")


def save_training_config(save_path: str, config: Dict):
    """保存训练配置，方便追踪实验"""
    import json
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 训练配置已保存到: {save_path}")


if __name__ == "__main__":
    # 使用示例
    print("""
    使用示例：
    
    # 1. 加载旧checkpoint，跳过不兼容的层
    from lit_gpt.model import GPT
    from lit_gpt.config import Config
    from lit_gpt.checkpoint_adapter import load_checkpoint_skip_incompatible, create_staged_optimizer
    
    config = Config.from_name("your_config")
    model = GPT(config)
    
    # 加载checkpoint
    model = load_checkpoint_skip_incompatible(
        model, 
        "path/to/old_checkpoint.pth",
        verbose=True
    )
    
    # 2. 第一阶段：Warmup训练（1-2个epoch）
    optimizer_warmup = create_staged_optimizer(model, stage="warmup", warmup_lr=1e-3)
    
    for epoch in range(2):
        # 训练循环...
        pass
    
    # 3. 第二阶段：端到端微调
    optimizer_finetune = create_staged_optimizer(model, stage="finetune", finetune_lr=1e-5)
    
    for epoch in range(remaining_epochs):
        # 训练循环...
        pass
    """)

