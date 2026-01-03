#!/usr/bin/env python3
"""
批量推理脚本 - 使用数据集载入多个样本进行推理
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import torch
from tqdm.auto import tqdm

# Support running without installing as a package (match training script behavior)
import sys
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import GPT, Config
from lit_gpt.mamba_simple import Mamba
from sft.datasets.serializaitonDEEMOS import deserialize
from sft.datasets.data_utils import to_mesh
from sft.datasets.DatasetDEEMOS import Sample_Dataset
import trimesh
import numpy as np
import torch.utils.data as data
import random


def validate_and_filter_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    验证并过滤 faces，移除包含超出 vertices 范围索引的 face
    
    Args:
        vertices: [N, 3] 形状的顶点数组
        faces: [M, 3] 形状的面数组，包含顶点索引
        
    Returns:
        过滤后的 faces 数组
    """
    num_vertices = len(vertices)
    max_valid_idx = num_vertices - 1
    
    # 检查每个 face 的所有索引是否在有效范围内
    valid_mask = np.all((faces >= 0) & (faces <= max_valid_idx), axis=1)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"    警告: 发现 {invalid_count} 个无效 faces（索引超出范围），已过滤")
    
    filtered_faces = faces[valid_mask]
    
    # 如果过滤后没有有效的 faces，返回空数组
    if len(filtered_faces) == 0:
        print(f"    警告: 过滤后没有有效的 faces，返回空数组")
    
    return filtered_faces


def load_model(ckpt_path: str, model_name: str = "Samba_1.3B", device: str = "cuda", dtype: str = "bf16", 
               use_dynamic_window: bool = True, window_thresholds: list = None) -> GPT:
    """加载模型"""
    config = Config.from_name(model_name)
    config.padded_vocab_size=(2*4**3)+(8**3)+(16**3) +1 +1  #4736+2
    config.block_size = 20000

    train_ctx = 20000
    infer_ctx_max = 50000
    config.block_size = infer_ctx_max

    # 你训练时的 RoPE context
    config.rope_ntk_base_ctx = 20000        # 训练时 block_size
    config.rope_ntk_factor = infer_ctx_max / config.rope_ntk_base_ctx  # 比如 2 / 4
    config.rope_scaling_type = "ntk"        # 打个 flag

    # 原来 RoPE 的 base（你 build_rope_cache 里默认是 10000）
    config.rope_theta_base = 10000.0


    
    # config.condense_ratio = infer_ctx_max / train_ctx  
    print(f"condense_ratio: {config.condense_ratio}")
    config.use_dynamic_window = use_dynamic_window
    if window_thresholds:
        config.window_thresholds = window_thresholds
    model = GPT(config)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)

    if dtype.lower() in ("bf16", "bfloat16"):
        model.to(device=device, dtype=torch.bfloat16)
    elif dtype.lower() in ("fp16", "half"):
        model.to(device=device, dtype=torch.float16)
    else:
        model.to(device=device)
    return model

@torch.no_grad()
def generate_with_kvcache(
    model: GPT,
    start_token_id: int,
    pc: torch.Tensor = None,
    max_new_tokens: int = 50,
    max_seq_length: Optional[int] = None,
    temperature: float = 0.0,
    batch_size: int = 4,      # 新增：并行 batch 个数
    eos_token_id: int = 4737,  # 终止符 ID
):
    """
    批量生成函数：将起始 token 和点云在 batch 维复制 batch_size 份，并行解码。
    返回：长度为 batch_size 的 list，每个元素是对应样本的 token 列表。
    """
    device = next(model.parameters()).device
    block_size = model.config.block_size
    max_seq_length = int(max_seq_length or block_size)

    if hasattr(model, "reset_cache"):
        model.reset_cache()

    # ---- 准备 batch 维度 ----
    # seq: [B, 1]
    seq = torch.full((batch_size, 1), fill_value=start_token_id, dtype=torch.long, device=device)

    # pc: 期望 [B, N, C]，如果传进来是 [1, N, C] 或 [N, C]，统一扩成 [B, N, C]
    if pc is not None:
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)  # [1, N, C]
        if pc.size(0) == 1 and batch_size > 1:
            pc = pc.repeat(batch_size, 1, 1)  # 复制到 batch 维
        elif pc.size(0) != batch_size:
            # 强制匹配 batch 维（更稳妥）
            pc = pc[:1].repeat(batch_size, 1, 1)
    # ---- 第一个 token ----
    # input_pos 形状按模型实现可广播；保持 [1] 即可在大多数实现下广播到 batch
    input_pos = torch.tensor([0], dtype=torch.long, device=device)
    out = model(seq, max_seq_length=max_seq_length, input_pos=input_pos, pc=pc).logits  # [B, 1, V]
    logits = out[:, -1, :]  # [B, V]

    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)  # [B, V]
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
    seq = torch.cat([seq, next_token], dim=1)  # [B, 2]
    
    # 检查第一个 token 是否所有都是终止符
    if torch.all(next_token.squeeze(-1) == eos_token_id):
        return [seq[i].tolist() for i in range(seq.size(0))]

    # ---- 后续 tokens ----
    for t in tqdm(range(1, max_new_tokens), total=max_new_tokens - 1, desc="Decoding", leave=False):
        input_pos = torch.tensor([t], dtype=torch.long, device=device)
        token_in = seq[:, -1:]  # [B, 1]
        out = model(token_in, max_seq_length=max_seq_length, input_pos=input_pos, pc=None).logits  # [B, 1, V]
        logits = out[:, -1, :]  # [B, V]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
        seq = torch.cat([seq, next_token], dim=1)  # [B, T+1]
        
        # 检查是否所有 batch 的下一个 token 都是终止符
        if torch.all(next_token.squeeze(-1) == eos_token_id):
            break

    # seq: [B, T_total]，返回按 batch 切开的 Python list
    return [seq[i].tolist() for i in range(seq.size(0))]


def create_dataloader(data_path: str, point_num: int = 16384, batch_size: int = 1) -> data.DataLoader:
    """创建数据加载器"""
    def collate_as_list(batch):
        out = {}
        for item in batch:
            for k, v in item.items():
                out.setdefault(k, []).append(v)
        return out
    
    dataset = Sample_Dataset(
        point_num=point_num, 
        use_H5=False, 
        use_uid=True,
        path=data_path
    )
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_as_list,
    )
    
    return dataloader

def main():
    parser = argparse.ArgumentParser(description="批量KV-Cache推理 for Samba GPT")
    parser.add_argument("--ckpt", type=str, 
                       default="/deemos-research-area-d/meshgen/code/Samba/out/tsz128x16k_100B_Samba_2.2B_22L/Samba-DEEMOS-11-20-15/iter-018000-ckpt.pth",
                       help="Path to training checkpoint (.pth)")
    parser.add_argument("--model_name", type=str, default="Samba_2.4B_22L")
    parser.add_argument("--start_token", type=int, default=4736)
    parser.add_argument("--max_new_tokens", type=int, default=50000)
    parser.add_argument("--max_seq_len", type=int, default=50000)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    
    # 动态窗口控制参数
    parser.add_argument("--use_dynamic_window", action="store_true", default=False,
                       help="Enable dynamic window control for KV cache")
    parser.add_argument("--no_dynamic_window", action="store_false", dest="use_dynamic_window",
                       help="Disable dynamic window control")
    parser.add_argument("--window_thresholds", type=int, nargs="+", default=[],
                       help="Window thresholds for dynamic window control")
    
    # 数据集相关参数
    parser.add_argument("--data_path", type=str, default="/deemos-research-area-d/meshgen/code/Samba/data/testdata",
                       help="Path to dataset directory")
    parser.add_argument("--point_num", type=int, default=16384,
                       help="Number of points in point cloud")
    parser.add_argument("--num_samples", type=int, default=13,
                       help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=300000,
                       help="Starting index of samples to process")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    print(f"批量推理配置:")
    print(f"  动态窗口控制: {'启用' if args.use_dynamic_window else '禁用'}")
    if args.use_dynamic_window:
        print(f"  窗口阈值: {args.window_thresholds}")
    print(f"  最大序列长度: {args.max_seq_len}")
    print(f"  数据集路径: {args.data_path}")
    print(f"  点云点数: {args.point_num}")
    print(f"  处理样本数: {args.num_samples}")
    print(f"  起始索引: {args.start_idx}")

    # 创建数据加载器
    print("加载数据集...")
    dataloader = create_dataloader(args.data_path, args.point_num, batch_size=1)
    dataset = dataloader.dataset
    print(f"数据集大小: {len(dataset)}")

    # 加载模型
    print("加载模型...")
    model = load_model(
        args.ckpt, 
        model_name=args.model_name, 
        device=args.device, 
        dtype=args.dtype,
        use_dynamic_window=args.use_dynamic_window,
        window_thresholds=args.window_thresholds
    )

    # 创建输出目录
    output_dir = Path("meshes/TestData_1201_NoScaleUp_50k_no_dynamic_window")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理样本
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    print(f"处理样本 {args.start_idx} 到 {end_idx-1}...")
    random.seed(928)

    # 生成 10 个 1~1,500,000 之间的随机整数
    nums = [random.randint(1, 1_500_000) for _ in range(args.num_samples)]

    for idd in range(args.num_samples * 2):
        # idx  = nums[idd]
        idx = idd % args.num_samples
        print(f"\n处理样本 {idx}/{end_idx-1}...")

        # 获取样本数据
        sample_data = dataset[idx]

        # 提取点云数据
        pc = sample_data.get('pc', None)


        if pc is None:
            pc = sample_data.get('pc_normal', None)

        if pc is not None:
            # 统一为 [1, N, C] 并放到设备
            if pc.dim() == 2:
                pc = pc.unsqueeze(0)
            pc = pc.to(args.device)
            print(f"  原始点云形状: {pc.shape}")  # [1, N, C]
        else:
            print(f"  警告: 样本 {idx} 未找到点云数据")
            pc = None

        # 保存单份点云（可选）
        if pc is not None:
            ply_filename = f'{output_dir}/{idd}_pc_sample_{idx}.ply'
            pointcloud = trimesh.points.PointCloud(pc[0].detach().cpu().numpy()[..., :3])
            pointcloud.export(ply_filename)
            print(f"  点云保存到: {ply_filename}")
        target_dtype = next(model.conditioner.parameters()).dtype if hasattr(model, "conditioner") and model.conditioner is not None else next(model.parameters()).dtype
        pc = pc.to(args.device, dtype=target_dtype)
        # ===== 批量推理：把起始 token 和点云在 batch 维复制 4 份，同时解码 =====
        B = 10
        tokens_batch = generate_with_kvcache(
            model,
            start_token_id=args.start_token,
            pc=pc,  # 函数内部会扩成 [B, N, C]
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_len,
            temperature=args.temperature,
            batch_size=B,
        )

        # 生成并保存每个 batch 的 mesh
        for b in range(B):
            tokens = tokens_batch[b]
            
            # 第一个mesh：使用前20000个token（不包括start_token）
            code_20k = tokens[1:20001] if len(tokens) > 20001 else tokens[1:]
            if len(code_20k) > 0 and False:
                vertices_20k, faces_20k = deserialize(code_20k)
                faces_20k = faces_20k.reshape(-1, 3)
                
                # 验证并过滤 faces
                faces_20k = validate_and_filter_faces(vertices_20k, faces_20k)
                
                if len(faces_20k) > 0:
                    mesh_20k = trimesh.Trimesh(vertices=vertices_20k, faces=faces_20k, process=False)
                    output_filename_20k = output_dir / f'{idd}_mesh_sample_{idx}_b{b}_{idd}_20k.obj'
                    mesh_20k.export(str(output_filename_20k))
                    print(f"  [b{b}] 20k tokens mesh: {len(code_20k)} tokens, {vertices_20k.shape[0]} vertices -> {output_filename_20k}")
                else:
                    print(f"  [b{b}] 20k tokens: 跳过，没有有效的 faces")
            
            # 第二个mesh：使用所有token
            code_full = tokens[1:]
            vertices_full, faces_full = deserialize(code_full)
            faces_full = faces_full.reshape(-1, 3)
            
            # 验证并过滤 faces
            faces_full = validate_and_filter_faces(vertices_full, faces_full)
            
            if len(faces_full) > 0:
                mesh_full = trimesh.Trimesh(vertices=vertices_full, faces=faces_full, process=False)
                output_filename_full = output_dir / f'{idd}_mesh_sample_{idx}_b{b}_{idd}_full.obj'
                mesh_full.export(str(output_filename_full))
                print(f"  [b{b}] full tokens mesh: {len(code_full)} tokens, {vertices_full.shape[0]} vertices -> {output_filename_full}")
            else:
                print(f"  [b{b}] full tokens: 跳过，没有有效的 faces")

            


    print(f"\n批量处理完成! 结果保存在 {output_dir}")

if __name__ == "__main__":
    main()

