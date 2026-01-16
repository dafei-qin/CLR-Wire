# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import glob
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
# from sft.datasets.DatasetDEEMOS import Sample_Dataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops
from lit_gpt.utils import chunked_cross_entropy, num_parameters
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
from sft.datasets.serializaitonDEEMOS import deserialize
import random
import os
from datetime import datetime
import numpy as np
import trimesh
import warnings
import einops


# Dafei's import

sys.path.insert(0, str(Path(__file__).parent))
# Add project root to sys.path to import src.utils
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.utils.import_tools import load_dataset_from_config, load_model_from_config
from src.utils.gpt_tools import tokenize_bspline_poles
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", message="When using.*NO_SHARD.*")

# ========== 训练参数（默认值，将从 YAML 配置文件中读取） ==========
# 这些变量将在 setup() 函数中从 config_dict.trainer 读取并更新
# model_name = "Diff_LLaMA_551M"  # 默认值
# train_config = "HY1024_tsz128x16k_100B_ScaleUp20k_unlockCondition"  # 默认值
# name = None  # 将在 setup 中计算
# out_dir = None  # 将在 setup 中设置
# devices = torch.cuda.device_count() or 1
# use_sample_dataset = True
# freeze_conditioner = False
# conditioner_lr_scale = 1.0
# fsdp_state_dict_type = "full"
# max_tokens = 1e9
# global_batch_size = 32
# micro_batch_size = 8
# learning_rate = 1e-4
# total_evals = 400
# warmup_tokens = None  # 将在 setup 中计算
# log_step_interval = 10
# save_step_interval = 2500
# eval_step_interval = 100000000000000
# num_extrapol = 4
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.95
# grad_clip = 1.0
# decay_lr = True
# min_lr = 1e-5
# num_epochs = 20
# batch_size = None  # 将在 setup 中计算
# gradient_accumulation_steps = None  # 将在 setup 中计算
# log_iter_interval = None  # 将在 setup 中计算

# hparams 将在 setup 函数中从配置读取后创建
hparams = {}




# ---------------------------
# Chamfer Distance 计算函数（优化版本）
# ---------------------------
def sample_points_from_mesh(vertices, faces, num_samples=1024):
    """
    从 mesh 表面随机采样点
    
    Args:
        vertices: (N, 3) numpy array
        faces: (M, 3) numpy array
        num_samples: 采样点数
    
    Returns:
        sampled_points: (num_samples, 3) numpy array
    """
    try:
        # 创建 trimesh 对象
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # 从表面采样点
        sampled_points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        
        return sampled_points
    except Exception as e:
        # 如果采样失败，直接使用顶点并随机采样
        if len(vertices) >= num_samples:
            indices = np.random.choice(len(vertices), num_samples, replace=False)
            return vertices[indices]
        else:
            # 顶点不够，重复采样
            indices = np.random.choice(len(vertices), num_samples, replace=True)
            return vertices[indices]


def compute_chamfer_distance_fast(pred_points, gt_points):
    """
    快速计算两个点云之间的 Chamfer Distance
    
    Args:
        pred_points: (N, 3) numpy array 或 torch tensor
        gt_points: (M, 3) numpy array 或 torch tensor
    
    Returns:
        chamfer_dist: 双向 Chamfer Distance
    """
    try:
        # 转换为 torch tensor（在 GPU 上）
        if not torch.is_tensor(pred_points):
            pred_points = torch.from_numpy(pred_points).float()
        if not torch.is_tensor(gt_points):
            gt_points = torch.from_numpy(gt_points).float()
        
        # 确保在 GPU 上
        if not pred_points.is_cuda:
            pred_points = pred_points.cuda()
        if not gt_points.is_cuda:
            gt_points = gt_points.cuda()
        
        # 计算双向 Chamfer Distance
        # pred -> gt
        dist_matrix = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)  # (N, M)
        min_dist_pred_to_gt = dist_matrix.min(dim=1)[0]  # (N,)
        
        # gt -> pred
        min_dist_gt_to_pred = dist_matrix.min(dim=0)[0]  # (M,)
        
        # Chamfer Distance (双向平均)
        chamfer_dist = (min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()).item()
        
        return chamfer_dist
    
    except Exception as e:
        return float('inf')


def validate_and_filter_faces(vertices, faces):
    """验证并过滤 faces，移除包含超出 vertices 范围索引的 face"""
    if len(vertices) == 0 or len(faces) == 0:
        return np.array([])
    
    num_vertices = len(vertices)
    max_valid_idx = num_vertices - 1
    valid_mask = np.all((faces >= 0) & (faces <= max_valid_idx), axis=1)
    
    filtered_faces = faces[valid_mask]
    return filtered_faces


def tokens_to_mesh_with_sampling(tokens, pad_id=4737, num_samples=1024):
    """
    将 token 序列解码为 mesh，并从表面采样点
    
    Args:
        tokens: token 序列 (numpy array 或 torch tensor)
        pad_id: padding token id
        num_samples: 采样点数
    
    Returns:
        sampled_points: (num_samples, 3) numpy array，如果解码失败返回 None
    """
    try:
        # 转为 numpy
        if torch.is_tensor(tokens):
            tokens = tokens.detach().cpu().numpy()
        
        # 移除 padding
        tokens = tokens[tokens != pad_id]
        
        if len(tokens) == 0:
            return None
        
        # 解码为 mesh
        vertices, faces = deserialize(tokens)
        
        if len(vertices) == 0:
            return None
        
        # 过滤无效的 faces
        faces = faces.reshape(-1, 3)
        faces = validate_and_filter_faces(vertices, faces)
        
        
        if len(faces) == 0:
            # 没有有效的 face，直接使用顶点
            if len(vertices) >= num_samples:
                indices = np.random.choice(len(vertices), num_samples, replace=False)
                return vertices[indices]
            else:
                indices = np.random.choice(len(vertices), num_samples, replace=True)
                return vertices[indices]
        
        # 从表面采样点
        sampled_points = sample_points_from_mesh(vertices, faces, num_samples)
        
        return sampled_points
    
    except Exception as e:
        return None

# ---------------------------
# 新增：严格搬运模块（含未注册 tensor）
# ---------------------------
def move_module_strict(module: torch.nn.Module, device: torch.device, dtype: torch.dtype | None = None):
    """把 module 的参数、buffers 以及未注册的裸 tensor 属性都搬到 device/dtype。"""
    module.to(device=device, dtype=dtype)
    for sub in module.modules():
        param_names = set(n for n, _ in sub.named_parameters(recurse=False))
        buffer_names = set(n for n, _ in sub.named_buffers(recurse=False))
        for name, value in vars(sub).items():
            if name in param_names or name in buffer_names:
                continue
            if isinstance(value, torch.Tensor):
                setattr(sub, name, value.to(device=device, dtype=(dtype or value.dtype)))


# ---------------------------
# Conditioner Checkpoint 辅助函数（统一保存/加载）
# ---------------------------
def save_checkpoint_with_conditioner(fabric, checkpoint_path: Path, state: dict) -> None:
    """
    保存 checkpoint，包含主模型和 conditioner。
    由于 conditioner 在 FSDP ignored_modules 中，需要手动保存。
    """
    model = state["model"]
    raw_model = model.module if hasattr(model, 'module') else model
    
    # 在保存前，手动将 conditioner 的 state_dict 加入到 state 中
    # if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
    #     # 只在 rank 0 准备 conditioner state
    #     if fabric.global_rank == 0:
    #         state['conditioner_state_dict'] = raw_model.conditioner.state_dict()
    #         state['freeze_conditioner'] = freeze_conditioner
    
    # 保存完整的 state（包含 conditioner）
    fabric.save(checkpoint_path, {key: value for key, value in state.items() if key != 'vae'})
    
    # 清理临时添加的 key
    # if 'conditioner_state_dict' in state:
    #     del state['conditioner_state_dict']
    # if 'freeze_conditioner' in state:
    #     del state['freeze_conditioner']
    
    if fabric.global_rank == 0:
        fabric.print(f"💾 Checkpoint saved to {str(checkpoint_path)!r}")
        if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
            fabric.print(f"   ✅ Conditioner included in checkpoint")


def load_checkpoint_with_conditioner(fabric, checkpoint_path: Path, state: dict, model) -> None:
    """
    加载 checkpoint，包含主模型和 conditioner。
    """
    # 加载 checkpoint
    state = {key:value for key, value in state.items() if key != 'vae'}
    fabric.load(checkpoint_path, state)
    return state
    


def setup(
    config_path: Optional[str] = None,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    resume: Union[bool, Path] = False,
    warm_start_ckpt: Optional[Path] = None, 
) -> None:
    # ========== 加载配置文件（如果提供） ==========
    config_dict = None
    if config_path is not None and Path(config_path).exists():
        config_dict = OmegaConf.load(config_path)
        print(f"📂 Loaded config from: {config_path}")
    elif config_path is not None:
        print(f"⚠️  Config file not found: {config_path}, using default settings")
    
    # ========== 从配置文件读取训练参数 ==========
    global model_name, train_config, train_name, name, out_dir, devices, num_nodes, total_gpus, use_sample_dataset
    global freeze_conditioner, conditioner_lr_scale, fsdp_state_dict_type
    global max_tokens, global_batch_size, micro_batch_size, learning_rate
    global total_evals, warmup_tokens, log_step_interval, save_step_interval
    global eval_step_interval, num_extrapol, weight_decay, beta1, beta2
    global grad_clip, decay_lr, min_lr, num_epochs, batch_size
    global gradient_accumulation_steps, log_iter_interval
    global find_unused_parameters, uncond_prob
    
    if config_dict is not None and "trainer" in config_dict:
        trainer_cfg = config_dict.trainer
        print("📋 Loading trainer parameters from config...")
        
        # 读取所有训练参数
        if "model_name" in trainer_cfg:
            model_name = trainer_cfg.model_name
        if "train_config" in trainer_cfg:
            train_config = trainer_cfg.train_config
        if 'train_name' in trainer_cfg:
            train_name = trainer_cfg.train_name
        if "out_dir" in trainer_cfg:
            out_dir = Path(trainer_cfg.out_dir)
        if "use_sample_dataset" in trainer_cfg:
            use_sample_dataset = trainer_cfg.use_sample_dataset
        if "freeze_conditioner" in trainer_cfg:
            freeze_conditioner = trainer_cfg.freeze_conditioner
        if "conditioner_lr_scale" in trainer_cfg:
            conditioner_lr_scale = trainer_cfg.conditioner_lr_scale
        if "fsdp_state_dict_type" in trainer_cfg:
            fsdp_state_dict_type = trainer_cfg.fsdp_state_dict_type
        if "max_tokens" in trainer_cfg:
            max_tokens = float(trainer_cfg.max_tokens)
        if "global_batch_size" in trainer_cfg:
            global_batch_size = trainer_cfg.global_batch_size
        if "micro_batch_size" in trainer_cfg:
            micro_batch_size = trainer_cfg.micro_batch_size
        if "learning_rate" in trainer_cfg:
            learning_rate = float(trainer_cfg.learning_rate)
        if "total_evals" in trainer_cfg:
            total_evals = trainer_cfg.total_evals
        if "warmup_tokens" in trainer_cfg and trainer_cfg.warmup_tokens is not None:
            warmup_tokens = int(trainer_cfg.warmup_tokens)
        elif "warmup_tokens" not in trainer_cfg or trainer_cfg.warmup_tokens is None:
            # 如果没有设置或为 null，则计算
            warmup_tokens = int(max_tokens * 0.05)
        if "log_step_interval" in trainer_cfg:
            log_step_interval = trainer_cfg.log_step_interval
        if "save_step_interval" in trainer_cfg:
            save_step_interval = trainer_cfg.save_step_interval
        if "eval_step_interval" in trainer_cfg:
            eval_step_interval = trainer_cfg.eval_step_interval
        if "num_extrapol" in trainer_cfg:
            num_extrapol = trainer_cfg.num_extrapol
        if "weight_decay" in trainer_cfg:
            weight_decay = float(trainer_cfg.weight_decay)
        if "beta1" in trainer_cfg:
            beta1 = trainer_cfg.beta1
        if "beta2" in trainer_cfg:
            beta2 = trainer_cfg.beta2
        if "grad_clip" in trainer_cfg:
            grad_clip = trainer_cfg.grad_clip
        if "decay_lr" in trainer_cfg:
            decay_lr = trainer_cfg.decay_lr
        if "min_lr" in trainer_cfg:
            min_lr = float(trainer_cfg.min_lr)
        if "num_epochs" in trainer_cfg:
            num_epochs = trainer_cfg.num_epochs
        if "find_unused_parameters" in trainer_cfg:
            find_unused_parameters = trainer_cfg.find_unused_parameters
        else:
            find_unused_parameters = False
        if "uncond_prob" in trainer_cfg:
            uncond_prob = float(trainer_cfg.uncond_prob)
        else:
            uncond_prob = 0.0  # 默认不做 unconditioned 训练
        print(f"   ✓ Loaded {len([k for k in trainer_cfg.keys()])} trainer parameters")
    else:
        print("⚠️  No 'trainer' section in config, using default values")
        # 使用默认值计算
        warmup_tokens = int(max_tokens * 0.05)
        uncond_prob = 0.0  # 默认不做 unconditioned 训练
    
    # ========== 计算派生参数 ==========
    # 计算 name（如果未设置）

    name = train_config + "_" +  train_name + "_" + model_name
    
    # 计算 batch_size 和 gradient_accumulation_steps
    # 🔥 修复：从环境变量读取节点数，避免 world_size 不匹配
    devices = torch.cuda.device_count() or 1
    num_nodes = int(os.environ.get("MLP_WORKER_NUM", 1))  # 从环境变量读取节点数
    total_gpus = devices * num_nodes  # 总 GPU 数 = 每节点GPU数 × 节点数
    
    batch_size = global_batch_size // total_gpus  # 使用总 GPU 数计算
    gradient_accumulation_steps = batch_size // micro_batch_size
    assert gradient_accumulation_steps > 0, f"gradient_accumulation_steps must be > 0, got {gradient_accumulation_steps}"
    
    # 计算 log_iter_interval
    log_iter_interval = log_step_interval
    
    # 确保 out_dir 已设置
    if out_dir is None:
        # 如果没有在配置中设置，使用默认路径
        out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name / f"Samba-DEEMOS-{datetime.now().strftime('%m-%d-%H')}"
        out_dir = Path(out_dir)
    
    print(f"📊 Training configuration:")
    print(f"   - Model: {model_name}, Config: {train_config}")
    print(f"   - Output dir: {out_dir}")
    print(f"   - Nodes: {num_nodes}, GPUs per node: {devices}, Total GPUs: {total_gpus}")
    print(f"   - Batch size per GPU: {batch_size}, Micro batch: {micro_batch_size}")
    print(f"   - Global batch size: {global_batch_size}")
    print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   - Learning rate: {learning_rate}, Max tokens: {max_tokens}")
    print(f"   - Warmup tokens: {warmup_tokens}")
    print(f"   - Uncond prob: {uncond_prob} (CFG training)")
    
    # ========== 创建 hparams 字典（用于保存 checkpoint） ==========
    global hparams
    # 收集所有训练相关的超参数
    hparams = {
        "model_name": model_name,
        "train_config": train_config,
        "train_name": train_name,
        "name": name,
        "out_dir": str(out_dir),
        "num_nodes": num_nodes,
        "devices": devices,
        "total_gpus": total_gpus,
        "use_sample_dataset": use_sample_dataset,
        "freeze_conditioner": freeze_conditioner,
        "conditioner_lr_scale": conditioner_lr_scale,
        "fsdp_state_dict_type": fsdp_state_dict_type,
        "max_tokens": max_tokens,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "learning_rate": learning_rate,
        "total_evals": total_evals,
        "warmup_tokens": warmup_tokens,
        "log_step_interval": log_step_interval,
        "save_step_interval": save_step_interval,
        "eval_step_interval": eval_step_interval,
        "num_extrapol": num_extrapol,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "grad_clip": grad_clip,
        "decay_lr": decay_lr,
        "min_lr": min_lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "log_iter_interval": log_iter_interval,
        "find_unused_parameters": find_unused_parameters,
        "uncond_prob": uncond_prob,
    }
    
    wandb_logger = WandbLogger(project=train_config, name=name)

    # ========== 加载模型 ==========
    if config_dict is not None and "model" in config_dict:
        # 使用配置文件加载模型
        print("📦 Loading model from config...")
        # print(config_dict.model.params.config)
        # exit()
        # 将 OmegaConf 对象转换为普通字典，避免 Literal 类型注解验证错误
        config_params = OmegaConf.to_container(config_dict.model.params.config, resolve=True)
        config_obj = Config(**config_params)
        # 将整个 config_dict 转换为普通字典，避免 OmegaConf 类型验证
        config_dict_plain = OmegaConf.to_container(config_dict, resolve=True)
        config_dict_plain["model"]["params"]["config"] = config_obj
        model = load_model_from_config(config_dict_plain, device='cpu', strict=False)
        vae = load_model_from_config(config_dict, device = 'cpu', section='vae')
        

    if warm_start_ckpt is not None:
        try:
            ckpt = torch.load(warm_start_ckpt, map_location="cpu")
            model_state = ckpt.get("model", ckpt)
            missing, unexpected = model.load_state_dict(model_state, strict=False)
            print(f"Warm-start loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"Warm-start failed: {e}")

    # ========== 🔥 修复：避免将大型模块放入 ignored_modules ==========
    # ignored_modules 中的模块不会被 FSDP 分片，每个 GPU 都会保留完整副本
    # 这会导致 GPU 0（通常负责协调）显存占用过高
    # 
    # 修复策略：
    # 1. 如果 michel/conditioner 很大，考虑让 FSDP 分片它们
    # 2. 如果必须 ignore，至少要知道每张卡都会占用额外显存
    # 
    # 临时方案：继续 ignore michel（因为它来自外部库），但添加警告
    ignored = []
    michel_module = getattr(model, "michel", None)
    if isinstance(michel_module, torch.nn.Module):
        ignored.append(michel_module)
        if fabric.global_rank == 0 if hasattr(locals().get('fabric'), 'global_rank') else True:
            michel_params = sum(p.numel() for p in michel_module.parameters()) / 1e6
            print(f"⚠️  WARNING: 'michel' module ({michel_params:.1f}M params) in ignored_modules")
            print(f"   Each GPU will hold a FULL copy → {michel_params:.1f}M params × {total_gpus} GPUs")
            print(f"   Consider reducing model size or not using ignored_modules for large modules")
    
    # 不要 ignore conditioner - 让 FSDP 分片它
    # （如果遇到问题，可以手动添加）

    # 4) 创建 FSDPStrategy（初始化时就传入 ignored_modules）
    from torch.distributed.fsdp import BackwardPrefetch
    
    # strategy = FSDPStrategy(
    #     auto_wrap_policy={Block},
    #     state_dict_type=fsdp_state_dict_type,  # 可配置：full（慢但兼容）或 sharded（快）
    #     ignored_modules=ignored if ignored else None,  # 如果为空则传 None
    #     use_orig_params=True,
    #     # cpu_offload=True,  # 可选：如果显存紧张，考虑开启
    #     # limit_all_gathers=True,  # 🔥 优化：限制 all-gather 操作，减少显存峰值
    #     # activation_checkpointing_policy={Block},  # 🔥 新增：activation checkpointing，减少50%激活值显存
    #     # backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # 🔥 优化：预取梯度，减少显存峰值
    # )
    strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)

    # 5) 创建 Fabric 并 launch
    fabric = L.Fabric(
        devices=devices,
        num_nodes=num_nodes,  # 🔥 修复：传入节点数，避免 world_size 不匹配
        strategy=strategy,
        precision="bf16-mixed",  # 🔥 启用混合精度：显存减半，速度提升
        # precision="32",  # 如果bf16有问题，可以尝试 "16-mixed"
        loggers=[wandb_logger],
    )
    fabric.launch()

    # 8) 进入主流程
    main(fabric, model, vae, config_dict, train_data_dir, val_data_dir, resume)

def main(fabric, model, vae, config_dict, train_data_dir, val_data_dir, resume, **overides):
    monitor = Monitor(fabric, window_size=1, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(42)
    # 这里不再 from_name/重新建模了，直接用传进来的 model
    config = model.config

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        fabric=fabric,
        config_dict=config_dict,
        seed=42,
    )

    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    

    fabric.print(f"Loading model with {config.__dict__}")
    fabric.print(f"Total parameters {num_parameters(model):,}")
    if hasattr(model, "conditioner") and model.conditioner is not None:
        fabric.print(f"Conditioner parameters {num_parameters(model.conditioner):,}")
    if hasattr(model, "michel") and model.michel is not None:
        fabric.print(f"Michel parameters {num_parameters(model.michel):,}")
    fabric.print(model)
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(idx, name, param.shape)

    # 统一由 Fabric/FSDP 搬到各自 rank 的设备
    model = fabric.setup(model)
    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'michel') and raw_model.michel is not None:
        move_module_strict(raw_model.michel, fabric.device)
        use_michelangelo_grad = config_dict['trainer']['use_michelangelo_grad']
        if use_michelangelo_grad:
            for param in raw_model.michel.parameters():
                param.requires_grad = True
        else:
            for param in raw_model.michel.parameters():
                param.requires_grad = False

        fabric.print(f"✅ michel module moved to {fabric.device}")
        fabric.print(f"✅ michel module grad set to {use_michelangelo_grad}")
    vae = fabric.setup(vae)
    # freeze vae
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    # 标记 VAE 的 encode 方法为 forward 方法，以便 FSDP 正确处理
    # 根据 Lightning Fabric 的要求，需要在 setup 后标记自定义的 forward 方法
    vae.mark_forward_method('encode')

    # ========== 构建优化器（区分 conditioner 和主干的学习率） ==========
    # if not freeze_conditioner and hasattr(model, "conditioner") and model.conditioner is not None:
    #     # 获取原始模型（处理 FSDP 包装）
    #     raw_model = model.module if hasattr(model, 'module') else model
        
    #     # 分离 conditioner 参数和其他参数
    #     conditioner_params = []
    #     other_params = []
    #     conditioner_param_ids = set(id(p) for p in raw_model.conditioner.parameters())
        
    #     for name, param in raw_model.named_parameters():
    #         if param.requires_grad:
    #             if id(param) in conditioner_param_ids:
    #                 conditioner_params.append(param)
    #             else:
    #                 other_params.append(param)
        
    #     # 使用不同学习率的参数组
    #     param_groups = [
    #         {"params": other_params, "lr": learning_rate},
    #         {"params": conditioner_params, "lr": learning_rate * conditioner_lr_scale, "name": "conditioner"},
    #     ]
        
    #     fabric.print(f"🔧 Optimizer setup with separate learning rates:")
    #     fabric.print(f"   - Main model params: {len(other_params)}, lr={learning_rate}")
    #     fabric.print(f"   - Conditioner params: {len(conditioner_params)}, lr={learning_rate * conditioner_lr_scale}")
        
    #     optimizer = torch.optim.AdamW(
    #         param_groups, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    #     )
    # else:
        # 原有逻辑：所有参数使用相同学习率
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "vae": vae,  "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0, "epoch": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"\n{'='*60}")
        fabric.print(f"📥 Resuming training from {resume}")
        fabric.print(f"   FSDP state_dict_type: {fsdp_state_dict_type}")
        fabric.print(f"   Current freeze_conditioner: {freeze_conditioner}")
        fabric.print(f"{'='*60}")
        
        # 记录加载时间
        t0 = time.perf_counter()
        resume_path = Path(resume) if not isinstance(resume, Path) else resume
        
        # ========== 智能加载：处理 optimizer 参数组不匹配 ==========
        skip_optimizer = False
        if fabric.global_rank == 0:
            try:
                ckpt_preview = torch.load(resume_path, map_location='cpu', weights_only=False)
                if 'optimizer' in ckpt_preview:
                    saved_param_groups = len(ckpt_preview['optimizer'].get('param_groups', []))
                    current_param_groups = len(optimizer.param_groups)
                    if saved_param_groups != current_param_groups:
                        fabric.print(f"⚠️  Optimizer param groups mismatch: saved={saved_param_groups}, current={current_param_groups}")
                        fabric.print(f"   This is normal when switching freeze_conditioner setting.")
                        fabric.print(f"   Will skip optimizer state and re-initialize.")
                        skip_optimizer = True
                del ckpt_preview
            except Exception as e:
                fabric.print(f"⚠️  Could not preview checkpoint: {e}")
        
        # 广播 skip_optimizer 给所有 rank
        if fabric.world_size > 1:
            skip_tensor = torch.tensor([skip_optimizer], dtype=torch.int32, device=fabric.device)
            torch.distributed.broadcast(skip_tensor, src=0)
            skip_optimizer = bool(skip_tensor.item())
        
        if skip_optimizer:
            # 只加载模型权重，不加载 optimizer
            state_model_only = {"model": model, "iter_num": 0, "step_count": 0, "epoch": 0}
            state_load = load_checkpoint_with_conditioner(fabric, resume_path, state_model_only, model)
            state_load['vae'] = state['vae']
            state = state_load
            
            # 恢复训练状态
            state["iter_num"] = state_model_only.get("iter_num", 0)
            state["step_count"] = state_model_only.get("step_count", 0)
            state["epoch"] = state_model_only.get("epoch", 0)
            
            fabric.print(f"✅ Model weights loaded, optimizer re-initialized")
            fabric.print(f"   Resumed from iter={state['iter_num']}, step={state['step_count']}, epoch={state['epoch']}")
        else:
            # 正常加载（包含 optimizer

            state_load = load_checkpoint_with_conditioner(fabric, resume_path, state, model)
            state_load['vae'] = state['vae']
            state = state_load
            fabric.print(f"✅ Full checkpoint loaded (model + optimizer + conditioner)")
        print(f"state['iter_num'] = {state['iter_num']}, state['step_count'] = {state['step_count']}, state['epoch'] = {state['epoch']}")
        
        t1 = time.perf_counter()
        fabric.print(f"⏱️  Total resume time: {t1 - t0:.2f}s")
        fabric.print(f"{'='*60}\n")
        
        fabric.barrier()  # 确保所有 rank 同步

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")






def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    vae = state["vae"]
    optimizer = state["optimizer"]

    # if val_dataloader is not None:
    #     validate(fabric, model, val_dataloader)  # sanity check

    # ------- 仍然在 meta 上估 FLOPs，但关闭 conditioner ------- ??
    # 为什么关闭conditioner?
    # Set to true now
    with torch.device("meta"):
        meta_model = GPT(model.config, build_conditioner=True)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        del meta_model, x
    # ------------------------------------------------------

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step()

    # 使用 epoch 计数的调度参数
    steps_per_epoch = len(train_dataloader)
    total_iters = steps_per_epoch * max(1, num_epochs)
    max_iters = total_iters
    warmup_iters = max(1, int(0.01 * total_iters)) if decay_lr else 0
    initial_iter = state["iter_num"]

    loss_func = FusedCrossEntropyLoss()
    
    # 🔥 训练前诊断：检查显存分布和 condition 流程
    if state["iter_num"] == 0:
        # 检查显存分布
        if fabric.world_size > 1:
            fabric.barrier()
            torch.cuda.synchronize()
            
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            reserved_mb = torch.cuda.memory_reserved() / 1024**2
            
            # 收集所有rank的显存信息
            allocated_tensor = torch.tensor([allocated_mb], device=fabric.device)
            reserved_tensor = torch.tensor([reserved_mb], device=fabric.device)
            
            all_allocated = [torch.zeros_like(allocated_tensor) for _ in range(fabric.world_size)]
            all_reserved = [torch.zeros_like(reserved_tensor) for _ in range(fabric.world_size)]
            
            torch.distributed.all_gather(all_allocated, allocated_tensor)
            torch.distributed.all_gather(all_reserved, reserved_tensor)
            
            if fabric.global_rank == 0:
                fabric.print("\n" + "="*60)
                fabric.print("📊 GPU Memory Distribution (MB)")
                fabric.print("="*60)
                
                all_alloc_list = [t.item() for t in all_allocated]
                all_resv_list = [t.item() for t in all_reserved]
                
                for i, (alloc, resv) in enumerate(zip(all_alloc_list, all_resv_list)):
                    fabric.print(f"  GPU {i}: Allocated {alloc:>8.1f} MB | Reserved {resv:>8.1f} MB")
                
                max_alloc = max(all_alloc_list)
                min_alloc = min(all_alloc_list)
                avg_alloc = sum(all_alloc_list) / len(all_alloc_list)
                imbalance = (max_alloc - min_alloc) / avg_alloc * 100 if avg_alloc > 0 else 0
                
                fabric.print(f"\n  Max: {max_alloc:.1f} MB (GPU {all_alloc_list.index(max_alloc)})")
                fabric.print(f"  Min: {min_alloc:.1f} MB (GPU {all_alloc_list.index(min_alloc)})")
                fabric.print(f"  Avg: {avg_alloc:.1f} MB")
                fabric.print(f"  Imbalance: {imbalance:.1f}%")
                
                if imbalance > 20:
                    fabric.print(f"\n⚠️  WARNING: High memory imbalance ({imbalance:.1f}%)!")
                    fabric.print(f"     GPU {all_alloc_list.index(max_alloc)} uses {max_alloc - avg_alloc:.1f} MB more than average")
                    fabric.print(f"     Check if large modules are in FSDP ignored_modules")
                
                fabric.print("="*60 + "\n")
            
            fabric.barrier()
    
    # 训练前诊断：检查 condition 流程
    if fabric.global_rank == 0 and state["iter_num"] == 0:
        fabric.print("\n" + "="*60)
        fabric.print("🔍 Condition Pipeline Diagnostic Check")
        fabric.print("="*60)
        raw_model = model.module if hasattr(model, 'module') else model
        if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
            is_frozen = not any(p.requires_grad for p in raw_model.conditioner.parameters())
            fabric.print("✅ Conditioner found (ShapeVAE from config.yaml)")
            fabric.print(f"   - Status: {'🔒 FROZEN' if is_frozen else '🔓 TRAINABLE'}")
            fabric.print(f"   - Mode: {'eval' if not raw_model.conditioner.training else 'train'}")
            fabric.print(f"   - Dtype: {next(raw_model.conditioner.parameters()).dtype}")
            fabric.print(f"   - Device: {next(raw_model.conditioner.parameters()).device}")
            
            if not is_frozen:
                cond_params = sum(p.numel() for p in raw_model.conditioner.parameters() if p.requires_grad)
                fabric.print(f"   - Trainable params: {cond_params:,}")
                fabric.print(f"   - Learning rate scale: {conditioner_lr_scale}x")
            
            # 验证 ShapeVAE 的编解码流程
            fabric.print(f"\n📋 ShapeVAE Architecture:")
            fabric.print(f"   - Latent shape: {raw_model.conditioner.latent_shape}")
            fabric.print(f"   - Encoder: PointCrossAttentionEncoder")
            fabric.print(f"   - Decoder: post_kl + Transformer")
            fabric.print(f"   - Expected workflow: pc → encode() → latent_codes → decode() → features")
            
            # 检查各组件的 dtype
            fabric.print(f"\n🔧 Component Dtypes:")
            if hasattr(raw_model.conditioner, 'encoder'):
                fabric.print(f"   - Encoder: {next(raw_model.conditioner.encoder.parameters()).dtype}")
            if hasattr(raw_model.conditioner, 'pre_kl'):
                fabric.print(f"   - Pre-KL: {raw_model.conditioner.pre_kl.weight.dtype}")
            if hasattr(raw_model.conditioner, 'post_kl'):
                fabric.print(f"   - Post-KL: {raw_model.conditioner.post_kl.weight.dtype}")
            if hasattr(raw_model.conditioner, 'transformer'):
                fabric.print(f"   - Transformer: {next(raw_model.conditioner.transformer.parameters()).dtype}")
            
            if hasattr(raw_model, 'linear'):
                fabric.print(f"\n✅ Linear projection layer found")
                fabric.print(f"   - input_dim: {raw_model.linear.in_features}")
                fabric.print(f"   - output_dim: {raw_model.linear.out_features}")
                fabric.print(f"   - dtype: {raw_model.linear.weight.dtype}")
                fabric.print(f"   - trainable: {raw_model.linear.weight.requires_grad}")
            
            # 统计 cross-attention 层数量
            cross_attn_count = sum(1 for name, _ in raw_model.named_modules() if 'cross_attn' in name)
            fabric.print(f"\n✅ Found {cross_attn_count} CrossAttention layers")
            
            # 根据降采样配置计算期望的 token 数
            num_latents = raw_model.conditioner.latent_shape[0]
            if hasattr(raw_model, 'condition_downsample_factor'):
                expected_tokens = num_latents // raw_model.condition_downsample_factor
                fabric.print(f"   Expected condition tokens: {expected_tokens} ({num_latents} → downsample by {raw_model.condition_downsample_factor}x)")
                if hasattr(raw_model, 'condition_downsample') and raw_model.condition_downsample is not None:
                    fabric.print(f"   Downsample method: learnable (MLP)")
                else:
                    fabric.print(f"   Downsample method: average pooling")
            else:
                fabric.print(f"   Expected condition tokens: {num_latents} (no downsampling)")
        else:
            fabric.print("⚠️  WARNING: No conditioner found!")
        fabric.print("="*60 + "\n")
    
    # 以 epoch 为单位训练
    for epoch in range(state.get("epoch", 0), max(1, num_epochs)):
        try:
            sampler = getattr(train_dataloader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
        except Exception:
            pass
        idddx = 0
        for train_data in train_dataloader:
            train_data = [_t[train_data[-1]] for _t in train_data[:-1]]
            points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask = train_data




            # First, tokenize the bspline poles
            all_tokens_padded = tokenize_bspline_poles(vae, train_dataloader.dataset, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask)

            

            idddx += 1
            if state["iter_num"] >= max_iters:
                break

            # determine and set the learning rate for this iteration
            lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate

            for param_group in optimizer.param_groups:
                # # 如果是 conditioner 参数组，使用缩放的学习率
                # if param_group.get("name") == "conditioner":
                #     param_group["lr"] = lr * conditioner_lr_scale
                # else:
                    param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            pc = torch.cat([points, normals], dim=-1).to(fabric.device)
            

            all_tokens_padded = all_tokens_padded.to(fabric.device)
            lengths = train_dataloader.dataset.max_tokens - (all_tokens_padded == train_dataloader.dataset.pad_id).sum(dim=1)
            lengths = torch.tensor(lengths, device=fabric.device, dtype=torch.long)
            maxL = max(lengths)
            minL = min(lengths)

            merged_token_tensor = all_tokens_padded


            input_token = merged_token_tensor[:, :-1].contiguous()
            target_token = merged_token_tensor[:, 1:].contiguous()
            batch_size, seq_len = target_token.shape
            
            # 计算位置 mask：每个样本的有效 target 长度 = 原始长度 - 1
            valid_lens = (lengths - 1).clamp(min=1, max=seq_len)  # (B,)
            pos = torch.arange(seq_len, device=fabric.device).unsqueeze(0)  # (1, T)
            pad_mask = (pos < valid_lens.unsqueeze(1)).to(torch.float32)  # (B, T), True 表示有效位置


            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
            
            # 🔥 Classifier-Free Guidance (CFG) 训练：随机 drop conditioning
            # 根据 uncond_prob 概率决定是否使用 conditioning
            use_conditioning = True
            if uncond_prob > 0 and random.random() < uncond_prob:
                use_conditioning = False
            
            # 监控 condition embeddings 统计信息（每 500 步）
            # monitor_condition = (state["iter_num"] % 500 == 0) and fabric.global_rank == 0
            
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_token, pc=pc, window_size=9000, use_conditioning=use_conditioning).logits

                
                # 使用位置 mask 计算 loss（不依赖 token id）
                # logits: (B, T, vocab_size), target_token: (B, T)
                per_token_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),  # (B*T, vocab_size)
                    target_token.reshape(-1),  # (B*T,)
                    reduction='none'
                ).view(batch_size, seq_len)  # (B, T)
                
                # 只对有效位置计算 loss（使用位置 mask）
                masked_loss = per_token_loss * pad_mask  # (B, T)
                per_sample_loss = masked_loss.sum(dim=1) / valid_lens.to(masked_loss.dtype)  # (B,)
                loss = per_sample_loss.mean()  # scalar
                
                with torch.no_grad():
                    pred_tokens = torch.argmax(logits, dim=-1)
                    acc = (pred_tokens == target_token) * pad_mask
                    acc_per_sample = acc.sum(dim=1) / valid_lens.to(acc.dtype)
                    acc = acc.mean()

                
                fabric.backward(loss / gradient_accumulation_steps)

            if not is_accumulating:
                # ========== 同步 conditioner 梯度（如果解锁训练） ==========
                # 由于 conditioner 在 FSDP 的 ignored_modules 中，多 GPU 时梯度不会自动同步
                # 需要手动进行 all-reduce
                # if not freeze_conditioner and fabric.world_size > 1:
                #     raw_model = model.module if hasattr(model, 'module') else model
                #     if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
                #         for param in raw_model.conditioner.parameters():
                #             if param.grad is not None:
                #                 # all-reduce 梯度并取平均
                #                 torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                #                 param.grad.div_(fabric.world_size)
                
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                
                # 监控 condition 相关层的梯度（每 100 步）
                # 🔥 修复：减少 rank 0 的显存开销 - 只监控关键参数，不遍历所有参数
                # if state["step_count"] % 100 == 0 and fabric.global_rank == 0:
                #     condition_grad_stats = {}
                #     cross_attn_grads = []
                #     linear_grads = []
                    
                #     # 🔥 优化：不要遍历所有参数，只监控指定的关键模块
                #     raw_model = model.module if hasattr(model, 'module') else model
                    
                #     # 只监控 conditioner 相关的参数（如果存在）
                #     if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
                #         for name, param in raw_model.conditioner.named_parameters():
                #             if param.grad is not None:
                #                 # 避免调用 .norm() 创建临时tensor，直接计算
                #                 with torch.no_grad():
                #                     grad_norm = param.grad.detach().norm().item()
                #                 if 'weight' in name:
                #                     # 只记录少数代表性的层
                #                     if 'encoder' in name or 'transformer' in name:
                #                         condition_grad_stats[f"grad/conditioner_{name.split('.')[0]}"] = grad_norm
                    
                #     # 监控 linear projection 层（如果存在）
                #     if hasattr(raw_model, 'linear'):
                #         if raw_model.linear.weight.grad is not None:
                #             with torch.no_grad():
                #                 grad_norm = raw_model.linear.weight.grad.detach().norm().item()
                #             condition_grad_stats["grad/linear_weight"] = grad_norm
                    
                #     if condition_grad_stats:
                #         fabric.print(f"\n[Condition Gradient Monitor @ step {state['step_count']}]")
                #         for key, value in condition_grad_stats.items():
                #             fabric.print(f"  {key}: {value:.6f}")
                        
                #         # 记录到 wandb
                #         fabric.log_dict(condition_grad_stats, state["step_count"])

                if state["step_count"] % 100 == 0 and fabric.global_rank == 0:
                    condition_grad_stats = {}
                    cross_attn_grads = []
                    linear_grads = []
                    
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            # 监控 cross-attention 层
                            if 'cross_attn' in name:
                                # 只记录权重层的梯度，跳过具体参数名以保持简洁
                                if 'weight' in name:
                                    cross_attn_grads.append(grad_norm)
                                    layer_type = name.split('.')[-2]  # q_proj, kv_proj, out_proj
                                    condition_grad_stats[f"grad/cross_attn_{layer_type}"] = grad_norm
                            # 监控 condition projection 层
                            elif name.endswith('linear.weight') or name.endswith('linear.bias'):
                                linear_grads.append(grad_norm)
                                condition_grad_stats[f"grad/{name.split('.')[-1]}"] = grad_norm
                    
                    if condition_grad_stats:
                        # 计算统计量
                        if cross_attn_grads:
                            condition_grad_stats["grad/cross_attn_mean"] = sum(cross_attn_grads) / len(cross_attn_grads)
                            condition_grad_stats["grad/cross_attn_max"] = max(cross_attn_grads)
                        if linear_grads:
                            condition_grad_stats["grad/linear_mean"] = sum(linear_grads) / len(linear_grads)
                        
                        fabric.print(f"\n[Condition Gradient Monitor @ step {state['step_count']}]")
                        if cross_attn_grads:
                            fabric.print(f"  CrossAttention: mean={condition_grad_stats['grad/cross_attn_mean']:.6f}, "
                                       f"max={condition_grad_stats['grad/cross_attn_max']:.6f}, "
                                       f"count={len(cross_attn_grads)}")
                        if linear_grads:
                            fabric.print(f"  Linear projection: mean={condition_grad_stats['grad/linear_mean']:.6f}")
                        
                        # 记录到 wandb
                        fabric.log_dict(condition_grad_stats, state["step_count"])
                
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
            elif fabric.device.type == "xla":
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            state["iter_num"] += 1


            total_lengths += input_token.size(1)
            t1 = time.perf_counter()
            elapsed_iters = max(1, state['iter_num'] - initial_iter)
            remaining_hours = (t1 - total_t0) / elapsed_iters * max(0, (max_iters - state['iter_num'])) / 3600
            
            # 打印训练信息（包含 conditioning 状态）
            cond_status = "✓" if use_conditioning else "✗"
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, acc {acc_per_sample.mean().item():.4f}, "
                f"cond {cond_status}, "
                f"iter: {idddx}/{len(train_dataloader)} , epoch: {state['epoch']}, gap: {maxL-minL}, lr: {lr}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                f" remaining time: {remaining_hours:.2f} hours. "
                f" or {remaining_hours / 24:.2f} days. "
            )

            monitor.on_train_batch_end(
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                fabric.world_size,
                state["step_count"],
                flops_per_batch=estimated_flops,
                lengths=total_lengths,
                train_loss = loss.item(),
                lr = lr,
                FWLoss = 0.0,
                cd_loss = 0.0,
            )


            if not is_accumulating and state["step_count"] % save_step_interval == 0:
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                save_checkpoint_with_conditioner(fabric, checkpoint_path, state)
                fabric.barrier()  # 确保所有 rank 同步

            if state["iter_num"] >= max_iters:
                break

        state["epoch"] = epoch + 1
        if state["iter_num"] >= max_iters:
            break

    # ========== 训练结束，保存最终 checkpoint ==========
    final_checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-final-ckpt.pth"
    fabric.print(f"\n🏁 Training finished! Saving final checkpoint to {str(final_checkpoint_path)!r}")
    save_checkpoint_with_conditioner(fabric, final_checkpoint_path, state)
    fabric.barrier()


def create_dataloaders(
    batch_size: int,
    fabric,
    config_dict: Optional[dict] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        batch_size: 批次大小
        fabric: Lightning Fabric 实例
        config_dict: 配置字典（OmegaConf 格式），如果为 None 则使用默认配置
        seed: 随机种子
    
    Returns:
        train_dataloader, val_dataloader
    """
    # def collate_as_list(batch):
    #     out = {}
    #     for item in batch:
    #         for k, v in item.items():
    #             out.setdefault(k, []).append(v)
    #     return out

    # ========== 加载数据集 ==========
    if config_dict is not None:
        # 使用配置文件加载数据集
        print("📂 Loading datasets from config...")
        train_dataset = load_dataset_from_config(config_dict, section="data_train")
        # try:
        #     val_dataset = load_dataset_from_config(config_dict, section="data_val")
        #     print('✅ Validation dataset loaded')
        # except (ValueError, KeyError) as e:
        #     print(f'⚠️  No validation dataset found in config: {e}')
        #     val_dataset = None
    else:
        # 向后兼容：使用旧的硬编码方式
        print("⚠️  No config provided, using legacy dataset loading...")
        # 这里可以保留旧的逻辑，或者抛出错误
        raise ValueError(
            "config_dict is required. Please provide a config file with 'data_train' section. "
            "Example: python pretrain.py --config_path path/to/config.yaml"
        )

    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=fabric.world_size,
        rank=fabric.global_rank,
        shuffle=True,
        drop_last=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=8,  # 增加 worker 数量（原来是4）
        pin_memory=True,
        # collate_fn=collate_as_list,
        sampler=sampler,
        prefetch_factor=8,  # 每个 worker 预取4个批次（默认是2）
        persistent_workers=True,  
    )

    # 返回数据加载器（验证集可能为 None）
    return train_dataloader, None

# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI
    CLI(setup)
