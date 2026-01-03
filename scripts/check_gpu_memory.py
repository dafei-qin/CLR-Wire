#!/usr/bin/env python3
"""
GPU显存占用诊断工具
用于分析FSDP训练中各个GPU的显存分布

用法：
1. 在训练脚本中导入并在关键位置调用
2. 或作为独立脚本运行查看当前GPU状态

Example:
    from scripts.check_gpu_memory import print_memory_stats, log_module_memory
    
    # 在模型setup后
    print_memory_stats(fabric, "After model setup")
    log_module_memory(model, fabric)
"""

import torch
import torch.distributed as dist
from typing import Optional, Dict
import gc

def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    获取GPU显存信息（MB）
    """
    if device is None:
        device = torch.cuda.current_device()
    
    if isinstance(device, int):
        device_id = device
    elif hasattr(device, 'index'):
        device_id = device.index
    else:
        device_id = 0
    
    torch.cuda.synchronize(device_id)
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**2
    reserved = torch.cuda.memory_reserved(device_id) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**2
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**2
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'max_allocated': max_allocated,
        'total': total,
        'free': total - reserved
    }

def print_memory_stats(fabric=None, tag: str = ""):
    """
    打印当前rank的显存统计
    """
    if fabric is not None:
        rank = fabric.global_rank
        world_size = fabric.world_size
    elif dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    mem_info = get_gpu_memory_info()
    
    msg = (
        f"[Rank {rank}/{world_size}] {tag}\n"
        f"  Allocated: {mem_info['allocated']:.2f} MB\n"
        f"  Reserved:  {mem_info['reserved']:.2f} MB\n"
        f"  Max Alloc: {mem_info['max_allocated']:.2f} MB\n"
        f"  Total:     {mem_info['total']:.2f} MB\n"
        f"  Free:      {mem_info['free']:.2f} MB"
    )
    
    if fabric is not None:
        fabric.print(msg)
    else:
        print(msg, flush=True)
    
    # 收集所有rank的信息
    if world_size > 1:
        all_allocated = [torch.zeros(1, device='cuda') for _ in range(world_size)]
        my_allocated = torch.tensor([mem_info['allocated']], device='cuda')
        dist.all_gather(all_allocated, my_allocated)
        
        if rank == 0:
            all_allocated_list = [t.item() for t in all_allocated]
            print(f"\n📊 Memory Distribution Across GPUs:")
            for i, alloc in enumerate(all_allocated_list):
                print(f"  GPU {i}: {alloc:.2f} MB")
            
            max_mem = max(all_allocated_list)
            min_mem = min(all_allocated_list)
            avg_mem = sum(all_allocated_list) / len(all_allocated_list)
            imbalance = (max_mem - min_mem) / avg_mem * 100 if avg_mem > 0 else 0
            
            print(f"\n  Max: {max_mem:.2f} MB (GPU {all_allocated_list.index(max_mem)})")
            print(f"  Min: {min_mem:.2f} MB (GPU {all_allocated_list.index(min_mem)})")
            print(f"  Avg: {avg_mem:.2f} MB")
            print(f"  Imbalance: {imbalance:.1f}%")
            
            if imbalance > 20:
                print(f"\n⚠️  WARNING: High memory imbalance detected!")
                print(f"     GPU {all_allocated_list.index(max_mem)} has {imbalance:.1f}% more memory than average")

def log_module_memory(model, fabric=None, top_n: int = 10):
    """
    记录模型各个模块的显存占用
    """
    rank = fabric.global_rank if fabric else 0
    
    if rank != 0:
        return
    
    module_memory = {}
    
    for name, module in model.named_modules():
        if name == "":
            name = "root"
        
        param_mem = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)) / 1024**2
        buffer_mem = sum(b.numel() * b.element_size() for b in module.buffers(recurse=False)) / 1024**2
        
        total_mem = param_mem + buffer_mem
        
        if total_mem > 0:
            module_memory[name] = {
                'params': param_mem,
                'buffers': buffer_mem,
                'total': total_mem
            }
    
    # 排序并打印top N
    sorted_modules = sorted(module_memory.items(), key=lambda x: x[1]['total'], reverse=True)
    
    print(f"\n📦 Top {top_n} Largest Modules (MB):")
    print(f"{'Module Name':<50} {'Params':>10} {'Buffers':>10} {'Total':>10}")
    print("-" * 82)
    
    for name, mem in sorted_modules[:top_n]:
        print(f"{name:<50} {mem['params']:>10.2f} {mem['buffers']:>10.2f} {mem['total']:>10.2f}")
    
    # 检查ignored modules
    raw_model = model.module if hasattr(model, 'module') else model
    
    print(f"\n🔍 Checking FSDP ignored modules:")
    if hasattr(raw_model, 'conditioner') and raw_model.conditioner is not None:
        cond_params = sum(p.numel() * p.element_size() for p in raw_model.conditioner.parameters()) / 1024**2
        print(f"  ⚠️  Conditioner: {cond_params:.2f} MB (NOT sharded - full copy on each GPU!)")
    
    if hasattr(raw_model, 'michel') and raw_model.michel is not None:
        michel_params = sum(p.numel() * p.element_size() for p in raw_model.michel.parameters()) / 1024**2
        print(f"  ⚠️  Michelangelo: {michel_params:.2f} MB (NOT sharded - full copy on each GPU!)")

def cleanup_memory():
    """
    清理GPU缓存
    """
    gc.collect()
    torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

if __name__ == "__main__":
    print("=" * 80)
    print("GPU Memory Diagnostic Tool")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        exit(1)
    
    num_gpus = torch.cuda.device_count()
    print(f"\n✅ Found {num_gpus} GPU(s)\n")
    
    for i in range(num_gpus):
        mem_info = get_gpu_memory_info(i)
        print(f"GPU {i}:")
        print(f"  Allocated: {mem_info['allocated']:.2f} MB")
        print(f"  Reserved:  {mem_info['reserved']:.2f} MB")
        print(f"  Total:     {mem_info['total']:.2f} MB")
        print(f"  Free:      {mem_info['free']:.2f} MB")
        print()

