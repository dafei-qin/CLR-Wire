#!/usr/bin/env python3
"""
分布式训练死锁诊断工具

用法：
1. 在训练脚本中导入并使用装饰器包装可疑的函数
2. 当程序hang住时，查看输出日志找到最后一个被调用的函数
3. 检查该函数是否有条件性的集体通信操作

Example:
    from scripts.diagnose_distributed_hang import trace_dist_call
    
    @trace_dist_call("tokenize_bspline")
    def tokenize_bspline_poles(...):
        ...
"""

import functools
import torch
import torch.distributed as dist
import time
import os
import sys
import traceback

# 全局开关，可以通过环境变量控制
ENABLE_TRACE = os.environ.get("TRACE_DIST_CALLS", "1") == "1"
TRACE_FILE = os.environ.get("TRACE_FILE", None)

def get_rank():
    """获取当前进程的rank"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    """获取world size"""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1

def log_trace(message, rank=None):
    """记录trace信息"""
    if rank is None:
        rank = get_rank()
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [Rank {rank}] {message}"
    
    if TRACE_FILE:
        with open(f"{TRACE_FILE}.rank{rank}", "a") as f:
            f.write(log_msg + "\n")
    else:
        print(log_msg, flush=True)

def trace_dist_call(func_name=None, check_sync=True):
    """
    装饰器：追踪分布式调用
    
    Args:
        func_name: 函数名称（用于日志）
        check_sync: 是否在函数前后进行同步检查
    """
    def decorator(func):
        nonlocal func_name
        if func_name is None:
            func_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not ENABLE_TRACE:
                return func(*args, **kwargs)
            
            rank = get_rank()
            world_size = get_world_size()
            
            # 记录函数调用开始
            log_trace(f">>> ENTER {func_name}", rank)
            
            # 可选：在函数调用前进行barrier检查
            if check_sync and world_size > 1:
                try:
                    log_trace(f"    Pre-barrier check for {func_name}", rank)
                    dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
                    log_trace(f"    Pre-barrier passed for {func_name}", rank)
                except Exception as e:
                    log_trace(f"    Pre-barrier FAILED for {func_name}: {e}", rank)
            
            # 执行函数
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                log_trace(f"<<< EXIT {func_name} (took {elapsed:.3f}s)", rank)
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                log_trace(f"XXX ERROR in {func_name} (after {elapsed:.3f}s): {e}", rank)
                log_trace(f"    Traceback: {traceback.format_exc()}", rank)
                raise
        
        return wrapper
    return decorator

def check_collective_op(op_name, tensor=None):
    """
    检查集体通信操作
    
    在调用任何集体通信前调用此函数，确保所有rank都会执行
    
    Example:
        if condition:
            check_collective_op("all_reduce", my_tensor)
            dist.all_reduce(my_tensor)
        else:
            # 即使条件不满足，也要调用检查
            check_collective_op("all_reduce", dummy_tensor)
            dist.all_reduce(dummy_tensor)
    """
    rank = get_rank()
    world_size = get_world_size()
    
    if world_size <= 1:
        return
    
    log_trace(f"    Preparing collective op: {op_name}", rank)
    
    # 检查所有rank是否都到达这里
    # 使用一个全局计数器（每个rank +1）
    if tensor is not None:
        log_trace(f"    Tensor shape: {tensor.shape}, device: {tensor.device}", rank)

def detect_conditional_fsdp_call(model, method_name="forward"):
    """
    检测条件性FSDP调用
    
    在可能有条件调用的地方使用：
    
    Example:
        if data_is_valid:
            detect_conditional_fsdp_call(model, "forward")
            output = model(input)
    """
    rank = get_rank()
    world_size = get_world_size()
    
    if world_size <= 1:
        return
    
    # 创建一个标志tensor，表示当前rank是否会调用该方法
    will_call = torch.tensor([1], dtype=torch.int32, device=torch.cuda.current_device())
    
    # 收集所有rank的标志
    all_flags = [torch.zeros_like(will_call) for _ in range(world_size)]
    dist.all_gather(all_flags, will_call)
    
    # 检查是否所有rank都会调用
    all_flags_list = [f.item() for f in all_flags]
    if not all(f == 1 for f in all_flags_list):
        log_trace(
            f"⚠️  WARNING: Conditional FSDP call detected for {model.__class__.__name__}.{method_name}",
            rank
        )
        log_trace(f"    Ranks calling: {[i for i, f in enumerate(all_flags_list) if f == 1]}", rank)
        log_trace(f"    Ranks NOT calling: {[i for i, f in enumerate(all_flags_list) if f == 0]}", rank)
        log_trace(f"    This may cause a DEADLOCK!", rank)

def trace_batch_data(batch_info, rank=None):
    """
    记录batch数据信息
    
    Example:
        trace_batch_data({
            "batch_idx": idx,
            "num_valid_patches": patches_valid.shape[0],
            "has_bspline": (tokens == -2).any().item()
        })
    """
    if rank is None:
        rank = get_rank()
    
    info_str = ", ".join(f"{k}={v}" for k, v in batch_info.items())
    log_trace(f"    BATCH: {info_str}", rank)

def sync_print(*args, **kwargs):
    """
    同步打印：确保所有rank都打印后才继续
    """
    rank = get_rank()
    world_size = get_world_size()
    
    # 打印信息
    print(f"[Rank {rank}]", *args, **kwargs, flush=True)
    
    # 同步
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

if __name__ == "__main__":
    print("分布式训练死锁诊断工具")
    print("="*60)
    print("\n使用方法：")
    print("1. 设置环境变量：export TRACE_DIST_CALLS=1")
    print("2. 可选设置trace文件：export TRACE_FILE=/path/to/trace_log")
    print("3. 在代码中使用装饰器：")
    print("   @trace_dist_call('my_function')")
    print("   def my_function(...):")
    print("       ...")
    print("\n4. 运行训练，当hang住时查看日志")
    print("5. 找到最后一个ENTER但没有EXIT的函数")
    print("6. 检查该函数是否有条件性的集体通信操作")
    print("="*60)

