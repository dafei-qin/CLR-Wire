"""
Sample script with trained conditioner support.
从 checkpoint 中加载训练过的 conditioner 权重进行推理。
"""
import argparse
import os
import torch
from tqdm import tqdm
from lit_gpt.model_cache import GPTCache, Config
from safetensors.torch import load_file
from sft.datasets.DatasetDEEMOS import Sample_Dataset
from sft.datasets.serializaitonDEEMOS import deserialize
import numpy as np
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime
from pathlib import Path
import trimesh


def validate_and_filter_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    验证并过滤 faces，移除包含超出 vertices 范围索引的 face
    """
    num_vertices = len(vertices)
    max_valid_idx = num_vertices - 1
    valid_mask = np.all((faces >= 0) & (faces <= max_valid_idx), axis=1)
    
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"    警告: 发现 {invalid_count} 个无效 faces（索引超出范围），已过滤")
    
    filtered_faces = faces[valid_mask]
    if len(filtered_faces) == 0:
        print(f"    警告: 过滤后没有有效的 faces，返回空数组")
    
    return filtered_faces


def setup_distributed_mode(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed_mode():
    dist.destroy_process_group()


def add_gumbel_noise(logits, temperature):
    '''
    As suggested by https://arxiv.org/pdf/2409.02908, we use float64 for the gumbel max method.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def load_checkpoint_with_conditioner(model: torch.nn.Module, checkpoint_path: Path, local_rank: int = 0) -> bool:
    """
    从统一的 checkpoint 中加载主模型和 conditioner 权重。
    
    Args:
        model: GPTCache 模型
        checkpoint_path: checkpoint 路径
        local_rank: 当前 GPU rank
        
    Returns:
        True 如果成功加载 conditioner，False 如果没有找到
    """
    checkpoint_path = Path(checkpoint_path)
    
    try:
        if local_rank == 0:
            print(f"📂 Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 提取主模型 state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model_state = checkpoint["model"]
        else:
            model_state = checkpoint
        
        # 加载主模型权重（strict=False 允许缺失/多余的键）
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        
        if local_rank == 0:
            print(f"✅ Main model loaded")
            if missing:
                print(f"   Missing keys: {len(missing)}")
            if unexpected:
                print(f"   Unexpected keys: {len(unexpected)}")
        
        # 尝试加载 conditioner（从统一 checkpoint 中）
        conditioner_loaded = False
        if 'conditioner_state_dict' in checkpoint:
            if local_rank == 0:
                print(f"📂 Loading conditioner from unified checkpoint...")
            
            conditioner_state = checkpoint['conditioner_state_dict']
            saved_freeze = checkpoint.get('freeze_conditioner', True)
        
        # 获取原始模型（处理 DDP 包装）
        raw_model = model.module if hasattr(model, 'module') else model
        
        if raw_model.conditioner is not None:
                raw_model.conditioner.load_state_dict(conditioner_state)
                conditioner_loaded = True
            
                if local_rank == 0:
                    print(f"✅ Conditioner loaded from unified checkpoint!")
                    print(f"   └── Was trained with freeze_conditioner={saved_freeze}")
        else:
            if local_rank == 0:
                print(f"⚠️  No conditioner found in checkpoint")
                print(f"   Using default initialization (pretrained or random)")
        
        return conditioner_loaded
    except Exception as e:
        if local_rank == 0:
            print(f"⚠️  Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()
        return False


@torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, temperature=0.5,
                      context_length=90000, window_size=9000, device='cuda',
                      output_path=None, local_rank=None, i=None):
    gpt.eval()
    N = prompt.shape[0]
    end_list = [0 for _ in range(N)]
    
    with tqdm(total=context_length-1, desc="Processing", disable=local_rank != 0) as pbar:
        for cur_pos in range(prompt.shape[1], context_length):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if cur_pos >= 9001 and (cur_pos - 9001) % 4500 == 0:
                    start = 4500 + ((cur_pos - 9001) // 4500) * 4500
                else:
                    start = cur_pos - 1
                input_pos = torch.arange(cur_pos, dtype=torch.long, device=device)
                prompt_input = prompt[:, start:cur_pos]
                logits = gpt(prompt_input, pc=pc, start=start, window_size=window_size, input_pos=input_pos)[:, -1]
                pc = None  # 只在第一次使用 pc

            logits_with_noise = add_gumbel_noise(logits, temperature)
            next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)

            prompt = torch.cat([prompt, next_token], dim=-1)

            pbar.set_description(f"start:{start}, cur_pos:{cur_pos}, length:{prompt_input.size(1)}")
            pbar.update(1)

            for u in range(N):
                if end_list[u] == 0:
                    if next_token[u] == torch.tensor([4737], device=device):
                        end_list[u] = 1
            if sum(end_list) == N:
                break
                
    return prompt, cur_pos


def first(it):
    return it[0]


def custom_collate(data, pad_id):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first=True, padding_value=pad_id)
        else:
            datum = list(datum)
        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


def build_dataloader_func(bs, dataset, local_rank, world_size):
    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=local_rank, shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=0,
        drop_last=False,
        collate_fn=partial(custom_collate, pad_id=4737)
    )
    return dataloader


@torch.inference_mode()
def get_model_answers(local_rank, world_size, args):
    model_path = args.model_path
    model_id = args.model_id
    steps = args.steps
    temperature = args.temperature
    base_output_path = args.output_path
    point_num = args.point_num
    repeat_num = args.repeat_num
    load_trained_conditioner = args.load_trained_conditioner

    setup_distributed_mode(local_rank, world_size)
    
    # 创建基于时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_output_path, timestamp)
    
    # ========== 模型配置 ==========
    model_name = f"Diff_LLaMA_{model_id}M"
    config = Config.from_name(model_name)
    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"🚀 Sample with Conditioner")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {model_path}")
        print(f"Load trained conditioner: {load_trained_conditioner}")
        print(config)
    
    config.padded_vocab_size = (2*4**3) + (8**3) + (16**3) + 1 + 1  # 4736+2
    config.block_size = 270000
    
    # ========== 创建模型 ==========
    model = GPTCache(config, build_conditioner=True).to('cuda')
    
    # ========== 加载统一 checkpoint（包含主模型和 conditioner）==========
    if local_rank == 0:
        print(f"\n📂 Loading unified checkpoint...")
    
    conditioner_loaded = load_checkpoint_with_conditioner(model, model_path, local_rank)
    
    # ========== 设置 conditioner 为 eval 模式 ==========
    if model.conditioner is not None:
        model.conditioner.eval()
        for p in model.conditioner.parameters():
            p.requires_grad = False
        if local_rank == 0:
            print(f"✅ Conditioner set to eval mode")
    
    # ========== DDP 包装 ==========
    model = DDP(model, device_ids=[local_rank])
    
    if local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        print(f"\n📁 Output directory: {output_path}")
        print(f"{'='*60}\n")
    
    # 同步所有进程
    dist.barrier()
    os.makedirs(output_path, exist_ok=True)

    # ========== 数据集 ==========
    dataset = Sample_Dataset(point_num=point_num, use_uid=False, use_H5=False)
    dataloader = build_dataloader_func(1, dataset, local_rank, world_size)
    
    # ========== 推理循环 ==========
    for i, test_batch in tqdm(enumerate(dataloader), disable=local_rank != 0):
        cond_pc = test_batch['pc'].to('cuda')
        if local_rank == 0:
            print(f"\n[Sample {i}] Point cloud shape: {cond_pc.shape}")
        
        # 保存点云
        points = cond_pc[0].cpu().numpy()
        point_cloud = trimesh.points.PointCloud(points[..., 0:3])
        point_cloud.export(f'{output_path}/{local_rank}_{i}_pc.ply')
        
        # 采样
        output_ids, _ = ar_sample_kvcache(
            model,
            prompt=torch.tensor([[4736]]).to('cuda').repeat(repeat_num, 1),
            pc=cond_pc.repeat(repeat_num, 1, 1),
            window_size=9000,
            temperature=temperature,
            context_length=steps,
            device='cuda',
            output_path=output_path,
            local_rank=local_rank,
            i=i
        )
        
        # 处理输出
        for u in range(repeat_num):
            code = output_ids[u][1:]
            index = (code >= 4737).nonzero()
            if index.numel() > 0:
                code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
            else:
                code = code.cpu().numpy().astype(np.int64)
            
            vertices, faces = deserialize(code)
            faces = faces.reshape(-1, 3)
            faces = validate_and_filter_faces(vertices, faces)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            mesh.export(f'{output_path}/{local_rank}_{i}_{u}_mesh.ply')
            
            if local_rank == 0:
                print(f"   └── Saved mesh {u}: {len(vertices)} vertices, {len(faces)} faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample with trained conditioner support")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the checkpoint (.pth file). "
             "Conditioner will be loaded from the unified checkpoint if available.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="2121",
        help="Model ID (e.g., 2121 for Diff_LLaMA_2121M)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Maximum sampling steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./mesh_output_conditioner',
        help="Base output directory",
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=4,
        help="Number of samples to generate per input",
    )
    parser.add_argument(
        "--point_num",
        type=int,
        default=81920,
        help="Number of points in point cloud",
    )
    parser.add_argument(
        "--load_trained_conditioner",
        action="store_true",
        default=False,
        help="此参数已弃用。Conditioner 现在直接从主 checkpoint 中加载（如果存在）。",
    )
    
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    get_model_answers(
        local_rank=local_rank,
        world_size=world_size,
        args=args
    )

