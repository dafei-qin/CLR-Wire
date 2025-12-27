import argparse
import os
import torch
from tqdm import tqdm
from lit_gpt.model_cache import GPTCache, Config
from safetensors.torch import load_file
from sft.datasets.DatasetDEEMOS import Sample_Dataset
import os
from tqdm import tqdm
import trimesh
from sft.datasets.serializaitonDEEMOS import deserialize
# from sft.datasets.data_utils import to_mesh
import numpy as np
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence
from functools import partial
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import datetime

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

@ torch.no_grad()
def ar_sample_kvcache(gpt, prompt, pc, temperature=0.5, \
                        context_length=90000, window_size=9000,device='cuda',\
                        output_path=None,local_rank=None,i=None):
    gpt.eval()
    N        = prompt.shape[0]
    end_list = [0 for _ in range(N)]
    with tqdm(total=context_length-1, desc="Processing") as pbar:
        for cur_pos in range(prompt.shape[1], context_length):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if cur_pos >= 9001 and (cur_pos - 9001)%4500 == 0:
                    start = 4500 + ((cur_pos - 9001)//4500)*4500
                else:
                    start = cur_pos-1
                input_pos    = torch.arange(cur_pos, dtype=torch.long, device=device)
                prompt_input = prompt[:, start:cur_pos]
                logits = gpt(prompt_input, pc=pc,start = start,window_size=window_size, input_pos=input_pos)[:, -1]
                pc     = None

            logits_with_noise = add_gumbel_noise(logits, temperature)
            next_token = torch.argmax(logits_with_noise, dim=-1, keepdim=True)

            prompt = torch.cat([prompt, next_token], dim=-1)

            pbar.set_description(f"with start:{start},cur_pos:{cur_pos},length:{prompt_input.size(1)}")
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
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output

def build_dataloader_func(bs, dataset, local_rank, world_size):
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=bs,
        num_workers=0,
        drop_last = False,
        collate_fn = partial(custom_collate, pad_id = 4737)
    )
    return dataloader

@torch.inference_mode()
def get_model_answers(
    local_rank,
    world_size
):
    model_path  = args.model_path
    model_id    = args.model_id
    steps       = args.steps
    temperature = args.temperature
    path        = args.input_path
    base_output_path = args.output_path
    point_num   = args.point_num
    uid_list    = args.uid_list.split(",")
    repeat_num  = args.repeat_num

    setup_distributed_mode(local_rank, world_size)
    
    # 创建基于时间戳的输出目录（分布式设置后创建，确保同步）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_output_path, timestamp)
    model_name = f"Diff_LLaMA_{model_id}M"
    config = Config.from_name(model_name)
    print(config)
    config.padded_vocab_size=(2*4**3)+(8**3)+(16**3) +1 +1  #4736+2
    config.block_size = 270000
    model = GPTCache(config).to('cuda')
    file_ext = model_path.split(".")[-1].lower()
    if file_ext == "safetensors":
        loaded_state = load_file(model_path)
    elif file_ext == "bin":
        loaded_state = torch.load(model_path, map_location='cpu', weights_only=False)
    elif file_ext == "pth":
        # .pth 文件可能是完整的 checkpoint（包含 model, optimizer 等）或直接的 state_dict
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            loaded_state = checkpoint["model"]  # 从 checkpoint 中提取 model state_dict
        else:
            loaded_state = checkpoint  # 直接是 state_dict
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: safetensors, bin, pth")
    model.load_state_dict(loaded_state, strict=False)
    model       = DDP(model, device_ids=[local_rank])
    if local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        print(f"输出目录: {output_path}")
    
    # 同步所有进程，确保目录已创建
    dist.barrier()
    
    # 所有进程都需要创建目录（以防万一）
    os.makedirs(output_path, exist_ok=True)

    dataset    = Sample_Dataset(point_num = point_num,use_uid=False,use_H5=False)
    dataloader = build_dataloader_func(1, dataset, local_rank, world_size)
    
    for i, test_batch in tqdm(enumerate(dataloader)):
        cond_pc = test_batch['pc'].to('cuda')
        print(cond_pc.shape)
        points = cond_pc[0].cpu().numpy()
        point_cloud = trimesh.points.PointCloud(points[..., 0:3])
        point_cloud.export(f'{output_path}/{local_rank}_{i}_pc.ply')
        
        output_ids, _ = ar_sample_kvcache(model,
                                prompt = torch.tensor([[4736]]).to('cuda').repeat(repeat_num,1),
                                pc = cond_pc.repeat(repeat_num,1,1),
                                window_size=9000,
                                temperature=temperature,
                                context_length=steps,
                                device='cuda',
                                output_path=output_path,local_rank=local_rank,i=i)
        for u in range(repeat_num):
            code = output_ids[u][1:]
            index = (code >= 4737).nonzero()
            if index.numel() > 0:
                code = code[:index[0, 0].item()].cpu().numpy().astype(np.int64)
            else:
                code = code.cpu()
            vertices, faces = deserialize(code)
            faces = faces.reshape(-1, 3)
            faces = validate_and_filter_faces(vertices, faces)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            mesh.export(f'{output_path}/{local_rank}_{i}_{u}_mesh.ply') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/deemos-research-area-d/meshgen/code/HG_DEEMOS/out/tsz128x16k_100B_ScaleUp20k_Diff_LLaMA_2121M/Samba-DEEMOS-12-02-11/iter-000040-ckpt.pth",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model_id", type=str, default="2121", help="A custom name for the model."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000, 
        help="sampling steps.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./output_pc_aug'
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/deemos-research-area-d/meshgen/code/Samba/data/testdata"
    )
    parser.add_argument(
        "--repeat_num",
        type=int,
        default=4
    )
    parser.add_argument(
        "--point_num",
        type=int,
        default=81920
    )
    parser.add_argument(
        "--uid_list",
        type=str,
        default=''
    )
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    get_model_answers(
                local_rank=local_rank,
                world_size=world_size
    )
