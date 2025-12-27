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
import einops

# Support running without installing as a package (match training script behavior)
import sys
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

# Add project root to sys.path to import src.utils
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import trimesh
import numpy as np
import torch.utils.data as data
import random
from omegaconf import OmegaConf
import polyscope as ps
from PIL import Image
# Import model and config classes
from lit_gpt.model import GPT, Config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.utils.gpt_tools import tokenize_bspline_poles
from sft.datasets.serializaitonDEEMOS import deserialize
from myutils.surface import visualize_json_interset


def visualize_points(all_points, radius=0.002):
    """Visualize point cloud from all_points array."""
    if all_points is None or len(all_points) == 0:
        return None
    
    # Register point cloud
    cloud = ps.register_point_cloud("points", all_points, radius=radius)
    return cloud


def render_single_view(surfaces, all_points, view_name, output_path):
    """Render a single view and save screenshot."""
    # Clear previous structures
    ps.remove_all_structures()
    
    # Visualize surfaces (plot=False to avoid re-initializing polyscope)
    if surfaces:
        vis_data = visualize_json_interset(surfaces, plot=False, plot_gui=False, tol=1e-5, ps_header=view_name)
        # Manually register surfaces since plot=False
        for surface_data in vis_data.values():
            if 'vertices' in surface_data and 'faces' in surface_data:
                vertices = surface_data['vertices']
                faces = surface_data['faces']
                surface_index = surface_data.get('surface_index', 0)
                surface_type = surface_data.get('surface_type', 'unknown')
                ps.register_surface_mesh(
                    f"{view_name}_{surface_index:03d}_{surface_type}",
                    vertices,
                    faces,
                    transparency=0.7
                )
    
    # Visualize points
    if all_points is not None and len(all_points) > 0:
        visualize_points(all_points, radius=0.002)
    
    # Set viewport size and camera
    ps.set_view_projection_mode("perspective")
    ps.reset_camera_to_home_view()
    
    # Take screenshot
    ps.screenshot(output_path, transparent_bg=False)
    
    return output_path


def set_camera_view(view_type, distance=5.0):
    """
    Set camera to a specific orthographic view.
    
    Args:
        view_type: 'front', 'left', or 'top'
        distance: Distance from origin to camera position
    """
    # Set orthographic projection for engineering views
    ps.set_view_projection_mode("orthographic")
    
    # Use small offset to avoid collinear issues with up direction
    epsilon = 1e-6
    
    if view_type == 'front':
        # 正视图：从 Z 轴正方向看（相机在 Z 轴正方向）
        # 上方向为 Y 轴正方向
        ps.set_up_dir("y_up")
        # Add tiny offset to avoid exact alignment issues
        camera_pos = (epsilon, 0.0, distance)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    elif view_type == 'left':
        # 左视图：从 X 轴正方向看（相机在 X 轴正方向）
        # 上方向为 Y 轴正方向
        ps.set_up_dir("y_up")
        # Add tiny offset to avoid exact alignment issues
        camera_pos = (distance, epsilon, 0.0)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    elif view_type == 'top':
        # 顶视图：从 Y 轴正方向看（相机在 Y 轴正方向）
        # 上方向为 Z 轴正方向（避免与观察方向共线）
        ps.set_up_dir("z_up")
        # Add tiny offset to avoid exact alignment issues
        camera_pos = (epsilon, distance, 0.0)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    else:
        raise ValueError(f"Unknown view_type: {view_type}. Must be 'front', 'left', or 'top'")


def render_three_views(surfaces, all_points, base_output_path, view_distance=5.0):
    """
    Render three orthographic views: front view, left view, and top view.
    
    Args:
        surfaces: Surface data to render
        all_points: Point cloud data to render (optional)
        base_output_path: Base path for output images (without extension)
                          Output files will be: {base_output_path}_front.png,
                                                {base_output_path}_left.png,
                                                {base_output_path}_top.png
        view_distance: Distance from origin to camera position
    
    Returns:
        dict: Dictionary with keys 'front', 'left', 'top' containing output paths
    """
    output_paths = {}
    
    # Clear previous structures first
    ps.remove_all_structures()
    
    # Visualize surfaces (plot=False to avoid re-initializing polyscope)
    if surfaces:
        vis_data = visualize_json_interset(surfaces, plot=False, plot_gui=False, tol=1e-5, ps_header="three_views")
        # Manually register surfaces since plot=False
        for surface_data in vis_data.values():
            if 'vertices' in surface_data and 'faces' in surface_data:
                vertices = surface_data['vertices']
                faces = surface_data['faces']
                surface_index = surface_data.get('surface_index', 0)
                surface_type = surface_data.get('surface_type', 'unknown')
                ps.register_surface_mesh(
                    f"three_views_{surface_index:03d}_{surface_type}",
                    vertices,
                    faces,
                    transparency=0.7
                )
    
    # Visualize points
    if all_points is not None and len(all_points) > 0:
        visualize_points(all_points, radius=0.002)
    
    # Render front view (正视图)
    ps.reset_camera_to_home_view()
    set_camera_view('front', distance=view_distance)
    front_path = f"{base_output_path}_front.png"
    ps.screenshot(front_path, transparent_bg=False)
    output_paths['front'] = front_path
    
    # Render left view (左视图)
    ps.reset_camera_to_home_view()
    set_camera_view('left', distance=view_distance)
    left_path = f"{base_output_path}_left.png"
    ps.screenshot(left_path, transparent_bg=False)
    output_paths['left'] = left_path
    
    # Render top view (顶视图)
    ps.reset_camera_to_home_view()
    set_camera_view('top', distance=view_distance)
    top_path = f"{base_output_path}_top.png"
    ps.screenshot(top_path, transparent_bg=False)
    output_paths['top'] = top_path
    
    return output_paths


def create_three_views_grid(pc_paths, gt_paths, pred_paths, output_path):
    """
    Create a 3x3 grid of three views (front, left, top) for pc, gt, and pred.
    
    Layout:
        [pc_front]  [pc_left]  [pc_top]
        [gt_front]  [gt_left]  [gt_top]
        [pred_front] [pred_left] [pred_top]
    
    Args:
        pc_paths: dict with keys 'front', 'left', 'top' containing paths to pc view images
        gt_paths: dict with keys 'front', 'left', 'top' containing paths to gt view images
        pred_paths: dict with keys 'front', 'left', 'top' containing paths to pred view images
        output_path: path to save the final 3x3 grid image (jpg format)
    """
    # Load all images
    images = {}
    target_size = None
    
    for name, paths_dict in [('pc', pc_paths), ('gt', gt_paths), ('pred', pred_paths)]:
        for view in ['front', 'left', 'top']:
            if view in paths_dict and Path(paths_dict[view]).exists():
                img = Image.open(paths_dict[view])
                images[f'{name}_{view}'] = img
                # Use first successfully loaded image size as target
                if target_size is None:
                    target_size = img.size
            else:
                # Will create blank image later with correct size
                print(f"Warning: {paths_dict.get(view, 'unknown')} not found, will use blank image")
                images[f'{name}_{view}'] = None
    
    # If no images were loaded, use default size
    if target_size is None:
        target_size = (800, 600)
    
    # Create blank images for missing ones
    for key in images:
        if images[key] is None:
            images[key] = Image.new('RGB', target_size, color='white')
    
    # Resize all images to the same size
    for key in images:
        if images[key].size != target_size:
            images[key] = images[key].resize(target_size, Image.Resampling.LANCZOS)
    
    # Create 3x3 grid
    # Row 1: pc views
    row1 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row1.paste(images['pc_front'], (0, 0))
    row1.paste(images['pc_left'], (target_size[0], 0))
    row1.paste(images['pc_top'], (target_size[0] * 2, 0))
    
    # Row 2: gt views
    row2 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row2.paste(images['gt_front'], (0, 0))
    row2.paste(images['gt_left'], (target_size[0], 0))
    row2.paste(images['gt_top'], (target_size[0] * 2, 0))
    
    # Row 3: pred views
    row3 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row3.paste(images['pred_front'], (0, 0))
    row3.paste(images['pred_left'], (target_size[0], 0))
    row3.paste(images['pred_top'], (target_size[0] * 2, 0))
    
    # Combine rows vertically
    final_image = Image.new('RGB', (target_size[0] * 3, target_size[1] * 3))
    final_image.paste(row1, (0, 0))
    final_image.paste(row2, (0, target_size[1]))
    final_image.paste(row3, (0, target_size[1] * 2))
    
    # Save as JPG
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.jpg' and output_path.suffix.lower() != '.jpeg':
        output_path = output_path.with_suffix('.jpg')
    
    final_image.save(output_path, 'JPEG', quality=95)
    print(f"  九宫格图像保存到: {output_path}")
    
    # Delete all temporary PNG files after creating the grid
    all_png_paths = []
    for paths_dict in [pc_paths, gt_paths, pred_paths]:
        for view in ['front', 'left', 'top']:
            if view in paths_dict:
                png_path = Path(paths_dict[view])
                if png_path.exists() and png_path.suffix.lower() == '.png':
                    all_png_paths.append(png_path)
    
    for png_path in all_png_paths:
        try:
            png_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete {png_path}: {e}")
    
    return output_path





def load_model(ckpt_path: str, config_dict: dict, device: str = "cuda", dtype: str = "bf16") -> GPT:
    """
    从配置文件加载模型
    
    Args:
        ckpt_path: checkpoint 路径
        config_dict: 配置字典
        device: 设备
        dtype: 数据类型
    """
    if "model" not in config_dict:
        raise ValueError("config_dict must contain 'model' section")
    
    print("📦 Loading model from config...")
    # 将 OmegaConf 对象转换为普通字典，避免 Literal 类型注解验证错误
    config_params = OmegaConf.to_container(config_dict.model.params.config, resolve=True)
    config_obj = Config(**config_params)
    # 将整个 config_dict 转换为普通字典，避免 OmegaConf 类型验证
    config_dict_plain = OmegaConf.to_container(config_dict, resolve=True)
    config_dict_plain["model"]["params"]["config"] = config_obj
    model = load_model_from_config(config_dict_plain, device=device, strict=False)
    
    # 如果提供了 checkpoint 路径，加载权重（覆盖配置文件中的 checkpoint）
    if ckpt_path is not None and Path(ckpt_path).exists():
        print(f"📥 Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    # 设置数据类型和设备
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
        token_in = seq[:, :]  # [B, 1]
        # out = model(token_in, max_seq_length=max_seq_length, input_pos=input_pos, pc=None).logits  # [B, 1, V]
        out = model(token_in, max_seq_length=max_seq_length, pc=pc).logits  # [B, 1, V]
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


def create_dataloader(config_dict: dict, batch_size: int = 1) -> data.DataLoader:
    """
    从配置文件创建数据加载器
    
    Args:
        config_dict: 配置字典
        batch_size: 批次大小
    # """

    
    # 使用配置文件加载数据集
    print("📂 Loading dataset from config...")
    dataset = load_dataset_from_config(config_dict, section="data_val")
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        # collate_fn=collate_as_list,
    )
    
    return dataloader

@torch.no_grad()
def detokenize_bspline_poles(vae, tokens_unwarp):
    bspline_tokens_recovered = []
    for token in tokens_unwarp:
        if token[0] == 5:  # Bspline surface
            bspline_tokens_recovered.append(token[1:7]) # six tokens starting from the second index
    if len(bspline_tokens_recovered) == 0:
        bspline_poles_for_detok = torch.empty((0, 4, 4, 4))
    
    else:
        bspline_tokens_recovered = torch.stack(bspline_tokens_recovered)
        z_quantized_recover = vae.indices_to_latent(bspline_tokens_recovered)
        x_recon = vae.decode(z_quantized_recover)
        x_recon = einops.rearrange(x_recon, "b c h w -> b h w c")
        num_bsplines = x_recon.shape[0]
        new_poles = torch.zeros((num_bsplines, 4, 4, 4))
        new_poles[..., :3] = x_recon
        new_poles[..., 3] = 1.0

        bspline_poles_for_detok = new_poles

    return bspline_poles_for_detok

def main():
    parser = argparse.ArgumentParser(description="批量KV-Cache推理 for Samba GPT")
    parser.add_argument("--ckpt", type=str, required=True,
                       help="Path to training checkpoint (.pth)")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--max_new_tokens", type=int, default=50000)
    parser.add_argument("--max_seq_len", type=int, default=50000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_samples", type=int, default=13,
                       help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=300000,
                       help="Starting index of samples to process")
    parser.add_argument("--output_dir", type=str, default="meshes/TestData_1201_NoScaleUp_50k_no_dynamic_window",
                       help="Output directory")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # ========== 加载配置文件 ==========
    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    config_dict = OmegaConf.load(args.config_path)
    print(f"📂 Loaded config from: {args.config_path}")

    print(f"批量推理配置:")
    print(f"  配置文件: {args.config_path}")
    print(f"  最大序列长度: {args.max_seq_len}")
    print(f"  处理样本数: {args.num_samples}")
    print(f"  起始索引: {args.start_idx}")

    # 创建数据加载器
    print("加载数据集...")
    dataloader = create_dataloader(config_dict, batch_size=1)
    dataset = dataloader.dataset
    print(f"数据集大小: {len(dataset)}")

    # 加载模型
    print("加载模型...")
    model = load_model(
        args.ckpt, 
        config_dict=config_dict,
        device=args.device, 
        dtype=args.dtype
    )
    vae = load_model_from_config(config_dict, section="vae")
    vae.eval()
    vae.to(args.device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理样本
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    print(f"处理样本 {args.start_idx} 到 {end_idx-1}...")
    random.seed(928)



    for idd in range(args.num_samples * 2):
        # idx  = nums[idd]
        idx = idd % args.num_samples
        print(f"\n处理样本 {idx}/{end_idx-1}...")

        # 获取样本数据

        train_data = dataset[idx]
        train_data = [_t[train_data[-1]] for _t in train_data[:-1]]
        train_data = [torch.from_numpy(_t).to(args.device) for _t in train_data]
        points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask = train_data
        all_tokens_padded = tokenize_bspline_poles(vae, dataset, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask)

        pc = torch.cat([points, normals], dim=-1)

        if pc.dim() == 2:
            pc = pc.unsqueeze(0)


        # 保存单份点云（可选）
        if pc is not None:
            ply_filename = f'{output_dir}/{idd}_pc_sample_{idx}.ply'
            pointcloud = trimesh.points.PointCloud(pc[0].detach().cpu().numpy()[..., :3])
            pointcloud.export(ply_filename)
            print(f"  点云保存到: {ply_filename}")
        
        pred_path = output_dir / f'{idd}_sample_{idx}_pred.png'
        gt_path = output_dir / f'{idd}_sample_{idx}_gt.png'
        pc_path = output_dir / f'{idd}_sample_{idx}_pc.png'

        target_dtype = next(model.conditioner.parameters()).dtype if hasattr(model, "conditioner") and model.conditioner is not None else next(model.parameters()).dtype
        pc = pc.to(args.device, dtype=target_dtype)

        # ===== 批量推理：把起始 token 和点云在 batch 维复制 4 份，同时解码 =====
        B = 4
        tokens_batch = generate_with_kvcache(
            model,
            start_token_id=dataset.start_id,
            pc=pc,  # 函数内部会扩成 [B, N, C]
            max_new_tokens=args.max_new_tokens,
            max_seq_length=args.max_seq_len,
            temperature=args.temperature,
            batch_size=B,
            eos_token_id=dataset.end_id
        )

        # 生成并保存每个 batch 的 mesh
        for b in range(B):
            tokens = tokens_batch[b]
            tokens = torch.tensor(tokens).to(args.device)
            

            tokens_unwarp = dataset.unwarp_codes(tokens)

            # bspline_tokens_recovered = []
            bspline_poles_for_detok = detokenize_bspline_poles(vae, tokens_unwarp)
            # detok_surfaces = dataset.detokenize(tokens, bspline_poles_for_detok)


            tokens_gt = all_tokens_padded[0]
            tokens_gt = tokens_gt[~(tokens_gt == dataset.pad_id)]
            tokens_gt_unwarp = dataset.unwarp_codes(tokens_gt)
            bspline_poles_for_detok_gt = detokenize_bspline_poles(vae, tokens_gt_unwarp)

            tokens = tokens.cpu()
            tokens_gt = tokens_gt.cpu()
            bspline_poles_for_detok = bspline_poles_for_detok.cpu().numpy()
            bspline_poles_for_detok_gt = bspline_poles_for_detok_gt.cpu().numpy()
            surfaces_pred = dataset.detokenize(tokens.numpy(), bspline_poles_for_detok)
            surfaces_gt = dataset.detokenize(tokens_gt.numpy(), bspline_poles_for_detok_gt)


            # Extract base path without extension for three views
            pred_base = str(pred_path).replace('.png', '')
            gt_base = str(gt_path).replace('.png', '')
            pc_base = str(pc_path).replace('.png', '')
            
            # Render three views for each type
            pred_paths = render_three_views(surfaces_pred, None, pred_base)
            gt_paths = render_three_views(surfaces_gt, None, gt_base)
            pc_paths = render_three_views(None, pc[0, ..., :3].detach().cpu().numpy(), pc_base)
            
            # Create 3x3 grid combining all views
            grid_output_path = output_dir / f'{idd}_sample_{idx}_grid.jpg'
            create_three_views_grid(pc_paths, gt_paths, pred_paths, str(grid_output_path))


    print(f"\n批量处理完成! 结果保存在 {output_dir}")

if __name__ == "__main__":
    ps.init("openGL3_egl")
    main()

