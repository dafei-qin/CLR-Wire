#!/usr/bin/env python3
"""
数据集评估脚本 - 仅渲染 GT surface 和 GT PC，不加载模型
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
# Import config classes
from src.utils.import_tools import load_dataset_from_config, load_model_from_config
from src.utils.gpt_tools import tokenize_bspline_poles
from myutils.surface import visualize_json_interset


def visualize_points(all_points, radius=0.002):
    """Visualize point cloud from all_points array."""
    if all_points is None or len(all_points) == 0:
        return None
    
    # Register point cloud
    cloud = ps.register_point_cloud("points", all_points, radius=radius)
    return cloud


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


def create_two_views_grid(pc_paths, gt_paths, output_path):
    """
    Create a 2x3 grid of three views (front, left, top) for pc and gt.
    
    Layout:
        [pc_front]  [pc_left]  [pc_top]
        [gt_front]  [gt_left]  [gt_top]
    
    Args:
        pc_paths: dict with keys 'front', 'left', 'top' containing paths to pc view images
        gt_paths: dict with keys 'front', 'left', 'top' containing paths to gt view images
        output_path: path to save the final 2x3 grid image (jpg format)
    """
    # Load all images
    images = {}
    target_size = None
    
    for name, paths_dict in [('pc', pc_paths), ('gt', gt_paths)]:
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
    
    # Create 2x3 grid
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
    
    # Combine rows vertically
    final_image = Image.new('RGB', (target_size[0] * 3, target_size[1] * 2))
    final_image.paste(row1, (0, 0))
    final_image.paste(row2, (0, target_size[1]))
    
    # Save as JPG
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.jpg' and output_path.suffix.lower() != '.jpeg':
        output_path = output_path.with_suffix('.jpg')
    
    final_image.save(output_path, 'JPEG', quality=95)
    print(f"  六宫格图像保存到: {output_path}")
    
    # Delete all temporary PNG files after creating the grid
    all_png_paths = []
    for paths_dict in [pc_paths, gt_paths]:
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


@torch.no_grad()
def detokenize_bspline_poles(vae, tokens_unwarp):
    """从 tokens 中提取并解码 bspline poles（如果需要 VAE 解码）"""
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


def create_dataloader(config_dict: dict, batch_size: int = 1) -> data.DataLoader:
    """
    从配置文件创建数据加载器
    
    Args:
        config_dict: 配置字典
        batch_size: 批次大小
    """
    
    # 使用配置文件加载数据集
    print("📂 Loading dataset from config...")
    dataset = load_dataset_from_config(config_dict, section="data_val")
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="数据集评估脚本 - 仅渲染 GT surface 和 GT PC")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to config YAML file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=13,
                       help="Number of samples to process")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index of samples to process")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                       help="Output directory")
    parser.add_argument("--use_vae", action="store_true",
                       help="Whether to use VAE to decode bspline tokens (if tokens are encoded)")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")

    # ========== 加载配置文件 ==========
    if not Path(args.config_path).exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    config_dict = OmegaConf.load(args.config_path)
    print(f"📂 Loaded config from: {args.config_path}")

    print(f"数据集评估配置:")
    print(f"  配置文件: {args.config_path}")
    print(f"  处理样本数: {args.num_samples}")
    print(f"  起始索引: {args.start_idx}")

    # 创建数据加载器
    print("加载数据集...")
    dataloader = create_dataloader(config_dict, batch_size=1)
    dataset = dataloader.dataset
    print(f"数据集大小: {len(dataset)}")

    # 如果需要使用 VAE 解码 bspline tokens，加载 VAE
    vae = None
    if args.use_vae:
        print("加载 VAE 模型（用于解码 bspline tokens）...")
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

    for idx in range(args.start_idx, end_idx):
        print(f"\n处理样本 {idx}/{end_idx-1}...")

        # 获取样本数据
        train_data = dataset[idx]
        train_data = [_t[train_data[-1]] for _t in train_data[:-1]]
        train_data = [torch.from_numpy(_t).to(args.device) for _t in train_data]
        points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask = train_data

        # 如果使用 VAE，需要将 bspline poles 编码到 tokens 中
        if args.use_vae and vae is not None:
            all_tokens_padded = tokenize_bspline_poles(vae, dataset, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask)

        pc = torch.cat([points, normals], dim=-1)

        if pc.dim() == 2:
            pc = pc.unsqueeze(0)

        # 保存点云（可选）
        if pc is not None:
            ply_filename = f'{output_dir}/{idx}_pc_sample_{idx}.ply'
            pointcloud = trimesh.points.PointCloud(pc[0].detach().cpu().numpy()[..., :3])
            pointcloud.export(ply_filename)
            print(f"  点云保存到: {ply_filename}")
        
        gt_path = output_dir / f'{idx}_sample_{idx}_gt.png'
        pc_path = output_dir / f'{idx}_sample_{idx}_pc.png'

        # 处理 GT tokens
        tokens_gt = all_tokens_padded[0]
        tokens_gt = tokens_gt[~(tokens_gt == dataset.pad_id)]
        tokens_gt_unwarp = dataset.unwarp_codes(tokens_gt)
        
        # 提取 bspline poles
        if args.use_vae and vae is not None:
            # 如果 tokens 中已经有 bspline tokens（通过 VAE 编码的），需要解码
            bspline_poles_for_detok_gt = detokenize_bspline_poles(vae, tokens_gt_unwarp)
        else:
            # 直接使用原始的 bspline_poles
            # 计算 tokens 中有多少个 bspline surfaces
            num_bspline_in_tokens = sum(1 for token in tokens_gt_unwarp if token[0] == 5)
            # 从 all_bspline_poles_padded 中提取前 num_bspline_in_tokens 个有效的 poles
            if num_bspline_in_tokens == 0:
                bspline_poles_for_detok_gt = torch.empty((0, 4, 4, 4), device=args.device)
            else:
                # 根据 valid_mask 提取有效的 poles
                # all_bspline_valid_mask 中 True 的位置对应有效的 poles
                if isinstance(all_bspline_valid_mask, torch.Tensor):
                    valid_mask = all_bspline_valid_mask
                else:
                    valid_mask = torch.from_numpy(all_bspline_valid_mask).to(args.device)
                
                # 提取所有有效的 poles
                valid_poles = all_bspline_poles_padded[valid_mask]
                # 取前 num_bspline_in_tokens 个
                num_to_take = min(num_bspline_in_tokens, len(valid_poles))
                if num_to_take == 0:
                    bspline_poles_for_detok_gt = torch.empty((0, 4, 4, 4), device=args.device)
                else:
                    bspline_poles_for_detok_gt = valid_poles[:num_to_take]

        tokens_gt = tokens_gt.cpu()
        bspline_poles_for_detok_gt = bspline_poles_for_detok_gt.cpu().numpy()
        surfaces_gt = dataset.detokenize(tokens_gt.numpy(), bspline_poles_for_detok_gt)

        # Extract base path without extension for three views
        gt_base = str(gt_path).replace('.png', '')
        pc_base = str(pc_path).replace('.png', '')
        
        # Render three views for each type
        gt_paths = render_three_views(surfaces_gt, None, gt_base)
        pc_paths = render_three_views(None, pc[0, ..., :3].detach().cpu().numpy(), pc_base)
        
        # Create 2x3 grid combining pc and gt views
        grid_output_path = output_dir / f'{idx}_sample_{idx}_grid.jpg'
        create_two_views_grid(pc_paths, gt_paths, str(grid_output_path))

    print(f"\n批量处理完成! 结果保存在 {output_dir}")

if __name__ == "__main__":
    ps.init("openGL3_egl")
    main()

