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
import json
import shutil
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
import open3d as o3d
import random
from omegaconf import OmegaConf
import polyscope as ps
from PIL import Image
import colorsys
# Import model and config classes
from lit_gpt.model import GPT, Config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.utils.gpt_tools import tokenize_bspline_poles
from sft.datasets.serializaitonDEEMOS import deserialize
from myutils.surface import visualize_json_interset


def save_surfaces_as_mesh(surfaces, output_path):
    """
    将 surfaces 转换为 mesh 并保存为 .obj 文件。
    每个 surface 使用不同颜色以便区分。
    
    Args:
        surfaces: 从 dataset.detokenize 返回的 surface 列表
        output_path: 输出 .obj 文件路径
    """
    if not surfaces or len(surfaces) == 0:
        print(f"Warning: No surfaces to save for {output_path}")
        return
    
    # 使用 visualize_json_interset 获取 vertices 和 faces
    vis_data = visualize_json_interset(surfaces, plot=False, plot_gui=False, tol=1e-5, ps_header="")
    
    if not vis_data or len(vis_data) == 0:
        print(f"Warning: No visualization data generated for {output_path}")
        return
    
    # 生成颜色列表（使用不同颜色区分每个 surface）
    num_surfaces = len(vis_data)
    colors = generate_distinct_colors(num_surfaces)
    
    mesh_list = []
    for i, (surface_key, surface_data) in enumerate(vis_data.items()):
        if 'vertices' in surface_data and 'faces' in surface_data:
            vertices = surface_data['vertices']
            faces = surface_data['faces']
            
            # 确保 vertices 和 faces 是 numpy 数组
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.detach().cpu().numpy()
            
            # 创建 trimesh mesh
            try:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                
                # 设置面颜色
                if mesh.visual.kind is None:
                    mesh.visual = trimesh.visual.ColorVisuals()
                
                # 为每个面设置颜色
                face_colors = np.tile(colors[i], (len(mesh.faces), 1))
                mesh.visual.face_colors = face_colors
                
                mesh_list.append(mesh)
            except Exception as e:
                print(f"Warning: Failed to create mesh for surface {i} (key: {surface_key}): {e}")
                continue
    
    if len(mesh_list) == 0:
        print(f"Warning: No valid meshes to save for {output_path}")
        return
    
    # 合并所有 mesh
    try:
        combined_mesh = trimesh.util.concatenate(mesh_list)
        
        # 确保输出路径是 .obj 格式
        output_path = Path(output_path)
        if output_path.suffix.lower() != '.obj':
            output_path = output_path.with_suffix('.obj')
        
        # 保存 mesh
        combined_mesh.export(str(output_path))
        print(f"  Mesh保存到: {output_path} (包含 {len(mesh_list)} 个 surfaces)")
    except Exception as e:
        print(f"Error: Failed to save mesh to {output_path}: {e}")


def generate_distinct_colors(num_colors):
    """
    生成一组易于区分的颜色。
    
    Args:
        num_colors: 需要生成的颜色数量
    
    Returns:
        numpy array of shape (num_colors, 4) with RGBA values in [0, 255]
    """
    if num_colors == 0:
        return np.array([])
    
    # 使用 HSV 色彩空间生成均匀分布的颜色
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        saturation = 0.7 + 0.3 * (i % 2)  # 交替使用高饱和度和中等饱和度
        value = 0.8 + 0.2 * ((i // 2) % 2)  # 交替使用亮度和中等亮度
        
        # 转换为 RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # 转换为 [0, 255] 范围并添加 alpha
        rgba = [int(c * 255) for c in rgb] + [255]
        colors.append(rgba)
    
    return np.array(colors, dtype=np.uint8)


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
                if vertices.shape[0] == 0:
                    continue
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


def create_three_views_grid(pc_highres_paths, pc_lowres_paths, gt_paths, pred_paths, output_path):
    """
    Create a 4x3 grid of three views (front, left, top) for high-res pc, low-res pc, gt, and pred.
    
    Layout:
        [pc_highres_front]  [pc_highres_left]  [pc_highres_top]   (10240 points, no noise)
        [pc_lowres_front]   [pc_lowres_left]   [pc_lowres_top]    (4096 points, with noise)
        [gt_front]          [gt_left]          [gt_top]            (Ground truth surfaces)
        [pred_front]        [pred_left]        [pred_top]          (Predicted surfaces)
    
    Args:
        pc_highres_paths: dict with keys 'front', 'left', 'top' containing paths to high-res pc view images
        pc_lowres_paths: dict with keys 'front', 'left', 'top' containing paths to low-res pc view images
        gt_paths: dict with keys 'front', 'left', 'top' containing paths to gt view images
        pred_paths: dict with keys 'front', 'left', 'top' containing paths to pred view images
        output_path: path to save the final 4x3 grid image (jpg format)
    """
    # Load all images
    images = {}
    target_size = None
    
    for name, paths_dict in [('pc_highres', pc_highres_paths), ('pc_lowres', pc_lowres_paths), 
                              ('gt', gt_paths), ('pred', pred_paths)]:
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
    
    # Create 4x3 grid
    # Row 1: high-res pc views (10240 points, no noise)
    row1 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row1.paste(images['pc_highres_front'], (0, 0))
    row1.paste(images['pc_highres_left'], (target_size[0], 0))
    row1.paste(images['pc_highres_top'], (target_size[0] * 2, 0))
    
    # Row 2: low-res pc views (4096 points, with noise)
    row2 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row2.paste(images['pc_lowres_front'], (0, 0))
    row2.paste(images['pc_lowres_left'], (target_size[0], 0))
    row2.paste(images['pc_lowres_top'], (target_size[0] * 2, 0))
    
    # Row 3: gt views
    row3 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row3.paste(images['gt_front'], (0, 0))
    row3.paste(images['gt_left'], (target_size[0], 0))
    row3.paste(images['gt_top'], (target_size[0] * 2, 0))
    
    # Row 4: pred views
    row4 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row4.paste(images['pred_front'], (0, 0))
    row4.paste(images['pred_left'], (target_size[0], 0))
    row4.paste(images['pred_top'], (target_size[0] * 2, 0))
    
    # Combine rows vertically
    final_image = Image.new('RGB', (target_size[0] * 3, target_size[1] * 4))
    final_image.paste(row1, (0, 0))
    final_image.paste(row2, (0, target_size[1]))
    final_image.paste(row3, (0, target_size[1] * 2))
    final_image.paste(row4, (0, target_size[1] * 3))
    
    # Save as JPG
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.jpg' and output_path.suffix.lower() != '.jpeg':
        output_path = output_path.with_suffix('.jpg')
    
    final_image.save(output_path, 'JPEG', quality=95)
    print(f"  4x3网格图像保存到: {output_path}")
    
    # Delete all temporary PNG files after creating the grid
    all_png_paths = []
    for paths_dict in [pc_highres_paths, pc_lowres_paths, gt_paths, pred_paths]:
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
    else:
        print(f"No checkpoint found at {ckpt_path}, exit")
        exit()
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
    out = model(seq, max_seq_length=max_seq_length, input_pos=input_pos, pc=pc).logits  # [B, 1, V] - Cache PC features
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
        token_in = seq[:, -1:]  # [B, 1] - Only pass the latest token for KV cache
        # Reuse cached PC features by passing pc=None
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
    parser.add_argument("--do_inference", type=str, default='True', help="Whether to do inference")
    parser.add_argument("--output_dir", type=str, default="meshes/TestData_1201_NoScaleUp_50k_no_dynamic_window",
                       help="Output directory")
    parser.add_argument('--aug_num', type=int, default=1, help="Number of augmentations")
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
    args.do_inference = args.do_inference.lower() == 'true'
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
    
    num_iter = os.path.basename(args.ckpt).split('-')[1]
    




    for idx in range(args.start_idx, end_idx):
        # idx  = nums[idd]
        # idx = idd % args.num_samples
        print(f"\n处理样本 {idx}/{end_idx-1}...")

        # 获取样本数据

        for j in range(args.aug_num):
            # 4 times of augmentation

            train_data = dataset[idx]
            npz_path = dataset.npz_path[idx % len(dataset.npz_path)]
            json_path = npz_path.replace('.npz', '.json')
            
            # copy raw json to the output folder
            shutil.copy(json_path, output_dir / f'{idx}_raw_gt.json')
            
            # Load and save graph topology (nodes and edges)
            npz_data_graph = np.load(npz_path, allow_pickle=True)
            graph_nodes = npz_data_graph['graph_nodes']
            graph_edges = npz_data_graph['graph_edges']
            graph_output_path = output_dir / f'{idx}_graph.npz'
            np.savez(graph_output_path, nodes=graph_nodes, edges=graph_edges)
            print(f"  图拓扑保存到: {graph_output_path}")

            if train_data[-1] != True:
                print(f'invalid sample found for idx: {idx} with batch {j}')
                continue
            
            # Extract data - new dataset returns 8 values (including highres point clouds)
            train_data_list = [_t[train_data[-1]] for _t in train_data[:-1]]
            train_data_list = [torch.from_numpy(_t).to(args.device) for _t in train_data_list]
            
            # Unpack: 5 original values + 2 highres values
            if len(train_data_list) == 7:  # New inference dataset
                points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, points_highres, normals_highres = train_data_list
            else:  # Old dataset (fallback compatibility)
                points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask = train_data_list
                points_highres = None
                normals_highres = None

            print('points_highres: ', points_highres.shape)
            
            all_tokens_padded = tokenize_bspline_poles(vae, dataset, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask)

            pc = torch.cat([points, normals], dim=-1)

            if pc.dim() == 2:
                pc = pc.unsqueeze(0)


            # 保存单份点云（可选）
            if pc is not None:
                ply_filename = f'{output_dir}/{idx}_batch_{j}.ply'
                pc_numpy = pc[0].detach().cpu().numpy()
                vertices = pc_numpy[..., :3]
                normals_pc = pc_numpy[..., 3:6] if pc_numpy.shape[-1] >= 6 else None
                
                # 使用 open3d 保存点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)
                if normals_pc is not None:
                    print('save_normals')
                    pcd.normals = o3d.utility.Vector3dVector(normals_pc)
                o3d.io.write_point_cloud(ply_filename, pcd)
                print(f"  点云保存到: {ply_filename}")
            
            # 保存高分辨率点云（10240点，无噪声，有旋转）
            if points_highres is not None:
                ply_filename_highres = f'{output_dir}/{idx}_batch_{j}_highres.ply'
                vertices_highres = points_highres[0].detach().cpu().numpy()
                normals_highres_np = normals_highres[0].detach().cpu().numpy() if normals_highres is not None else None
                
                # 使用 open3d 保存高分辨率点云
                pcd_highres = o3d.geometry.PointCloud()
                pcd_highres.points = o3d.utility.Vector3dVector(vertices_highres.astype(float))
                if normals_highres_np is not None:
                    print('save_highres_normals')
                    pcd_highres.normals = o3d.utility.Vector3dVector(normals_highres_np.astype(float))
                o3d.io.write_point_cloud(ply_filename_highres, pcd_highres)
                print(f"  高分辨率点云保存到: {ply_filename_highres} (点数: {len(vertices_highres)})")

            target_dtype = next(model.conditioner.parameters()).dtype if hasattr(model, "conditioner") and model.conditioner is not None else next(model.parameters()).dtype
            pc = pc.to(args.device, dtype=target_dtype)

            # ===== 批量推理：把起始 token 和点云在 batch 维复制 4 份，同时解码 =====
            B = 2
            print('do_inference: ', args.do_inference)
            if args.do_inference:
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
            else:
                tokens_batch = None

            # 生成并保存每个 batch 的 mesh
            for b in range(1):
                if tokens_batch is None:
                    tokens =  all_tokens_padded[0]
                    tokens = tokens[~(tokens == dataset.pad_id)]
                else:
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

                # 保存 surfaces 到 json 文件
                pred_json_path = output_dir / f'{idx}_batch_{j}_pred_iter_{num_iter}.json'
                gt_json_path = output_dir / f'{idx}_batch_{j}_gt_iter_{num_iter}.json'
                with open(pred_json_path, 'w') as f:
                    json.dump(surfaces_pred, f)
                with open(gt_json_path, 'w') as f:
                    json.dump(surfaces_gt, f)

                # 保存 mesh 到 .obj 文件
                pred_mesh_path = output_dir / f'{idx}_batch_{j}_pred_iter_{num_iter}.obj'
                gt_mesh_path = output_dir / f'{idx}_batch_{j}_gt_iter_{num_iter}.obj'
                
                save_surfaces_as_mesh(surfaces_pred, str(pred_mesh_path))
                if b == 0:
                    save_surfaces_as_mesh(surfaces_gt, str(gt_mesh_path))

                # 渲染三视图
                pred_base = str(output_dir / f'{idx}_pred_iter_{num_iter}').replace('.png', '')
                gt_base = str(output_dir / f'{idx}_gt_iter_{num_iter}').replace('.png', '')
                pc_highres_base = str(output_dir / f'{idx}_pc_highres_iter_{num_iter}').replace('.png', '')
                pc_lowres_base = str(output_dir / f'{idx}_pc_lowres_iter_{num_iter}').replace('.png', '')
                
                # Render three views for each type
                pred_paths = render_three_views(surfaces_pred, None, pred_base)
                gt_paths = render_three_views(surfaces_gt, None, gt_base)
                
                # Render high-res point cloud (10240 points, no noise, with rotation)
                if points_highres is not None:
                    pc_highres_numpy = points_highres[0].detach().cpu().float().numpy()
                    pc_highres_paths = render_three_views(None, pc_highres_numpy, pc_highres_base)
                else:
                    # Fallback to using low-res if high-res not available
                    pc_highres_numpy = pc[0, ..., :3].detach().cpu().float().numpy()
                    pc_highres_paths = render_three_views(None, pc_highres_numpy, pc_highres_base)
                
                # Render low-res point cloud (4096 points, with noise, with rotation)
                pc_lowres_numpy = pc[0, ..., :3].detach().cpu().float().numpy()
                pc_lowres_paths = render_three_views(None, pc_lowres_numpy, pc_lowres_base)
                
                # Create 4x3 grid combining all views
                grid_output_path = output_dir / f'{idx}_batch_{j}_grid_iter_{num_iter}.jpg'
                create_three_views_grid(pc_highres_paths, pc_lowres_paths, gt_paths, pred_paths, str(grid_output_path))


    print(f"\n批量处理完成! 结果保存在 {output_dir}")

if __name__ == "__main__":
    ps.init("openGL3_egl")
    main()

