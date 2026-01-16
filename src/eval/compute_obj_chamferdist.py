"""
计算GT和预测结果之间的Chamfer Distance
预先构建路径映射表以提高效率
支持两种方法：
1. PLY点云直接计算
2. JSON表面采样后计算
"""

import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import json
import torch
from PIL import Image

# 添加上级目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from plyfile import PlyData
except ImportError:
    print("Please install plyfile: pip install plyfile")
    exit(1)

try:
    import polyscope as ps
except ImportError:
    print("Warning: polyscope not installed, rendering will be disabled")
    ps = None

# 从 compute_json_chamferdist.py 导入工具函数
from src.utils.surface_tools import params_to_samples


def read_ply(ply_path):
    """读取PLY文件，返回点云坐标 (N, 3)"""
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return points.astype(np.float32)


def load_json_surfaces(json_path):
    """加载JSON表面文件"""
    with open(json_path, 'r') as f:
        surfaces = json.load(f)
    return surfaces


def surface_to_samples(surface, resolution=16):
    """将单个表面转换为采样点云"""
    surface_type = surface['type']
    
    if surface_type == 'bspline_surface':
        surface['poles'] = np.array(surface['poles'])
    else:
        surface['location'] = torch.tensor(surface['location'])
        surface['direction'] = torch.tensor(surface['direction'])
        surface['scalar'] = torch.tensor(surface['scalar'])
        surface['uv'] = torch.tensor(surface['uv'])
    
    samples = params_to_samples(
        torch.zeros([]),
        surface['type'],
        resolution,
        resolution,
        surface_json=surface
    )
    
    if isinstance(samples, torch.Tensor):
        samples = samples.squeeze(0).numpy() if samples.dim() > 3 else samples.numpy()
    
    return samples


def json_to_pointcloud(json_path, resolution=16):
    """从JSON文件生成点云 (N, 3)"""
    surfaces = load_json_surfaces(json_path)
    
    points_list = []
    for surface in surfaces:
        try:
            samples = surface_to_samples(surface, resolution)
            points_list.append(samples.reshape(-1, 3))
        except Exception as e:
            # 跳过有问题的表面，继续处理其他表面
            print(f"警告: 处理表面失败，跳过该表面: {e}")
            continue
    
    if len(points_list) == 0:
        return None
    
    all_points = np.concatenate(points_list, axis=0)
    return all_points.astype(np.float32)


def visualize_points(all_points, radius=0.002):
    """Visualize point cloud from all_points array."""
    if all_points is None or len(all_points) == 0 or ps is None:
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
    if ps is None:
        return
    
    # Set orthographic projection for engineering views
    ps.set_view_projection_mode("orthographic")
    
    # Use small offset to avoid collinear issues with up direction
    epsilon = 1e-6
    
    if view_type == 'front':
        # 正视图：从 Z 轴正方向看（相机在 Z 轴正方向）
        # 上方向为 Y 轴正方向
        ps.set_up_dir("y_up")
        camera_pos = (epsilon, 0.0, distance)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    elif view_type == 'left':
        # 左视图：从 X 轴正方向看（相机在 X 轴正方向）
        # 上方向为 Y 轴正方向
        ps.set_up_dir("y_up")
        camera_pos = (distance, epsilon, 0.0)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    elif view_type == 'top':
        # 顶视图：从 Y 轴正方向看（相机在 Y 轴正方向）
        # 上方向为 Z 轴正方向（避免与观察方向共线）
        ps.set_up_dir("z_up")
        camera_pos = (epsilon, distance, 0.0)
        target_pos = (0.0, 0.0, 0.0)
        ps.look_at(camera_pos, target_pos)
    else:
        raise ValueError(f"Unknown view_type: {view_type}. Must be 'front', 'left', or 'top'")


def render_three_views(all_points, base_output_path, view_distance=5.0):
    """
    Render three orthographic views: front view, left view, and top view.
    
    Args:
        all_points: Point cloud data to render (N, 3)
        base_output_path: Base path for output images (without extension)
        view_distance: Distance from origin to camera position
    
    Returns:
        dict: Dictionary with keys 'front', 'left', 'top' containing output paths
    """
    if ps is None or all_points is None or len(all_points) == 0:
        return {}
    
    output_paths = {}
    
    # Clear previous structures first
    ps.remove_all_structures()
    
    # Visualize points
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


def create_two_rows_grid(gt_paths, pred_paths, output_path):
    """
    Create a 2x3 grid of three views (front, left, top) for gt and pred.
    
    Layout:
        [gt_front]  [gt_left]  [gt_top]
        [pred_front] [pred_left] [pred_top]
    
    Args:
        gt_paths: dict with keys 'front', 'left', 'top' containing paths to gt view images
        pred_paths: dict with keys 'front', 'left', 'top' containing paths to pred view images
        output_path: path to save the final 2x3 grid image (jpg format)
    """
    if ps is None:
        return None
    
    # Load all images
    images = {}
    target_size = None
    
    for name, paths_dict in [('gt', gt_paths), ('pred', pred_paths)]:
        for view in ['front', 'left', 'top']:
            if view in paths_dict and Path(paths_dict[view]).exists():
                img = Image.open(paths_dict[view])
                images[f'{name}_{view}'] = img
                # Use first successfully loaded image size as target
                if target_size is None:
                    target_size = img.size
            else:
                # Will create blank image later with correct size
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
    # Row 1: gt views
    row1 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row1.paste(images['gt_front'], (0, 0))
    row1.paste(images['gt_left'], (target_size[0], 0))
    row1.paste(images['gt_top'], (target_size[0] * 2, 0))
    
    # Row 2: pred views
    row2 = Image.new('RGB', (target_size[0] * 3, target_size[1]))
    row2.paste(images['pred_front'], (0, 0))
    row2.paste(images['pred_left'], (target_size[0], 0))
    row2.paste(images['pred_top'], (target_size[0] * 2, 0))
    
    # Combine rows vertically
    final_image = Image.new('RGB', (target_size[0] * 3, target_size[1] * 2))
    final_image.paste(row1, (0, 0))
    final_image.paste(row2, (0, target_size[1]))
    
    # Save as JPG
    output_path = Path(output_path)
    if output_path.suffix.lower() not in ['.jpg', '.jpeg']:
        output_path = output_path.with_suffix('.jpg')
    
    final_image.save(output_path, 'JPEG', quality=95)
    
    # Delete all temporary PNG files after creating the grid
    all_png_paths = []
    for paths_dict in [gt_paths, pred_paths]:
        for view in ['front', 'left', 'top']:
            if view in paths_dict:
                png_path = Path(paths_dict[view])
                if png_path.exists() and png_path.suffix.lower() == '.png':
                    all_png_paths.append(png_path)
    
    for png_path in all_png_paths:
        try:
            png_path.unlink()
        except Exception as e:
            pass
    
    return output_path


def compute_chamfer_distance_fast(pred_points, gt_points, bidirectional=False):
    """
    快速计算两个点云之间的 Chamfer Distance (使用GPU)
    
    Args:
        pred_points: (N, 3) numpy array 或 torch tensor
        gt_points: (M, 3) numpy array 或 torch tensor
        bidirectional: 是否计算双向 Chamfer Distance (默认False，只计算 pred->gt)
    
    Returns:
        chamfer_dist: 单向或双向 Chamfer Distance
    """
    try:
        # 转换为 torch tensor（在 GPU 上）
        if not torch.is_tensor(pred_points):
            pred_points = torch.from_numpy(pred_points).float()
        if not torch.is_tensor(gt_points):
            gt_points = torch.from_numpy(gt_points).float()
        
        # 确保在 GPU 上（如果可用）
        if torch.cuda.is_available():
            if not pred_points.is_cuda:
                pred_points = pred_points.cuda()
            if not gt_points.is_cuda:
                gt_points = gt_points.cuda()
        
        # 计算距离矩阵
        dist_matrix = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)  # (N, M)
        
        # pred -> gt
        min_dist_pred_to_gt = dist_matrix.min(dim=1)[0]  # (N,)
        pred_to_gt_mean = min_dist_pred_to_gt.mean()
        
        if bidirectional:
            # gt -> pred
            min_dist_gt_to_pred = dist_matrix.min(dim=0)[0]  # (M,)
            gt_to_pred_mean = min_dist_gt_to_pred.mean()
            
            # Chamfer Distance (双向求和)
            chamfer_dist = (pred_to_gt_mean + gt_to_pred_mean).item()
        else:
            # 单向 Chamfer Distance (pred -> gt)
            chamfer_dist = pred_to_gt_mean.item()
        
        return chamfer_dist
    
    except Exception as e:
        print(f"警告: GPU计算失败，回退到CPU: {e}")
        return float('inf')


def build_result_path_mapping(results_root):
    """
    构建结果路径映射表
    返回: dict[gt_filename] -> list[result_ply_paths]
    """
    print("正在构建结果路径映射表...")
    mapping = defaultdict(list)
    
    results_root = Path(results_root)
    
    # 收集所有 brep_results 目录
    brep_results_dirs = []
    
    # 方法1: 查找 process_X/brep_results 结构
    for process_dir in sorted(results_root.glob("process_*")):
        brep_results_dir = process_dir / "brep_results"
        if brep_results_dir.exists():
            brep_results_dirs.append(brep_results_dir)
    
    # 方法2: 如果没有 process_* 文件夹，检查是否直接有 brep_results
    if not brep_results_dirs:
        direct_brep = results_root / "brep_results"
        if direct_brep.exists():
            brep_results_dirs.append(direct_brep)
            print(f"  使用直接 brep_results 目录: {direct_brep}")
    
    # 遍历所有找到的 brep_results 目录
    for brep_results_dir in brep_results_dirs:
        # 遍历每个样本文件夹
        for sample_dir in brep_results_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            
            sample_name = sample_dir.name
            
            # 遍历所有子文件夹 (00_00, 00_01, ...)
            for sub_dir in sample_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                separate_faces_ply = sub_dir / "separate_faces.ply"
                if separate_faces_ply.exists():
                    mapping[sample_name].append(str(separate_faces_ply))
    
    print(f"映射表构建完成: 找到 {len(mapping)} 个GT样本的结果")
    
    # 打印统计信息
    result_counts = [len(v) for v in mapping.values()]
    if result_counts:
        print(f"  每个GT样本的结果数: min={min(result_counts)}, "
              f"max={max(result_counts)}, mean={np.mean(result_counts):.2f}")
    
    return mapping


def compute_chamfer_distances(gt_root, results_root, output_dir, resolution=16, disable_json=False, render_views=False, bidirectional=False):
    """
    批量计算Chamfer Distance
    支持两种方法：PLY点云和JSON表面采样
    
    Args:
        disable_json: 如果为True，则禁用JSON-based计算
        render_views: 如果为True，渲染GT和result的三视图对比（2x3网格）
        bidirectional: 如果为True，计算双向Chamfer Distance；否则只计算pred->gt单向距离
    """
    gt_root = Path(gt_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 构建结果路径映射表
    result_mapping = build_result_path_mapping(results_root)
    
    # 2. 收集所有GT文件（同时查找.ply和.json）
    print("\n正在收集GT文件...")
    gt_files = []
    
    # 收集所有PLY文件的路径
    ply_file_paths = []
    
    # 方法1: 从子目录中查找（原有逻辑）
    for index_dir in sorted(gt_root.iterdir()):
        if not index_dir.is_dir():
            continue
        
        ply_files = list(index_dir.glob("*.ply"))
        ply_file_paths.extend(ply_files)
    
    # 方法2: 如果没找到，直接从根目录查找
    if not ply_file_paths:
        ply_file_paths = list(gt_root.glob("*.ply"))
        print(f"  从根目录直接读取PLY文件")
    
    # 处理所有找到的PLY文件
    for ply_file in ply_file_paths:
        gt_name = ply_file.stem  # 去掉 .ply 扩展名
        
        # 检查是否有对应的结果
        if gt_name in result_mapping:
            # 查找JSON文件（支持多种格式）
            json_path = None
            
            if not disable_json:
                # 优先查找 tokenized JSON 文件
                tokenized_json_file = ply_file.parent / (ply_file.stem + '_tokenized.json')
                
                if tokenized_json_file.exists():
                    json_path = str(tokenized_json_file)
                else:
                    # 如果没有tokenized文件，查找 _gt_iter_*.json 格式
                    # 尝试多种可能的前缀匹配方式
                    gt_json_files = []
                    
                    # 方式1: 完整的stem (如 0_batch_0_highres_gt_iter_*.json)
                    gt_json_files.extend(list(ply_file.parent.glob(ply_file.stem + '_gt_iter_*.json')))
                    
                    # 方式2: 移除常见的后缀 (如移除_highres，变成 0_batch_0_gt_iter_*.json)
                    if ply_file.stem.endswith('_highres'):
                        base_name = ply_file.stem[:-len('_highres')]
                        gt_json_files.extend(list(ply_file.parent.glob(base_name + '_gt_iter_*.json')))
                    
                    # 排除 _pred_iter_*.json 文件
                    gt_json_files = [f for f in gt_json_files if '_pred_iter_' not in f.name]
                    
                    # 去重（如果同一个文件通过不同方式找到）
                    gt_json_files = list(set(gt_json_files))
                    
                    if gt_json_files:
                        # 如果找到多个，使用第一个
                        json_path = str(sorted(gt_json_files)[0])
                
                # 如果既没有tokenized也没有gt_iter文件，跳过该样本
                if json_path is None:
                    continue
            
            gt_files.append({
                'gt_ply_path': str(ply_file),
                'gt_json_path': json_path,
                'gt_name': gt_name,
                'result_paths': result_mapping[gt_name]
            })
    
    print(f"找到 {len(gt_files)} 个有对应结果的GT文件")
    
    # 统计有json的GT数量
    if not disable_json:
        num_with_json = sum(1 for gf in gt_files if gf['gt_json_path'] is not None)
        num_tokenized = sum(1 for gf in gt_files if gf['gt_json_path'] and '_tokenized.json' in gf['gt_json_path'])
        num_gt_iter = sum(1 for gf in gt_files if gf['gt_json_path'] and '_gt_iter_' in gf['gt_json_path'])
        print(f"  其中 {num_with_json} 个有对应的JSON文件")
        if num_tokenized > 0:
            print(f"    - {num_tokenized} 个 tokenized JSON")
        if num_gt_iter > 0:
            print(f"    - {num_gt_iter} 个 gt_iter JSON")
    else:
        print(f"  JSON-based计算已禁用")
    
    if len(gt_files) == 0:
        print("没有找到匹配的GT-结果对，退出")
        return
    
    # 3. 计算Chamfer Distance (两种方法)
    print("\n开始计算Chamfer Distance...")
    if not disable_json:
        print(f"  使用分辨率: {resolution}x{resolution} (用于JSON采样)")
    else:
        print(f"  仅计算PLY-based Chamfer Distance")
    
    # 分别存储两种方法的结果
    # 按GT样本分组存储
    ply_cd_by_sample = {}  # {gt_name: [cd1, cd2, ...]}
    json_cd_by_sample = {}
    ply_results_log = []
    json_results_log = []
    
    for gt_info in tqdm(gt_files, desc="计算CD", leave=True):
        gt_ply_path = gt_info['gt_ply_path']
        gt_json_path = gt_info['gt_json_path']
        gt_name = gt_info['gt_name']
        result_paths = gt_info['result_paths']
        
        print(f"\n{'='*80}")
        print(f"处理GT样本: {gt_name}")
        print(f"  计算模式: {'双向 Chamfer Distance (pred<->gt)' if bidirectional else '单向 Chamfer Distance (pred->gt)'}")
        print(f"  GT PLY: {gt_ply_path}")
        if gt_json_path:
            print(f"  GT JSON: {gt_json_path}")
        print(f"  结果数量: {len(result_paths)}")
        
        # 读取GT数据
        gt_ply_points = None
        gt_json_points = None
        
        # 方法1: 读取PLY点云
        try:
            gt_ply_points = read_ply(gt_ply_path)
            print(f"  ✓ 读取PLY成功: {len(gt_ply_points)} 点")
        except Exception as e:
            print(f"  ✗ 错误: 读取PLY失败: {e}")
        
        # 方法2: 从JSON生成点云（如果存在且未禁用）
        if not disable_json and gt_json_path is not None:
            try:
                gt_json_points = json_to_pointcloud(gt_json_path, resolution)
                if gt_json_points is None:
                    print(f"  ✗ 警告: JSON未能生成点云")
                else:
                    print(f"  ✓ JSON采样成功: {len(gt_json_points)} 点")
            except Exception as e:
                print(f"  ✗ 错误: 处理JSON失败: {e}")

        
        # 对每个结果计算CD
        for idx, result_path in enumerate(result_paths, 1):
            print(f"  [{idx}/{len(result_paths)}] Result: {result_path}")
            try:
                result_points = read_ply(result_path)
                print(f"      读取结果PLY: {len(result_points)} 点")
                
                # 方法1: PLY-based CD
                if gt_ply_points is not None:
                    try:
                        cd_ply = compute_chamfer_distance_fast(result_points, gt_ply_points, bidirectional=bidirectional)
                        
                        # 按样本分组存储
                        if gt_name not in ply_cd_by_sample:
                            ply_cd_by_sample[gt_name] = []
                        ply_cd_by_sample[gt_name].append(cd_ply)
                        
                        ply_results_log.append({
                            'gt_path': gt_ply_path,
                            'gt_name': gt_name,
                            'result_path': result_path,
                            'chamfer_distance': float(cd_ply),
                            'gt_num_points': len(gt_ply_points),
                            'result_num_points': len(result_points),
                            'method': 'ply',
                            'bidirectional': bidirectional
                        })
                        print(f"      PLY-based CD: {cd_ply:.6f}")
                    except Exception as e:
                        print(f"      ✗ PLY-based CD计算失败: {e}")
                
                # 方法2: JSON-based CD
                if gt_json_points is not None:
                    try:
                        cd_json = compute_chamfer_distance_fast(result_points, gt_json_points, bidirectional=bidirectional)
                        
                        # 按样本分组存储
                        if gt_name not in json_cd_by_sample:
                            json_cd_by_sample[gt_name] = []
                        json_cd_by_sample[gt_name].append(cd_json)
                        
                        json_results_log.append({
                            'gt_path': gt_json_path,
                            'gt_name': gt_name,
                            'result_path': result_path,
                            'chamfer_distance': float(cd_json),
                            'gt_num_points': len(gt_json_points),
                            'result_num_points': len(result_points),
                            'method': 'json',
                            'resolution': resolution,
                            'bidirectional': bidirectional
                        })
                        print(f"      JSON-based CD: {cd_json:.6f}")
                    except Exception as e:
                        print(f"      ✗ JSON-based CD计算失败: {e}")
                
                # 渲染三视图对比（如果启用）
                if render_views and ps is not None and gt_ply_points is not None:
                    try:
                        # 创建渲染输出目录（直接在renders下，不创建子文件夹）
                        render_dir = output_dir / "renders"
                        render_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 生成唯一的文件名前缀（使用result路径的父目录名，如"00_03"）
                        result_basename = Path(result_path).parent.name
                        
                        # 渲染GT三视图（在文件名中包含gt_name）
                        gt_base = str(render_dir / f"{gt_name}_gt")
                        gt_paths = render_three_views(gt_ply_points, gt_base)
                        
                        # 渲染结果三视图
                        result_base = str(render_dir / f"{gt_name}_result_{result_basename}")
                        result_paths_dict = render_three_views(result_points, result_base)
                        
                        # 创建2x3网格图
                        grid_output = render_dir / f"{gt_name}_compare_{result_basename}.jpg"
                        create_two_rows_grid(gt_paths, result_paths_dict, str(grid_output))
                        print(f"      渲染保存到: {grid_output}")
                    except Exception as e:
                        print(f"      ✗ 渲染失败: {e}")
                        
            except Exception as e:
                print(f"      ✗ 读取结果失败: {e}")
                continue
        
        # 打印该样本的统计
        if gt_name in ply_cd_by_sample and len(ply_cd_by_sample[gt_name]) > 0:
            ply_cds = ply_cd_by_sample[gt_name]
            print(f"  样本统计 (PLY): min={min(ply_cds):.6f}, avg={np.mean(ply_cds):.6f}, max={max(ply_cds):.6f}")
        if gt_name in json_cd_by_sample and len(json_cd_by_sample[gt_name]) > 0:
            json_cds = json_cd_by_sample[gt_name]
            print(f"  样本统计 (JSON): min={min(json_cds):.6f}, avg={np.mean(json_cds):.6f}, max={max(json_cds):.6f}")
    
    # 4. 统计和保存结果
    print("\n" + "="*70)
    print("Chamfer Distance 统计结果:")
    print("="*70)
    
    def print_and_save_stats(cd_by_sample, log_list, method_name, output_dir, is_bidirectional):
        """
        打印和保存统计信息
        计算三种统计：all (所有CD), min (每样本最小CD), mean (每样本平均CD)
        """
        if len(cd_by_sample) == 0:
            print(f"\n【{method_name}】未计算出任何Chamfer Distance")
            return None
        
        # 计算三种统计数据
        all_cds = []
        min_cds = []
        mean_cds = []
        
        for sample_name, cd_list in cd_by_sample.items():
            all_cds.extend(cd_list)
            min_cds.append(min(cd_list))
            mean_cds.append(np.mean(cd_list))
        
        all_cds = np.array(all_cds)
        min_cds = np.array(min_cds)
        mean_cds = np.array(mean_cds)
        
        # 打印统计信息
        print(f"\n【{method_name}】")
        print(f"  样本数: {len(cd_by_sample)}")
        print(f"  总CD计算数: {len(all_cds)}")
        print(f"  平均每样本结果数: {len(all_cds) / len(cd_by_sample):.2f}")
        
        # All CD统计
        print(f"\n  【All】所有CD值的统计:")
        print(f"    Mean: {np.mean(all_cds):.6f}")
        print(f"    Median: {np.median(all_cds):.6f}")
        print(f"    Std: {np.std(all_cds):.6f}")
        
        # Min CD统计
        print(f"\n  【Best】每样本最小CD的统计:")
        print(f"    Mean: {np.mean(min_cds):.6f}")
        print(f"    Median: {np.median(min_cds):.6f}")
        print(f"    Std: {np.std(min_cds):.6f}")
        
        # Mean CD统计
        print(f"\n  【Avg】每样本平均CD的统计:")
        print(f"    Mean: {np.mean(mean_cds):.6f}")
        print(f"    Median: {np.median(mean_cds):.6f}")
        print(f"    Std: {np.std(mean_cds):.6f}")
        
        # 保存详细结果到JSON
        method_suffix = method_name.lower().replace('-', '_').replace(' ', '_')
        json_output = output_dir / f"chamfer_distances_{method_suffix}_detailed.json"
        with open(json_output, 'w') as f:
            json.dump(log_list, f, indent=2)
        print(f"\n  详细结果已保存到: {json_output}")
        
        # 保存统计摘要
        summary = {
            'method': method_name,
            'bidirectional': is_bidirectional,
            'num_samples': len(cd_by_sample),
            'total_count': len(all_cds),
            'avg_results_per_sample': float(len(all_cds) / len(cd_by_sample)),
            'all_cds': {
                'mean': float(np.mean(all_cds)),
                'median': float(np.median(all_cds)),
                'std': float(np.std(all_cds)),
                'min': float(np.min(all_cds)),
                'max': float(np.max(all_cds)),
                'percentiles': {
                    '25': float(np.percentile(all_cds, 25)),
                    '50': float(np.percentile(all_cds, 50)),
                    '75': float(np.percentile(all_cds, 75)),
                    '90': float(np.percentile(all_cds, 90)),
                    '95': float(np.percentile(all_cds, 95)),
                    '99': float(np.percentile(all_cds, 99)),
                }
            },
            'best_per_sample': {
                'mean': float(np.mean(min_cds)),
                'median': float(np.median(min_cds)),
                'std': float(np.std(min_cds)),
                'min': float(np.min(min_cds)),
                'max': float(np.max(min_cds)),
                'percentiles': {
                    '25': float(np.percentile(min_cds, 25)),
                    '50': float(np.percentile(min_cds, 50)),
                    '75': float(np.percentile(min_cds, 75)),
                    '90': float(np.percentile(min_cds, 90)),
                    '95': float(np.percentile(min_cds, 95)),
                    '99': float(np.percentile(min_cds, 99)),
                }
            },
            'avg_per_sample': {
                'mean': float(np.mean(mean_cds)),
                'median': float(np.median(mean_cds)),
                'std': float(np.std(mean_cds)),
                'min': float(np.min(mean_cds)),
                'max': float(np.max(mean_cds)),
                'percentiles': {
                    '25': float(np.percentile(mean_cds, 25)),
                    '50': float(np.percentile(mean_cds, 50)),
                    '75': float(np.percentile(mean_cds, 75)),
                    '90': float(np.percentile(mean_cds, 90)),
                    '95': float(np.percentile(mean_cds, 95)),
                    '99': float(np.percentile(mean_cds, 99)),
                }
            }
        }
        
        summary_output = output_dir / f"chamfer_distances_{method_suffix}_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  统计摘要已保存到: {summary_output}")
        
        # 保存原始数据（用于绘图）
        np_output_all = output_dir / f"chamfer_distances_{method_suffix}_all.npy"
        np.save(np_output_all, all_cds)
        np_output_min = output_dir / f"chamfer_distances_{method_suffix}_best.npy"
        np.save(np_output_min, min_cds)
        np_output_mean = output_dir / f"chamfer_distances_{method_suffix}_avg.npy"
        np.save(np_output_mean, mean_cds)
        print(f"  原始数据已保存 (all/best/avg)")
        
        return summary
    
    # 统计和保存PLY-based结果
    ply_stats = print_and_save_stats(
        ply_cd_by_sample, 
        ply_results_log, 
        "PLY-based", 
        output_dir,
        bidirectional
    )
    
    # 统计和保存JSON-based结果
    if not disable_json:
        json_stats = print_and_save_stats(
            json_cd_by_sample, 
            json_results_log, 
            "JSON-based", 
            output_dir,
            bidirectional
        )
    else:
        json_stats = None
        print(f"\n【JSON-based】已禁用")
    
    # 保存对比摘要
    if ply_stats is not None or json_stats is not None:
        comparison = {
            'ply_based': ply_stats,
            'json_based': json_stats,
            'resolution': resolution,
            'bidirectional': bidirectional
        }
        
        comparison_output = output_dir / "chamfer_distances_comparison.json"
        with open(comparison_output, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\n对比摘要已保存到: {comparison_output}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="批量计算GT和预测结果之间的Chamfer Distance (支持PLY和JSON两种方法)"
    )
    parser.add_argument(
        '--gt_root',
        type=str,
        default='/home/qindafei/CAD/data/abc_step_pc_correct_normal/00',
        help='GT数据根目录'
    )
    parser.add_argument(
        '--results_root',
        type=str,
        default='/home/qindafei/CAD/HoLa/results/00',
        help='结果数据根目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/qindafei/CAD/CLR-Wire/src/eval/chamfer_results',
        help='输出目录'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        default=16,
        help='JSON表面采样分辨率 (默认: 16x16)'
    )
    parser.add_argument(
        '--use-tokenized-json',
        action='store_true',
        help='优先使用tokenized JSON文件（*_tokenized.json）'
    )
    parser.add_argument(
        '--disable-json',
        action='store_true',
        help='禁用JSON-based计算，只计算PLY-based Chamfer Distance'
    )
    parser.add_argument(
        '--render-views',
        action='store_true',
        help='渲染GT和result的三视图对比（2x3网格）'
    )
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        help='计算双向Chamfer Distance (pred<->gt)；默认只计算单向 (pred->gt)'
    )
    
    args = parser.parse_args()
    
    # 初始化polyscope（如果需要渲染）
    if args.render_views and ps is not None:
        try:
            ps.init("openGL3_egl")
            print("Polyscope初始化成功")
        except Exception as e:
            print(f"警告: Polyscope初始化失败: {e}")
            print("渲染功能将被禁用")
            args.render_views = False
    
    print("="*70)
    if args.disable_json:
        print("Chamfer Distance 批量计算 (仅PLY)")
    else:
        print("Chamfer Distance 批量计算 (PLY + JSON)")
    print("="*70)
    print(f"GT根目录: {args.gt_root}")
    print(f"结果根目录: {args.results_root}")
    print(f"输出目录: {args.output_dir}")
    if not args.disable_json:
        print(f"JSON采样分辨率: {args.resolution}x{args.resolution}")
        print(f"优先使用tokenized JSON: 是")
    else:
        print(f"JSON计算: 已禁用")
    print(f"渲染三视图: {'是' if args.render_views else '否'}")
    print(f"Chamfer Distance模式: {'双向 (pred<->gt)' if args.bidirectional else '单向 (pred->gt)'}")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    compute_chamfer_distances(
        gt_root=args.gt_root,
        results_root=args.results_root,
        output_dir=args.output_dir,
        resolution=args.resolution,
        disable_json=args.disable_json,
        render_views=args.render_views,
        bidirectional=args.bidirectional
    )
    
    print("\n完成！")


if __name__ == "__main__":
    main()

