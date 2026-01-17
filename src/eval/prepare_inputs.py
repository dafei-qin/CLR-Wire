"""
准备输入数据：从OBJ文件采样带normal的点云并保存为PLY格式
功能：
1. 遍历输入文件夹的所有obj文件
2. 随机采样20480个点，带normal
3. normalize其bbox到[-1, 1]
4. 保存ply带normal到同路径的同名称+.ply
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:
    print("Error: Please install open3d: pip install open3d")
    sys.exit(1)


def normalize_points(points):
    """
    将点云normalize到[-1, 1]的bounding box内
    
    Args:
        points: numpy array of shape (N, 3)
    
    Returns:
        normalized points: numpy array of shape (N, 3)
    """
    # 计算bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    
    # 计算中心点
    center = (min_coords + max_coords) / 2.0
    
    # 计算最大范围
    max_range = np.max(max_coords - min_coords)
    
    # 避免除零
    if max_range < 1e-8:
        return points - center
    
    # normalize到[-1, 1]
    normalized_points = (points - center) / (max_range / 2.0)
    
    return normalized_points


def sample_point_cloud_from_mesh(mesh_path, num_samples=20480):
    """
    从mesh文件采样带normal的点云
    
    Args:
        mesh_path: mesh文件路径
        num_samples: 采样点数量，默认20480
    
    Returns:
        points: numpy array of shape (N, 3)
        normals: numpy array of shape (N, 3)
    """
    # 读取mesh
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    if not mesh.has_triangles():
        raise ValueError(f"Mesh {mesh_path} has no triangles")
    
    # 如果mesh没有顶点法向量，计算它们
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # 采样点云（带normal）
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    # 如果采样后的点云没有法向量，从mesh估计
    if not pcd.has_normals():
        # 使用Poisson surface reconstruction的方式估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        # 统一法向量方向
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # 转换为numpy数组
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    return points, normals


def process_obj_file(obj_path, num_samples=20480, output_path=None):
    """
    处理单个OBJ文件：采样、normalize、保存
    
    Args:
        obj_path: OBJ文件路径
        num_samples: 采样点数量，默认20480
        output_path: 输出PLY文件路径，如果为None则保存在同路径下
    
    Returns:
        success: 是否成功处理
    """
    try:
        # 采样点云
        points, normals = sample_point_cloud_from_mesh(obj_path, num_samples)
        
        # Normalize到[-1, 1]
        normalized_points = normalize_points(points)
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(normalized_points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # 确定输出路径
        if output_path is None:
            output_path = Path(obj_path).with_suffix('.ply')
        else:
            output_path = Path(output_path)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存PLY文件
        o3d.io.write_point_cloud(str(output_path), pcd)
        
        return True
        
    except Exception as e:
        print(f"\nError processing {obj_path}: {str(e)}")
        return False


def process_folder(input_folder, num_samples=20480, recursive=True):
    """
    遍历文件夹处理所有OBJ文件
    
    Args:
        input_folder: 输入文件夹路径
        num_samples: 采样点数量，默认20480
        recursive: 是否递归处理子文件夹
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"Error: Input folder {input_folder} does not exist")
        return
    
    # 查找所有OBJ文件
    if recursive:
        obj_files = list(input_path.rglob("*.obj"))
    else:
        obj_files = list(input_path.glob("*.obj"))
    
    if len(obj_files) == 0:
        print(f"No OBJ files found in {input_folder}")
        return
    
    print(f"Found {len(obj_files)} OBJ files")
    
    # 统计结果
    success_count = 0
    fail_count = 0
    
    # 处理每个OBJ文件
    for obj_file in tqdm(obj_files, desc="Processing OBJ files"):
        success = process_obj_file(obj_file, num_samples)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Total files: {len(obj_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="从OBJ文件采样带normal的点云并保存为PLY格式"
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="输入文件夹路径（包含OBJ文件）"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20480,
        help="采样点数量（默认：20480）"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="不递归处理子文件夹"
    )
    
    args = parser.parse_args()
    
    # 处理文件夹
    process_folder(
        args.input_folder,
        num_samples=args.num_samples,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()

