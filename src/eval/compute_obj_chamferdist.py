"""
计算GT和预测结果之间的Chamfer Distance
预先构建路径映射表以提高效率
"""

import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import json

try:
    from plyfile import PlyData
except ImportError:
    print("Please install plyfile: pip install plyfile")
    exit(1)


def read_ply(ply_path):
    """读取PLY文件，返回点云坐标 (N, 3)"""
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    return points.astype(np.float32)


def chamfer_distance(points1, points2):
    """
    计算两个点云之间的Chamfer Distance
    points1: (N1, 3)
    points2: (N2, 3)
    """
    # 计算 points1 到 points2 的最近距离
    # 使用分块计算避免内存溢出
    chunk_size = 1000
    dist1_sum = 0.0
    
    for i in range(0, len(points1), chunk_size):
        chunk = points1[i:i+chunk_size]
        # (chunk_size, 1, 3) - (1, N2, 3) -> (chunk_size, N2, 3)
        diff = chunk[:, None, :] - points2[None, :, :]
        dist = np.sum(diff ** 2, axis=2)  # (chunk_size, N2)
        min_dist = np.min(dist, axis=1)  # (chunk_size,)
        dist1_sum += np.sum(min_dist)
    
    # 计算 points2 到 points1 的最近距离
    dist2_sum = 0.0
    for i in range(0, len(points2), chunk_size):
        chunk = points2[i:i+chunk_size]
        diff = chunk[:, None, :] - points1[None, :, :]
        dist = np.sum(diff ** 2, axis=2)
        min_dist = np.min(dist, axis=1)
        dist2_sum += np.sum(min_dist)
    
    # Chamfer Distance = 平均双向最近点距离
    cd = (dist1_sum / len(points1) + dist2_sum / len(points2)) / 2.0
    
    return cd


def build_result_path_mapping(results_root):
    """
    构建结果路径映射表
    返回: dict[gt_filename] -> list[result_ply_paths]
    """
    print("正在构建结果路径映射表...")
    mapping = defaultdict(list)
    
    results_root = Path(results_root)
    
    # 遍历所有 process_X 文件夹
    for process_dir in sorted(results_root.glob("process_*")):
        brep_results_dir = process_dir / "brep_results"
        if not brep_results_dir.exists():
            continue
        
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


def compute_chamfer_distances(gt_root, results_root, output_dir):
    """
    批量计算Chamfer Distance
    """
    gt_root = Path(gt_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 构建结果路径映射表
    result_mapping = build_result_path_mapping(results_root)
    
    # 2. 收集所有GT文件
    print("\n正在收集GT文件...")
    gt_files = []
    
    for index_dir in sorted(gt_root.iterdir()):
        if not index_dir.is_dir():
            continue
        
        # 跳过空文件夹（如00000000）
        ply_files = list(index_dir.glob("*.ply"))
        if not ply_files:
            continue
        
        for ply_file in ply_files:
            gt_name = ply_file.stem  # 去掉 .ply 扩展名
            
            # 检查是否有对应的结果
            if gt_name in result_mapping:
                gt_files.append({
                    'gt_path': str(ply_file),
                    'gt_name': gt_name,
                    'result_paths': result_mapping[gt_name]
                })
    
    print(f"找到 {len(gt_files)} 个有对应结果的GT文件")
    
    if len(gt_files) == 0:
        print("没有找到匹配的GT-结果对，退出")
        return
    
    # 3. 计算Chamfer Distance
    print("\n开始计算Chamfer Distance...")
    all_chamfer_distances = []
    results_log = []
    
    for gt_info in tqdm(gt_files, desc="计算CD"):
        gt_path = gt_info['gt_path']
        gt_name = gt_info['gt_name']
        result_paths = gt_info['result_paths']
        
        try:
            # 读取GT点云
            gt_points = read_ply(gt_path)
            
            # 计算每个结果的CD
            for result_path in result_paths:
                try:
                    result_points = read_ply(result_path)
                    cd = chamfer_distance(gt_points, result_points)
                    
                    all_chamfer_distances.append(cd)
                    results_log.append({
                        'gt_path': gt_path,
                        'gt_name': gt_name,
                        'result_path': result_path,
                        'chamfer_distance': float(cd),
                        'gt_num_points': len(gt_points),
                        'result_num_points': len(result_points)
                    })
                    
                except Exception as e:
                    print(f"\n错误: 计算 {result_path} 时失败: {e}")
                    continue
                    
        except Exception as e:
            print(f"\n错误: 读取 {gt_path} 时失败: {e}")
            continue
    
    # 4. 统计和保存结果
    print("\n" + "="*60)
    print("Chamfer Distance 统计结果:")
    print("="*60)
    
    if len(all_chamfer_distances) > 0:
        cd_array = np.array(all_chamfer_distances)
        
        print(f"总计算数: {len(cd_array)}")
        print(f"平均值 (Mean): {np.mean(cd_array):.6f}")
        print(f"中位数 (Median): {np.median(cd_array):.6f}")
        print(f"标准差 (Std): {np.std(cd_array):.6f}")
        print(f"最小值 (Min): {np.min(cd_array):.6f}")
        print(f"最大值 (Max): {np.max(cd_array):.6f}")
        print(f"\n百分位数:")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"  {p}%: {np.percentile(cd_array, p):.6f}")
        
        # 保存详细结果到JSON
        json_output = output_dir / "chamfer_distances_detailed.json"
        with open(json_output, 'w') as f:
            json.dump(results_log, f, indent=2)
        print(f"\n详细结果已保存到: {json_output}")
        
        # 保存统计摘要
        summary = {
            'total_count': len(cd_array),
            'mean': float(np.mean(cd_array)),
            'median': float(np.median(cd_array)),
            'std': float(np.std(cd_array)),
            'min': float(np.min(cd_array)),
            'max': float(np.max(cd_array)),
            'percentiles': {
                '25': float(np.percentile(cd_array, 25)),
                '50': float(np.percentile(cd_array, 50)),
                '75': float(np.percentile(cd_array, 75)),
                '90': float(np.percentile(cd_array, 90)),
                '95': float(np.percentile(cd_array, 95)),
                '99': float(np.percentile(cd_array, 99)),
            }
        }
        
        summary_output = output_dir / "chamfer_distances_summary.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"统计摘要已保存到: {summary_output}")
        
        # 保存原始数据（用于绘图）
        np_output = output_dir / "chamfer_distances.npy"
        np.save(np_output, cd_array)
        print(f"原始数据已保存到: {np_output}")
        
    else:
        print("未计算出任何Chamfer Distance")


def main():
    parser = argparse.ArgumentParser(
        description="批量计算GT和预测结果之间的Chamfer Distance"
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
    
    args = parser.parse_args()
    
    print("="*60)
    print("Chamfer Distance 批量计算")
    print("="*60)
    print(f"GT根目录: {args.gt_root}")
    print(f"结果根目录: {args.results_root}")
    print(f"输出目录: {args.output_dir}")
    print("="*60)
    
    compute_chamfer_distances(
        gt_root=args.gt_root,
        results_root=args.results_root,
        output_dir=args.output_dir
    )
    
    print("\n完成！")


if __name__ == "__main__":
    main()

