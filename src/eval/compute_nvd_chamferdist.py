"""
计算NVDNet预测结果和GT之间的Chamfer Distance
只使用PLY点云直接计算
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

try:
    from plyfile import PlyData
except ImportError:
    print("Please install plyfile: pip install plyfile")
    exit(1)


def read_ply(ply_path, max_points=10000):
    """
    读取PLY文件，返回点云坐标 (N, 3)
    如果点数超过max_points，则随机采样max_points个点
    
    Args:
        ply_path: PLY文件路径
        max_points: 最大点数，超过则随机采样 (默认10000)
    
    Returns:
        points: (N, 3) numpy array
    """
    ply_data = PlyData.read(ply_path)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    points = points.astype(np.float32)
    
    # 如果点数超过阈值，随机采样
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    return points


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


def build_pred_gt_pairs(pred_root, gt_root):
    """
    构建pred-gt文件对
    返回: list[dict] with keys: 'gt_path', 'pred_path', 'sample_name'
    """
    print("正在构建pred-gt文件对...")
    
    pred_root = Path(pred_root)
    gt_root = Path(gt_root)
    
    pairs = []
    
    # 遍历gt文件夹中的所有ply文件
    gt_files = sorted(gt_root.glob("*.ply"))
    
    print(f"找到 {len(gt_files)} 个GT文件")
    
    for gt_file in gt_files:
        sample_name = gt_file.stem  # 例如: 0_batch_0_highres
        
        # 构建对应的pred文件路径
        # pred路径: mesh/{sample_name}/mesh/0total_mesh.ply
        pred_file = pred_root / sample_name / "mesh" / "0total_mesh.ply"
        
        if pred_file.exists():
            pairs.append({
                'gt_path': str(gt_file),
                'pred_path': str(pred_file),
                'sample_name': sample_name
            })
        else:
            print(f"警告: 未找到pred文件 {pred_file}")
    
    print(f"成功匹配 {len(pairs)} 对pred-gt文件")
    
    return pairs


def compute_chamfer_distances(pred_root, gt_root, output_dir, bidirectional=False, max_points=10000, num_to_process=None):
    """
    批量计算Chamfer Distance
    
    Args:
        pred_root: pred数据根目录 (mesh文件夹)
        gt_root: GT数据根目录 (poisson文件夹)
        output_dir: 输出目录
        bidirectional: 如果为True，计算双向Chamfer Distance；否则只计算pred->gt单向距离
        max_points: 点云最大点数，超过则随机采样 (默认10000)
        num_to_process: 只处理前N个样本，None表示处理所有样本 (默认None)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 构建pred-gt文件对
    pairs = build_pred_gt_pairs(pred_root, gt_root)
    
    if len(pairs) == 0:
        print("没有找到匹配的pred-gt对，退出")
        return
    
    # 如果指定了num_to_process，只处理前N个样本
    total_pairs = len(pairs)
    if num_to_process is not None and num_to_process > 0:
        pairs = pairs[:num_to_process]
        print(f"\n限制处理数量: 只处理前 {len(pairs)}/{total_pairs} 个样本")
    
    # 2. 计算Chamfer Distance
    print("\n开始计算Chamfer Distance...")
    print(f"  计算模式: {'双向 (pred<->gt)' if bidirectional else '单向 (pred->gt)'}")
    print(f"  点云采样: 超过{max_points}点的将被随机采样到{max_points}点")
    
    # 存储结果
    all_cds = []
    results_log = []
    
    for pair in tqdm(pairs, desc="计算CD", leave=True):
        gt_path = pair['gt_path']
        pred_path = pair['pred_path']
        sample_name = pair['sample_name']
        
        try:
            # 读取GT和pred点云（会自动采样）
            gt_points = read_ply(gt_path, max_points=max_points)
            pred_points = read_ply(pred_path, max_points=max_points)
            
            # 计算Chamfer Distance
            cd = compute_chamfer_distance_fast(pred_points, gt_points, bidirectional=bidirectional)
            
            all_cds.append(cd)
            
            results_log.append({
                'sample_name': sample_name,
                'gt_path': gt_path,
                'pred_path': pred_path,
                'chamfer_distance': float(cd),
                'gt_num_points': len(gt_points),
                'pred_num_points': len(pred_points),
                'bidirectional': bidirectional,
                'note': f'Points sampled to {max_points} if exceeded'
            })
            
        except Exception as e:
            print(f"\n错误: 处理样本 {sample_name} 失败: {e}")
            continue
    
    # 3. 统计和保存结果
    print("\n" + "="*70)
    print("Chamfer Distance 统计结果:")
    print("="*70)
    
    if len(all_cds) == 0:
        print("未计算出任何Chamfer Distance")
        return
    
    all_cds = np.array(all_cds)
    
    # 打印统计信息
    print(f"\n样本数: {len(all_cds)}")
    print(f"\nChamfer Distance统计:")
    print(f"  Mean: {np.mean(all_cds):.6f}")
    print(f"  Median: {np.median(all_cds):.6f}")
    print(f"  Std: {np.std(all_cds):.6f}")
    print(f"  Min: {np.min(all_cds):.6f}")
    print(f"  Max: {np.max(all_cds):.6f}")
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(all_cds, p):.6f}")
    
    # 保存详细结果到JSON
    json_output = output_dir / "chamfer_distances_detailed.json"
    with open(json_output, 'w') as f:
        json.dump(results_log, f, indent=2)
    print(f"\n详细结果已保存到: {json_output}")
    
    # 保存统计摘要
    summary = {
        'method': 'nvdnet_ply_based',
        'bidirectional': bidirectional,
        'max_points': max_points,
        'num_samples': len(all_cds),
        'statistics': {
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
        }
    }
    
    summary_output = output_dir / "chamfer_distances_summary.json"
    with open(summary_output, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"统计摘要已保存到: {summary_output}")
    
    # 保存原始数据（用于绘图）
    np_output = output_dir / "chamfer_distances_all.npy"
    np.save(np_output, all_cds)
    print(f"原始数据已保存到: {np_output}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="批量计算NVDNet预测结果和GT之间的Chamfer Distance"
    )
    parser.add_argument(
        '--pred_root',
        type=str,
        default='/home/qindafei/CAD/baseline_data/test_9_highres_nvdnet/mesh',
        help='预测结果根目录 (mesh文件夹)'
    )
    parser.add_argument(
        '--gt_root',
        type=str,
        default='/home/qindafei/CAD/baseline_data/test_9_highres_nvdnet/poisson',
        help='GT数据根目录 (poisson文件夹)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/qindafei/CAD/CLR-Wire/src/eval/nvdnet_chamfer_results',
        help='输出目录'
    )
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        help='计算双向Chamfer Distance (pred<->gt)；默认只计算单向 (pred->gt)'
    )
    parser.add_argument(
        '--max_points',
        type=int,
        default=10000,
        help='点云最大点数，超过则随机采样 (默认: 10000)'
    )
    parser.add_argument(
        '--num_to_process',
        type=int,
        default=None,
        help='只处理前N个样本，用于测试 (默认: None，处理所有样本)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("NVDNet Chamfer Distance 批量计算")
    print("="*70)
    print(f"Pred根目录: {args.pred_root}")
    print(f"GT根目录: {args.gt_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"Chamfer Distance模式: {'双向 (pred<->gt)' if args.bidirectional else '单向 (pred->gt)'}")
    print(f"点云采样策略: 超过{args.max_points}点将随机采样到{args.max_points}点")
    if args.num_to_process is not None:
        print(f"处理数量限制: 只处理前 {args.num_to_process} 个样本")
    else:
        print(f"处理数量限制: 处理所有样本")
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    compute_chamfer_distances(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        output_dir=args.output_dir,
        bidirectional=args.bidirectional,
        max_points=args.max_points,
        num_to_process=args.num_to_process
    )
    
    print("\n完成！")


if __name__ == "__main__":
    main()

