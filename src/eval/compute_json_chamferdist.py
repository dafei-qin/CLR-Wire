"""Compute Chamfer Distance between pred/gt JSON files (all surfaces as one point cloud)."""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import argparse
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from tqdm import tqdm

# Import chamferdist library
try:
    from chamferdist import ChamferDistance
except ImportError:
    print("Installing chamferdist library...")
    import subprocess
    subprocess.check_call(["pip", "install", "chamferdist"])
    from chamferdist import ChamferDistance

# Import params_to_samples from surface_tools
from src.utils.surface_tools import params_to_samples


def load_json_surfaces(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        surfaces = json.load(f)
    return surfaces


def surface_to_samples(surface: Dict, resolution: int = 16) -> np.ndarray:
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



def compute_chamfer_distance_fast(pred_points, gt_points):
    """
    快速计算两个点云之间的 Chamfer Distance
    
    Args:
        pred_points: (N, 3) numpy array 或 torch tensor
        gt_points: (M, 3) numpy array 或 torch tensor
    
    Returns:
        chamfer_dist: 双向 Chamfer Distance
    """
    try:
        # 转换为 torch tensor（在 GPU 上）
        if not torch.is_tensor(pred_points):
            pred_points = torch.from_numpy(pred_points).float()
        if not torch.is_tensor(gt_points):
            gt_points = torch.from_numpy(gt_points).float()
        
        # 确保在 GPU 上
        if not pred_points.is_cuda:
            pred_points = pred_points.cuda()
        if not gt_points.is_cuda:
            gt_points = gt_points.cuda()
        
        # 计算双向 Chamfer Distance
        # pred -> gt
        dist_matrix = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)  # (N, M)
        min_dist_pred_to_gt = dist_matrix.min(dim=1)[0]  # (N,)
        
        # gt -> pred
        min_dist_gt_to_pred = dist_matrix.min(dim=0)[0]  # (M,)
        
        # Chamfer Distance (双向平均)
        chamfer_dist = (min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()).item()
        
        return chamfer_dist
    
    except Exception as e:
        return float('inf')


def compute_chamfer_distance(pred_samples: np.ndarray, gt_samples: np.ndarray) -> float:
    pred_tensor = torch.from_numpy(pred_samples).float().unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_samples).float().unsqueeze(0)
    # chamfer_dist_fn = ChamferDistance()
    # dist = chamfer_dist_fn(pred_tensor, gt_tensor)
    return compute_chamfer_distance_fast(pred_samples, gt_samples)


def get_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return points.min(axis=0), points.max(axis=0)

def process_json_pair(pred_surfaces: List[Dict], gt_surfaces: List[Dict], resolution: int = 16) -> Tuple[float, np.ndarray, np.ndarray]:
    try:
        pred_points_list = []
        for pred_surf in pred_surfaces:
            samples = surface_to_samples(pred_surf, resolution)
            pred_points_list.append(samples.reshape(-1, 3))
        
        gt_points_list = []
        for gt_surf in gt_surfaces:
            samples = surface_to_samples(gt_surf, resolution)
            gt_points_list.append(samples.reshape(-1, 3))
        
        pred_all = np.concatenate(pred_points_list, axis=0)
        gt_all = np.concatenate(gt_points_list, axis=0)
        
        chamfer_dist = compute_chamfer_distance(pred_all, gt_all)
        
        return chamfer_dist, pred_all, gt_all
    
    except Exception as e:
        print(f"Error processing JSON pair: {e}")
        return None, None, None


def parse_filename(filename: str) -> Tuple[str, str]:
    """Parse filename to extract idx and batch_idx. Returns (idx, batch_idx)."""
    parts = filename.split('_batch_')
    idx = parts[0]
    batch_idx = parts[1].split('_')[0]
    return idx, batch_idx

def find_json_pairs(test_dir: str, checkpoint: str) -> Dict[str, List[Tuple[str, str]]]:
    """Find all pred/gt pairs grouped by idx."""
    test_path = Path(test_dir)
    gt_pattern = f"*_gt_iter_{checkpoint}.json"
    gt_files = sorted(test_path.glob(gt_pattern))
    
    pairs_by_idx = {}
    for gt_file in gt_files:
        pred_file = gt_file.parent / gt_file.name.replace("_gt_iter_", "_pred_iter_")
        if pred_file.exists():
            idx, batch_idx = parse_filename(gt_file.name)
            if idx not in pairs_by_idx:
                pairs_by_idx[idx] = []
            pairs_by_idx[idx].append((str(pred_file), str(gt_file)))
        else:
            print(f"Warning: No pred file found for {gt_file.name}")
    
    return pairs_by_idx


def compute_chamfer_distances_for_checkpoint(
    test_dir: str, 
    checkpoint: str, 
    resolution: int = 16
) -> Dict[str, float]:
    pairs_by_idx = find_json_pairs(test_dir, checkpoint)
    
    if not pairs_by_idx:
        print(f"No valid pred/gt pairs found for checkpoint {checkpoint}")
        return {}
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_idx.values())
    print(f"Found {len(pairs_by_idx)} unique indices with {total_pairs} total pairs")
    
    min_chamfer_dists = []
    
    for idx in tqdm(sorted(pairs_by_idx.keys(), key=lambda x: int(x)), desc="Processing indices"):
        pairs = pairs_by_idx[idx]
        idx_chamfer_dists = []
        
        print(f"\nIdx {idx} ({len(pairs)} batch variants):")
        for pred_path, gt_path in pairs:
            pred_surfaces = load_json_surfaces(pred_path)
            gt_surfaces = load_json_surfaces(gt_path)
            
            chamfer_dist, pred_points, gt_points = process_json_pair(pred_surfaces, gt_surfaces, resolution)
            
            if chamfer_dist is not None and pred_points is not None and gt_points is not None:
                gt_bbox_min, gt_bbox_max = get_bbox(gt_points)
                pred_bbox_min, pred_bbox_max = get_bbox(pred_points)
                
                gt_name = os.path.basename(gt_path)
                pred_name = os.path.basename(pred_path)
                
                gt_bbox_str = f"[{gt_bbox_min[0]:.3f},{gt_bbox_min[1]:.3f},{gt_bbox_min[2]:.3f}]-[{gt_bbox_max[0]:.3f},{gt_bbox_max[1]:.3f},{gt_bbox_max[2]:.3f}]"
                pred_bbox_str = f"[{pred_bbox_min[0]:.3f},{pred_bbox_min[1]:.3f},{pred_bbox_min[2]:.3f}]-[{pred_bbox_max[0]:.3f},{pred_bbox_max[1]:.3f},{pred_bbox_max[2]:.3f}]"
                
                print(f"  {gt_name} {gt_bbox_str} {pred_name} {pred_bbox_str} CD={chamfer_dist:.6f}")
                idx_chamfer_dists.append(chamfer_dist)
        
        if idx_chamfer_dists:
            min_cd = min(idx_chamfer_dists)
            min_chamfer_dists.append(min_cd)
            print(f"  -> Min CD for idx {idx}: {min_cd:.6f}")
    
    if not min_chamfer_dists:
        print("No valid Chamfer distances computed!")
        return {}
    
    chamfer_array = np.array(min_chamfer_dists)
    
    stats = {
        'mean': float(np.mean(chamfer_array)),
        'median': float(np.median(chamfer_array)),
        'std': float(np.std(chamfer_array)),
        'min': float(np.min(chamfer_array)),
        'max': float(np.max(chamfer_array)),
        'percentile_5': float(np.percentile(chamfer_array, 5)),
        'percentile_95': float(np.percentile(chamfer_array, 95)),
        'num_indices': len(min_chamfer_dists),
        'total_pairs': total_pairs
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute Chamfer Distance between pred and gt JSON files.')
    parser.add_argument('--test_dir', type=str,
        default='/deemos-research-area-d/meshgen/cad/checkpoints/GPT_INIT_142M/train_0110_michel_4096_full/test_00')
    parser.add_argument('--checkpoint', type=str, default='230000')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    print(f"Checkpoint: {args.checkpoint}, Resolution: {args.resolution}x{args.resolution}\n")
    
    stats = compute_chamfer_distances_for_checkpoint(args.test_dir, args.checkpoint, args.resolution)
    
    if stats:
        print(f"\n{'='*60}")
        print(f"Checkpoint {args.checkpoint} - Unique indices: {stats['num_indices']} (from {stats['total_pairs']} pairs)")
        print(f"Mean: {stats['mean']:.6f} | Median: {stats['median']:.6f} | Std: {stats['std']:.6f}")
        print(f"Min: {stats['min']:.6f} | Max: {stats['max']:.6f}")
        print(f"5th: {stats['percentile_5']:.6f} | 95th: {stats['percentile_95']:.6f}")
        print(f"{'='*60}")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()



'''
080000
140000
185000
230000
253500'''
