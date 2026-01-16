"""Compute Type Matching Rate between pred/gt JSON files."""
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

# Import params_to_samples from surface_tools
from src.utils.surface_tools import params_to_samples


def load_json_surfaces(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        surfaces = json.load(f)
    return surfaces


def surface_to_samples(surface: Dict, resolution: int = 16) -> np.ndarray:
    """Convert surface to samples without modifying input dict."""
    surface_type = surface['type']
    
    # Create a copy to avoid modifying the original
    surface_copy = surface.copy()
    
    if surface_type == 'bspline_surface':
        surface_copy['poles'] = np.array(surface_copy['poles'])
    else:
        surface_copy['location'] = torch.tensor(surface_copy['location'])
        surface_copy['direction'] = torch.tensor(surface_copy['direction'])
        surface_copy['scalar'] = torch.tensor(surface_copy['scalar'])
        surface_copy['uv'] = torch.tensor(surface_copy['uv'])

    samples = params_to_samples(
        torch.zeros([]),
        surface_copy['type'],
        resolution,
        resolution,
        surface_json=surface_copy
    )
    
    if isinstance(samples, torch.Tensor):
        samples = samples.squeeze(0).numpy() if samples.dim() > 3 else samples.numpy()
    
    return samples


def surfaces_to_typed_points(surfaces: List[Dict], resolution: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert all surfaces to point cloud with type labels.
    
    Args:
        surfaces: List of surface dictionaries
        resolution: Sampling resolution for each surface
    
    Returns:
        points: (N, 3) numpy array of all sampled points
        types: (N,) numpy array of surface type strings
    """
    points_list = []
    types_list = []
    
    for surface in surfaces:
        surface_type = surface['type']
        samples = surface_to_samples(surface, resolution)
        samples_flat = samples.reshape(-1, 3)
        
        points_list.append(samples_flat)
        types_list.extend([surface_type] * len(samples_flat))
    
    all_points = np.concatenate(points_list, axis=0)
    all_types = np.array(types_list)
    
    return all_points, all_types


def get_bbox_diagonal(points):
    """Compute diagonal length of bounding box for normalization."""
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diagonal = np.linalg.norm(bbox_max - bbox_min)
    return diagonal


def compute_type_matching_symmetric(pred_points, pred_types, gt_points, gt_types, 
                                   distance_threshold=None, relative_threshold=0.01):
    """
    Compute symmetric surface matching metrics: Precision, Recall, F1.
    
    Surface Precision (SP): pred points that can be explained by GT surfaces
    Surface Recall (SR): GT points that can be explained by pred surfaces
    Surface F1: harmonic mean of SP and SR
    
    Args:
        pred_points: (N, 3) numpy array
        pred_types: (N,) numpy array of type strings
        gt_points: (M, 3) numpy array
        gt_types: (M,) numpy array of type strings
        distance_threshold: Absolute distance threshold. If None, use relative_threshold
        relative_threshold: Relative threshold as fraction of bbox diagonal (default 0.01)
    
    Returns:
        metrics: Dict with precision, recall, f1, and detailed stats
    """
    try:
        # Convert to torch tensors
        if not torch.is_tensor(pred_points):
            pred_points_tensor = torch.from_numpy(pred_points).float()
        else:
            pred_points_tensor = pred_points.float()
            
        if not torch.is_tensor(gt_points):
            gt_points_tensor = torch.from_numpy(gt_points).float()
        else:
            gt_points_tensor = gt_points.float()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            pred_points_tensor = pred_points_tensor.cuda()
            gt_points_tensor = gt_points_tensor.cuda()
        
        # Determine distance threshold
        if distance_threshold is None:
            # Compute bbox diagonal from combined point cloud
            all_points = np.concatenate([pred_points, gt_points], axis=0)
            bbox_diag = get_bbox_diagonal(all_points)
            threshold = bbox_diag * relative_threshold
        else:
            threshold = distance_threshold
        
        # Compute distance matrix
        dist_matrix = torch.cdist(pred_points_tensor.unsqueeze(0), gt_points_tensor.unsqueeze(0), p=2).squeeze(0)  # (N, M)
        
        # ========== Precision: pred → gt ==========
        # Find nearest gt point for each pred point
        min_dist_pred_to_gt, nearest_gt_indices = dist_matrix.min(dim=1)  # (N,)
        min_dist_pred_to_gt = min_dist_pred_to_gt.cpu().numpy()
        nearest_gt_indices = nearest_gt_indices.cpu().numpy()
        
        # Get matched gt types
        matched_gt_types = gt_types[nearest_gt_indices]
        
        # Check both distance and type matching
        distance_valid = (min_dist_pred_to_gt < threshold)
        type_matches = (pred_types == matched_gt_types)
        full_matches_pred = distance_valid & type_matches
        
        num_matched_pred = full_matches_pred.sum()
        precision = num_matched_pred / len(pred_points) if len(pred_points) > 0 else 0.0
        
        # ========== Recall: gt → pred ==========
        # Find nearest pred point for each gt point
        min_dist_gt_to_pred, nearest_pred_indices = dist_matrix.min(dim=0)  # (M,)
        min_dist_gt_to_pred = min_dist_gt_to_pred.cpu().numpy()
        nearest_pred_indices = nearest_pred_indices.cpu().numpy()
        
        # Get matched pred types
        matched_pred_types = pred_types[nearest_pred_indices]
        
        # Check both distance and type matching
        distance_valid_gt = (min_dist_gt_to_pred < threshold)
        type_matches_gt = (gt_types == matched_pred_types)
        full_matches_gt = distance_valid_gt & type_matches_gt
        
        num_matched_gt = full_matches_gt.sum()
        recall = num_matched_gt / len(gt_points) if len(gt_points) > 0 else 0.0
        
        # ========== F1 Score ==========
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        # ========== Per-type statistics (for precision) ==========
        per_type_stats = {}
        unique_pred_types = np.unique(pred_types)
        
        for surf_type in unique_pred_types:
            mask = (pred_types == surf_type)
            type_pred_count = mask.sum()
            type_matched_count = full_matches_pred[mask].sum()
            type_precision = type_matched_count / type_pred_count if type_pred_count > 0 else 0.0
            
            per_type_stats[surf_type] = {
                'pred_count': int(type_pred_count),
                'matched_count': int(type_matched_count),
                'precision': float(type_precision)
            }
        
        # ========== Per-type statistics (for recall) ==========
        per_type_recall = {}
        unique_gt_types = np.unique(gt_types)
        
        for surf_type in unique_gt_types:
            mask = (gt_types == surf_type)
            type_gt_count = mask.sum()
            type_matched_count = full_matches_gt[mask].sum()
            type_recall = type_matched_count / type_gt_count if type_gt_count > 0 else 0.0
            
            per_type_recall[surf_type] = {
                'gt_count': int(type_gt_count),
                'matched_count': int(type_matched_count),
                'recall': float(type_recall)
            }
        
        metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'num_matched_pred': int(num_matched_pred),
            'num_matched_gt': int(num_matched_gt),
            'threshold_used': float(threshold),
            'per_type_precision': per_type_stats,
            'per_type_recall': per_type_recall
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error in compute_type_matching_symmetric: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_matched_pred': 0,
            'num_matched_gt': 0,
            'threshold_used': 0.0,
            'per_type_precision': {},
            'per_type_recall': {}
        }


def process_json_pair(pred_surfaces: List[Dict], gt_surfaces: List[Dict], 
                      resolution: int = 16, distance_threshold=None, 
                      relative_threshold=0.01) -> Dict:
    """
    Process a pair of pred/gt JSON files with symmetric metrics.
    
    Args:
        pred_surfaces: List of predicted surfaces
        gt_surfaces: List of GT surfaces
        resolution: Sampling resolution
        distance_threshold: Absolute distance threshold
        relative_threshold: Relative threshold as fraction of bbox diagonal
    
    Returns:
        metrics: Dict containing precision, recall, f1, and detailed statistics
    """
    try:
        pred_points, pred_types = surfaces_to_typed_points(pred_surfaces, resolution)
        gt_points, gt_types = surfaces_to_typed_points(gt_surfaces, resolution)
        
        metrics = compute_type_matching_symmetric(
            pred_points, pred_types, gt_points, gt_types,
            distance_threshold, relative_threshold
        )
        
        # Add point counts
        metrics['num_pred_points'] = len(pred_points)
        metrics['num_gt_points'] = len(gt_points)
        
        return metrics
    
    except Exception as e:
        print(f"Error processing JSON pair: {e}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'num_matched_pred': 0,
            'num_matched_gt': 0,
            'num_pred_points': 0,
            'num_gt_points': 0,
            'threshold_used': 0.0,
            'per_type_precision': {},
            'per_type_recall': {}
        }


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


def aggregate_per_type_stats(all_metrics: List[Dict]) -> Dict:
    """
    Aggregate per-type statistics across all idx results.
    
    Args:
        all_metrics: List of metrics dicts from each idx
    
    Returns:
        Global per-type statistics with precision and recall
    """
    # Aggregate precision stats
    precision_aggregates = {}
    for metrics in all_metrics:
        for surf_type, stats in metrics['per_type_precision'].items():
            if surf_type not in precision_aggregates:
                precision_aggregates[surf_type] = {
                    'total_pred': 0,
                    'total_matched': 0,
                    'precisions': []
                }
            
            precision_aggregates[surf_type]['total_pred'] += stats['pred_count']
            precision_aggregates[surf_type]['total_matched'] += stats['matched_count']
            precision_aggregates[surf_type]['precisions'].append(stats['precision'])
    
    # Aggregate recall stats
    recall_aggregates = {}
    for metrics in all_metrics:
        for surf_type, stats in metrics['per_type_recall'].items():
            if surf_type not in recall_aggregates:
                recall_aggregates[surf_type] = {
                    'total_gt': 0,
                    'total_matched': 0,
                    'recalls': []
                }
            
            recall_aggregates[surf_type]['total_gt'] += stats['gt_count']
            recall_aggregates[surf_type]['total_matched'] += stats['matched_count']
            recall_aggregates[surf_type]['recalls'].append(stats['recall'])
    
    # Compute global statistics for each type
    all_types = set(list(precision_aggregates.keys()) + list(recall_aggregates.keys()))
    global_type_stats = {}
    
    for surf_type in all_types:
        stats = {}
        
        # Precision stats
        if surf_type in precision_aggregates:
            p_agg = precision_aggregates[surf_type]
            stats['total_pred_points'] = p_agg['total_pred']
            stats['total_matched_pred_points'] = p_agg['total_matched']
            stats['global_precision'] = p_agg['total_matched'] / p_agg['total_pred'] if p_agg['total_pred'] > 0 else 0.0
            stats['mean_precision_per_idx'] = float(np.mean(p_agg['precisions']))
            stats['std_precision_per_idx'] = float(np.std(p_agg['precisions']))
        else:
            stats['total_pred_points'] = 0
            stats['total_matched_pred_points'] = 0
            stats['global_precision'] = 0.0
            stats['mean_precision_per_idx'] = 0.0
            stats['std_precision_per_idx'] = 0.0
        
        # Recall stats
        if surf_type in recall_aggregates:
            r_agg = recall_aggregates[surf_type]
            stats['total_gt_points'] = r_agg['total_gt']
            stats['total_matched_gt_points'] = r_agg['total_matched']
            stats['global_recall'] = r_agg['total_matched'] / r_agg['total_gt'] if r_agg['total_gt'] > 0 else 0.0
            stats['mean_recall_per_idx'] = float(np.mean(r_agg['recalls']))
            stats['std_recall_per_idx'] = float(np.std(r_agg['recalls']))
        else:
            stats['total_gt_points'] = 0
            stats['total_matched_gt_points'] = 0
            stats['global_recall'] = 0.0
            stats['mean_recall_per_idx'] = 0.0
            stats['std_recall_per_idx'] = 0.0
        
        # F1 score
        if stats['global_precision'] + stats['global_recall'] > 0:
            stats['global_f1'] = 2 * stats['global_precision'] * stats['global_recall'] / (stats['global_precision'] + stats['global_recall'])
        else:
            stats['global_f1'] = 0.0
        
        global_type_stats[surf_type] = stats
    
    return global_type_stats


def compute_type_matching_for_checkpoint(
    test_dir: str, 
    checkpoint: str, 
    resolution: int = 16,
    range_min: int = None,
    range_max: int = None,
    distance_threshold=None,
    relative_threshold=0.01
) -> Tuple[Dict, List[Dict], Dict]:
    """
    Compute type matching rates for all pred/gt pairs in checkpoint.
    
    Args:
        test_dir: Directory containing test files
        checkpoint: Checkpoint iteration number
        resolution: Sampling resolution
        range_min: Minimum idx to process (inclusive), None means no minimum
        range_max: Maximum idx to process (inclusive), None means no maximum
        distance_threshold: Absolute distance threshold for matching
        relative_threshold: Relative threshold as fraction of bbox diagonal
    
    Returns:
        stats: Overall statistics (precision, recall, f1)
        idx_results: Results for each idx (sorted by F1 descending)
        global_type_stats: Global per-type statistics
    """
    pairs_by_idx = find_json_pairs(test_dir, checkpoint)
    
    if not pairs_by_idx:
        print(f"No valid pred/gt pairs found for checkpoint {checkpoint}")
        return {}, [], {}
    
    # Filter by range if specified
    if range_min is not None or range_max is not None:
        filtered_pairs = {}
        for idx, pairs in pairs_by_idx.items():
            idx_int = int(idx)
            if range_min is not None and idx_int < range_min:
                continue
            if range_max is not None and idx_int > range_max:
                continue
            filtered_pairs[idx] = pairs
        pairs_by_idx = filtered_pairs
        
        if not pairs_by_idx:
            print(f"No indices found in range [{range_min}, {range_max}]")
            return {}, [], {}
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_idx.values())
    range_str = f" (range: {range_min if range_min is not None else 'start'} - {range_max if range_max is not None else 'end'})" if (range_min is not None or range_max is not None) else ""
    print(f"Found {len(pairs_by_idx)} unique indices with {total_pairs} total pairs{range_str}")
    
    idx_results = []
    
    for idx in tqdm(sorted(pairs_by_idx.keys(), key=lambda x: int(x)), desc="Processing indices"):
        pairs = pairs_by_idx[idx]
        idx_metrics_list = []
        idx_file_info = []
        
        print(f"\nIdx {idx} ({len(pairs)} batch variants):")
        for pred_path, gt_path in pairs:
            pred_surfaces = load_json_surfaces(pred_path)
            gt_surfaces = load_json_surfaces(gt_path)
            
            metrics = process_json_pair(
                pred_surfaces, gt_surfaces, resolution,
                distance_threshold, relative_threshold
            )
            
            gt_name = os.path.basename(gt_path)
            pred_name = os.path.basename(pred_path)
            
            print(f"  {gt_name} (GT:{metrics['num_gt_points']}pts) {pred_name} (Pred:{metrics['num_pred_points']}pts)")
            print(f"    P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f} τ={metrics['threshold_used']:.6f}")
            
            idx_metrics_list.append(metrics['f1'])
            idx_file_info.append({
                'gt_path': gt_path,
                'pred_path': pred_path,
                'metrics': metrics
            })
        
        if idx_metrics_list:
            # Select the variant with HIGHEST F1 score
            max_idx = np.argmax(idx_metrics_list)
            best_metrics = idx_file_info[max_idx]['metrics']
            
            result = {
                'idx': idx,
                'gt_path': idx_file_info[max_idx]['gt_path'],
                'pred_path': idx_file_info[max_idx]['pred_path'],
            }
            result.update(best_metrics)
            
            idx_results.append(result)
            print(f"  -> Best F1 for idx {idx}: {best_metrics['f1']:.4f} (P={best_metrics['precision']:.4f}, R={best_metrics['recall']:.4f})")
    
    if not idx_results:
        print("No valid matching rates computed!")
        return {}, [], {}
    
    # Sort by F1 score (descending)
    idx_results_sorted = sorted(idx_results, key=lambda x: x['f1'], reverse=True)
    
    # Compute overall statistics
    precisions = np.array([r['precision'] for r in idx_results])
    recalls = np.array([r['recall'] for r in idx_results])
    f1s = np.array([r['f1'] for r in idx_results])
    
    stats = {
        'precision': {
            'mean': float(np.mean(precisions)),
            'median': float(np.median(precisions)),
            'std': float(np.std(precisions)),
            'min': float(np.min(precisions)),
            'max': float(np.max(precisions)),
        },
        'recall': {
            'mean': float(np.mean(recalls)),
            'median': float(np.median(recalls)),
            'std': float(np.std(recalls)),
            'min': float(np.min(recalls)),
            'max': float(np.max(recalls)),
        },
        'f1': {
            'mean': float(np.mean(f1s)),
            'median': float(np.median(f1s)),
            'std': float(np.std(f1s)),
            'min': float(np.min(f1s)),
            'max': float(np.max(f1s)),
            'percentile_5': float(np.percentile(f1s, 5)),
            'percentile_95': float(np.percentile(f1s, 95)),
        },
        'num_indices': len(idx_results),
        'total_pairs': total_pairs
    }
    
    # Aggregate per-type statistics
    global_type_stats = aggregate_per_type_stats(idx_results)
    
    return stats, idx_results_sorted, global_type_stats


def main():
    parser = argparse.ArgumentParser(description='Compute Symmetric Type Matching Metrics (Precision/Recall/F1) between pred and gt JSON files.')
    parser.add_argument('--test_dir', type=str,
        default='/deemos-research-area-d/meshgen/cad/checkpoints/GPT_INIT_142M/train_0110_michel_4096_full/test_00')
    parser.add_argument('--checkpoint', type=str, default='253500')
    parser.add_argument('--resolution', type=int, default=16,
        help='Sampling resolution for each surface')
    parser.add_argument('--output', type=str, default='type_matching_results.json')
    parser.add_argument('--range_min', type=int, default=None,
        help='Minimum idx to process (inclusive)')
    parser.add_argument('--range_max', type=int, default=None,
        help='Maximum idx to process (inclusive)')
    parser.add_argument('--distance_threshold', type=float, default=None,
        help='Absolute distance threshold for point matching. If None, use relative_threshold.')
    parser.add_argument('--relative_threshold', type=float, default=0.01,
        help='Relative distance threshold as fraction of bbox diagonal (default: 0.01)')
    
    args = parser.parse_args()
    
    range_info = ""
    if args.range_min is not None or args.range_max is not None:
        range_info = f", Range: [{args.range_min if args.range_min is not None else 'start'} - {args.range_max if args.range_max is not None else 'end'}]"
    
    threshold_info = f"Absolute τ={args.distance_threshold}" if args.distance_threshold is not None else f"Relative τ={args.relative_threshold}×bbox_diag"
    print(f"Checkpoint: {args.checkpoint}, Resolution: {args.resolution}x{args.resolution}, {threshold_info}{range_info}\n")
    
    stats, idx_results, global_type_stats = compute_type_matching_for_checkpoint(
        args.test_dir, args.checkpoint, args.resolution, 
        args.range_min, args.range_max,
        args.distance_threshold, args.relative_threshold
    )
    
    if stats:
        print(f"\n{'='*80}")
        print(f"Checkpoint {args.checkpoint} - Unique indices: {stats['num_indices']} (from {stats['total_pairs']} pairs)")
        print(f"{'-'*80}")
        print(f"{'Metric':<15} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"{'-'*80}")
        print(f"{'Precision':<15} {stats['precision']['mean']:>10.4f} {stats['precision']['median']:>10.4f} "
              f"{stats['precision']['std']:>10.4f} {stats['precision']['min']:>10.4f} {stats['precision']['max']:>10.4f}")
        print(f"{'Recall':<15} {stats['recall']['mean']:>10.4f} {stats['recall']['median']:>10.4f} "
              f"{stats['recall']['std']:>10.4f} {stats['recall']['min']:>10.4f} {stats['recall']['max']:>10.4f}")
        print(f"{'F1':<15} {stats['f1']['mean']:>10.4f} {stats['f1']['median']:>10.4f} "
              f"{stats['f1']['std']:>10.4f} {stats['f1']['min']:>10.4f} {stats['f1']['max']:>10.4f}")
        print(f"{'-'*80}")
        print(f"F1 Percentiles: 5th={stats['f1']['percentile_5']:.4f}, 95th={stats['f1']['percentile_95']:.4f}")
        print(f"{'='*80}")
        
        print(f"\nPer-Type Global Statistics:")
        print(f"{'-'*100}")
        print(f"{'Type':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Pred Pts':>12} {'GT Pts':>12}")
        print(f"{'-'*100}")
        for surf_type, type_stats in sorted(global_type_stats.items()):
            prec = type_stats['global_precision']
            rec = type_stats['global_recall']
            f1 = type_stats['global_f1']
            pred_pts = type_stats['total_pred_points']
            gt_pts = type_stats['total_gt_points']
            print(f"{surf_type:<20} {prec:>12.4f} {rec:>12.4f} {f1:>12.4f} {pred_pts:>12d} {gt_pts:>12d}")
        print(f"{'-'*100}")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'checkpoint': args.checkpoint,
                'resolution': args.resolution,
                'distance_threshold': args.distance_threshold,
                'relative_threshold': args.relative_threshold,
                'range_min': args.range_min,
                'range_max': args.range_max,
                'statistics': stats,
                'per_type_global_stats': global_type_stats,
                'results_sorted_by_f1': idx_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to: {args.output}")
            print(f"Saved {len(idx_results)} index results sorted by F1 score (descending)")


if __name__ == '__main__':
    main()

