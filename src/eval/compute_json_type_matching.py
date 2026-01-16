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


def compute_type_matching_fast(pred_points, pred_types, gt_points, gt_types):
    """
    Compute type matching rate between pred and gt point clouds.
    
    For each pred point, find its nearest gt point and check if types match.
    
    Args:
        pred_points: (N, 3) numpy array
        pred_types: (N,) numpy array of type strings
        gt_points: (M, 3) numpy array
        gt_types: (M,) numpy array of type strings
    
    Returns:
        matching_rate: Overall matching rate (0-1)
        per_type_stats: Dict with per-type statistics
        num_matches: Number of matched points
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
        
        # Compute distance matrix
        dist_matrix = torch.cdist(pred_points_tensor.unsqueeze(0), gt_points_tensor.unsqueeze(0), p=2).squeeze(0)  # (N, M)
        
        # Find nearest gt point for each pred point
        nearest_indices = dist_matrix.argmin(dim=1).cpu().numpy()  # (N,)
        
        # Get matched gt types
        matched_gt_types = gt_types[nearest_indices]
        
        # Check type matches
        type_matches = (pred_types == matched_gt_types)
        num_matches = type_matches.sum()
        matching_rate = num_matches / len(pred_types)
        
        # Per-type statistics
        per_type_stats = {}
        unique_types = np.unique(pred_types)
        
        for surf_type in unique_types:
            mask = (pred_types == surf_type)
            type_pred_count = mask.sum()
            type_matched_count = type_matches[mask].sum()
            type_rate = type_matched_count / type_pred_count if type_pred_count > 0 else 0.0
            
            per_type_stats[surf_type] = {
                'pred_count': int(type_pred_count),
                'matched_count': int(type_matched_count),
                'rate': float(type_rate)
            }
        
        return float(matching_rate), per_type_stats, int(num_matches)
    
    except Exception as e:
        print(f"Error in compute_type_matching_fast: {e}")
        return 0.0, {}, 0


def process_json_pair(pred_surfaces: List[Dict], gt_surfaces: List[Dict], resolution: int = 16) -> Tuple[float, Dict, int, int, int]:
    """
    Process a pair of pred/gt JSON files.
    
    Returns:
        matching_rate: Overall matching rate
        per_type_stats: Per-type statistics
        num_pred_points: Number of pred points
        num_gt_points: Number of gt points
        num_matches: Number of matched points
    """
    try:
        pred_points, pred_types = surfaces_to_typed_points(pred_surfaces, resolution)
        gt_points, gt_types = surfaces_to_typed_points(gt_surfaces, resolution)
        
        matching_rate, per_type_stats, num_matches = compute_type_matching_fast(
            pred_points, pred_types, gt_points, gt_types
        )
        
        return matching_rate, per_type_stats, len(pred_points), len(gt_points), num_matches
    
    except Exception as e:
        print(f"Error processing JSON pair: {e}")
        return 0.0, {}, 0, 0, 0


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


def aggregate_per_type_stats(all_per_type_stats: List[Dict]) -> Dict:
    """
    Aggregate per-type statistics across all idx results.
    
    Args:
        all_per_type_stats: List of per_type_stats dicts from each idx
    
    Returns:
        Global per-type statistics
    """
    type_aggregates = {}
    
    for per_type_stats in all_per_type_stats:
        for surf_type, stats in per_type_stats.items():
            if surf_type not in type_aggregates:
                type_aggregates[surf_type] = {
                    'total_pred': 0,
                    'total_matched': 0,
                    'rates': []
                }
            
            type_aggregates[surf_type]['total_pred'] += stats['pred_count']
            type_aggregates[surf_type]['total_matched'] += stats['matched_count']
            type_aggregates[surf_type]['rates'].append(stats['rate'])
    
    # Compute global statistics for each type
    global_type_stats = {}
    for surf_type, agg in type_aggregates.items():
        global_type_stats[surf_type] = {
            'total_pred_points': agg['total_pred'],
            'total_matched_points': agg['total_matched'],
            'global_rate': agg['total_matched'] / agg['total_pred'] if agg['total_pred'] > 0 else 0.0,
            'mean_rate_per_idx': float(np.mean(agg['rates'])),
            'std_rate_per_idx': float(np.std(agg['rates'])),
            'num_indices': len(agg['rates'])
        }
    
    return global_type_stats


def compute_type_matching_for_checkpoint(
    test_dir: str, 
    checkpoint: str, 
    resolution: int = 16,
    range_min: int = None,
    range_max: int = None
) -> Tuple[Dict, List[Dict], Dict]:
    """
    Compute type matching rates for all pred/gt pairs in checkpoint.
    
    Args:
        test_dir: Directory containing test files
        checkpoint: Checkpoint iteration number
        resolution: Sampling resolution
        range_min: Minimum idx to process (inclusive), None means no minimum
        range_max: Maximum idx to process (inclusive), None means no maximum
    
    Returns:
        stats: Overall statistics
        idx_results: Results for each idx (sorted by matching rate descending)
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
        idx_matching_rates = []
        idx_file_info = []
        
        print(f"\nIdx {idx} ({len(pairs)} batch variants):")
        for pred_path, gt_path in pairs:
            pred_surfaces = load_json_surfaces(pred_path)
            gt_surfaces = load_json_surfaces(gt_path)
            
            matching_rate, per_type_stats, num_pred, num_gt, num_matches = process_json_pair(
                pred_surfaces, gt_surfaces, resolution
            )
            
            gt_name = os.path.basename(gt_path)
            pred_name = os.path.basename(pred_path)
            
            print(f"  {gt_name} (GT:{num_gt}pts) {pred_name} (Pred:{num_pred}pts) "
                  f"Match={matching_rate:.4f} ({num_matches}/{num_pred})")
            
            idx_matching_rates.append(matching_rate)
            idx_file_info.append({
                'gt_path': gt_path,
                'pred_path': pred_path,
                'matching_rate': matching_rate,
                'per_type_stats': per_type_stats,
                'num_pred_points': num_pred,
                'num_gt_points': num_gt,
                'num_matches': num_matches
            })
        
        if idx_matching_rates:
            # Select the variant with HIGHEST matching rate
            max_idx = np.argmax(idx_matching_rates)
            max_rate = idx_matching_rates[max_idx]
            best_files = idx_file_info[max_idx]
            
            idx_results.append({
                'idx': idx,
                'matching_rate': max_rate,
                'gt_path': best_files['gt_path'],
                'pred_path': best_files['pred_path'],
                'per_type_stats': best_files['per_type_stats'],
                'num_pred_points': best_files['num_pred_points'],
                'num_gt_points': best_files['num_gt_points'],
                'num_matches': best_files['num_matches']
            })
            print(f"  -> Max Matching Rate for idx {idx}: {max_rate:.4f}")
    
    if not idx_results:
        print("No valid matching rates computed!")
        return {}, [], {}
    
    # Sort by matching rate (descending)
    idx_results_sorted = sorted(idx_results, key=lambda x: x['matching_rate'], reverse=True)
    
    # Compute overall statistics
    matching_rates = np.array([r['matching_rate'] for r in idx_results])
    
    stats = {
        'mean': float(np.mean(matching_rates)),
        'median': float(np.median(matching_rates)),
        'std': float(np.std(matching_rates)),
        'min': float(np.min(matching_rates)),
        'max': float(np.max(matching_rates)),
        'percentile_5': float(np.percentile(matching_rates, 5)),
        'percentile_95': float(np.percentile(matching_rates, 95)),
        'num_indices': len(idx_results),
        'total_pairs': total_pairs
    }
    
    # Aggregate per-type statistics
    all_per_type_stats = [r['per_type_stats'] for r in idx_results]
    global_type_stats = aggregate_per_type_stats(all_per_type_stats)
    
    return stats, idx_results_sorted, global_type_stats


def main():
    parser = argparse.ArgumentParser(description='Compute Type Matching Rate between pred and gt JSON files.')
    parser.add_argument('--test_dir', type=str,
        default='/deemos-research-area-d/meshgen/cad/checkpoints/GPT_INIT_142M/train_0110_michel_4096_full/test_00')
    parser.add_argument('--checkpoint', type=str, default='253500')
    parser.add_argument('--resolution', type=int, default=16)
    parser.add_argument('--output', type=str, default='type_matching_results.json')
    parser.add_argument('--range_min', type=int, default=None,
        help='Minimum idx to process (inclusive)')
    parser.add_argument('--range_max', type=int, default=None,
        help='Maximum idx to process (inclusive)')
    
    args = parser.parse_args()
    
    range_info = ""
    if args.range_min is not None or args.range_max is not None:
        range_info = f", Range: [{args.range_min if args.range_min is not None else 'start'} - {args.range_max if args.range_max is not None else 'end'}]"
    print(f"Checkpoint: {args.checkpoint}, Resolution: {args.resolution}x{args.resolution}{range_info}\n")
    
    stats, idx_results, global_type_stats = compute_type_matching_for_checkpoint(
        args.test_dir, args.checkpoint, args.resolution, args.range_min, args.range_max
    )
    
    if stats:
        print(f"\n{'='*60}")
        print(f"Checkpoint {args.checkpoint} - Unique indices: {stats['num_indices']} (from {stats['total_pairs']} pairs)")
        print(f"Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f} | Std: {stats['std']:.4f}")
        print(f"Min: {stats['min']:.4f} | Max: {stats['max']:.4f}")
        print(f"5th: {stats['percentile_5']:.4f} | 95th: {stats['percentile_95']:.4f}")
        print(f"{'='*60}")
        
        print(f"\nPer-Type Global Statistics:")
        print(f"{'-'*60}")
        for surf_type, type_stats in sorted(global_type_stats.items()):
            print(f"{surf_type}:")
            print(f"  Total Pred Points: {type_stats['total_pred_points']}")
            print(f"  Total Matched Points: {type_stats['total_matched_points']}")
            print(f"  Global Rate: {type_stats['global_rate']:.4f}")
            print(f"  Mean Rate per Idx: {type_stats['mean_rate_per_idx']:.4f} ± {type_stats['std_rate_per_idx']:.4f}")
            print(f"  Num Indices: {type_stats['num_indices']}")
        print(f"{'-'*60}")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            output_data = {
                'checkpoint': args.checkpoint,
                'resolution': args.resolution,
                'range_min': args.range_min,
                'range_max': args.range_max,
                'statistics': stats,
                'per_type_global_stats': global_type_stats,
                'results_sorted_by_matching_rate': idx_results
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nSaved to: {args.output}")
            print(f"Saved {len(idx_results)} index results sorted by matching rate (descending)")


if __name__ == '__main__':
    main()

