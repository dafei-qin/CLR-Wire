"""
Script to sample points from surfaces in JSON files.

This script:
1. Loads surface data from JSON files
2. Samples each surface proportionally to its surface area
3. Generates uniform UV grid samples for each surface
4. Strictly limits total points to max_points
5. Saves all sampled points to NPY files

The output NPY files maintain the same relative directory structure as input JSON files.
"""

import numpy as np
import sys
import json
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v1 import dataset_compound
from src.tools.sample_simple_surface import sample_surface_uniform


def estimate_surface_area(params, surface_type_idx, num_u=16, num_v=16):
    """
    Estimate surface area by sampling and computing mesh area.
    
    Args:
        params: Surface parameters
        surface_type_idx: Surface type index
        num_u: Number of samples in u direction for area estimation
        num_v: Number of samples in v direction for area estimation
        
    Returns:
        Estimated surface area (float)
    """
    try:
        # Sample surface on a grid
        points = sample_surface_uniform(
            params,
            surface_type_idx,
            num_u=num_u,
            num_v=num_v,
            flatten=False
        )  # Shape: (num_v, num_u, 3)
        
        # Compute area by summing areas of grid quads
        total_area = 0.0
        
        for i in range(num_v - 1):
            for j in range(num_u - 1):
                # Get four corners of quad
                p00 = points[i, j]
                p01 = points[i, j+1]
                p10 = points[i+1, j]
                p11 = points[i+1, j+1]
                
                # Split quad into two triangles and compute their areas
                # Triangle 1: p00, p01, p10
                v1 = p01 - p00
                v2 = p10 - p00
                area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
                
                # Triangle 2: p01, p11, p10
                v1 = p11 - p01
                v2 = p10 - p01
                area2 = 0.5 * np.linalg.norm(np.cross(v1, v2))
                
                total_area += area1 + area2
        
        return total_area
    
    except Exception as e:
        print(f"Warning: Failed to estimate area for surface type {surface_type_idx}: {e}")
        return 0.0


def compute_grid_size(target_points):
    """
    Compute num_u and num_v to get approximately target_points.
    
    Args:
        target_points: Target number of points
        
    Returns:
        (num_u, num_v) tuple
    """
    # Solve num_u * num_v ≈ target_points
    # Use square grid as default, then adjust
    sqrt_points = int(np.sqrt(target_points))
    
    # Try to get as close as possible
    num_u = max(2, sqrt_points)
    num_v = max(2, int(np.ceil(target_points / num_u)))
    
    # Adjust to get closer to target
    actual_points = num_u * num_v
    if actual_points > target_points:
        # Try reducing num_v
        num_v = max(2, int(target_points / num_u))
    
    return num_u, num_v


def allocate_points_uniformly(n_surfaces, max_points, min_points_per_surface=4):
    """
    Allocate sampling points uniformly across all surfaces.
    Each surface gets approximately the same number of points.
    
    Args:
        n_surfaces: Number of surfaces
        max_points: Maximum total points to allocate
        min_points_per_surface: Minimum points per surface (must be >= 4 for 2x2 grid)
        
    Returns:
        List of point allocations for each surface
    """
    # Base allocation: divide equally
    points_per_surface = max_points // n_surfaces
    remainder = max_points % n_surfaces
    
    # Ensure minimum
    points_per_surface = max(points_per_surface, min_points_per_surface)
    
    # Allocate base points to all surfaces
    allocations = [points_per_surface] * n_surfaces
    
    # Distribute remainder to first few surfaces
    for i in range(remainder):
        allocations[i] += 1
    
    # Check if we exceed max_points due to minimum constraint
    current_total = sum(allocations)
    if current_total > max_points:
        # Need to reduce, but respect minimum
        diff = current_total - max_points
        # Try to remove from surfaces with most points
        indices_by_allocation = sorted(range(n_surfaces), 
                                      key=lambda i: allocations[i], 
                                      reverse=True)
        removed = 0
        for idx in indices_by_allocation:
            if removed >= diff:
                break
            can_remove = allocations[idx] - min_points_per_surface
            to_remove = min(can_remove, diff - removed)
            if to_remove > 0:
                allocations[idx] -= to_remove
                removed += to_remove
    
    final_total = sum(allocations)
    if final_total != max_points:
        print(f"Warning: Could not allocate exactly {max_points} points with uniform distribution, got {final_total}")
    
    return allocations


def allocate_points_by_area(areas, max_points, min_points_per_surface=4):
    """
    Allocate sampling points to surfaces proportionally to their areas.
    
    Args:
        areas: List of surface areas
        max_points: Maximum total points to allocate
        min_points_per_surface: Minimum points per surface (must be >= 4 for 2x2 grid)
        
    Returns:
        List of point allocations for each surface
    """
    n_surfaces = len(areas)
    total_area = sum(areas)
    
    if total_area == 0:
        # Equal distribution if all areas are zero
        points_per_surface = max_points // n_surfaces
        remainder = max_points % n_surfaces
        allocations = [points_per_surface] * n_surfaces
        for i in range(remainder):
            allocations[i] += 1
        return allocations
    
    # First pass: allocate proportionally
    allocations = []
    for area in areas:
        if area > 0:
            points = int(np.round(max_points * area / total_area))
            allocations.append(max(min_points_per_surface, points))
        else:
            allocations.append(min_points_per_surface)
    
    # Adjust to meet exact max_points constraint
    current_total = sum(allocations)
    
    if current_total < max_points:
        # Add remaining points to largest surfaces
        diff = max_points - current_total
        # Sort by area (descending)
        indices_by_area = sorted(range(n_surfaces), key=lambda i: areas[i], reverse=True)
        for i in range(diff):
            allocations[indices_by_area[i % n_surfaces]] += 1
    
    elif current_total > max_points:
        # Remove excess points from largest surfaces (but keep minimum)
        diff = current_total - max_points
        # Sort by allocation (descending)
        indices_by_allocation = sorted(range(n_surfaces), 
                                      key=lambda i: allocations[i], 
                                      reverse=True)
        removed = 0
        for idx in indices_by_allocation:
            if removed >= diff:
                break
            # Can remove if above minimum
            can_remove = allocations[idx] - min_points_per_surface
            to_remove = min(can_remove, diff - removed)
            if to_remove > 0:
                allocations[idx] -= to_remove
                removed += to_remove
    
    # Final adjustment: if still over, force reduce from largest
    current_total = sum(allocations)
    if current_total > max_points:
        diff = current_total - max_points
        indices_by_allocation = sorted(range(n_surfaces), 
                                      key=lambda i: allocations[i], 
                                      reverse=True)
        for i in range(diff):
            idx = indices_by_allocation[i % n_surfaces]
            if allocations[idx] > min_points_per_surface:
                allocations[idx] -= 1
    
    # Ensure exact total
    final_total = sum(allocations)
    if final_total != max_points:
        print(f"Warning: Could not allocate exactly {max_points} points, got {final_total}")
    
    return allocations


def sample_surfaces_from_json(json_path, dataset, max_points=2048, area_estimation_grid=16, 
                             skip_bspline=True, sampling_mode='area'):
    """
    Sample points from all surfaces in a JSON file.
    
    Args:
        json_path: Path to JSON file
        dataset: Dataset instance for parsing surfaces
        max_points: Maximum total number of points to sample
        area_estimation_grid: Grid size for area estimation
        skip_bspline: Whether to skip B-spline surfaces
        sampling_mode: Sampling strategy
            - 'area': Allocate points proportionally to surface area (default)
            - 'uniform': Allocate equal points to each surface
        
    Returns:
        sampled_points: (N, 3) array of sampled points, where N ≤ max_points
        surface_labels: (N,) array of surface indices for each point
    """
    # Load JSON
    with open(json_path, 'r') as f:
        surfaces_data = json.load(f)
    
    if not surfaces_data:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    if skip_bspline:
        surfaces_data = [surface_dict for surface_dict in surfaces_data if surface_dict['type'] != 'bspline_surface']
    # Parse surfaces and estimate areas
    valid_surfaces = []
    areas = []
    
    for i, surface_dict in enumerate(surfaces_data):
#         try:
            # Parse surface
            params, surface_type_idx = dataset._parse_surface(surface_dict)
            
            if surface_type_idx == -1:
                continue
            
            # Estimate area
            area = estimate_surface_area(
                params, 
                surface_type_idx, 
                num_u=area_estimation_grid,
                num_v=area_estimation_grid
            )
            
            valid_surfaces.append((params, surface_type_idx))
            areas.append(area)
            
        # except Exception as e:
        #     print(f"Warning: Failed to parse surface {i} in {json_path}: {e}")
        #     continue
    
    if not valid_surfaces:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    # Allocate points based on sampling mode
    if sampling_mode == 'uniform':
        point_allocations = allocate_points_uniformly(len(valid_surfaces), max_points, min_points_per_surface=4)
    elif sampling_mode == 'area':
        point_allocations = allocate_points_by_area(areas, max_points, min_points_per_surface=4)
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}. Use 'area' or 'uniform'.")
    
    # Sample each surface
    all_points = []
    all_labels = []
    
    for surf_idx, ((params, surface_type_idx), num_points) in enumerate(zip(valid_surfaces, point_allocations)):
        try:
            # Determine grid size
            num_u, num_v = compute_grid_size(num_points)
            
            # Sample surface
            points = sample_surface_uniform(
                params,
                surface_type_idx,
                num_u=num_u,
                num_v=num_v,
                flatten=True
            )  # Shape: (num_u * num_v, 3)
            
            actual_points = len(points)
            
            # Adjust if we got more or fewer points than allocated
            if actual_points > num_points:
                # Randomly subsample
                indices = np.random.choice(actual_points, num_points, replace=False)
                points = points[indices]
            elif actual_points < num_points:
                # Oversample with replacement
                indices = np.random.choice(actual_points, num_points, replace=True)
                points = points[indices]
            
            # Add points and labels
            all_points.append(points)
            all_labels.append(np.full(len(points), surf_idx, dtype=np.int32))
            
        except Exception as e:
            print(f"Warning: Failed to sample surface {surf_idx} (type {surface_type_idx}): {e}")
            continue
    
    if not all_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    # Concatenate all points
    sampled_points = np.concatenate(all_points, axis=0).astype(np.float32)
    surface_labels = np.concatenate(all_labels, axis=0)
    
    return sampled_points, surface_labels


def get_output_path(input_json_path, input_root, output_root):
    """
    Get the output NPY file path maintaining the same relative directory structure.
    
    Args:
        input_json_path: Full path to input JSON file
        input_root: Root directory of input JSON files
        output_root: Root directory for output NPY files
        
    Returns:
        Full path to output NPY file
    """
    input_path = Path(input_json_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Get relative path from input root
    relative_path = input_path.relative_to(input_root)
    
    # Change extension from .json to .npy
    relative_path = relative_path.with_suffix('.npy')
    
    # Construct output path
    output_path = output_root / relative_path
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Sample points from surfaces in JSON files'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for NPY files (will maintain same subdirectory structure)'
    )
    parser.add_argument(
        '--max_points',
        type=int,
        default=2048,
        help='Maximum number of points to sample per JSON file (default: 2048)'
    )
    parser.add_argument(
        '--area_estimation_grid',
        type=int,
        default=16,
        help='Grid size for area estimation (default: 16)'
    )
    parser.add_argument(
        '--save_labels',
        action='store_true',
        help='Also save surface labels for each point'
    )
    parser.add_argument(
        '--sampling_mode',
        type=str,
        default='area',
        choices=['area', 'uniform'],
        help='Sampling strategy: "area" (proportional to surface area) or "uniform" (equal per surface) (default: area)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.input_dir}")
    dataset = dataset_compound(args.input_dir, canonical=False)
    print(f"Found {len(dataset)} JSON files")
    
    # Create output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files
    print(f"\nProcessing JSON files...")
    print(f"Sampling mode: {args.sampling_mode}")
    print(f"  - 'area': Points allocated proportionally to surface area")
    print(f"  - 'uniform': Equal points for each surface")
    print(f"Max points per file: {args.max_points}")
    print(f"Area estimation grid: {args.area_estimation_grid}x{args.area_estimation_grid}")
    
    failed_samples = []
    total_points_saved = 0
    
    for idx in tqdm(range(len(dataset.json_names)), desc="Sampling surfaces"):
        json_path = dataset.json_names[idx]
        
        # try:
            # Sample surfaces
            sampled_points, surface_labels = sample_surfaces_from_json(
                json_path,
                dataset,
                max_points=args.max_points,
                area_estimation_grid=args.area_estimation_grid,
                sampling_mode=args.sampling_mode
            )
        
        if len(sampled_points) == 0:
            print(f"\nWarning: No valid points sampled from {json_path}")
            continue
        
        # Get output path
        output_path = get_output_path(json_path, args.input_dir, args.output_dir)
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save points
        np.save(output_path, sampled_points)
        total_points_saved += len(sampled_points)
        
        # Optionally save labels
        if args.save_labels:
            label_path = output_path.with_suffix('.labels.npy')
            np.save(label_path, surface_labels)
            
        # except Exception as e:
        #     print(f"\nError processing {json_path}: {e}")
        #     failed_samples.append((idx, json_path, str(e)))
        #     continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Successfully processed: {len(dataset.json_names) - len(failed_samples)} files")
    print(f"Failed: {len(failed_samples)} files")
    print(f"Total points saved: {total_points_saved:,}")
    print(f"Average points per file: {total_points_saved / max(1, len(dataset.json_names) - len(failed_samples)):.1f}")
    
    if failed_samples:
        print("\nFailed samples:")
        for idx, path, error in failed_samples[:10]:  # Show first 10
            print(f"  [{idx}] {path}")
            print(f"       Error: {error}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()


