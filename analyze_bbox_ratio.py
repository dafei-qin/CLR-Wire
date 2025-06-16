#!/usr/bin/env python3
"""
Analyze JSON files to find the optimal bounding box ratio for each surface.

This script processes a directory of JSON files containing CAD data. For each surface,
it performs a line search to find the smallest bounding box ratio (from 1.0 to 1.5)
that encloses all of its adjacent surfaces.

The results, including the optimal ratio and related statistics, are saved to a CSV file.
"""

import argparse
import csv
import json
import os
import multiprocessing
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def default_ratio_analysis():
    return {'surfaces_in_bbox': [], 'adj_not_in_bbox': []}


def find_json_files(input_dir):
    """Recursively find all JSON files in the input directory."""
    input_path = Path(input_dir)
    json_files = sorted(list(input_path.rglob("*.json")))
    return [str(f) for f in json_files]


def get_surfaces_and_adjacency(data):
    """Extract surfaces and their adjacency relationships from the raw JSON data."""
    surface_adjacency = defaultdict(set)
    surfaces = {}
    for item in data:
        if 'idx' in item and len(item['idx']) == 2:
            idx_a, idx_b = item['idx']
            if idx_a == idx_b:
                if 'points' in item and item['points']:
                    surfaces[idx_a] = item
            else:
                surface_adjacency[idx_a].add(idx_b)
                surface_adjacency[idx_b].add(idx_a)
    
    for key in list(surface_adjacency.keys()):
        surface_adjacency[key] = sorted(list(surface_adjacency[key]))
        
    return surfaces, dict(surface_adjacency)


def upsample_surface(vertices, factor=4):
    """
    Upsamples the surface vertices assuming they form a square grid.
    """
    num_points = vertices.shape[0]
    grid_size = int(np.sqrt(num_points))

    if grid_size * grid_size != num_points:
        # Not a square grid, cannot upsample this way.
        return None

    # Reshape to a grid for each coordinate
    grid_x = vertices[:, 0].reshape((grid_size, grid_size))
    grid_y = vertices[:, 1].reshape((grid_size, grid_size))
    grid_z = vertices[:, 2].reshape((grid_size, grid_size))

    # Upsample each coordinate grid using linear interpolation
    zoomed_x = zoom(grid_x, factor, order=1)
    zoomed_y = zoom(grid_y, factor, order=1)
    zoomed_z = zoom(grid_z, factor, order=1)

    # Combine back into a list of points
    upsampled_vertices = np.vstack([zoomed_x.ravel(), zoomed_y.ravel(), zoomed_z.ravel()]).T
    
    return upsampled_vertices


def is_surface_in_bbox(surface_verts, bbox_min, bbox_max):
    """
    Check if any vertex of a surface is within the given bounding box.
    If not, upsample the vertices and check again.
    """
    # Initial check with original vertices
    if np.any(np.all((surface_verts >= bbox_min) & (surface_verts <= bbox_max), axis=1)):
        return True

    # If the initial check fails, try upsampling the surface.
    upsampled_verts = upsample_surface(surface_verts, factor=4)
    
    if upsampled_verts is not None:
        # Check again with the upsampled vertices.
        return np.any(np.all((upsampled_verts >= bbox_min) & (upsampled_verts <= bbox_max), axis=1))

    # If upsampling is not possible, return the result from the original check (which was False).
    return False


def analyze_file(file_path):
    """
    Analyzes a single JSON file to gather statistics for various bounding box ratios.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    surfaces, adjacency_map = get_surfaces_and_adjacency(data)
    
    file_stats = {
        'num_adjacent_dist': [],
        'ratio_analysis': defaultdict(default_ratio_analysis)
    }
    ratios = np.arange(1.0, 1.06, 0.02)

    for surface_idx, surface_data in surfaces.items():
        if not surface_data or not surface_data.get('points'):
            continue

        adjacent_indices = adjacency_map.get(surface_idx, [])
        file_stats['num_adjacent_dist'].append(len(adjacent_indices))

        selected_verts = np.array(surface_data['points']).reshape(-1, 3)
        min_coords = np.min(selected_verts, axis=0)
        max_coords = np.max(selected_verts, axis=0)
        surface_size = max_coords - min_coords
        surface_size[surface_size < 1e-6] = 1e-6
        center = (min_coords + max_coords) / 2.0

        for ratio in ratios:
            ratio = round(ratio, 2)
            bbox_dims = ratio * surface_size
            
            if surface_data.get("type") == 'plane':
                min_idx = np.argmin(bbox_dims)
                if bbox_dims[min_idx] < 0.01:
                    bbox_dims[min_idx] = 0.01

            half_dims = bbox_dims / 2.0
            bbox_min = center - half_dims
            bbox_max = center + half_dims

            surfaces_in_bbox = set()
            for other_idx, other_surface in surfaces.items():
                if other_idx == surface_idx:
                    continue
                if other_surface.get('points'):
                    other_verts = np.array(other_surface['points']).reshape(-1, 3)
                    if is_surface_in_bbox(other_verts, bbox_min, bbox_max):
                        surfaces_in_bbox.add(other_idx)

            adj_set = set(adjacent_indices)
            num_adj_not_in_bbox = len(adj_set - surfaces_in_bbox)
            
            stats = file_stats['ratio_analysis'][ratio]
            stats['surfaces_in_bbox'].append(len(surfaces_in_bbox))
            stats['adj_not_in_bbox'].append(num_adj_not_in_bbox)
            
    return file_stats


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSON files to find optimal bounding box ratios for a dataset."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing JSON files."
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    
    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    json_files = find_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in {input_dir}.")
        return

    print(f"Found {len(json_files)} JSON files. Analyzing...")

    all_num_adjacent = []
    all_ratio_stats = defaultdict(lambda: {'surfaces_in_bbox': [], 'adj_not_in_bbox': []})

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for analysis.")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(analyze_file, json_files)
        
        for file_results in tqdm(results_iterator, total=len(json_files), desc="Processing files"):
            if file_results:
                all_num_adjacent.extend(file_results['num_adjacent_dist'])
                for ratio, stats in file_results['ratio_analysis'].items():
                    all_ratio_stats[ratio]['surfaces_in_bbox'].extend(stats['surfaces_in_bbox'])
                    all_ratio_stats[ratio]['adj_not_in_bbox'].extend(stats['adj_not_in_bbox'])

    if not all_num_adjacent:
        print("No processable surfaces found in any files.")
        return
    
    # --- Print statistics ---

    print("\n--- Distribution of Number of Adjacent Surfaces ---")
    adj_array = np.array(all_num_adjacent)
    print(f"Total surfaces analyzed: {len(adj_array)}")
    if len(adj_array) > 0:
        print(f"  Min: {np.min(adj_array)}")
        print(f"  Max: {np.max(adj_array)}")
        print(f"  Mean: {np.mean(adj_array):.2f}")
        print(f"  Median: {np.median(adj_array)}")
        print(f"  75th percentile: {np.percentile(adj_array, 75)}")
        print(f"  95th percentile: {np.percentile(adj_array, 95)}")

    print("\n--- Bounding Box Ratio Analysis ---")
    sorted_ratios = sorted(all_ratio_stats.keys())

    for ratio in sorted_ratios:
        stats = all_ratio_stats[ratio]
        surfaces_in_bbox_dist = np.array(stats['surfaces_in_bbox'])
        adj_not_in_bbox_dist = np.array(stats['adj_not_in_bbox'])
        
        total_cases = len(adj_not_in_bbox_dist)
        if total_cases > 0:
            all_captured_count = np.sum(adj_not_in_bbox_dist == 0)
            all_captured_percent = (all_captured_count / total_cases) * 100
        else:
            all_captured_percent = 0

        print(f"\n--- Ratio: {ratio:.2f} ---")
        print(f"  Percentage of surfaces capturing ALL adjacent surfaces: {all_captured_percent:.2f}%")

        if len(adj_not_in_bbox_dist) > 0:
            print("  Distribution of 'adjacent surfaces NOT in bbox':")
            print(f"    Mean: {np.mean(adj_not_in_bbox_dist):.2f}, "
                  f"Max: {np.max(adj_not_in_bbox_dist)}, "
                  f"95th percentile: {np.percentile(adj_not_in_bbox_dist, 95):.2f}")
        
        if len(surfaces_in_bbox_dist) > 0:
            print("  Distribution of 'total surfaces in bbox':")
            print(f"    Mean: {np.mean(surfaces_in_bbox_dist):.2f}, "
                  f"Max: {np.max(surfaces_in_bbox_dist)}, "
                  f"95th percentile: {np.percentile(surfaces_in_bbox_dist, 95):.2f}")


if __name__ == "__main__":
    main() 