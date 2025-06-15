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
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm


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


def is_surface_in_bbox(surface_verts, bbox_min, bbox_max):
    """Check if any vertex of a surface is within the given bounding box."""
    return np.any(np.all((surface_verts >= bbox_min) & (surface_verts <= bbox_max), axis=1))


def find_optimal_ratio(selected_surface, all_surfaces, adjacent_indices):
    """
    Performs a line search for the optimal bbox ratio for a single surface.
    The optimal ratio is the smallest one that contains all adjacent surfaces.
    """
    if not selected_surface or not selected_surface.get('points'):
        return 1.5, False

    selected_verts = np.array(selected_surface['points']).reshape(-1, 3)
    min_coords = np.min(selected_verts, axis=0)
    max_coords = np.max(selected_verts, axis=0)
    surface_size = max_coords - min_coords
    surface_size[surface_size < 1e-6] = 1e-6

    # Pre-fetch vertex data for adjacent surfaces to speed up the loop
    adjacent_surfaces_verts = []
    for adj_idx in adjacent_indices:
        adj_surface = all_surfaces.get(adj_idx)
        if adj_surface and adj_surface.get('points'):
            adjacent_surfaces_verts.append(np.array(adj_surface['points']).reshape(-1, 3))
    
    if not adjacent_surfaces_verts:
        return 1.0, True  # No adjacent surfaces to capture

    for ratio in np.arange(1.0, 1.52, 0.02):
        ratio = round(ratio, 2)
        bbox_dims = ratio * surface_size
        
        if selected_surface.get("type") == 'plane':
            min_idx = np.argmin(bbox_dims)
            if bbox_dims[min_idx] < 0.01:
                bbox_dims[min_idx] = 0.01

        center = (min_coords + max_coords) / 2.0
        half_dims = bbox_dims / 2.0
        bbox_min = center - half_dims
        bbox_max = center + half_dims

        all_adj_captured = True
        for adj_verts in adjacent_surfaces_verts:
            if not is_surface_in_bbox(adj_verts, bbox_min, bbox_max):
                all_adj_captured = False
                break
        
        if all_adj_captured:
            return ratio, True

    return 1.5, False  # Loop finished without capturing all adjacent surfaces


def analyze_file(file_path):
    """Analyze a single JSON file and return statistics for each surface."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    surfaces, adjacency_map = get_surfaces_and_adjacency(data)
    results = []

    for surface_idx, surface_data in surfaces.items():
        adjacent_indices = adjacency_map.get(surface_idx, [])
        
        optimal_ratio, all_captured = find_optimal_ratio(surface_data, surfaces, adjacent_indices)
        
        # Calculate final statistics based on the optimal ratio
        selected_verts = np.array(surface_data['points']).reshape(-1, 3)
        min_coords = np.min(selected_verts, axis=0)
        max_coords = np.max(selected_verts, axis=0)
        surface_size = max_coords - min_coords
        surface_size[surface_size < 1e-6] = 1e-6
        bbox_dims = optimal_ratio * surface_size
        if surface_data.get("type") == 'plane':
            min_idx = np.argmin(bbox_dims)
            if bbox_dims[min_idx] < 0.01:
                bbox_dims[min_idx] = 0.01
        
        center = (min_coords + max_coords) / 2.0
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

        results.append({
            "file_path": file_path,
            "surface_index": surface_idx,
            "surface_type": surface_data.get("type", "unknown"),
            "num_adjacent": len(adjacent_indices),
            "optimal_ratio": optimal_ratio,
            "all_adj_captured": all_captured,
            "num_in_bbox_at_optimal": len(surfaces_in_bbox),
            "num_adj_not_in_bbox_at_optimal": num_adj_not_in_bbox
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze JSON files to find optimal bounding box ratios."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing JSON files."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to the output CSV file."
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_file = Path(args.output)
    
    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    json_files = find_json_files(input_dir)
    if not json_files:
        print(f"No JSON files found in {input_dir}.")
        return

    print(f"Found {len(json_files)} JSON files. Analyzing...")

    all_results = []
    for file_path in tqdm(json_files, desc="Processing files"):
        file_results = analyze_file(file_path)
        all_results.extend(file_results)

    if not all_results:
        print("No processable surfaces found in any files.")
        return

    # Write results to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_path", "surface_index", "surface_type", "num_adjacent", 
        "optimal_ratio", "all_adj_captured", 
        "num_in_bbox_at_optimal", "num_adj_not_in_bbox_at_optimal"
    ]
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nAnalysis complete. Results saved to {output_file}")
    except IOError as e:
        print(f"\nError writing to output file {output_file}: {e}")


if __name__ == "__main__":
    main() 