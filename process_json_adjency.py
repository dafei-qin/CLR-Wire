#!/usr/bin/env python3
"""
Process surface data from JSON files into individual pickle files.

This script reads CAD data from JSON files, extracts information for each surface,
and computes its local context, including nearby surfaces and adjacency information.
Each surface is saved as an individual pickle file with the naming convention:
<uid>_<surface_idx>.pkl
"""
import argparse
import json
import multiprocessing
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

SURFACE_TYPE_MAP = {
    'plane': 0,
    'cylinder': 1,
    'cone': 2,
    'sphere': 3,
    'torus': 4,
    'bspline_surface': 5,
}
BBOX_RATIO = 1.10
MAX_NEARBY_SURFACES = 20

def find_json_files(input_dir):
    """Recursively find all JSON files in the input directory."""
    input_path = Path(input_dir)
    return sorted([str(f) for f in input_path.rglob("*.json")])

def extract_uid_from_filename(file_path):
    """Extract UID from JSON filename."""
    filename = Path(file_path).stem
    return filename

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
    return surfaces, dict(surface_adjacency)

def upsample_surface(vertices, factor=4):
    """
    Upsamples the surface vertices assuming they form a square grid.
    """

    grid_x = vertices[:, 0]
    grid_y = vertices[:, 1]
    grid_z = vertices[:, 2]

    zoomed_x = zoom(grid_x, factor, order=1)
    zoomed_y = zoom(grid_y, factor, order=1)
    zoomed_z = zoom(grid_z, factor, order=1)

    upsampled_vertices = np.vstack([zoomed_x.ravel(), zoomed_y.ravel(), zoomed_z.ravel()]).T
    
    return upsampled_vertices

def is_surface_in_bbox(surface_verts, bbox_min, bbox_max):
    """
    Check if any vertex of a surface is within the given bounding box.
    If not, upsample the vertices and check again.
    """
    if np.any(np.all((surface_verts >= bbox_min) & (surface_verts <= bbox_max), axis=1)):
        return True

    upsampled_verts = upsample_surface(surface_verts, factor=4)
    
    if upsampled_verts is not None:
        return np.any(np.all((upsampled_verts >= bbox_min) & (upsampled_verts <= bbox_max), axis=1))

    return False

def process_file(args_tuple):
    """
    Process a single JSON file and save individual surface data.
    """
    file_path, output_dir = args_tuple
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return 0

    uid = extract_uid_from_filename(file_path)
    surfaces, adjacency_map = get_surfaces_and_adjacency(data)
    saved_count = 0

    for s_idx, s_data in surfaces.items():
        if not s_data.get('points') or s_data.get('type') not in SURFACE_TYPE_MAP:
            continue

        s_points = np.array(s_data['points']).reshape(-1, 3)
        s_type = SURFACE_TYPE_MAP[s_data['type']]
        s_bbox = np.array([np.min(s_points, axis=0), np.max(s_points, axis=0)])

        # Calculate enlarged bbox
        center = np.mean(s_bbox, axis=0)
        size = s_bbox[1] - s_bbox[0]
        size[size < 1e-6] = 1e-6
        enlarged_size = size * BBOX_RATIO
        
        if s_data.get("type") == 'plane':
            min_idx = np.argmin(enlarged_size)
            if enlarged_size[min_idx] < 0.01:
                enlarged_size[min_idx] = 0.01

        bbox_min = center - enlarged_size / 2.0
        bbox_max = center + enlarged_size / 2.0

        # Find nearby surfaces
        nearby_surfaces = []
        for o_idx, o_data in surfaces.items():
            if o_idx == s_idx or not o_data.get('points') or o_data.get('type') not in SURFACE_TYPE_MAP:
                continue
            
            o_points = np.array(o_data['points']).reshape(-1, 3)
            if is_surface_in_bbox(o_points, bbox_min, bbox_max):
                nearby_surfaces.append({
                    'idx': o_idx,
                    'type': SURFACE_TYPE_MAP[o_data['type']],
                    'points': o_points.reshape(32, 32, 3),
                    'bbox': np.array([np.min(o_points, axis=0), np.max(o_points, axis=0)])
                })

        if len(nearby_surfaces) > MAX_NEARBY_SURFACES:
            continue
        if len(nearby_surfaces) == 0:
            continue
            
        # Create adjacency mask
        s_adj = adjacency_map.get(s_idx, set())
        adj_mask = np.zeros(MAX_NEARBY_SURFACES, dtype=np.int8)
        
        for i, nearby in enumerate(nearby_surfaces):
            if nearby['idx'] in s_adj:
                adj_mask[i] = 1

        # Prepare surface data
        surface_data = {
            'type': s_type,
            'points': s_points.reshape(32, 32, 3),
            'bbox': s_bbox,
            'nearby': nearby_surfaces,
            'adj_mask': adj_mask,
        }

        # Save individual surface file
        output_filename = f"{uid}_{s_idx}.pkl"
        output_path = Path(output_dir) / output_filename
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(surface_data, f)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {output_path}: {e}")

    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Process surface data from JSON to individual pickle files.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing JSON files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for individual pickle files.")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes to use (default: CPU count).")
    args = parser.parse_args()

    json_files = find_json_files(args.input)
    if not json_files:
        print(f"No JSON files found in {args.input}")
        return

    print(f"Found {len(json_files)} JSON files. Starting processing...")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for multiprocessing
    process_args = [(file_path, str(output_dir)) for file_path in json_files]
    
    num_processes = args.processes or multiprocessing.cpu_count()
    total_surfaces_saved = 0
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_iterator = pool.imap_unordered(process_file, process_args)
        
        for surfaces_saved in tqdm(results_iterator, total=len(json_files), desc="Processing files"):
            total_surfaces_saved += surfaces_saved

    print(f"Processing complete! Saved {total_surfaces_saved} individual surface files to {output_dir}")

if __name__ == "__main__":
    main() 