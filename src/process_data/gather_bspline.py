#!/usr/bin/env python3
"""
Script to extract B-spline surface data from JSON files and save as numpy arrays.

This script recursively processes JSON files containing CAD data, extracts B-spline
surface parameters, and saves them as .npy files maintaining the original directory structure.
"""

import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_bspline_data(bspline_face):
    """
    Extract B-spline surface data from a face dictionary and convert to a single vector.
    
    Args:
        bspline_face (dict): Dictionary containing B-spline surface data with keys:
            - 'scalar': list containing degrees, pole counts, knot counts, knots, and multiplicities
            - 'u_periodic': boolean indicating if surface is periodic in u direction
            - 'v_periodic': boolean indicating if surface is periodic in v direction
            - 'poles': 2D list of control points, each with [x, y, z, w]
    
    Returns:
        np.ndarray: 1D array containing all B-spline parameters in order:
            [u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v,
             is_u_periodic, is_v_periodic, u_knots..., v_knots..., u_mults..., v_mults...,
             flattened_poles...]
    """
    scalar_data = bspline_face["scalar"]
    
    # Extract basic parameters (first 6 elements of scalar)
    u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v = map(int, scalar_data[:6])
    
    # Extract periodicity flags
    is_u_periodic = int(bspline_face["u_periodic"])
    is_v_periodic = int(bspline_face["v_periodic"])
    
    # Extract knot and multiplicity lists from scalar data
    u_knots_list = scalar_data[6 : 6 + num_knots_u]
    v_knots_list = scalar_data[6 + num_knots_u : 6 + num_knots_u + num_knots_v]
    u_mults_list = scalar_data[6 + num_knots_u + num_knots_v : 6 + num_knots_u + num_knots_v + num_knots_u]
    v_mults_list = scalar_data[6 + num_knots_u + num_knots_v + num_knots_u :]
    
    # Extract and flatten poles
    poles_data = bspline_face["poles"]
    # poles_data is shape (num_poles_u, num_poles_v, 4) where each pole is [x, y, z, w]
    flattened_poles = []
    for i in range(num_poles_u):
        for j in range(num_poles_v):
            flattened_poles.extend(poles_data[i][j])  # Append [x, y, z, w]
    
    # Construct the final vector
    vector = [
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v,
        is_u_periodic, is_v_periodic
    ]
    vector.extend(u_knots_list)
    vector.extend(v_knots_list)
    vector.extend(u_mults_list)
    vector.extend(v_mults_list)
    vector.extend(flattened_poles)
    
    return np.array(vector, dtype=np.float64)


def process_json_file(json_path, input_dir, save_dir):
    """
    Process a single JSON file and save B-spline surfaces found in it.
    
    Args:
        json_path (Path): Path to the JSON file
        input_dir (Path): Root input directory (for computing relative paths)
        save_dir (Path): Root save directory
    
    Returns:
        int: Number of B-spline surfaces saved
    """
    try:
        with open(json_path, 'r') as f:
            cad_data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return 0
    
    # Handle both list and dict formats
    if isinstance(cad_data, dict):
        # Might be a dict with a 'faces' key or similar
        if 'faces' in cad_data:
            faces_list = cad_data['faces']
        else:
            # Assume the dict itself represents faces
            faces_list = [cad_data]
    elif isinstance(cad_data, list):
        faces_list = cad_data
    else:
        print(f"Unexpected data format in {json_path}")
        return 0
    
    # Compute relative path to maintain directory structure
    rel_path = json_path.relative_to(input_dir)
    output_subdir = save_dir / rel_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Process each face in the file
    bspline_count = 0
    for face in faces_list:
        if not isinstance(face, dict):
            continue
            
        surface_type = face.get('type', '')
        if surface_type == 'bspline_surface':
            try:
                # Extract B-spline data
                bspline_vector = extract_bspline_data(face)
                
                # Generate output filename
                surface_idx = face.get('idx', [bspline_count])[0]
                output_filename = f"surface_{surface_idx:03d}.npy"
                output_path = output_subdir / output_filename
                
                # Save as numpy array
                np.save(output_path, bspline_vector)
                bspline_count += 1
                
            except Exception as e:
                print(f"Error processing B-spline in {json_path}, face {face.get('idx', 'unknown')}: {e}")
    
    return bspline_count


def gather_bspline_surfaces(input_dir, save_dir):
    """
    Recursively process all JSON files in input_dir and save B-spline surfaces to save_dir.
    
    Args:
        input_dir (str or Path): Directory containing JSON files
        save_dir (str or Path): Directory where .npy files will be saved
    """
    input_dir = Path(input_dir)
    save_dir = Path(save_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files recursively
    json_files = list(input_dir.rglob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    total_bsplines = 0
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        count = process_json_file(json_path, input_dir, save_dir)
        total_bsplines += count
    
    print(f"\nProcessing complete!")
    print(f"Total B-spline surfaces saved: {total_bsplines}")
    print(f"Output directory: {save_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract B-spline surfaces from JSON files and save as numpy arrays"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing JSON files to process"
    )
    parser.add_argument(
        "save_dir",
        type=str,
        help="Directory where .npy files will be saved (maintains subfolder structure)"
    )
    
    args = parser.parse_args()
    
    gather_bspline_surfaces(args.input_dir, args.save_dir)


if __name__ == "__main__":
    main()

