import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

def find_json_files(directory):
    """Find all .json files in directory and subdirectories"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def is_surface_type(feature_type):
    """Check if the feature type is a surface"""
    surface_types = {'plane', 'cylinder', 'cone', 'sphere', 'torus', 'bezier_surface', 'bspline_surface'}
    return feature_type in surface_types

def is_curve_type(feature_type):
    """Check if the feature type is a curve"""
    curve_types = {'line', 'circle', 'ellipse', 'hyperbola', 'parabola', 'bezier_curve', 'bspline_curve'}
    return feature_type in curve_types

def process_single_json(json_file_path):
    """Process a single JSON file and extract surface and curve data"""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file_path}: {e}")
        return None
    
    # Create mappings for surfaces and group curves by face indices in one pass
    surface_map = {}
    curves_by_face = {}  # face_idx -> list of curves
    
    for feature in data:
        if 'type' not in feature or 'idx' not in feature:
            continue
            
        if is_surface_type(feature['type']):
            face_idx = feature['idx'][0]  # For surfaces, idx is [face_idx, face_idx]
            surface_map[face_idx] = feature
            # Initialize curves list for this face if not exists
            if face_idx not in curves_by_face:
                curves_by_face[face_idx] = []
                
        elif is_curve_type(feature['type']):
            curve_idx = feature['idx']
            # For edges, idx is [face1_idx, face2_idx]
            # Add curve to both faces it connects
            for face_idx in curve_idx:
                if face_idx not in curves_by_face:
                    curves_by_face[face_idx] = []
                curves_by_face[face_idx].append(feature)
    
    # Initialize result lists
    surface_cp_list = []
    surface_points_list = []
    curve_cp_lists = []
    curve_points_lists = []
    surface_indices = []
    surface_types = []
    
    # Process each surface (single loop)
    for face_idx, surface in surface_map.items():
        # Store surface data
        surface_approximation = surface.get('approximation', None)
        surface_points = surface.get('points', None)
        
        if surface_approximation is not None:
            surface_cp_list.append(surface_approximation)
        else:
            surface_cp_list.append([])
            
        if surface_points is not None:
            surface_points_list.append(surface_points)
        else:
            surface_points_list.append([])
        
        # Get curves on this surface (already grouped)
        curves_on_surface = curves_by_face.get(face_idx, [])
        curves_on_surface_cp = []
        curves_on_surface_points = []
        
        for curve in curves_on_surface:
            curve_approximation = curve.get('approximation', None)
            curve_points = curve.get('points', None)
            
            if curve_approximation is not None:
                curves_on_surface_cp.append(curve_approximation)
            else:
                curves_on_surface_cp.append([])
                
            if curve_points is not None:
                curves_on_surface_points.append(curve_points)
            else:
                curves_on_surface_points.append([])
        
        curve_cp_lists.append(curves_on_surface_cp)
        curve_points_lists.append(curves_on_surface_points)
        surface_indices.append(face_idx)
        surface_types.append(surface.get('type', 'unknown'))
    
    # Extract UID from filename (remove .json extension)
    filename = os.path.basename(json_file_path)
    uid = os.path.splitext(filename)[0]  # Remove .json extension
    
    return {
        'surface_cp_list': surface_cp_list,
        'surface_points_list': surface_points_list,
        'curve_cp_lists': curve_cp_lists,
        'curve_points_lists': curve_points_lists,
        'surface_indices': surface_indices,
        'surface_types': surface_types,
        'file_path': json_file_path,
        'uid': uid
    }

def process_directory(input_directory, output_directory):
    """Process all JSON files in directory and save individual NPZ files per UID"""
    json_files = find_json_files(input_directory)
    print(f"Found {len(json_files)} JSON files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process each JSON file and save individual NPZ
    successful_files = 0
    total_surfaces = 0
    total_curves = 0
    
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        result = process_single_json(json_file)
        
        if result is not None:
            uid = result['uid']
            
            # Convert lists to numpy arrays with proper handling of variable-length data
            # For fixed-size data, convert to regular arrays
            surface_indices_array = np.array(result['surface_indices'])
            surface_types_array = np.array(result['surface_types'])
            file_paths_array = np.array([result['file_path']] * len(result['surface_cp_list']))
            uids_array = np.array([uid] * len(result['surface_cp_list']))
            
            # For variable-length data, use object arrays
            surface_cp_array = np.array(result['surface_cp_list'], dtype=object)
            surface_points_array = np.array(result['surface_points_list'], dtype=object)
            curve_cp_array = np.array(result['curve_cp_lists'], dtype=object)
            curve_points_array = np.array(result['curve_points_lists'], dtype=object)
            
            # Save individual NPZ file for this UID
            output_file = os.path.join(output_directory, f"{uid}.npz")
            np.savez_compressed(
                output_file,
                surface_cp_list=surface_cp_array,
                surface_points_list=surface_points_array,
                curve_cp_lists=curve_cp_array,
                curve_points_lists=curve_points_array,
                surface_indices=surface_indices_array,
                surface_types=surface_types_array,
                file_paths=file_paths_array,
                uids=uids_array
            )
            
            # Update statistics
            successful_files += 1
            num_surfaces = len(result['surface_cp_list'])
            num_curves = sum(len(curves) for curves in result['curve_cp_lists'])
            total_surfaces += num_surfaces
            total_curves += num_curves
            
            if successful_files % 100 == 0:  # Progress update every 100 files
                print(f"  Processed {successful_files} files, {total_surfaces} surfaces so far...")
    
    print(f"\nSuccessfully processed {successful_files}/{len(json_files)} files")
    print(f"Total surfaces: {total_surfaces}")
    print(f"Output directory: {output_directory}")
    
    # Print some statistics
    avg_curves_per_surface = total_curves / total_surfaces if total_surfaces else 0
    avg_surfaces_per_file = total_surfaces / successful_files if successful_files else 0
    
    print(f"\nStatistics:")
    print(f"  Total surfaces: {total_surfaces}")
    print(f"  Total curves: {total_curves}")
    print(f"  Average curves per surface: {avg_curves_per_surface:.2f}")
    print(f"  Average surfaces per file: {avg_surfaces_per_file:.2f}")
    print(f"  NPZ files created: {successful_files}")
    
    # Create a summary file with overall statistics
    summary_file = os.path.join(output_directory, "dataset_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Dataset Processing Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Total JSON files found: {len(json_files)}\n")
        f.write(f"Successfully processed: {successful_files}\n")
        f.write(f"Total surfaces: {total_surfaces}\n")
        f.write(f"Total curves: {total_curves}\n")
        f.write(f"Average curves per surface: {avg_curves_per_surface:.2f}\n")
        f.write(f"Average surfaces per file: {avg_surfaces_per_file:.2f}\n")
        f.write(f"NPZ files created: {successful_files}\n")
        f.write(f"Output directory: {output_directory}\n")
    
    print(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Process JSON CAD files and extract surface/curve data")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing JSON files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for individual NPZ files")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Process directory and create individual NPZ files
    process_directory(args.input, args.output)

if __name__ == "__main__":
    main() 