import sys
import json
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from utils.surface import visualize_json_interset

# Global variables
_npz_files = []
_current_idx = 0
_max_idx = 0
_data_folder = ""
_current_data = None
_show_valid_pc = True
_show_invalid_pc = True
_show_surfaces = True
_ps_initialized = False
_point_cloud_handlers = {}
_surface_group = None
_valid_pc_group = None
_invalid_pc_group = None
_surfaces = {}


def scan_npz_files(folder_path: str) -> List[str]:
    """Scan folder for .npz files and return sorted list of paths"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Warning: Folder '{folder_path}' does not exist or is not a directory")
        return []
    
    npz_files = sorted(folder.glob("*.npz"))
    npz_files = [str(f) for f in npz_files]
    
    print(f"Found {len(npz_files)} .npz files in {folder_path}")
    return npz_files


def load_npz_data(npz_path: str) -> Dict:
    """Load npz file and corresponding json file"""
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} does not exist")
        return None
    
    # Load npz
    data = np.load(npz_path, allow_pickle=True)
    points_list = data['points']  # List of arrays, each (H*W, 3)
    normals_list = data['normals']  # List of arrays, each (H*W, 3)
    masks_list = data['masks']  # List of arrays, each (H*W,) boolean
    
    # Load corresponding json
    json_path = npz_path.replace('.npz', '.json')
    surface_jsons = None
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            surface_jsons = json.load(f)
        print(f"Loaded {len(surface_jsons)} surfaces from {json_path}")
    else:
        print(f"Warning: JSON file not found: {json_path}")
    
    return {
        'points': points_list,
        'normals': normals_list,
        'masks': masks_list,
        'surface_jsons': surface_jsons,
        'npz_path': npz_path,
        'json_path': json_path if os.path.exists(json_path) else None
    }


def reset_scene():
    """Clear all polyscope structures except groups"""
    if not _ps_initialized:
        return
    # Remove all structures but groups will be preserved
    ps.remove_all_structures()


def update_visualization():
    """Update visualization with current npz file"""
    global _current_idx, _current_data, _npz_files
    global _show_valid_pc, _show_invalid_pc, _show_surfaces
    global _point_cloud_handlers, _surfaces
    global _surface_group, _valid_pc_group, _invalid_pc_group
    
    if not _ps_initialized or not _npz_files:
        return
    
    if _current_idx < 0 or _current_idx >= len(_npz_files):
        print(f"Invalid index: {_current_idx}")
        return
    
    # Clear scene (groups are preserved)
    reset_scene()
    
    # Load data
    npz_path = _npz_files[_current_idx]
    print(f"\n{'='*60}")
    print(f"Loading: {os.path.basename(npz_path)}")
    print(f"{'='*60}")
    
    _current_data = load_npz_data(npz_path)
    
    if _current_data is None:
        print("Failed to load data")
        return
    
    points_list = _current_data['points']
    normals_list = _current_data['normals']
    masks_list = _current_data['masks']
    surface_jsons = _current_data['surface_jsons']
    
    print(f"Number of surfaces: {len(points_list)}")
    
    # Visualize point clouds for each surface
    _point_cloud_handlers = {}
    total_valid = 0
    total_invalid = 0
    
    for i in range(len(points_list)):
        points = points_list[i]
        normals = normals_list[i]
        masks = masks_list[i]
        
        # Split into valid and invalid points
        valid_mask = masks.astype(bool)
        invalid_mask = ~valid_mask
        
        valid_points = points[valid_mask]
        invalid_points = points[invalid_mask]
        
        total_valid += len(valid_points)
        total_invalid += len(invalid_points)
        
        # Register valid points (green)
        if len(valid_points) > 0:
            pc_name = f"surface_{i:03d}_valid"
            try:
                pc_handler = ps.register_point_cloud(pc_name, valid_points, radius=0.002)
                pc_handler.set_color([0.2, 0.8, 0.2])  # Green
                pc_handler.add_to_group(_valid_pc_group)
                _point_cloud_handlers[pc_name] = pc_handler
            except Exception as e:
                print(f"Warning: Could not register valid point cloud {i}: {e}")
        
        # Register invalid points (red)
        if len(invalid_points) > 0:
            pc_name = f"surface_{i:03d}_invalid"
            try:
                pc_handler = ps.register_point_cloud(pc_name, invalid_points, radius=0.002)
                pc_handler.set_color([0.8, 0.2, 0.2])  # Red
                pc_handler.add_to_group(_invalid_pc_group)
                _point_cloud_handlers[pc_name] = pc_handler
            except Exception as e:
                print(f"Warning: Could not register invalid point cloud {i}: {e}")
    
    print(f"Point clouds: {total_valid} valid (green), {total_invalid} invalid (red)")
    
    # Visualize surfaces using visualize_json_interset
    _surfaces = {}
    if surface_jsons is not None and len(surface_jsons) > 0:
        try:
            print(f"Visualizing {len(surface_jsons)} surfaces...")
            _surfaces = visualize_json_interset(
                surface_jsons, 
                plot=True, 
                plot_gui=False, 
                tol=1e-3,
                ps_header=f'surface_{_current_idx}'
            )
            
            # Add all surfaces to group
            for surface_key, surface_data in _surfaces.items():
                if 'ps_handler' in surface_data and surface_data['ps_handler'] is not None:
                    surface_data['ps_handler'].add_to_group(_surface_group)
            
            print(f"Successfully visualized {len(_surfaces)} surfaces")
        except Exception as e:
            print(f"Error visualizing surfaces: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No surface JSON data available")
    
    # Set group visibility
    _valid_pc_group.set_enabled(_show_valid_pc)
    _invalid_pc_group.set_enabled(_show_invalid_pc)
    _surface_group.set_enabled(_show_surfaces)
    
    print(f"{'='*60}\n")


def callback():
    """Polyscope callback function for UI controls"""
    global _current_idx, _max_idx
    global _show_valid_pc, _show_invalid_pc, _show_surfaces
    global _valid_pc_group, _invalid_pc_group, _surface_group
    global _current_data
    
    psim.Text("Trimmed Point Cloud Visualizer")
    psim.Separator()
    
    # File info
    if _npz_files and _current_idx >= 0 and _current_idx < len(_npz_files):
        current_file = os.path.basename(_npz_files[_current_idx])
        psim.Text(f"File: {current_file}")
        psim.Text(f"Index: {_current_idx} / {_max_idx}")
    
    psim.Separator()
    
    # Navigation controls
    psim.Text("Navigation:")
    
    # Slider
    slider_changed, slider_idx = psim.SliderInt("File Index", _current_idx, 0, _max_idx)
    if slider_changed and slider_idx != _current_idx:
        _current_idx = slider_idx
        update_visualization()
    
    # Navigation buttons
    psim.PushItemWidth(80)
    if psim.Button("First"):
        if _current_idx != 0:
            _current_idx = 0
            update_visualization()
    
    psim.SameLine()
    if psim.Button("Prev"):
        if _current_idx > 0:
            _current_idx -= 1
            update_visualization()
    
    psim.SameLine()
    if psim.Button("Next"):
        if _current_idx < _max_idx:
            _current_idx += 1
            update_visualization()
    
    psim.SameLine()
    if psim.Button("Last"):
        if _current_idx != _max_idx:
            _current_idx = _max_idx
            update_visualization()
    psim.PopItemWidth()
    
    # Visibility controls
    if _valid_pc_group is not None and _invalid_pc_group is not None and _surface_group is not None:
        psim.Separator()
        psim.Text("Visibility Controls:")
        
        changed_valid, _show_valid_pc = psim.Checkbox("Show Valid Points (Green)", _show_valid_pc)
        if changed_valid:
            _valid_pc_group.set_enabled(_show_valid_pc)
        
        changed_invalid, _show_invalid_pc = psim.Checkbox("Show Invalid Points (Red)", _show_invalid_pc)
        if changed_invalid:
            _invalid_pc_group.set_enabled(_show_invalid_pc)
        
        changed_surfaces, _show_surfaces = psim.Checkbox("Show Surfaces", _show_surfaces)
        if changed_surfaces:
            _surface_group.set_enabled(_show_surfaces)
    
    # Statistics
    if _current_data is not None:
        psim.Separator()
        psim.Text("Statistics:")
        
        points_list = _current_data['points']
        masks_list = _current_data['masks']
        
        total_valid = sum(np.sum(mask) for mask in masks_list)
        total_invalid = sum(np.sum(~mask.astype(bool)) for mask in masks_list)
        total_points = total_valid + total_invalid
        
        psim.Text(f"Surfaces: {len(points_list)}")
        psim.Text(f"Total points: {total_points}")
        psim.Text(f"Valid: {total_valid} ({100*total_valid/total_points:.1f}%)")
        psim.Text(f"Invalid: {total_invalid} ({100*total_invalid/total_points:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize trimmed point clouds from NPZ files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'data_folder',
        type=str,
        help='Folder containing .npz files to visualize'
    )
    parser.add_argument(
        '--start_idx',
        type=int,
        default=0,
        help='Starting file index'
    )
    
    args = parser.parse_args()
    
    global _npz_files, _current_idx, _max_idx, _data_folder, _ps_initialized
    
    # Scan for npz files
    _data_folder = args.data_folder
    _npz_files = scan_npz_files(_data_folder)
    
    if not _npz_files:
        print(f"Error: No .npz files found in {_data_folder}")
        return 1
    
    _max_idx = len(_npz_files) - 1
    _current_idx = min(args.start_idx, _max_idx)
    
    print(f"\nFound {len(_npz_files)} files")
    print(f"Starting at index {_current_idx}")
    
    # Initialize polyscope
    ps.init()
    _ps_initialized = True
    
    # Create groups (only once at initialization)
    global _surface_group, _valid_pc_group, _invalid_pc_group
    _surface_group = ps.create_group("Surfaces")
    _valid_pc_group = ps.create_group("Valid Point Clouds")
    _invalid_pc_group = ps.create_group("Invalid Point Clouds")
    
    # Set user callback
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show interface
    ps.show()
    
    return 0


if __name__ == '__main__':
    exit(main())

