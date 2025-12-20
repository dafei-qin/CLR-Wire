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
_show_edges = True
_ps_initialized = False
_point_cloud_handlers = {}
_surface_group = None
_valid_pc_group = None
_invalid_pc_group = None
_edges_group = None
_surfaces = {}
_edge_handlers = {}


def scan_npz_files(folder_path: str) -> List[str]:
    """Scan folder for .npz files and return sorted list of paths"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Warning: Folder '{folder_path}' does not exist or is not a directory")
        return []
    
    npz_files = sorted(folder.rglob("*.npz"))
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
    
    # Load graph data if available
    graph_nodes = None
    graph_edges = None
    if 'graph_nodes' in data:
        graph_nodes = data['graph_nodes']
        print(f"Loaded graph with {len(graph_nodes)} nodes")
    if 'graph_edges' in data:
        graph_edges = data['graph_edges']
        print(f"Loaded {len(graph_edges)} edges")
    
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
        'graph_nodes': graph_nodes,
        'graph_edges': graph_edges,
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
    global _show_valid_pc, _show_invalid_pc, _show_surfaces, _show_edges
    global _point_cloud_handlers, _surfaces, _edge_handlers
    global _surface_group, _valid_pc_group, _invalid_pc_group, _edges_group
    
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
    
    # Visualize adjacency edges
    _edge_handlers = {}
    graph_edges = _current_data['graph_edges']
    graph_nodes = _current_data['graph_nodes']
    
    if graph_edges is not None and len(graph_edges) > 0:
        print(f"Visualizing {len(graph_edges)} adjacency edges...")
        
        # Compute centroids of valid points for each surface
        surface_centroids = {}
        for i in range(len(points_list)):
            points = points_list[i]
            masks = masks_list[i]
            valid_mask = masks.astype(bool)
            valid_points = points[valid_mask]
            
            if len(valid_points) > 0:
                centroid = np.mean(valid_points, axis=0)
                surface_centroids[i] = centroid
            else:
                # Fallback to all points if no valid points
                if len(points) > 0:
                    surface_centroids[i] = np.mean(points, axis=0)
        
        # Build edge visualization
        edge_count = 0
        for edge_idx, edge in enumerate(graph_edges):
            face_i, face_j = int(edge[0]), int(edge[1])
            
            # Check if both surfaces have centroids
            if face_i in surface_centroids and face_j in surface_centroids:
                centroid_i = surface_centroids[face_i]
                centroid_j = surface_centroids[face_j]
                
                # Create edge as curve network with 2 nodes and 1 edge
                edge_nodes = np.array([centroid_i, centroid_j])  # (2, 3)
                edge_connections = np.array([[0, 1]])  # (1, 2)
                
                edge_name = f"edge_{edge_idx:04d}_f{face_i}_f{face_j}"
                try:
                    edge_handler = ps.register_curve_network(
                        edge_name, 
                        edge_nodes, 
                        edge_connections,
                        radius=0.0015
                    )
                    edge_handler.set_color([0.3, 0.3, 0.8])  # Blue for edges
                    edge_handler.add_to_group(_edges_group)
                    _edge_handlers[edge_name] = edge_handler
                    edge_count += 1
                except Exception as e:
                    print(f"Warning: Could not visualize edge {edge_idx}: {e}")
        
        print(f"Successfully visualized {edge_count} edges")
    else:
        print("No graph edge data available")
    
    # Set group visibility
    _valid_pc_group.set_enabled(_show_valid_pc)
    _invalid_pc_group.set_enabled(_show_invalid_pc)
    _surface_group.set_enabled(_show_surfaces)
    _edges_group.set_enabled(_show_edges)
    
    print(f"{'='*60}\n")


def callback():
    """Polyscope callback function for UI controls"""
    global _current_idx, _max_idx
    global _show_valid_pc, _show_invalid_pc, _show_surfaces, _show_edges
    global _valid_pc_group, _invalid_pc_group, _surface_group, _edges_group
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
        
        if _edges_group is not None:
            changed_edges, _show_edges = psim.Checkbox("Show Adjacency Edges (Blue)", _show_edges)
            if changed_edges:
                _edges_group.set_enabled(_show_edges)
    
    # Statistics
    if _current_data is not None:
        psim.Separator()
        psim.Text("Statistics:")
        
        points_list = _current_data['points']
        masks_list = _current_data['masks']
        graph_edges = _current_data.get('graph_edges', None)
        
        total_valid = sum(np.sum(mask) for mask in masks_list)
        total_invalid = sum(np.sum(~mask.astype(bool)) for mask in masks_list)
        total_points = total_valid + total_invalid
        
        psim.Text(f"Surfaces: {len(points_list)}")
        if graph_edges is not None:
            psim.Text(f"Adjacency edges: {len(graph_edges)}")
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
    global _surface_group, _valid_pc_group, _invalid_pc_group, _edges_group
    _surface_group = ps.create_group("Surfaces")
    _valid_pc_group = ps.create_group("Valid Point Clouds")
    _invalid_pc_group = ps.create_group("Invalid Point Clouds")
    _edges_group = ps.create_group("Adjacency Edges")
    
    # Set user callback
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show interface
    ps.show()
    
    return 0


if __name__ == '__main__':
    exit(main())

