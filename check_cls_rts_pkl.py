#!/usr/bin/env python3
"""
Visualize surface meshes from aggregated .pkl dataset using polyscope.

This script loads a .pkl file created by aggregate_surface_dataset.py and
visualizes the surface points as meshes, showing the surface type for each.
Supports filtering by surface type both via command line and interactive GUI.

Usage:
    python check_cls_rts_pkl.py path/to/data.pkl [--max-surfaces N] [--surface-type TYPE]
    
Examples:
    # Show all surfaces
    python check_cls_rts_pkl.py data.pkl
    
    # Show only first 10 surfaces
    python check_cls_rts_pkl.py data.pkl --max-surfaces 10
    
    # Show only cylinder surfaces
    python check_cls_rts_pkl.py data.pkl --surface-type cylinder
    
    # Show first 5 plane surfaces
    python check_cls_rts_pkl.py data.pkl --max-surfaces 5 --surface-type plane
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import polyscope as ps
import polyscope.imgui as psim


# Surface type mapping (inverse of CLASS_MAPPING from aggregate_surface_dataset.py)
CLASS_NAMES = {
    0: "plane",
    1: "cylinder", 
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "bspline_surface",
    6: "bezier_surface",
}

# Colors for different surface types
SURFACE_COLORS = {
    0: (1.0, 0.0, 0.0, 1.0),    # plane - red
    1: (0.0, 1.0, 0.0, 1.0),    # cylinder - green
    2: (0.0, 0.0, 1.0, 1.0),    # cone - blue
    3: (1.0, 1.0, 0.0, 1.0),    # sphere - yellow
    4: (1.0, 0.0, 1.0, 1.0),    # torus - magenta
    5: (0.0, 1.0, 1.0, 1.0),    # bspline_surface - cyan
    6: (1.0, 0.5, 0.0, 1.0),    # bezier_surface - orange
}


def create_grid_faces(rows: int, cols: int) -> np.ndarray:
    """
    Create triangle faces for a structured grid of points.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        Array of shape (2*(rows-1)*(cols-1), 3) containing triangle indices
    """
    faces = []
    
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Current grid cell corners
            top_left = i * cols + j
            top_right = i * cols + (j + 1)
            bottom_left = (i + 1) * cols + j
            bottom_right = (i + 1) * cols + (j + 1)
            
            # Two triangles per grid cell
            faces.append([top_left, bottom_left, top_right])
            faces.append([top_right, bottom_left, bottom_right])
    
    return np.array(faces, dtype=np.int32)


def load_dataset(pkl_path: Path) -> Dict[str, Any]:
    """Load the .pkl dataset file."""
    print(f"Loading dataset from {pkl_path}...")
    with pkl_path.open('rb') as f:
        data = pickle.load(f)
    
    print(f"Dataset contains {len(data['points'])} surfaces")
    print("Surface type distribution:")
    
    # Count surfaces by type
    unique_labels, counts = np.unique(data['class_label'], return_counts=True)
    for label, count in zip(unique_labels, counts):
        surface_name = CLASS_NAMES.get(label, f"unknown_{label}")
        print(f"  {surface_name}: {count}")
    
    return data


def visualize_surfaces(data: Dict[str, Any], max_surfaces: int = None, surface_type_filter: str = None):
    """Visualize surfaces using polyscope."""
    
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    points_array = data['points']
    class_labels = data['class_label']
    uids = data.get('uid', [f'surface_{i}' for i in range(len(points_array))])
    indices = data.get('index', list(range(len(points_array))))
    
    # Filter by surface type if specified
    if surface_type_filter is not None:
        # Find the class ID for the requested surface type
        target_class_id = None
        for class_id, name in CLASS_NAMES.items():
            if name.lower() == surface_type_filter.lower():
                target_class_id = class_id
                break
        
        if target_class_id is None:
            print(f"Error: Unknown surface type '{surface_type_filter}'")
            print(f"Available types: {list(CLASS_NAMES.values())}")
            return
        
        # Filter data to only include the specified surface type
        filtered_indices = [i for i, label in enumerate(class_labels) if label == target_class_id]
        
        if not filtered_indices:
            print(f"No surfaces of type '{surface_type_filter}' found in dataset")
            return
        
        print(f"Filtering to {len(filtered_indices)} surfaces of type '{surface_type_filter}'")
        
        # Apply filtering
        points_array = [points_array[i] for i in filtered_indices]
        class_labels = [class_labels[i] for i in filtered_indices]
        uids = [uids[i] if i < len(uids) else f'surface_{i}' for i in filtered_indices]
        indices = [indices[i] if i < len(indices) else i for i in filtered_indices]
    
    # Limit number of surfaces if specified
    n_surfaces = len(points_array)
    if max_surfaces is not None:
        n_surfaces = min(max_surfaces, n_surfaces)
        print(f"Showing first {n_surfaces} surfaces out of {len(points_array)}")
    
    # Create grid faces (assuming all surfaces are 32x32 grids)
    grid_faces = create_grid_faces(32, 32)
    
    # Store original data for random selection (moved before usage)
    original_points = data['points']
    original_labels = data['class_label']
    original_uids = data.get('uid', [f'surface_{i}' for i in range(len(original_points))])
    original_indices = data.get('index', list(range(len(original_points))))
    
    # Initially load random surfaces automatically
    # Note: surfaces will be loaded automatically when load_random_surfaces() is called in the UI setup
    print(f"\nVisualization ready! Use the GUI to load random surfaces.")
    print("Available surface types and colors:")
    for class_id, name in CLASS_NAMES.items():
        if class_id in np.unique(original_labels):
            color = SURFACE_COLORS.get(class_id, (0.5, 0.5, 0.5, 1.0))
            print(f"  {name}: RGB{color}")
    
    # State for dynamic filtering and random selection
    current_filter = [0]  # 0 = All, 1-7 = specific surface types
    filter_options = ["All"] + list(CLASS_NAMES.values())
    visibility_state = {}  # Track visibility of each surface type
    
    # Current displayed data (will be updated by random selection)
    current_points = points_array
    current_labels = class_labels
    current_uids = uids
    current_indices = indices
    current_n_surfaces = n_surfaces
    
    # Initialize visibility state
    loaded_labels = class_labels[:n_surfaces]
    unique_loaded_labels = np.unique(loaded_labels)
    for label in unique_loaded_labels:
        visibility_state[label] = True
    
    def get_available_indices():
        """Get available indices based on current surface type filter."""
        if surface_type_filter is not None:
            # Find the class ID for the requested surface type
            target_class_id = None
            for class_id, name in CLASS_NAMES.items():
                if name.lower() == surface_type_filter.lower():
                    target_class_id = class_id
                    break
            
            if target_class_id is not None:
                return [i for i, label in enumerate(original_labels) if label == target_class_id]
        
        return list(range(len(original_points)))
    
    def load_random_surfaces():
        """Load random surfaces and update the visualization."""
        nonlocal current_points, current_labels, current_uids, current_indices, current_n_surfaces, visibility_state
        
        # Get available indices based on current filter
        available_indices = get_available_indices()
        
        if not available_indices:
            print("No surfaces available for random selection")
            return
        
        # Determine how many surfaces to load
        target_count = min(max_surfaces if max_surfaces else 20, len(available_indices))
        
        # Randomly sample indices
        random_indices = random.sample(available_indices, target_count)
        
        print(f"Loading {target_count} random surfaces...")
        
        # Clear existing surfaces
        ps.remove_all_structures()
        
        # Update current data
        current_points = [original_points[i] for i in random_indices]
        current_labels = [original_labels[i] for i in random_indices]
        current_uids = [original_uids[i] for i in random_indices]
        current_indices = [original_indices[i] for i in random_indices]
        current_n_surfaces = len(current_points)
        
        # Reset visibility state for new surfaces
        unique_labels = np.unique(current_labels)
        visibility_state = {label: True for label in unique_labels}
        
        # Add new surfaces to polyscope
        for i in range(current_n_surfaces):
            points_grid = current_points[i]  # Shape: (32, 32, 3)
            class_label = current_labels[i]
            uid = current_uids[i]
            surf_index = current_indices[i]
            
            # Flatten points grid to (32*32, 3)
            points_flat = points_grid.reshape(-1, 3)
            
            # Create mesh name with surface type
            surface_type = CLASS_NAMES.get(class_label, f"unknown_{class_label}")
            mesh_name = f"{surface_type}_{uid}_{surf_index}"
            
            # Add mesh to polyscope
            mesh = ps.register_surface_mesh(mesh_name, points_flat, grid_faces)
            
            # Set color based on surface type
            color = SURFACE_COLORS.get(class_label, (0.5, 0.5, 0.5, 1.0))
            mesh.set_color(color)
            
            # Add surface type as metadata
            mesh.add_scalar_quantity("surface_type_id", 
                                    np.full(len(points_flat), class_label))
        
        print(f"Loaded {current_n_surfaces} random surfaces")

    def update_surface_visibility():
        """Update surface visibility based on current settings."""
        for i in range(current_n_surfaces):
            class_label = current_labels[i]
            surface_type = CLASS_NAMES.get(class_label, f"unknown_{class_label}")
            uid = current_uids[i]
            surf_index = current_indices[i]
            mesh_name = f"{surface_type}_{uid}_{surf_index}"
            
            # Show/hide based on visibility state
            try:
                mesh = ps.get_surface_mesh(mesh_name)
                mesh.set_enabled(visibility_state.get(class_label, True))
            except:
                pass  # Mesh might not exist
    
    # Custom UI callback to show surface information
    def ui_callback():
        nonlocal current_filter, visibility_state
        
        psim.TextUnformatted("Surface Dataset Visualization")
        psim.Separator()
        
        # Random selection controls
        psim.TextUnformatted("Random Surface Selection:")
        
        # Random selection button
        if psim.Button("🎲 Load Random Surfaces"):
            load_random_surfaces()
        
        psim.SameLine()
        
        # Show current selection info
        available_count = len(get_available_indices())
        psim.TextUnformatted(f"(Available: {available_count})")
        
        psim.Separator()
        
        # Surface filtering controls
        psim.TextUnformatted("Surface Type Visibility Controls:")
        
        # Show surface type counts and visibility toggles
        if current_n_surfaces > 0:
            loaded_labels = current_labels[:current_n_surfaces]
            unique_labels, counts = np.unique(loaded_labels, return_counts=True)
            
            visibility_changed = False
            
            for label, count in zip(unique_labels, counts):
                surface_name = CLASS_NAMES.get(label, f"unknown_{label}")
                color = SURFACE_COLORS.get(label, (0.5, 0.5, 0.5, 1.0))
                
                # Visibility checkbox
                current_visibility = visibility_state.get(label, True)
                changed, new_visibility = psim.Checkbox(f"##vis_{label}", current_visibility)
                if changed:
                    visibility_state[label] = new_visibility
                    visibility_changed = True
                
                psim.SameLine()
                
                # Colored surface name and count
                psim.PushStyleColor(psim.ImGuiCol_Text, color)
                psim.TextUnformatted(f"{surface_name}: {count}")
                psim.PopStyleColor()
            
            # Update visibility if any checkbox changed
            if visibility_changed:
                update_surface_visibility()
            
            psim.Separator()
            
            # Quick filter buttons
            psim.TextUnformatted("Quick Filters:")
            
            if psim.Button("Show All"):
                for label in visibility_state:
                    visibility_state[label] = True
                update_surface_visibility()
            
            psim.SameLine()
            
            if psim.Button("Hide All"):
                for label in visibility_state:
                    visibility_state[label] = False
                update_surface_visibility()
            
            # Individual surface type quick buttons
            for label in unique_labels:
                surface_name = CLASS_NAMES.get(label, f"unknown_{label}")
                if psim.Button(f"Only {surface_name}"):
                    # Hide all, then show only this type
                    for l in visibility_state:
                        visibility_state[l] = (l == label)
                    update_surface_visibility()
                if label != unique_labels[-1]:  # Don't add same line after last button
                    psim.SameLine()
        else:
            psim.TextUnformatted("No surfaces loaded. Click 'Load Random Surfaces' to start.")
        
        psim.Separator()
        psim.TextUnformatted(f"Currently displayed: {current_n_surfaces} surfaces")
        psim.TextUnformatted(f"Total in dataset: {len(original_points)} surfaces")
        
        if surface_type_filter:
            psim.TextUnformatted(f"Command line filter: {surface_type_filter}")
        
        if max_surfaces:
            psim.TextUnformatted(f"Max surfaces per load: {max_surfaces}")
        else:
            psim.TextUnformatted("Max surfaces per load: 20 (default)")
    
    ps.set_user_callback(ui_callback)
    
    # Automatically load some random surfaces on startup
    print("Loading initial random surfaces...")
    load_random_surfaces()
    
    # Show the visualization
    ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize surface meshes from .pkl dataset using polyscope"
    )
    parser.add_argument("pkl_file", type=Path, 
                       help="Path to .pkl file created by aggregate_surface_dataset.py")
    parser.add_argument("--max-surfaces", type=int, default=None,
                       help="Maximum number of surfaces to visualize (default: all)")
    parser.add_argument("--surface-type", type=str, default=None,
                       help="Filter to show only specific surface type. "
                            f"Available types: {list(CLASS_NAMES.values())}")
    
    args = parser.parse_args()
    
    if not args.pkl_file.exists():
        print(f"Error: File {args.pkl_file} does not exist")
        return
    
    # Validate surface type if provided
    if args.surface_type is not None:
        valid_types = [name.lower() for name in CLASS_NAMES.values()]
        if args.surface_type.lower() not in valid_types:
            print(f"Error: Invalid surface type '{args.surface_type}'")
            print(f"Available types: {list(CLASS_NAMES.values())}")
            return
    
    # Load dataset
    try:
        data = load_dataset(args.pkl_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Visualize surfaces
    try:
        visualize_surfaces(data, args.max_surfaces, args.surface_type)
    except Exception as e:
        print(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main() 