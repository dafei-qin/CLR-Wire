#!/usr/bin/env python3
"""
Visualize surface meshes from aggregated .pkl dataset using polyscope.

This script loads a .pkl file created by aggregate_surface_dataset.py and
visualizes the surface points as meshes, showing the surface type for each.

Usage:
    python visualize_surface_meshes.py path/to/data.pkl [--max-surfaces N]
"""

import argparse
import pickle
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
    0: (1.0, 0.0, 0.0),    # plane - red
    1: (0.0, 1.0, 0.0),    # cylinder - green
    2: (0.0, 0.0, 1.0),    # cone - blue
    3: (1.0, 1.0, 0.0),    # sphere - yellow
    4: (1.0, 0.0, 1.0),    # torus - magenta
    5: (0.0, 1.0, 1.0),    # bspline_surface - cyan
    6: (1.0, 0.5, 0.0),    # bezier_surface - orange
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


def visualize_surfaces(data: Dict[str, Any], max_surfaces: int = None):
    """Visualize surfaces using polyscope."""
    
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    points_array = data['points']
    class_labels = data['class_label']
    uids = data.get('uid', [f'surface_{i}' for i in range(len(points_array))])
    indices = data.get('index', list(range(len(points_array))))
    
    # Limit number of surfaces if specified
    n_surfaces = len(points_array)
    if max_surfaces is not None:
        n_surfaces = min(max_surfaces, n_surfaces)
        print(f"Showing first {n_surfaces} surfaces out of {len(points_array)}")
    
    # Create grid faces (assuming all surfaces are 32x32 grids)
    grid_faces = create_grid_faces(32, 32)
    
    # Add each surface as a mesh
    for i in range(n_surfaces):
        points_grid = points_array[i]  # Shape: (32, 32, 3)
        class_label = class_labels[i]
        uid = uids[i] if i < len(uids) else f'surface_{i}'
        surf_index = indices[i] if i < len(indices) else i
        
        # Flatten points grid to (32*32, 3)
        points_flat = points_grid.reshape(-1, 3)
        
        # Create mesh name with surface type
        surface_type = CLASS_NAMES.get(class_label, f"unknown_{class_label}")
        mesh_name = f"{surface_type}_{uid}_{surf_index}"
        
        # Add mesh to polyscope
        mesh = ps.register_surface_mesh(mesh_name, points_flat, grid_faces)
        
        # Set color based on surface type
        color = SURFACE_COLORS.get(class_label, (0.5, 0.5, 0.5))
        mesh.set_color(color)
        
        # Add surface type as metadata
        mesh.add_scalar_quantity("surface_type_id", 
                                np.full(len(points_flat), class_label))
    
    print(f"\nVisualization ready! Added {n_surfaces} surface meshes.")
    print("Surface types and colors:")
    for class_id, name in CLASS_NAMES.items():
        if class_id in class_labels[:n_surfaces]:
            color = SURFACE_COLORS.get(class_id, (0.5, 0.5, 0.5))
            print(f"  {name}: RGB{color}")
    
    # Custom UI callback to show surface information
    def ui_callback():
        psim.TextUnformatted("Surface Dataset Visualization")
        psim.Separator()
        psim.TextUnformatted(f"Total surfaces loaded: {n_surfaces}")
        psim.Separator()
        psim.TextUnformatted("Surface Types:")
        
        # Show surface type counts for loaded surfaces
        loaded_labels = class_labels[:n_surfaces]
        unique_labels, counts = np.unique(loaded_labels, return_counts=True)
        
        for label, count in zip(unique_labels, counts):
            surface_name = CLASS_NAMES.get(label, f"unknown_{label}")
            color = SURFACE_COLORS.get(label, (0.5, 0.5, 0.5))
            psim.PushStyleColor(psim.ImGuiCol_Text, *color, 1.0)
            psim.TextUnformatted(f"  {surface_name}: {count}")
            psim.PopStyleColor()
    
    ps.set_user_callback(ui_callback)
    
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
    
    args = parser.parse_args()
    
    if not args.pkl_file.exists():
        print(f"Error: File {args.pkl_file} does not exist")
        return
    
    # Load dataset
    try:
        data = load_dataset(args.pkl_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Visualize surfaces
    try:
        visualize_surfaces(data, args.max_surfaces)
    except Exception as e:
        print(f"Error during visualization: {e}")
        return


if __name__ == "__main__":
    main() 