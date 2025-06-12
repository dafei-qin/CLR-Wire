#!/usr/bin/env python3
"""
Interactive surface examination tool to identify out-of-distribution (OOD) surfaces.
Uses polyscope UI to filter surfaces by type and transformation parameters,
then randomly visualizes matching samples.

Usage:
$ python examine_surface_ood.py data.pkl

Controls:
- Surface Type: Select which surface types to include
- Scaling Range: Filter by X, Y, Z scaling values
- Rotation Range: Filter by X, Y, Z rotation values  
- Translation Range: Filter by X, Y, Z translation values
- Random Sample: Click to visualize a random surface matching filters
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import polyscope as ps
from typing import Dict, Tuple, List, Optional
import random

# Surface type mapping (from aggregate_surface_dataset.py)
CLASS_MAPPING = {
    0: "plane",
    1: "cylinder", 
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "bspline_surface",
    6: "bezier_surface",
}

class SurfaceOODExaminer:
    def __init__(self, data: Dict):
        self.data = data
        self.filtered_indices = np.arange(len(data['class_label']))
        self.current_surface_idx = None
        
        # Initialize filter ranges from data statistics
        self.init_filter_ranges()
        
        # UI state
        self.selected_surface_types = set(range(7))  # All types initially
        self.scaling_ranges = [(0.0, 2.0)] * 3
        self.rotation_ranges = [(-2.0, 2.0)] * 3  
        self.translation_ranges = [(-10.0, 10.0)] * 3
        
    def init_filter_ranges(self):
        """Initialize filter ranges based on data statistics."""
        self.scaling_stats = []
        self.rotation_stats = []
        self.translation_stats = []
        
        for i in range(3):  # X, Y, Z axes
            # Scaling stats
            scale_vals = self.data['scaling'][:, i]
            self.scaling_stats.append({
                'min': float(np.min(scale_vals)),
                'max': float(np.max(scale_vals)),
                'mean': float(np.mean(scale_vals)),
                'std': float(np.std(scale_vals))
            })
            
            # Rotation stats  
            rot_vals = self.data['rotation'][:, i]
            self.rotation_stats.append({
                'min': float(np.min(rot_vals)),
                'max': float(np.max(rot_vals)),
                'mean': float(np.mean(rot_vals)),
                'std': float(np.std(rot_vals))
            })
            
            # Translation stats
            trans_vals = self.data['translation'][:, i]
            self.translation_stats.append({
                'min': float(np.min(trans_vals)),
                'max': float(np.max(trans_vals)),
                'mean': float(np.mean(trans_vals)),
                'std': float(np.std(trans_vals))
            })
    
    def apply_filters(self):
        """Apply all active filters to get matching surface indices."""
        mask = np.ones(len(self.data['class_label']), dtype=bool)
        
        # Surface type filter
        if len(self.selected_surface_types) < 7:  # Not all types selected
            type_mask = np.isin(self.data['class_label'], list(self.selected_surface_types))
            mask &= type_mask
        
        # Scaling filters
        for i, (min_val, max_val) in enumerate(self.scaling_ranges):
            scale_mask = (self.data['scaling'][:, i] >= min_val) & (self.data['scaling'][:, i] <= max_val)
            mask &= scale_mask
            
        # Rotation filters
        for i, (min_val, max_val) in enumerate(self.rotation_ranges):
            rot_mask = (self.data['rotation'][:, i] >= min_val) & (self.data['rotation'][:, i] <= max_val)
            mask &= rot_mask
            
        # Translation filters
        for i, (min_val, max_val) in enumerate(self.translation_ranges):
            trans_mask = (self.data['translation'][:, i] >= min_val) & (self.data['translation'][:, i] <= max_val)
            mask &= trans_mask
            
        self.filtered_indices = np.where(mask)[0]
        print(f"Filters applied: {len(self.filtered_indices)} surfaces match criteria")
    
    def visualize_random_surface(self):
        """Randomly select and visualize a surface from filtered results."""
        if len(self.filtered_indices) == 0:
            print("No surfaces match the current filters!")
            return
            
        # Randomly select a surface
        random_idx = random.choice(self.filtered_indices)
        self.current_surface_idx = random_idx
        
        # Get surface data
        points = self.data['points'][random_idx]  # Shape: (32, 32, 3)
        surface_type = CLASS_MAPPING[self.data['class_label'][random_idx]]
        uid = self.data['uid'][random_idx]
        index = self.data['index'][random_idx]
        
        # Get transformation parameters
        scaling = self.data['scaling'][random_idx]
        rotation = self.data['rotation'][random_idx]
        translation = self.data['translation'][random_idx]
        
        print(f"\nDisplaying surface {random_idx}:")
        print(f"  Type: {surface_type}")
        print(f"  UID: {uid}")
        print(f"  Index: {index}")
        print(f"  Scaling: [{scaling[0]:.3f}, {scaling[1]:.3f}, {scaling[2]:.3f}]")
        print(f"  Rotation: [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}]")
        print(f"  Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        
        # Clear previous surface
        if ps.has_surface_mesh("current_surface"):
            ps.remove_surface_mesh("current_surface")
            
        # Create mesh from grid points
        vertices, faces = self.grid_to_mesh(points)
        
        # Add to polyscope
        mesh = ps.register_surface_mesh("current_surface", vertices, faces)
        mesh.set_color([0.8, 0.2, 0.2])  # Red color
        
        # Update camera to focus on surface
        ps.look_at_dir([0, 0, 1], [0, 1, 0])
    
    def grid_to_mesh(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 32x32x3 grid points to vertices and faces for mesh visualization."""
        H, W, _ = points.shape
        vertices = points.reshape(-1, 3)
        
        # Create triangular faces from grid
        faces = []
        for i in range(H - 1):
            for j in range(W - 1):
                # Two triangles per quad
                v0 = i * W + j
                v1 = i * W + (j + 1)
                v2 = (i + 1) * W + j
                v3 = (i + 1) * W + (j + 1)
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
                
        return vertices, np.array(faces)
    
    def create_ui(self):
        """Create polyscope UI controls."""
        
        # Surface type selection
        def surface_type_callback():
            # Create checkboxes for each surface type
            for class_id, name in CLASS_MAPPING.items():
                if ps.ImGuiCheckbox(f"{name}##type_{class_id}", class_id in self.selected_surface_types):
                    if class_id in self.selected_surface_types:
                        self.selected_surface_types.remove(class_id)
                    else:
                        self.selected_surface_types.add(class_id)
        
        # Scaling range controls
        def scaling_range_callback():
            ps.ImGuiText("Scaling Ranges")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                stats = self.scaling_stats[i]
                min_val, max_val = self.scaling_ranges[i]
                
                ps.ImGuiText(f"Scale {axis}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
                changed, new_range = ps.ImGuiSliderFloat2(
                    f"##scale_{axis}", 
                    min_val, max_val,
                    v_min=stats['min'] - 0.1,
                    v_max=stats['max'] + 0.1
                )
                if changed:
                    self.scaling_ranges[i] = new_range
        
        # Rotation range controls  
        def rotation_range_callback():
            ps.ImGuiText("Rotation Ranges")
            for i, axis in enumerate(['X', 'Y', 'Z']):
                stats = self.rotation_stats[i]
                min_val, max_val = self.rotation_ranges[i]
                
                ps.ImGuiText(f"Rotation {axis}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
                changed, new_range = ps.ImGuiSliderFloat2(
                    f"##rot_{axis}",
                    min_val, max_val, 
                    v_min=stats['min'] - 0.5,
                    v_max=stats['max'] + 0.5
                )
                if changed:
                    self.rotation_ranges[i] = new_range
        
        # Translation range controls
        def translation_range_callback():
            ps.ImGuiText("Translation Ranges") 
            for i, axis in enumerate(['X', 'Y', 'Z']):
                stats = self.translation_stats[i]
                min_val, max_val = self.translation_ranges[i]
                
                ps.ImGuiText(f"Translation {axis}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
                changed, new_range = ps.ImGuiSliderFloat2(
                    f"##trans_{axis}",
                    min_val, max_val,
                    v_min=stats['min'] - 2.0,
                    v_max=stats['max'] + 2.0
                )
                if changed:
                    self.translation_ranges[i] = new_range
        
        # Control buttons
        def control_callback():
            if ps.ImGuiButton("Apply Filters"):
                self.apply_filters()
                
            ps.ImGuiSameLine()
            if ps.ImGuiButton("Random Sample"):
                self.visualize_random_surface()
                
            ps.ImGuiSameLine()
            if ps.ImGuiButton("Reset Filters"):
                self.reset_filters()
                
            # Show current filter status
            ps.ImGuiText(f"Matching surfaces: {len(self.filtered_indices)}")
            if self.current_surface_idx is not None:
                surface_type = CLASS_MAPPING[self.data['class_label'][self.current_surface_idx]]
                ps.ImGuiText(f"Current surface: {self.current_surface_idx} ({surface_type})")
        
        # Register UI callbacks
        ps.set_user_callback(lambda: [
            ps.ImGuiSetNextWindowPos((10, 10)),
            ps.ImGuiBegin("Surface Type Filter"),
            surface_type_callback(),
            ps.ImGuiEnd(),
            
            ps.ImGuiSetNextWindowPos((10, 200)),
            ps.ImGuiBegin("Scaling Filter"),
            scaling_range_callback(),
            ps.ImGuiEnd(),
            
            ps.ImGuiSetNextWindowPos((10, 400)),
            ps.ImGuiBegin("Rotation Filter"),
            rotation_range_callback(),
            ps.ImGuiEnd(),
            
            ps.ImGuiSetNextWindowPos((10, 600)),
            ps.ImGuiBegin("Translation Filter"),
            translation_range_callback(),
            ps.ImGuiEnd(),
            
            ps.ImGuiSetNextWindowPos((350, 10)),
            ps.ImGuiBegin("Controls"),
            control_callback(),
            ps.ImGuiEnd(),
        ])
    
    def reset_filters(self):
        """Reset all filters to default values."""
        self.selected_surface_types = set(range(7))
        
        # Reset to statistical ranges (mean ± 2*std)
        for i in range(3):
            scale_stats = self.scaling_stats[i]
            self.scaling_ranges[i] = (
                max(scale_stats['min'], scale_stats['mean'] - 2*scale_stats['std']),
                min(scale_stats['max'], scale_stats['mean'] + 2*scale_stats['std'])
            )
            
            rot_stats = self.rotation_stats[i]
            self.rotation_ranges[i] = (
                max(rot_stats['min'], rot_stats['mean'] - 2*rot_stats['std']),
                min(rot_stats['max'], rot_stats['mean'] + 2*rot_stats['std'])
            )
            
            trans_stats = self.translation_stats[i]
            self.translation_ranges[i] = (
                max(trans_stats['min'], trans_stats['mean'] - 2*trans_stats['std']),
                min(trans_stats['max'], trans_stats['mean'] + 2*trans_stats['std'])
            )
        
        self.apply_filters()
        print("Filters reset to default ranges")


def load_dataset(pkl_path: Path) -> Dict:
    """Load the pickled dataset."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to examine surfaces with potentially OOD attributes")
    parser.add_argument("dataset", type=Path,
                       help="Path to the pickled dataset file (.pkl)")
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"Error: Dataset file {args.dataset} does not exist!")
        return
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    
    # Print dataset info
    print(f"Dataset contains {len(data['class_label'])} surfaces")
    unique_labels, counts = np.unique(data["class_label"], return_counts=True)
    print("Surface type distribution:")
    for label, count in zip(unique_labels, counts):
        surface_name = CLASS_MAPPING.get(label, f"unknown_{label}")
        print(f"  {surface_name}: {count} samples")
    
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")
    
    # Create examiner
    examiner = SurfaceOODExaminer(data)
    examiner.create_ui()
    examiner.apply_filters()  # Apply initial filters
    
    print("\nInteractive examination tool started!")
    print("Use the UI controls to filter surfaces and click 'Random Sample' to visualize.")
    print("Look for surfaces with extreme transformation parameters that might be OOD.")
    
    # Run polyscope
    ps.show()


if __name__ == "__main__":
    main() 