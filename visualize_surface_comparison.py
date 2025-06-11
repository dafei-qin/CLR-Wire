#!/usr/bin/env python3
"""
Script to visualize surface comparisons using polyscope.
Compares original surfaces from JSON files (logan_process output) with
new surfaces sampled from basic surfaces + rotation, translation, scaling.

For basic surfaces (plane, cylinder, sphere, cone, torus):
- Shows original surface using surface['points']
- Shows transformed surface using surface_sampler

For B-spline surfaces:
- Only shows original surface
"""

import argparse
import json
import math
import numpy as np
import os
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import polyscope as ps
import polyscope.imgui as psim

# Import the surface sampler
from surface_sampler import SurfaceSampler
from convert_surface_to_transformations import SurfaceTransformationConverter


class SurfaceComparisonVisualizer:
    def __init__(self):
        # Core components
        self.surface_sampler = SurfaceSampler()
        self.transformation_converter = SurfaceTransformationConverter()
        
        # Data storage
        self.current_json_files = []
        self.current_file_index = 0
        self.current_surfaces_data = []
        self.current_file_path = ""
        
        # Visualization settings
        self.surface_resolution = 32
        self.show_original = True
        self.show_transformed = True
        self.surface_transparency = 0.7  # Less transparent by default
        self.wireframe_mode = False
        self.overlay_mode = True  # Show surfaces at same position for overlay comparison
        
        # Surface type filtering
        self.show_all_types = True
        self.surface_type_filters = {
            "plane": True,
            "cylinder": True,
            "sphere": True,
            "cone": True,
            "torus": True,
            "bezier_surface": True,
            "bspline_surface": True
        }
        
        # Basic surface types that support transformation
        self.basic_surface_types = {
            "plane", "cylinder", "sphere", "cone", "torus"
        }
        
        # Colors for different surface types
        self.surface_colors = {
            "plane": [0.8, 0.2, 0.2],        # Red
            "cylinder": [0.2, 0.8, 0.2],     # Green  
            "cone": [0.2, 0.2, 0.8],         # Blue
            "sphere": [0.8, 0.8, 0.2],       # Yellow
            "torus": [0.8, 0.2, 0.8],        # Magenta
            "bezier_surface": [0.2, 0.8, 0.8], # Cyan
            "bspline_surface": [0.8, 0.5, 0.2], # Orange
            "unknown": [0.5, 0.5, 0.5],      # Gray
        }
        
        # Object tracking
        self.original_objects = []
        self.transformed_objects = []
    
    def should_show_surface_type(self, surface_type: str) -> bool:
        """Check if a surface type should be displayed based on current filters"""
        if self.show_all_types:
            return True
        return self.surface_type_filters.get(surface_type, False)
    
    def find_json_files(self, directory: str) -> List[str]:
        """Find all JSON files in the directory and subdirectories"""
        json_files = []
        directory_path = Path(directory)
        
        if directory_path.is_file() and directory_path.suffix == '.json':
            json_files.append(str(directory_path))
        elif directory_path.is_dir():
            # Search for JSON files recursively in all subdirectories
            json_pattern = os.path.join(directory, "**", "*.json")
            json_files = glob.glob(json_pattern, recursive=True)
        
        return sorted(json_files)
    
    def load_json_data(self, json_path: str) -> List[Dict[str, Any]]:
        """Load surface data from JSON file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            return []
    
    def create_surface_mesh(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create mesh vertices and faces from a grid of points"""
        if points is None or len(points) == 0:
            return None, None
            
        # Handle different point array shapes
        if len(points.shape) == 3:
            height, width = points.shape[:2]
            vertices = points.reshape(-1, 3)
        elif len(points.shape) == 2 and points.shape[1] == 3:
            # Try to reshape to square grid
            total_points = len(points)
            side_length = int(np.sqrt(total_points))
            if side_length * side_length == total_points:
                points = points.reshape(side_length, side_length, 3)
                height, width = side_length, side_length
                vertices = points.reshape(-1, 3)
            else:
                # Assume it's already a list of vertices
                vertices = points
                return vertices, None  # Cannot create faces without grid structure
        else:
            return None, None
        
        # Create faces by connecting adjacent points in the grid
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                # Current quad vertices (in flattened index)
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
                
                # Create two triangles for each quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        faces = np.array(faces)
        return vertices, faces
    
    def get_original_surface_points(self, surface_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract original surface points from JSON data"""
        points = surface_data.get("points")
        if points is not None:
            return np.array(points)
        return None
    
    def calculate_transformation_on_the_fly(self, surface_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate r, t, s transformation on-the-fly"""
        surface_type = surface_data.get("type")
        
        if surface_type not in self.basic_surface_types:
            return surface_data
        
        # Use the transformation converter to calculate r, t, s
        try:
            transformed_data = self.transformation_converter.convert_surface(surface_data)
            return transformed_data
        except Exception as e:
            print(f"Error calculating transformation for {surface_type}: {e}")
            return surface_data
    
    def sample_transformed_surface(self, surface_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Sample points from transformed basic surface"""
        surface_type = surface_data.get("type")
        
        if surface_type not in self.basic_surface_types:
            return None
        
        # Calculate transformation on-the-fly
        transformed_data = self.calculate_transformation_on_the_fly(surface_data)
        
        # Set appropriate parameter ranges for each surface type
        num_u, num_v = self.surface_resolution, self.surface_resolution
        
        if surface_type == "plane":
            u_coords = np.linspace(-2, 2, num_u)
            v_coords = np.linspace(-2, 2, num_v)
        elif surface_type == "cylinder":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(-1, 1, num_v)
        elif surface_type == "sphere":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, np.pi, num_v)
        elif surface_type == "cone":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, 1, num_v)
        elif surface_type == "torus":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, 2*np.pi, num_v)
        else:
            return None
        
        try:
            # Sample the transformed surface
            sampled_points = self.surface_sampler.sample_surface(
                transformed_data, u_coords, v_coords
            )
            return sampled_points
        except Exception as e:
            print(f"Error sampling {surface_type}: {e}")
            return None
    
    def clear_all_objects(self):
        """Clear all polyscope objects"""
        try:
            ps.remove_all_structures()
        except:
            pass
        
        self.original_objects = []
        self.transformed_objects = []
    
    def visualize_current_file(self):
        """Visualize surfaces from the current JSON file"""
        if not self.current_surfaces_data:
            return
        
        self.clear_all_objects()
        
        print(f"\n=== Visualizing {self.current_file_path} ===")
        print(f"Found {len(self.current_surfaces_data)} surfaces")
        
        original_count = 0
        transformed_count = 0
        bspline_count = 0
        
        for i, surface_data in enumerate(self.current_surfaces_data):
            surface_type = surface_data.get("type", "unknown")
            surface_idx = surface_data.get("idx", i)
            
            # Skip if this surface type is filtered out
            if not self.should_show_surface_type(surface_type):
                continue
            
            print(f"Processing surface {i}: type={surface_type}, idx={surface_idx}")
            

            
            # Visualize original surface
            if self.show_original:
                original_points = self.get_original_surface_points(surface_data)
                if original_points is not None:
                    vertices, faces = self.create_surface_mesh(original_points)
                    if vertices is not None:
                        # Create mesh name
                        mesh_name = f"original_{surface_type}_{i}"
                        
                        if faces is not None:
                            ps_mesh = ps.register_surface_mesh(mesh_name, vertices, faces)
                        else:
                            # Register as point cloud if no faces
                            ps_mesh = ps.register_point_cloud(mesh_name, vertices)
                        
                        # Set color and properties
                        color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                        ps_mesh.set_color(color)
                        
                        if self.overlay_mode:
                            ps_mesh.set_transparency(0.5)  # Moderate transparency in overlay mode
                        else:
                            ps_mesh.set_transparency(self.surface_transparency)
                        
                        if self.wireframe_mode and hasattr(ps_mesh, 'set_edge_width'):
                            ps_mesh.set_edge_width(1.0)
                            ps_mesh.set_edge_color([0.2, 0.2, 0.2])
                        
                        self.original_objects.append(mesh_name)
                        original_count += 1
            
            # Visualize transformed surface (only for basic surfaces)
            if self.show_transformed and surface_type in self.basic_surface_types:
                transformed_points = self.sample_transformed_surface(surface_data)
                if transformed_points is not None:
                    vertices, faces = self.create_surface_mesh(transformed_points)
                    if vertices is not None and faces is not None:
                        # Apply spatial offset only if not in overlay mode
                        if self.overlay_mode:
                            vertices_final = vertices  # Same position as original
                        else:
                            offset = np.array([5.0, 0.0, 0.0])  # 5 units offset in X direction
                            vertices_final = vertices + offset
                        
                        # Create mesh name
                        mesh_name = f"transformed_{surface_type}_{i}"
                        ps_mesh = ps.register_surface_mesh(mesh_name, vertices_final, faces)
                        
                        # Set different visual properties for transformed surface
                        color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                        if self.overlay_mode:
                            # In overlay mode, use darker color with wireframe
                            darker_color = [c * 0.5 for c in color]  # Darker for contrast
                            ps_mesh.set_color(darker_color)
                            ps_mesh.set_edge_width(1.5)
                            ps_mesh.set_edge_color([0.0, 0.0, 0.0])  # Black edges
                            ps_mesh.set_transparency(0.3)  # Less transparent for visibility
                        else:
                            # In separated mode, use darker color
                            darker_color = [c * 0.7 for c in color]  # Make it darker
                            ps_mesh.set_color(darker_color)
                            ps_mesh.set_transparency(self.surface_transparency)
                        
                        if self.wireframe_mode:
                            ps_mesh.set_edge_width(1.0)
                            ps_mesh.set_edge_color([0.1, 0.1, 0.1])
                        
                        self.transformed_objects.append(mesh_name)
                        transformed_count += 1
            
            # Count B-spline surfaces
            if surface_type in ["bezier_surface", "bspline_surface"]:
                bspline_count += 1
        
        print(f"Visualization Summary:")
        print(f"  Original surfaces: {original_count}")
        print(f"  Transformed surfaces: {transformed_count}")
        print(f"  B-spline surfaces (original only): {bspline_count}")
        
        # Update camera to fit all objects
        try:
            ps.reset_camera_to_home_view()
            # Also try to set a reasonable camera position
            ps.look_at((0., 0., 3.), (0., 0., 0.))
        except Exception as e:
            print(f"Camera reset warning: {e}")
    
    def load_file(self, file_index: int):
        """Load a specific JSON file by index"""
        if 0 <= file_index < len(self.current_json_files):
            self.current_file_index = file_index
            self.current_file_path = self.current_json_files[file_index]
            self.current_surfaces_data = self.load_json_data(self.current_file_path)
            self.visualize_current_file()
    
    def create_gui_callback(self):
        """Create the polyscope GUI callback"""
        def gui_callback():
            # File navigation
            psim.Text("File Navigation")
            if len(self.current_json_files) > 1:
                changed, new_index = psim.SliderInt(
                    "File Index", 
                    self.current_file_index, 
                    0, 
                    len(self.current_json_files) - 1
                )
                if changed:
                    self.load_file(new_index)
                
                # Previous/Next buttons
                if psim.Button("Previous File") and self.current_file_index > 0:
                    self.load_file(self.current_file_index - 1)
                psim.SameLine()
                if psim.Button("Next File") and self.current_file_index < len(self.current_json_files) - 1:
                    self.load_file(self.current_file_index + 1)
            
            # Current file info
            psim.Text(f"Current file: {Path(self.current_file_path).name}")
            psim.Text(f"File {self.current_file_index + 1} of {len(self.current_json_files)}")
            
            psim.Separator()
            
            # Visualization controls
            psim.Text("Visualization Settings")
            
            # Show/hide toggles
            changed, self.show_original = psim.Checkbox("Show Original Surfaces", self.show_original)
            if changed:
                self.visualize_current_file()
            
            changed, self.show_transformed = psim.Checkbox("Show Transformed Surfaces", self.show_transformed)
            if changed:
                self.visualize_current_file()
            
            # Surface resolution
            changed, self.surface_resolution = psim.SliderInt(
                "Surface Resolution", 
                self.surface_resolution, 
                8, 
                64
            )
            if changed:
                self.visualize_current_file()
            
            # Transparency
            changed, self.surface_transparency = psim.SliderFloat(
                "Surface Transparency", 
                self.surface_transparency, 
                0.0, 
                1.0
            )
            if changed:
                self.visualize_current_file()
            
            # Wireframe mode
            changed, self.wireframe_mode = psim.Checkbox("Wireframe Mode", self.wireframe_mode)
            if changed:
                self.visualize_current_file()
            
            # Overlay mode
            changed, self.overlay_mode = psim.Checkbox("Overlay Mode (same position)", self.overlay_mode)
            if changed:
                self.visualize_current_file()
            
            psim.Separator()
            
            # Surface Type Filtering
            psim.Text("Surface Type Filters")
            
            # Show all types toggle
            changed, self.show_all_types = psim.Checkbox("Show All Types", self.show_all_types)
            if changed:
                self.visualize_current_file()
            
            # Individual surface type filters (only show when not showing all)
            if not self.show_all_types:
                psim.Text("Select types to show:")
                filter_changed = False
                
                # Basic surface types
                for surface_type in ["plane", "cylinder", "sphere", "cone", "torus"]:
                    changed, self.surface_type_filters[surface_type] = psim.Checkbox(
                        f"Show {surface_type}", 
                        self.surface_type_filters[surface_type]
                    )
                    if changed:
                        filter_changed = True
                
                # B-spline surface types
                for surface_type in ["bezier_surface", "bspline_surface"]:
                    changed, self.surface_type_filters[surface_type] = psim.Checkbox(
                        f"Show {surface_type}", 
                        self.surface_type_filters[surface_type]
                    )
                    if changed:
                        filter_changed = True
                
                # Quick filter buttons
                psim.Text("Quick filters:")
                if psim.Button("Basic Surfaces Only"):
                    for key in self.surface_type_filters:
                        self.surface_type_filters[key] = key in self.basic_surface_types
                    filter_changed = True
                
                psim.SameLine()
                if psim.Button("B-spline Surfaces Only"):
                    for key in self.surface_type_filters:
                        self.surface_type_filters[key] = key in ["bezier_surface", "bspline_surface"]
                    filter_changed = True
                
                psim.SameLine()
                if psim.Button("Cones Only"):
                    for key in self.surface_type_filters:
                        self.surface_type_filters[key] = (key == "cone")
                    filter_changed = True
                
                if filter_changed:
                    self.visualize_current_file()
            
            psim.Separator()
            
            # Statistics
            psim.Text("Surface Statistics")
            if self.current_surfaces_data:
                surface_type_counts = {}
                visible_surface_type_counts = {}
                basic_surface_count = 0
                bspline_surface_count = 0
                visible_count = 0
                
                for surface_data in self.current_surfaces_data:
                    surface_type = surface_data.get("type", "unknown")
                    surface_type_counts[surface_type] = surface_type_counts.get(surface_type, 0) + 1
                    
                    if surface_type in self.basic_surface_types:
                        basic_surface_count += 1
                    elif surface_type in ["bezier_surface", "bspline_surface"]:
                        bspline_surface_count += 1
                    
                    # Count visible surfaces
                    if self.should_show_surface_type(surface_type):
                        visible_surface_type_counts[surface_type] = visible_surface_type_counts.get(surface_type, 0) + 1
                        visible_count += 1
                
                psim.Text(f"Total surfaces: {len(self.current_surfaces_data)}")
                psim.Text(f"Visible surfaces: {visible_count}")
                psim.Text(f"Basic surfaces: {basic_surface_count}")
                psim.Text(f"B-spline surfaces: {bspline_surface_count}")
                
                psim.Text("Surface type breakdown (total/visible):")
                for surface_type, count in surface_type_counts.items():
                    visible_count_for_type = visible_surface_type_counts.get(surface_type, 0)
                    color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                    status = "✓" if self.should_show_surface_type(surface_type) else "✗"
                    psim.Text(f"  {status} {surface_type}: {count}/{visible_count_for_type}")
                    psim.SameLine()
                    psim.Text(f"RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
            
            psim.Separator()
            
            # Help text
            psim.Text("Help")
            psim.Text("Original surfaces: from surface['points']")
            psim.Text("Transformed surfaces: basic surfaces + r,t,s")
            psim.Text("B-spline surfaces: original only")
            psim.Text("Use surface type filters to show specific types")
            psim.Text("'Cones Only' button useful for debugging cones")
            psim.Text("Overlay mode: surfaces at same position for comparison")
            psim.Text("Non-overlay: transformed surfaces offset by +5 in X")
            
        return gui_callback
    
    def run(self, directory: str):
        """Main execution function"""
        # Find JSON files
        self.current_json_files = self.find_json_files(directory)
        
        if not self.current_json_files:
            print(f"No JSON files found in {directory}")
            return
        
        print(f"Found {len(self.current_json_files)} JSON files")
        for i, json_file in enumerate(self.current_json_files[:5]):
            print(f"  {i}: {Path(json_file).name}")
        if len(self.current_json_files) > 5:
            print(f"  ... and {len(self.current_json_files) - 5} more")
        
        # Initialize polyscope
        ps.init()
        ps.set_user_callback(self.create_gui_callback())
        
        # Load first file
        if self.current_json_files:
            self.load_file(0)
        
        print(f"\n🎛️ Interactive Controls:")
        print(f"   - Use file navigation to switch between JSON files")
        print(f"   - Toggle original vs transformed surface display")
        print(f"   - Adjust surface resolution and transparency")
        print(f"   - Use surface type filters to focus on specific types")
        print(f"   - Toggle overlay mode for better comparison")
        print(f"   - Original surfaces use surface['points']")
        print(f"   - Transformed surfaces use surface_sampler with r,t,s")
        print(f"   - B-spline surfaces show original only")
        print()
        
        # Show the visualization
        ps.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize surface comparisons with polyscope")
    parser.add_argument("directory", help="Directory containing JSON files from logan_process")
    
    args = parser.parse_args()
    
    visualizer = SurfaceComparisonVisualizer()
    visualizer.run(args.directory)


if __name__ == "__main__":
    main() 