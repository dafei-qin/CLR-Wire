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
import colorsys
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
        self.use_aabb_filtering = True  # Use AABB for constrained surface sampling
        self.show_aabb_wireframes = False  # Show AABB bounds as wireframe boxes
        self.show_coordinate_axes = True  # Show X, Y, Z coordinate axes
        self.show_curves = True  # Show curves as wireframes
        self.curve_transparency = 0.8  # Transparency for curve wireframes
        self.use_uv_color_gradient = False  # Use UV-based color gradient shading
        self.show_surface_centers = True  # Show surface centers as points
        self.show_surface_normals = True  # Show surface normals as vectors
        self.normal_length = 1.0  # Length of normal vectors
        
        # Surface type filtering
        self.show_all_types = True
        self.surface_type_filters = {
            "plane": True,
            "cylinder": True,
            "sphere": True,
            "cone": True,
            "torus": True,
            "bezier_surface": True,
            "bspline_surface": True,
            "line": True,
            "circle": True,
            "ellipse": True,
            "bspline_curve": True,
            "bezier_curve": True,
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
            "line": [0.9, 0.9, 0.9],         # Light gray
            "circle": [0.7, 0.3, 0.7],       # Purple
            "ellipse": [0.3, 0.7, 0.3],      # Light green
            "bspline_curve": [0.9, 0.5, 0.1], # Dark orange
            "bezier_curve": [0.1, 0.5, 0.9],  # Dark blue
            "unknown": [0.5, 0.5, 0.5],      # Gray
        }
        
        # Object tracking
        self.original_objects = []
        self.transformed_objects = []
        self.wireframe_objects = []
        self.axes_objects = []
        self.curve_objects = []
        self.center_objects = []  # Surface centers
        self.normal_objects = []  # Surface normals
    
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
    
    def generate_uv_colors(self, vertices: np.ndarray, height: int, width: int) -> Optional[np.ndarray]:
        """
        Generate UV-based color gradient for surface vertices.
        
        Args:
            vertices: Surface vertices (n_vertices, 3)
            height: Number of rows in parameter grid
            width: Number of columns in parameter grid
            
        Returns:
            colors: RGB colors for each vertex (n_vertices, 3)
        """
        if vertices is None or len(vertices) == 0:
            return None
        
        total_vertices = len(vertices)
        expected_vertices = height * width
        
        if total_vertices != expected_vertices:
            print(f"Warning: UV color mapping mismatch - expected {expected_vertices} vertices, got {total_vertices}")
            # Fall back to simple gradient based on vertex index
            colors = np.zeros((total_vertices, 3))
            for i in range(total_vertices):
                t = i / max(1, total_vertices - 1)  # Normalize to [0, 1]
                # Create a nice color gradient: blue -> cyan -> green -> yellow -> red
                if t < 0.25:
                    # Blue to cyan
                    s = t / 0.25
                    colors[i] = [0, s, 1]
                elif t < 0.5:
                    # Cyan to green
                    s = (t - 0.25) / 0.25
                    colors[i] = [0, 1, 1-s]
                elif t < 0.75:
                    # Green to yellow
                    s = (t - 0.5) / 0.25
                    colors[i] = [s, 1, 0]
                else:
                    # Yellow to red
                    s = (t - 0.75) / 0.25
                    colors[i] = [1, 1-s, 0]
            return colors
        
        # Generate UV-based colors
        colors = np.zeros((total_vertices, 3))
        
        for i in range(height):
            for j in range(width):
                vertex_idx = i * width + j
                
                # Normalize UV coordinates to [0, 1]
                u = j / max(1, width - 1)
                v = i / max(1, height - 1)
                
                # Create color based on UV coordinates
                # Option 1: U maps to red-green, V maps to blue
                # colors[vertex_idx] = [u, v, 0.5]
                
                # Option 2: More visually appealing UV color mapping
                # U direction: varies hue, V direction: varies brightness
                hue = u  # U coordinate controls hue (0=red, 0.33=green, 0.67=blue)
                saturation = 0.8  # High saturation for vivid colors
                value = 0.3 + 0.7 * v  # V coordinate controls brightness (0.3 to 1.0)
                
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                colors[vertex_idx] = list(rgb)
        
        return colors
    
    def visualize_surface_center_and_normal(self, surface_data: Dict[str, Any], surface_index: int):
        """
        Visualize surface center and normal vector.
        
        Args:
            surface_data: Surface data containing location and direction
            surface_index: Index of the surface for naming
        """
        surface_type = surface_data.get("type", "unknown")
        
        # Skip if this surface type is filtered out
        if not self.should_show_surface_type(surface_type):
            return
        
        # Extract location (center) and direction (normal)
        location = surface_data.get("location")
        direction = surface_data.get("direction")
        
        if location is None or direction is None or len(location) == 0 or len(direction) == 0:
            print(f"  Skipping center/normal for {surface_type} {surface_index}: missing location or direction")
            return
        
        # Handle different formats
        if isinstance(location[0], list):
            center = np.array(location[0])  # location is [[x, y, z]]
        else:
            center = np.array(location)  # location is [x, y, z]
        
        if isinstance(direction[0], list):
            normal = np.array(direction[0])  # direction is [[nx, ny, nz]]
        else:
            normal = np.array(direction)  # direction is [nx, ny, nz]
        
        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length > 1e-10:
            normal = normal / normal_length
        else:
            print(f"  Warning: Zero-length normal for {surface_type} {surface_index}")
            return
        
        try:
            # Visualize center point
            if self.show_surface_centers:
                center_name = f"center_{surface_type}_{surface_index}"
                ps_center = ps.register_point_cloud(center_name, center.reshape(1, 3))
                
                # Set color based on surface type
                color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                ps_center.set_color(color)
                ps_center.set_radius(0.02)  # Make centers visible
                
                self.center_objects.append(center_name)
                print(f"  Visualized center for {surface_type} {surface_index} at {center}")
            
            # Visualize normal vector
            if self.show_surface_normals:
                # Create normal vector endpoint
                normal_end = center + normal * self.normal_length
                
                # Create points and edges for the normal vector
                normal_points = np.array([center, normal_end])
                normal_edges = np.array([[0, 1]])
                
                normal_name = f"normal_{surface_type}_{surface_index}"
                ps_normal = ps.register_curve_network(normal_name, normal_points, normal_edges)
                
                # Set color and properties
                # Use a darker version of the surface color for normals
                color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                darker_color = [c * 0.6 for c in color]  # Darker for contrast
                ps_normal.set_color(darker_color)
                ps_normal.set_radius(0.008)  # Thicker than wireframes but thinner than axes
                
                self.normal_objects.append(normal_name)
                print(f"  Visualized normal for {surface_type} {surface_index}: direction {normal}")
                
        except Exception as e:
            print(f"  Error visualizing center/normal for {surface_type} {surface_index}: {e}")
    
    def create_surface_mesh(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Create mesh vertices and faces from a grid of points"""
        if points is None or len(points) == 0:
            return None, None, None
            
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
                return vertices, None, None  # Cannot create faces without grid structure
        else:
            return None, None, None
        
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
        return vertices, faces, (height, width)
    
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
    
    def sample_transformed_surface(self, surface_data: Dict[str, Any], surface_index: int = 0) -> Optional[np.ndarray]:
        """Sample points from transformed basic surface"""
        surface_type = surface_data.get("type")
        
        if surface_type not in self.basic_surface_types:
            return None
        
        # Use pre-calculated transformation data if available
        transformation = surface_data.get("converted_transformation")
        if transformation is None:
            print(f"Warning: No converted_transformation found for {surface_type}, calculating on-the-fly")
            # Calculate transformation on-the-fly and get the converted surface data
            transformed_data = self.calculate_transformation_on_the_fly(surface_data)
        else:
            # Use the surface data directly since it already contains the transformation
            transformed_data = surface_data
        
        # Set appropriate parameter ranges for each surface type
        num_u, num_v = self.surface_resolution, self.surface_resolution
        
        # Always use default parameter ranges for full surface sampling
        if surface_type == "plane":
            u_coords = np.linspace(-1, 1, num_u)
            v_coords = np.linspace(-1, 1, num_v)
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
        
        print(f"Sampling full {surface_type} surface with default parameter ranges")
        
        try:
            # Sample the full transformed surface
            print(f"  About to sample {surface_type} with shape u_coords: {u_coords.shape}, v_coords: {v_coords.shape}")
            print(f"  transformed_data keys: {list(transformed_data.keys())}")
            if 'converted_transformation' in transformed_data:
                print(f"  Has converted_transformation with keys: {list(transformed_data['converted_transformation'].keys())}")
            
            sampled_points = self.surface_sampler.sample_surface(
                transformed_data, u_coords, v_coords
            )
            print(f"  Successfully sampled {surface_type}, got shape: {sampled_points.shape}")
            
            # Apply 3D AABB filtering if enabled and bounds are available
            if (self.use_aabb_filtering and sampled_points is not None and 
                len(sampled_points) > 0):
                
                # Get pre-calculated transformation data
                transformation = transformed_data.get("converted_transformation")
                if transformation is not None:
                    aabb_min = transformation.get("aabb_min")
                    aabb_max = transformation.get("aabb_max")
                    translation = transformation.get("translation")
                    rotation = transformation.get("rotation")
                    scaling = transformation.get("scaling")
                    print('scaling: ', scaling, 'rotation: ', rotation, 'translation: ', translation)
                    if (aabb_min is not None and aabb_max is not None and
                        translation is not None and rotation is not None and scaling is not None):
                        
                        # Handle different shapes of sampled_points
                        original_shape = sampled_points.shape
                        if len(original_shape) == 3 and original_shape[2] == 3:
                            # Grid format (height, width, 3) - flatten for processing
                            points_flat = sampled_points.reshape(-1, 3)
                        elif len(original_shape) == 2 and original_shape[1] == 3:
                            # Already flat format (n_points, 3)
                            points_flat = sampled_points
                        else:
                            print(f"AABB filtering for {surface_type}: unexpected points shape {original_shape}")
                            return sampled_points
                        
                        # Convert sampled points to standard space
                        standard_points = self.transformation_converter.transform_points_to_standard_space(
                            points_flat, translation, rotation, scaling
                        )
                        
                        # Debug: Print point ranges in both world and standard space
                        world_min = np.min(points_flat, axis=0)
                        world_max = np.max(points_flat, axis=0)
                        standard_min = np.min(standard_points, axis=0)
                        standard_max = np.max(standard_points, axis=0)
                        
                        print(f"  Debug {surface_type} surface {surface_index} points:")
                        print(f"    World space range: min={world_min}, max={world_max}")
                        print(f"    Standard space range: min={standard_min}, max={standard_max}")
                        print(f"    AABB bounds: min={aabb_min}, max={aabb_max}")
                        print(f"    Transformation: t={translation}, s={scaling}")
                        
                        # Additional debugging for cone surfaces
                        if surface_type == "cone":
                            print(f"    Cone specific debug:")
                            cone_location = surface_data.get("location", [[0, 0, 0]])[0]
                            cone_direction = surface_data.get("direction", [[0, 0, 1]])[0]
                            cone_scalars = surface_data.get("scalar", [0, 1])
                            print(f"      Original cone location: {cone_location}")
                            print(f"      Original cone direction: {cone_direction}")
                            print(f"      Original cone scalars (semi_angle, radius): {cone_scalars}")
                            
                            # Check if this cone's AABB seems reasonable
                            aabb_size = np.array(aabb_max) - np.array(aabb_min)
                            print(f"      AABB size: {aabb_size}")
                            print(f"      Standard space extent: {standard_max - standard_min}")
                        
                        # Apply AABB filtering in standard space
                        aabb_min_np = np.array(aabb_min)
                        aabb_max_np = np.array(aabb_max)
                        
                        # Add small tolerance to avoid numerical precision issues
                        tolerance = 1e-6
                        aabb_min_with_tol = aabb_min_np - tolerance
                        aabb_max_with_tol = aabb_max_np + tolerance
                        
                        # Find points within AABB (with tolerance)
                        mask = np.all((standard_points >= aabb_min_with_tol) & (standard_points <= aabb_max_with_tol), axis=1)
                        
                        # Debug: Show how many points are within each axis bound
                        x_in_bounds = np.sum((standard_points[:, 0] >= aabb_min_with_tol[0]) & (standard_points[:, 0] <= aabb_max_with_tol[0]))
                        y_in_bounds = np.sum((standard_points[:, 1] >= aabb_min_with_tol[1]) & (standard_points[:, 1] <= aabb_max_with_tol[1]))
                        z_in_bounds = np.sum((standard_points[:, 2] >= aabb_min_with_tol[2]) & (standard_points[:, 2] <= aabb_max_with_tol[2]))
                        print(f"    Points within bounds: X: {x_in_bounds}/{len(points_flat)}, Y: {y_in_bounds}/{len(points_flat)}, Z: {z_in_bounds}/{len(points_flat)}")
                        print(f"    Points within all bounds: {np.sum(mask)}/{len(points_flat)}")
                        
                        if np.any(mask):
                            # Filter the sampled points (in world space)
                            filtered_points_flat = points_flat[mask]
                            print(f"AABB filtering for {surface_type}: kept {len(filtered_points_flat)}/{len(points_flat)} points")
                            
                            # Check if we have enough points for a meaningful visualization
                            if len(filtered_points_flat) < 10:  # Very few points, might be too aggressive
                                print(f"  Warning: Very few points after filtering ({len(filtered_points_flat)}), returning full surface instead")
                                return sampled_points
                            
                            # Return filtered points in original format if possible
                            # For visualization, we typically need flat format anyway
                            return filtered_points_flat
                        else:
                            print(f"AABB filtering for {surface_type}: no points within bounds, returning full surface")
                            print(f"  AABB bounds: min={aabb_min}, max={aabb_max}")
                            print(f"  Standard points range: min={standard_min}, max={standard_max}")
                            
                            # For cones, if no points are within bounds, there might be an issue with AABB calculation
                            if surface_type == "cone":
                                print(f"  Warning: Cone AABB filtering failed completely, check AABB calculation")
                                # For cones that fail AABB completely, return some points for debugging
                                # Return a subset of points rather than nothing
                                subset_size = min(100, len(points_flat))
                                print(f"  Returning subset of {subset_size} points for debugging")
                                return points_flat[:subset_size]
                            
                            return sampled_points
                    else:
                        print(f"AABB filtering for {surface_type}: incomplete transformation data")
                        print(f"  aabb_min: {aabb_min}, aabb_max: {aabb_max}")
                        print(f"  translation: {translation is not None}, rotation: {rotation is not None}, scaling: {scaling is not None}")
                        return sampled_points
                else:
                    print(f"AABB filtering for {surface_type}: no converted_transformation found, using full surface")
                    return sampled_points
            
            return sampled_points
        except Exception as e:
            print(f"Error sampling {surface_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_aabb_wireframe(self, aabb_min, aabb_max, transformation, surface_type, surface_index):
        """
        Create wireframe box visualization for AABB bounds.
        
        Args:
            aabb_min: Minimum bounds in standard space
            aabb_max: Maximum bounds in standard space  
            transformation: Transformation parameters
            surface_type: Type of surface
            surface_index: Index of surface for naming
        """
        if aabb_min is None or aabb_max is None:
            return
        
        # Create 8 corners of the AABB box in standard space
        corners_standard = np.array([
            [aabb_min[0], aabb_min[1], aabb_min[2]],  # 0: min corner
            [aabb_max[0], aabb_min[1], aabb_min[2]],  # 1: +x
            [aabb_min[0], aabb_max[1], aabb_min[2]],  # 2: +y  
            [aabb_max[0], aabb_max[1], aabb_min[2]],  # 3: +x+y
            [aabb_min[0], aabb_min[1], aabb_max[2]],  # 4: +z
            [aabb_max[0], aabb_min[1], aabb_max[2]],  # 5: +x+z
            [aabb_min[0], aabb_max[1], aabb_max[2]],  # 6: +y+z
            [aabb_max[0], aabb_max[1], aabb_max[2]],  # 7: max corner
        ])
        
        # Transform corners to world space (inverse of the transformation used for points)
        # This is the forward transformation: Scale -> Rotate -> Translate
        translation = np.array(transformation["translation"])
        rotation = np.array(transformation["rotation"])
        scaling = np.array(transformation["scaling"])
        
        # Apply forward transformation
        corners_scaled = corners_standard * scaling  # Apply scaling
        corners_rotated = corners_scaled @ rotation.T  # Apply rotation  
        corners_world = corners_rotated + translation  # Apply translation
        
        # Define edges of the wireframe box (connecting corner indices)
        edges = np.array([
            [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom face
            [4, 5], [5, 7], [7, 6], [6, 4],  # Top face  
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ])
        
        # Register wireframe in polyscope
        wireframe_name = f"aabb_wireframe_{surface_type}_{surface_index}"
        try:
            ps_wireframe = ps.register_curve_network(wireframe_name, corners_world, edges)
            ps_wireframe.set_color([1.0, 1.0, 1.0])  # White wireframes
            ps_wireframe.set_radius(0.002)  # Thin lines
            self.wireframe_objects.append(wireframe_name)
            print(f"  Created AABB wireframe for {surface_type} {surface_index}")
        except Exception as e:
            print(f"  Error creating wireframe for {surface_type} {surface_index}: {e}")
    
    def create_coordinate_axes(self):
        """
        Create coordinate axes visualization with length 1.
        X-axis: Red, Y-axis: Green, Z-axis: Blue
        """
        if not self.show_coordinate_axes:
            return
        
        try:
            # Origin point
            origin = np.array([0.0, 0.0, 0.0])
            
            # X-axis (Red)
            x_end = np.array([1.0, 0.0, 0.0])
            x_points = np.array([origin, x_end])
            x_edges = np.array([[0, 1]])
            x_axis = ps.register_curve_network("x_axis", x_points, x_edges)
            x_axis.set_color([1.0, 0.0, 0.0])  # Red
            x_axis.set_radius(0.005)  # Slightly thicker than wireframes
            self.axes_objects.append("x_axis")
            
            # Y-axis (Green)  
            y_end = np.array([0.0, 1.0, 0.0])
            y_points = np.array([origin, y_end])
            y_edges = np.array([[0, 1]])
            y_axis = ps.register_curve_network("y_axis", y_points, y_edges)
            y_axis.set_color([0.0, 1.0, 0.0])  # Green
            y_axis.set_radius(0.005)
            self.axes_objects.append("y_axis")
            
            # Z-axis (Blue)
            z_end = np.array([0.0, 0.0, 1.0])
            z_points = np.array([origin, z_end])
            z_edges = np.array([[0, 1]])
            z_axis = ps.register_curve_network("z_axis", z_points, z_edges)
            z_axis.set_color([0.0, 0.0, 1.0])  # Blue
            z_axis.set_radius(0.005)
            self.axes_objects.append("z_axis")
            
            print("  Created coordinate axes (X=Red, Y=Green, Z=Blue)")
            
        except Exception as e:
            print(f"  Error creating coordinate axes: {e}")
    
    def get_curve_points(self, curve_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract curve points from JSON data"""
        points = curve_data.get("points")
        if points is not None:
            points_array = np.array(points)
            # Handle different curve point formats
            if len(points_array.shape) == 3 and points_array.shape[2] == 3:
                # Grid format - flatten to line
                return points_array.reshape(-1, 3)
            elif len(points_array.shape) == 2 and points_array.shape[1] == 3:
                # Already in correct format
                return points_array
            elif len(points_array.shape) == 2 and points_array.shape[0] == 3:
                # Transposed format
                return points_array.T
        return None
    
    def is_curve_type(self, geometry_type: str) -> bool:
        """Check if the geometry type is a curve"""
        curve_types = {"line", "circle", "ellipse", "bspline_curve", "bezier_curve"}
        return geometry_type in curve_types
    
    def visualize_curve(self, curve_data: Dict[str, Any], curve_index: int):
        """Visualize a single curve as wireframe"""
        curve_type = curve_data.get("type", "unknown")
        
        if not self.show_curves or not self.should_show_surface_type(curve_type):
            return
        
        # Get curve points
        curve_points = self.get_curve_points(curve_data)
        if curve_points is None or len(curve_points) < 2:
            print(f"  Skipping {curve_type} {curve_index}: insufficient points")
            return
        
        try:
            # Create edges connecting consecutive points
            num_points = len(curve_points)
            edges = []
            
            # For closed curves (circle, ellipse), connect last point back to first
            is_closed_curve = curve_type in {"circle", "ellipse"}
            
            for i in range(num_points - 1):
                edges.append([i, i + 1])
            
            if is_closed_curve and num_points > 2:
                edges.append([num_points - 1, 0])  # Close the curve
            
            edges = np.array(edges)
            
            # Register curve network in polyscope
            curve_name = f"curve_{curve_type}_{curve_index}"
            ps_curve = ps.register_curve_network(curve_name, curve_points, edges)
            
            # Set color and properties
            color = self.surface_colors.get(curve_type, self.surface_colors["unknown"])
            ps_curve.set_color(color)
            ps_curve.set_radius(0.003)  # Thin curves
            
            # Apply transparency if not fully opaque
            if self.curve_transparency < 1.0:
                ps_curve.set_transparency(1.0 - self.curve_transparency)
            
            self.curve_objects.append(curve_name)
            print(f"  Visualized {curve_type} curve {curve_index} with {num_points} points")
            
        except Exception as e:
            print(f"  Error visualizing {curve_type} {curve_index}: {e}")
    
    def clear_all_objects(self):
        """Clear all polyscope objects"""
        try:
            ps.remove_all_structures()
        except:
            pass
        
        self.original_objects = []
        self.transformed_objects = []
        self.wireframe_objects = []
        self.axes_objects = []
        self.curve_objects = []
        self.center_objects = []
        self.normal_objects = []
    
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
        curve_count = 0
        
        for i, surface_data in enumerate(self.current_surfaces_data):
            surface_type = surface_data.get("type", "unknown")
            surface_idx = surface_data.get("idx", i)
            
            # Skip if this surface type is filtered out
            if not self.should_show_surface_type(surface_type):
                continue
            
            print(f"Processing surface {i}: type={surface_type}, idx={surface_idx}")
            
            # Handle curves separately
            if self.is_curve_type(surface_type):
                self.visualize_curve(surface_data, i)
                curve_count += 1
                continue
            
            # Visualize original surface
            if self.show_original:
                original_points = self.get_original_surface_points(surface_data)
                if original_points is not None:
                    vertices, faces, grid_dims = self.create_surface_mesh(original_points)
                    if vertices is not None:
                        # Create mesh name
                        mesh_name = f"original_{surface_type}_{i}"
                        
                        if faces is not None:
                            ps_mesh = ps.register_surface_mesh(mesh_name, vertices, faces)
                        else:
                            # Register as point cloud if no faces
                            ps_mesh = ps.register_point_cloud(mesh_name, vertices)
                        
                        # Set color and properties
                        if self.use_uv_color_gradient and grid_dims is not None and faces is not None:
                            # Apply UV-based color gradient
                            height, width = grid_dims
                            uv_colors = self.generate_uv_colors(vertices, height, width)
                            if uv_colors is not None:
                                ps_mesh.add_color_quantity("UV Gradient", uv_colors, enabled=True)
                            else:
                                # Fall back to solid color
                                color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                                ps_mesh.set_color(color)
                        else:
                            # Use solid color
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
                
                # Visualize surface center and normal for original surface
                self.visualize_surface_center_and_normal(surface_data, i)
            
            # Visualize transformed surface (only for basic surfaces)
            if self.show_transformed and surface_type in self.basic_surface_types:
                transformed_points = self.sample_transformed_surface(surface_data, i)
                if transformed_points is not None:
                    vertices, faces, grid_dims = self.create_surface_mesh(transformed_points)
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
                        if self.use_uv_color_gradient and grid_dims is not None:
                            # Apply UV-based color gradient
                            height, width = grid_dims
                            uv_colors = self.generate_uv_colors(vertices_final, height, width)
                            if uv_colors is not None:
                                ps_mesh.add_color_quantity("UV Gradient", uv_colors, enabled=True)
                            else:
                                # Fall back to solid color
                                color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                                darker_color = [c * 0.7 for c in color]
                                ps_mesh.set_color(darker_color)
                        else:
                            # Use solid color
                            color = self.surface_colors.get(surface_type, self.surface_colors["unknown"])
                            if self.overlay_mode:
                                # In overlay mode, use darker color with wireframe
                                darker_color = [c * 0.5 for c in color]  # Darker for contrast
                                ps_mesh.set_color(darker_color)
                            else:
                                # In separated mode, use darker color
                                darker_color = [c * 0.7 for c in color]  # Make it darker
                                ps_mesh.set_color(darker_color)
                        
                        if self.overlay_mode:
                            ps_mesh.set_edge_width(1.5)
                            ps_mesh.set_edge_color([0.0, 0.0, 0.0])  # Black edges
                            ps_mesh.set_transparency(0.3)  # Less transparent for visibility
                        else:
                            ps_mesh.set_transparency(self.surface_transparency)
                        
                        if self.wireframe_mode:
                            ps_mesh.set_edge_width(1.0)
                            ps_mesh.set_edge_color([0.1, 0.1, 0.1])
                        
                        self.transformed_objects.append(mesh_name)
                        transformed_count += 1
                
                # Create AABB wireframe if requested and transformation data is available
                if (self.show_aabb_wireframes and surface_type in self.basic_surface_types):
                    transformation = surface_data.get("converted_transformation")
                    if transformation is None:
                        # Use on-the-fly calculation if needed
                        transformed_data = self.calculate_transformation_on_the_fly(surface_data)
                        transformation = transformed_data.get("converted_transformation")
                    
                    if transformation is not None:
                        aabb_min = transformation.get("aabb_min")
                        aabb_max = transformation.get("aabb_max")
                        if aabb_min is not None and aabb_max is not None:
                            self.create_aabb_wireframe(aabb_min, aabb_max, transformation, surface_type, i)
            
            # Count B-spline surfaces
            if surface_type in ["bezier_surface", "bspline_surface"]:
                bspline_count += 1
        
        print(f"Visualization Summary:")
        print(f"  Original surfaces: {original_count}")
        print(f"  Transformed surfaces: {transformed_count}")
        print(f"  B-spline surfaces (original only): {bspline_count}")
        print(f"  Curve surfaces: {curve_count}")
        print(f"  Surface centers: {len(self.center_objects)}")
        print(f"  Surface normals: {len(self.normal_objects)}")
        
        # Create coordinate axes
        self.create_coordinate_axes()
        
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
            
            # AABB filtering
            changed, self.use_aabb_filtering = psim.Checkbox("Use AABB Filtering", self.use_aabb_filtering)
            if changed:
                self.visualize_current_file()
            
            # Show AABB wireframes
            changed, self.show_aabb_wireframes = psim.Checkbox("Show AABB Wireframes", self.show_aabb_wireframes)
            if changed:
                self.visualize_current_file()
            
            # Show coordinate axes
            changed, self.show_coordinate_axes = psim.Checkbox("Show Coordinate Axes", self.show_coordinate_axes)
            if changed:
                self.visualize_current_file()
            
            # Show curves
            changed, self.show_curves = psim.Checkbox("Show Curves", self.show_curves)
            if changed:
                self.visualize_current_file()
            
            # Curve transparency
            changed, self.curve_transparency = psim.SliderFloat(
                "Curve Transparency", 
                self.curve_transparency, 
                0.0, 
                1.0
            )
            if changed:
                self.visualize_current_file()
            
            # UV color gradient
            changed, self.use_uv_color_gradient = psim.Checkbox("UV Color Gradient", self.use_uv_color_gradient)
            if changed:
                self.visualize_current_file()
            
            # Surface centers and normals
            changed, self.show_surface_centers = psim.Checkbox("Show Surface Centers", self.show_surface_centers)
            if changed:
                self.visualize_current_file()
            
            changed, self.show_surface_normals = psim.Checkbox("Show Surface Normals", self.show_surface_normals)
            if changed:
                self.visualize_current_file()
            
            # Normal length control
            changed, self.normal_length = psim.SliderFloat(
                "Normal Vector Length", 
                self.normal_length, 
                0.1, 
                3.0
            )
            if changed:
                self.visualize_current_file()
            
            # Add debugging option
            if psim.Button("Debug: Disable AABB temporarily"):
                self.use_aabb_filtering = False
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
                
                # Curve types
                for curve_type in ["line", "circle", "ellipse", "bspline_curve", "bezier_curve"]:
                    changed, self.surface_type_filters[curve_type] = psim.Checkbox(
                        f"Show {curve_type}", 
                        self.surface_type_filters[curve_type]
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
                
                psim.SameLine()
                if psim.Button("Curves Only"):
                    curve_types = {"line", "circle", "ellipse", "bspline_curve", "bezier_curve"}
                    for key in self.surface_type_filters:
                        self.surface_type_filters[key] = key in curve_types
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
                curve_count = 0
                visible_count = 0
                
                for surface_data in self.current_surfaces_data:
                    surface_type = surface_data.get("type", "unknown")
                    surface_type_counts[surface_type] = surface_type_counts.get(surface_type, 0) + 1
                    
                    if surface_type in self.basic_surface_types:
                        basic_surface_count += 1
                    elif surface_type in ["bezier_surface", "bspline_surface"]:
                        bspline_surface_count += 1
                    elif self.is_curve_type(surface_type):
                        curve_count += 1
                    
                    # Count visible surfaces
                    if self.should_show_surface_type(surface_type):
                        visible_surface_type_counts[surface_type] = visible_surface_type_counts.get(surface_type, 0) + 1
                        visible_count += 1
                
                psim.Text(f"Total surfaces: {len(self.current_surfaces_data)}")
                psim.Text(f"Visible surfaces: {visible_count}")
                psim.Text(f"Basic surfaces: {basic_surface_count}")
                psim.Text(f"B-spline surfaces: {bspline_surface_count}")
                psim.Text(f"Curves: {curve_count}")
                
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
            psim.Text("Curves: shown as colored wireframes")
            psim.Text("Use surface type filters to show specific types")
            psim.Text("'Cones Only' button useful for debugging cones")
            psim.Text("'Curves Only' button to show only curve geometries")
            psim.Text("Overlay mode: surfaces at same position for comparison")
            psim.Text("Non-overlay: transformed surfaces offset by +5 in X")
            psim.Text("AABB Filtering: use 3D bounds to constrain sampling")
            psim.Text("  (when off: shows full infinite surfaces)")
            psim.Text("AABB Wireframes: show bounding boxes as white wireframes")
            psim.Text("  (helps debug why filtering removes points)")
            psim.Text("Coordinate Axes: X=Red, Y=Green, Z=Blue (length=1)")
            psim.Text("  (helps understand spatial orientation)")
            psim.Text("UV Color Gradient: colors surfaces by parameter space")
            psim.Text("  (U direction = hue variation, V direction = brightness)")
            psim.Text("  (helps visualize surface parameterization flow)")
            psim.Text("Surface Centers: shows surface['location'] as colored points")
            psim.Text("Surface Normals: shows surface['direction'] as vectors")
            psim.Text("  (helps understand surface orientation and positioning)")
            psim.Text("")
            psim.Text("IMPORTANT:")
            psim.Text("For best results, first run:")
            psim.Text("python convert_surface_to_transformations.py input.json")
            psim.Text("This pre-calculates transformation data for surfaces.")
            psim.Text("Without this, on-the-fly calculation is slower and")
            psim.Text("may not have AABB filtering data.")
            
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