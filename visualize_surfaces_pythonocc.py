#!/usr/bin/env python3
"""
Surface visualizer using PythonOCC to create B-spline surfaces from control points.
This script loads JSON data containing surface information and visualizes them using polyscope,
with B-spline surfaces created using the proper PythonOCC API.
"""

import argparse
import json
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import List, Dict, Any, Tuple, Optional
import os
import glob
import re

# PythonOCC imports for B-spline surface creation and evaluation
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface, Geom_BezierCurve, Geom_BSplineCurve
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
PYTHONOCC_AVAILABLE = True
#     print("✅ PythonOCC is available - using proper B-spline surface evaluation")
# except ImportError:
#     PYTHONOCC_AVAILABLE = False
#     print("⚠️  PythonOCC not available - falling back to manual B-spline evaluation")


class PythonOCCSurfaceVisualizer:
    def __init__(self):
        # Define color mapping for different surface types
        self.surface_colors = {
            "plane": [0.8, 0.2, 0.2],        # Red
            "cylinder": [0.2, 0.8, 0.2],     # Green  
            "cone": [0.2, 0.2, 0.8],         # Blue
            "sphere": [0.8, 0.8, 0.2],       # Yellow
            "torus": [0.8, 0.2, 0.8],        # Magenta
            "bezier_surface": [0.2, 0.8, 0.8], # Cyan
            "bspline_surface": [0.8, 0.5, 0.2], # Orange
        }
        
        # Define colors for edge/curve types  
        self.edge_colors = {
            "line": [0.5, 0.5, 0.5],         # Gray
            "circle": [1.0, 0.0, 0.0],       # Bright Red
            "ellipse": [0.0, 1.0, 0.0],      # Bright Green
            "hyperbola": [0.0, 0.0, 1.0],    # Bright Blue
            "parabola": [1.0, 1.0, 0.0],     # Bright Yellow
            "bezier_curve": [1.0, 0.0, 1.0], # Bright Magenta
            "bspline_curve": [0.0, 1.0, 1.0], # Bright Cyan
        }
        
        # Track visibility state for each category
        self.surface_visibility = {k: True for k in self.surface_colors.keys()}
        self.edge_visibility = {k: True for k in self.edge_colors.keys()}
        
        # Store registered polyscope objects for each category
        self.surface_objects = {k: [] for k in self.surface_colors.keys()}
        self.edge_objects = {k: [] for k in self.edge_colors.keys()}
        
        # Store control point objects
        self.control_point_objects = []  # Point clouds for control points
        self.control_grid_objects = []   # Grid lines for control points
        
        # Model selection state
        self.available_models = []
        self.current_model_index = 0
        self.current_model_path = None
        self.current_data = None
        self.show_edges = True
        
        # Surface evaluation settings
        self.use_pythonocc_bspline = PYTHONOCC_AVAILABLE
        self.evaluation_resolution = 32  # Resolution for surface evaluation
        self.show_wireframe = False  # Show wireframe to see mesh density
        
        # Counter for unique object names to force polyscope refresh
        self.refresh_counter = 0
        
        # Control points visualization settings
        self.show_control_points = True  # Show control points for B-spline surfaces
        self.show_control_grid = True   # Show control point grid connections
        self.control_point_size = 0.02  # Size of control point spheres
        
        # Surface selection and transparency settings
        self.selected_surface_index = -1  # Index of currently selected surface (-1 = none)
        self.surface_transparency = 0.8   # Default transparency for surfaces
        self.surface_metadata = []        # Store surface metadata for selection
        self.surface_control_points = {}  # Map surface index to control point objects
        self.structure_name_to_surface_index = {}  # Map polyscope structure names to surface indices
        
        # Click-based selection settings
        self.click_selection_enabled = True  # Enable/disable click selection

    def find_json_models(self, directory: str) -> List[Dict[str, str]]:
        """Find all JSON files with UID pattern in the directory"""
        models = []
        
        # Search for JSON files recursively
        json_pattern = os.path.join(directory, "**", "*.json")
        json_files = glob.glob(json_pattern, recursive=True)
        
        # Filter for files that match UID pattern (8 digits)
        uid_pattern = re.compile(r'(\d{8})\.json$')
        
        for json_file in json_files:
            filename = os.path.basename(json_file)
            match = uid_pattern.search(filename)
            if match:
                uid = match.group(1)
                models.append({
                    'uid': uid,
                    'path': json_file,
                    'filename': filename,
                    'relative_path': os.path.relpath(json_file, directory)
                })
        
        # Sort by UID
        models.sort(key=lambda x: x['uid'])
        return models

    def load_json_data(self, json_path: str) -> List[Dict[str, Any]]:
        """Load JSON data from file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def create_bspline_surface_pythonocc(self, control_points: List[List[float]]) -> Optional[Geom_BSplineSurface]:
        """
        Create a B-spline surface using PythonOCC from 4x4 control points
        
        Args:
            control_points: List of 16 [x,y,z] control points (4x4 grid)
            
        Returns:
            Geom_BSplineSurface object or None if creation fails
        """
        if not PYTHONOCC_AVAILABLE:
            return None
            
        try:
            # Convert control points to TColgp_Array2OfPnt (1-indexed, 4x4)
            poles = TColgp_Array2OfPnt(1, 4, 1, 4)
            
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    pt = control_points[idx]
                    poles.SetValue(i + 1, j + 1, gp_Pnt(pt[0], pt[1], pt[2]))
            
            # Define knot vectors for cubic B-spline (degree 3)
            # For 4 control points, we need 8 knots: [0,0,0,0,1,1,1,1]
            u_knots = TColStd_Array1OfReal(1, 2)
            v_knots = TColStd_Array1OfReal(1, 2)
            u_knots.SetValue(1, 0.0)
            u_knots.SetValue(2, 1.0)
            v_knots.SetValue(1, 0.0)
            v_knots.SetValue(2, 1.0)
            
            # Define multiplicities (how many times each knot is repeated)
            u_mults = TColStd_Array1OfInteger(1, 2)
            v_mults = TColStd_Array1OfInteger(1, 2)
            u_mults.SetValue(1, 4)  # Multiplicity 4 for clamped spline
            u_mults.SetValue(2, 4)
            v_mults.SetValue(1, 4)
            v_mults.SetValue(2, 4)
            
            # Create the B-spline surface (degree 3 in both directions)
            surface = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, 3, 3)
            
            return surface
            
        except Exception as e:
            print(f"Error creating B-spline surface: {e}")
            return None

    def evaluate_bspline_surface_pythonocc(self, surface: Geom_BSplineSurface, resolution: int = 32) -> np.ndarray:
        """
        Evaluate B-spline surface using PythonOCC at given resolution
        
        Args:
            surface: Geom_BSplineSurface object
            resolution: Number of evaluation points in each direction
            
        Returns:
            np.ndarray: Shape (resolution, resolution, 3) with evaluated points
        """
        try:
            # Create parameter arrays
            u_params = np.linspace(0.0, 1.0, resolution)
            v_params = np.linspace(0.0, 1.0, resolution)
            
            # Initialize output array
            surface_points = np.zeros((resolution, resolution, 3))
            
            # Evaluate surface at each parameter point
            for u_idx, u in enumerate(u_params):
                for v_idx, v in enumerate(v_params):
                    point = surface.Value(u, v)
                    surface_points[u_idx, v_idx] = [point.X(), point.Y(), point.Z()]
            
            return surface_points
            
        except Exception as e:
            print(f"Error evaluating B-spline surface: {e}")
            return None

    def fallback_bspline_evaluation(self, control_points: List[List[float]], resolution: int = 32) -> np.ndarray:
        """
        Fallback B-spline evaluation when PythonOCC is not available
        Uses manual cubic B-spline basis functions that match PythonOCC's clamped B-spline behavior
        """
        # Reshape control points to 4x4x3 array
        control_points = np.array(control_points).reshape(4, 4, 3)
        
        # Create parameter arrays
        u_params = np.linspace(0, 1, resolution)
        v_params = np.linspace(0, 1, resolution)
        
        # Initialize output array
        surface_points = np.zeros((resolution, resolution, 3))
        
        # B-spline basis functions for degree 3 (cubic) with clamped knot vector [0,0,0,0,1,1,1,1]
        def cubic_bspline_basis_clamped(t, i):
            """Cubic B-spline basis function for clamped spline (matches PythonOCC behavior)"""
            if i == 0:
                return (1 - t)**3
            elif i == 1:
                return 3*t*(1-t)**2
            elif i == 2:
                return 3*t**2*(1-t)
            elif i == 3:
                return t**3
            else:
                return 0.0
        
        # Evaluate surface at each parameter point
        for u_idx, u in enumerate(u_params):
            for v_idx, v in enumerate(v_params):
                point = np.zeros(3)
                
                # Sum over all control points with basis functions
                for i in range(4):
                    for j in range(4):
                        basis_u = cubic_bspline_basis_clamped(u, i)
                        basis_v = cubic_bspline_basis_clamped(v, j)
                        weight = basis_u * basis_v
                        point += weight * control_points[i, j]
                
                surface_points[u_idx, v_idx] = point
        
        return surface_points

    def get_surface_points(self, item: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        Get surface points using PythonOCC B-spline evaluation or fallback methods
        
        Args:
            item: Surface data item from JSON
            
        Returns:
            tuple: (surface_points, method_used)
                - surface_points: np.ndarray of shape (res, res, 3)
                - method_used: string describing the method used
        """
        # Try to use B-spline approximation first
        approximation = item.get("approximation")
        if approximation is not None and len(approximation) == 16:
            
            if self.use_pythonocc_bspline and PYTHONOCC_AVAILABLE:
                # Use PythonOCC B-spline surface
                surface = self.create_bspline_surface_pythonocc(approximation)
                if surface is not None:
                    surface_points = self.evaluate_bspline_surface_pythonocc(surface, self.evaluation_resolution)
                    if surface_points is not None:
                        return surface_points, f"PythonOCC B-spline ({self.evaluation_resolution}×{self.evaluation_resolution})"
            
            # Fallback to manual B-spline evaluation
            surface_points = self.fallback_bspline_evaluation(approximation, self.evaluation_resolution)
            return surface_points, f"Manual B-spline ({self.evaluation_resolution}×{self.evaluation_resolution})"
        
        # Fall back to original points
        points = item.get("points")
        if points is not None:
            original_points = np.array(points)
            return original_points, f"Original points ({original_points.shape[0]}×{original_points.shape[1]})"
        
        return None, "No surface data"

    def create_surface_mesh(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Create mesh vertices and faces from a grid of points
        Returns (vertices, faces) where vertices is Nx3 and faces is Mx3
        """
        if points is None:
            return None, None
            
        # Handle different resolutions
        if len(points.shape) == 3:
            height, width = points.shape[:2]
        else:
            # Try to reshape to square grid
            total_points = len(points)
            side_length = int(np.sqrt(total_points))
            if side_length * side_length == total_points:
                points = np.array(points).reshape(side_length, side_length, 3)
                height, width = side_length, side_length
            else:
                # Assume 32x32 if shape is unclear
                points = np.array(points).reshape(32, 32, 3)
                height, width = 32, 32
        
        # Flatten points to get vertices
        vertices = points.reshape(-1, 3)
        
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

    def create_curve_points(self, points: List[List[float]]) -> np.ndarray:
        """
        Create curve points from a list of 3D points
        Returns points as Nx3 array
        """
        if isinstance(points, list):
            return np.array(points)
        else:
            return points

    def create_control_point_visualization(self, control_points: List[List[float]], surface_type: str, surface_index: int) -> tuple:
        """
        Create visualization for B-spline control points
        
        Args:
            control_points: List of 16 [x,y,z] control points (4x4 grid)
            surface_type: Type of surface (for naming)
            surface_index: Index of surface (for naming)
            
        Returns:
            tuple: (point_cloud_object, grid_object) or (None, None) if creation fails
        """
        if not control_points or len(control_points) != 16:
            return None, None
            
        try:
            # Convert to numpy array and reshape to 4x4x3
            cp_array = np.array(control_points).reshape(4, 4, 3)
            
            # Flatten for point cloud
            cp_points = cp_array.reshape(-1, 3)
            
            point_cloud_obj = None
            grid_obj = None
            
            if self.show_control_points:
                # Create point cloud for control points
                cp_name = f"control_points_{surface_type}_{surface_index}_{self.refresh_counter}"
                point_cloud_obj = ps.register_point_cloud(cp_name, cp_points)
                point_cloud_obj.set_color([1.0, 0.5, 0.0])  # Orange color for control points
                point_cloud_obj.set_radius(self.control_point_size)
            
            if self.show_control_grid:
                # Create grid connections between control points
                grid_edges = []
                
                # Add horizontal connections (u-direction)
                for i in range(4):
                    for j in range(3):
                        v1 = i * 4 + j
                        v2 = i * 4 + (j + 1)
                        grid_edges.append([v1, v2])
                
                # Add vertical connections (v-direction)
                for i in range(3):
                    for j in range(4):
                        v1 = i * 4 + j
                        v2 = (i + 1) * 4 + j
                        grid_edges.append([v1, v2])
                
                grid_edges = np.array(grid_edges)
                
                # Create grid curve network
                grid_name = f"control_grid_{surface_type}_{surface_index}_{self.refresh_counter}"
                grid_obj = ps.register_curve_network(grid_name, cp_points, edges=grid_edges)
                grid_obj.set_color([0.8, 0.4, 0.0])  # Darker orange for grid lines
                grid_obj.set_radius(0.005)  # Thin lines for grid
            
            return point_cloud_obj, grid_obj
            
        except Exception as e:
            print(f"Error creating control point visualization: {e}")
            return None, None

    def clear_all_objects(self):
        """Clear all registered polyscope objects"""
        try:
            ps.remove_all_structures()
        except:
            pass
        
        # Reset object tracking
        for surface_type in self.surface_objects:
            self.surface_objects[surface_type] = []
        for edge_type in self.edge_objects:
            self.edge_objects[edge_type] = []
        
        # Reset control point tracking
        self.control_point_objects = []
        self.control_grid_objects = []
        
        # Reset surface selection state
        self.selected_surface_index = -1
        self.surface_metadata = []
        self.surface_control_points = {}
        self.structure_name_to_surface_index = {}

    def load_model(self, model_index: int):
        """Load a specific model by index"""
        if 0 <= model_index < len(self.available_models):
            model = self.available_models[model_index]
            
            print(f"Loading model: {model['uid']} ({model['relative_path']})")
            
            try:
                # Increment refresh counter for new object names
                self.refresh_counter += 1
                
                # Clear existing objects
                self.clear_all_objects()
                
                # Load new data
                self.current_data = self.load_json_data(model['path'])
                self.current_model_path = model['path']
                self.current_model_index = model_index
                
                # Validate data
                if not self.current_data or not isinstance(self.current_data, list):
                    raise ValueError("Invalid or empty JSON data")
                
                # Visualize the new model
                self.visualize_current_model()
                
                print(f"✅ Successfully loaded {len(self.current_data)} features from {model['filename']}")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error in model {model['uid']}: {e}")
                self.current_data = None
            except Exception as e:
                print(f"❌ Error loading model {model['uid']}: {e}")
                self.current_data = None

    def load_next_valid_model(self, direction: int = 1):
        """Load the next valid model in the given direction"""
        start_index = self.current_model_index
        attempts = 0
        max_attempts = len(self.available_models)
        
        while attempts < max_attempts:
            new_index = (self.current_model_index + direction) % len(self.available_models)
            
            old_index = self.current_model_index
            self.current_model_index = new_index
            self.load_model(new_index)
            
            if self.current_data is not None:
                break
                
            attempts += 1
            
        if attempts >= max_attempts:
            print("⚠️  No valid models found in the directory")
            self.current_model_index = start_index

    def update_surface_evaluation_mode(self):
        """Update surface evaluation when switching between modes or changing resolution"""
        if self.current_data is not None:
            print(f"🔄 Updating surface evaluation - Mode: {'PythonOCC' if self.use_pythonocc_bspline else 'Manual'}, Resolution: {self.evaluation_resolution}")
            
            # Increment refresh counter to force new object names
            self.refresh_counter += 1
            
            # Clear existing objects completely
            self.clear_all_objects()
            
            # Re-visualize with new objects
            self.visualize_current_model()

    def update_visibility(self):
        """Update visibility of all objects based on current state"""
        # Update surface visibility
        for surface_type, objects in self.surface_objects.items():
            visible = self.surface_visibility[surface_type]
            for obj in objects:
                obj.set_enabled(visible)
        
        # Update edge visibility  
        for edge_type, objects in self.edge_objects.items():
            visible = self.edge_visibility[edge_type]
            for obj in objects:
                obj.set_enabled(visible)
        
        # Update control point visibility based on selection
        if self.selected_surface_index >= 0:
            # Show control points only for selected surface
            for surface_idx, (cp_obj, grid_obj) in self.surface_control_points.items():
                show_cp = (surface_idx == self.selected_surface_index and self.show_control_points)
                show_grid = (surface_idx == self.selected_surface_index and self.show_control_grid)
                
                if cp_obj is not None:
                    cp_obj.set_enabled(show_cp)
                if grid_obj is not None:
                    grid_obj.set_enabled(show_grid)
        else:
            # Show all control points if no selection
            for surface_idx, (cp_obj, grid_obj) in self.surface_control_points.items():
                if cp_obj is not None:
                    cp_obj.set_enabled(self.show_control_points)
                if grid_obj is not None:
                    grid_obj.set_enabled(self.show_control_grid)

    def update_wireframe_mode(self):
        """Update wireframe mode for all surface objects"""
        for surface_type, objects in self.surface_objects.items():
            for obj in objects:
                try:
                    if self.show_wireframe:
                        obj.set_edge_width(1.0)
                        obj.set_edge_color([0.2, 0.2, 0.2])
                    else:
                        obj.set_edge_width(0.0)
                except:
                    pass
        print(f"🔲 Wireframe mode: {'ON' if self.show_wireframe else 'OFF'}")

    def select_surface(self, surface_index: int):
        """Select a surface and update visualization accordingly"""
        if surface_index < 0 or surface_index >= len(self.surface_metadata):
            # Deselect all surfaces
            self.selected_surface_index = -1
        else:
            self.selected_surface_index = surface_index
        
        # Update transparency and highlighting
        self.update_surface_highlighting()
        
        # Update control point visibility
        self.update_visibility()
        
        # Print selection info
        if self.selected_surface_index >= 0:
            surface_info = self.surface_metadata[self.selected_surface_index]
            print(f"🎯 Selected surface {self.selected_surface_index}: {surface_info['type']} (method: {surface_info['method']})")
        else:
            print("🔄 Deselected all surfaces")

    def update_surface_highlighting(self):
        """Update surface highlighting based on selection"""
        surface_idx = 0
        for surface_type, objects in self.surface_objects.items():
            for obj in objects:
                if self.selected_surface_index >= 0 and surface_idx == self.selected_surface_index:
                    # Highlight selected surface (opaque)
                    obj.set_transparency(0.1)
                    # Optional: slightly brighten the color
                    base_color = self.surface_colors[surface_type]
                    highlighted_color = [min(1.0, c * 1.2) for c in base_color]
                    obj.set_color(highlighted_color)
                else:
                    # Normal transparency for non-selected surfaces
                    obj.set_transparency(self.surface_transparency)
                    obj.set_color(self.surface_colors[surface_type])
                surface_idx += 1

    def create_gui_callback(self):
        """Create the GUI callback function for polyscope"""
        def gui_callback():
            # --- Handle Mouse Clicks for Surface Selection ---
            if self.click_selection_enabled:
                io = psim.GetIO()
                if io.MouseClicked[0]:  # Left mouse button clicked
                    mouse_pos = io.MousePos
                    screen_coords = (mouse_pos[0], mouse_pos[1])
                    pick_result = ps.pick(screen_coords=screen_coords)
                    self.handle_surface_click(pick_result)
            
            psim.PushItemWidth(200)
            
            # Surface evaluation mode panel
            if psim.TreeNode("Surface Evaluation"):
                psim.Text("B-spline Surface Method:")
                psim.Separator()
                
                # Method selection
                if PYTHONOCC_AVAILABLE:
                    if psim.RadioButton("PythonOCC B-spline", self.use_pythonocc_bspline):
                        if not self.use_pythonocc_bspline:
                            self.use_pythonocc_bspline = True
                            self.update_surface_evaluation_mode()
                    
                    if psim.RadioButton("Manual B-spline", not self.use_pythonocc_bspline):
                        if self.use_pythonocc_bspline:
                            self.use_pythonocc_bspline = False
                            self.update_surface_evaluation_mode()
                else:
                    psim.Text("PythonOCC not available")
                    psim.Text("Using manual B-spline evaluation")
                
                psim.Separator()
                
                # Resolution control
                psim.Text("Surface Resolution:")
                changed_res, new_res = psim.SliderInt("Resolution", self.evaluation_resolution, 16, 64)
                if changed_res:
                    self.evaluation_resolution = new_res
                    self.update_surface_evaluation_mode()
                
                psim.Text(f"Grid size: {self.evaluation_resolution}×{self.evaluation_resolution}")
                
                psim.Separator()
                
                # Wireframe mode toggle
                changed_wireframe, self.show_wireframe = psim.Checkbox("Show Wireframe", self.show_wireframe)
                if changed_wireframe:
                    self.update_wireframe_mode()
                
                psim.TreePop()
            
            # Model selection panel
            if psim.TreeNode("Model Selection"):
                if len(self.available_models) > 0:
                    current_model = self.available_models[self.current_model_index]
                    psim.Text(f"Current: {current_model['uid']} ({current_model['filename']})")
                    psim.Text(f"Path: {current_model['relative_path']}")
                    psim.Separator()
                    
                    # Model navigation buttons
                    if psim.Button("Previous Model"):
                        self.load_next_valid_model(-1)
                    
                    psim.SameLine()
                    if psim.Button("Next Model"):
                        self.load_next_valid_model(1)
                    
                    # Model dropdown selector
                    model_names = [f"{m['uid']} - {m['filename']}" for m in self.available_models]
                    changed, new_index = psim.Combo("Select Model", self.current_model_index, model_names)
                    if changed:
                        self.load_model(new_index)
                    
                    psim.Separator()
                    psim.Text(f"Total Models: {len(self.available_models)}")
                    psim.Text(f"Model {self.current_model_index + 1} of {len(self.available_models)}")
                else:
                    psim.Text("No models loaded")
                
                psim.TreePop()
            
            # Surface controls
            if psim.TreeNode("Surface Controls"):
                psim.Text("Show/Hide Surface Types:")
                psim.Separator()
                
                # Global surface controls
                if psim.Button("Show All Surfaces"):
                    for key in self.surface_visibility:
                        self.surface_visibility[key] = True
                    self.update_visibility()
                
                psim.SameLine()
                if psim.Button("Hide All Surfaces"):
                    for key in self.surface_visibility:
                        self.surface_visibility[key] = False
                    self.update_visibility()
                
                psim.Separator()
                
                # Individual surface type controls
                for surface_type in self.surface_colors.keys():
                    if surface_type in self.surface_objects and len(self.surface_objects[surface_type]) > 0:
                        count = len(self.surface_objects[surface_type])
                        color = self.surface_colors[surface_type]
                        
                        # Color indicator
                        psim.PushStyleColor(psim.ImGuiCol_Button, (*color, 1.0))
                        psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (*[c*1.2 for c in color], 1.0))
                        psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (*[c*0.8 for c in color], 1.0))
                        
                        changed, self.surface_visibility[surface_type] = psim.Checkbox(
                            f"{surface_type.title()} ({count})", 
                            self.surface_visibility[surface_type]
                        )
                        
                        psim.PopStyleColor(3)
                        
                        if changed:
                            self.update_visibility()
                
                psim.TreePop()
            
            # Edge controls
            if psim.TreeNode("Edge Controls"):
                psim.Text("Show/Hide Edge Types:")
                psim.Separator()
                
                # Global edge controls
                if psim.Button("Show All Edges"):
                    for key in self.edge_visibility:
                        self.edge_visibility[key] = True
                    self.update_visibility()
                
                psim.SameLine()
                if psim.Button("Hide All Edges"):
                    for key in self.edge_visibility:
                        self.edge_visibility[key] = False
                    self.update_visibility()
                
                psim.Separator()
                
                # Individual edge type controls
                for edge_type in self.edge_colors.keys():
                    if edge_type in self.edge_objects and len(self.edge_objects[edge_type]) > 0:
                        count = len(self.edge_objects[edge_type])
                        color = self.edge_colors[edge_type]
                        
                        # Color indicator
                        psim.PushStyleColor(psim.ImGuiCol_Button, (*color, 1.0))
                        psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, (*[c*1.2 for c in color], 1.0))
                        psim.PushStyleColor(psim.ImGuiCol_ButtonActive, (*[c*0.8 for c in color], 1.0))
                        
                        changed, self.edge_visibility[edge_type] = psim.Checkbox(
                            f"{edge_type.title()} ({count})", 
                            self.edge_visibility[edge_type]
                        )
                        
                        psim.PopStyleColor(3)
                        
                        if changed:
                            self.update_visibility()
                
                psim.TreePop()
            
            # Control Points panel
            if psim.TreeNode("Control Points"):
                psim.Text("B-spline Control Points:")
                psim.Separator()
                
                # Control points toggle
                changed_cp, self.show_control_points = psim.Checkbox("Show Control Points", self.show_control_points)
                if changed_cp:
                    self.update_visibility()
                
                # Control grid toggle
                changed_grid, self.show_control_grid = psim.Checkbox("Show Control Grid", self.show_control_grid)
                if changed_grid:
                    self.update_visibility()
                
                psim.Separator()
                
                # Control point size slider
                if self.show_control_points:
                    psim.Text("Control Point Size:")
                    changed_size, new_size = psim.SliderFloat("Point Size", self.control_point_size, 0.005, 0.05)
                    if changed_size:
                        self.control_point_size = new_size
                        # Update existing control point sizes
                        for obj in self.control_point_objects:
                            obj.set_radius(self.control_point_size)
                
                psim.Text("Orange points show B-spline control points")
                psim.Text("Grid shows control point connectivity")
                
                psim.TreePop()
            
            # Surface Selection panel
            if psim.TreeNode("Surface Selection"):
                psim.Text("Interactive Surface Selection:")
                psim.Separator()
                
                # Click selection toggle
                changed_click, self.click_selection_enabled = psim.Checkbox("Enable Click Selection", self.click_selection_enabled)
                if self.click_selection_enabled:
                    psim.TextColored((0.0, 0.8, 0.0, 1.0), "Click on surfaces to select them")
                else:
                    psim.TextColored((0.8, 0.8, 0.0, 1.0), "Click selection disabled")
                
                psim.Separator()
                
                # Surface list for selection
                if len(self.surface_metadata) > 0:
                    surface_names = []
                    for i, surface_info in enumerate(self.surface_metadata):
                        name = f"{i}: {surface_info['type']} ({surface_info['method']})"
                        surface_names.append(name)
                    
                    # Add "None" option for deselection
                    surface_names.insert(0, "None (show all control points)")
                    
                    current_selection = self.selected_surface_index + 1  # +1 because of "None" option
                    changed, new_selection = psim.Combo("Select Surface", current_selection, surface_names)
                    if changed:
                        self.select_surface(new_selection - 1)  # -1 because of "None" option
                    
                    psim.Separator()
                    
                    # Transparency control
                    psim.Text("Surface Transparency:")
                    changed_trans, new_trans = psim.SliderFloat("Transparency", self.surface_transparency, 0.1, 1.0)
                    if changed_trans:
                        self.surface_transparency = new_trans
                        self.update_surface_highlighting()
                    
                    psim.Separator()
                    
                    # Selection info
                    if self.selected_surface_index >= 0:
                        surface_info = self.surface_metadata[self.selected_surface_index]
                        psim.Text(f"Selected: {surface_info['type']}")
                        psim.Text(f"Method: {surface_info['method']}")
                        psim.Text("Control points shown for selected surface only")
                    else:
                        psim.Text("No surface selected")
                        psim.Text("Control points shown for all surfaces")
                else:
                    psim.Text("No surfaces available for selection")
                
                psim.TreePop()
            
            # Statistics panel
            if psim.TreeNode("Statistics"):
                psim.Text("Geometry Statistics:")
                psim.Separator()
                
                total_surfaces = sum(len(objs) for objs in self.surface_objects.values())
                total_edges = sum(len(objs) for objs in self.edge_objects.values())
                
                psim.Text(f"Total Surfaces: {total_surfaces}")
                psim.Text(f"Total Edges: {total_edges}")
                
                if self.current_data:
                    psim.Text(f"Total Features: {len(self.current_data)}")
                
                # Control point statistics
                psim.Text(f"Control Points: {len(self.control_point_objects)}")
                psim.Text(f"Control Grids: {len(self.control_grid_objects)}")
                
                psim.Separator()
                
                # Show method being used
                if PYTHONOCC_AVAILABLE:
                    method = "PythonOCC" if self.use_pythonocc_bspline else "Manual"
                else:
                    method = "Manual (PythonOCC N/A)"
                psim.Text(f"B-spline Method: {method}")
                psim.Text(f"Resolution: {self.evaluation_resolution}×{self.evaluation_resolution}")
                
                psim.Separator()
                psim.Text("Surface Breakdown:")
                for surface_type, objects in self.surface_objects.items():
                    if len(objects) > 0:
                        psim.Text(f"  {surface_type.title()}: {len(objects)}")
                
                psim.Separator()
                psim.Text("Edge Breakdown:")
                for edge_type, objects in self.edge_objects.items():
                    if len(objects) > 0:
                        psim.Text(f"  {edge_type.title()}: {len(objects)}")
                
                psim.TreePop()
            
            psim.PopItemWidth()
        
        return gui_callback

    def visualize_current_model(self):
        """Visualize the currently loaded model"""
        if not self.current_data:
            return
            
        surface_count = {}
        edge_count = {}
        pythonocc_count = 0
        manual_count = 0
        original_count = 0
        
        # Reset surface metadata and control points tracking
        self.surface_metadata = []
        self.surface_control_points = {}
        surface_index = 0  # Global surface index across all types
        
        for item in self.current_data:
            item_type = item.get("type", "unknown")
            
            # Check if this is a surface type
            if item_type in self.surface_colors:
                # Handle surfaces
                surface_points, method_used = self.get_surface_points(item)
                if surface_points is not None and len(surface_points) > 0:
                    try:
                        vertices, faces = self.create_surface_mesh(surface_points)
                        
                        if vertices is not None and faces is not None:
                            # Create unique name for this surface
                            if item_type not in surface_count:
                                surface_count[item_type] = 0
                            surface_count[item_type] += 1
                            
                            mesh_name = f"{item_type}_{surface_count[item_type]}_{self.refresh_counter}"
                            
                            # Register mesh with polyscope
                            ps_mesh = ps.register_surface_mesh(mesh_name, vertices, faces)
                            
                            # Set color based on surface type
                            ps_mesh.set_color(self.surface_colors[item_type])
                            
                            # Apply default transparency
                            ps_mesh.set_transparency(self.surface_transparency)
                            
                            # Apply wireframe mode if enabled
                            if self.show_wireframe:
                                try:
                                    ps_mesh.set_edge_width(1.0)
                                    ps_mesh.set_edge_color([0.2, 0.2, 0.2])
                                except:
                                    pass
                            
                            # Store the object for visibility control
                            self.surface_objects[item_type].append(ps_mesh)
                            
                            # Create control point visualization if approximation exists
                            approximation = item.get("approximation")
                            cp_obj, grid_obj = None, None
                            if approximation is not None and len(approximation) == 16:
                                cp_obj, grid_obj = self.create_control_point_visualization(
                                    approximation, item_type, surface_count[item_type]
                                )
                                if cp_obj is not None:
                                    self.control_point_objects.append(cp_obj)
                                if grid_obj is not None:
                                    self.control_grid_objects.append(grid_obj)
                            
                            # Count evaluation methods
                            if "PythonOCC" in method_used:
                                pythonocc_count += 1
                            elif "Manual" in method_used:
                                manual_count += 1
                            else:
                                original_count += 1
                            
                            print(f"  📊 {item_type}: {method_used}")
                        
                            # Store surface metadata
                            self.surface_metadata.append({
                                'type': item_type,
                                'method': method_used
                            })
                            self.surface_control_points[surface_index] = (cp_obj, grid_obj)
                            
                            # Map structure name to surface index for click selection
                            self.structure_name_to_surface_index[mesh_name] = surface_index
                            
                            surface_index += 1
                        
                    except Exception as e:
                        print(f"Error processing {item_type} surface: {e}")
                        
            elif self.show_edges and item_type in self.edge_colors:
                # Handle edges/curves
                points = item.get("points")
                if points is not None and len(points) > 0:
                    try:
                        curve_points = self.create_curve_points(points)
                        
                        # Create unique name for this edge
                        if item_type not in edge_count:
                            edge_count[item_type] = 0
                        edge_count[item_type] += 1
                        
                        curve_name = f"{item_type}_{edge_count[item_type]}_{self.refresh_counter}"
                        
                        # Create edges connecting consecutive points
                        num_points = curve_points.shape[0]
                        edges = np.array([[i, i+1] for i in range(num_points-1)])
                        
                        # Register curve with polyscope
                        ps_curve = ps.register_curve_network(curve_name, curve_points, edges=edges)
                        
                        # Set color based on edge type
                        ps_curve.set_color(self.edge_colors[item_type])
                        ps_curve.set_radius(0.005)
                        
                        # Store the object for visibility control
                        self.edge_objects[item_type].append(ps_curve)
                        
                    except Exception as e:
                        print(f"Error processing {item_type} curve: {e}")

        # Print summary
        print(f"\nVisualization Summary:")
        print(f"Surface types found: {list(surface_count.keys())}")
        print(f"Surface counts: {surface_count}")
        if self.show_edges:
            print(f"Edge types found: {list(edge_count.keys())}")
            print(f"Edge counts: {edge_count}")
        
        print(f"\nSurface Evaluation Methods:")
        if pythonocc_count > 0:
            print(f"  PythonOCC B-spline: {pythonocc_count} surfaces")
        if manual_count > 0:
            print(f"  Manual B-spline: {manual_count} surfaces")
        if original_count > 0:
            print(f"  Original points: {original_count} surfaces")
        
        print(f"\nControl Point Visualization:")
        print(f"  Control point clouds: {len(self.control_point_objects)}")
        print(f"  Control grids: {len(self.control_grid_objects)}")
        
        # Apply initial surface highlighting and control point visibility
        self.update_surface_highlighting()
        self.update_visibility()

    def visualize_directory(self, directory: str, show_edges: bool = True):
        """
        Visualize JSON models from a directory with interactive model switching
        """
        self.show_edges = show_edges
        
        # Find all available models
        print(f"Searching for JSON models in: {directory}")
        self.available_models = self.find_json_models(directory)
        
        if not self.available_models:
            print("No JSON files with UID pattern found in directory!")
            return
        
        print(f"Found {len(self.available_models)} models:")
        for i, model in enumerate(self.available_models[:5]):
            print(f"  {model['uid']}: {model['relative_path']}")
        if len(self.available_models) > 5:
            print(f"  ... and {len(self.available_models) - 5} more")
        
        # Initialize polyscope
        ps.init()
        
        # Load the first valid model
        if self.available_models:
            self.current_model_index = -1
            self.load_next_valid_model(1)
        
        # Set up the GUI callback
        ps.set_user_callback(self.create_gui_callback())
        
        print(f"\n🎛️ Interactive Controls:")
        print(f"   - Use 'Surface Evaluation' to switch between PythonOCC and manual B-spline")
        print(f"   - Adjust resolution for surface sampling quality")
        print(f"   - Use 'Model Selection' panel to switch between models")
        print(f"   - Toggle geometry types on/off")
        print(f"   - View detailed statistics for each model")
        print()
        
        # Show the visualization
        ps.show()

    def visualize_surfaces(self, data: List[Dict[str, Any]], show_edges: bool = True):
        """
        Visualize surfaces from JSON data using polyscope (single-file method)
        """
        self.current_data = data
        self.show_edges = show_edges
        
        ps.init()
        
        self.visualize_current_model()
        
        # Set up the GUI callback
        ps.set_user_callback(self.create_gui_callback())
        
        print(f"\nUse the control panel to switch between PythonOCC and manual B-spline evaluation!")
        print(f"Toggle geometry types and adjust surface resolution as needed!")
        
        # Show the visualization
        ps.show()

    def handle_surface_click(self, pick_result):
        """
        Handle a click on a surface to select it.
        
        Args:
            pick_result: Polyscope pick result from ps.pick()
        """
        if not self.click_selection_enabled:
            return
        
        if not pick_result.is_hit:
            # Clicked on empty space - deselect all
            self.select_surface(-1)
            return
        
        # Parse structure name to get surface information
        structure_name = pick_result.structure_name
        
        # Our surface meshes are named like: "{surface_type}_{count}_{refresh_counter}"
        # We need to map this back to our surface index
        surface_index = self.get_surface_index_from_structure_name(structure_name)
        
        if surface_index is not None:
            self.select_surface(surface_index)
            print(f"🖱️ Clicked surface {surface_index}: {self.surface_metadata[surface_index]['type']}")
        else:
            print(f"🖱️ Clicked on non-surface structure: {structure_name}")

    def get_surface_index_from_structure_name(self, structure_name: str) -> Optional[int]:
        """
        Map a polyscope structure name back to our surface index.
        
        Args:
            structure_name: Name of the clicked structure
            
        Returns:
            Optional[int]: Surface index, or None if not a surface
        """
        try:
            # Check if we have this structure name mapped to a surface index
            if structure_name in self.structure_name_to_surface_index:
                return self.structure_name_to_surface_index[structure_name]
            
            # Structure not found - it might be a control point or edge
            return None
            
        except Exception as e:
            print(f"Error parsing structure name '{structure_name}': {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Visualize surfaces from JSON BREP data using PythonOCC B-spline evaluation")
    parser.add_argument("input", type=str, help="Path to JSON file or directory containing JSON files")
    parser.add_argument("--no-edges", action="store_true", help="Don't show edges/curves, only surfaces")
    parser.add_argument("--manual-bspline", action="store_true", help="Use manual B-spline evaluation instead of PythonOCC")
    parser.add_argument("--resolution", type=int, default=32, help="Surface evaluation resolution (default: 32)")
    
    args = parser.parse_args()
    
    # Validate resolution
    if args.resolution < 8 or args.resolution > 128:
        print("Warning: Resolution should be between 8 and 128, using default 32")
        args.resolution = 32
    
    # Create visualizer
    visualizer = PythonOCCSurfaceVisualizer()
    
    # Set initial parameters
    if args.manual_bspline or not PYTHONOCC_AVAILABLE:
        visualizer.use_pythonocc_bspline = False
    visualizer.evaluation_resolution = args.resolution
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file mode
        try:
            data = visualizer.load_json_data(args.input)
            print(f"Loaded {len(data)} features from {args.input}")
            
            # Visualize
            show_edges = not args.no_edges
            visualizer.visualize_surfaces(data, show_edges=show_edges)
            
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return
            
    elif os.path.isdir(args.input):
        # Directory mode
        show_edges = not args.no_edges
        visualizer.visualize_directory(args.input, show_edges=show_edges)
        
    else:
        print(f"Error: '{args.input}' is neither a file nor a directory")
        return


if __name__ == "__main__":
    main() 