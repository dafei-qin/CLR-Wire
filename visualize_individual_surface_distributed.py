#!/usr/bin/env python3
"""
Individual surface visualizer for distributed NPZ files (one per UID).
Allows browsing through UIDs and surfaces within each UID.
"""

import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import List, Dict, Any, Tuple, Optional
import os
import glob

# Import the distributed loader
from load_distributed_npz import DistributedNPZLoader

# PythonOCC imports for B-spline surface creation and evaluation
try:
    from OCC.Core.gp import gp_Pnt
    from OCC.Core.Geom import Geom_BSplineSurface, Geom_BSplineCurve
    from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt
    from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    PYTHONOCC_AVAILABLE = True
    print("✅ PythonOCC is available - using proper B-spline evaluation")
except ImportError:
    PYTHONOCC_AVAILABLE = False
    print("⚠️  PythonOCC not available - falling back to manual evaluation")


class DistributedSurfaceVisualizer:
    def __init__(self):
        # Data storage
        self.loader = None
        self.available_uids = []
        self.current_uid_idx = 0
        self.current_surface_idx = 0
        self.current_uid_data = None
        
        # Current UID data
        self.surface_cp_list = None
        self.surface_points_list = None
        self.curve_cp_lists = None
        self.curve_points_lists = None
        self.surface_indices = None
        self.surface_types = None
        self.uids = None
        self.file_paths = None
        
        # Counts
        self.total_uids = 0
        self.surfaces_in_current_uid = 0
        
        # Visualization settings
        self.surface_resolution = 32
        self.curve_resolution = 64
        self.show_control_points = True
        self.show_wireframe = False
        self.surface_transparency = 0.8
        self.control_point_size = 0.02
        self.curve_radius = 0.005  # Thickness of curve lines
        self.use_bspline_mode = True  # Toggle between B-spline and direct points mode
        
        # Colors
        self.surface_color = [0.2, 0.8, 0.2]  # Default Green
        self.curve_color = [1.0, 0.0, 0.0]    # Red (not used when gradient is enabled)
        self.control_point_color = [0.0, 0.0, 1.0]  # Blue
        
        # Curve gradient colors
        self.curve_gradient_start = [1.0, 0.5, 0.5]  # Light red
        self.curve_gradient_end = [0.8, 0.0, 0.0]    # Dark red
        self.use_curve_gradient = True  # Toggle for gradient vs solid color
        
        # Surface type color mapping
        self.surface_type_colors = {
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
        self.surface_object = None
        self.curve_objects = []
        self.control_point_objects = []

    def load_distributed_data(self, npz_directory: str) -> bool:
        """Load distributed NPZ data"""
        try:
            self.loader = DistributedNPZLoader(npz_directory)
            self.available_uids = self.loader.available_uids
            self.total_uids = len(self.available_uids)
            
            print(f"✅ Loaded distributed NPZ data:")
            print(f"   Total UIDs: {self.total_uids}")
            
            if self.total_uids > 0:
                self.load_current_uid()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading distributed NPZ files: {e}")
            return False

    def load_current_uid(self) -> bool:
        """Load data for current UID"""
        if self.current_uid_idx >= len(self.available_uids):
            return False
        
        current_uid = self.available_uids[self.current_uid_idx]
        self.current_uid_data = self.loader.load_single_uid(current_uid)
        
        if self.current_uid_data is None:
            return False
        
        # Extract data arrays
        self.surface_cp_list = self.current_uid_data['surface_cp_list']
        self.surface_points_list = self.current_uid_data['surface_points_list']
        self.curve_cp_lists = self.current_uid_data['curve_cp_lists']
        self.curve_points_lists = self.current_uid_data['curve_points_lists']
        self.surface_indices = self.current_uid_data['surface_indices']
        self.surface_types = self.current_uid_data['surface_types']
        self.uids = self.current_uid_data['uids']
        self.file_paths = self.current_uid_data['file_paths']
        
        self.surfaces_in_current_uid = len(self.surface_cp_list)
        self.current_surface_idx = 0  # Reset to first surface
        
        print(f"Loaded UID {current_uid}: {self.surfaces_in_current_uid} surfaces")
        return True

    # Copy B-spline evaluation methods from original visualizer
    def create_bspline_surface_pythonocc(self, control_points: List[List[float]]) -> Optional[Geom_BSplineSurface]:
        """Create B-spline surface using PythonOCC from 4x4 control points"""
        if not PYTHONOCC_AVAILABLE or len(control_points) != 16:
            return None
            
        try:
            poles = TColgp_Array2OfPnt(1, 4, 1, 4)
            
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    pt = control_points[idx]
                    poles.SetValue(i + 1, j + 1, gp_Pnt(pt[0], pt[1], pt[2]))
            
            u_knots = TColStd_Array1OfReal(1, 2)
            v_knots = TColStd_Array1OfReal(1, 2)
            u_knots.SetValue(1, 0.0)
            u_knots.SetValue(2, 1.0)
            v_knots.SetValue(1, 0.0)
            v_knots.SetValue(2, 1.0)
            
            u_mults = TColStd_Array1OfInteger(1, 2)
            v_mults = TColStd_Array1OfInteger(1, 2)
            u_mults.SetValue(1, 4)
            u_mults.SetValue(2, 4)
            v_mults.SetValue(1, 4)
            v_mults.SetValue(2, 4)
            
            surface = Geom_BSplineSurface(poles, u_knots, v_knots, u_mults, v_mults, 3, 3)
            return surface
            
        except Exception as e:
            print(f"Error creating B-spline surface: {e}")
            return None

    def evaluate_bspline_surface_pythonocc(self, surface: Geom_BSplineSurface, resolution: int = 32) -> np.ndarray:
        """Evaluate B-spline surface using PythonOCC"""
        try:
            u_params = np.linspace(0.0, 1.0, resolution)
            v_params = np.linspace(0.0, 1.0, resolution)
            surface_points = np.zeros((resolution, resolution, 3))
            
            for u_idx, u in enumerate(u_params):
                for v_idx, v in enumerate(v_params):
                    point = surface.Value(u, v)
                    surface_points[u_idx, v_idx] = [point.X(), point.Y(), point.Z()]
            
            return surface_points
            
        except Exception as e:
            print(f"Error evaluating B-spline surface: {e}")
            return None

    def create_bspline_curve_pythonocc(self, control_points: List[List[float]]) -> Optional[Geom_BSplineCurve]:
        """Create B-spline curve using PythonOCC from 4 control points"""
        if not PYTHONOCC_AVAILABLE or len(control_points) != 4:
            return None
            
        try:
            poles = TColgp_Array1OfPnt(1, 4)
            for i, pt in enumerate(control_points):
                poles.SetValue(i + 1, gp_Pnt(pt[0], pt[1], pt[2]))
            
            knots = TColStd_Array1OfReal(1, 2)
            knots.SetValue(1, 0.0)
            knots.SetValue(2, 1.0)
            
            mults = TColStd_Array1OfInteger(1, 2)
            mults.SetValue(1, 4)
            mults.SetValue(2, 4)
            
            curve = Geom_BSplineCurve(poles, knots, mults, 3)
            return curve
            
        except Exception as e:
            print(f"Error creating B-spline curve: {e}")
            return None

    def evaluate_bspline_curve_pythonocc(self, curve: Geom_BSplineCurve, resolution: int = 64) -> np.ndarray:
        """Evaluate B-spline curve using PythonOCC"""
        try:
            t_params = np.linspace(0.0, 1.0, resolution)
            curve_points = np.zeros((resolution, 3))
            
            for i, t in enumerate(t_params):
                point = curve.Value(t)
                curve_points[i] = [point.X(), point.Y(), point.Z()]
            
            return curve_points
            
        except Exception as e:
            print(f"Error evaluating B-spline curve: {e}")
            return None

    def fallback_surface_evaluation(self, control_points: List[List[float]], resolution: int = 32) -> np.ndarray:
        """Fallback surface evaluation when PythonOCC is not available"""
        cp = np.array(control_points).reshape(4, 4, 3)
        
        u_params = np.linspace(0, 1, resolution)
        v_params = np.linspace(0, 1, resolution)
        surface_points = np.zeros((resolution, resolution, 3))
        
        for u_idx, u in enumerate(u_params):
            for v_idx, v in enumerate(v_params):
                p00, p01 = cp[0, 0], cp[0, 3]
                p10, p11 = cp[3, 0], cp[3, 3]
                
                p0 = (1 - u) * p00 + u * p10
                p1 = (1 - u) * p01 + u * p11
                point = (1 - v) * p0 + v * p1
                
                surface_points[u_idx, v_idx] = point
        
        return surface_points

    def fallback_curve_evaluation(self, control_points: List[List[float]], resolution: int = 64) -> np.ndarray:
        """Fallback curve evaluation when PythonOCC is not available"""
        cp = np.array(control_points)
        t_params = np.linspace(0, 1, resolution)
        curve_points = np.zeros((resolution, 3))
        
        for i, t in enumerate(t_params):
            if t <= 0.33:
                alpha = t / 0.33
                curve_points[i] = (1 - alpha) * cp[0] + alpha * cp[1]
            elif t <= 0.66:
                alpha = (t - 0.33) / 0.33
                curve_points[i] = (1 - alpha) * cp[1] + alpha * cp[2]
            else:
                alpha = (t - 0.66) / 0.34
                curve_points[i] = (1 - alpha) * cp[2] + alpha * cp[3]
        
        return curve_points

    def create_surface_mesh(self, points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create mesh vertices and faces from surface points"""
        if points is None:
            return None, None
            
        height, width, _ = points.shape
        vertices = points.reshape(-1, 3)
        
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return vertices, np.array(faces)

    def clear_all_objects(self):
        """Clear all polyscope objects"""
        if self.surface_object:
            ps.remove_surface_mesh(self.surface_object)
            self.surface_object = None
            
        for obj_name in self.curve_objects:
            ps.remove_curve_network(obj_name)
        self.curve_objects = []
        
        for obj_name in self.control_point_objects:
            ps.remove_point_cloud(obj_name)
        self.control_point_objects = []

    def change_uid(self, new_uid_idx: int):
        """Change to a different UID"""
        if new_uid_idx < 0 or new_uid_idx >= self.total_uids:
            return
            
        self.current_uid_idx = new_uid_idx
        if self.load_current_uid():
            self.visualize_current_surface()

    def visualize_current_surface(self):
        """Visualize current surface"""
        if (self.current_uid_data is None or 
            self.current_surface_idx >= self.surfaces_in_current_uid):
            return
            
        self.clear_all_objects()
        
        # Get current data
        surface_cp = self.surface_cp_list[self.current_surface_idx]
        surface_points = self.surface_points_list[self.current_surface_idx]
        curves_cp = self.curve_cp_lists[self.current_surface_idx]
        curves_points = self.curve_points_lists[self.current_surface_idx]
        surface_type = self.surface_types[self.current_surface_idx]
        
        current_uid = self.available_uids[self.current_uid_idx]
        
        print(f"\n=== Visualizing UID {current_uid}, Surface {self.current_surface_idx} ===")
        print(f"Surface Type: {surface_type}")
        print(f"Control points: {len(surface_cp) if type(surface_cp) == list else 0}")
        print(f"Direct surface points: {np.array(surface_points).shape if len(surface_points) > 0 else 'None'}")
        print(f"Curves: {len(curves_cp)}")
        print(f"Mode: {'B-spline' if self.use_bspline_mode else 'Direct Points'}")
        
        # Visualize surface
        surface_points_eval = None
        
        if self.use_bspline_mode and len(surface_cp) == 16:  # B-spline mode with 4x4 control points
            if PYTHONOCC_AVAILABLE:
                bspline_surface = self.create_bspline_surface_pythonocc(surface_cp)
                if bspline_surface:
                    surface_points_eval = self.evaluate_bspline_surface_pythonocc(
                        bspline_surface, self.surface_resolution)
                else:
                    surface_points_eval = self.fallback_surface_evaluation(
                        surface_cp, self.surface_resolution)
            else:
                surface_points_eval = self.fallback_surface_evaluation(
                    surface_cp, self.surface_resolution)
        
        elif not self.use_bspline_mode and len(surface_points) > 0:  # Direct points mode
            surface_points_array = np.array(surface_points)
            if surface_points_array.shape == (32, 32, 3):
                surface_points_eval = surface_points_array
            else:
                print(f"Warning: Expected surface points shape (32, 32, 3), got {surface_points_array.shape}")
        
        else:  # Fallback: try the other mode if current mode doesn't work
            if self.use_bspline_mode:
                # B-spline mode failed, try direct points
                if len(surface_points) > 0:
                    surface_points_array = np.array(surface_points)
                    if surface_points_array.shape == (32, 32, 3):
                        surface_points_eval = surface_points_array
                        print("B-spline mode failed, using direct points as fallback")
            else:
                # Direct points mode failed, try B-spline
                if len(surface_cp) == 16:
                    if PYTHONOCC_AVAILABLE:
                        bspline_surface = self.create_bspline_surface_pythonocc(surface_cp)
                        if bspline_surface:
                            surface_points_eval = self.evaluate_bspline_surface_pythonocc(
                                bspline_surface, self.surface_resolution)
                        else:
                            surface_points_eval = self.fallback_surface_evaluation(
                                surface_cp, self.surface_resolution)
                    else:
                        surface_points_eval = self.fallback_surface_evaluation(
                            surface_cp, self.surface_resolution)
                    if surface_points_eval is not None:
                        print("Direct points mode failed, using B-spline as fallback")
        
        # Create surface mesh if we have valid points
        if surface_points_eval is not None:
            vertices, faces = self.create_surface_mesh(surface_points_eval)
            if vertices is not None and faces is not None:
                surface_name = f"surface_{self.current_uid_idx}_{self.current_surface_idx}"
                ps_mesh = ps.register_surface_mesh(surface_name, vertices, faces)
                surface_color = self.surface_type_colors.get(surface_type, self.surface_color)
                ps_mesh.set_color(surface_color)
                ps_mesh.set_transparency(self.surface_transparency)
                if self.show_wireframe:
                    ps_mesh.set_edge_width(1.0)
                self.surface_object = surface_name
        
        # Visualize control points (only in B-spline mode)
        if self.show_control_points and self.use_bspline_mode and len(surface_cp) == 16:
            cp_array = np.array(surface_cp)
            cp_name = f"surface_cp_{self.current_uid_idx}_{self.current_surface_idx}"
            ps_cp = ps.register_point_cloud(cp_name, cp_array)
            ps_cp.set_color(self.control_point_color)
            ps_cp.set_radius(self.control_point_size)
            self.control_point_objects.append(cp_name)
        
        # Visualize curves
        for curve_idx, curve_cp in enumerate(curves_cp):
            curve_points_eval = None
            
            if self.use_bspline_mode and len(curve_cp) == 4:  # B-spline mode with 4 control points
                if PYTHONOCC_AVAILABLE:
                    bspline_curve = self.create_bspline_curve_pythonocc(curve_cp)
                    if bspline_curve:
                        curve_points_eval = self.evaluate_bspline_curve_pythonocc(
                            bspline_curve, self.curve_resolution)
                    else:
                        curve_points_eval = self.fallback_curve_evaluation(
                            curve_cp, self.curve_resolution)
                else:
                    curve_points_eval = self.fallback_curve_evaluation(
                        curve_cp, self.curve_resolution)
            
            elif not self.use_bspline_mode and curve_idx < len(curves_points) and len(curves_points[curve_idx]) > 0:  # Direct points mode
                curve_points_array = np.array(curves_points[curve_idx])
                if len(curve_points_array.shape) == 2 and curve_points_array.shape[1] == 3:
                    # Check if it's the expected (32, 3) shape
                    if curve_points_array.shape[0] == 32:
                        curve_points_eval = curve_points_array
                    else:
                        curve_points_eval = curve_points_array  # Use whatever shape we have
                        print(f"Warning: Expected curve points shape (32, 3), got {curve_points_array.shape}")
            
            else:  # Fallback: try the other mode if current mode doesn't work
                if self.use_bspline_mode:
                    # B-spline mode failed, try direct points
                    if curve_idx < len(curves_points) and len(curves_points[curve_idx]) > 0:
                        curve_points_array = np.array(curves_points[curve_idx])
                        if len(curve_points_array.shape) == 2 and curve_points_array.shape[1] == 3:
                            curve_points_eval = curve_points_array
                else:
                    # Direct points mode failed, try B-spline
                    if len(curve_cp) == 4:
                        if PYTHONOCC_AVAILABLE:
                            bspline_curve = self.create_bspline_curve_pythonocc(curve_cp)
                            if bspline_curve:
                                curve_points_eval = self.evaluate_bspline_curve_pythonocc(
                                    bspline_curve, self.curve_resolution)
                            else:
                                curve_points_eval = self.fallback_curve_evaluation(
                                    curve_cp, self.curve_resolution)
                        else:
                            curve_points_eval = self.fallback_curve_evaluation(
                                curve_cp, self.curve_resolution)
            
            # Create curve visualization if we have valid points
            if curve_points_eval is not None:
                edges = np.array([[i, i + 1] for i in range(len(curve_points_eval) - 1)])
                curve_name = f"curve_{self.current_uid_idx}_{self.current_surface_idx}_{curve_idx}"
                ps_curve = ps.register_curve_network(curve_name, curve_points_eval, edges)
                
                # Apply coloring based on gradient setting
                if self.use_curve_gradient:
                    # Create UV gradient colors
                    num_points = len(curve_points_eval)
                    gradient_colors = np.zeros((num_points, 3))
                    for i in range(num_points):
                        # Parameter t goes from 0 to 1 along the curve
                        t = i / (num_points - 1) if num_points > 1 else 0
                        # Interpolate between start and end colors
                        start_color = np.array(self.curve_gradient_start)
                        end_color = np.array(self.curve_gradient_end)
                        gradient_colors[i] = (1 - t) * start_color + t * end_color
                    
                    # Add the gradient as a color quantity
                    ps_curve.add_color_quantity("gradient", gradient_colors, enabled=True)
                else:
                    # Use solid color
                    ps_curve.set_color(self.curve_color)
                ps_curve.set_radius(self.curve_radius)
                self.curve_objects.append(curve_name)
        
        # Automatically fit camera to the new surface
        ps.reset_camera_to_home_view()

    def create_gui_callback(self):
        """Create GUI callback for polyscope"""
        def gui_callback():
            # UID navigation
            psim.Text("UID Navigation")
            psim.Text(f"Total UIDs: {self.total_uids}")
            
            if self.total_uids > 0:
                current_uid = self.available_uids[self.current_uid_idx] if self.current_uid_idx < len(self.available_uids) else "N/A"
                psim.Text(f"Current UID: {current_uid} ({self.current_uid_idx + 1}/{self.total_uids})")
                psim.Text(f"Surfaces in UID: {self.surfaces_in_current_uid}")
                
                # UID navigation Buttons
                if psim.Button("Previous UID"):
                    new_idx = max(0, self.current_uid_idx - 1)
                    if new_idx != self.current_uid_idx:
                        self.change_uid(new_idx)
                
                psim.SameLine()
                if psim.Button("Next UID"):
                    new_idx = min(self.total_uids - 1, self.current_uid_idx + 1)
                    if new_idx != self.current_uid_idx:
                        self.change_uid(new_idx)
                
                # UID slider
                changed, new_uid_idx = psim.SliderInt("UID Index", self.current_uid_idx, 0, self.total_uids - 1)
                if changed:
                    self.change_uid(new_uid_idx)
                
                psim.Separator()
                
                # Surface navigation within current UID
                psim.Text("Surface Navigation")
                if self.surfaces_in_current_uid > 0:
                    surface_type = self.surface_types[self.current_surface_idx] if self.current_surface_idx < len(self.surface_types) else "N/A"
                    num_curves = len(self.curve_cp_lists[self.current_surface_idx]) if self.current_surface_idx < len(self.curve_cp_lists) else 0
                    
                    # Surface data availability info
                    surface_cp = self.surface_cp_list[self.current_surface_idx] if self.current_surface_idx < len(self.surface_cp_list) else []
                    surface_points = self.surface_points_list[self.current_surface_idx] if self.current_surface_idx < len(self.surface_points_list) else []
                    has_bspline_data = len(surface_cp) == 16
                    has_direct_points = len(surface_points) > 0 and np.array(surface_points).shape == (32, 32, 3)
                    
                    psim.Text(f"Current Surface: {self.current_surface_idx + 1}/{self.surfaces_in_current_uid}")
                    psim.Text(f"Surface Type: {surface_type}")
                    psim.Text(f"Curves: {num_curves}")
                    
                    # Data availability indicators
                    psim.Text("Data Availability:")
                    bspline_color = (0.0, 1.0, 0.0, 1.0) if has_bspline_data else (1.0, 0.0, 0.0, 1.0)
                    direct_color = (0.0, 1.0, 0.0, 1.0) if has_direct_points else (1.0, 0.0, 0.0, 1.0)
                    psim.TextColored(bspline_color, f"  B-spline CP: {'✓' if has_bspline_data else '✗'}")
                    psim.TextColored(direct_color, f"  Direct Points: {'✓' if has_direct_points else '✗'}")
                    
                    # Current mode indicator
                    mode_text = "B-spline Mode" if self.use_bspline_mode else "Direct Points Mode"
                    mode_color = (0.0, 0.8, 1.0, 1.0)  # Cyan with alpha
                    psim.TextColored(mode_color, f"Current Mode: {mode_text}")
                    
                    # Surface navigation Buttons
                    if psim.Button("Previous Surface"):
                        new_idx = max(0, self.current_surface_idx - 1)
                        if new_idx != self.current_surface_idx:
                            self.current_surface_idx = new_idx
                            self.visualize_current_surface()
                    
                    psim.SameLine()
                    if psim.Button("Next Surface"):
                        new_idx = min(self.surfaces_in_current_uid - 1, self.current_surface_idx + 1)
                        if new_idx != self.current_surface_idx:
                            self.current_surface_idx = new_idx
                            self.visualize_current_surface()
                    
                    # Surface slider
                    changed, new_surface_idx = psim.SliderInt("Surface Index", self.current_surface_idx, 0, self.surfaces_in_current_uid - 1)
                    if changed:
                        self.current_surface_idx = new_surface_idx
                        self.visualize_current_surface()
                
                psim.Separator()
                
                # Visualization settings
                psim.Text("Visualization Settings")
                
                # Visualization mode toggle
                changed, self.use_bspline_mode = psim.Checkbox("Use B-spline Mode", self.use_bspline_mode)
                if changed:
                    self.visualize_current_surface()
                
                psim.SameLine()
                psim.Text("(Unchecked = Direct Points Mode)")
                
                changed, self.show_control_points = psim.Checkbox("Show Control Points", self.show_control_points)
                if changed:
                    self.visualize_current_surface()
                
                changed, self.show_wireframe = psim.Checkbox("Show Wireframe", self.show_wireframe)
                if changed:
                    self.visualize_current_surface()
                
                changed, self.surface_transparency = psim.SliderFloat("Surface Transparency", self.surface_transparency, 0.0, 1.0)
                if changed:
                    if self.surface_object:
                        ps.get_surface_mesh(self.surface_object).set_transparency(self.surface_transparency)
                
                changed, self.control_point_size = psim.SliderFloat("Control Point Size", self.control_point_size, 0.001, 0.1)
                if changed:
                    for obj_name in self.control_point_objects:
                        ps.get_point_cloud(obj_name).set_radius(self.control_point_size)
                
                changed, self.curve_radius = psim.SliderFloat("Curve Thickness", self.curve_radius, 0.001, 0.05)
                if changed:
                    for obj_name in self.curve_objects:
                        ps.get_curve_network(obj_name).set_radius(self.curve_radius)
                
                # Curve gradient settings
                changed, self.use_curve_gradient = psim.Checkbox("Use Curve Gradient", self.use_curve_gradient)
                if changed:
                    self.visualize_current_surface()
                
                if self.use_curve_gradient:
                    psim.Text("Gradient Colors:")
                    changed, self.curve_gradient_start = psim.ColorEdit3("Start Color (Light)", self.curve_gradient_start)
                    if changed:
                        self.visualize_current_surface()
                    
                    changed, self.curve_gradient_end = psim.ColorEdit3("End Color (Dark)", self.curve_gradient_end)
                    if changed:
                        self.visualize_current_surface()
                else:
                    changed, self.curve_color = psim.ColorEdit3("Curve Color", self.curve_color)
                    if changed:
                        self.visualize_current_surface()
                
                psim.Separator()
                
                # Resolution settings
                psim.Text("Resolution Settings")
                
                changed, self.surface_resolution = psim.SliderInt("Surface Resolution", self.surface_resolution, 8, 64)
                if changed:
                    self.visualize_current_surface()
                
                changed, self.curve_resolution = psim.SliderInt("Curve Resolution", self.curve_resolution, 16, 128)
                if changed:
                    self.visualize_current_surface()
                
                psim.Separator()
                
                # Visualization mode explanation
                psim.Text("Visualization Modes:")
                psim.Text("  B-spline Mode: Uses 4x4 control points to reconstruct surfaces")
                psim.Text("                 and 4 control points for curves via B-spline evaluation")
                psim.Text("  Direct Points: Uses pre-computed (32,32,3) surface points")
                psim.Text("                 and (32,3) curve points directly")
                
                psim.Separator()
                
                # Surface type color legend
                psim.Text("Surface Type Colors:")
                for surf_type, color in self.surface_type_colors.items():
                    psim.Text(f"  {surf_type}")
                    psim.SameLine()
                    psim.Text(f"RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        
        return gui_callback

    def run(self, npz_directory: str):
        """Main execution function"""
        # Load data
        if not self.load_distributed_data(npz_directory):
            return
        
        # Initialize polyscope
        ps.init()
        ps.set_user_callback(self.create_gui_callback())
        
        # Visualize first surface if available
        if self.total_uids > 0 and self.surfaces_in_current_uid > 0:
            self.visualize_current_surface()
        
        # Start polyscope
        ps.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize individual surfaces from distributed NPZ files")
    parser.add_argument("--input", type=str, required=True, help="Directory containing UID.npz files")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"❌ Input directory does not exist: {args.input}")
        return
    
    visualizer = DistributedSurfaceVisualizer()
    visualizer.run(args.input)


if __name__ == "__main__":
    main() 