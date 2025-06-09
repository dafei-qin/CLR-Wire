#!/usr/bin/env python3
"""
Individual surface visualizer that loads NPZ files and visualizes one surface with its related curves.
Supports changing surface index through polyscope UI.
"""

import argparse
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import List, Dict, Any, Tuple, Optional
import os

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


class IndividualSurfaceVisualizer:
    def __init__(self):
        # Data storage
        self.npz_data = None
        self.surface_cp_list = None
        self.surface_points_list = None
        self.curve_cp_lists = None
        self.curve_points_lists = None
        self.surface_indices = None
        self.surface_types = None
        self.uids = None
        self.file_paths = None
        
        # Current state
        self.current_surface_idx = 0
        self.total_surfaces = 0
        
        # Visualization settings
        self.surface_resolution = 32
        self.curve_resolution = 64
        self.show_control_points = True
        self.show_wireframe = False
        self.surface_transparency = 0.8
        self.control_point_size = 0.02
        
        # Colors
        self.surface_color = [0.2, 0.8, 0.2]  # Default Green
        self.curve_color = [1.0, 0.0, 0.0]    # Red
        self.control_point_color = [0.0, 0.0, 1.0]  # Blue
        
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

    def load_npz_data(self, npz_path: str) -> bool:
        """Load NPZ data file"""
        try:
            self.npz_data = np.load(npz_path, allow_pickle=True)
            
            self.surface_cp_list = self.npz_data['surface_cp_list']
            self.surface_points_list = self.npz_data['surface_points_list']
            self.curve_cp_lists = self.npz_data['curve_cp_lists']
            self.curve_points_lists = self.npz_data['curve_points_lists']
            self.surface_indices = self.npz_data['surface_indices']
            self.surface_types = self.npz_data['surface_types']
            self.uids = self.npz_data['uids']
            self.file_paths = self.npz_data['file_paths']
            
            self.total_surfaces = len(self.surface_cp_list)
            
            print(f"✅ Loaded NPZ data:")
            print(f"   Total surfaces: {self.total_surfaces}")
            print(f"   Keys: {list(self.npz_data.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading NPZ file: {e}")
            return False

    def create_bspline_surface_pythonocc(self, control_points: List[List[float]]) -> Optional[Geom_BSplineSurface]:
        """Create B-spline surface using PythonOCC from 4x4 control points"""
        if not PYTHONOCC_AVAILABLE or len(control_points) != 16:
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
            u_knots = TColStd_Array1OfReal(1, 2)
            v_knots = TColStd_Array1OfReal(1, 2)
            u_knots.SetValue(1, 0.0)
            u_knots.SetValue(2, 1.0)
            v_knots.SetValue(1, 0.0)
            v_knots.SetValue(2, 1.0)
            
            # Define multiplicities
            u_mults = TColStd_Array1OfInteger(1, 2)
            v_mults = TColStd_Array1OfInteger(1, 2)
            u_mults.SetValue(1, 4)
            u_mults.SetValue(2, 4)
            v_mults.SetValue(1, 4)
            v_mults.SetValue(2, 4)
            
            # Create B-spline surface (degree 3 in both directions)
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
            # Convert control points to TColgp_Array1OfPnt (1-indexed)
            poles = TColgp_Array1OfPnt(1, 4)
            for i, pt in enumerate(control_points):
                poles.SetValue(i + 1, gp_Pnt(pt[0], pt[1], pt[2]))
            
            # Define knot vector for cubic B-spline curve (degree 3)
            knots = TColStd_Array1OfReal(1, 2)
            knots.SetValue(1, 0.0)
            knots.SetValue(2, 1.0)
            
            # Define multiplicities
            mults = TColStd_Array1OfInteger(1, 2)
            mults.SetValue(1, 4)
            mults.SetValue(2, 4)
            
            # Create B-spline curve (degree 3)
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
        # Simple bilinear interpolation for demonstration
        # In practice, you'd implement proper B-spline evaluation
        cp = np.array(control_points).reshape(4, 4, 3)
        
        u_params = np.linspace(0, 1, resolution)
        v_params = np.linspace(0, 1, resolution)
        surface_points = np.zeros((resolution, resolution, 3))
        
        for u_idx, u in enumerate(u_params):
            for v_idx, v in enumerate(v_params):
                # Bilinear interpolation between corner points
                p00, p01 = cp[0, 0], cp[0, 3]
                p10, p11 = cp[3, 0], cp[3, 3]
                
                p0 = (1 - u) * p00 + u * p10
                p1 = (1 - u) * p01 + u * p11
                point = (1 - v) * p0 + v * p1
                
                surface_points[u_idx, v_idx] = point
        
        return surface_points

    def fallback_curve_evaluation(self, control_points: List[List[float]], resolution: int = 64) -> np.ndarray:
        """Fallback curve evaluation when PythonOCC is not available"""
        # Simple linear interpolation between control points
        cp = np.array(control_points)
        t_params = np.linspace(0, 1, resolution)
        curve_points = np.zeros((resolution, 3))
        
        for i, t in enumerate(t_params):
            # Simple interpolation through control points
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
        
        # Flatten points to create vertices
        vertices = points.reshape(-1, 3)
        
        # Create faces (triangles)
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                # Current quad vertices
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
                
                # Add two triangles for each quad
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

    def visualize_surface(self, surface_idx: int):
        """Visualize a specific surface with its curves"""
        if surface_idx < 0 or surface_idx >= self.total_surfaces:
            return
            
        self.clear_all_objects()
        
        # Get surface data
        surface_cp = self.surface_cp_list[surface_idx]
        surface_points = self.surface_points_list[surface_idx]
        curves_cp = self.curve_cp_lists[surface_idx]
        curves_points = self.curve_points_lists[surface_idx]
        uid = self.uids[surface_idx]
        surface_index = self.surface_indices[surface_idx]
        surface_type = self.surface_types[surface_idx]
        
        print(f"\n=== Visualizing Surface {surface_idx} ===")
        print(f"UID: {uid}")
        print(f"Surface Index: {surface_index}")
        print(f"Surface Type: {surface_type}")
        print(f"Control points: {len(surface_cp) if surface_cp else 0}")
        print(f"Curves: {len(curves_cp)}")
        
        # Visualize surface
        if len(surface_cp) == 16:  # 4x4 control points
            # Try PythonOCC first
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
                vertices, faces = self.create_surface_mesh(surface_points_eval)
                if vertices is not None and faces is not None:
                    surface_name = f"surface_{surface_idx}"
                    ps_mesh = ps.register_surface_mesh(surface_name, vertices, faces)
                    # Set color based on surface type
                    surface_color = self.surface_type_colors.get(surface_type, self.surface_color)
                    ps_mesh.set_color(surface_color)
                    ps_mesh.set_transparency(self.surface_transparency)
                    if self.show_wireframe:
                        ps_mesh.set_edge_width(1.0)
                    self.surface_object = surface_name
        
        elif len(surface_points) > 0:
            # Use pre-computed surface points
            surface_points_array = np.array(surface_points)
            if surface_points_array.shape == (32, 32, 3):
                vertices, faces = self.create_surface_mesh(surface_points_array)
                if vertices is not None and faces is not None:
                    surface_name = f"surface_{surface_idx}"
                    ps_mesh = ps.register_surface_mesh(surface_name, vertices, faces)
                    # Set color based on surface type
                    surface_color = self.surface_type_colors.get(surface_type, self.surface_color)
                    ps_mesh.set_color(surface_color)
                    ps_mesh.set_transparency(self.surface_transparency)
                    if self.show_wireframe:
                        ps_mesh.set_edge_width(1.0)
                    self.surface_object = surface_name
        
        # Visualize control points
        if self.show_control_points and len(surface_cp) == 16:
            cp_array = np.array(surface_cp)
            cp_name = f"surface_cp_{surface_idx}"
            ps_cp = ps.register_point_cloud(cp_name, cp_array)
            ps_cp.set_color(self.control_point_color)
            ps_cp.set_radius(self.control_point_size)
            self.control_point_objects.append(cp_name)
        
        # Visualize curves
        for curve_idx, curve_cp in enumerate(curves_cp):
            if len(curve_cp) == 4:  # 4 control points for cubic curve
                # Try PythonOCC first
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
                
                if curve_points_eval is not None:
                    # Create curve edges
                    edges = np.array([[i, i + 1] for i in range(len(curve_points_eval) - 1)])
                    curve_name = f"curve_{surface_idx}_{curve_idx}"
                    ps_curve = ps.register_curve_network(curve_name, curve_points_eval, edges)
                    ps_curve.set_color(self.curve_color)
                    ps_curve.set_radius(0.005)
                    self.curve_objects.append(curve_name)
            
            elif curve_idx < len(curves_points) and len(curves_points[curve_idx]) > 0:
                # Use pre-computed curve points
                curve_points_array = np.array(curves_points[curve_idx])
                if len(curve_points_array.shape) == 2 and curve_points_array.shape[1] == 3:
                    edges = np.array([[i, i + 1] for i in range(len(curve_points_array) - 1)])
                    curve_name = f"curve_{surface_idx}_{curve_idx}"
                    ps_curve = ps.register_curve_network(curve_name, curve_points_array, edges)
                    ps_curve.set_color(self.curve_color)
                    ps_curve.set_radius(0.005)
                    self.curve_objects.append(curve_name)
        
        # Update current index
        self.current_surface_idx = surface_idx

    def create_gui_callback(self):
        """Create GUI callback for polyscope"""
        def gui_callback():
            # Surface navigation
            psim.text("Surface Navigation")
            psim.text(f"Total surfaces: {self.total_surfaces}")
            
            if self.total_surfaces > 0:
                # Current surface info
                uid = self.uids[self.current_surface_idx] if self.current_surface_idx < len(self.uids) else "N/A"
                surface_index = self.surface_indices[self.current_surface_idx] if self.current_surface_idx < len(self.surface_indices) else "N/A"
                surface_type = self.surface_types[self.current_surface_idx] if self.current_surface_idx < len(self.surface_types) else "N/A"
                num_curves = len(self.curve_cp_lists[self.current_surface_idx]) if self.current_surface_idx < len(self.curve_cp_lists) else 0
                
                psim.text(f"Current: {self.current_surface_idx}")
                psim.text(f"UID: {uid}")
                psim.text(f"Surface Index: {surface_index}")
                psim.text(f"Surface Type: {surface_type}")
                psim.text(f"Curves: {num_curves}")
                
                psim.separator()
                
                # Navigation buttons
                if psim.button("Previous Surface"):
                    new_idx = max(0, self.current_surface_idx - 1)
                    if new_idx != self.current_surface_idx:
                        self.visualize_surface(new_idx)
                
                psim.same_line()
                if psim.button("Next Surface"):
                    new_idx = min(self.total_surfaces - 1, self.current_surface_idx + 1)
                    if new_idx != self.current_surface_idx:
                        self.visualize_surface(new_idx)
                
                # Direct surface index input
                psim.separator()
                changed, new_idx = psim.slider_int("Surface Index", self.current_surface_idx, 0, self.total_surfaces - 1)
                if changed:
                    self.visualize_surface(new_idx)
                
                # Jump buttons for large datasets
                psim.separator()
                if psim.button("Jump -100"):
                    new_idx = max(0, self.current_surface_idx - 100)
                    self.visualize_surface(new_idx)
                
                psim.same_line()
                if psim.button("Jump +100"):
                    new_idx = min(self.total_surfaces - 1, self.current_surface_idx + 100)
                    self.visualize_surface(new_idx)
                
            psim.separator()
            
            # Visualization settings
            psim.text("Visualization Settings")
            
            changed, self.show_control_points = psim.checkbox("Show Control Points", self.show_control_points)
            if changed:
                self.visualize_surface(self.current_surface_idx)
            
            changed, self.show_wireframe = psim.checkbox("Show Wireframe", self.show_wireframe)
            if changed:
                self.visualize_surface(self.current_surface_idx)
            
            changed, self.surface_transparency = psim.slider_float("Surface Transparency", self.surface_transparency, 0.0, 1.0)
            if changed:
                if self.surface_object:
                    ps.get_surface_mesh(self.surface_object).set_transparency(self.surface_transparency)
            
            changed, self.control_point_size = psim.slider_float("Control Point Size", self.control_point_size, 0.001, 0.1)
            if changed:
                for obj_name in self.control_point_objects:
                    ps.get_point_cloud(obj_name).set_radius(self.control_point_size)
            
            psim.separator()
            
            # Resolution settings
            psim.text("Resolution Settings")
            
            changed, self.surface_resolution = psim.slider_int("Surface Resolution", self.surface_resolution, 8, 64)
            if changed:
                self.visualize_surface(self.current_surface_idx)
            
            changed, self.curve_resolution = psim.slider_int("Curve Resolution", self.curve_resolution, 16, 128)
            if changed:
                self.visualize_surface(self.current_surface_idx)
            
            psim.separator()
            
            # Surface type color legend
            psim.text("Surface Type Colors:")
            for surf_type, color in self.surface_type_colors.items():
                # Create a small colored square indicator
                psim.text(f"  {surf_type}")
                psim.same_line()
                # Show color as RGB values since we can't draw colored squares easily
                psim.text(f"RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        
        return gui_callback

    def run(self, npz_path: str):
        """Main execution function"""
        # Load data
        if not self.load_npz_data(npz_path):
            return
        
        # Initialize polyscope
        ps.init()
        ps.set_user_callback(self.create_gui_callback())
        
        # Visualize first surface
        if self.total_surfaces > 0:
            self.visualize_surface(0)
        
        # Start polyscope
        ps.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize individual surfaces with curves from NPZ file")
    parser.add_argument("--input", type=str, required=True, help="Input NPZ file path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        return
    
    visualizer = IndividualSurfaceVisualizer()
    visualizer.run(args.input)


if __name__ == "__main__":
    main() 