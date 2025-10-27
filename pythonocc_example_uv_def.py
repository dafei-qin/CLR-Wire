#!/usr/bin/env python3
"""
Interactive UV Parameter Visualizer for Different Surface Types
This script demonstrates how UV parameters define different surface types
and allows interactive adjustment of UV bounds.
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from typing import Dict, Any, Optional, Tuple, List
import sys

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Ax3, gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus
from OCC.Core.Geom import Geom_BSplineSurface, Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface, Geom_ToroidalSurface
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface

from occwl.face import Face

class UVParameterVisualizer:
    def __init__(self):
        # Surface type definitions
        self.surface_types = [
            "plane",
            "cylinder", 
            "cone",
            "sphere",
            "torus",
            "bspline_surface"
        ]
        
        # Current surface selection
        self.current_surface_type = "cylinder"
        self.current_surface_index = 1  # Index in surface_types list
        
        # UV parameter ranges for each surface type
        # Using ±1e5 for all ranges to allow maximum flexibility
        two_pi = 2.0 * np.pi
        param_range = 1e5
        
        self.uv_params = {
            "plane": {
                "u_min": 0.0, "u_max": 10.0,
                "v_min": 0.0, "v_max": 10.0,
                "u_range": [-param_range, param_range],  # Allowable range
                "v_range": [-param_range, param_range]
            },
            "cylinder": {
                "u_min": 0.0, "u_max": two_pi,  # 0 to 2*pi
                "v_min": 0.0, "v_max": 10.0,
                "u_range": [-param_range, param_range],
                "v_range": [-param_range, param_range]
            },
            "cone": {
                "u_min": 0.0, "u_max": two_pi,
                "v_min": 0.0, "v_max": 10.0,
                "u_range": [-param_range, param_range],
                "v_range": [-param_range, param_range]
            },
            "sphere": {
                "u_min": 0.0, "u_max": two_pi,  # 0 to 2*pi (longitude)
                "v_min": -np.pi/2, "v_max": np.pi/2,  # -pi/2 to pi/2 (latitude)
                "u_range": [-param_range, param_range],
                "v_range": [-param_range, param_range]
            },
            "torus": {
                "u_min": 0.0, "u_max": two_pi,
                "v_min": 0.0, "v_max": two_pi,
                "u_range": [-param_range, param_range],
                "v_range": [-param_range, param_range]
            },
            "bspline_surface": {
                "u_min": 0.0, "u_max": 1.0,
                "v_min": 0.0, "v_max": 1.0,
                "u_range": [-param_range, param_range],
                "v_range": [-param_range, param_range]
            }
        }
        
        # Geometric parameters for each surface
        self.surface_params = {
            "plane": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "x_direction": [1.0, 0.0, 0.0]
            },
            "cylinder": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "x_direction": [1.0, 0.0, 0.0],
                "radius": 5.0
            },
            "cone": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "x_direction": [1.0, 0.0, 0.0],
                "radius": 5.0,
                "semi_angle": 0.5  # radians (~28.6 degrees)
            },
            "sphere": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "x_direction": [1.0, 0.0, 0.0],
                "radius": 8.0
            },
            "torus": {
                "position": [0.0, 0.0, 0.0],
                "direction": [0.0, 0.0, 1.0],
                "x_direction": [1.0, 0.0, 0.0],
                "major_radius": 8.0,
                "minor_radius": 3.0
            },
            "bspline_surface": {
                "position": [0.0, 0.0, 0.0]  # Offset for B-spline control points
            }
        }
        
        # Visualization settings
        self.mesh_quality = 0.1  # Smaller = finer mesh
        self.show_uv_grid = True
        self.uv_grid_density_u = 8
        self.uv_grid_density_v = 8
        self.show_surface = True
        self.surface_transparency = 0.5
        self.show_reference_cube = True
        
        # Polyscope objects
        self.surface_mesh_obj = None
        self.uv_grid_obj = None
        self.uv_grid_points_obj = None
        self.reference_cube_obj = None
        self.coordinate_frame_obj = None
        self.position_point_obj = None
        
        # Visualization settings for coordinate frame
        self.show_coordinate_frame = True
        self.coordinate_frame_scale = 5.0
        
        # Visualization settings for position point
        self.show_position_point = True
        self.position_point_radius = 0.1
        
        # Curvature analysis settings
        self.show_curvature_analysis = True
        self.curvature_grid_size = 8
        self.max_curvature_value = 0.0
        self.curvature_points_obj = None
        self.curvature_values = None
        
    def normalize_vector(self, vec):
        """Normalize a vector to unit length"""
        vec = np.array(vec, dtype=np.float64)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            # If vector is too small, return a default
            return np.array([0.0, 0.0, 1.0])
        return vec / norm
    
    def orthogonalize_vectors(self, direction, x_direction):
        """
        Orthogonalize direction and x_direction vectors.
        - Normalize direction
        - Project x_direction to be perpendicular to direction
        - Calculate y_direction as cross product
        
        Returns:
            tuple: (normalized_direction, orthogonal_x_direction, y_direction)
        """
        # Normalize direction (Z-axis of local frame)
        dir_normalized = self.normalize_vector(direction)
        
        # Project x_direction to be perpendicular to direction
        x_dir = np.array(x_direction, dtype=np.float64)
        
        # Remove component parallel to direction: x_dir_perp = x_dir - (x_dir · dir) * dir
        x_dir_perp = x_dir - np.dot(x_dir, dir_normalized) * dir_normalized
        
        # Normalize the perpendicular component
        x_dir_normalized = self.normalize_vector(x_dir_perp)
        
        # Calculate y_direction as cross product: y = z × x
        y_dir = np.cross(dir_normalized, x_dir_normalized)
        y_dir_normalized = self.normalize_vector(y_dir)
        
        return dir_normalized, x_dir_normalized, y_dir_normalized
    
    def update_surface_coordinate_system(self, surface_type):
        """Update the coordinate system for a surface to ensure orthogonality"""
        if surface_type not in self.surface_params:
            return
        
        params = self.surface_params[surface_type]
        if "direction" in params and "x_direction" in params:
            direction = params["direction"]
            x_direction = params["x_direction"]
            
            # Orthogonalize
            dir_norm, x_dir_norm, y_dir_norm = self.orthogonalize_vectors(direction, x_direction)
            
            # Update the parameters
            params["direction"] = dir_norm.tolist()
            params["x_direction"] = x_dir_norm.tolist()
            
            # Store Y direction for display (not used in construction but useful for info)
            params["y_direction"] = y_dir_norm.tolist()
    
    def create_coordinate_frame_visualization(self):
        """Create visualization of the local coordinate frame at the surface position"""
        if self.current_surface_type not in self.surface_params:
            return None
        
        params = self.surface_params[self.current_surface_type]
        
        if "position" not in params or "direction" not in params or "x_direction" not in params:
            return None
        
        position = np.array(params["position"])
        
        # Ensure coordinate system is orthogonal
        self.update_surface_coordinate_system(self.current_surface_type)
        
        x_dir = np.array(params["x_direction"])
        direction = np.array(params["direction"])
        y_dir = np.array(params.get("y_direction", np.cross(direction, x_dir)))
        
        # Create three arrows from position
        points = []
        edges = []
        
        scale = self.coordinate_frame_scale
        
        # X axis (red) - starts at position
        points.append(position)
        points.append(position + x_dir * scale)
        edges.append([0, 1])
        
        # Y axis (green)
        points.append(position)
        points.append(position + y_dir * scale)
        edges.append([2, 3])
        
        # Z axis (blue) - direction
        points.append(position)
        points.append(position + direction * scale)
        edges.append([4, 5])
        
        return np.array(points), np.array(edges)
    
    def create_reference_cube(self):
        """Create a reference cube with left bottom corner at (0,0,0) and size 1"""
        # Define the 8 vertices of the cube
        vertices = np.array([
            [0.0, 0.0, 0.0],  # 0: origin
            [1.0, 0.0, 0.0],  # 1: +x
            [1.0, 1.0, 0.0],  # 2: +x, +y
            [0.0, 1.0, 0.0],  # 3: +y
            [0.0, 0.0, 1.0],  # 4: +z
            [1.0, 0.0, 1.0],  # 5: +x, +z
            [1.0, 1.0, 1.0],  # 6: +x, +y, +z
            [0.0, 1.0, 1.0],  # 7: +y, +z
        ])
        
        # Define the 12 triangular faces (2 per cube face)
        faces = np.array([
            # Bottom face (z=0)
            [0, 1, 2], [0, 2, 3],
            # Top face (z=1)
            [4, 6, 5], [4, 7, 6],
            # Front face (y=0)
            [0, 5, 1], [0, 4, 5],
            # Back face (y=1)
            [3, 2, 6], [3, 6, 7],
            # Left face (x=0)
            [0, 3, 7], [0, 7, 4],
            # Right face (x=1)
            [1, 5, 6], [1, 6, 2],
        ])
        
        return vertices, faces
    
    def extract_mesh_from_face(self, face):
        """Extract mesh vertices and faces from a TopoDS_Face"""
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        
        if triangulation is None:
            return None, None
        
        # Get transformation
        trsf = location.Transformation()
        
        # Extract vertices
        num_nodes = triangulation.NbNodes()
        num_triangles = triangulation.NbTriangles()
        
        vertices = []
        for i in range(1, num_nodes + 1):
            local_pnt = triangulation.Node(i)
            global_pnt = local_pnt.Transformed(trsf)
            vertices.append([global_pnt.X(), global_pnt.Y(), global_pnt.Z()])
        
        # Extract faces
        faces = []
        for i in range(1, num_triangles + 1):
            triangle = triangulation.Triangle(i)
            v1_idx, v2_idx, v3_idx = triangle.Get()
            faces.append([v1_idx - 1, v2_idx - 1, v3_idx - 1])
        
        return np.array(vertices), np.array(faces)
    
    def build_plane_surface(self):
        """Build a plane surface with current UV parameters"""
        params = self.surface_params["plane"]
        uv = self.uv_params["plane"]
        
        position = gp_Pnt(params["position"][0], params["position"][1], params["position"][2])
        direction = gp_Dir(params["direction"][0], params["direction"][1], params["direction"][2])
        x_direction = gp_Dir(params["x_direction"][0], params["x_direction"][1], params["x_direction"][2])
        
        ax3 = gp_Ax3(position, direction, x_direction)
        plane = gp_Pln(ax3)
        
        face_builder = BRepBuilderAPI_MakeFace(plane, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"])
        face = face_builder.Face()
        
        # Get the underlying Geom_Surface for UV evaluation
        geom_surface = BRep_Tool.Surface(face)
        
        return face, geom_surface
    
    def build_cylinder_surface(self):
        """Build a cylindrical surface with current UV parameters"""
        params = self.surface_params["cylinder"]
        uv = self.uv_params["cylinder"]
        
        position = gp_Pnt(params["position"][0], params["position"][1], params["position"][2])
        direction = gp_Dir(params["direction"][0], params["direction"][1], params["direction"][2])
        x_direction = gp_Dir(params["x_direction"][0], params["x_direction"][1], params["x_direction"][2])
        
        ax3 = gp_Ax3(position, direction, x_direction)
        cylinder = gp_Cylinder(ax3, params["radius"])
        
        face_builder = BRepBuilderAPI_MakeFace(cylinder, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"])
        face = face_builder.Face()
        
        geom_surface = BRep_Tool.Surface(face)
        
        return face, geom_surface
    
    def build_cone_surface(self):
        """Build a conical surface with current UV parameters"""
        params = self.surface_params["cone"]
        uv = self.uv_params["cone"]
        
        position = gp_Pnt(params["position"][0], params["position"][1], params["position"][2])
        direction = gp_Dir(params["direction"][0], params["direction"][1], params["direction"][2])
        x_direction = gp_Dir(params["x_direction"][0], params["x_direction"][1], params["x_direction"][2])
        
        ax3 = gp_Ax3(position, direction, x_direction)
        cone = gp_Cone(ax3, params["semi_angle"], params["radius"])
        
        face_builder = BRepBuilderAPI_MakeFace(cone, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"])
        face = face_builder.Face()
        
        geom_surface = BRep_Tool.Surface(face)
        
        return face, geom_surface
    
    def build_sphere_surface(self):
        """Build a spherical surface with current UV parameters"""
        params = self.surface_params["sphere"]
        uv = self.uv_params["sphere"]
        
        position = gp_Pnt(params["position"][0], params["position"][1], params["position"][2])
        direction = gp_Dir(params["direction"][0], params["direction"][1], params["direction"][2])
        x_direction = gp_Dir(params["x_direction"][0], params["x_direction"][1], params["x_direction"][2])
        
        ax3 = gp_Ax3(position, direction, x_direction)
        sphere = gp_Sphere(ax3, params["radius"])
        
        face_builder = BRepBuilderAPI_MakeFace(sphere, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"])
        face = face_builder.Face()
        
        geom_surface = BRep_Tool.Surface(face)
        
        return face, geom_surface
    
    def build_torus_surface(self):
        """Build a toroidal surface with current UV parameters"""
        params = self.surface_params["torus"]
        uv = self.uv_params["torus"]
        
        position = gp_Pnt(params["position"][0], params["position"][1], params["position"][2])
        direction = gp_Dir(params["direction"][0], params["direction"][1], params["direction"][2])
        x_direction = gp_Dir(params["x_direction"][0], params["x_direction"][1], params["x_direction"][2])
        
        ax3 = gp_Ax3(position, direction, x_direction)
        torus = gp_Torus(ax3, params["major_radius"], params["minor_radius"])
        
        face_builder = BRepBuilderAPI_MakeFace(torus, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"])
        face = face_builder.Face()
        
        geom_surface = BRep_Tool.Surface(face)
        
        return face, geom_surface
    
    def build_bspline_surface(self):
        """Build a B-spline surface with current UV parameters"""
        params = self.surface_params["bspline_surface"]
        uv = self.uv_params["bspline_surface"]
        
        # Get position offset
        offset = params["position"]
        
        # Create a 4x4 control point grid for a simple B-spline surface
        # This creates a wavy surface for demonstration
        control_points = TColgp_Array2OfPnt(1, 4, 1, 4)
        for i in range(4):
            for j in range(4):
                x = (i - 1.5) * 5.0 + offset[0]
                y = (j - 1.5) * 5.0 + offset[1]
                z = np.sin(i * 0.8) * np.cos(j * 0.8) * 3.0 + offset[2]
                control_points.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))
        
        # Define knot vectors (clamped cubic B-spline)
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
        
        # Create B-spline surface
        bspline_surface = Geom_BSplineSurface(control_points, u_knots, v_knots, u_mults, v_mults, 3, 3)
        
        # Create face from B-spline surface
        face_builder = BRepBuilderAPI_MakeFace(bspline_surface, uv["u_min"], uv["u_max"], uv["v_min"], uv["v_max"], 1e-6)
        face = face_builder.Face()
        
        return face, bspline_surface
    
    def build_current_surface(self):
        """Build the currently selected surface type"""
        if self.current_surface_type == "plane":
            return self.build_plane_surface()
        elif self.current_surface_type == "cylinder":
            return self.build_cylinder_surface()
        elif self.current_surface_type == "cone":
            return self.build_cone_surface()
        elif self.current_surface_type == "sphere":
            return self.build_sphere_surface()
        elif self.current_surface_type == "torus":
            return self.build_torus_surface()
        elif self.current_surface_type == "bspline_surface":
            return self.build_bspline_surface()
        else:
            return None, None
    
    def compute_curvature_on_grid(self, face, uv_params):
        """Compute max curvature on a UV grid using occwl.face.Face"""
        u_min = uv_params["u_min"]
        u_max = uv_params["u_max"]
        v_min = uv_params["v_min"]
        v_max = uv_params["v_max"]
        
        try:
            # Create occwl Face object
            occwl_face = Face(face)
            
            # Sample on grid
            grid_size = self.curvature_grid_size
            points = []
            curvature_values = []
            
            print(f"🔍 Computing curvature on {grid_size}x{grid_size} grid...")
            print(f"   UV range: U=[{u_min:.3f}, {u_max:.3f}], V=[{v_min:.3f}, {v_max:.3f}]")
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Compute UV coordinates
                    u = u_min + (u_max - u_min) * i / (grid_size - 1) if grid_size > 1 else u_min
                    v = v_min + (v_max - v_min) * j / (grid_size - 1) if grid_size > 1 else v_min
                    
                    try:
                        # Get max curvature at this UV location
                        max_curv = occwl_face.max_curvature([u, v])
                        
                        # Get the 3D point at this UV location
                        # geom_surface = occwl_face.surface
                        # pnt = geom_surface.Value(u, v)
                        points.append(occwl_face.point([u, v]))
                        # points.append([pnt.X(), pnt.Y(), pnt.Z()])
                        curvature_values.append(max_curv)
                    except Exception as e:
                        # Skip invalid UV points
                        print(f"   ⚠️ Failed at UV({u:.3f}, {v:.3f}): {e}")
                        continue
            
            print(f"   ✅ Computed curvature at {len(points)} points")
            
            if len(points) > 0:
                points = np.array(points)
                curvature_values = np.array(curvature_values)
                max_curvature = np.max(curvature_values)
                return points, curvature_values, max_curvature
            else:
                print(f"   ❌ No valid curvature points computed")
                return None, None, 0.0
                
        except Exception as e:
            print(f"⚠️ Curvature computation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0
    
    def create_uv_grid(self, geom_surface, uv_params):
        """Create UV grid lines on the surface"""
        u_min = uv_params["u_min"]
        u_max = uv_params["u_max"]
        v_min = uv_params["v_min"]
        v_max = uv_params["v_max"]
        
        all_points = []
        all_edges = []
        point_idx = 0
        
        # Create U-direction lines (constant V)
        for i in range(self.uv_grid_density_v + 1):
            v = v_min + (v_max - v_min) * i / self.uv_grid_density_v
            line_points = []
            
            for j in range(self.uv_grid_density_u + 1):
                u = u_min + (u_max - u_min) * j / self.uv_grid_density_u
                try:
                    pnt = geom_surface.Value(u, v)
                    line_points.append([pnt.X(), pnt.Y(), pnt.Z()])
                except:
                    # Skip invalid UV points
                    continue
            
            if len(line_points) > 1:
                start_idx = len(all_points)
                all_points.extend(line_points)
                # Create edges for this line
                for k in range(len(line_points) - 1):
                    all_edges.append([start_idx + k, start_idx + k + 1])
        
        # Create V-direction lines (constant U)
        for i in range(self.uv_grid_density_u + 1):
            u = u_min + (u_max - u_min) * i / self.uv_grid_density_u
            line_points = []
            
            for j in range(self.uv_grid_density_v + 1):
                v = v_min + (v_max - v_min) * j / self.uv_grid_density_v
                try:
                    pnt = geom_surface.Value(u, v)
                    line_points.append([pnt.X(), pnt.Y(), pnt.Z()])
                except:
                    continue
            
            if len(line_points) > 1:
                start_idx = len(all_points)
                all_points.extend(line_points)
                for k in range(len(line_points) - 1):
                    all_edges.append([start_idx + k, start_idx + k + 1])
        
        return np.array(all_points), np.array(all_edges)
    
    def update_visualization(self):
        """Update the visualization with current parameters"""
        try:
            # Build surface
            face, geom_surface = self.build_current_surface()
            
            if face is None or geom_surface is None:
                print(f"Failed to build {self.current_surface_type} surface")
                return
            
            # Mesh the surface
            mesher = BRepMesh_IncrementalMesh(face, self.mesh_quality, True, 0.5)
            mesher.Perform()
            
            if not mesher.IsDone():
                print("Meshing failed")
                return
            
            # Extract mesh
            vertices, faces = self.extract_mesh_from_face(face)
            
            if vertices is None or faces is None or len(vertices) == 0:
                print("Failed to extract mesh")
                return
            
            # Update surface mesh
            if self.show_surface:
                if self.surface_mesh_obj is not None:
                    ps.remove_surface_mesh("surface")
                
                self.surface_mesh_obj = ps.register_surface_mesh("surface", vertices, faces)
                self.surface_mesh_obj.set_transparency(self.surface_transparency)
                self.surface_mesh_obj.set_color([0.2, 0.6, 0.9])
            else:
                if self.surface_mesh_obj is not None:
                    ps.remove_surface_mesh("surface")
                    self.surface_mesh_obj = None
            
            # Update UV grid
            if self.show_uv_grid:
                uv_params = self.uv_params[self.current_surface_type]
                grid_points, grid_edges = self.create_uv_grid(geom_surface, uv_params)
                
                if len(grid_points) > 0 and len(grid_edges) > 0:
                    if self.uv_grid_obj is not None:
                        ps.remove_curve_network("uv_grid")
                    
                    self.uv_grid_obj = ps.register_curve_network("uv_grid", grid_points, grid_edges)
                    self.uv_grid_obj.set_color([1.0, 0.5, 0.0])
                    self.uv_grid_obj.set_radius(0.01)
                    
                    # Also show grid intersection points
                    if self.uv_grid_points_obj is not None:
                        ps.remove_point_cloud("uv_grid_points")
                    
                    self.uv_grid_points_obj = ps.register_point_cloud("uv_grid_points", grid_points)
                    self.uv_grid_points_obj.set_color([1.0, 0.0, 0.0])
                    self.uv_grid_points_obj.set_radius(0.015)
            else:
                if self.uv_grid_obj is not None:
                    ps.remove_curve_network("uv_grid")
                    self.uv_grid_obj = None
                if self.uv_grid_points_obj is not None:
                    ps.remove_point_cloud("uv_grid_points")
                    self.uv_grid_points_obj = None
            
            # Update coordinate frame visualization
            if self.show_coordinate_frame:
                frame_data = self.create_coordinate_frame_visualization()
                if frame_data is not None:
                    frame_points, frame_edges = frame_data
                    
                    if self.coordinate_frame_obj is not None:
                        ps.remove_curve_network("coordinate_frame")
                    
                    self.coordinate_frame_obj = ps.register_curve_network("coordinate_frame", frame_points, frame_edges)
                    
                    # Color the axes: X=red, Y=green, Z=blue
                    # Create per-edge colors
                    edge_colors = np.array([
                        [1.0, 0.0, 0.0],  # X axis - red
                        [0.0, 1.0, 0.0],  # Y axis - green
                        [0.0, 0.0, 1.0],  # Z axis - blue (direction)
                    ])
                    self.coordinate_frame_obj.add_color_quantity("axis_colors", edge_colors, defined_on='edges', enabled=True)
                    self.coordinate_frame_obj.set_radius(0.02)
            else:
                if self.coordinate_frame_obj is not None:
                    ps.remove_curve_network("coordinate_frame")
                    self.coordinate_frame_obj = None
            
            # Update position point visualization
            if self.show_position_point:
                params = self.surface_params.get(self.current_surface_type, {})
                if "position" in params:
                    position = np.array([params["position"]], dtype=np.float64)
                    
                    if self.position_point_obj is not None:
                        ps.remove_point_cloud("position_point")
                    
                    self.position_point_obj = ps.register_point_cloud("position_point", position)
                    self.position_point_obj.set_color([1.0, 0.0, 1.0])  # Magenta color
                    self.position_point_obj.set_radius(self.position_point_radius)
            else:
                if self.position_point_obj is not None:
                    ps.remove_point_cloud("position_point")
                    self.position_point_obj = None
            
            # Compute and visualize curvature
            if self.show_curvature_analysis:
                uv_params = self.uv_params[self.current_surface_type]
                curv_points, curv_values, max_curv = self.compute_curvature_on_grid(face, uv_params)
                
                if curv_points is not None and curv_values is not None:
                    self.max_curvature_value = max_curv
                    self.curvature_values = curv_values
                    
                    if self.curvature_points_obj is not None:
                        ps.remove_point_cloud("curvature_points")
                    
                    self.curvature_points_obj = ps.register_point_cloud("curvature_points", curv_points)
                    self.curvature_points_obj.set_radius(0.03)
                    
                    # Add curvature values as a scalar quantity with color map
                    self.curvature_points_obj.add_scalar_quantity("max_curvature", curv_values, enabled=True, cmap='turbo')
                else:
                    self.max_curvature_value = 0.0
                    self.curvature_values = None
            else:
                if self.curvature_points_obj is not None:
                    ps.remove_point_cloud("curvature_points")
                    self.curvature_points_obj = None
                self.max_curvature_value = 0.0
                self.curvature_values = None
            
            print(f"✅ Updated {self.current_surface_type} surface with {len(vertices)} vertices, {len(faces)} faces")
            if self.show_curvature_analysis and self.max_curvature_value > 0:
                print(f"📊 Max curvature: {self.max_curvature_value:.6f}")
            
        except Exception as e:
            print(f"❌ Error updating visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def gui_callback(self):
        """GUI callback for polyscope"""
        psim.PushItemWidth(200)
        
        # Surface type selection
        if psim.TreeNode("Surface Type"):
            psim.Text("Select surface type to visualize:")
            psim.Separator()
            
            for i, surface_type in enumerate(self.surface_types):
                if psim.RadioButton(surface_type.replace("_", " ").title(), i == self.current_surface_index):
                    if i != self.current_surface_index:
                        self.current_surface_index = i
                        self.current_surface_type = surface_type
                        self.update_visualization()
            
            psim.TreePop()
        
        # UV Parameters
        if psim.TreeNode("UV Parameters"):
            psim.Text(f"UV bounds for {self.current_surface_type}:")
            psim.Separator()
            
            uv = self.uv_params[self.current_surface_type]
            updated = False
            
            # U min/max
            psim.Text("U Parameter:")
            psim.Text(f"  Range: [{uv['u_range'][0]:.6f}, {uv['u_range'][1]:.6f}]")
            changed_u_min, new_u_min = psim.InputFloat("U Min", uv["u_min"], step=1e-8, step_fast=1e-4, format="%.8f")
            if changed_u_min:
                # Clamp to valid range and ensure u_min < u_max
                new_u_min = max(uv["u_range"][0], min(uv["u_range"][1], new_u_min))
                if new_u_min < uv["u_max"]:
                    uv["u_min"] = new_u_min
                    updated = True
            
            changed_u_max, new_u_max = psim.InputFloat("U Max", uv["u_max"], step=1e-8, step_fast=1e-4, format="%.8f")
            if changed_u_max:
                # Clamp to valid range and ensure u_max > u_min
                new_u_max = max(uv["u_range"][0], min(uv["u_range"][1], new_u_max))
                if new_u_max > uv["u_min"]:
                    uv["u_max"] = new_u_max
                    updated = True
            
            psim.Separator()
            
            # V min/max
            psim.Text("V Parameter:")
            psim.Text(f"  Range: [{uv['v_range'][0]:.6f}, {uv['v_range'][1]:.6f}]")
            changed_v_min, new_v_min = psim.InputFloat("V Min", uv["v_min"], step=1e-8, step_fast=1e-4, format="%.8f")
            if changed_v_min:
                # Clamp to valid range and ensure v_min < v_max
                new_v_min = max(uv["v_range"][0], min(uv["v_range"][1], new_v_min))
                if new_v_min < uv["v_max"]:
                    uv["v_min"] = new_v_min
                    updated = True
            
            changed_v_max, new_v_max = psim.InputFloat("V Max", uv["v_max"], step=1e-8, step_fast=1e-4, format="%.8f")
            if changed_v_max:
                # Clamp to valid range and ensure v_max > v_min
                new_v_max = max(uv["v_range"][0], min(uv["v_range"][1], new_v_max))
                if new_v_max > uv["v_min"]:
                    uv["v_max"] = new_v_max
                    updated = True
            
            psim.Separator()
            
            # Reset button
            if psim.Button("Reset UV to Default"):
                self.reset_uv_params()
                updated = True
            
            if updated:
                self.update_visualization()
            
            psim.TreePop()
        
        # Geometric Parameters
        if psim.TreeNode("Geometric Parameters"):
            params = self.surface_params.get(self.current_surface_type, {})
            updated = False
            
            # Position controls (available for all surfaces)
            if "position" in params:
                psim.Text("Position (Center):")
                psim.Separator()
                
                changed_x, new_x = psim.InputFloat("X##pos", params["position"][0], step=0.1, step_fast=1.0, format="%.6f")
                if changed_x:
                    new_x = max(-1e5, min(1e5, new_x))
                    params["position"][0] = new_x
                    updated = True
                
                changed_y, new_y = psim.InputFloat("Y##pos", params["position"][1], step=0.1, step_fast=1.0, format="%.6f")
                if changed_y:
                    new_y = max(-1e5, min(1e5, new_y))
                    params["position"][1] = new_y
                    updated = True
                
                changed_z, new_z = psim.InputFloat("Z##pos", params["position"][2], step=0.1, step_fast=1.0, format="%.6f")
                if changed_z:
                    new_z = max(-1e5, min(1e5, new_z))
                    params["position"][2] = new_z
                    updated = True
                
                psim.Separator()
            
            # Direction controls (Z-axis of local coordinate system)
            if "direction" in params:
                psim.Text("Direction (Z-axis, normalized):")
                psim.Separator()
                
                changed_dx, new_dx = psim.InputFloat("X##dir", params["direction"][0], step=0.01, step_fast=0.1, format="%.6f")
                if changed_dx:
                    new_dx = max(-1e5, min(1e5, new_dx))
                    params["direction"][0] = new_dx
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                changed_dy, new_dy = psim.InputFloat("Y##dir", params["direction"][1], step=0.01, step_fast=0.1, format="%.6f")
                if changed_dy:
                    new_dy = max(-1e5, min(1e5, new_dy))
                    params["direction"][1] = new_dy
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                changed_dz, new_dz = psim.InputFloat("Z##dir", params["direction"][2], step=0.01, step_fast=0.1, format="%.6f")
                if changed_dz:
                    new_dz = max(-1e5, min(1e5, new_dz))
                    params["direction"][2] = new_dz
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                psim.Separator()
            
            # X Direction controls (X-axis of local coordinate system)
            if "x_direction" in params:
                psim.Text("X Direction (X-axis, orthogonalized):")
                psim.Separator()
                
                changed_xdx, new_xdx = psim.InputFloat("X##xdir", params["x_direction"][0], step=0.01, step_fast=0.1, format="%.6f")
                if changed_xdx:
                    new_xdx = max(-1e5, min(1e5, new_xdx))
                    params["x_direction"][0] = new_xdx
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                changed_xdy, new_xdy = psim.InputFloat("Y##xdir", params["x_direction"][1], step=0.01, step_fast=0.1, format="%.6f")
                if changed_xdy:
                    new_xdy = max(-1e5, min(1e5, new_xdy))
                    params["x_direction"][1] = new_xdy
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                changed_xdz, new_xdz = psim.InputFloat("Z##xdir", params["x_direction"][2], step=0.01, step_fast=0.1, format="%.6f")
                if changed_xdz:
                    new_xdz = max(-1e5, min(1e5, new_xdz))
                    params["x_direction"][2] = new_xdz
                    self.update_surface_coordinate_system(self.current_surface_type)
                    updated = True
                
                psim.Separator()
                
                # Display Y direction (computed automatically)
                if "y_direction" in params:
                    psim.Text("Y Direction (computed, Y-axis):")
                    y_dir = params["y_direction"]
                    psim.Text(f"  X: {y_dir[0]:.3f}, Y: {y_dir[1]:.3f}, Z: {y_dir[2]:.3f}")
                    psim.Separator()
            
            # Shape-specific parameters
            if "radius" in params:
                psim.Text("Radius:")
                changed, new_val = psim.InputFloat("##radius", params["radius"], step=0.1, step_fast=1.0, format="%.6f")
                if changed:
                    new_val = max(-1e5, min(1e5, new_val))
                    params["radius"] = new_val
                    updated = True
            
            if "semi_angle" in params:
                psim.Text("Semi Angle (radians):")
                changed, new_val = psim.InputFloat("##semi_angle", params["semi_angle"], step=0.01, step_fast=0.1, format="%.6f")
                if changed:
                    new_val = max(-1e5, min(1e5, new_val))
                    params["semi_angle"] = new_val
                    updated = True
            
            if "major_radius" in params:
                psim.Text("Major Radius:")
                changed, new_val = psim.InputFloat("##major_radius", params["major_radius"], step=0.1, step_fast=1.0, format="%.6f")
                if changed:
                    new_val = max(-1e5, min(1e5, new_val))
                    params["major_radius"] = new_val
                    updated = True
            
            if "minor_radius" in params:
                psim.Text("Minor Radius:")
                changed, new_val = psim.InputFloat("##minor_radius", params["minor_radius"], step=0.1, step_fast=1.0, format="%.6f")
                if changed:
                    new_val = max(-1e5, min(1e5, new_val))
                    params["minor_radius"] = new_val
                    updated = True
            
            if updated:
                self.update_visualization()
            
            psim.TreePop()
        
        # Visualization Settings
        if psim.TreeNode("Visualization Settings"):
            psim.Separator()
            
            # Surface visibility
            changed_surface, self.show_surface = psim.Checkbox("Show Surface", self.show_surface)
            if changed_surface:
                self.update_visualization()
            
            # Surface transparency
            if self.show_surface:
                changed_trans, new_trans = psim.SliderFloat(
                    "Surface Transparency", self.surface_transparency, 0.0, 1.0
                )
                if changed_trans:
                    self.surface_transparency = new_trans
                    if self.surface_mesh_obj is not None:
                        self.surface_mesh_obj.set_transparency(self.surface_transparency)
            
            psim.Separator()
            
            # UV grid visibility
            changed_grid, self.show_uv_grid = psim.Checkbox("Show UV Grid", self.show_uv_grid)
            if changed_grid:
                self.update_visualization()
            
            # UV grid density
            if self.show_uv_grid:
                changed_u_density, new_u_density = psim.SliderInt(
                    "U Grid Lines", self.uv_grid_density_u, 2, 20
                )
                if changed_u_density:
                    self.uv_grid_density_u = new_u_density
                    self.update_visualization()
                
                changed_v_density, new_v_density = psim.SliderInt(
                    "V Grid Lines", self.uv_grid_density_v, 2, 20
                )
                if changed_v_density:
                    self.uv_grid_density_v = new_v_density
                    self.update_visualization()
            
            psim.Separator()
            
            # Mesh quality
            changed_quality, new_quality = psim.SliderFloat(
                "Mesh Quality", self.mesh_quality, 0.01, 1.0
            )
            if changed_quality:
                self.mesh_quality = new_quality
                self.update_visualization()
            
            psim.Text("(Smaller = finer mesh)")
            
            psim.Separator()
            
            # Reference cube visibility
            changed_cube, self.show_reference_cube = psim.Checkbox("Show Reference Cube", self.show_reference_cube)
            if changed_cube and self.reference_cube_obj is not None:
                self.reference_cube_obj.set_enabled(self.show_reference_cube)
            
            psim.Separator()
            
            # Coordinate frame visibility
            changed_frame, self.show_coordinate_frame = psim.Checkbox("Show Coordinate Frame", self.show_coordinate_frame)
            if changed_frame:
                self.update_visualization()
            
            # Coordinate frame scale
            if self.show_coordinate_frame:
                changed_scale, new_scale = psim.SliderFloat("Frame Scale", self.coordinate_frame_scale, 1.0, 15.0)
                if changed_scale:
                    self.coordinate_frame_scale = new_scale
                    self.update_visualization()
                
                psim.Text("Red=X, Green=Y, Blue=Z(direction)")
            
            psim.Separator()
            
            # Position point visibility
            changed_pos_point, self.show_position_point = psim.Checkbox("Show Position Point", self.show_position_point)
            if changed_pos_point:
                self.update_visualization()
            
            # Position point radius
            if self.show_position_point:
                changed_radius, new_radius = psim.SliderFloat("Point Radius", self.position_point_radius, 0.01, 1.0)
                if changed_radius:
                    self.position_point_radius = new_radius
                    if self.position_point_obj is not None:
                        self.position_point_obj.set_radius(self.position_point_radius)
                
                psim.Text("Magenta point shows geometry center")
            
            psim.Separator()
            
            # Curvature analysis
            changed_curv, self.show_curvature_analysis = psim.Checkbox("Show Curvature Analysis", self.show_curvature_analysis)
            if changed_curv:
                self.update_visualization()
            
            if self.show_curvature_analysis:
                changed_grid, new_grid = psim.SliderInt("Curvature Grid Size", self.curvature_grid_size, 4, 16)
                if changed_grid:
                    self.curvature_grid_size = new_grid
                    self.update_visualization()
                
                psim.Text("Color-coded max curvature values")
                if self.max_curvature_value > 0:
                    psim.Text(f"Max: {self.max_curvature_value:.6e}")
            
            psim.TreePop()
        
        # Information panel
        if psim.TreeNode("Surface Information"):
            psim.Separator()
            
            uv = self.uv_params[self.current_surface_type]
            params = self.surface_params.get(self.current_surface_type, {})
            
            psim.Text(f"Current Surface: {self.current_surface_type.replace('_', ' ').title()}")
            psim.Separator()
            
            # UV parameters
            psim.Text("UV Parameters:")
            psim.Text(f"  U Range: [{uv['u_min']:.3f}, {uv['u_max']:.3f}]")
            psim.Text(f"  V Range: [{uv['v_min']:.3f}, {uv['v_max']:.3f}]")
            psim.Text(f"  U Span: {uv['u_max'] - uv['u_min']:.3f}")
            psim.Text(f"  V Span: {uv['v_max'] - uv['v_min']:.3f}")
            
            psim.Separator()
            
            # Coordinate system
            if "position" in params:
                psim.Text("Coordinate System:")
                pos = params["position"]
                psim.Text(f"  Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
                
                if "direction" in params:
                    dir_vec = params["direction"]
                    psim.Text(f"  Direction (Z): [{dir_vec[0]:.3f}, {dir_vec[1]:.3f}, {dir_vec[2]:.3f}]")
                    norm = np.linalg.norm(dir_vec)
                    psim.Text(f"    Magnitude: {norm:.4f}")
                
                if "x_direction" in params:
                    x_dir = params["x_direction"]
                    psim.Text(f"  X Direction: [{x_dir[0]:.3f}, {x_dir[1]:.3f}, {x_dir[2]:.3f}]")
                    norm_x = np.linalg.norm(x_dir)
                    psim.Text(f"    Magnitude: {norm_x:.4f}")
                
                if "y_direction" in params:
                    y_dir = params["y_direction"]
                    psim.Text(f"  Y Direction: [{y_dir[0]:.3f}, {y_dir[1]:.3f}, {y_dir[2]:.3f}]")
                    norm_y = np.linalg.norm(y_dir)
                    psim.Text(f"    Magnitude: {norm_y:.4f}")
                
                # Check orthogonality
                if "direction" in params and "x_direction" in params and "y_direction" in params:
                    psim.Separator()
                    dir_vec = np.array(params["direction"])
                    x_dir = np.array(params["x_direction"])
                    y_dir = np.array(params["y_direction"])
                    
                    dot_xy = abs(np.dot(x_dir, y_dir))
                    dot_xz = abs(np.dot(x_dir, dir_vec))
                    dot_yz = abs(np.dot(y_dir, dir_vec))
                    
                    psim.Text("Orthogonality Check:")
                    psim.Text(f"  X·Y: {dot_xy:.6f} (should be ~0)")
                    psim.Text(f"  X·Z: {dot_xz:.6f} (should be ~0)")
                    psim.Text(f"  Y·Z: {dot_yz:.6f} (should be ~0)")
            
            psim.Separator()
            
            # Curvature analysis results
            if self.show_curvature_analysis:
                psim.Text("Curvature Analysis:")
                psim.Text(f"  Grid Size: {self.curvature_grid_size} x {self.curvature_grid_size}")
                if self.curvature_values is not None and len(self.curvature_values) > 0:
                    psim.Text(f"  Max Curvature: {self.max_curvature_value:.6e}")
                    psim.Text(f"  Min Curvature: {np.min(self.curvature_values):.6e}")
                    psim.Text(f"  Mean Curvature: {np.mean(self.curvature_values):.6e}")
                    psim.Text(f"  Sample Points: {len(self.curvature_values)}")
                else:
                    psim.Text("  No curvature data available")
                psim.Separator()
            
            psim.Text("UV Parameterization Notes:")
            
            if self.current_surface_type == "plane":
                psim.TextWrapped("Plane: U and V are Cartesian coordinates in the plane")
            elif self.current_surface_type == "cylinder":
                psim.TextWrapped("Cylinder: U is angular (0 to 2π), V is height along axis")
            elif self.current_surface_type == "cone":
                psim.TextWrapped("Cone: U is angular (0 to 2π), V is distance along axis")
            elif self.current_surface_type == "sphere":
                psim.TextWrapped("Sphere: U is longitude (0 to 2π), V is latitude (-π/2 to π/2)")
            elif self.current_surface_type == "torus":
                psim.TextWrapped("Torus: U is major circle angle, V is minor circle angle (both 0 to 2π)")
            elif self.current_surface_type == "bspline_surface":
                psim.TextWrapped("B-spline: U and V are normalized parameters (0 to 1)")
            
            psim.TreePop()
        
        psim.PopItemWidth()
    
    def reset_uv_params(self):
        """Reset UV parameters to default values for current surface"""
        if self.current_surface_type == "plane":
            self.uv_params["plane"].update({
                "u_min": 0.0, "u_max": 10.0,
                "v_min": 0.0, "v_max": 10.0
            })
        elif self.current_surface_type == "cylinder":
            self.uv_params["cylinder"].update({
                "u_min": 0.0, "u_max": 6.28,
                "v_min": 0.0, "v_max": 10.0
            })
        elif self.current_surface_type == "cone":
            self.uv_params["cone"].update({
                "u_min": 0.0, "u_max": 6.28,
                "v_min": 0.0, "v_max": 10.0
            })
        elif self.current_surface_type == "sphere":
            self.uv_params["sphere"].update({
                "u_min": 0.0, "u_max": 6.28,
                "v_min": -1.57, "v_max": 1.57
            })
        elif self.current_surface_type == "torus":
            self.uv_params["torus"].update({
                "u_min": 0.0, "u_max": 6.28,
                "v_min": 0.0, "v_max": 6.28
            })
        elif self.current_surface_type == "bspline_surface":
            self.uv_params["bspline_surface"].update({
                "u_min": 0.0, "u_max": 1.0,
                "v_min": 0.0, "v_max": 1.0
            })
    
    def run(self):
        """Run the interactive visualizer"""
        ps.init()
        ps.set_ground_plane_mode("none")
        
        # Initialize coordinate systems for all surfaces
        for surface_type in self.surface_types:
            if surface_type != "bspline_surface":
                self.update_surface_coordinate_system(surface_type)
        
        # Create and register reference cube
        cube_vertices, cube_faces = self.create_reference_cube()
        self.reference_cube_obj = ps.register_surface_mesh("reference_cube", cube_vertices, cube_faces)
        self.reference_cube_obj.set_color([0.7, 0.7, 0.7])
        self.reference_cube_obj.set_transparency(0.3)
        self.reference_cube_obj.set_edge_width(1.5)
        self.reference_cube_obj.set_edge_color([0.3, 0.3, 0.3])
        
        # Initial visualization
        self.update_visualization()
        
        # Set up GUI callback
        ps.set_user_callback(self.gui_callback)
        
        print("=" * 70)
        print("Interactive UV Parameter Visualizer")
        print("=" * 70)
        print()
        print("🎛️ Controls:")
        print("  - Select different surface types")
        print("  - Adjust UV min/max parameters with sliders")
        print("  - Adjust position (center) of surfaces")
        print("  - Modify direction (Z-axis) and X-direction vectors")
        print("  - Y-direction is computed automatically (orthogonal)")
        print("  - Toggle surface and UV grid visibility")
        print("  - Modify geometric parameters (radius, angles, etc.)")
        print("  - Adjust UV grid density to see parameter lines")
        print()
        print("📐 UV Parameterization:")
        print("  - Orange lines show constant U and V parameter curves")
        print("  - Red points show grid intersections")
        print("  - Adjust parameters to see how surfaces are defined")
        print()
        print("📦 Reference Cube:")
        print("  - Gray cube from (0,0,0) to (1,1,1) for scale reference")
        print()
        print("🎯 Coordinate Frame:")
        print("  - Red axis: X-direction")
        print("  - Green axis: Y-direction (computed)")
        print("  - Blue axis: Z-direction (main direction)")
        print("  - All vectors are normalized and orthogonal")
        print()
        print("📍 Position Point:")
        print("  - Magenta point shows the geometry center/position")
        print("  - Adjustable radius in visualization settings")
        print()
        print("📊 Curvature Analysis:")
        print("  - Computes max curvature on a UV grid (default 8x8)")
        print("  - Color-coded points show curvature distribution")
        print("  - Uses occwl.face.Face.max_curvature() method")
        print("  - Displays max, min, and mean curvature values")
        print()
        print("=" * 70)
        
        # Show visualization
        ps.show()


def main():
    """Main entry point"""
    visualizer = UVParameterVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()

