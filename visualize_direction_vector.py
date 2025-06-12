#!/usr/bin/env python3
"""
Interactive direction vector visualizer using polyscope.
Converts user-input azimuthal and polar angles to 3D direction vectors
and visualizes them in polyscope with multiple representation options.

Usage:
$ python visualize_direction_vector.py

Controls:
- Enter azimuthal angle (phi) in degrees [-180, 180]
- Enter polar angle (theta) in degrees [0, 180]  
- Choose visualization style
- Interactive 3D view with polyscope

Based on latest polyscope documentation features.
"""

import argparse
import numpy as np
import polyscope as ps
from typing import Tuple, List, Optional
import math

class DirectionVectorVisualizer:
    def __init__(self):
        self.current_azimuthal = 0.0  # phi in degrees
        self.current_polar = 90.0     # theta in degrees
        self.vector_length = 1.0
        self.show_sphere = True
        self.show_coordinates = True
        self.vector_color = [1.0, 0.0, 0.0]  # Red
        self.coordinate_color = [0.5, 0.5, 0.5]  # Gray
        
        # UI state
        self.ui_azimuthal = self.current_azimuthal
        self.ui_polar = self.current_polar
        
    def spherical_to_cartesian(self, azimuthal_deg: float, polar_deg: float, radius: float = 1.0) -> np.ndarray:
        """
        Convert spherical coordinates to cartesian coordinates.
        
        Args:
            azimuthal_deg: Azimuthal angle phi in degrees [-180, 180]
            polar_deg: Polar angle theta in degrees [0, 180]
            radius: Distance from origin
            
        Returns:
            3D cartesian coordinates as numpy array [x, y, z]
        """
        # Convert to radians
        phi = math.radians(azimuthal_deg)
        theta = math.radians(polar_deg)
        
        # Spherical to cartesian conversion
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        
        return np.array([x, y, z])
    
    def create_sphere_mesh(self, resolution: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Create a unit sphere mesh for reference."""
        phi = np.linspace(0, 2*np.pi, resolution)
        theta = np.linspace(0, np.pi, resolution)
        
        vertices = []
        faces = []
        
        # Generate vertices
        for i, t in enumerate(theta):
            for j, p in enumerate(phi):
                x = math.sin(t) * math.cos(p)
                y = math.sin(t) * math.sin(p)
                z = math.cos(t)
                vertices.append([x, y, z])
        
        # Generate faces
        for i in range(len(theta) - 1):
            for j in range(len(phi) - 1):
                # Current quad vertices
                v0 = i * len(phi) + j
                v1 = i * len(phi) + (j + 1)
                v2 = (i + 1) * len(phi) + j
                v3 = (i + 1) * len(phi) + (j + 1)
                
                # Two triangles per quad
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return np.array(vertices), np.array(faces)
    
    def create_coordinate_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create coordinate system axes."""
        # Origin and axis endpoints
        points = np.array([
            [0, 0, 0],  # Origin
            [1.2, 0, 0],  # X-axis
            [0, 1.2, 0],  # Y-axis  
            [0, 0, 1.2],  # Z-axis
        ])
        
        # Vectors for each axis
        vectors = np.array([
            [1.2, 0, 0],   # X-axis vector
            [0, 1.2, 0],   # Y-axis vector
            [0, 0, 1.2],   # Z-axis vector
            [0, 0, 0],     # Dummy for origin
        ])
        
        return points, vectors
    
    def update_visualization(self):
        """Update the polyscope visualization with current parameters."""
        # Clear previous visualizations
        ps.remove_all_structures()
        
        # Calculate direction vector
        direction = self.spherical_to_cartesian(self.current_azimuthal, self.current_polar, self.vector_length)
        
        # 1. Create and display reference sphere
        if self.show_sphere:
            sphere_verts, sphere_faces = self.create_sphere_mesh()
            sphere_mesh = ps.register_surface_mesh("reference_sphere", sphere_verts, sphere_faces)
            sphere_mesh.set_color([0.8, 0.8, 0.8])
            sphere_mesh.set_transparency(0.3)
            sphere_mesh.set_material("wax")
        
        # 2. Create and display coordinate system
        if self.show_coordinates:
            coord_points, coord_vectors = self.create_coordinate_system()
            coord_cloud = ps.register_point_cloud("coordinate_axes", coord_points[:3])  # Skip origin
            coord_cloud.set_radius(0.02)
            
            # Add coordinate vectors
            coord_cloud.add_vector_quantity("axes", coord_vectors[:3], 
                                          enabled=True, 
                                          vectortype="ambient",
                                          length=1.0,
                                          radius=0.01,
                                          color=self.coordinate_color)
        
        # 3. Create and display the main direction vector
        origin = np.array([[0, 0, 0]])
        vector_array = np.array([direction])
        
        origin_cloud = ps.register_point_cloud("vector_origin", origin)
        origin_cloud.set_radius(0.05)
        origin_cloud.set_color([0.0, 0.0, 1.0])  # Blue origin
        
        # Add the direction vector
        origin_cloud.add_vector_quantity("direction_vector", vector_array,
                                        enabled=True,
                                        vectortype="ambient", 
                                        length=1.0,
                                        radius=0.03,
                                        color=self.vector_color)
        
        # 4. Add endpoint marker
        endpoint = np.array([direction])
        endpoint_cloud = ps.register_point_cloud("vector_endpoint", endpoint)
        endpoint_cloud.set_radius(0.04)
        endpoint_cloud.set_color(self.vector_color)
        endpoint_cloud.set_material("candy")
        
        # 5. Add angle visualization (arc on sphere)
        self.add_angle_visualization()
        
        print(f"Direction Vector Updated:")
        print(f"  Azimuthal (φ): {self.current_azimuthal:.1f}°")
        print(f"  Polar (θ): {self.current_polar:.1f}°") 
        print(f"  Cartesian: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
        print(f"  Magnitude: {np.linalg.norm(direction):.3f}")
    
    def add_angle_visualization(self):
        """Add visual aids for angle understanding."""
        # Create meridian line (constant azimuth)
        theta_range = np.linspace(0, np.pi, 50)
        phi_fixed = math.radians(self.current_azimuthal)
        
        meridian_points = []
        for theta in theta_range:
            x = math.sin(theta) * math.cos(phi_fixed)
            y = math.sin(theta) * math.sin(phi_fixed)
            z = math.cos(theta)
            meridian_points.append([x, y, z])
        
        meridian_points = np.array(meridian_points)
        meridian_cloud = ps.register_point_cloud("meridian_line", meridian_points)
        meridian_cloud.set_radius(0.005)
        meridian_cloud.set_color([1.0, 1.0, 0.0])  # Yellow
        
        # Create parallel line (constant polar angle)
        phi_range = np.linspace(-np.pi, np.pi, 50)
        theta_fixed = math.radians(self.current_polar)
        
        parallel_points = []
        for phi in phi_range:
            x = math.sin(theta_fixed) * math.cos(phi)
            y = math.sin(theta_fixed) * math.sin(phi)
            z = math.cos(theta_fixed)
            parallel_points.append([x, y, z])
        
        parallel_points = np.array(parallel_points)
        parallel_cloud = ps.register_point_cloud("parallel_line", parallel_points)
        parallel_cloud.set_radius(0.005) 
        parallel_cloud.set_color([0.0, 1.0, 1.0])  # Cyan
    
    def create_ui(self):
        """Create polyscope UI controls."""
        def ui_callback():
            # Angle input controls
            ps.ImGuiText("Direction Vector Controls")
            ps.ImGuiSeparator()
            
            # Azimuthal angle control
            ps.ImGuiText("Azimuthal Angle (φ)")
            ps.ImGuiText("Range: [-180°, 180°] (longitude)")
            changed_azi, new_azi = ps.ImGuiSliderFloat("##azimuthal", self.ui_azimuthal, 
                                                      v_min=-180.0, v_max=180.0)
            if changed_azi:
                self.ui_azimuthal = new_azi
            
            # Polar angle control  
            ps.ImGuiText("Polar Angle (θ)")
            ps.ImGuiText("Range: [0°, 180°] (latitude from north pole)")
            changed_pol, new_pol = ps.ImGuiSliderFloat("##polar", self.ui_polar,
                                                      v_min=0.0, v_max=180.0)
            if changed_pol:
                self.ui_polar = new_pol
            
            ps.ImGuiSeparator()
            
            # Update button
            if ps.ImGuiButton("Update Vector"):
                self.current_azimuthal = self.ui_azimuthal
                self.current_polar = self.ui_polar
                self.update_visualization()
            
            ps.ImGuiSameLine()
            if ps.ImGuiButton("Reset to Default"):
                self.ui_azimuthal = 0.0
                self.ui_polar = 90.0
                self.current_azimuthal = 0.0
                self.current_polar = 90.0
                self.update_visualization()
            
            ps.ImGuiSeparator()
            
            # Display options
            ps.ImGuiText("Display Options")
            
            changed_sphere, new_sphere = ps.ImGuiCheckbox("Show Reference Sphere", self.show_sphere)
            if changed_sphere:
                self.show_sphere = new_sphere
                self.update_visualization()
            
            changed_coord, new_coord = ps.ImGuiCheckbox("Show Coordinate System", self.show_coordinates)
            if changed_coord:
                self.show_coordinates = new_coord
                self.update_visualization()
            
            ps.ImGuiSeparator()
            
            # Vector properties
            ps.ImGuiText("Vector Properties")
            changed_len, new_len = ps.ImGuiSliderFloat("Vector Length", self.vector_length,
                                                      v_min=0.1, v_max=2.0)
            if changed_len:
                self.vector_length = new_len
                self.update_visualization()
            
            # Color controls
            changed_color, new_color = ps.ImGuiColorEdit3("Vector Color", self.vector_color)
            if changed_color:
                self.vector_color = new_color
                self.update_visualization()
            
            ps.ImGuiSeparator()
            
            # Information display
            ps.ImGuiText("Current Vector Information")
            direction = self.spherical_to_cartesian(self.current_azimuthal, self.current_polar, self.vector_length)
            ps.ImGuiText(f"Azimuthal (φ): {self.current_azimuthal:.1f}°")
            ps.ImGuiText(f"Polar (θ): {self.current_polar:.1f}°")
            ps.ImGuiText(f"Cartesian: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
            ps.ImGuiText(f"Magnitude: {np.linalg.norm(direction):.3f}")
            
            ps.ImGuiSeparator()
            
            # Quick presets
            ps.ImGuiText("Quick Presets")
            if ps.ImGuiButton("North Pole (0°, 0°)"):
                self.ui_azimuthal = 0.0
                self.ui_polar = 0.0
                self.current_azimuthal = 0.0
                self.current_polar = 0.0
                self.update_visualization()
            
            ps.ImGuiSameLine()
            if ps.ImGuiButton("Equator +X (0°, 90°)"):
                self.ui_azimuthal = 0.0
                self.ui_polar = 90.0
                self.current_azimuthal = 0.0
                self.current_polar = 90.0
                self.update_visualization()
            
            if ps.ImGuiButton("Equator +Y (90°, 90°)"):
                self.ui_azimuthal = 90.0
                self.ui_polar = 90.0
                self.current_azimuthal = 90.0
                self.current_polar = 90.0
                self.update_visualization()
            
            ps.ImGuiSameLine()
            if ps.ImGuiButton("South Pole (0°, 180°)"):
                self.ui_azimuthal = 0.0
                self.ui_polar = 180.0
                self.current_azimuthal = 0.0
                self.current_polar = 180.0
                self.update_visualization()
        
        ps.set_user_callback(ui_callback)

def main():
    parser = argparse.ArgumentParser(
        description="Interactive direction vector visualizer using polyscope")
    parser.add_argument("--azimuthal", "-a", type=float, default=0.0,
                       help="Initial azimuthal angle in degrees [-180, 180] (default: 0)")
    parser.add_argument("--polar", "-p", type=float, default=90.0,
                       help="Initial polar angle in degrees [0, 180] (default: 90)")
    parser.add_argument("--length", "-l", type=float, default=1.0,
                       help="Vector length (default: 1.0)")
    
    args = parser.parse_args()
    
    # Validate input ranges
    if not (-180 <= args.azimuthal <= 180):
        print("Warning: Azimuthal angle should be in range [-180, 180]. Clamping...")
        args.azimuthal = max(-180, min(180, args.azimuthal))
    
    if not (0 <= args.polar <= 180):
        print("Warning: Polar angle should be in range [0, 180]. Clamping...")
        args.polar = max(0, min(180, args.polar))
    
    # Initialize polyscope
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")  # Remove ground plane for clearer view
    
    # Create visualizer
    visualizer = DirectionVectorVisualizer()
    visualizer.current_azimuthal = args.azimuthal
    visualizer.current_polar = args.polar
    visualizer.ui_azimuthal = args.azimuthal
    visualizer.ui_polar = args.polar
    visualizer.vector_length = args.length
    
    # Setup UI and initial visualization
    visualizer.create_ui()
    visualizer.update_visualization()
    
    print("=== Direction Vector Visualizer ===")
    print("Controls:")
    print("  - Use sliders to adjust azimuthal (φ) and polar (θ) angles")
    print("  - Click 'Update Vector' to apply changes")
    print("  - Toggle display options for sphere and coordinate system")
    print("  - Try quick presets for common directions")
    print("\nSpherical Coordinate System:")
    print("  - Azimuthal (φ): -180° to 180° (longitude, rotation around Z-axis)")
    print("  - Polar (θ): 0° to 180° (latitude from north pole)")
    print("  - Yellow line: Meridian (constant azimuth)")
    print("  - Cyan line: Parallel (constant polar angle)")
    print("\nClose the window to exit.")
    
    # Run polyscope
    ps.show()

if __name__ == "__main__":
    main() 