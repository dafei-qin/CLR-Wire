#!/usr/bin/env python3
"""
Interactive B-spline surface viewer using the dataset_bspline class.

This script provides a simple interface to visualize B-spline surfaces
from the dataset with navigation and filtering capabilities.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import argparse

from src.dataset.dataset_bspline import dataset_bspline
from utils.surface import build_bspline_surface


class BSplineDatasetViewer:
    """Interactive viewer for B-spline surfaces using dataset_bspline."""
    
    def __init__(self, data_path):
        """
        Initialize the viewer.
        
        Args:
            data_path: Path to directory containing .npy B-spline files
        """
        print("="*80)
        print("B-SPLINE DATASET VIEWER")
        print("="*80)
        
        # Load dataset
        print(f"\n📂 Loading dataset from {data_path}...")
        self.dataset = dataset_bspline(data_path=data_path, replica=1)
        print(f"✓ Found {len(self.dataset)} surfaces")
        
        # Current state
        self.current_index = 0
        self.current_surface_data = None
        
        # Polyscope handles
        self.ps_mesh = None
        self.ps_control_points = None
        self.ps_control_mesh_u = None
        self.ps_control_mesh_v = None
        
        # Visualization settings
        self.show_control_points = True  # Show by default
        self.show_control_mesh = True    # Show by default
        self.control_point_radius = 0.02
        self.mesh_quality = 0.1
        self.surface_transparency = 0.8
        
        # Navigation settings
        self.auto_update = False
        
        print(f"✓ Viewer initialized with {len(self.dataset)} surfaces")
        
    def load_and_display_surface(self, index):
        """
        Load and display surface at the given index.
        
        Args:
            index: Index in the dataset
        """
        if len(self.dataset) == 0:
            print("No surfaces to display!")
            return
        
        # Clamp index
        index = max(0, min(index, len(self.dataset) - 1))
        self.current_index = index
        
        # Load surface data
        data_path_file = self.dataset.data_names[index]
        
        try:
            u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, \
                is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, \
                u_mults_list, v_mults_list, poles, valid = self.dataset.load_data(data_path_file)
            
            print(f"\n[{index + 1}/{len(self.dataset)}] Loading surface...")
            print(f"   File: {Path(data_path_file).name}")
            print(f"   Degrees: (u={u_degree}, v={v_degree})")
            print(f"   Control points: {num_poles_u} × {num_poles_v} = {num_poles_u * num_poles_v}")
            print(f"   Knots: U={num_knots_u}, V={num_knots_v}")
            print(f"   Periodic: U={bool(is_u_periodic)}, V={bool(is_v_periodic)}")
            
            # Prepare data for build_bspline_surface
            surface_data = {
                'scalar': [
                    u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v
                ] + list(u_knots_list) + list(v_knots_list) + list(u_mults_list) + list(v_mults_list),
                'u_periodic': bool(is_u_periodic),
                'v_periodic': bool(is_v_periodic),
                'poles': poles.tolist()  # Convert numpy array to list
            }
            
            # Build surface using pythonOCC
            print(f"   Building surface mesh...")
            occ_face, vertices, faces, attr_str = build_bspline_surface(
                surface_data, 
                tol=self.mesh_quality, 
                normalize_knots=True, 
                normalize_surface=True
            )
            
            # Remove old surface
            if self.ps_mesh is not None:
                ps.remove_surface_mesh("bspline_surface")
                self.ps_mesh = None
            
            # Display surface
            self.ps_mesh = ps.register_surface_mesh(
                "bspline_surface",
                vertices,
                faces,
                enabled=True,
                transparency=self.surface_transparency
            )
            self.ps_mesh.set_color((0.3, 0.7, 0.9))
            
            # Store current surface data
            self.current_surface_data = {
                'u_degree': u_degree,
                'v_degree': v_degree,
                'num_poles_u': num_poles_u,
                'num_poles_v': num_poles_v,
                'num_knots_u': num_knots_u,
                'num_knots_v': num_knots_v,
                'is_u_periodic': is_u_periodic,
                'is_v_periodic': is_v_periodic,
                'u_knots': u_knots_list,
                'v_knots': v_knots_list,
                'u_mults': u_mults_list,
                'v_mults': v_mults_list,
                'poles': poles,
                'file_path': data_path_file,
                'vertices': vertices,
                'faces': faces
            }
            
            # Update control visualizations
            self._update_control_visualizations(poles)
            
            print(f"   ✓ Surface displayed: {len(vertices)} vertices, {len(faces)} faces")
            
        except Exception as e:
            print(f"   ✗ Error loading surface: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_control_visualizations(self, poles):
        """Update control point and mesh visualizations."""
        # Remove existing visualizations
        if self.ps_control_points is not None:
            ps.remove_point_cloud("control_points")
            self.ps_control_points = None
        if self.ps_control_mesh_u is not None:
            ps.remove_curve_network("control_mesh_u")
            self.ps_control_mesh_u = None
        if self.ps_control_mesh_v is not None:
            ps.remove_curve_network("control_mesh_v")
            self.ps_control_mesh_v = None
        
        if not self.show_control_points and not self.show_control_mesh:
            return
        
        # Extract control points from poles array
        num_u, num_v, _ = poles.shape
        
        points = []
        for i in range(num_u):
            for j in range(num_v):
                x, y, z, w = poles[i, j]
                points.append([x, y, z])
        
        points = np.array(points)
        
        # Visualize control points
        if self.show_control_points and len(points) > 0:
            self.ps_control_points = ps.register_point_cloud(
                "control_points",
                points,
                enabled=True,
                radius=self.control_point_radius
            )
            self.ps_control_points.set_color((1.0, 0.2, 0.2))  # Red
        
        # Visualize control mesh
        if self.show_control_mesh and len(points) > 0:
            # Build edges in U direction
            u_edges = []
            for i in range(num_u):
                for j in range(num_v - 1):
                    idx1 = i * num_v + j
                    idx2 = i * num_v + (j + 1)
                    u_edges.append([idx1, idx2])
            
            if len(u_edges) > 0:
                self.ps_control_mesh_u = ps.register_curve_network(
                    "control_mesh_u",
                    points,
                    np.array(u_edges),
                    enabled=True
                )
                self.ps_control_mesh_u.set_color((0.3, 0.8, 0.3))  # Green
                self.ps_control_mesh_u.set_radius(self.control_point_radius * 0.5)
            
            # Build edges in V direction
            v_edges = []
            for i in range(num_u - 1):
                for j in range(num_v):
                    idx1 = i * num_v + j
                    idx2 = (i + 1) * num_v + j
                    v_edges.append([idx1, idx2])
            
            if len(v_edges) > 0:
                self.ps_control_mesh_v = ps.register_curve_network(
                    "control_mesh_v",
                    points,
                    np.array(v_edges),
                    enabled=True
                )
                self.ps_control_mesh_v.set_color((0.3, 0.3, 0.8))  # Blue
                self.ps_control_mesh_v.set_radius(self.control_point_radius * 0.5)
    
    def ui_callback(self):
        """ImGui callback for the UI."""
        
        # Navigation panel
        if psim.TreeNode("🎮 Navigation"):
            psim.TextUnformatted(f"Surface: {self.current_index + 1} / {len(self.dataset)}")
            
            # Slider for navigation
            changed, new_index = psim.SliderInt(
                "##surf_slider",
                self.current_index,
                v_min=0,
                v_max=max(0, len(self.dataset) - 1)
            )
            
            if changed and self.auto_update:
                self.load_and_display_surface(new_index)
            
            # Navigation buttons
            psim.SameLine()
            if psim.Button("◀ Prev"):
                self.load_and_display_surface(self.current_index - 1)
            
            psim.SameLine()
            if psim.Button("Next ▶"):
                self.load_and_display_surface(self.current_index + 1)
            
            psim.SameLine()
            if psim.Button("⟲ Reload"):
                self.load_and_display_surface(self.current_index)
            
            # Auto-update toggle
            _, self.auto_update = psim.Checkbox("Auto-update on slider", self.auto_update)
            
            psim.TreePop()
        
        # Visualization settings
        if psim.TreeNode("🎨 Visualization"):
            
            # Surface transparency
            changed, self.surface_transparency = psim.SliderFloat(
                "Surface Transparency",
                self.surface_transparency,
                v_min=0.0,
                v_max=1.0
            )
            if changed and self.ps_mesh is not None:
                self.ps_mesh.set_transparency(self.surface_transparency)
            
            # Mesh quality
            changed, self.mesh_quality = psim.SliderFloat(
                "Mesh Quality",
                self.mesh_quality,
                v_min=0.01,
                v_max=1.0
            )
            if changed:
                psim.TextUnformatted("  (Click Reload to apply)")
            
            psim.Separator()
            
            # Control points toggle
            changed, self.show_control_points = psim.Checkbox(
                "Show Control Points", 
                self.show_control_points
            )
            if changed and self.current_surface_data is not None:
                self._update_control_visualizations(self.current_surface_data['poles'])
            
            # Control mesh toggle
            changed, self.show_control_mesh = psim.Checkbox(
                "Show Control Mesh", 
                self.show_control_mesh
            )
            if changed and self.current_surface_data is not None:
                self._update_control_visualizations(self.current_surface_data['poles'])
            
            # Control point radius
            if self.show_control_points or self.show_control_mesh:
                changed, self.control_point_radius = psim.SliderFloat(
                    "Control Point Radius",
                    self.control_point_radius,
                    v_min=0.001,
                    v_max=0.1
                )
                if changed and self.current_surface_data is not None:
                    self._update_control_visualizations(self.current_surface_data['poles'])
            
            psim.TreePop()
        
        # Surface information
        if psim.TreeNode("ℹ️ Surface Info"):
            if self.current_surface_data is not None:
                data = self.current_surface_data
                
                psim.TextUnformatted(f"File: {Path(data['file_path']).name}")
                psim.Separator()
                
                psim.TextUnformatted(f"Degrees: (u={data['u_degree']}, v={data['v_degree']})")
                psim.TextUnformatted(f"Control Points: {data['num_poles_u']} × {data['num_poles_v']} = {data['num_poles_u'] * data['num_poles_v']}")
                psim.Separator()
                
                psim.TextUnformatted(f"Knots: U={data['num_knots_u']}, V={data['num_knots_v']}")
                psim.TextUnformatted(f"U Knots: {', '.join([f'{k:.3f}' for k in data['u_knots']])}")
                psim.TextUnformatted(f"V Knots: {', '.join([f'{k:.3f}' for k in data['v_knots']])}")
                psim.Separator()
                
                psim.TextUnformatted(f"U Multiplicities: {', '.join(map(str, map(int, data['u_mults'])))}")
                psim.TextUnformatted(f"V Multiplicities: {', '.join(map(str, map(int, data['v_mults'])))}")
                psim.Separator()
                
                psim.TextUnformatted(f"U Periodic: {'Yes' if data['is_u_periodic'] else 'No'}")
                psim.TextUnformatted(f"V Periodic: {'Yes' if data['is_v_periodic'] else 'No'}")
                psim.Separator()
                
                psim.TextUnformatted(f"Mesh: {len(data['vertices'])} vertices, {len(data['faces'])} faces")
                
                # Weight statistics
                poles = data['poles']
                weights = poles[:, :, 3].flatten()
                is_rational = not np.allclose(weights, 1.0)
                
                psim.Separator()
                psim.TextUnformatted(f"Rational: {'Yes' if is_rational else 'No'}")
                if is_rational:
                    psim.TextUnformatted(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
                    psim.TextUnformatted(f"  Weight mean: {weights.mean():.4f}")
                    psim.TextUnformatted(f"  Weight std: {weights.std():.4f}")
            else:
                psim.TextUnformatted("No surface loaded")
            
            psim.TreePop()
        
        # Dataset statistics
        if psim.TreeNode("📊 Dataset Info"):
            psim.TextUnformatted(f"Total surfaces in dataset: {len(self.dataset)}")
            psim.TextUnformatted(f"Current index: {self.current_index}")
            
            psim.TreePop()
    
    def run(self):
        """Run the interactive viewer."""
        print("\n🚀 Starting interactive viewer...")
        
        if len(self.dataset) == 0:
            print("❌ No surfaces to display!")
            return
        
        # Initialize polyscope
        ps.init()
        ps.set_program_name("B-Spline Dataset Viewer")
        ps.set_verbosity(0)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        
        # Set user callback
        ps.set_user_callback(self.ui_callback)
        
        # Load initial surface
        self.load_and_display_surface(0)
        
        # Show the viewer
        print("\n✓ Viewer ready!")
        print("\nControls:")
        print("  - Use the slider or Prev/Next buttons to navigate")
        print("  - Toggle control points and mesh visualization")
        print("  - Adjust mesh quality and transparency")
        print("\nPress Q or close the window to exit.")
        
        ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize B-spline surfaces from dataset"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to directory containing .npy B-spline files"
    )
    
    args = parser.parse_args()
    
    # Create and run viewer
    viewer = BSplineDatasetViewer(data_path=args.data_path)
    viewer.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

