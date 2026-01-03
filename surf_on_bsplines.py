#!/usr/bin/env python3
"""
Interactive B-spline surface viewer with filtering and navigation.

Features:
- Load complete B-spline dataset
- Filter surfaces by various criteria (degrees, knots, rational, etc.)
- Navigate through filtered surfaces with a slider
- Real-time visualization using polyscope
"""

import argparse
import pandas as pd
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
import json

# Import surface building functions
from myutils.surface import build_bspline_surface


class BSplineSurfaceViewer:
    """Interactive viewer for B-spline surfaces with filtering."""
    
    def __init__(self, data_path):
        """Initialize the viewer and load data."""
        print("="*80)
        print("B-SPLINE SURFACE VIEWER")
        print("="*80)
        
        # Load data
        print(f"\n📂 Loading data from {data_path}...")
        self.data_path = data_path
        self.df_full = self._load_data(data_path)
        print(f"✓ Loaded {len(self.df_full):,} surfaces")
        
        # Current state
        self.df_filtered = self.df_full.copy()
        self.current_index = 0
        self.current_surface = None
        self.ps_mesh_unnormalized = None
        self.ps_mesh_normalized = None
        self.ps_mesh_normalized_surface = None
        self.ps_control_points = None
        self.ps_control_mesh_u = None
        self.ps_control_mesh_v = None
        
        # Filter settings
        self.filter_settings = {
            # Degree filters
            'u_degree_min': int(self.df_full['u_degree'].min()),
            'u_degree_max': int(self.df_full['u_degree'].max()),
            'v_degree_min': int(self.df_full['v_degree'].min()),
            'v_degree_max': int(self.df_full['v_degree'].max()),
            
            # Rationality filter
            'show_rational': True,
            'show_polynomial': True,
            
            # Periodicity filter
            'show_u_periodic': True,
            'show_u_nonperiodic': True,
            'show_v_periodic': True,
            'show_v_nonperiodic': True,
            
            # Control points filter
            'poles_u_min': int(self.df_full['num_poles_u'].min()),
            'poles_u_max': int(self.df_full['num_poles_u'].max()),
            'poles_v_min': int(self.df_full['num_poles_v'].min()),
            'poles_v_max': int(self.df_full['num_poles_v'].max()),
            'total_poles_min': int(self.df_full['total_poles'].min()),
            'total_poles_max': int(self.df_full['total_poles'].max()),
            
            # Knot structure filter
            'show_bezier': True,
            'show_general_bspline': True,
            
            # Weight statistics (for rational surfaces)
            'weight_std_min': 0.0,
            'weight_std_max': float(self.df_full['std_weight'].max()),
            
            # Multiplicity pattern filter
            'enable_mult_filter': False,
            'u_mult_pattern': '',  # e.g., "4,4" or "4,2,2,4"
            'v_mult_pattern': '',  # e.g., "3,3" or "3,2,3"
        }
        
        # UI state
        self.show_info_window = True
        self.show_filter_window = True
        self.auto_update = True
        self.mesh_quality = 0.1  # Mesh refinement parameter
        
        # Visualization toggles
        self.show_control_points = False
        self.show_control_mesh = False
        self.control_point_radius = 0.02
        
        # Statistics
        self._compute_statistics()
        
        print(f"\n✓ Viewer initialized")
        print(f"   Total surfaces: {len(self.df_full):,}")
        print(f"   Degree range U: [{self.filter_settings['u_degree_min']}, {self.filter_settings['u_degree_max']}]")
        print(f"   Degree range V: [{self.filter_settings['v_degree_min']}, {self.filter_settings['v_degree_max']}]")
    
    def _load_data(self, path):
        """Load B-spline data from parquet or CSV."""
        path = Path(path)
        
        if path.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix == '.csv':
            print("   Converting CSV to dataframe (this may take a moment)...")
            df = pd.read_csv(path)
            # Convert JSON strings to lists
            array_columns = ['u_knots', 'v_knots', 'u_mults', 'v_mults', 'poles_flat']
            for col in array_columns:
                if col in df.columns:
                    df[col] = df[col].apply(json.loads)
            return df
        elif path.suffix == '.pkl':
            return pd.read_pickle(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        df = self.df_full
        
        self.stats = {
            'total': len(df),
            'rational': int(df['is_rational'].sum()),
            'polynomial': int((~df['is_rational']).sum()),
            'u_periodic': int(df['u_periodic'].sum()),
            'v_periodic': int(df['v_periodic'].sum()),
            'degree_distribution': df.groupby(['u_degree', 'v_degree']).size().sort_values(ascending=False).head(10),
            'poles_distribution': df.groupby(['num_poles_u', 'num_poles_v']).size().sort_values(ascending=False).head(10),
        }
        
        # Classify Bezier vs general B-spline
        bezier_mask = self._is_bezier_patch(df)
        self.stats['bezier'] = int(bezier_mask.sum())
        self.stats['general_bspline'] = int((~bezier_mask).sum())
    
    def _is_bezier_patch(self, df):
        """Check if surfaces are Bezier patches (2 knots with full multiplicity)."""
        # A Bezier patch has exactly 2 knots in each direction with multiplicity = degree + 1
        is_bezier = (df['num_knots_u'] == 2) & (df['num_knots_v'] == 2)
        # We'd need to check multiplicities too, but that requires unpacking the lists
        # For now, we use num_knots as a proxy
        return is_bezier
    
    def _filter_by_multiplicities(self, df, u_pattern, v_pattern):
        """
        Filter surfaces by multiplicity patterns.
        
        Args:
            df: DataFrame to filter
            u_pattern: String like "4,4" or "4,2,2,4" for U direction
            v_pattern: String like "3,3" or "3,2,3" for V direction
        
        Returns:
            Boolean mask for matching surfaces
        """
        mask = np.ones(len(df), dtype=bool)
        
        # Parse U multiplicity pattern
        if u_pattern.strip():
            try:
                u_target = [int(x.strip()) for x in u_pattern.split(',')]
                u_mask = df['u_mults'].apply(lambda x: list(x) == u_target)
                mask &= u_mask
            except ValueError:
                print(f"Warning: Invalid U multiplicity pattern: {u_pattern}")
        
        # Parse V multiplicity pattern
        if v_pattern.strip():
            try:
                v_target = [int(x.strip()) for x in v_pattern.split(',')]
                v_mask = df['v_mults'].apply(lambda x: list(x) == v_target)
                mask &= v_mask
            except ValueError:
                print(f"Warning: Invalid V multiplicity pattern: {v_pattern}")
        
        return mask
    
    def apply_filters(self):
        """Apply all filters to the dataset."""
        df = self.df_full.copy()
        s = self.filter_settings
        
        # Degree filters
        df = df[(df['u_degree'] >= s['u_degree_min']) & (df['u_degree'] <= s['u_degree_max'])]
        df = df[(df['v_degree'] >= s['v_degree_min']) & (df['v_degree'] <= s['v_degree_max'])]
        
        # Rationality filter
        rational_mask = df['is_rational']
        show_mask = np.zeros(len(df), dtype=bool)
        if s['show_rational']:
            show_mask |= rational_mask
        if s['show_polynomial']:
            show_mask |= ~rational_mask
        df = df[show_mask]
        
        # Periodicity filter
        u_periodic_mask = df['u_periodic']
        show_u_mask = np.zeros(len(df), dtype=bool)
        if s['show_u_periodic']:
            show_u_mask |= u_periodic_mask
        if s['show_u_nonperiodic']:
            show_u_mask |= ~u_periodic_mask
        df = df[show_u_mask]
        
        v_periodic_mask = df['v_periodic']
        show_v_mask = np.zeros(len(df), dtype=bool)
        if s['show_v_periodic']:
            show_v_mask |= v_periodic_mask
        if s['show_v_nonperiodic']:
            show_v_mask |= ~v_periodic_mask
        df = df[show_v_mask]
        
        # Control points filter
        df = df[(df['num_poles_u'] >= s['poles_u_min']) & (df['num_poles_u'] <= s['poles_u_max'])]
        df = df[(df['num_poles_v'] >= s['poles_v_min']) & (df['num_poles_v'] <= s['poles_v_max'])]
        df = df[(df['total_poles'] >= s['total_poles_min']) & (df['total_poles'] <= s['total_poles_max'])]
        
        # Knot structure filter
        bezier_mask = self._is_bezier_patch(df)
        show_knot_mask = np.zeros(len(df), dtype=bool)
        if s['show_bezier']:
            show_knot_mask |= bezier_mask
        if s['show_general_bspline']:
            show_knot_mask |= ~bezier_mask
        df = df[show_knot_mask]
        
        # Weight statistics filter (for rational surfaces)
        df = df[(df['std_weight'] >= s['weight_std_min']) & (df['std_weight'] <= s['weight_std_max'])]
        
        # Multiplicity pattern filter
        if s['enable_mult_filter']:
            mult_mask = self._filter_by_multiplicities(df, s['u_mult_pattern'], s['v_mult_pattern'])
            df = df[mult_mask]
        
        self.df_filtered = df.reset_index(drop=True)
        
        # Reset current index if out of bounds
        if self.current_index >= len(self.df_filtered):
            self.current_index = max(0, len(self.df_filtered) - 1)
        
        return len(self.df_filtered)
    
    def load_and_display_surface(self, index):
        """Load and display the surface at the given index."""
        if len(self.df_filtered) == 0:
            print("No surfaces match the current filters!")
            if self.ps_mesh_unnormalized is not None:
                ps.remove_surface_mesh("bspline_surface_unnormalized")
                self.ps_mesh_unnormalized = None
            if self.ps_mesh_normalized is not None:
                ps.remove_surface_mesh("bspline_surface_normalized")
                self.ps_mesh_normalized = None
            if self.ps_mesh_normalized_surface is not None:
                ps.remove_surface_mesh("bspline_surface_normalized_surface")
                self.ps_mesh_normalized_surface = None
            return
        
        # Clamp index
        index = max(0, min(index, len(self.df_filtered) - 1))
        self.current_index = index
        
        row = self.df_filtered.iloc[index]
        
        # Convert row to surface data dict compatible with build_bspline_surface
        surface_data = {
            'scalar': [
                row['u_degree'].item(),
                row['v_degree'].item(),
                row['num_poles_u'].item(),
                row['num_poles_v'].item(),
                row['num_knots_u'].item(),
                row['num_knots_v'].item(),
            ] + row['u_knots'] + row['v_knots'] + row['u_mults'] + row['v_mults'],
            'u_periodic': row['u_periodic'].item(),
            'v_periodic': row['v_periodic'].item(),
            'poles': self._reshape_poles(row),
        }
        
        try:
            # Build the surface using pythonOCC
            print(f"\n[{index+1}/{len(self.df_filtered)}] Building surface...")
            print(f"   File: {Path(row['file_path']).name}")
            print(f"   Degrees: ({row['u_degree']}, {row['v_degree']})")
            print(f"   Control points: {row['num_poles_u']} × {row['num_poles_v']}")
            print(f"   Rational: {row['is_rational']}")
            
            # Build surface with different normalization settings
            print(f"   Building with normalize_knots=False, normalize_surface=False...")
            occ_face_unnorm, vertices_unnorm, faces_unnorm, _ = build_bspline_surface(
                surface_data, tol=self.mesh_quality, normalize_knots=False, normalize_surface=False
            )
            
            print(f"   Building with normalize_knots=True, normalize_surface=False...")
            occ_face_norm, vertices_norm, faces_norm, _ = build_bspline_surface(
                surface_data, tol=self.mesh_quality, normalize_knots=True, normalize_surface=False
            )
            
            print(f"   Building with normalize_knots=True, normalize_surface=True...")
            occ_face_norm_surf, vertices_norm_surf, faces_norm_surf, _ = build_bspline_surface(
                surface_data, tol=self.mesh_quality, normalize_knots=True, normalize_surface=True
            )
            
            # Display surface 1: unnormalized
            if self.ps_mesh_unnormalized is not None:
                ps.remove_surface_mesh("bspline_surface_unnormalized")
            
            self.ps_mesh_unnormalized = ps.register_surface_mesh(
                "bspline_surface_unnormalized",
                vertices_unnorm,
                faces_unnorm,
                enabled=True
            )
            
            # Display surface 2: normalize_knots only
            if self.ps_mesh_normalized is not None:
                ps.remove_surface_mesh("bspline_surface_normalized")
            
            self.ps_mesh_normalized = ps.register_surface_mesh(
                "bspline_surface_normalized",
                vertices_norm,
                faces_norm,
                enabled=True
            )
            
            # Display surface 3: normalize_knots + normalize_surface
            if self.ps_mesh_normalized_surface is not None:
                ps.remove_surface_mesh("bspline_surface_normalized_surface")
            
            self.ps_mesh_normalized_surface = ps.register_surface_mesh(
                "bspline_surface_normalized_surface",
                vertices_norm_surf,
                faces_norm_surf,
                enabled=True
            )
            
            # Set colors - different for each version
            self.ps_mesh_unnormalized.set_color((0.9, 0.3, 0.3))  # Red
            self.ps_mesh_normalized.set_color((0.3, 0.9, 0.3))  # Green
            self.ps_mesh_normalized_surface.set_color((0.3, 0.3, 0.9))  # Blue
            
            self.current_surface = row
            
            # Visualize control points and mesh (for unnormalized version)
            self._update_control_visualizations(surface_data['poles'])
            
            print(f"   ✓ Surfaces displayed (overlapping):")
            print(f"     Red (no normalization): {len(vertices_unnorm)} vertices, {len(faces_unnorm)} faces")
            print(f"     Green (normalize_knots only): {len(vertices_norm)} vertices, {len(faces_norm)} faces")
            print(f"     Blue (normalize_knots + normalize_surface): {len(vertices_norm_surf)} vertices, {len(faces_norm_surf)} faces")
            
        except Exception as e:
            print(f"   ✗ Error building surface: {e}")
            import traceback
            traceback.print_exc()
    
    def _reshape_poles(self, row):
        """Reshape flat poles array to 2D grid."""
        poles_flat = row['poles_flat']
        num_poles_u = int(row['num_poles_u'])
        num_poles_v = int(row['num_poles_v'])
        
        poles_grid = []
        idx = 0
        for i in range(num_poles_u):
            row_poles = []
            for j in range(num_poles_v):
                x = poles_flat[idx]
                y = poles_flat[idx + 1]
                z = poles_flat[idx + 2]
                w = poles_flat[idx + 3]
                row_poles.append([x, y, z, w])
                idx += 4
            poles_grid.append(row_poles)
        
        return poles_grid
    
    def _update_control_visualizations(self, poles_grid):
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
        
        # Extract control points
        num_u = len(poles_grid)
        num_v = len(poles_grid[0]) if num_u > 0 else 0
        
        points = []
        for i in range(num_u):
            for j in range(num_v):
                x, y, z, w = poles_grid[i][j]
                # Use homogeneous coordinates (multiply by weight for rational surfaces)
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
        
        # Main control window
        if psim.TreeNode("🎮 Surface Navigation"):
            
            # Navigation controls
            psim.TextUnformatted(f"Surface: {self.current_index + 1} / {len(self.df_filtered)}")
            
            # Slider for navigation
            changed, new_index = psim.SliderInt(
                "##surf_slider",
                self.current_index,
                v_min=0,
                v_max=max(0, len(self.df_filtered) - 1)
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
            
            # Mesh quality
            changed, self.mesh_quality = psim.SliderFloat(
                "Mesh Quality",
                self.mesh_quality,
                v_min=0.01,
                v_max=1.0
            )
            if changed:
                psim.TextUnformatted("Click Reload to apply")
            
            psim.Separator()
            psim.TextUnformatted("Visualization Options:")
            
            # Control points toggle
            changed, self.show_control_points = psim.Checkbox("Show Control Points", self.show_control_points)
            if changed:
                self._update_control_visualizations(self._reshape_poles(self.current_surface) if self.current_surface is not None else [])
            
            # Control mesh toggle
            changed, self.show_control_mesh = psim.Checkbox("Show Control Mesh", self.show_control_mesh)
            if changed:
                self._update_control_visualizations(self._reshape_poles(self.current_surface) if self.current_surface is not None else [])
            
            # Control point radius
            if self.show_control_points or self.show_control_mesh:
                changed, self.control_point_radius = psim.SliderFloat(
                    "Control Point Radius",
                    self.control_point_radius,
                    v_min=0.001,
                    v_max=0.1
                )
                if changed:
                    self._update_control_visualizations(self._reshape_poles(self.current_surface) if self.current_surface is not None else [])
            
            psim.TreePop()
        
        # Filter window
        if psim.TreeNode("🔍 Filters"):
            
            filters_changed = False
            s = self.filter_settings
            
            # Degree filters
            if psim.TreeNode("Degree Filters"):
                psim.TextUnformatted(f"Full range U: [{self.df_full['u_degree'].min()}, {self.df_full['u_degree'].max()}]")
                changed, s['u_degree_min'] = psim.SliderInt(
                    "U Degree Min", s['u_degree_min'],
                    v_min=int(self.df_full['u_degree'].min()),
                    v_max=int(self.df_full['u_degree'].max())
                )
                filters_changed |= changed
                
                changed, s['u_degree_max'] = psim.SliderInt(
                    "U Degree Max", s['u_degree_max'],
                    v_min=int(self.df_full['u_degree'].min()),
                    v_max=int(self.df_full['u_degree'].max())
                )
                filters_changed |= changed
                
                psim.TextUnformatted(f"Full range V: [{self.df_full['v_degree'].min()}, {self.df_full['v_degree'].max()}]")
                changed, s['v_degree_min'] = psim.SliderInt(
                    "V Degree Min", s['v_degree_min'],
                    v_min=int(self.df_full['v_degree'].min()),
                    v_max=int(self.df_full['v_degree'].max())
                )
                filters_changed |= changed
                
                changed, s['v_degree_max'] = psim.SliderInt(
                    "V Degree Max", s['v_degree_max'],
                    v_min=int(self.df_full['v_degree'].min()),
                    v_max=int(self.df_full['v_degree'].max())
                )
                filters_changed |= changed
                
                psim.TreePop()
            
            # Rationality filter
            if psim.TreeNode("Rationality"):
                changed, s['show_rational'] = psim.Checkbox("Show Rational", s['show_rational'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['rational']:,})")
                
                changed, s['show_polynomial'] = psim.Checkbox("Show Polynomial", s['show_polynomial'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['polynomial']:,})")
                
                psim.TreePop()
            
            # Periodicity filter
            if psim.TreeNode("Periodicity"):
                changed, s['show_u_periodic'] = psim.Checkbox("U Periodic", s['show_u_periodic'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['u_periodic']:,})")
                
                changed, s['show_u_nonperiodic'] = psim.Checkbox("U Non-periodic", s['show_u_nonperiodic'])
                filters_changed |= changed
                
                changed, s['show_v_periodic'] = psim.Checkbox("V Periodic", s['show_v_periodic'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['v_periodic']:,})")
                
                changed, s['show_v_nonperiodic'] = psim.Checkbox("V Non-periodic", s['show_v_nonperiodic'])
                filters_changed |= changed
                
                psim.TreePop()
            
            # Control points filter
            if psim.TreeNode("Control Points"):
                changed, s['poles_u_min'] = psim.SliderInt(
                    "Poles U Min", s['poles_u_min'],
                    v_min=int(self.df_full['num_poles_u'].min()),
                    v_max=int(self.df_full['num_poles_u'].max())
                )
                filters_changed |= changed
                
                changed, s['poles_u_max'] = psim.SliderInt(
                    "Poles U Max", s['poles_u_max'],
                    v_min=int(self.df_full['num_poles_u'].min()),
                    v_max=int(self.df_full['num_poles_u'].max())
                )
                filters_changed |= changed
                
                changed, s['poles_v_min'] = psim.SliderInt(
                    "Poles V Min", s['poles_v_min'],
                    v_min=int(self.df_full['num_poles_v'].min()),
                    v_max=int(self.df_full['num_poles_v'].max())
                )
                filters_changed |= changed
                
                changed, s['poles_v_max'] = psim.SliderInt(
                    "Poles V Max", s['poles_v_max'],
                    v_min=int(self.df_full['num_poles_v'].min()),
                    v_max=int(self.df_full['num_poles_v'].max())
                )
                filters_changed |= changed
                
                changed, s['total_poles_min'] = psim.SliderInt(
                    "Total Poles Min", s['total_poles_min'],
                    v_min=int(self.df_full['total_poles'].min()),
                    v_max=int(self.df_full['total_poles'].max())
                )
                filters_changed |= changed
                
                changed, s['total_poles_max'] = psim.SliderInt(
                    "Total Poles Max", s['total_poles_max'],
                    v_min=int(self.df_full['total_poles'].min()),
                    v_max=int(self.df_full['total_poles'].max())
                )
                filters_changed |= changed
                
                psim.TreePop()
            
            # Knot structure filter
            if psim.TreeNode("Knot Structure"):
                changed, s['show_bezier'] = psim.Checkbox("Show Bézier Patches", s['show_bezier'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['bezier']:,})")
                
                changed, s['show_general_bspline'] = psim.Checkbox("Show General B-splines", s['show_general_bspline'])
                filters_changed |= changed
                psim.SameLine()
                psim.TextUnformatted(f"({self.stats['general_bspline']:,})")
                
                psim.TreePop()
            
            # Weight statistics
            if psim.TreeNode("Weight Statistics (Rational)"):
                changed, s['weight_std_min'] = psim.SliderFloat(
                    "Weight Std Min", s['weight_std_min'],
                    v_min=0.0,
                    v_max=float(self.df_full['std_weight'].max())
                )
                filters_changed |= changed
                
                changed, s['weight_std_max'] = psim.SliderFloat(
                    "Weight Std Max", s['weight_std_max'],
                    v_min=0.0,
                    v_max=float(self.df_full['std_weight'].max())
                )
                filters_changed |= changed
                
                psim.TreePop()
            
            # Multiplicity pattern filter
            if psim.TreeNode("Multiplicity Patterns"):
                psim.TextUnformatted("Filter by exact knot multiplicity patterns")
                psim.TextUnformatted("Example: '4,4' for Bezier, '4,2,2,4' for C1")
                
                changed, s['enable_mult_filter'] = psim.Checkbox("Enable Multiplicity Filter", s['enable_mult_filter'])
                filters_changed |= changed
                
                if s['enable_mult_filter']:
                    psim.PushItemWidth(150)
                    changed, s['u_mult_pattern'] = psim.InputText("U Multiplicities", s['u_mult_pattern'])
                    filters_changed |= changed
                    psim.PopItemWidth()
                    
                    psim.PushItemWidth(150)
                    changed, s['v_mult_pattern'] = psim.InputText("V Multiplicities", s['v_mult_pattern'])
                    filters_changed |= changed
                    psim.PopItemWidth()
                    
                    psim.TextUnformatted("Examples:")
                    psim.TextUnformatted("  Simple Bezier: 4,4")
                    psim.TextUnformatted("  C1 continuity: 4,2,2,4")
                    psim.TextUnformatted("  C0 continuity: 4,3,3,4")
                    
                    # Show current surface multiplicities as reference
                    if self.current_surface is not None:
                        row = self.current_surface
                        psim.Separator()
                        psim.TextUnformatted("Current surface:")
                        psim.TextUnformatted(f"  U: {','.join(map(str, row['u_mults']))}")
                        psim.TextUnformatted(f"  V: {','.join(map(str, row['v_mults']))}")
                
                psim.TreePop()
            
            # Apply filters button
            if filters_changed or psim.Button("Apply Filters"):
                n_filtered = self.apply_filters()
                print(f"\n🔍 Filters applied: {n_filtered:,} surfaces match")
                if n_filtered > 0:
                    self.load_and_display_surface(0)
            
            psim.SameLine()
            if psim.Button("Reset Filters"):
                # Reset all filters to defaults
                self.filter_settings['u_degree_min'] = int(self.df_full['u_degree'].min())
                self.filter_settings['u_degree_max'] = int(self.df_full['u_degree'].max())
                self.filter_settings['v_degree_min'] = int(self.df_full['v_degree'].min())
                self.filter_settings['v_degree_max'] = int(self.df_full['v_degree'].max())
                self.filter_settings['show_rational'] = True
                self.filter_settings['show_polynomial'] = True
                self.filter_settings['show_u_periodic'] = True
                self.filter_settings['show_u_nonperiodic'] = True
                self.filter_settings['show_v_periodic'] = True
                self.filter_settings['show_v_nonperiodic'] = True
                self.filter_settings['poles_u_min'] = int(self.df_full['num_poles_u'].min())
                self.filter_settings['poles_u_max'] = int(self.df_full['num_poles_u'].max())
                self.filter_settings['poles_v_min'] = int(self.df_full['num_poles_v'].min())
                self.filter_settings['poles_v_max'] = int(self.df_full['num_poles_v'].max())
                self.filter_settings['total_poles_min'] = int(self.df_full['total_poles'].min())
                self.filter_settings['total_poles_max'] = int(self.df_full['total_poles'].max())
                self.filter_settings['show_bezier'] = True
                self.filter_settings['show_general_bspline'] = True
                self.filter_settings['weight_std_min'] = 0.0
                self.filter_settings['weight_std_max'] = float(self.df_full['std_weight'].max())
                self.filter_settings['enable_mult_filter'] = False
                self.filter_settings['u_mult_pattern'] = ''
                self.filter_settings['v_mult_pattern'] = ''
                
                self.apply_filters()
                self.load_and_display_surface(0)
            
            psim.TreePop()
        
        # Info window
        if psim.TreeNode("ℹ️ Current Surface Info"):
            if self.current_surface is not None:
                row = self.current_surface
                
                # Display legend for surface comparison
                psim.TextUnformatted("📍 Surface Comparison (overlapping):")
                psim.TextColored((0.9, 0.3, 0.3, 1.0), "  🔴 RED:")
                psim.TextUnformatted("    normalize_knots=False, normalize_surface=False")
                psim.TextColored((0.3, 0.9, 0.3, 1.0), "  🟢 GREEN:")
                psim.TextUnformatted("    normalize_knots=True, normalize_surface=False")
                psim.TextColored((0.3, 0.3, 0.9, 1.0), "  🔵 BLUE:")
                psim.TextUnformatted("    normalize_knots=True, normalize_surface=True")
                psim.Separator()
                
                psim.TextUnformatted(f"File: {Path(row['file_path']).name}")
                psim.TextUnformatted(f"File Index: {row['file_idx']}, Face Index: {row['face_idx']}")
                psim.Separator()
                
                psim.TextUnformatted(f"Degrees: ({row['u_degree']}, {row['v_degree']})")
                psim.TextUnformatted(f"Control Points: {row['num_poles_u']} × {row['num_poles_v']} = {row['total_poles']}")
                psim.TextUnformatted(f"Knots: U={row['num_knots_u']}, V={row['num_knots_v']}")
                psim.TextUnformatted(f"Multiplicities U: {','.join(map(str, row['u_mults']))}")
                psim.TextUnformatted(f"Multiplicities V: {','.join(map(str, row['v_mults']))}")
                psim.TextUnformatted(f"Knot Vector: U={','.join(map(str, row['u_knots']))}, V={','.join(map(str, row['v_knots']))}")
                psim.Separator()
                
                psim.TextUnformatted(f"Rational: {'Yes' if row['is_rational'] else 'No'}")
                if row['is_rational']:
                    psim.TextUnformatted(f"  Weight range: [{row['min_weight']:.4f}, {row['max_weight']:.4f}]")
                    psim.TextUnformatted(f"  Weight mean: {row['mean_weight']:.4f}")
                    psim.TextUnformatted(f"  Weight std: {row['std_weight']:.4f}")
                psim.Separator()
                
                psim.TextUnformatted(f"U Periodic: {'Yes' if row['u_periodic'] else 'No'}")
                psim.TextUnformatted(f"V Periodic: {'Yes' if row['v_periodic'] else 'No'}")
                
            else:
                psim.TextUnformatted("No surface loaded")
            
            psim.TreePop()
        
        # Statistics window
        if psim.TreeNode("📊 Dataset Statistics"):
            psim.TextUnformatted(f"Total surfaces: {self.stats['total']:,}")
            psim.TextUnformatted(f"Filtered surfaces: {len(self.df_filtered):,}")
            psim.Separator()
            
            psim.TextUnformatted(f"Rational: {self.stats['rational']:,} ({100*self.stats['rational']/self.stats['total']:.1f}%)")
            psim.TextUnformatted(f"Polynomial: {self.stats['polynomial']:,} ({100*self.stats['polynomial']/self.stats['total']:.1f}%)")
            psim.Separator()
            
            psim.TextUnformatted(f"Bézier patches: {self.stats['bezier']:,}")
            psim.TextUnformatted(f"General B-splines: {self.stats['general_bspline']:,}")
            psim.Separator()
            
            psim.TextUnformatted(f"U Periodic: {self.stats['u_periodic']:,}")
            psim.TextUnformatted(f"V Periodic: {self.stats['v_periodic']:,}")
            
            psim.TreePop()
    
    def run(self):
        """Run the interactive viewer."""
        print("\n🚀 Starting interactive viewer...")
        
        # Initialize polyscope
        ps.init()
        ps.set_program_name("B-Spline Surface Viewer")
        ps.set_verbosity(0)
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode("shadow_only")
        
        # Set user callback
        ps.set_user_callback(self.ui_callback)
        
        # Load initial surface
        self.apply_filters()
        if len(self.df_filtered) > 0:
            self.load_and_display_surface(0)
        
        # Show the viewer
        print("\n✓ Viewer ready!")
        print("\nControls:")
        print("  - Use the slider or Prev/Next buttons to navigate")
        print("  - Apply filters to narrow down surfaces")
        print("  - Click on a surface to see details")
        print("\nPress Q or close the window to exit.")
        
        ps.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive B-spline surface viewer"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="assets/bspline_complete_data.csv",
        help="Path to B-spline data (CSV, Parquet, or PKL)"
    )
    args = parser.parse_args()
    
    # Create and run viewer
    viewer = BSplineSurfaceViewer(args.data)
    viewer.run()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

