"""
Quick test for sample_simple_surface.py (without requiring actual dataset)

This creates synthetic surfaces and tests the sampling functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polyscope as ps

from src.dataset.dataset_v1 import SURFACE_TYPE_MAP
from src.tools.sample_simple_surface import sample_surface_uniform, recover_surface_dict


def create_synthetic_cylinder():
    """Create synthetic cylinder parameters."""
    P = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # UV: [sin(u_c), cos(u_c), u_half_norm, height, 0, 0, 0, 0]
    UV = np.array([0.0, 1.0, 0.0, 2.0, 0, 0, 0, 0], dtype=np.float32)
    
    # Scalar: [log(radius)]
    scalar = np.array([np.log(0.5)], dtype=np.float32)
    
    params = np.concatenate([P, D, X, UV, scalar])
    return params, SURFACE_TYPE_MAP['cylinder'], "Cylinder (r=0.5, h=2.0)"


def create_synthetic_sphere():
    """Create synthetic sphere parameters."""
    P = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # UV: [dir_x, dir_y, dir_z, u_h_norm, v_h_norm, 0, 0, 0]
    UV = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0, 0, 0], dtype=np.float32)
    
    # Scalar: [log(radius)]
    scalar = np.array([np.log(0.3)], dtype=np.float32)
    
    params = np.concatenate([P, D, X, UV, scalar])
    return params, SURFACE_TYPE_MAP['sphere'], "Sphere (r=0.3)"


def create_synthetic_plane():
    """Create synthetic plane parameters."""
    P = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # UV: [u_min, u_max, v_min, v_max, 0, 0, 0, 0]
    UV = np.array([-0.5, 0.5, -0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
    
    params = np.concatenate([P, D, X, UV])
    return params, SURFACE_TYPE_MAP['plane'], "Plane (1x1)"


def create_synthetic_torus():
    """Create synthetic torus parameters."""
    P = np.array([-2.0, 0.0, 0.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # UV: [sin_u, cos_u, u_half, sin_v, cos_v, v_half, 0, 0]
    UV = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0, 0], dtype=np.float32)
    
    # Scalar: [log(major_radius), log(minor_radius)]
    scalar = np.array([np.log(0.5), np.log(0.15)], dtype=np.float32)
    
    params = np.concatenate([P, D, X, UV, scalar])
    return params, SURFACE_TYPE_MAP['torus'], "Torus (R=0.5, r=0.15)"


def create_synthetic_cone():
    """Create synthetic cone parameters."""
    P = np.array([0.0, -2.0, 0.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # UV: [sin_u, cos_u, u_half, v_c, v_half, 0, 0, 0]
    UV = np.array([0.0, 1.0, 1.0, 0.5, 0.5, 0, 0, 0], dtype=np.float32)
    
    # Scalar: [semi_angle_norm, log(radius)]
    semi_angle = np.pi / 6  # 30 degrees
    scalar = np.array([semi_angle / (np.pi/2), np.log(0.2)], dtype=np.float32)
    
    params = np.concatenate([P, D, X, UV, scalar])
    return params, SURFACE_TYPE_MAP['cone'], "Cone (angle=30°, r=0.2)"


def main():
    print("=" * 60)
    print("Quick Test: sample_simple_surface.py")
    print("=" * 60)
    print()
    
    # Create synthetic surfaces
    surfaces = [
        create_synthetic_cylinder(),
        create_synthetic_sphere(),
        create_synthetic_plane(),
        create_synthetic_torus(),
        create_synthetic_cone(),
    ]
    
    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("tile")
    
    print("Sampling synthetic surfaces...\n")
    
    # Sample each surface
    for i, (params, surf_type_idx, description) in enumerate(surfaces):
        try:
            # Sample with different densities for visualization
            num_u, num_v = 32, 32
            
            points = sample_surface_uniform(
                params=params,
                surface_type_idx=surf_type_idx,
                num_u=num_u,
                num_v=num_v,
                flatten=True,
            )
            
            # Register in polyscope
            cloud = ps.register_point_cloud(
                f"surface_{i}_{description}",
                points,
                radius=0.003,
            )
            
            # Use different colors for different surfaces
            colors = [
                (0.8, 0.2, 0.2),  # Red
                (0.2, 0.8, 0.2),  # Green
                (0.2, 0.2, 0.8),  # Blue
                (0.8, 0.8, 0.2),  # Yellow
                (0.8, 0.2, 0.8),  # Magenta
            ]
            cloud.set_color(colors[i])
            
            print(f"✓ [{i}] {description:25s}: sampled {points.shape[0]:4d} points")
            
            # Also recover and print surface info
            surf_dict = recover_surface_dict(params, surf_type_idx)
            print(f"    Type: {surf_dict['type']}")
            print(f"    UV range: u=[{surf_dict['uv'][0]:.2f}, {surf_dict['uv'][1]:.2f}], "
                  f"v=[{surf_dict['uv'][2]:.2f}, {surf_dict['uv'][3]:.2f}]")
            if surf_dict['scalar']:
                print(f"    Scalars: {surf_dict['scalar']}")
            print()
            
        except Exception as e:
            print(f"✗ [{i}] {description}: FAILED - {e}\n")
    
    print("=" * 60)
    print("Visualization ready!")
    print("=" * 60)
    print("\nSurface colors:")
    print("  - Red: Cylinder")
    print("  - Green: Sphere")
    print("  - Blue: Plane")
    print("  - Yellow: Torus")
    print("  - Magenta: Cone")
    print("\nPress 'q' to quit")
    
    ps.show()


if __name__ == "__main__":
    main()


