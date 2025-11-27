"""
Test script for sample_simple_surface.py

This script visualizes both:
1. Original surfaces from JSON using visualize_json_interset (meshified)
2. Sampled point clouds using sample_simple_surface

Both visualizations are shown in the same polyscope window for comparison.
Use the slider in the GUI to switch between different JSON files.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP_INV
from src.tools.sample_simple_surface import sample_surface_uniform
from utils.surface import build_plane_face, build_second_order_surface, build_bspline_surface


# Global state for interactive visualization
_json_files = []
_dataset = None
_current_index = 0
_pending_index = 0
_num_u = 32
_num_v = 32
_show_mesh = True
_show_samples = True
_status_message = ""


def load_json_data(json_path: str):
    """Load surface data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _set_status(message: str):
    """Set status message."""
    global _status_message
    _status_message = message


def _refresh_visualization():
    """Refresh visualization with current index."""
    global _current_index, _json_files, _dataset, _num_u, _num_v, _show_mesh, _show_samples
    
    # Clear all existing structures
    ps.remove_all_structures()
    
    json_path = str(_json_files[_current_index])
    json_filename = Path(json_path).name
    
    _set_status(f"Loading {json_filename}...")
    
    # Load JSON data
    try:
        surfaces_data = load_json_data(json_path)
    except Exception as e:
        _set_status(f"Failed to load JSON: {e}")
        return
    
    num_surfaces = len(surfaces_data)
    _set_status(f"Loaded {num_surfaces} surfaces from {json_filename}")
    
    # === Part 1: Visualize original surfaces (meshified) ===
    if _show_mesh:
        for face in surfaces_data:
            surface_type = face['type']
            surface_index = face['idx'][0]
            
            try:
                if surface_type == 'plane':
                    occ_face, vertices, faces, _ = build_plane_face(face, tol=1e-2, meshify=True)
                elif surface_type in ['cylinder', 'cone', 'torus', 'sphere']:
                    occ_face, vertices, faces, _ = build_second_order_surface(face, tol=1e-2, meshify=True)
                elif surface_type == 'bspline_surface':
                    occ_face, vertices, faces, _ = build_bspline_surface(
                        face, tol=1e-1, normalize_surface=False, normalize_knots=True
                    )
                else:
                    continue
                
                if len(vertices) > 0 and len(faces) > 0:
                    ps.register_surface_mesh(
                        f"mesh_{surface_index:03d}_{surface_type}",
                        np.array(vertices),
                        np.array(faces),
                        transparency=0.5
                    )
            except Exception as e:
                print(f"  [Mesh {surface_index:03d}] {surface_type}: Failed - {e}")
    
    # === Part 2: Sample point clouds ===
    if _show_samples:
        # Load parameters from dataset
        params_batch, types_batch, mask_batch, _, _, _ = _dataset[_current_index]
        
        num_sampled = 0
        for surf_idx in range(num_surfaces):
            if mask_batch[surf_idx] < 0.5:
                continue
            
            params = params_batch[surf_idx]
            surf_type_idx = types_batch[surf_idx].item()
            surf_type_name = SURFACE_TYPE_MAP_INV.get(surf_type_idx, "unknown")
            
            # Skip bspline surfaces
            if surf_type_name == 'bspline_surface':
                continue
            
            try:
                points = sample_surface_uniform(
                    params=params,
                    surface_type_idx=surf_type_idx,
                    num_u=_num_u,
                    num_v=_num_v,
                    flatten=True,
                )
                
                cloud = ps.register_point_cloud(
                    f"samples_{surf_idx:03d}_{surf_type_name}",
                    points,
                    radius=0.003,
                )
                cloud.set_color((0.1, 0.6, 0.9))
                num_sampled += 1
                
            except Exception as e:
                print(f"  [Sample {surf_idx:03d}] {surf_type_name}: Failed - {e}")
        
        _set_status(f"{json_filename} | Surfaces: {num_surfaces} | Sampled: {num_sampled}")


def _polyscope_callback():
    """Polyscope GUI callback for interactive controls."""
    global _current_index, _pending_index, _json_files, _num_u, _num_v, _show_mesh, _show_samples
    
    psim.TextUnformatted("Test: sample_simple_surface.py")
    psim.Separator()
    
    # JSON file selector
    total_files = len(_json_files)
    psim.TextUnformatted(f"Total JSON files: {total_files}")
    
    changed_slider, new_index = psim.SliderInt(
        "JSON Index",
        _pending_index,
        v_min=0,
        v_max=max(0, total_files - 1)
    )
    
    if changed_slider and new_index != _current_index:
        _pending_index = new_index
        _current_index = new_index
        _refresh_visualization()
    
    # Sampling parameters
    psim.Separator()
    changed_u, new_u = psim.InputInt("num_u", _num_u)
    if changed_u and new_u > 0:
        _num_u = new_u
        if _show_samples:
            _refresh_visualization()
    
    changed_v, new_v = psim.InputInt("num_v", _num_v)
    if changed_v and new_v > 0:
        _num_v = new_v
        if _show_samples:
            _refresh_visualization()
    
    # Display options
    psim.Separator()
    changed_mesh, new_mesh = psim.Checkbox("Show Mesh", _show_mesh)
    if changed_mesh:
        _show_mesh = new_mesh
        _refresh_visualization()
    
    changed_samples, new_samples = psim.Checkbox("Show Samples", _show_samples)
    if changed_samples:
        _show_samples = new_samples
        _refresh_visualization()
    
    # Refresh button
    if psim.Button("Refresh"):
        _refresh_visualization()
    
    # Status
    psim.Separator()
    psim.TextWrapped(_status_message or "Ready")
    
    # Help
    psim.Separator()
    psim.TextUnformatted("Controls:")
    psim.TextUnformatted("  - Slider: Switch JSON files")
    psim.TextUnformatted("  - num_u/v: Change sampling density")
    psim.TextUnformatted("  - Checkboxes: Toggle mesh/samples")
    psim.TextUnformatted("  - Press 'q' to quit")


def main():
    global _json_files, _dataset, _current_index, _pending_index, _num_u, _num_v, _show_mesh, _show_samples
    
    parser = argparse.ArgumentParser(
        description="Test sample_simple_surface by comparing with original JSON surfaces. "
                    "Use the slider in GUI to switch between JSON files."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset JSON directory"
    )
    parser.add_argument(
        "--num_u",
        type=int,
        default=32,
        help="Initial number of samples in u direction (default: 32)"
    )
    parser.add_argument(
        "--num_v",
        type=int,
        default=32,
        help="Initial number of samples in v direction (default: 32)"
    )
    parser.add_argument(
        "--mesh_only",
        action="store_true",
        help="Show only meshified surfaces (no sampling)"
    )
    parser.add_argument(
        "--samples_only",
        action="store_true",
        help="Show only sampled point clouds (no mesh)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Initial JSON index to display (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Find JSON files in dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    _json_files = sorted(list(dataset_path.rglob("*.json")))
    if not _json_files:
        print(f"Error: No JSON files found in {dataset_path}")
        return
    
    print("=" * 60)
    print("Test: sample_simple_surface.py (Interactive)")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Found {len(_json_files)} JSON files")
    
    # Initialize global parameters
    _num_u = args.num_u
    _num_v = args.num_v
    _show_mesh = not args.samples_only
    _show_samples = not args.mesh_only
    _current_index = min(args.start_index, len(_json_files) - 1)
    _pending_index = _current_index
    
    print(f"Initial sampling: num_u={_num_u}, num_v={_num_v}")
    print(f"Show mesh: {_show_mesh}")
    print(f"Show samples: {_show_samples}")
    print(f"Starting at index: {_current_index}")
    
    # Load dataset
    print("\nLoading dataset...")
    _dataset = dataset_compound(
        json_dir=str(dataset_path),
        max_num_surfaces=500,
        canonical=False  # No canonical transformation
    )
    print(f"✓ Dataset loaded with {len(_dataset)} files")
    
    # Initialize polyscope
    print("\nInitializing visualization...")
    ps.init()
    ps.set_ground_plane_mode("tile")
    ps.set_user_callback(_polyscope_callback)
    
    # Load initial visualization
    _refresh_visualization()
    
    print("\n" + "=" * 60)
    print("Visualization ready!")
    print("=" * 60)
    print("\nInteractive controls:")
    print("  - Use slider to switch between JSON files")
    print("  - Adjust num_u/num_v to change sampling density")
    print("  - Toggle checkboxes to show/hide mesh and samples")
    print("  - Press 'q' to quit")
    print()
    
    # Show polyscope window
    ps.show()
    
    print("\nDone!")


if __name__ == "__main__":
    main()

