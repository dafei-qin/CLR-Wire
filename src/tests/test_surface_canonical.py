"""
Test script for surface canonical space transformation

This script tests the to_canonical and from_canonical functions by:
1. Loading surfaces from the dataset
2. Processing surfaces through _parse_surface and _recover_surface (GT)
3. Transforming surfaces to canonical space
4. Visualizing original, canonical, and recovered surfaces
5. Verifying the round-trip transformation accuracy
"""

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
from pathlib import Path

# Add project paths
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from utils.surface import visualize_json_interset


def process_surface_through_pipeline(surface_dict, dataset):
    """
    Process a surface through the full pipeline:
    1. Parse surface (apply dataset filters)
    2. Recover surface (get GT)
    3. Transform to canonical
    4. Transform back from canonical
    
    Args:
        surface_dict: Raw surface dictionary from JSON
        dataset: dataset_compound instance
    
    Returns:
        dict with keys: 'raw', 'gt', 'canonical', 'recovered', 'metrics', 'transform'
    """
    result = {
        'raw': surface_dict,
        'gt': None,
        'canonical': None,
        'recovered': None,
        'metrics': {},
        'transform': {}
    }
    
    try:
        # Step 1: Parse surface (apply dataset normalization)
        params, surface_type_idx = dataset._parse_surface(surface_dict)
        
        if surface_type_idx == -1 or params is None:
            result['metrics']['valid'] = False
            result['metrics']['error'] = 'Invalid surface (filtered out by dataset)'
            return result
        
        # Step 2: Recover surface to get GT
        gt_surface = dataset._recover_surface(params, surface_type_idx)
        gt_surface['idx'] = surface_dict.get('idx', [0, 0])
        gt_surface['orientation'] = surface_dict.get('orientation', 'Forward')
        gt_surface['poles'] = []
        result['gt'] = gt_surface
        result['metrics']['valid'] = True
        
        # Skip bspline surfaces for now
        if gt_surface['type'] == 'bspline_surface':
            result['metrics']['skipped'] = True
            result['metrics']['reason'] = 'B-spline surfaces not supported'
            return result
        
        # Step 3: Transform to canonical space
        canonical_surface, shift, rotation, scale = to_canonical(gt_surface)
        result['canonical'] = canonical_surface
        result['transform']['shift'] = shift
        result['transform']['rotation'] = rotation
        result['transform']['scale'] = scale
        
        # Step 4: Transform back from canonical
        recovered_surface = from_canonical(canonical_surface, shift, rotation, scale)
        result['recovered'] = recovered_surface
        
        # Step 5: Calculate transformation accuracy metrics
        result['metrics'].update(calculate_transformation_metrics(gt_surface, recovered_surface))
        
    except Exception as e:
        result['metrics']['valid'] = False
        result['metrics']['error'] = str(e)
        print(f"Error processing surface: {e}")
    
    return result


def calculate_transformation_metrics(gt_surface, recovered_surface):
    """Calculate metrics comparing GT and recovered surfaces"""
    metrics = {}
    
    # Type match
    metrics['type_match'] = gt_surface['type'] == recovered_surface['type']
    
    # Location difference
    gt_loc = np.array(gt_surface['location'][0])
    rec_loc = np.array(recovered_surface['location'][0])
    metrics['location_diff'] = np.linalg.norm(gt_loc - rec_loc)
    metrics['location_max_diff'] = np.abs(gt_loc - rec_loc).max()
    
    # Direction difference
    gt_dir = np.array(gt_surface['direction'])
    rec_dir = np.array(recovered_surface['direction'])
    metrics['direction_diff'] = np.linalg.norm(gt_dir - rec_dir)
    metrics['direction_max_diff'] = np.abs(gt_dir - rec_dir).max()
    
    # UV difference
    gt_uv = np.array(gt_surface['uv'])
    rec_uv = np.array(recovered_surface['uv'])
    metrics['uv_diff'] = np.linalg.norm(gt_uv - rec_uv)
    metrics['uv_max_diff'] = np.abs(gt_uv - rec_uv).max()
    
    # Scalar difference
    if len(gt_surface['scalar']) > 0:
        gt_scalar = np.array(gt_surface['scalar'])
        rec_scalar = np.array(recovered_surface['scalar'])
        metrics['scalar_diff'] = np.linalg.norm(gt_scalar - rec_scalar)
        metrics['scalar_max_diff'] = np.abs(gt_scalar - rec_scalar).max()
    else:
        metrics['scalar_diff'] = 0.0
        metrics['scalar_max_diff'] = 0.0
    
    # Overall accuracy (max of all differences)
    metrics['overall_max_diff'] = max(
        metrics['location_max_diff'],
        metrics['direction_max_diff'],
        metrics['uv_max_diff'],
        metrics['scalar_max_diff']
    )
    
    # Pass/fail threshold
    threshold = 1e-6
    metrics['pass'] = metrics['overall_max_diff'] < threshold
    
    return metrics


def verify_canonical_properties(canonical_surface):
    """Verify that the canonical surface has the expected properties"""
    checks = {}
    
    # Check location is near origin
    loc = np.array(canonical_surface['location'][0])
    checks['location_at_origin'] = np.linalg.norm(loc) < 1e-5
    
    # Check direction is near (0, 0, 1)
    direction = np.array(canonical_surface['direction'][0])
    expected_dir = np.array([0, 0, 1])
    checks['direction_is_z'] = np.linalg.norm(direction - expected_dir) < 1e-5
    
    # Check X direction is near (1, 0, 0)
    x_dir = np.array(canonical_surface['direction'][1])
    expected_x = np.array([1, 0, 0])
    checks['x_direction_is_x'] = np.linalg.norm(x_dir - expected_x) < 1e-5
    
    # Check scale for different surface types
    surface_type = canonical_surface['type']
    if surface_type in ['cylinder', 'sphere']:
        # Radius should be ~1.0
        radius = canonical_surface['scalar'][0]
        checks['radius_is_one'] = abs(radius - 1.0) < 1e-5
    elif surface_type == 'cone':
        # Radius at v=0 should be ~1.0
        radius = canonical_surface['scalar'][1]
        checks['radius_is_one'] = abs(radius - 1.0) < 1e-5
    elif surface_type == 'torus':
        # Major radius should be ~1.0
        major_radius = canonical_surface['scalar'][0]
        checks['major_radius_is_one'] = abs(major_radius - 1.0) < 1e-5
    elif surface_type == 'plane':
        # Max UV dimension should be ~1.0
        uv = canonical_surface['uv']
        max_dim = max(abs(uv[1] - uv[0]), abs(uv[3] - uv[2]))
        checks['max_uv_is_one'] = abs(max_dim - 1.0) < 1e-5
    
    checks['all_pass'] = all(checks.values())
    return checks


# Global variables for interactive visualization
dataset = None
current_file_idx = 0
current_surface_idx = 0
max_file_idx = 0
current_surfaces = []
current_results = []
show_raw = False
show_gt = True
show_canonical = True
show_recovered = True
show_axes = True
show_cube = True
visualization_offset = 0.0  # Distance between visualizations


def create_coordinate_axes(scale=1.0):
    """Create coordinate axes visualization"""
    # Define axes endpoints
    axes_points = np.array([
        [0, 0, 0],  # Origin
        [scale, 0, 0],  # X axis
        [0, scale, 0],  # Y axis
        [0, 0, scale],  # Z axis
    ])
    
    # Define edges
    axes_edges = np.array([
        [0, 1],  # X axis
        [0, 2],  # Y axis
        [0, 3],  # Z axis
    ])
    
    return axes_points, axes_edges


def create_unit_cube():
    """Create unit cube vertices and edges"""
    # Cube vertices (centered at origin, size 1x1x1)
    cube_points = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ])
    
    # Cube edges
    cube_edges = np.array([
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    
    return cube_points, cube_edges


def load_dataset(dataset_path):
    """Load the dataset"""
    global dataset, max_file_idx
    
    dataset = dataset_compound(dataset_path)
    max_file_idx = len(dataset) - 1
    print(f"Loaded dataset with {len(dataset)} JSON files")


def load_file(file_idx):
    """Load all surfaces from a JSON file"""
    global current_surfaces, current_results
    
    json_path = dataset.json_names[file_idx]
    print(f"\nLoading file: {json_path}")
    
    with open(json_path, 'r') as f:
        current_surfaces = json.load(f)
    
    print(f"Loaded {len(current_surfaces)} surfaces from file")
    
    # Process all surfaces
    current_results = []
    valid_count = 0
    for i, surface_dict in enumerate(current_surfaces):
        result = process_surface_through_pipeline(surface_dict, dataset)
        current_results.append(result)
        if result['metrics'].get('valid', False) and not result['metrics'].get('skipped', False):
            valid_count += 1
    
    print(f"Successfully processed {valid_count} valid surfaces")
    return len(current_surfaces)


def update_visualization():
    """Update the visualization with current surface"""
    global current_surface_idx, current_results
    
    # Clear existing structures
    ps.remove_all_structures()
    
    if current_surface_idx >= len(current_results):
        print(f"Invalid surface index: {current_surface_idx}")
        return
    
    result = current_results[current_surface_idx]
    
    # Check if surface is valid
    if not result['metrics'].get('valid', False):
        print(f"Surface {current_surface_idx} is invalid: {result['metrics'].get('error', 'Unknown error')}")
        return
    
    if result['metrics'].get('skipped', False):
        print(f"Surface {current_surface_idx} skipped: {result['metrics'].get('reason', 'Unknown reason')}")
        return
    
    offset_x = 0.0
    
    # Visualize raw surface (optional)
    if show_raw and result['raw'] is not None:
        try:
            raw_data = [result['raw']]
            raw_surfaces = visualize_json_interset(raw_data, plot=True, plot_gui=False, tol=1e-3, ps_header=f'raw_{current_surface_idx}')
            # Apply offset
            for surf_data in raw_surfaces.values():
                if surf_data['ps_handler'] is not None:
                    surf_data['ps_handler'].translate([offset_x, 0, 0])
            offset_x += visualization_offset
        except Exception as e:
            print(f"Error visualizing raw surface: {e}")
    
    # Visualize GT surface
    if show_gt and result['gt'] is not None:
        try:
            gt_data = [result['gt']]
            gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-3, ps_header=f'gt_{current_surface_idx}')
            # Apply offset
            for surf_data in gt_surfaces.values():
                if surf_data['ps_handler'] is not None:
                    surf_data['ps_handler'].translate([offset_x, 0, 0])
                    surf_data['ps_handler'].set_color([0.2, 0.8, 0.2])  # Green for GT
            offset_x += visualization_offset
        except Exception as e:
            print(f"Error visualizing GT surface: {e}")
    
    # Visualize canonical surface
    if show_canonical and result['canonical'] is not None:
        try:
            canonical_data = [result['canonical']]
            canonical_surfaces = visualize_json_interset(canonical_data, plot=True, plot_gui=False, tol=1e-3, ps_header=f'canonical_{current_surface_idx}')
            # Apply offset
            for surf_data in canonical_surfaces.values():
                if surf_data['ps_handler'] is not None:
                    surf_data['ps_handler'].translate([offset_x, 0, 0])
                    surf_data['ps_handler'].set_color([0.2, 0.2, 0.8])  # Blue for canonical
            offset_x += visualization_offset
        except Exception as e:
            print(f"Error visualizing canonical surface: {e}")
    
    # Visualize recovered surface
    if show_recovered and result['recovered'] is not None:
        try:
            recovered_data = [result['recovered']]
            recovered_surfaces = visualize_json_interset(recovered_data, plot=True, plot_gui=False, tol=1e-3, ps_header=f'recovered_{current_surface_idx}')
            # Apply offset
            for surf_data in recovered_surfaces.values():
                if surf_data['ps_handler'] is not None:
                    surf_data['ps_handler'].translate([offset_x, 0, 0])
                    surf_data['ps_handler'].set_color([0.8, 0.2, 0.2])  # Red for recovered
        except Exception as e:
            print(f"Error visualizing recovered surface: {e}")
    
    # Add coordinate axes
    if show_axes:
        try:
            axes_points, axes_edges = create_coordinate_axes(scale=1.5)
            axes_network = ps.register_curve_network("coordinate_axes", axes_points, axes_edges)
            axes_network.set_radius(0.005)
            axes_network.set_color([0.5, 0.5, 0.5])
            
            # Add colored axes for clarity
            # X axis - Red
            x_axis_points = np.array([[0, 0, 0], [1.5, 0, 0]])
            x_axis_edges = np.array([[0, 1]])
            x_axis = ps.register_curve_network("x_axis", x_axis_points, x_axis_edges)
            x_axis.set_radius(0.008)
            x_axis.set_color([1.0, 0.0, 0.0])
            
            # Y axis - Green
            y_axis_points = np.array([[0, 0, 0], [0, 1.5, 0]])
            y_axis_edges = np.array([[0, 1]])
            y_axis = ps.register_curve_network("y_axis", y_axis_points, y_axis_edges)
            y_axis.set_radius(0.008)
            y_axis.set_color([0.0, 1.0, 0.0])
            
            # Z axis - Blue
            z_axis_points = np.array([[0, 0, 0], [0, 0, 1.5]])
            z_axis_edges = np.array([[0, 1]])
            z_axis = ps.register_curve_network("z_axis", z_axis_points, z_axis_edges)
            z_axis.set_radius(0.008)
            z_axis.set_color([0.0, 0.0, 1.0])
        except Exception as e:
            print(f"Error visualizing axes: {e}")
    
    # Add unit cube
    if show_cube:
        try:
            cube_points, cube_edges = create_unit_cube()
            cube_network = ps.register_curve_network("unit_cube", cube_points, cube_edges)
            cube_network.set_radius(0.003)
            cube_network.set_color([0.7, 0.7, 0.7])
        except Exception as e:
            print(f"Error visualizing cube: {e}")
    
    # Print current surface info
    print(f"\n{'='*60}")
    print(f"Surface {current_surface_idx} - Type: {result['gt']['type']}")
    print(f"{'='*60}")
    
    # Print metrics
    metrics = result['metrics']
    print(f"\nTransformation Metrics:")
    print(f"  Location diff: {metrics.get('location_diff', 'N/A'):.2e} (max: {metrics.get('location_max_diff', 'N/A'):.2e})")
    print(f"  Direction diff: {metrics.get('direction_diff', 'N/A'):.2e} (max: {metrics.get('direction_max_diff', 'N/A'):.2e})")
    print(f"  UV diff: {metrics.get('uv_diff', 'N/A'):.2e} (max: {metrics.get('uv_max_diff', 'N/A'):.2e})")
    print(f"  Scalar diff: {metrics.get('scalar_diff', 'N/A'):.2e} (max: {metrics.get('scalar_max_diff', 'N/A'):.2e})")
    print(f"  Overall max diff: {metrics.get('overall_max_diff', 'N/A'):.2e}")
    print(f"  PASS: {metrics.get('pass', False)}")
    
    # Print canonical properties
    if result['canonical'] is not None:
        canonical_checks = verify_canonical_properties(result['canonical'])
        print(f"\nCanonical Properties:")
        for check, passed in canonical_checks.items():
            print(f"  {check}: {'✓' if passed else '✗'}")
    
    # Print transformation parameters
    if result['transform']:
        print(f"\nTransformation Parameters:")
        print(f"  Shift: {result['transform']['shift']}")
        print(f"  Scale: {result['transform']['scale']}")
        print(f"  Rotation:\n{result['transform']['rotation']}")


def callback():
    """Polyscope callback function for UI controls"""
    global current_file_idx, current_surface_idx, max_file_idx
    global show_raw, show_gt, show_canonical, show_recovered, show_axes, show_cube, current_results
    
    psim.TextUnformatted("Surface Canonical Space Transformation Test")
    psim.Separator()
    
    # File index slider
    changed, new_file_idx = psim.SliderInt("File Index", current_file_idx, 0, max_file_idx)
    if changed:
        current_file_idx = new_file_idx
        num_surfaces = load_file(current_file_idx)
        current_surface_idx = 0
        if num_surfaces > 0:
            update_visualization()
    
    # Surface index slider
    if current_results:
        max_surface_idx = len(current_results) - 1
        changed, new_surface_idx = psim.SliderInt("Surface Index", current_surface_idx, 0, max_surface_idx)
        if changed:
            current_surface_idx = new_surface_idx
            update_visualization()
    
    psim.Separator()
    psim.TextUnformatted(f"Current File: {current_file_idx}/{max_file_idx}")
    if current_results:
        psim.TextUnformatted(f"Current Surface: {current_surface_idx}/{len(current_results)-1}")
    
    # Display current surface metrics
    if current_results and current_surface_idx < len(current_results):
        result = current_results[current_surface_idx]
        if result['metrics'].get('valid', False):
            psim.Separator()
            psim.TextUnformatted("Transformation Metrics:")
            metrics = result['metrics']
            
            if not metrics.get('skipped', False):
                psim.TextUnformatted(f"Overall max diff: {metrics.get('overall_max_diff', 0):.2e}")
                
                pass_status = "PASS ✓" if metrics.get('pass', False) else "FAIL ✗"
                if metrics.get('pass', False):
                    psim.PushStyleColor(psim.ImGuiCol_Text, (0.2, 0.8, 0.2, 1.0))
                else:
                    psim.PushStyleColor(psim.ImGuiCol_Text, (0.8, 0.2, 0.2, 1.0))
                psim.TextUnformatted(pass_status)
                psim.PopStyleColor()
            else:
                psim.TextUnformatted(f"Skipped: {metrics.get('reason', 'Unknown')}")
    
    # Visibility controls
    psim.Separator()
    psim.TextUnformatted("Visibility Controls:")
    
    changed, show_raw = psim.Checkbox("Show Raw", show_raw)
    if changed:
        update_visualization()
    
    changed, show_gt = psim.Checkbox("Show GT (Green)", show_gt)
    if changed:
        update_visualization()
    
    changed, show_canonical = psim.Checkbox("Show Canonical (Blue)", show_canonical)
    if changed:
        update_visualization()
    
    changed, show_recovered = psim.Checkbox("Show Recovered (Red)", show_recovered)
    if changed:
        update_visualization()
    
    psim.Separator()
    psim.TextUnformatted("Reference Objects:")
    
    changed, show_axes = psim.Checkbox("Show Axes (RGB=XYZ)", show_axes)
    if changed:
        update_visualization()
    
    changed, show_cube = psim.Checkbox("Show Unit Cube", show_cube)
    if changed:
        update_visualization()
    
    # Action buttons
    psim.Separator()
    if psim.Button("Refresh"):
        update_visualization()
    
    if psim.Button("Next Valid Surface"):
        find_next_valid_surface(1)
    
    if psim.Button("Previous Valid Surface"):
        find_next_valid_surface(-1)
    
    if psim.Button("Test All Surfaces in File"):
        test_all_surfaces_in_file()


def find_next_valid_surface(direction=1):
    """Find the next (or previous) valid surface"""
    global current_surface_idx, current_results
    
    start_idx = current_surface_idx
    current_surface_idx += direction
    
    while 0 <= current_surface_idx < len(current_results):
        result = current_results[current_surface_idx]
        if result['metrics'].get('valid', False) and not result['metrics'].get('skipped', False):
            update_visualization()
            return
        current_surface_idx += direction
    
    # Wrap around or restore
    current_surface_idx = start_idx
    print("No more valid surfaces in this direction")


def test_all_surfaces_in_file():
    """Test all surfaces in the current file and report statistics"""
    global current_results
    
    print(f"\n{'='*60}")
    print(f"Testing all surfaces in file {current_file_idx}")
    print(f"{'='*60}")
    
    total = len(current_results)
    valid = 0
    skipped = 0
    passed = 0
    failed = 0
    
    max_diffs = []
    
    for i, result in enumerate(current_results):
        if not result['metrics'].get('valid', False):
            continue
        
        valid += 1
        
        if result['metrics'].get('skipped', False):
            skipped += 1
            continue
        
        if result['metrics'].get('pass', False):
            passed += 1
        else:
            failed += 1
            print(f"  Surface {i} ({result['gt']['type']}) FAILED with max diff: {result['metrics']['overall_max_diff']:.2e}")
        
        max_diffs.append(result['metrics']['overall_max_diff'])
    
    print(f"\nResults:")
    print(f"  Total surfaces: {total}")
    print(f"  Valid surfaces: {valid}")
    print(f"  Skipped surfaces: {skipped}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if max_diffs:
        print(f"\nAccuracy Statistics:")
        print(f"  Mean max diff: {np.mean(max_diffs):.2e}")
        print(f"  Median max diff: {np.median(max_diffs):.2e}")
        print(f"  Max max diff: {np.max(max_diffs):.2e}")
        print(f"  Min max diff: {np.min(max_diffs):.2e}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_surface_canonical.py <dataset_path>")
        print("Example: python test_surface_canonical.py c:/drivers/CAD/data/examples")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Load dataset
    print("Loading dataset...")
    load_dataset(dataset_path)
    
    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")
    
    # Set callback
    ps.set_user_callback(callback)
    
    # Load initial file and surface
    print("Loading initial file...")
    num_surfaces = load_file(current_file_idx)
    if num_surfaces > 0:
        update_visualization()
    
    # Show the interface
    print("\nStarting interactive visualization...")
    print("Use the sliders to navigate between files and surfaces")
    print("Toggle visibility to compare different stages")
    ps.show()

