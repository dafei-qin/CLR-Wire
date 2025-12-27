import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v2 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from myutils.surface import visualize_json_interset

def apply_scale_to_control_points(control_points: np.ndarray, scale: float) -> np.ndarray:
    """
    Apply uniform scaling to control points.
    
    Args:
        control_points: Array of shape (..., 3) where last dimension is [x, y, z]
        scale: Scale factor (original bbox max dimension)
    
    Returns:
        Scaled control points in original space
    """
    if scale <= 0:
        return control_points
    # From canonical [-1, 1] to original space
    # In canonical: points are normalized by (scale / 2.0)
    # To recover: multiply by (scale / 2.0)
    return control_points * (scale / 2.0)


def apply_rotation_to_control_points(control_points: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """
    Apply rotation to control points.
    
    Args:
        control_points: Array of shape (..., 3)
        rotation: 3x3 rotation matrix (from canonical to original)
    
    Returns:
        Rotated control points
    """
    original_shape = control_points.shape
    flat_points = control_points.reshape(-1, 3)
    # rotation is from canonical to original, so we apply it directly
    rotated = flat_points @ rotation.T
    return rotated.reshape(original_shape)


def apply_shift_to_control_points(control_points: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Apply translation (shift) to control points.
    
    Args:
        control_points: Array of shape (..., 3)
        shift: 3D translation vector (centroid)
    
    Returns:
        Shifted control points
    """
    return control_points + shift


def recover_bspline_from_canonical(shift, rotation, scale, params, surface_type_idx):
    """
    Recover bspline surface from canonical space parameters.
    
    Args:
        params: Parameter array from dataset_v2
        surface_type_idx: Surface type index (should be 5 for bspline_surface)
    
    Returns:
        Dictionary with recovered bspline surface
    """
    if surface_type_idx != SURFACE_TYPE_MAP.get('bspline_surface', -1):
        raise ValueError(f"Expected bspline_surface (type 5), got {surface_type_idx}")
    
    # Extract transformation info from P, D, X, UV
    # shift = params[:3]
    # rotation_row0 = params[3:6]
    # rotation_row1 = params[6:9]
    # rotation_row2 = params[9:12]
    # scale = float(params[12])
    
    # Reconstruct rotation matrix
    # rotation = np.array([rotation_row0, rotation_row1, rotation_row2], dtype=np.float64)
    
    # Extract 48D control points (in canonical space [-1, 1])
    scalar_params = params[17:65]
    control_points_canonical = scalar_params.reshape(4, 4, 3)
    
    # Recover to original space: Scale -> Rotate -> Translate

    control_points_original = (control_points_canonical * scale) @ rotation + shift
    
    # Build bspline surface dict
    u_degree = 3
    v_degree = 3
    num_poles_u = 4
    num_poles_v = 4
    num_knots_u = 2
    num_knots_v = 2
    
    u_knots = [0.0, 1.0]
    v_knots = [0.0, 1.0]
    u_mults = [4, 4]
    v_mults = [4, 4]
    
    # Build scalar array for bspline
    scalar = [u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v]
    scalar.extend(u_knots)
    scalar.extend(v_knots)
    scalar.extend(u_mults)
    scalar.extend(v_mults)
    
    # Build poles with weights = 1
    poles = []
    for i in range(4):
        row = []
        for j in range(4):
            x, y, z = control_points_original[i, j]
            row.append([float(x), float(y), float(z), 1.0])
        poles.append(row)
    
    return {
        'type': 'bspline_surface',
        'scalar': scalar,
        'poles': poles,
        'u_periodic': False,
        'v_periodic': False,
    }


def to_json_original(params_tensor, types_tensor, mask_tensor, dataset, shifts=None, rotations=None, scales=None, apply_from_canonical=False, only_bspline=False):
    """Convert processed parameters back to original JSON format for comparison"""
    json_data = []
    SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    
    for i in range(len(params_tensor)):
        if mask_tensor[i] == 0:  # Skip invalid surfaces
            continue
        
        surface_type_idx = types_tensor[i].item()
        surface_type = SURFACE_TYPE_MAP_INVERSE[surface_type_idx]
        
        # Filter: skip non-bspline surfaces if only_bspline is True
        if only_bspline and surface_type != 'bspline_surface':
            continue
        
        # Get valid param length
        if surface_type == 'bspline_surface':
            valid_len = 65  # 17 + 48
        else:
            valid_len = 17 + SCALAR_DIM_MAP.get(surface_type, 0)
        
        params = params_tensor[i][:valid_len]
        
        print(f'Converting surface {i} of type {surface_type}')
        
        # Special handling for bspline_surface
        if surface_type == 'bspline_surface' and apply_from_canonical:
            recovered_surface = recover_bspline_from_canonical(shifts[i], rotations[i], scales[i], params.numpy(), surface_type_idx)
        else:
            recovered_surface = dataset._recover_surface(params.numpy(), surface_type_idx)
            
            # Apply from_canonical if requested and canonical data is provided
            if apply_from_canonical and shifts is not None and rotations is not None and scales is not None:
                if surface_type != 'bspline_surface':  # Already handled above
                    recovered_surface = from_canonical(recovered_surface, shifts[i], rotations[i], scales[i])
        
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)
    
    return json_data


def load_original_json(json_path):
    """Load original JSON data for comparison"""
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_geometry_metrics(original_data, processed_data):
    """Calculate metrics to verify geometry preservation"""
    metrics = {}
    
    # Count surfaces
    metrics['original_count'] = len(original_data)
    metrics['processed_count'] = len(processed_data)
    
    # Compare surface types
    original_types = [surf['type'] for surf in original_data]
    processed_types = [surf['type'] for surf in processed_data]
    metrics['type_match'] = original_types == processed_types
    
    # Count bspline surfaces
    metrics['bspline_count'] = sum(1 for t in original_types if t == 'bspline_surface')
    
    # Compare parameter ranges
    if len(original_data) > 0 and len(processed_data) > 0:
        if len(original_data) != len(processed_data):
            print(f'Original data and processed data have different lengths: {len(original_data)} != {len(processed_data)}')
            return metrics
        
        # Find a non-bspline surface for UV/scalar comparison
        for orig_surf, proc_surf in zip(original_data, processed_data):
            if orig_surf['type'] != 'bspline_surface':
                # Compare UV bounds
                if 'uv' in orig_surf and 'uv' in proc_surf:
                    orig_uv = orig_surf['uv']
                    proc_uv = proc_surf['uv']
                    metrics['uv_difference'] = np.abs(np.array(orig_uv) - np.array(proc_uv)).max()
                
                # Compare scalar parameters
                if 'scalar' in orig_surf and 'scalar' in proc_surf:
                    orig_scalar = np.array(orig_surf['scalar'])
                    proc_scalar = np.array(proc_surf['scalar'])
                    if len(orig_scalar) > 0 and len(proc_scalar) > 0:
                        metrics['scalar_difference'] = np.abs(orig_scalar - proc_scalar).max()
                    else:
                        metrics['scalar_difference'] = 0
                break
    
    return metrics


# Global variables for interactive visualization
dataset = None
current_idx = 0
max_idx = 0
original_group = None
processed_group = None
original_surfaces = {}
processed_surfaces = {}
show_original = True
show_processed = True
current_metrics = {}
detect_closed = False
current_is_u_closed = None
current_is_v_closed = None
only_show_bspline = False  # Filter to show only bspline surfaces


def load_dataset():
    global dataset, max_idx, detect_closed
    
    parser = argparse.ArgumentParser(description='Test Dataset V2 with bspline surface support')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--detect_closed', action='store_true', help='Enable closed surface detection')
    parser.add_argument('--bspline_fit_threshold', type=float, default=1e-3, help='BSpline fitting error threshold')
    
    args = parser.parse_args()
    detect_closed = args.detect_closed
    
    dataset = dataset_compound(
        args.dataset_path, 
        canonical=True, 
        detect_closed=detect_closed,
        bspline_fit_threshold=args.bspline_fit_threshold
    )
    max_idx = len(dataset) - 1
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Detect closed: {detect_closed}")
    print(f"BSpline fit threshold: {args.bspline_fit_threshold}")


def process_sample(idx):
    """Process a single sample and return both original and processed data"""
    global dataset, current_metrics, detect_closed, current_is_u_closed, current_is_v_closed, only_show_bspline
    
    # Get processed data from dataset
    if detect_closed:
        params_tensor, types_tensor, mask_tensor, shifts, rotations, scales, is_u_closed_tensor, is_v_closed_tensor = dataset[idx]
        current_is_u_closed = is_u_closed_tensor
        current_is_v_closed = is_v_closed_tensor
    else:
        params_tensor, types_tensor, mask_tensor, shifts, rotations, scales = dataset[idx]
        current_is_u_closed = None
        current_is_v_closed = None
    
    print(f'Processing file: {dataset.json_names[idx]}')
    json_path = dataset.json_names[idx]
    
    # Load original JSON data
    original_data = load_original_json(json_path)
    
    # Apply filter to original data if needed
    if only_show_bspline:
        original_data = [surf for surf in original_data if surf['type'] == 'bspline_surface']
    
    # Convert processed data back to JSON format
    # Apply from_canonical to transform back from canonical space
    mask_bool = mask_tensor.bool()
    mask_np = mask_bool.cpu().numpy().astype(bool)
    params_tensor_valid = params_tensor[mask_np]
    types_tensor_valid = types_tensor[mask_np]
    shifts_valid = shifts[mask_np]
    rotations_valid = rotations[mask_np]
    scales_valid = scales[mask_np]
    
    processed_data = to_json_original(
        params_tensor_valid, 
        types_tensor_valid, 
        mask_tensor, 
        dataset, 
        shifts=shifts_valid,
        rotations=rotations_valid,
        scales=scales_valid,
        apply_from_canonical=True,  # Apply from_canonical transformation
        only_bspline=only_show_bspline  # Apply bspline filter
    )
    
    # Calculate geometry preservation metrics
    current_metrics = calculate_geometry_metrics(original_data, processed_data)
    
    print(f"Sample {idx} metrics:")
    print(f"  Original surfaces: {current_metrics['original_count']}")
    print(f"  Processed surfaces: {current_metrics['processed_count']}")
    print(f"  BSpline surfaces: {current_metrics.get('bspline_count', 0)}")
    print(f"  Type match: {current_metrics['type_match']}")
    print(f"  UV difference: {current_metrics.get('uv_difference', 'N/A')}")
    print(f"  Scalar difference: {current_metrics.get('scalar_difference', 'N/A')}")
    
    if detect_closed:
        # Count closed surfaces (only for valid surfaces)
        valid_is_u_closed = current_is_u_closed[mask_bool]
        valid_is_v_closed = current_is_v_closed[mask_bool]
        u_closed_count = valid_is_u_closed.sum().item()
        v_closed_count = valid_is_v_closed.sum().item()
        both_closed_count = (valid_is_u_closed & valid_is_v_closed).sum().item()
        print(f"  U-closed surfaces: {u_closed_count}")
        print(f"  V-closed surfaces: {v_closed_count}")
        print(f"  Both closed: {both_closed_count}")
    
    return original_data, processed_data


def get_closed_color(is_u_closed, is_v_closed):
    """
    Determine color based on closed status.
    
    Returns:
        RGB tuple (values in [0, 1])
    """
    if is_u_closed and is_v_closed:
        return (1.0, 1.0, 0.0)  # Yellow - both closed
    elif is_u_closed:
        return (1.0, 0.0, 0.0)  # Red - u closed
    elif is_v_closed:
        return (0.0, 0.0, 1.0)  # Blue - v closed
    else:
        return (0.5, 0.5, 0.5)  # Gray - none closed


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, original_group, processed_group, original_surfaces, processed_surfaces
    global detect_closed, current_is_u_closed, current_is_v_closed, dataset
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Get mask to know which surfaces are valid
    if detect_closed:
        params_tensor, types_tensor, mask_tensor, shifts, rotations, scales, is_u_closed_tensor, is_v_closed_tensor = dataset[current_idx]
    else:
        params_tensor, types_tensor, mask_tensor, shifts, rotations, scales = dataset[current_idx]
    
    mask_bool = mask_tensor.bool()
    valid_indices = torch.where(mask_bool)[0].numpy()
    
    # Process current sample
    original_data, processed_data = process_sample(current_idx)
    
    # Visualize original surfaces
    try:
        original_surfaces = visualize_json_interset(original_data, plot=True, plot_gui=False, tol=1e-5, ps_header='original')
    except Exception as e:
        print(f'Error visualizing original data: {e}')
        return
    
    # Add original surfaces to group and apply colors
    for i, (surface_key, surface_data) in enumerate(original_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(original_group)
            
            # Apply color based on closed status if detect_closed is enabled
            if detect_closed and current_is_u_closed is not None and current_is_v_closed is not None:
                if i < len(valid_indices):
                    valid_idx = valid_indices[i]
                    is_u = current_is_u_closed[valid_idx].item()
                    is_v = current_is_v_closed[valid_idx].item()
                    color = get_closed_color(is_u, is_v)
                    surface_data['ps_handler'].set_color(color)
    
    # Visualize processed surfaces
    try:
        processed_surfaces = visualize_json_interset(processed_data, plot=True, plot_gui=False, tol=1e-5, ps_header='processed')
    except Exception as e:
        print(f'Error visualizing processed data: {e}')
        return
    
    # Add processed surfaces to group and apply colors based on closed status
    for i, (surface_key, surface_data) in enumerate(processed_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(processed_group)
            
            # Apply color based on closed status if detect_closed is enabled
            if detect_closed and current_is_u_closed is not None and current_is_v_closed is not None:
                if i < len(valid_indices):
                    valid_idx = valid_indices[i]
                    is_u = current_is_u_closed[valid_idx].item()
                    is_v = current_is_v_closed[valid_idx].item()
                    color = get_closed_color(is_u, is_v)
                    surface_data['ps_handler'].set_color(color)
    
    # Configure groups with current visibility settings
    original_group.set_enabled(show_original)
    processed_group.set_enabled(show_processed)
    
    print(f"Visualized {len(original_surfaces)} original surfaces and {len(processed_surfaces)} processed surfaces")


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_original, show_processed, current_metrics
    global detect_closed, current_is_u_closed, current_is_v_closed, only_show_bspline
    
    psim.Text("Dataset V2 Geometry Preservation Test")
    psim.Text("(with BSpline Surface Support)")
    psim.Separator()
    
    # Filter controls
    psim.Text("=== Surface Type Filter ===")
    changed_filter, only_show_bspline = psim.Checkbox("Only Show BSpline Surfaces", only_show_bspline)
    if changed_filter:
        update_visualization()
    
    psim.Separator()
    
    # Index slider
    changed, new_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if changed:
        current_idx = new_idx
        update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Display metrics
    if current_metrics:
        psim.Separator()
        psim.Text("Geometry Preservation Metrics:")
        psim.Text(f"Original surfaces: {current_metrics.get('original_count', 'N/A')}")
        psim.Text(f"Processed surfaces: {current_metrics.get('processed_count', 'N/A')}")
        psim.Text(f"BSpline surfaces: {current_metrics.get('bspline_count', 0)}")
        psim.Text(f"Type match: {current_metrics.get('type_match', 'N/A')}")
        if 'uv_difference' in current_metrics:
            psim.Text(f"UV difference: {current_metrics['uv_difference']:.6f}")
        if 'scalar_difference' in current_metrics:
            psim.Text(f"Scalar difference: {current_metrics['scalar_difference']:.6f}")
    
    # Display closed surface statistics if enabled
    # if detect_closed and current_is_u_closed is not None and current_is_v_closed is not None:
    #     psim.Separator()
    #     psim.Text("Closed Surface Detection:")
        
    #     # Get mask to count only valid surfaces
    #     if detect_closed:
    #         result = dataset[current_idx]
    #         mask_tensor = result[2]
    #         mask_bool = mask_tensor.bool()
    #         valid_is_u_closed = current_is_u_closed[mask_bool]
    #         valid_is_v_closed = current_is_v_closed[mask_bool]
    #     else:
    #         valid_is_u_closed = current_is_u_closed
    #         valid_is_v_closed = current_is_v_closed
        
    #     u_closed_count = valid_is_u_closed.sum().item()
    #     v_closed_count = valid_is_v_closed.sum().item()
    #     both_closed_count = (valid_is_u_closed & valid_is_v_closed).sum().item()
    #     none_closed_count = (~valid_is_u_closed & ~valid_is_v_closed).sum().item()
        
    #     psim.TextColored((1.0, 0.0, 0.0, 1.0), f"Red: U-closed only ({u_closed_count - both_closed_count})")
    #     psim.TextColored((0.0, 0.0, 1.0, 1.0), f"Blue: V-closed only ({v_closed_count - both_closed_count})")
    #     psim.TextColored((1.0, 1.0, 0.0, 1.0), f"Yellow: Both closed ({both_closed_count})")
    #     psim.TextColored((0.5, 0.5, 0.5, 1.0), f"Gray: None closed ({none_closed_count})")
    
    # Group controls
    if original_group is not None:
        psim.Separator()
        psim.Text("Group Controls:")
        changed, show_original = psim.Checkbox("Show Original", show_original)
        if changed:
            original_group.set_enabled(show_original)
        
        changed, show_processed = psim.Checkbox("Show Processed", show_processed)
        if changed:
            processed_group.set_enabled(show_processed)
    
    # Additional controls
    psim.Separator()
    if psim.Button("Refresh Current Sample"):
        update_visualization()
    
    if psim.Button("Test All Samples"):
        test_all_samples()


def test_all_samples():
    """Test all samples and report overall statistics"""
    print("\n" + "="*50)
    print("Testing all samples for geometry preservation...")
    total_samples = len(dataset)
    type_matches = 0
    uv_differences = []
    scalar_differences = []
    bspline_total = 0
    u_closed_total = 0
    v_closed_total = 0
    both_closed_total = 0
    
    for i in range(min(10, total_samples)):  # Test first 10 samples
        try:
            original_data, processed_data = process_sample(i)
            metrics = calculate_geometry_metrics(original_data, processed_data)
            
            if metrics['type_match']:
                type_matches += 1
            
            bspline_total += metrics.get('bspline_count', 0)
            
            if 'uv_difference' in metrics:
                uv_differences.append(metrics['uv_difference'])
            if 'scalar_difference' in metrics:
                scalar_differences.append(metrics['scalar_difference'])
            
            # Collect closed surface statistics
            if detect_closed and current_is_u_closed is not None and current_is_v_closed is not None:
                if detect_closed:
                    result = dataset[i]
                    mask_tensor = result[2]
                    mask_bool = mask_tensor.bool()
                    valid_is_u_closed = current_is_u_closed[mask_bool]
                    valid_is_v_closed = current_is_v_closed[mask_bool]
                else:
                    valid_is_u_closed = current_is_u_closed
                    valid_is_v_closed = current_is_v_closed
                    
                u_closed_total += valid_is_u_closed.sum().item()
                v_closed_total += valid_is_v_closed.sum().item()
                both_closed_total += (valid_is_u_closed & valid_is_v_closed).sum().item()
                
        except Exception as e:
            print(f"Error testing sample {i}: {e}")
    
    print(f"\nTest Results (first {min(10, total_samples)} samples):")
    print(f"Type matches: {type_matches}/{min(10, total_samples)} ({100*type_matches/min(10, total_samples):.1f}%)")
    print(f"Total BSpline surfaces: {bspline_total}")
    if uv_differences:
        print(f"Average UV difference: {np.mean(uv_differences):.6f}")
        print(f"Max UV difference: {np.max(uv_differences):.6f}")
    if scalar_differences:
        print(f"Average scalar difference: {np.mean(scalar_differences):.6f}")
        print(f"Max scalar difference: {np.max(scalar_differences):.6f}")
    
    if detect_closed:
        print(f"\nClosed Surface Statistics:")
        print(f"Total U-closed: {u_closed_total}")
        print(f"Total V-closed: {v_closed_total}")
        print(f"Total Both-closed: {both_closed_total}")
    
    print("="*50)


if __name__ == '__main__':
    # Initialize
    load_dataset()
    
    # Initialize polyscope
    ps.init()
    
    # Create groups
    original_group = ps.create_group("Original Surfaces")
    processed_group = ps.create_group("Processed Surfaces")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()

