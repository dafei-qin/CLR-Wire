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

from src.dataset.dataset_v3_tokenize import dataset_compound_tokenize
from src.dataset.dataset_v2 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import from_canonical
from utils.surface import visualize_json_interset


def recover_bspline_from_canonical(shift, rotation, scale, control_points_canonical):
    """
    Recover bspline surface from canonical space control points.
    
    Args:
        shift: 3D translation vector (centroid)
        rotation: 3x3 rotation matrix
        scale: Scale factor (original max dimension)
        control_points_canonical: 4x4x3 control points in canonical space [-1, 1]
    
    Returns:
        Dictionary with recovered bspline surface
    """
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


def calculate_tokenization_metrics(original_surfaces, tokenized_surfaces, codes, types):
    """Calculate metrics to verify tokenization quality"""
    metrics = {}
    
    # Count surfaces
    metrics['surface_count'] = len(original_surfaces)
    metrics['tokenized_count'] = len(tokenized_surfaces)
    
    # Analyze codes
    if len(codes) > 0:
        # Count non-padding codes (-1)
        valid_codes = codes[codes != -1]
        metrics['total_codes'] = len(valid_codes)
        metrics['unique_codes'] = len(np.unique(valid_codes))
        metrics['code_min'] = valid_codes.min() if len(valid_codes) > 0 else 0
        metrics['code_max'] = valid_codes.max() if len(valid_codes) > 0 else 0
    
    # Compare UV and scalar differences
    uv_diffs = []
    scalar_diffs = []
    
    for orig_surf, token_surf in zip(original_surfaces, tokenized_surfaces):
        if orig_surf['type'] == 'bspline_surface':
            continue
            
        # Compare UV
        if 'uv' in orig_surf and 'uv' in token_surf:
            orig_uv = np.array(orig_surf['uv'])
            token_uv = np.array(token_surf['uv'])
            uv_diff = np.abs(orig_uv - token_uv).max()
            uv_diffs.append(uv_diff)
        
        # Compare scalar
        if 'scalar' in orig_surf and 'scalar' in token_surf:
            orig_scalar = np.array(orig_surf['scalar'])
            token_scalar = np.array(token_surf['scalar'])
            if len(orig_scalar) > 0 and len(token_scalar) > 0:
                scalar_diff = np.abs(orig_scalar - token_scalar).max()
                scalar_diffs.append(scalar_diff)
    
    if uv_diffs:
        metrics['uv_max_diff'] = np.max(uv_diffs)
        metrics['uv_mean_diff'] = np.mean(uv_diffs)
    
    if scalar_diffs:
        metrics['scalar_max_diff'] = np.max(scalar_diffs)
        metrics['scalar_mean_diff'] = np.mean(scalar_diffs)
    
    # Count surface types
    type_counts = {}
    SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}
    for surface in tokenized_surfaces:
        # type_name = SURFACE_TYPE_MAP_INV.get(type_idx, 'unknown')
        type_name = surface['type']
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    metrics['type_counts'] = type_counts
    
    return metrics


def format_codes_for_display(codes, types):
    """Format codes for readable display"""
    SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}
    
    display_lines = []
    for i, (code_vec, type_idx) in enumerate(zip(codes, types)):
        if type(type_idx) == torch.Tensor:
            type_idx = type_idx.item()
        type_name = SURFACE_TYPE_MAP_INV.get(type_idx, 'unknown')
        # Filter out padding (-1)
        valid_codes = code_vec[code_vec != -1]
        if len(valid_codes) > 0:
            code_str = ', '.join([str(c) for c in valid_codes])
            display_lines.append(f"Surface {i} ({type_name}): [{code_str}]")
        else:
            display_lines.append(f"Surface {i} ({type_name}): [no codes]")
    
    return display_lines


# Global variables for interactive visualization
dataset = None
current_idx = 0
max_idx = 0
pending_idx = 0  # Pending index for "Go To Index" input
original_group = None
tokenized_group = None
original_surfaces = {}
tokenized_surfaces = {}
show_original = True
show_tokenized = True
current_metrics = {}
current_codes = None
current_types = None
codebook_size = 1024
detect_closed = False
only_show_bspline = False
apply_from_canonical = True  # Apply from_canonical transformation by default
current_json_path = ""  # Current JSON file path being processed


def load_dataset():
    global dataset, max_idx, codebook_size, detect_closed
    
    parser = argparse.ArgumentParser(description='Test Dataset V3 Tokenize with visualization')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--codebook_size', type=int, default=1024, help='Codebook size for quantization')
    parser.add_argument('--detect_closed', action='store_true', help='Enable closed surface detection')
    parser.add_argument('--bspline_fit_threshold', type=float, default=1e-3, help='BSpline fitting error threshold')
    
    args = parser.parse_args()
    codebook_size = args.codebook_size
    detect_closed = args.detect_closed
    
    dataset = dataset_compound_tokenize(
        args.dataset_path,
        canonical=True,
        detect_closed=detect_closed,
        bspline_fit_threshold=args.bspline_fit_threshold,
        codebook_size=codebook_size
    )
    max_idx = len(dataset.dataset_compound) - 1
    print(f"Loaded dataset with {len(dataset.dataset_compound)} samples")
    print(f"Codebook size: {codebook_size}")
    print(f"Detect closed: {detect_closed}")
    print(f"BSpline fit threshold: {args.bspline_fit_threshold}")


def process_sample(idx):
    """Process a single sample and return both original and tokenized data"""
    global dataset, current_metrics, current_codes, current_types, detect_closed, only_show_bspline, apply_from_canonical, current_json_path
    
    # Get data from tokenize dataset
    result = dataset[idx]
    tokenized_surfaces_list = result[0]
    codes = result[1]
    types = result[2]

    shifts = result[3]
    rotations = result[4]
    scales = result[5]
    
    current_codes = codes
    current_types = types
    
    # Get original data from underlying dataset
    json_path = dataset.dataset_compound.json_names[idx]
    current_json_path = json_path  # Update global variable
    print(f'Processing file: {json_path}')
    
    # Load original JSON
    with open(json_path, 'r') as f:
        original_data_full = json.load(f)
    
    # Get processed data from dataset_compound (before tokenization)
    if detect_closed:
        params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales, is_u_closed_tensor, is_v_closed_tensor = dataset.dataset_compound[idx]
    else:
        params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales = dataset.dataset_compound[idx]
    
    # Filter valid surfaces based on mask

    valid_mask = mask_tensor.bool()
    valid_count = valid_mask.sum().item()
    params_tensor = params_tensor[valid_mask.bool()]
    types_tensor = types_tensor[valid_mask.bool()]
    all_shifts = all_shifts[valid_mask.bool()]
    all_rotations = all_rotations[valid_mask.bool()]
    all_scales = all_scales[valid_mask.bool()]
    if detect_closed:
        is_u_closed_tensor = is_u_closed_tensor[valid_mask.bool()]
        is_v_closed_tensor = is_v_closed_tensor[valid_mask.bool()]
    
    # Recover original surfaces (before tokenization) and apply from_canonical
    original_surfaces_list = []
    for i in range(valid_count):
        recon_surface = dataset.dataset_compound._recover_surface(
            params_tensor[i].numpy(), 
            types_tensor[i].item()
        )
        
        # Apply from_canonical transformation if enabled
        if apply_from_canonical:
            surface_type = recon_surface['type']
            if surface_type == 'cone':
                print()
            if surface_type == 'bspline_surface':
                # For bspline, extract control points and transform
                scalar_params = params_tensor[i].numpy()[17:65]
                control_points_canonical = scalar_params.reshape(4, 4, 3)
                recon_surface = recover_bspline_from_canonical(
                    all_shifts[i], 
                    all_rotations[i], 
                    all_scales[i].item(),
                    control_points_canonical
                )
            else:
                # For other surfaces, use from_canonical
                recon_surface = from_canonical(
                    recon_surface, 
                    all_shifts[i], 
                    all_rotations[i], 
                    all_scales[i].item()
                )
        
        recon_surface['idx'] = [i, i]
        recon_surface['orientation'] = 'Forward'
        original_surfaces_list.append(recon_surface)
    
    # Apply from_canonical to tokenized surfaces as well
    if apply_from_canonical:
        tokenized_surfaces_transformed = []
        for i, surf in enumerate(tokenized_surfaces_list):
            if surf['type'] == 'bspline_surface':
                # For bspline, we need to extract control points from the surface dict
                # The poles are in canonical space, we need to transform them back
                # Extract poles: poles is a list of rows, each row is a list of [x, y, z, w]
                poles_flat = []
                for row in surf['poles']:
                    for pole in row:
                        poles_flat.extend(pole[:3])  # x, y, z only
                control_points_canonical = np.array(poles_flat).reshape(4, 4, 3)
                
                # Transform back to original space
                surf_transformed = recover_bspline_from_canonical(
                    shifts[i], 
                    rotations[i], 
                    scales[i],
                    control_points_canonical
                )
            else:
                # For other surfaces, use from_canonical directly
                surf_transformed = from_canonical(
                    surf, 
                    shifts[i], 
                    rotations[i], 
                    scales[i]
                )
            surf_transformed['idx'] = [i, i]
            surf_transformed['orientation'] = 'Forward'
            tokenized_surfaces_transformed.append(surf_transformed)
        tokenized_surfaces_list = tokenized_surfaces_transformed
    else:
        # If not applying from_canonical, still need to add idx and orientation
        for i, surf in enumerate(tokenized_surfaces_list):
            surf['idx'] = [i, i]
            surf['orientation'] = 'Forward'
    
    # Apply filter if needed
    if only_show_bspline:
        original_surfaces_list = [s for s in original_surfaces_list if s['type'] == 'bspline_surface']
        tokenized_surfaces_list = [s for s in tokenized_surfaces_list if s['type'] == 'bspline_surface']
        # Filter codes and types too
        bspline_indices = [i for i, s in enumerate(tokenized_surfaces_list) if s['type'] == 'bspline_surface']
        if len(bspline_indices) > 0:
            codes = codes[bspline_indices]
            types = types[bspline_indices]
    
    # Calculate metrics
    current_metrics = calculate_tokenization_metrics(
        original_surfaces_list, 
        tokenized_surfaces_list,
        codes,
        types
    )
    
    print(f"Sample {idx} metrics:")
    print(f"  Total surfaces: {current_metrics['surface_count']}")
    print(f"  Tokenized surfaces: {current_metrics['tokenized_count']}")
    if 'total_codes' in current_metrics:
        print(f"  Total codes: {current_metrics['total_codes']}")
        print(f"  Unique codes: {current_metrics['unique_codes']}")
        print(f"  Code range: [{current_metrics['code_min']}, {current_metrics['code_max']}]")
    if 'uv_max_diff' in current_metrics:
        print(f"  UV max diff: {current_metrics['uv_max_diff']:.6f}")
        print(f"  UV mean diff: {current_metrics['uv_mean_diff']:.6f}")
    if 'scalar_max_diff' in current_metrics:
        print(f"  Scalar max diff: {current_metrics['scalar_max_diff']:.6f}")
        print(f"  Scalar mean diff: {current_metrics['scalar_mean_diff']:.6f}")
    
    print(f"  Surface type counts: {current_metrics['type_counts']}")
    
    return original_surfaces_list, tokenized_surfaces_list


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, pending_idx, original_group, tokenized_group, original_surfaces, tokenized_surfaces
    
    # Sync pending_idx with current_idx
    pending_idx = current_idx
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    original_data, tokenized_data = process_sample(current_idx)
    
    # Visualize original surfaces (before tokenization)
    try:
        original_surfaces = visualize_json_interset(
            original_data, 
            plot=True, 
            plot_gui=False, 
            tol=1e-5, 
            ps_header='original'
        )
    except Exception as e:
        print(f'Error visualizing original data: {e}')
        original_surfaces = {}
    
    # Add original surfaces to group
    for surface_key, surface_data in original_surfaces.items():
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(original_group)
            # Set default color (blue-ish)
            surface_data['ps_handler'].set_color((0.3, 0.5, 0.9))
    
    # Visualize tokenized surfaces
    try:
        tokenized_surfaces = visualize_json_interset(
            tokenized_data, 
            plot=True, 
            plot_gui=False, 
            tol=1e-5, 
            ps_header='tokenized'
        )
    except Exception as e:
        print(f'Error visualizing tokenized data: {e}')
        tokenized_surfaces = {}
    
    # Add tokenized surfaces to group
    for surface_key, surface_data in tokenized_surfaces.items():
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(tokenized_group)
            # Set different color (orange-ish)
            surface_data['ps_handler'].set_color((0.9, 0.5, 0.2))
    
    # Configure groups with current visibility settings
    original_group.set_enabled(show_original)
    tokenized_group.set_enabled(show_tokenized)
    
    print(f"Visualized {len(original_surfaces)} original surfaces and {len(tokenized_surfaces)} tokenized surfaces")


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_original, show_tokenized, current_metrics
    global current_codes, current_types, codebook_size, only_show_bspline, apply_from_canonical, pending_idx, current_json_path
    
    psim.Text("Dataset V3 Tokenize Test")
    psim.Text(f"(Codebook Size: {codebook_size})")
    psim.Separator()
    
    # Display current JSON path
    if current_json_path:
        psim.TextWrapped(f"Current File: {current_json_path}")
        psim.Separator()
    
    # Transformation controls
    psim.Text("=== Transformation ===")
    changed_canonical, apply_from_canonical = psim.Checkbox("Apply from_canonical (to original space)", apply_from_canonical)
    if changed_canonical:
        update_visualization()
    
    psim.Separator()
    
    # Filter controls
    psim.Text("=== Surface Type Filter ===")
    changed_filter, only_show_bspline = psim.Checkbox("Only Show BSpline Surfaces", only_show_bspline)
    if changed_filter:
        update_visualization()
    
    psim.Separator()
    
    # Index controls
    psim.Text("=== Index Controls ===")
    slider_changed, slider_idx = psim.SliderInt("Sample Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    # Go To Index with button
    input_changed, input_idx = psim.InputInt("Go To Index", pending_idx)
    if input_changed:
        pending_idx = max(0, min(max_idx, input_idx))
    
    psim.SameLine()
    if psim.Button("Go"):
        if pending_idx != current_idx:
            current_idx = pending_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Display metrics
    if current_metrics:
        psim.Separator()
        psim.Text("=== Tokenization Metrics ===")
        psim.Text(f"Total surfaces: {current_metrics.get('surface_count', 'N/A')}")
        psim.Text(f"Tokenized surfaces: {current_metrics.get('tokenized_count', 'N/A')}")
        
        if 'total_codes' in current_metrics:
            psim.Text(f"Total codes: {current_metrics['total_codes']}")
            psim.Text(f"Unique codes: {current_metrics['unique_codes']}")
            psim.Text(f"Code range: [{current_metrics['code_min']}, {current_metrics['code_max']}]")
        
        if 'uv_max_diff' in current_metrics:
            psim.Text(f"UV max diff: {current_metrics['uv_max_diff']:.6f}")
            psim.Text(f"UV mean diff: {current_metrics['uv_mean_diff']:.6f}")
        
        if 'scalar_max_diff' in current_metrics:
            psim.Text(f"Scalar max diff: {current_metrics['scalar_max_diff']:.6f}")
            psim.Text(f"Scalar mean diff: {current_metrics['scalar_mean_diff']:.6f}")
        
        if 'type_counts' in current_metrics:
            psim.Separator()
            psim.Text("Surface Type Distribution:")
            for type_name, count in current_metrics['type_counts'].items():
                psim.Text(f"  {type_name}: {count}")
    
    # Display codes
    if current_codes is not None and current_types is not None:
        psim.Separator()
        psim.Text("=== Quantization Codes ===")
        psim.Text("(Format: [u_code0, u_code1, v_code0, v_code1, scalar0, scalar1])")
        
        code_lines = format_codes_for_display(current_codes, current_types)
        for line in code_lines[:10]:  # Show first 10 surfaces
            psim.Text(line)
        
        if len(code_lines) > 10:
            psim.Text(f"... and {len(code_lines) - 10} more surfaces")
    
    # Group controls
    if original_group is not None:
        psim.Separator()
        psim.Text("=== Visualization Controls ===")
        changed, show_original = psim.Checkbox("Show Original (Blue)", show_original)
        if changed:
            original_group.set_enabled(show_original)
        
        changed, show_tokenized = psim.Checkbox("Show Tokenized (Orange)", show_tokenized)
        if changed:
            tokenized_group.set_enabled(show_tokenized)
    
    # Additional controls
    psim.Separator()
    if psim.Button("Refresh Current Sample"):
        update_visualization()
    
    if psim.Button("Test Multiple Samples"):
        test_multiple_samples()


def test_multiple_samples():
    """Test multiple samples and report overall statistics"""
    global apply_from_canonical
    print("\n" + "="*50)
    print("Testing multiple samples for tokenization quality...")
    print(f"from_canonical transformation: {'enabled' if apply_from_canonical else 'disabled'}")
    total_samples = len(dataset.dataset_compound)
    test_count = min(10, total_samples)
    
    all_uv_diffs = []
    all_scalar_diffs = []
    all_code_ranges = []
    all_unique_codes = []
    type_distribution = {}
    
    for i in range(test_count):
        try:
            original_data, tokenized_data = process_sample(i)
            
            if 'uv_max_diff' in current_metrics:
                all_uv_diffs.append(current_metrics['uv_max_diff'])
            
            if 'scalar_max_diff' in current_metrics:
                all_scalar_diffs.append(current_metrics['scalar_max_diff'])
            
            if 'unique_codes' in current_metrics:
                all_unique_codes.append(current_metrics['unique_codes'])
                all_code_ranges.append((current_metrics['code_min'], current_metrics['code_max']))
            
            # Accumulate type distribution
            for type_name, count in current_metrics.get('type_counts', {}).items():
                type_distribution[type_name] = type_distribution.get(type_name, 0) + count
                
        except Exception as e:
            print(f"Error testing sample {i}: {e}")
    
    print(f"\nTest Results (first {test_count} samples):")
    print(f"Tested samples: {test_count}")
    
    if all_uv_diffs:
        print(f"\nUV Quantization Error:")
        print(f"  Mean max diff: {np.mean(all_uv_diffs):.6f}")
        print(f"  Overall max diff: {np.max(all_uv_diffs):.6f}")
        print(f"  Min max diff: {np.min(all_uv_diffs):.6f}")
    
    if all_scalar_diffs:
        print(f"\nScalar Quantization Error:")
        print(f"  Mean max diff: {np.mean(all_scalar_diffs):.6f}")
        print(f"  Overall max diff: {np.max(all_scalar_diffs):.6f}")
        print(f"  Min max diff: {np.min(all_scalar_diffs):.6f}")
    
    if all_unique_codes:
        print(f"\nCode Statistics:")
        print(f"  Mean unique codes per sample: {np.mean(all_unique_codes):.1f}")
        print(f"  Max unique codes: {np.max(all_unique_codes)}")
        print(f"  Total code range observed: [0, {max([r[1] for r in all_code_ranges])}]")
    
    if type_distribution:
        print(f"\nSurface Type Distribution (across all tested samples):")
        for type_name, count in sorted(type_distribution.items()):
            print(f"  {type_name}: {count}")
    
    print("="*50)


if __name__ == '__main__':
    # Initialize
    load_dataset()
    
    # Initialize polyscope
    ps.init()
    
    # Create groups
    original_group = ps.create_group("Original Surfaces")
    tokenized_group = ps.create_group("Tokenized Surfaces")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()

