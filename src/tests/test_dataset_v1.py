import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
import argparse
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from utils.surface import visualize_json_interset

def to_json_original(params_tensor, types_tensor, mask_tensor, dataset, shifts=None, rotations=None, scales=None, apply_from_canonical=False):
    """Convert processed parameters back to original JSON format for comparison"""
    json_data = []
    SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    
    for i in range(len(params_tensor)):
        if mask_tensor[i] == 0:  # Skip invalid surfaces
            continue
            
        params = params_tensor[i][:17 + SCALAR_DIM_MAP[SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]]]
        surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        
        print(f'Converting surface {i} of type {surface_type}')
        recovered_surface = dataset._recover_surface(params.numpy(), types_tensor[i].item())
        
        # Apply from_canonical if requested and canonical data is provided
        if apply_from_canonical and shifts is not None and rotations is not None and scales is not None:
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
    
    # Compare parameter ranges
    if len(original_data) > 0 and len(processed_data) > 0:
        if len(original_data) != len(processed_data):
            print(f'Original data and processed data have different lengths: {len(original_data)} != {len(processed_data)}')
            return metrics
        # Compare first surface as example
        orig_surf = original_data[0]
        proc_surf = processed_data[0]
        
        # Compare UV bounds
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
        else:
            metrics['scalar_difference'] = 0
    
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

def load_dataset():
    global dataset, max_idx, detect_closed
    
    parser = argparse.ArgumentParser(description='Test Dataset V1 with geometry preservation')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory')
    parser.add_argument('--detect_closed', action='store_true', help='Enable closed surface detection')
    
    args = parser.parse_args()
    detect_closed = args.detect_closed
    
    dataset = dataset_compound(args.dataset_path, canonical=True, detect_closed=detect_closed)
    max_idx = len(dataset) - 1
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Detect closed: {detect_closed}")

def process_sample(idx):
    """Process a single sample and return both original and processed data"""
    global dataset, current_metrics, detect_closed, current_is_u_closed, current_is_v_closed
    
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
        apply_from_canonical=True  # Apply from_canonical transformation
    )
    
    # Calculate geometry preservation metrics
    current_metrics = calculate_geometry_metrics(original_data, processed_data)
    
    print(f"Sample {idx} metrics:")
    print(f"  Original surfaces: {current_metrics['original_count']}")
    print(f"  Processed surfaces: {current_metrics['processed_count']}")
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
    global detect_closed, current_is_u_closed, current_is_v_closed
    
    psim.Text("Dataset V1 Geometry Preservation Test")
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
        psim.Text(f"Type match: {current_metrics.get('type_match', 'N/A')}")
        if 'uv_difference' in current_metrics:
            psim.Text(f"UV difference: {current_metrics['uv_difference']:.6f}")
        if 'scalar_difference' in current_metrics:
            psim.Text(f"Scalar difference: {current_metrics['scalar_difference']:.6f}")
    
    # Display closed surface statistics if enabled
    if detect_closed and current_is_u_closed is not None and current_is_v_closed is not None:
        psim.Separator()
        psim.Text("Closed Surface Detection:")
        
        # Get mask to count only valid surfaces
        if detect_closed:
            result = dataset[current_idx]
            mask_tensor = result[2]
            mask_bool = mask_tensor.bool()
            valid_is_u_closed = current_is_u_closed[mask_bool]
            valid_is_v_closed = current_is_v_closed[mask_bool]
        else:
            valid_is_u_closed = current_is_u_closed
            valid_is_v_closed = current_is_v_closed
        
        u_closed_count = valid_is_u_closed.sum().item()
        v_closed_count = valid_is_v_closed.sum().item()
        both_closed_count = (valid_is_u_closed & valid_is_v_closed).sum().item()
        none_closed_count = (~valid_is_u_closed & ~valid_is_v_closed).sum().item()
        
        psim.TextColored((1.0, 0.0, 0.0, 1.0), f"Red: U-closed only ({u_closed_count - both_closed_count})")
        psim.TextColored((0.0, 0.0, 1.0, 1.0), f"Blue: V-closed only ({v_closed_count - both_closed_count})")
        psim.TextColored((1.0, 1.0, 0.0, 1.0), f"Yellow: Both closed ({both_closed_count})")
        psim.TextColored((0.5, 0.5, 0.5, 1.0), f"Gray: None closed ({none_closed_count})")
    
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
    u_closed_total = 0
    v_closed_total = 0
    both_closed_total = 0
    
    for i in range(min(10, total_samples)):  # Test first 10 samples
        try:
            original_data, processed_data = process_sample(i)
            metrics = calculate_geometry_metrics(original_data, processed_data)
            
            if metrics['type_match']:
                type_matches += 1
            
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
