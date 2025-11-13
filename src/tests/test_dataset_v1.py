import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from utils.surface import visualize_json_interset

def to_json_original(params_tensor, types_tensor, mask_tensor, dataset):
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

def load_dataset():
    global dataset, max_idx
    
    if len(sys.argv) < 2:
        print("Usage: python test_dataset_v1.py <dataset_path>")
        sys.exit(1)
    
    dataset = dataset_compound(sys.argv[1], canonical=True)
    max_idx = len(dataset) - 1
    print(f"Loaded dataset with {len(dataset)} samples")

def process_sample(idx):
    """Process a single sample and return both original and processed data"""
    global dataset, current_metrics
    
    # Get processed data from dataset
    params_tensor, types_tensor, mask_tensor = dataset[idx]
    print(f'Processing file: {dataset.json_names[idx]}')
    json_path = dataset.json_names[idx]
    
    # Load original JSON data
    original_data = load_original_json(json_path)
    
    # Convert processed data back to JSON format
    processed_data = to_json_original(params_tensor, types_tensor, mask_tensor, dataset)
    
    # Calculate geometry preservation metrics
    current_metrics = calculate_geometry_metrics(original_data, processed_data)
    
    print(f"Sample {idx} metrics:")
    print(f"  Original surfaces: {current_metrics['original_count']}")
    print(f"  Processed surfaces: {current_metrics['processed_count']}")
    print(f"  Type match: {current_metrics['type_match']}")
    print(f"  UV difference: {current_metrics.get('uv_difference', 'N/A')}")
    print(f"  Scalar difference: {current_metrics.get('scalar_difference', 'N/A')}")
    
    return original_data, processed_data

def update_visualization():
    """Update the visualization with current index"""
    global current_idx, original_group, processed_group, original_surfaces, processed_surfaces
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    original_data, processed_data = process_sample(current_idx)
    
    # Visualize original surfaces
    try:
        original_surfaces = visualize_json_interset(original_data, plot=True, plot_gui=False, tol=1e-5, ps_header='original')
    except Exception as e:
        print(f'Error visualizing original data: {e}')
        return
    
    # Add original surfaces to group
    for i, (surface_key, surface_data) in enumerate(original_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(original_group)
    
    # Visualize processed surfaces
    try:
        processed_surfaces = visualize_json_interset(processed_data, plot=True, plot_gui=False, tol=1e-5, ps_header='processed')
    except Exception as e:
        print(f'Error visualizing processed data: {e}')
        return
    
    # Add processed surfaces to group
    for i, (surface_key, surface_data) in enumerate(processed_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(processed_group)
    
    # Configure groups with current visibility settings
    original_group.set_enabled(show_original)
    processed_group.set_enabled(show_processed)
    
    print(f"Visualized {len(original_surfaces)} original surfaces and {len(processed_surfaces)} processed surfaces")

def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_original, show_processed, current_metrics
    
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
