import numpy as np
import polyscope as ps
import random
import torch
from argparse import ArgumentParser
import os

# Flow model related imports
from src.flow.surface_flow import ZLDM, ZLDMPipeline, get_new_scheduler
from src.vae.layers import BSplineSurfaceLayer
from src.utils.config import NestedDictToClass, load_config
from einops import rearrange

# --- Global variables ---
data_control_points_all = None
data_types_all = None
num_samples_total = 0
current_sample_indices = []
model_grid_size = None

# Polyscope structure references
ps_gt_surface_structures = []
ps_generated_surface_structures = []
ps_control_point_structures = []

# UI controllable number of surfaces
num_surfaces_to_display_ui = [1]

# Flow Model and Config Globals
flow_model = None
flow_cfg = None
device = None
pipeline = None
cp2surface_layer = None
use_fp16 = True  # Enable FP16 inference by default

# For generation variations
ps_generated_variation_structures = {}
ui_source_slot_for_generation = [0]
ui_num_variations = [5]
ui_num_inference_steps = [50]
ui_cfg_scale = [1.0]

# UI controls
ui_show_control_points = [True]
ui_show_gt_surface = [True]
ui_show_generated_surface = [True]
ui_control_point_size = [0.02]

# Type filtering controls
available_types = []
selected_type_index = [0]  # Index of selected type (0 = "All Types")
filtered_sample_indices = []  # Indices of samples matching the selected type

# Distance metrics tracking
distance_metrics = {}  # Key: sample_idx, Value: dict with distance metrics
ui_show_distance_summary = [False]

# --- Load Data --- 
def load_npz_data(npz_file_path):
    """Load control points and types from NPZ file."""
    global data_control_points_all, data_types_all, num_samples_total
    global available_types, filtered_sample_indices
    
    try:
        data = np.load(npz_file_path, allow_pickle=True) 
    except FileNotFoundError:
        print(f"Error: Data file '{npz_file_path}' not found. Please check the path.")
        exit()
    
    # Load control points (required for flow model)
    if 'control_points' in data:
        data_control_points_all = data['control_points']
    else:
        # Assume the file directly contains control points
        data_control_points_all = data
    
    # Load surface types (optional)
    if 'types' in data:
        data_types_all = data['types']
        if isinstance(data_types_all, np.ndarray) and data_types_all.dtype.kind in ['U', 'S']:
            # Convert numpy string array to Python list
            data_types_all = data_types_all.tolist()
        elif isinstance(data_types_all, np.ndarray):
            # Convert other numpy arrays to list
            data_types_all = data_types_all.tolist()
        
        # Get unique types and prepare type filtering
        unique_types = sorted(list(set(data_types_all)))
        available_types = ["All Types"] + unique_types
        print(f"Found {len(unique_types)} surface types: {unique_types}")
        
        # Initialize filtered indices to all samples
        filtered_sample_indices = list(range(len(data_types_all)))
    else:
        print("Warning: 'types' not found in NPZ file. Type filtering will be disabled.")
        data_types_all = None
        available_types = ["All Types"]
        filtered_sample_indices = []
    
    # Validate data shapes
    if data_control_points_all.ndim != 3 or data_control_points_all.shape[-1] != 3:
        print(f"Error: Expected control points shape (M, 16, 3), got {data_control_points_all.shape}")
        exit()
    
    if data_control_points_all.shape[1] != 16:
        print(f"Error: Expected 16 control points per surface, got {data_control_points_all.shape[1]}")
        exit()
    
    num_samples_total = data_control_points_all.shape[0]

    if num_samples_total == 0:
        print("Error: No samples found in the data file.")
        exit()
    
    # Update filtered indices if no types were loaded
    if not filtered_sample_indices:
        filtered_sample_indices = list(range(num_samples_total))
    
    type_info = f" with {len(available_types)-1} types" if data_types_all else ""
    print(f"Loaded {num_samples_total} control point samples{type_info} from {npz_file_path}.")

# --- Flow Model Loading ---
def load_flow_model_and_config(config_path, checkpoint_path):
    """Load the surface flow model and config."""
    global flow_model, flow_cfg, device, pipeline, cp2surface_layer, model_grid_size, use_fp16
    
    print(f"Loading surface flow model from config: {config_path} and checkpoint: {checkpoint_path}")
    try:
        cfg_dict = load_config(config_path)
        flow_cfg = NestedDictToClass(cfg_dict)
    except Exception as e:
        print(f"Error loading flow config: {e}")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if CUDA supports FP16
    if device.type == "cuda" and use_fp16:
        try:
            # Test if FP16 is supported
            test_tensor = torch.zeros(1, device=device, dtype=torch.float16)
            print("FP16 inference enabled - using half precision for memory efficiency")
        except:
            print("FP16 not supported on this device, falling back to FP32")
            use_fp16 = False
    else:
        if device.type == "cpu":
            print("CPU device detected, using FP32 (FP16 not supported on CPU)")
            use_fp16 = False

    try:
        flow_model = ZLDM(
            depth=flow_cfg.model.depth,
            dim=flow_cfg.model.dim,
            latent_dim=flow_cfg.model.latent_dim,
            heads=flow_cfg.model.heads,
            pe=flow_cfg.model.pe,
            res=flow_cfg.model.sample_points_num,
            block_out_channels=flow_cfg.model.block_out_channels,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'ema_model' in checkpoint:
            ema_model = checkpoint['ema']
            ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
            flow_model.load_state_dict(ema_model, strict=False)
            print("Loaded EMA model weights for surface flow.")
        elif 'model' in checkpoint:
            flow_model.load_state_dict(checkpoint['model'])
            print("Loaded model weights for surface flow.")
        else:
            flow_model.load_state_dict(checkpoint)
            print("Loaded raw model state_dict for surface flow.")
        
        flow_model.to(device)
        
        # Convert to half precision if using FP16
        if use_fp16:
            flow_model.half()
            print("Model converted to FP16")
        
        flow_model.eval()
        print("Surface flow model loaded and set to evaluation mode.")
        
        # Initialize pipeline with appropriate dtype
        pipeline_dtype = torch.float16 if use_fp16 else torch.float32
        scheduler = get_new_scheduler(getattr(flow_cfg, 'prediction_type', 'sample'))
        pipeline = ZLDMPipeline(flow_model, scheduler, dtype=pipeline_dtype)
        
        # Initialize B-spline surface layer
        model_grid_size = flow_cfg.model.sample_points_num
        cp2surface_layer = BSplineSurfaceLayer(resolution=model_grid_size)
        cp2surface_layer.to(device)
        
        # Convert B-spline layer to half precision if using FP16
        if use_fp16:
            cp2surface_layer.half()
        
        print(f"Pipeline initialized with grid size: {model_grid_size}x{model_grid_size}, dtype: {pipeline_dtype}")
        
    except Exception as e:
        print(f"Error initializing or loading surface flow model: {e}")
        exit()

# --- Type filtering functions ---
def update_filtered_sample_indices():
    """Update the filtered sample indices based on the selected type."""
    global filtered_sample_indices, selected_type_index, available_types, data_types_all
    
    if data_types_all is None or selected_type_index[0] == 0:
        # "All Types" selected or no types available
        filtered_sample_indices = list(range(num_samples_total))
    else:
        # Specific type selected
        selected_type = available_types[selected_type_index[0]]
        filtered_sample_indices = [i for i, surf_type in enumerate(data_types_all) if surf_type == selected_type]
    
    print(f"Filtered to {len(filtered_sample_indices)} samples of type '{available_types[selected_type_index[0]]}'")

def get_type_for_sample(sample_idx):
    """Get the type string for a given sample index."""
    if data_types_all is None or sample_idx >= len(data_types_all):
        return "Unknown"
    return data_types_all[sample_idx]

# --- Helper to get a new random sample index ---
def get_new_random_sample_idx(exclude_indices):
    """Get a new random sample index from filtered samples, avoiding those already shown if possible."""
    if len(filtered_sample_indices) == 0: 
        return -1
    
    # Convert exclude_indices to set for faster lookup
    exclude_set = set(exclude_indices)
    
    # Get available indices from filtered samples that are not excluded
    available_indices = [idx for idx in filtered_sample_indices if idx not in exclude_set]
    
    if not available_indices:
        # All filtered samples are already shown, pick any from filtered
        if filtered_sample_indices:
            return random.choice(filtered_sample_indices)
        else:
            return -1
    
    return random.choice(available_indices)

# --- Function to generate surface using the flow model ---
def generate_surface_with_flow(control_points_16x3, num_inference_steps=50, cfg_scale=1.0):
    """
    Generate a surface using the loaded flow model.
    
    Args:
        control_points_16x3: numpy array of shape (16, 3) - the input control points
        num_inference_steps: number of denoising steps
        cfg_scale: classifier-free guidance scale
        
    Returns:
        tuple: (generated_control_points, generated_surface) - generated control points and surface
    """
    global flow_model, flow_cfg, device, pipeline, cp2surface_layer, use_fp16
    
    if flow_model is None or flow_cfg is None or device is None or pipeline is None:
        print("Flow model not loaded. Cannot generate surface.")
        return control_points_16x3, np.zeros((model_grid_size, model_grid_size, 3))
    
    # Prepare inputs with appropriate dtype
    input_dtype = torch.float16 if use_fp16 else torch.float32
    control_points_tensor = torch.from_numpy(control_points_16x3).to(dtype=input_dtype, device=device).unsqueeze(0)  # (1, 16, 3)
    
    # Use autocast for mixed precision if using FP16
    autocast_context = torch.cuda.amp.autocast() if use_fp16 and device.type == "cuda" else torch.no_grad()
    
    with autocast_context:
        # Generate surface from control points using B-spline layer
        pc_condition = cp2surface_layer(control_points_tensor)  # (1, grid_size, grid_size, 3)
        
        # Generate new control points using the flow model
        generated_cp = pipeline(
            pc=pc_condition,
            num_latents=3,
            num_samples=1,
            num_inference_steps=num_inference_steps,
            device=device,
            show_progress=True
        )  # (1, 16, 3)
        
        # Generate surface from the generated control points
        generated_surface = cp2surface_layer(generated_cp)  # (1, grid_size, grid_size, 3)
        
        # Convert to numpy (ensure float32 for numpy compatibility)
        generated_cp_np = generated_cp[0].float().cpu().numpy()  # (16, 3)
        generated_surface_np = generated_surface[0].float().cpu().numpy()  # (grid_size, grid_size, 3)
        
    return generated_cp_np, generated_surface_np

# --- Helper function for color gradient by index order ---
def get_color_gradient_by_index(num_points, color_offset=None):
    """Generate colors for points based on their index order using a gradient."""
    light_blue = np.array([0.5, 0.7, 1.0])
    light_yellow = np.array([1.0, 1.0, 0.5])

    if color_offset is not None:
        light_blue = light_blue + np.array(color_offset)
        light_yellow = light_yellow + np.array(color_offset)
        light_blue = np.clip(light_blue, 0.0, 1.0)
        light_yellow = np.clip(light_yellow, 0.0, 1.0)

    indices = np.arange(num_points)
    
    if num_points <= 1:
        normalized_indices = np.full(num_points, 0.5)
    else:
        normalized_indices = indices / (num_points - 1)
    
    normalized_indices = np.clip(normalized_indices, 0, 1)
    colors = (1 - normalized_indices[:, None]) * light_blue + normalized_indices[:, None] * light_yellow

    return colors

# --- Helper function to create faces for surface mesh ---
def create_surface_mesh_faces(grid_size):
    """Generate faces (triangles) for a surface represented as a grid of points."""
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Triangle 1
            v1 = i * grid_size + j
            v2 = (i + 1) * grid_size + j
            v3 = i * grid_size + (j + 1)
            faces.append([v1, v2, v3])

            # Triangle 2
            v4 = (i + 1) * grid_size + j
            v5 = (i + 1) * grid_size + (j + 1)
            v6 = i * grid_size + (j + 1)
            faces.append([v4, v5, v6])

    return np.array(faces)

# --- Surface distance computation functions ---
def compute_surface_distances(surface_a, surface_b):
    """
    Compute point-to-point distances between two surfaces.
    
    Args:
        surface_a: numpy array of shape (N, N, 3) - first surface
        surface_b: numpy array of shape (N, N, 3) - second surface
        
    Returns:
        tuple: (l2_norm, percentile_95) - L2 norm and 95th percentile of point distances
    """
    # Flatten surfaces to point clouds
    points_a = surface_a.reshape(-1, 3)  # (N*N, 3)
    points_b = surface_b.reshape(-1, 3)  # (N*N, 3)
    
    # Compute point-to-point distances
    point_distances = (np.square(points_a - points_b)).mean(axis=1)  # (N*N,)
    
    # Compute 95th percentile of point-to-point distances
    percentile_95 = np.percentile(point_distances, 95)
    
    return point_distances.mean(), percentile_95

def print_surface_distance_metrics(sample_idx, surface_type, surface_gt, generated_surface):
    """
    Print distance metrics between ground truth and generated surfaces.
    
    Args:
        sample_idx: Sample index for identification
        surface_type: Type of the surface
        surface_gt: Ground truth surface (N, N, 3)
        generated_surface: Generated surface (N, N, 3)
    """
    global distance_metrics
    
    print(f"\n📏 Surface Distance Metrics for Sample {sample_idx} ({surface_type}):")
    print("=" * 70)
    
    # GT vs Generated
    gt_gen_l2, gt_gen_p95 = compute_surface_distances(surface_gt, generated_surface)
    print(f"GT ↔ Generated:")
    print(f"  L2 Norm: {gt_gen_l2:.6f}")
    print(f"  95th Percentile: {gt_gen_p95:.6f}")
    
    # Store metrics for later analysis
    distance_metrics[sample_idx] = {
        'type': surface_type,
        'gt_gen_l2': gt_gen_l2,
        'gt_gen_p95': gt_gen_p95,
    }
    
    print("=" * 70)

def get_distance_summary_stats():
    """Get summary statistics for all computed distance metrics."""
    if not distance_metrics:
        return None
    
    # Collect all metrics
    gt_gen_l2_vals = [m['gt_gen_l2'] for m in distance_metrics.values()]
    gt_gen_p95_vals = [m['gt_gen_p95'] for m in distance_metrics.values()]
    
    return {
        'num_samples': len(distance_metrics),
        'gt_gen_l2_mean': np.mean(gt_gen_l2_vals),
        'gt_gen_l2_std': np.std(gt_gen_l2_vals),
        'gt_gen_p95_mean': np.mean(gt_gen_p95_vals),
        'gt_gen_p95_std': np.std(gt_gen_p95_vals),
    }

# --- Polyscope Plotting/Updating for Flow Model ---
def display_or_update_flow_results(slot_idx, sample_idx):
    """Display or update all results from flow model inference."""
    global ps_gt_surface_structures, ps_generated_surface_structures, ps_control_point_structures

    if data_control_points_all is None:
        print("Data not loaded. Cannot display surfaces.")
        return

    control_points_gt = data_control_points_all[sample_idx]  # Shape: (16, 3)

    # Generate surface from ground truth control points with appropriate dtype
    input_dtype = torch.float16 if use_fp16 else torch.float32
    control_points_tensor = torch.from_numpy(control_points_gt).to(dtype=input_dtype, device=device).unsqueeze(0)
    
    # Use autocast context for consistent precision handling
    autocast_context = torch.cuda.amp.autocast() if use_fp16 and device.type == "cuda" else torch.no_grad()
    
    with autocast_context:
        gt_surface_tensor = cp2surface_layer(control_points_tensor)[0]
        gt_surface = gt_surface_tensor.float().cpu().numpy()  # Ensure float32 for numpy
    
    # Generate new surface using flow model
    control_points_generated, generated_surface = generate_surface_with_flow(
        control_points_gt, 
        num_inference_steps=ui_num_inference_steps[0],
        cfg_scale=ui_cfg_scale[0]
    )
    
    # Compute and print surface distance metrics
    surface_type = get_type_for_sample(sample_idx)
    print_surface_distance_metrics(sample_idx, surface_type, gt_surface, generated_surface)

    # Prepare data for visualization
    gt_surface_points = gt_surface.reshape(-1, 3)
    generated_surface_points = generated_surface.reshape(-1, 3)

    # Get colors and faces
    gt_colors = get_color_gradient_by_index(len(gt_surface_points))
    gen_colors = get_color_gradient_by_index(len(generated_surface_points), color_offset=[0.2, -0.1, -0.3])
    
    grid_size = gt_surface.shape[0]
    faces = create_surface_mesh_faces(grid_size)

    # Ensure lists are long enough
    while len(ps_gt_surface_structures) <= slot_idx:
        ps_gt_surface_structures.append(None)
    while len(ps_generated_surface_structures) <= slot_idx:
        ps_generated_surface_structures.append(None)
    while len(ps_control_point_structures) <= slot_idx:
        ps_control_point_structures.append(None)

    # Structure names
    gt_name = f"gt_surface_{slot_idx}"
    gen_name = f"generated_surface_{slot_idx}"
    cp_gt_name = f"cp_gt_{slot_idx}"
    cp_gen_name = f"cp_generated_{slot_idx}"

    # Ground Truth Surface
    if ui_show_gt_surface[0]:
        if ps_gt_surface_structures[slot_idx] is None or not ps.has_surface_mesh(gt_name):
            ps_gt_surface_structures[slot_idx] = ps.register_surface_mesh(gt_name, gt_surface_points, faces)
        else:
            ps_gt_surface_structures[slot_idx].update_vertex_positions(gt_surface_points)
        
        ps_gt_surface_structures[slot_idx].add_color_quantity("index_color", gt_colors, enabled=True)
        ps_gt_surface_structures[slot_idx].set_edge_width(1.0)
        ps_gt_surface_structures[slot_idx].set_smooth_shade(True)
        ps_gt_surface_structures[slot_idx].set_transparency(0.7)
    elif ps_gt_surface_structures[slot_idx] is not None and ps.has_surface_mesh(gt_name):
        ps.remove_surface_mesh(gt_name)
        ps_gt_surface_structures[slot_idx] = None

    # Generated Surface
    if ui_show_generated_surface[0]:
        if ps_generated_surface_structures[slot_idx] is None or not ps.has_surface_mesh(gen_name):
            ps_generated_surface_structures[slot_idx] = ps.register_surface_mesh(gen_name, generated_surface_points, faces)
        else:
            ps_generated_surface_structures[slot_idx].update_vertex_positions(generated_surface_points)
        
        ps_generated_surface_structures[slot_idx].add_color_quantity("index_color", gen_colors, enabled=True)
        ps_generated_surface_structures[slot_idx].set_edge_width(1.0)
        ps_generated_surface_structures[slot_idx].set_smooth_shade(True)
    elif ps_generated_surface_structures[slot_idx] is not None and ps.has_surface_mesh(gen_name):
        ps.remove_surface_mesh(gen_name)
        ps_generated_surface_structures[slot_idx] = None

    # Control Points
    if ui_show_control_points[0]:
        # Ground truth control points
        if not ps.has_point_cloud(cp_gt_name):
            ps_cp_gt = ps.register_point_cloud(cp_gt_name, control_points_gt)
        else:
            ps_cp_gt = ps.get_point_cloud(cp_gt_name)
            ps_cp_gt.update_point_positions(control_points_gt)
        
        ps_cp_gt.set_color((0.2, 0.8, 0.2))  # Green for GT
        ps_cp_gt.set_radius(ui_control_point_size[0])

        # Generated control points
        if not ps.has_point_cloud(cp_gen_name):
            ps_cp_gen = ps.register_point_cloud(cp_gen_name, control_points_generated)
        else:
            ps_cp_gen = ps.get_point_cloud(cp_gen_name)
            ps_cp_gen.update_point_positions(control_points_generated)
        
        ps_cp_gen.set_color((0.8, 0.2, 0.2))  # Red for generated
        ps_cp_gen.set_radius(ui_control_point_size[0])
    else:
        # Remove control points if not showing
        if ps.has_point_cloud(cp_gt_name):
            ps.remove_point_cloud(cp_gt_name)
        if ps.has_point_cloud(cp_gen_name):
            ps.remove_point_cloud(cp_gen_name)

# --- Function to clear generated variations ---
def clear_generated_variations_for_slot(source_slot_idx):
    """Clear generated variations for a given slot."""
    global ps_generated_variation_structures
    if source_slot_idx in ps_generated_variation_structures:
        for i, var_ps_struct in enumerate(ps_generated_variation_structures[source_slot_idx]):
            var_name = f"generated_variation_surface_{source_slot_idx}_{i}"
            if var_ps_struct is not None and ps.has_surface_mesh(var_name):
                ps.remove_surface_mesh(var_name)
        ps_generated_variation_structures[source_slot_idx] = []
        print(f"Cleared generated variations for slot {source_slot_idx}.")

# --- Manage display count ---
def manage_surface_display_count(force_refresh_all=False):
    """Manage the total number of displayed surfaces based on UI."""
    global current_sample_indices
    desired_num = num_surfaces_to_display_ui[0]
    current_num_on_screen = len(current_sample_indices)

    if not ps.is_initialized(): 
        ps.init()

    if force_refresh_all:
        print(f"Refreshing all {current_num_on_screen} displayed surfaces.")
        temp_current_indices = list(current_sample_indices)
        for i in range(current_num_on_screen):
            new_sample_idx = get_new_random_sample_idx(temp_current_indices)
            current_sample_indices[i] = new_sample_idx
            display_or_update_flow_results(i, new_sample_idx)
            clear_generated_variations_for_slot(i)
            if new_sample_idx not in temp_current_indices:
                temp_current_indices.append(new_sample_idx)
        return

    # Add surfaces if desired > current
    while desired_num > len(current_sample_indices):
        slot_idx = len(current_sample_indices)
        new_sample_idx = get_new_random_sample_idx(current_sample_indices)
        if new_sample_idx == -1: 
            break
        
        current_sample_indices.append(new_sample_idx)
        print(f"Adding flow results at slot {slot_idx} with data sample {new_sample_idx}")
        display_or_update_flow_results(slot_idx, new_sample_idx)

    # Remove surfaces if desired < current
    while desired_num < len(current_sample_indices):
        slot_idx_to_remove = len(current_sample_indices) - 1
        print(f"Removing flow results from slot {slot_idx_to_remove}")
        
        # Remove all structures for this slot
        for structure_list in [ps_gt_surface_structures, ps_generated_surface_structures]:
            if len(structure_list) > slot_idx_to_remove and structure_list[slot_idx_to_remove] is not None:
                structure_list.pop()
        
        # Remove control points
        cp_gt_name = f"cp_gt_{slot_idx_to_remove}"
        cp_gen_name = f"cp_generated_{slot_idx_to_remove}"
        if ps.has_point_cloud(cp_gt_name):
            ps.remove_point_cloud(cp_gt_name)
        if ps.has_point_cloud(cp_gen_name):
            ps.remove_point_cloud(cp_gen_name)
        
        clear_generated_variations_for_slot(slot_idx_to_remove)
        current_sample_indices.pop()
        
        if slot_idx_to_remove in ps_generated_variation_structures:
            del ps_generated_variation_structures[slot_idx_to_remove]
        
    if current_num_on_screen != len(current_sample_indices):
        print(f"Now displaying {len(current_sample_indices)} flow result(s).")

# --- Function to register manual XYZ axes ---
def register_manual_xyz_axes(length=1.0, radius=0.02):
    """Register manual XYZ axes for reference."""
    if not ps.is_initialized(): 
        ps.init()

    # X-axis (Red)
    nodes_x = np.array([[0,0,0], [length,0,0]])
    edges_x = np.array([[0,1]])
    ps_x = ps.register_curve_network("x_axis", nodes_x, edges_x, radius=radius)
    ps_x.set_color((0.8, 0.1, 0.1))
    ps_x.set_radius(radius)

    # Y-axis (Green)
    nodes_y = np.array([[0,0,0], [0,length,0]])
    edges_y = np.array([[0,1]])
    ps_y = ps.register_curve_network("y_axis", nodes_y, edges_y)
    ps_y.set_color((0.1, 0.8, 0.1))
    ps_y.set_radius(radius)

    # Z-axis (Blue)
    nodes_z = np.array([[0,0,0], [0,0,length]])
    edges_z = np.array([[0,1]])
    ps_z = ps.register_curve_network("z_axis", nodes_z, edges_z)
    ps_z.set_color((0.1, 0.1, 0.8))
    ps_z.set_radius(radius)
    print("Manually registered XYZ axes.")

# --- Core Flow Generation Function ---
def generate_surface_variations(source_control_points, num_variations, num_inference_steps):
    """Generate surface variations using the flow model."""
    global flow_model, flow_cfg, device, pipeline, cp2surface_layer, use_fp16
    if flow_model is None or flow_cfg is None or device is None or pipeline is None:
        print("Flow model not loaded. Cannot generate variations.")
        return []

    if num_variations <= 0:
        print("Number of variations must be positive.")
        return []

    # Prepare inputs with appropriate dtype
    input_dtype = torch.float16 if use_fp16 else torch.float32
    control_points_tensor = torch.from_numpy(source_control_points).to(dtype=input_dtype, device=device).unsqueeze(0)
    
    # Use autocast for mixed precision if using FP16
    autocast_context = torch.cuda.amp.autocast() if use_fp16 and device.type == "cuda" else torch.no_grad()
    
    with autocast_context:
        pc_condition = cp2surface_layer(control_points_tensor)
        
        generated_surfaces_list = []
        for _ in range(num_variations):
            # Generate new control points
            generated_cp = pipeline(
                pc=pc_condition,
                num_latents=3,
                num_samples=1,
                num_inference_steps=num_inference_steps,
                device=device,
                show_progress=False
            )
            
            # Generate surface from control points
            generated_surface = cp2surface_layer(generated_cp)
            generated_surfaces_list.append(generated_surface[0].float().cpu().numpy())  # Ensure float32 for numpy
            
    print(f"Generated {len(generated_surfaces_list)} surface variations with {num_inference_steps} inference steps.")
    return generated_surfaces_list

# --- Polyscope User Interface Callback --- 
def my_ui_callback():
    """Main UI callback for the surface flow examination interface."""
    global num_surfaces_to_display_ui, current_sample_indices
    global ui_source_slot_for_generation, ui_num_variations, ui_num_inference_steps, ui_cfg_scale
    global ps_generated_variation_structures
    global ui_show_control_points, ui_show_gt_surface, ui_show_generated_surface, ui_control_point_size
    global model_grid_size
    global available_types, selected_type_index, filtered_sample_indices, data_types_all
    
    # --- Display current mode ---
    ps.imgui.TextColored((0.2, 0.8, 0.8, 1), "Mode: Surface Flow Model")
    
    # --- Display model information ---
    if model_grid_size is not None:
        precision_text = "FP16" if use_fp16 else "FP32"
        ps.imgui.TextColored((0.7, 0.7, 0.7, 1), f"Grid: {model_grid_size}x{model_grid_size}, Precision: {precision_text}")
    
    # --- Display type filtering information ---
    if data_types_all is not None:
        current_type = available_types[selected_type_index[0]]
        ps.imgui.TextColored((0.8, 0.9, 0.6, 1), f"Type Filter: {current_type} ({len(filtered_sample_indices)} samples)")
    
    ps.imgui.Separator()
    
    # --- Type Selector ---
    if len(available_types) > 1:  # Only show if we have types to choose from
        ps.imgui.Text("Surface Type Filter")
        
        changed_type, selected_type_index[0] = ps.imgui.Combo("Surface Type", selected_type_index[0], available_types)
        
        if changed_type:
            # Update filtered indices
            update_filtered_sample_indices()
            
            # Clear current displays and refresh with new type
            if len(current_sample_indices) > 0:
                manage_surface_display_count(force_refresh_all=True)
        
        ps.imgui.Separator()
    
    # --- Display Controls ---
    ps.imgui.Text("Display Controls")
    
    changed_show_cp, ui_show_control_points[0] = ps.imgui.Checkbox("Show Control Points", ui_show_control_points[0])
    changed_show_gt, ui_show_gt_surface[0] = ps.imgui.Checkbox("Show Ground Truth Surface", ui_show_gt_surface[0])
    changed_show_gen, ui_show_generated_surface[0] = ps.imgui.Checkbox("Show Generated Surface", ui_show_generated_surface[0])
    
    if ui_show_control_points[0]:
        ps.imgui.PushItemWidth(100)
        changed_cp_size, ui_control_point_size[0] = ps.imgui.SliderFloat("Control Point Size", ui_control_point_size[0], 0.005, 0.1)
        ps.imgui.PopItemWidth()
        if changed_cp_size:
            # Update control point sizes for all displayed structures
            for slot_idx in range(len(current_sample_indices)):
                cp_gt_name = f"cp_gt_{slot_idx}"
                cp_gen_name = f"cp_generated_{slot_idx}"
                if ps.has_point_cloud(cp_gt_name):
                    ps.get_point_cloud(cp_gt_name).set_radius(ui_control_point_size[0])
                if ps.has_point_cloud(cp_gen_name):
                    ps.get_point_cloud(cp_gen_name).set_radius(ui_control_point_size[0])
    
    # Update display if visibility options changed
    if changed_show_cp or changed_show_gt or changed_show_gen:
        for slot_idx in range(len(current_sample_indices)):
            sample_idx = current_sample_indices[slot_idx]
            display_or_update_flow_results(slot_idx, sample_idx)
    
    ps.imgui.Separator()
    
    # --- Surface Management ---
    ps.imgui.Text("Surface Management")
    
    ps.imgui.PushItemWidth(100)
    changed_num_display, num_surfaces_to_display_ui[0] = ps.imgui.InputInt("Num Surfaces", num_surfaces_to_display_ui[0], step=1, step_fast=5)
    ps.imgui.PopItemWidth()
    if num_surfaces_to_display_ui[0] < 0: 
        num_surfaces_to_display_ui[0] = 0 
    max_displayable_main = min(len(filtered_sample_indices), 10)  # Limit based on filtered samples
    if num_surfaces_to_display_ui[0] > max_displayable_main: 
        num_surfaces_to_display_ui[0] = max_displayable_main

    if changed_num_display:
        manage_surface_display_count(force_refresh_all=False)

    if ps.imgui.Button("Refresh Surfaces"):
        if len(current_sample_indices) > 0:
             manage_surface_display_count(force_refresh_all=True)
        else: 
             manage_surface_display_count(force_refresh_all=False)
    
    # Show information about currently displayed surfaces
    if len(current_sample_indices) > 0 and data_types_all is not None:
        ps.imgui.Text("Currently Displayed:")
        for i, sample_idx in enumerate(current_sample_indices):
            surface_type = get_type_for_sample(sample_idx)
            ps.imgui.Text(f"  Slot {i}: Sample {sample_idx} ({surface_type})")
        if len(current_sample_indices) > 3:  # Limit display to avoid clutter
            ps.imgui.Text(f"  ... and {len(current_sample_indices) - 3} more")
    
    ps.imgui.Separator()
    
    # --- Distance Metrics Summary ---
    ps.imgui.Text("Distance Metrics")
    
    changed_show_summary, ui_show_distance_summary[0] = ps.imgui.Checkbox("Show Distance Summary", ui_show_distance_summary[0])
    
    if ui_show_distance_summary[0]:
        stats = get_distance_summary_stats()
        if stats is not None:
            ps.imgui.Separator()
            ps.imgui.TextColored((1.0, 0.9, 0.5, 1), f"Distance Summary ({stats['num_samples']} samples):")
            
            # GT vs Generated comparison
            ps.imgui.Text("GT ↔ Generated:")
            ps.imgui.SameLine()
            ps.imgui.TextColored((0.8, 0.8, 0.8, 1), f"L2: {stats['gt_gen_l2_mean']:.4f}±{stats['gt_gen_l2_std']:.4f}, P95: {stats['gt_gen_p95_mean']:.4f}±{stats['gt_gen_p95_std']:.4f}")
        else:
            ps.imgui.TextColored((0.7, 0.7, 0.7, 1), "No distance metrics computed yet.")
    
    if ps.imgui.Button("Clear Distance Metrics"):
        distance_metrics.clear()
        print("Cleared all stored distance metrics.")
    
    ps.imgui.Separator()
    
    # --- Flow Generation Section ---
    ps.imgui.Text("Flow Generation Controls")

    num_currently_displayed = len(current_sample_indices)
    if num_currently_displayed > 0:
        ps.imgui.PushItemWidth(100)
        slot_changed, ui_source_slot_for_generation[0] = ps.imgui.InputInt("Source Slot Idx", ui_source_slot_for_generation[0])
        ps.imgui.PopItemWidth()
        if ui_source_slot_for_generation[0] < 0: 
            ui_source_slot_for_generation[0] = 0
        if ui_source_slot_for_generation[0] >= num_currently_displayed: 
            ui_source_slot_for_generation[0] = num_currently_displayed - 1
        
        # Generation parameters
        ps.imgui.PushItemWidth(100)
        _, ui_num_variations[0] = ps.imgui.InputInt("Num Variations", ui_num_variations[0])
        if ui_num_variations[0] < 1: 
            ui_num_variations[0] = 1
        ps.imgui.SameLine()
        _, ui_num_inference_steps[0] = ps.imgui.InputInt("Inference Steps", ui_num_inference_steps[0])
        if ui_num_inference_steps[0] < 1: 
            ui_num_inference_steps[0] = 1
        ps.imgui.PopItemWidth()

        if ps.imgui.Button(f"Generate Variations for Slot {ui_source_slot_for_generation[0]}"):
            source_slot_idx_val = ui_source_slot_for_generation[0]
            if 0 <= source_slot_idx_val < len(current_sample_indices):
                data_sample_idx_for_source = current_sample_indices[source_slot_idx_val]
                source_control_points = data_control_points_all[data_sample_idx_for_source]

                print(f"Generating {ui_num_variations[0]} surface variations for control points in slot {source_slot_idx_val} (data sample {data_sample_idx_for_source}) with {ui_num_inference_steps[0]} inference steps")
                
                # Clear old variations for this slot first
                clear_generated_variations_for_slot(source_slot_idx_val)
                
                newly_generated_surfaces = generate_surface_variations(source_control_points, ui_num_variations[0], ui_num_inference_steps[0])
                
                # Visualize the generated surface variations
                current_variations_for_slot = []
                
                # Define a consistent orange offset for all variations
                orange_offset = [0.3, -0.2, -0.4]
                
                for i, variation_surface_nxnxc in enumerate(newly_generated_surfaces):
                    var_name = f"generated_variation_surface_{source_slot_idx_val}_{i}"
                    
                    # Flatten surface for polyscope
                    variation_points = variation_surface_nxnxc.reshape(-1, 3)
                    grid_size = variation_surface_nxnxc.shape[0]
                    faces = create_surface_mesh_faces(grid_size)
                    
                    # Get colors with orange offset
                    variation_colors = get_color_gradient_by_index(len(variation_points), color_offset=orange_offset)
                    
                    if variation_points.shape[0] > 0:
                        var_ps_struct = ps.register_surface_mesh(var_name, variation_points, faces)
                        
                        var_ps_struct.add_color_quantity("index_color", variation_colors, enabled=True)
                        var_ps_struct.set_edge_width(0.5)
                        var_ps_struct.set_smooth_shade(True)
                        
                        current_variations_for_slot.append(var_ps_struct)
                    else:
                        current_variations_for_slot.append(None)
                        
                ps_generated_variation_structures[source_slot_idx_val] = current_variations_for_slot
                print(f"Displayed {len(current_variations_for_slot)} generated surface variations.")
                
            else:
                print(f"Invalid source slot index: {source_slot_idx_val}")
    else:
        ps.imgui.TextDisabled("No surfaces displayed to select as source.")

# --- Main Execution ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize Surface Flow Model Inference.")
    parser.add_argument('--npz_file', type=str, required=True, help="Path to the .npz file containing control points.")
    parser.add_argument('--config', type=str, required=True, help="Path to the flow model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the flow model checkpoint file.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16 for inference.")
    # parser.add_argument("--inference_steps", type=int, default=50, help="Number of inference steps for the flow model.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")

    # Set precision mode based on command line argument
    if args.fp32:
        use_fp16 = False
        print("FP32 mode requested via command line")

    # if args.inference_steps is not None:
    #     num_inference_steps = args.inference_steps
    #     print(f"Using {num_inference_steps} inference steps.")

    load_npz_data(args.npz_file)
    load_flow_model_and_config(args.config, args.checkpoint)

    # Initialize Polyscope
    ps.init()

    # Manually register XYZ axes
    axis_length = 1
    register_manual_xyz_axes(length=axis_length, radius=0.01 * axis_length) 

    # Set the user callback function
    ps.set_user_callback(my_ui_callback)

    # Initial display
    manage_surface_display_count()

    # Show the Polyscope GUI
    ps.show() 