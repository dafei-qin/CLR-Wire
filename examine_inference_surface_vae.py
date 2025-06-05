import numpy as np
import polyscope as ps # Import Polyscope
import random
import torch # Added
from argparse import ArgumentParser # Added
import os # Added

# VAE related imports
from src.vae.vae_surface import AutoencoderKL2D # Surface VAE model
from src.utils.config import NestedDictToClass, load_config # Added
from einops import rearrange # Added
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution # Added

# --- Global variables ---
data_gt_all = None
data_recon_all = None
num_samples_total = 0
current_sample_indices = [] # Stores the data sample_idx for each displayed surface pair
eval_onthefly = False # Added: flag to determine if we should reconstruct on-the-fly
npz_grid_size = None # Added: grid size in the original .npz data (e.g., 32 for 32x32)
model_grid_size = None # Added: grid size expected by the model
skip_factor = None # Added: skip factor for sampling

# Polyscope structure references (lists to hold multiple structures)
ps_gt_surface_structures = []
ps_recon_surface_structures = []

# UI controllable number of surfaces
num_surfaces_to_display_ui = [1] # ImGui input_int needs a list or array

# VAE Model and Config Globals
vae_model = None # Added
vae_cfg = None # Added
device = None # Added
latent_spatial_dim_global = None # Added

# For sampled variations
ps_sampled_variation_structures = {} # Key: source_slot_idx, Value: list of PS structures # Added
ui_source_slot_for_sampling = [0] # Added
ui_num_variations = [5] # Added
ui_sigma_scale = [0.5] # Added

# --- Function to apply skipped sampling to surface data ---
def apply_skipped_sampling(surface_data_nxnxc):
    """
    Apply skipped sampling to surface data if model expects smaller grid than npz provides.
    
    Args:
        surface_data_nxnxc: numpy array of shape (N_npz, N_npz, C) - original surface data
        
    Returns:
        numpy array of shape (N_model, N_model, C) - sampled surface data
    """
    global skip_factor, model_grid_size, npz_grid_size
    
    if skip_factor is None or skip_factor == 1:
        return surface_data_nxnxc  # No sampling needed
    
    # Apply skipped sampling along both spatial dimensions
    sampled_surface = surface_data_nxnxc[::skip_factor, ::skip_factor, :]
    
    # Ensure we get exactly the expected grid size
    if sampled_surface.shape[0] > model_grid_size or sampled_surface.shape[1] > model_grid_size:
        sampled_surface = sampled_surface[:model_grid_size, :model_grid_size, :]
    elif sampled_surface.shape[0] < model_grid_size or sampled_surface.shape[1] < model_grid_size:
        print(f"Warning: Skipped sampling resulted in {sampled_surface.shape[:2]} grid, but model expects {model_grid_size}x{model_grid_size}")
    
    return sampled_surface

# --- Load Data --- 
def load_npz_data(npz_file_path, use_onthefly_eval=False): # Modified to accept eval_onthefly flag
    global data_gt_all, data_recon_all, num_samples_total, eval_onthefly
    global npz_grid_size, model_grid_size, skip_factor
    eval_onthefly = use_onthefly_eval
    
    try:
        data = np.load(npz_file_path, allow_pickle=True) 
    except FileNotFoundError:
        print(f"Error: Data file '{npz_file_path}' not found. Please check the path.")
        exit()
    
    # if 'ground_truth' not in data:
    #     print(f"Error: 'ground_truth' not found in {npz_file_path}.")
    #     exit()
    try:
        data_gt_all_raw = data['ground_truth']
    except:
        data_gt_all_raw = data
    
    # Expect surface data to be (M, N, N, 3) where M is number of samples
    if data_gt_all_raw.ndim != 4 or data_gt_all_raw.shape[-1] != 3:
        print(f"Error: Expected surface data shape (M, N, N, 3), got {data_gt_all_raw.shape}")
        exit()
    
    npz_grid_size = data_gt_all_raw.shape[1]  # Assuming square grid
    if data_gt_all_raw.shape[1] != data_gt_all_raw.shape[2]:
        print(f"Warning: Non-square grid detected: {data_gt_all_raw.shape[1]}x{data_gt_all_raw.shape[2]}")
    
    print(f"NPZ file contains {npz_grid_size}x{npz_grid_size} grid per surface.")
    
    if eval_onthefly:
        print("Using on-the-fly evaluation mode - reconstructions will be computed using the VAE model.")
        data_recon_all = None  # Will be computed on-demand
    else:
        if 'reconstructions' not in data:
            print(f"Error: 'reconstructions' not found in {npz_file_path}.")
            exit()
        data_recon_all_raw = data['reconstructions']
        
    num_samples_total = data_gt_all_raw.shape[0]

    if num_samples_total == 0:
        print("Error: No samples found in the data file.")
        exit()
    
    # Store raw data initially - will be processed after model config is loaded
    data_gt_all = data_gt_all_raw
    if not eval_onthefly:
        data_recon_all = data_recon_all_raw
    
    if eval_onthefly:
        print(f"Loaded {num_samples_total} ground truth samples for on-the-fly reconstruction from {npz_file_path}.")
    else:
        print(f"Loaded {num_samples_total} samples successfully from {npz_file_path}.")

# --- Function to process data after model config is loaded ---
def process_data_with_model_config():
    """
    Process the loaded data based on model configuration.
    Apply skipped sampling if necessary.
    """
    global data_gt_all, data_recon_all, npz_grid_size, model_grid_size, skip_factor, vae_cfg
    
    if vae_cfg is None:
        print("Error: VAE config not loaded. Cannot process data.")
        return
    
    # The surface VAE config uses sample_points_num parameter
    model_grid_size = getattr(vae_cfg.model, 'sample_points_num', 16)  # Default to 16 if not specified
    print(f"Model expects {model_grid_size}x{model_grid_size} grid per surface.")
    
    if npz_grid_size == model_grid_size:
        skip_factor = 1
        print("NPZ and model grid sizes match. No sampling needed.")
    elif npz_grid_size > model_grid_size:
        if npz_grid_size % model_grid_size == 0:
            skip_factor = npz_grid_size // model_grid_size
            print(f"Applying skipped sampling with factor {skip_factor} ({npz_grid_size}x{npz_grid_size} -> {model_grid_size}x{model_grid_size} grid).")
            
            # Apply skipped sampling to ground truth data
            print("Processing ground truth data...")
            processed_gt = []
            for i in range(data_gt_all.shape[0]):
                processed_gt.append(apply_skipped_sampling(data_gt_all[i]))
            data_gt_all = np.array(processed_gt)
            
            # Apply skipped sampling to reconstruction data if available
            if data_recon_all is not None:
                print("Processing reconstruction data...")
                processed_recon = []
                for i in range(data_recon_all.shape[0]):
                    processed_recon.append(apply_skipped_sampling(data_recon_all[i]))
                data_recon_all = np.array(processed_recon)
        else:
            print(f"Warning: NPZ grid size ({npz_grid_size}) not evenly divisible by model grid size ({model_grid_size}).")
            skip_factor = npz_grid_size // model_grid_size
            if skip_factor == 0:
                skip_factor = 1
            print(f"Using skip factor {skip_factor}, may result in approximate grid size.")
            
            # Apply skipped sampling
            processed_gt = []
            for i in range(data_gt_all.shape[0]):
                processed_gt.append(apply_skipped_sampling(data_gt_all[i]))
            data_gt_all = np.array(processed_gt)
            
            if data_recon_all is not None:
                processed_recon = []
                for i in range(data_recon_all.shape[0]):
                    processed_recon.append(apply_skipped_sampling(data_recon_all[i]))
                data_recon_all = np.array(processed_recon)
    else:
        print(f"Error: NPZ has smaller grid ({npz_grid_size}) than model expects ({model_grid_size}). Cannot proceed.")
        exit()
    
    print(f"Data processing complete. Final data shape: {data_gt_all.shape}")

# --- VAE Model Loading ---
def calculate_latent_spatial_dim(cfg_model_vae): # Renamed arg for clarity
    initial_grid_size = getattr(cfg_model_vae, 'sample_points_num', 16)
    num_downsamples = 0
    if hasattr(cfg_model_vae, 'block_out_channels') and isinstance(cfg_model_vae.block_out_channels, (list, tuple)) and len(cfg_model_vae.block_out_channels) > 1:
        num_downsamples = len(cfg_model_vae.block_out_channels) - 1
    
    latent_spatial_dim = initial_grid_size // (2**num_downsamples)
    latent_spatial_dim = max(1, latent_spatial_dim) 
    print(f"Calculated latent spatial dimension: {latent_spatial_dim}x{latent_spatial_dim} (from initial grid {initial_grid_size}x{initial_grid_size} and {num_downsamples} downsamples)")
    return latent_spatial_dim

def load_vae_model_and_config(config_path, checkpoint_path):
    global vae_model, vae_cfg, device, latent_spatial_dim_global
    
    print(f"Loading VAE model from config: {config_path} and checkpoint: {checkpoint_path}")
    try:
        cfg_dict = load_config(config_path)
        vae_cfg = NestedDictToClass(cfg_dict)
    except Exception as e:
        print(f"Error loading VAE config: {e}")
        exit()

    # Process data with model config after loading config
    process_data_with_model_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        vae_model = AutoencoderKL2D(
            in_channels=vae_cfg.model.in_channels,
            out_channels=vae_cfg.model.out_channels,
            down_block_types=vae_cfg.model.down_block_types,
            up_block_types=vae_cfg.model.up_block_types,
            block_out_channels=vae_cfg.model.block_out_channels,
            layers_per_block=vae_cfg.model.layers_per_block,
            act_fn=vae_cfg.model.act_fn,
            latent_channels=vae_cfg.model.latent_channels,
            norm_num_groups=vae_cfg.model.norm_num_groups,
            sample_points_num=vae_cfg.model.sample_points_num,
            kl_weight=vae_cfg.model.kl_weight,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'ema_model' in checkpoint:
            vae_model.load_state_dict(checkpoint['ema_model'])
            print("Loaded EMA model weights for VAE.")
        elif 'model' in checkpoint:
            vae_model.load_state_dict(checkpoint['model'])
            print("Loaded model weights for VAE.")
        else:
            vae_model.load_state_dict(checkpoint)
            print("Loaded raw model state_dict for VAE.")
        
        vae_model.to(device)
        vae_model.eval()
        print("VAE model loaded and set to evaluation mode.")
    except Exception as e:
        print(f"Error initializing or loading VAE model: {e}")
        exit()
        
    latent_spatial_dim_global = calculate_latent_spatial_dim(vae_cfg.model)

# --- Helper to get a new random sample index (avoiding those already shown if possible) ---
def get_new_random_sample_idx(exclude_indices):
    if num_samples_total == 0: return -1
    if len(exclude_indices) >= num_samples_total: return random.randint(0, num_samples_total - 1) # All shown, just pick random
    
    new_idx = random.randint(0, num_samples_total - 1)
    while new_idx in exclude_indices:
        new_idx = random.randint(0, num_samples_total - 1)
    return new_idx

# --- Function to reconstruct a single surface using the VAE model ---
def reconstruct_surface_with_vae(surface_data_nxnxc):
    """
    Reconstruct a single surface using the loaded VAE model.
    
    Args:
        surface_data_nxnxc: numpy array of shape (N, N, C) - the ground truth surface (already processed)
        
    Returns:
        numpy array of shape (N_out, N_out, C) - the reconstructed surface
    """
    global vae_model, vae_cfg, device, model_grid_size
    
    if vae_model is None or vae_cfg is None or device is None:
        print("VAE model not loaded. Cannot reconstruct surface.")
        return surface_data_nxnxc  # Return original as fallback
    
    # Ensure we're using the correct grid size
    expected_grid_size = model_grid_size if model_grid_size is not None else vae_cfg.model.sample_points_num
    if surface_data_nxnxc.shape[0] != expected_grid_size or surface_data_nxnxc.shape[1] != expected_grid_size:
        print(f"Warning: Input surface has {surface_data_nxnxc.shape[:2]} grid, but model expects {expected_grid_size}x{expected_grid_size}")
    
    # Add batch dimension and rearrange: (1, N, N, C) -> (1, C, N, N)
    input_tensor = torch.from_numpy(surface_data_nxnxc).float().unsqueeze(0).to(device)
    input_tensor = rearrange(input_tensor, 'b h w c -> b c h w')
    
    with torch.no_grad():
        # Encode to latent space
        posterior = vae_model.encode(input_tensor).latent_dist
        z = posterior.mode()  # Use mode instead of sampling for deterministic reconstruction
        
        # Generate query points for decoding - create a grid of query points
        grid_size = expected_grid_size
        t = torch.linspace(0, 1, grid_size, device=device)
        t_grid = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)  # (grid_size, grid_size, 2)
        t_queries = t_grid.unsqueeze(0)  # (1, grid_size, grid_size, 2)
        
        # Decode back to surface
        reconstructed = vae_model.decode(z, t_queries).sample  # (1, C, N_out, N_out)
        
        # Remove batch dimension and rearrange to (N_out, N_out, C)
        reconstructed_surface = rearrange(reconstructed[0], 'c h w -> h w c').cpu().numpy()
        
    return reconstructed_surface

# --- Get reconstruction data (either from file or computed on-the-fly) ---
def get_reconstruction_data(sample_idx):
    """
    Get reconstruction data for a given sample index.
    If eval_onthefly is True, compute reconstruction using VAE model.
    Otherwise, return pre-computed reconstruction from data_recon_all.
    """
    global data_recon_all, data_gt_all, eval_onthefly
    
    if eval_onthefly:
        # Reconstruct on-the-fly using VAE model
        surface_gt_data = data_gt_all[sample_idx]  # Shape: (N, N, C)
        return reconstruct_surface_with_vae(surface_gt_data)
    else:
        # Use pre-computed reconstruction
        return data_recon_all[sample_idx]  # Shape: (N, N, C)

# --- Helper function for color gradient by index order ---
def get_color_gradient_by_index(num_points, color_offset=None):
    """
    Generates colors for points based on their index order using a gradient
    from light blue to light yellow, with optional color offset.
    num_points: number of points to generate colors for
    color_offset: optional RGB offset to add to the base colors (for variations)
    """
    # Define colors: light blue (R, G, B), light yellow (R, G, B)
    light_blue = np.array([0.5, 0.7, 1.0]) # Example light blue
    light_yellow = np.array([1.0, 1.0, 0.5]) # Example light yellow

    # Apply color offset if provided
    if color_offset is not None:
        light_blue = light_blue + np.array(color_offset)
        light_yellow = light_yellow + np.array(color_offset)
        # Clamp to valid color range [0, 1]
        light_blue = np.clip(light_blue, 0.0, 1.0)
        light_yellow = np.clip(light_yellow, 0.0, 1.0)

    # Create indices from 0 to num_points-1
    indices = np.arange(num_points)
    
    # Normalize indices to [0, 1]
    if num_points <= 1:
        # Handle case where there's only one point or no points
        normalized_indices = np.full(num_points, 0.5) # Assign middle color
    else:
        normalized_indices = indices / (num_points - 1)
    
    # Clamp normalized_indices to [0, 1] (should already be in range but safe check)
    normalized_indices = np.clip(normalized_indices, 0, 1)

    # Linear interpolation between light blue and light yellow
    # color = (1 - t) * light_blue + t * light_yellow, where t is normalized index
    colors = (1 - normalized_indices[:, None]) * light_blue + normalized_indices[:, None] * light_yellow

    return colors

# --- Helper function to create faces for surface mesh ---
def create_surface_mesh_faces(grid_size):
    """
    Generates faces (triangles) for a surface represented as a grid of points.
    """
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Triangle 1: (i, j), (i+1, j), (i, j+1)
            v1 = i * grid_size + j
            v2 = (i + 1) * grid_size + j
            v3 = i * grid_size + (j + 1)
            faces.append([v1, v2, v3])

            # Triangle 2: (i+1, j), (i+1, j+1), (i, j+1)
            v4 = (i + 1) * grid_size + j
            v5 = (i + 1) * grid_size + (j + 1)
            v6 = i * grid_size + (j + 1)
            faces.append([v4, v5, v6])

    return np.array(faces)

# --- Polyscope Plotting/Updating a single surface pair ---
def display_or_update_single_surface_pair(slot_idx, sample_idx):
    global ps_gt_surface_structures, ps_recon_surface_structures

    # Ensure data_gt_all is populated
    if data_gt_all is None:
        print("Data not loaded. Cannot display surfaces.")
        return

    surface_gt_grid = data_gt_all[sample_idx]  # Shape: (N, N, C)
    surface_recon_grid = get_reconstruction_data(sample_idx)  # Shape: (N, N, C)

    # Flatten grids for polyscope
    surface_gt_points = surface_gt_grid.reshape(-1, 3)  # (N*N, 3)
    surface_recon_points = surface_recon_grid.reshape(-1, 3)  # (N*N, 3)

    # Get colors based on the index order of points
    gt_colors = get_color_gradient_by_index(len(surface_gt_points))
    recon_colors = get_color_gradient_by_index(len(surface_recon_points))

    # Create faces for the mesh
    grid_size = surface_gt_grid.shape[0]
    faces = create_surface_mesh_faces(grid_size)

    gt_name = f"ground_truth_surface_{slot_idx}"
    recon_name = f"reconstructed_surface_{slot_idx}"

    # Ensure lists are long enough for Polyscope structures
    while len(ps_gt_surface_structures) <= slot_idx:
        ps_gt_surface_structures.append(None)
    while len(ps_recon_surface_structures) <= slot_idx:
        ps_recon_surface_structures.append(None)

    # Ground Truth Surface
    num_points_gt = surface_gt_points.shape[0]
    if num_points_gt > 0:
        if ps_gt_surface_structures[slot_idx] is None or not ps.has_surface_mesh(gt_name):
            ps_gt_surface_structures[slot_idx] = ps.register_surface_mesh(gt_name, surface_gt_points, faces)
        else:
            ps_gt_surface_structures[slot_idx].update_vertex_positions(surface_gt_points)
        
        # Apply index-based coloring and set other properties
        ps_gt_surface_structures[slot_idx].add_color_quantity("index_color", gt_colors, enabled=True)
        ps_gt_surface_structures[slot_idx].set_edge_width(1.0)
        ps_gt_surface_structures[slot_idx].set_smooth_shade(True)
        
    elif ps_gt_surface_structures[slot_idx] is not None and ps.has_surface_mesh(gt_name):
        ps.remove_surface_mesh(gt_name)
        ps_gt_surface_structures[slot_idx] = None

    # Reconstructed Surface
    num_points_recon = surface_recon_points.shape[0]
    if num_points_recon > 0:
        if ps_recon_surface_structures[slot_idx] is None or not ps.has_surface_mesh(recon_name):
            ps_recon_surface_structures[slot_idx] = ps.register_surface_mesh(recon_name, surface_recon_points, faces)
        else:
            ps_recon_surface_structures[slot_idx].update_vertex_positions(surface_recon_points)
        
        # Apply index-based coloring and set other properties
        ps_recon_surface_structures[slot_idx].add_color_quantity("index_color", recon_colors, enabled=True)
        ps_recon_surface_structures[slot_idx].set_edge_width(1.0)
        ps_recon_surface_structures[slot_idx].set_smooth_shade(True)
        
    elif ps_recon_surface_structures[slot_idx] is not None and ps.has_surface_mesh(recon_name):
        ps.remove_surface_mesh(recon_name)
        ps_recon_surface_structures[slot_idx] = None

# --- Function to clear sampled variations for a given slot ---
def clear_sampled_variations_for_slot(source_slot_idx):
    global ps_sampled_variation_structures
    if source_slot_idx in ps_sampled_variation_structures:
        for i, var_ps_struct in enumerate(ps_sampled_variation_structures[source_slot_idx]):
            var_name = f"sampled_variation_surface_{source_slot_idx}_{i}"
            if var_ps_struct is not None and ps.has_surface_mesh(var_name):
                ps.remove_surface_mesh(var_name)
        ps_sampled_variation_structures[source_slot_idx] = []
        print(f"Cleared sampled variations for slot {source_slot_idx}.")

# --- Manage the total number of displayed surfaces based on UI --- 
def manage_surface_display_count(force_refresh_all=False):
    global current_sample_indices, ps_gt_surface_structures, ps_recon_surface_structures, ps_sampled_variation_structures
    desired_num = num_surfaces_to_display_ui[0]
    current_num_on_screen = len(current_sample_indices)

    if not ps.is_initialized(): ps.init() # Should be initialized by main, but safe check

    if force_refresh_all:
        print(f"Refreshing all {current_num_on_screen} displayed GT/Recon surfaces.")
        temp_current_indices = list(current_sample_indices) # Keep a reference for exclusion
        for i in range(current_num_on_screen):
            new_sample_idx = get_new_random_sample_idx(temp_current_indices)
            current_sample_indices[i] = new_sample_idx
            display_or_update_single_surface_pair(i, new_sample_idx)
            clear_sampled_variations_for_slot(i) # Clear variations when base surface changes
            if new_sample_idx not in temp_current_indices: # For next iteration, avoid self-collision
                temp_current_indices.append(new_sample_idx)
        return

    # Add surfaces if desired > current
    while desired_num > len(current_sample_indices):
        slot_idx = len(current_sample_indices)
        new_sample_idx = get_new_random_sample_idx(current_sample_indices)
        if new_sample_idx == -1: break # No more samples
        
        current_sample_indices.append(new_sample_idx)
        print(f"Adding GT/Recon surface pair at slot {slot_idx} with data sample {new_sample_idx}")
        display_or_update_single_surface_pair(slot_idx, new_sample_idx)

    # Remove surfaces if desired < current
    while desired_num < len(current_sample_indices):
        slot_idx_to_remove = len(current_sample_indices) - 1
        print(f"Removing GT/Recon surface pair from slot {slot_idx_to_remove}")
        if ps_gt_surface_structures and len(ps_gt_surface_structures) > slot_idx_to_remove and ps_gt_surface_structures[slot_idx_to_remove] and ps.has_surface_mesh(f"ground_truth_surface_{slot_idx_to_remove}"):
            ps.remove_surface_mesh(f"ground_truth_surface_{slot_idx_to_remove}")
        if ps_recon_surface_structures and len(ps_recon_surface_structures) > slot_idx_to_remove and ps_recon_surface_structures[slot_idx_to_remove] and ps.has_surface_mesh(f"reconstructed_surface_{slot_idx_to_remove}"):
            ps.remove_surface_mesh(f"reconstructed_surface_{slot_idx_to_remove}")
        
        clear_sampled_variations_for_slot(slot_idx_to_remove) # Clear its variations

        current_sample_indices.pop()
        if ps_gt_surface_structures: ps_gt_surface_structures.pop()
        if ps_recon_surface_structures: ps_recon_surface_structures.pop()
        if slot_idx_to_remove in ps_sampled_variation_structures: # Clean up dict entry
            del ps_sampled_variation_structures[slot_idx_to_remove]
        
    # Update printout for current number
    if current_num_on_screen != len(current_sample_indices):
        print(f"Now displaying {len(current_sample_indices)} GT/Recon surface pair(s).")

# --- Function to register manual XYZ axes ---
def register_manual_xyz_axes(length=1.0, radius=0.02):
    if not ps.is_initialized(): ps.init()

    # X-axis (Red)
    nodes_x = np.array([[0,0,0], [length,0,0]])
    edges_x = np.array([[0,1]])
    ps_x = ps.register_curve_network("x_axis", nodes_x, edges_x, radius=radius)
    ps_x.set_color((0.8, 0.1, 0.1)); ps_x.set_radius(radius)

    # Y-axis (Green)
    nodes_y = np.array([[0,0,0], [0,length,0]])
    edges_y = np.array([[0,1]])
    ps_y = ps.register_curve_network("y_axis", nodes_y, edges_y)
    ps_y.set_color((0.1, 0.8, 0.1)); ps_y.set_radius(radius)

    # Z-axis (Blue)
    nodes_z = np.array([[0,0,0], [0,0,length]])
    edges_z = np.array([[0,1]])
    ps_z = ps.register_curve_network("z_axis", nodes_z, edges_z)
    ps_z.set_color((0.1, 0.1, 0.8)); ps_z.set_radius(radius)
    print("Manually registered XYZ axes.")

# --- Core VAE Sampling Function ---
def generate_latent_variations(source_surface_data_nxnxc, num_variations, sigma_val):
    global vae_model, vae_cfg, device, latent_spatial_dim_global, model_grid_size
    if vae_model is None or vae_cfg is None or device is None or latent_spatial_dim_global is None:
        print("VAE model not loaded. Cannot generate variations.")
        return []

    if sigma_val <= 0:
        print("Sigma scale must be positive.")
        return []

    # Input surface_data is expected as (N, N, C) from data_gt_all[idx] (already processed)
    # Add batch dim and rearrange: (1, N, N, C) -> (1, C, N, N)
    input_tensor = torch.from_numpy(source_surface_data_nxnxc).float().unsqueeze(0).to(device)
    input_tensor = rearrange(input_tensor, 'b h w c -> b c h w')
    
    # Ensure we're using the correct number of points
    expected_grid_size = model_grid_size if model_grid_size is not None else vae_cfg.model.sample_points_num
    
    generated_surfaces_list = []
    with torch.no_grad():
        posterior_orig = vae_model.encode(input_tensor).latent_dist
        original_mean = posterior_orig.mean 
        original_logvar = posterior_orig.logvar

        sigma_scale_tensor = torch.tensor(sigma_val, device=original_logvar.device, dtype=original_logvar.dtype)
        log_sigma_scale_val = torch.log(sigma_scale_tensor)
        scaled_logvar = original_logvar + 2 * log_sigma_scale_val
        
        moments_for_sampling_dist = torch.cat([original_mean, scaled_logvar], dim=1)
        sampling_distribution = DiagonalGaussianDistribution(moments_for_sampling_dist)

        z_samples_list = []
        for _ in range(num_variations):
            z_samples_list.append(sampling_distribution.sample()) 
        
        if not z_samples_list: return []
        z_to_decode = torch.cat(z_samples_list, dim=0) # (num_variations, latent_C, latent_H, latent_W)
        
        # Prepare query points 't' for the decoder - use the model's expected grid size
        grid_size = expected_grid_size
        t = torch.linspace(0, 1, grid_size, device=device)
        t_grid = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)  # (grid_size, grid_size, 2)
        t_queries = t_grid.unsqueeze(0).repeat(num_variations, 1, 1, 1)  # (num_variations, grid_size, grid_size, 2)
        
        decoded_surfaces_batch = vae_model.decode(z_to_decode, t_queries).sample # (num_variations, C, N_out, N_out)
        
        for i in range(decoded_surfaces_batch.shape[0]):
            # Rearrange to (N_out, N_out, C) for Polyscope
            generated_surfaces_list.append(rearrange(decoded_surfaces_batch[i], 'c h w -> h w c').cpu().numpy()) 
            
    print(f"Generated {len(generated_surfaces_list)} surface variations with sigma={sigma_val}.")
    return generated_surfaces_list

# --- Polyscope User Interface Callback --- 
def my_ui_callback():
    global num_surfaces_to_display_ui, current_sample_indices, data_gt_all
    global ui_source_slot_for_sampling, ui_num_variations, ui_sigma_scale, ps_sampled_variation_structures
    global eval_onthefly, npz_grid_size, model_grid_size, skip_factor  # Added sampling info
    
    # --- Display current evaluation mode ---
    if eval_onthefly:
        ps.imgui.TextColored((0.2, 0.8, 0.2, 1), "Mode: On-the-fly VAE Reconstruction")
    else:
        ps.imgui.TextColored((0.8, 0.8, 0.2, 1), "Mode: Pre-computed Reconstructions")
    
    # --- Display sampling information ---
    if npz_grid_size is not None and model_grid_size is not None:
        if skip_factor == 1:
            ps.imgui.TextColored((0.7, 0.7, 0.7, 1), f"Grid: {npz_grid_size}x{npz_grid_size} (NPZ) = {model_grid_size}x{model_grid_size} (Model) - No sampling")
        else:
            ps.imgui.TextColored((0.9, 0.7, 0.3, 1), f"Grid: {npz_grid_size}x{npz_grid_size} (NPZ) -> {model_grid_size}x{model_grid_size} (Model), Skip: {skip_factor}")
    
    ps.imgui.Separator()
    
    # --- GT/Recon Display Management ---
    ps.imgui.PushItemWidth(100)
    changed_num_display, num_surfaces_to_display_ui[0] = ps.imgui.InputInt("Num GT/Recon", num_surfaces_to_display_ui[0], step=1, step_fast=5)
    ps.imgui.PopItemWidth()
    if num_surfaces_to_display_ui[0] < 0: num_surfaces_to_display_ui[0] = 0 
    max_displayable_main = min(num_samples_total, 20)  # Limit to 20 surfaces for performance
    if num_surfaces_to_display_ui[0] > max_displayable_main: num_surfaces_to_display_ui[0] = max_displayable_main

    if changed_num_display:
        manage_surface_display_count(force_refresh_all=False)

    if ps.imgui.Button("Refresh GT/Recon Surfaces"):
        if len(current_sample_indices) > 0:
             manage_surface_display_count(force_refresh_all=True)
        else: 
             manage_surface_display_count(force_refresh_all=False)
    
    ps.imgui.Separator() # --- VAE Sampling Section ---
    ps.imgui.Text("VAE Sampling Controls")

    # Input for source slot for sampling
    num_currently_displayed = len(current_sample_indices)
    if num_currently_displayed > 0:
        ps.imgui.PushItemWidth(100)
        slot_changed, ui_source_slot_for_sampling[0] = ps.imgui.InputInt("Source Slot Idx", ui_source_slot_for_sampling[0])
        ps.imgui.PopItemWidth()
        if ui_source_slot_for_sampling[0] < 0: ui_source_slot_for_sampling[0] = 0
        if ui_source_slot_for_sampling[0] >= num_currently_displayed: ui_source_slot_for_sampling[0] = num_currently_displayed -1
        
        # Sampling parameters
        ps.imgui.PushItemWidth(100)
        _, ui_num_variations[0] = ps.imgui.InputInt("Num Variations", ui_num_variations[0])
        if ui_num_variations[0] < 1: ui_num_variations[0] = 1
        ps.imgui.SameLine()
        _, ui_sigma_scale[0] = ps.imgui.InputFloat("Sigma Scale", ui_sigma_scale[0], format="%.2f")
        if ui_sigma_scale[0] <= 0: ui_sigma_scale[0] = 0.01 # Must be positive
        ps.imgui.PopItemWidth()

        if ps.imgui.Button(f"Generate Variations for Slot {ui_source_slot_for_sampling[0]}"):
            source_slot_idx_val = ui_source_slot_for_sampling[0]
            if 0 <= source_slot_idx_val < len(current_sample_indices):
                data_sample_idx_for_source = current_sample_indices[source_slot_idx_val]
                source_surface_data = data_gt_all[data_sample_idx_for_source] # This is (N, N, C)

                print(f"Attempting to generate {ui_num_variations[0]} variations for GT surface in slot {source_slot_idx_val} (data sample {data_sample_idx_for_source}) with sigma {ui_sigma_scale[0]:.2f}")
                
                # Clear old variations for this slot first
                clear_sampled_variations_for_slot(source_slot_idx_val)
                
                newly_sampled_surfaces = generate_latent_variations(source_surface_data, ui_num_variations[0], ui_sigma_scale[0])
                
                # Visualize the generated surface variations
                current_variations_for_slot = []
                
                # Define a consistent orange offset for all variations
                orange_offset = [0.3, -0.2, -0.4]  # Shift towards orange/red spectrum
                
                for i, variation_surface_nxnxc in enumerate(newly_sampled_surfaces): # variation_surface_nxnxc is (N, N, C)
                    var_name = f"sampled_variation_surface_{source_slot_idx_val}_{i}"
                    
                    # Flatten surface for polyscope
                    variation_points = variation_surface_nxnxc.reshape(-1, 3)  # (N*N, 3)
                    grid_size = variation_surface_nxnxc.shape[0]
                    faces = create_surface_mesh_faces(grid_size)
                    
                    # Get colors based on the index order of points with orange offset
                    variation_colors = get_color_gradient_by_index(len(variation_points), color_offset=orange_offset)
                    
                    if variation_points.shape[0] > 0:
                        var_ps_struct = ps.register_surface_mesh(var_name, variation_points, faces)
                        
                        # Apply index-based coloring with orange offset and set properties
                        var_ps_struct.add_color_quantity("index_color", variation_colors, enabled=True)
                        var_ps_struct.set_edge_width(0.5) # Thinner edges for variations
                        var_ps_struct.set_smooth_shade(True)
                        
                        current_variations_for_slot.append(var_ps_struct)
                    else:
                        current_variations_for_slot.append(None)
                        
                ps_sampled_variation_structures[source_slot_idx_val] = current_variations_for_slot
                print(f"Displayed {len(current_variations_for_slot)} surface variations.")
                
            else:
                print(f"Invalid source slot index: {source_slot_idx_val}")
    else:
        ps.imgui.TextDisabled("No GT/Recon surfaces displayed to select as source.")

# --- Main Execution ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize Surface VAE Reconstructions and Sample Latent Variations.")
    parser.add_argument('--npz_file', type=str, required=True, help="Path to the .npz file containing 'ground_truth' and optionally 'reconstructions' (not needed with --eval_onthefly).")
    parser.add_argument('--config', type=str, required=True, help="Path to the VAE model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the VAE model checkpoint file.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--eval_onthefly', action='store_true', help="Reconstruct ground truth data using the VAE model instead of using pre-computed reconstructions.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")

    load_npz_data(args.npz_file, args.eval_onthefly) # Load ground truth and reconstruction data
    load_vae_model_and_config(args.config, args.checkpoint) # Load VAE model

    # Initialize Polyscope
    ps.init()

    # Manually register XYZ axes
    # You might want to adjust the length based on your data's typical scale
    axis_length = 1 # Example length, adjust as needed
    register_manual_xyz_axes(length=axis_length, radius=0.01 * axis_length) 

    # Set the user callback function
    ps.set_user_callback(my_ui_callback)

    # Initial display based on default num_surfaces_to_display_ui[0]
    manage_surface_display_count()

    # Show the Polyscope GUI. This is a blocking call.
    ps.show() 