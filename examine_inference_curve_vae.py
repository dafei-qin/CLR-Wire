import numpy as np
import polyscope as ps # Import Polyscope
import random
import torch # Added
from argparse import ArgumentParser # Added
import os # Added

# VAE related imports
from src.vae.vae_curve import AutoencoderKL1D # Added
from src.utils.config import NestedDictToClass, load_config # Added
from einops import rearrange # Added
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution # Added

# --- Global variables ---
data_gt_all = None
data_recon_all = None
num_samples_total = 0
current_sample_indices = [] # Stores the data sample_idx for each displayed curve pair
eval_onthefly = False # Added: flag to determine if we should reconstruct on-the-fly
npz_points_num = None # Added: number of points in the original .npz data
model_points_num = None # Added: number of points expected by the model
skip_factor = None # Added: skip factor for sampling

# Polyscope structure references (lists to hold multiple structures)
ps_gt_curve_structures = []
ps_recon_curve_structures = []

# UI controllable number of curves
num_curves_to_display_ui = [1] # ImGui input_int needs a list or array

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

# --- Function to apply skipped sampling to curve data ---
def apply_skipped_sampling(curve_data_cxn):
    """
    Apply skipped sampling to curve data if model expects fewer points than npz provides.
    
    Args:
        curve_data_cxn: numpy array of shape (C, N_points_npz) - original curve data
        
    Returns:
        numpy array of shape (C, N_points_model) - sampled curve data
    """
    global skip_factor, model_points_num, npz_points_num
    
    if skip_factor is None or skip_factor == 1:
        return curve_data_cxn  # No sampling needed
    
    # Apply skipped sampling along the points dimension (last dimension)
    sampled_curve = curve_data_cxn[..., ::skip_factor]
    
    # Ensure we get exactly the expected number of points
    if sampled_curve.shape[-1] > model_points_num:
        sampled_curve = sampled_curve[..., :model_points_num]
    elif sampled_curve.shape[-1] < model_points_num:
        print(f"Warning: Skipped sampling resulted in {sampled_curve.shape[-1]} points, but model expects {model_points_num}")
    
    return sampled_curve

# --- Load Data --- 
def load_npz_data(npz_file_path, use_onthefly_eval=False): # Modified to accept eval_onthefly flag
    global data_gt_all, data_recon_all, num_samples_total, eval_onthefly
    global npz_points_num, model_points_num, skip_factor
    eval_onthefly = use_onthefly_eval
    
    try:
        data = np.load(npz_file_path) 
    except FileNotFoundError:
        print(f"Error: Data file '{npz_file_path}' not found. Please check the path.")
        exit()
    
    if 'ground_truth' not in data:
        print(f"Error: 'ground_truth' not found in {npz_file_path}.")
        exit()

    data_gt_all_raw = data['ground_truth']
    npz_points_num = data_gt_all_raw.shape[-1]  # Get number of points in npz data
    print(f"NPZ file contains {npz_points_num} points per curve.")
    
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
    global data_gt_all, data_recon_all, npz_points_num, model_points_num, skip_factor, vae_cfg
    
    if vae_cfg is None:
        print("Error: VAE config not loaded. Cannot process data.")
        return
    
    model_points_num = vae_cfg.model.sample_points_num
    print(f"Model expects {model_points_num} points per curve.")
    
    if npz_points_num == model_points_num:
        skip_factor = 1
        print("NPZ and model point counts match. No sampling needed.")
    elif npz_points_num > model_points_num:
        if npz_points_num % model_points_num == 0:
            print("Skip downsampling")
            return 
            skip_factor = npz_points_num // model_points_num
            print(f"Applying skipped sampling with factor {skip_factor} ({npz_points_num} -> {model_points_num} points).")
            
            # Apply skipped sampling to ground truth data
            print("Processing ground truth data...")
            data_gt_all = apply_skipped_sampling(data_gt_all)
            
            # Apply skipped sampling to reconstruction data if available
            if data_recon_all is not None:
                print("Processing reconstruction data...")
                data_recon_all = apply_skipped_sampling(data_recon_all)
        else:
            print(f"Warning: NPZ points ({npz_points_num}) not evenly divisible by model points ({model_points_num}).")
            skip_factor = npz_points_num // model_points_num
            if skip_factor == 0:
                skip_factor = 1
            print(f"Using skip factor {skip_factor}, may result in approximate point count.")
            
            # Apply skipped sampling
            data_gt_all = apply_skipped_sampling(data_gt_all)
            if data_recon_all is not None:
                data_recon_all = apply_skipped_sampling(data_recon_all)
    else:
        print(f"Error: NPZ has fewer points ({npz_points_num}) than model expects ({model_points_num}). Cannot proceed.")
        exit()
    
    print(f"Data processing complete. Final data shape: {data_gt_all.shape}")

# --- VAE Model Loading ---
def calculate_latent_spatial_dim(cfg_model_vae): # Renamed arg for clarity
    initial_length = cfg_model_vae.sample_points_num
    num_downsamples = 0
    if hasattr(cfg_model_vae, 'block_out_channels') and isinstance(cfg_model_vae.block_out_channels, (list, tuple)) and len(cfg_model_vae.block_out_channels) > 1:
        num_downsamples = len(cfg_model_vae.block_out_channels) - 1
    
    latent_spatial_dim = initial_length // (2**num_downsamples)
    latent_spatial_dim = max(1, latent_spatial_dim) 
    print(f"Calculated latent spatial dimension: {latent_spatial_dim} (from initial length {initial_length} and {num_downsamples} downsamples)")
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
        vae_model = AutoencoderKL1D(
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

# --- Function to reconstruct a single curve using the VAE model ---
def reconstruct_curve_with_vae(curve_data_cxn):
    """
    Reconstruct a single curve using the loaded VAE model.
    
    Args:
        curve_data_cxn: numpy array of shape (C, N_points) - the ground truth curve (already processed)
        
    Returns:
        numpy array of shape (N_out, C) - the reconstructed curve points
    """
    global vae_model, vae_cfg, device, model_points_num
    
    if vae_model is None or vae_cfg is None or device is None:
        print("VAE model not loaded. Cannot reconstruct curve.")
        return curve_data_cxn.transpose()  # Return original as fallback
    
    # Ensure we're using the correct number of points
    expected_points = model_points_num if model_points_num is not None else vae_cfg.model.sample_points_num
    if curve_data_cxn.shape[-1] != expected_points:
        print(f"Warning: Input curve has {curve_data_cxn.shape[-1]} points, but model expects {expected_points}")
    
    # Add batch dimension: (1, C, N_points)
    input_tensor = torch.from_numpy(curve_data_cxn).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode to latent space
        posterior = vae_model.encode(input_tensor).latent_dist
        z = posterior.mode()  # Use mode instead of sampling for deterministic reconstruction
        
        # Generate query points for decoding - use the model's expected number of output points
        t_queries = torch.rand(1, expected_points, device=device)
        t_queries, _ = torch.sort(t_queries, dim=-1)
        
        # Decode back to curve
        reconstructed = vae_model.decode(z, t_queries).sample  # (1, C, N_out)
        
        # Remove batch dimension and transpose to (N_out, C)
        reconstructed_curve = reconstructed[0].cpu().numpy().transpose()
        
    return reconstructed_curve

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
        curve_gt_data = data_gt_all[sample_idx]  # Shape: (C, N_points)
        return reconstruct_curve_with_vae(curve_gt_data)
    else:
        # Use pre-computed reconstruction
        return data_recon_all[sample_idx].transpose()  # (N_out, C)

# --- Polyscope Plotting/Updating a single curve pair ---
def display_or_update_single_curve_pair(slot_idx, sample_idx):
    global ps_gt_curve_structures, ps_recon_curve_structures

    # Ensure data_gt_all is populated
    if data_gt_all is None:
        print("Data not loaded. Cannot display curves.")
        return

    curve_gt_points = data_gt_all[sample_idx].transpose() # (N_out, C)
    curve_recon_points = get_reconstruction_data(sample_idx)  # Modified to use new function

    gt_name = f"ground_truth_curve_{slot_idx}"
    recon_name = f"reconstructed_curve_{slot_idx}"

    # Ensure lists are long enough for Polyscope structures
    while len(ps_gt_curve_structures) <= slot_idx:
        ps_gt_curve_structures.append(None)
    while len(ps_recon_curve_structures) <= slot_idx:
        ps_recon_curve_structures.append(None)

    # Ground Truth Curve
    num_points_gt = curve_gt_points.shape[0]
    if num_points_gt > 1:
        if ps_gt_curve_structures[slot_idx] is None or not ps.has_curve_network(gt_name):
            edges_gt = np.array([[i, i + 1] for i in range(num_points_gt - 1)])
            ps_gt_curve_structures[slot_idx] = ps.register_curve_network(gt_name, curve_gt_points, edges_gt)
        else:
            ps_gt_curve_structures[slot_idx].update_node_positions(curve_gt_points)
        ps_gt_curve_structures[slot_idx].set_color((0.2, 0.2, 0.9)); ps_gt_curve_structures[slot_idx].set_radius(0.007) # Apply style always
    elif ps_gt_curve_structures[slot_idx] is not None and ps.has_curve_network(gt_name):
        ps.remove_curve_network(gt_name); ps_gt_curve_structures[slot_idx] = None

    # Reconstructed Curve
    num_points_recon = curve_recon_points.shape[0]
    if num_points_recon > 1:
        if ps_recon_curve_structures[slot_idx] is None or not ps.has_curve_network(recon_name):
            edges_recon = np.array([[i, i + 1] for i in range(num_points_recon - 1)])
            ps_recon_curve_structures[slot_idx] = ps.register_curve_network(recon_name, curve_recon_points, edges_recon)
        else:
            ps_recon_curve_structures[slot_idx].update_node_positions(curve_recon_points)
        ps_recon_curve_structures[slot_idx].set_color((0.2, 0.9, 0.2)); ps_recon_curve_structures[slot_idx].set_radius(0.007) # Apply style always
    elif ps_recon_curve_structures[slot_idx] is not None and ps.has_curve_network(recon_name):
        ps.remove_curve_network(recon_name); ps_recon_curve_structures[slot_idx] = None

# --- Function to clear sampled variations for a given slot ---
def clear_sampled_variations_for_slot(source_slot_idx):
    global ps_sampled_variation_structures
    if source_slot_idx in ps_sampled_variation_structures:
        for i, var_ps_struct in enumerate(ps_sampled_variation_structures[source_slot_idx]):
            var_name = f"sampled_variation_{source_slot_idx}_{i}"
            if var_ps_struct is not None and ps.has_curve_network(var_name):
                ps.remove_curve_network(var_name)
        ps_sampled_variation_structures[source_slot_idx] = []
        print(f"Cleared sampled variations for slot {source_slot_idx}.")

# --- Manage the total number of displayed curves based on UI --- 
def manage_curve_display_count(force_refresh_all=False):
    global current_sample_indices, ps_gt_curve_structures, ps_recon_curve_structures, ps_sampled_variation_structures
    desired_num = num_curves_to_display_ui[0]
    current_num_on_screen = len(current_sample_indices)

    if not ps.is_initialized(): ps.init() # Should be initialized by main, but safe check

    if force_refresh_all:
        print(f"Refreshing all {current_num_on_screen} displayed GT/Recon curves.")
        temp_current_indices = list(current_sample_indices) # Keep a reference for exclusion
        for i in range(current_num_on_screen):
            new_sample_idx = get_new_random_sample_idx(temp_current_indices)
            current_sample_indices[i] = new_sample_idx
            display_or_update_single_curve_pair(i, new_sample_idx)
            clear_sampled_variations_for_slot(i) # Clear variations when base curve changes
            if new_sample_idx not in temp_current_indices: # For next iteration, avoid self-collision
                temp_current_indices.append(new_sample_idx)
        return

    # Add curves if desired > current
    while desired_num > len(current_sample_indices):
        slot_idx = len(current_sample_indices)
        new_sample_idx = get_new_random_sample_idx(current_sample_indices)
        if new_sample_idx == -1: break # No more samples
        
        current_sample_indices.append(new_sample_idx)
        # Ensure structure lists are ready (handled in display_or_update)
        print(f"Adding GT/Recon curve pair at slot {slot_idx} with data sample {new_sample_idx}")
        display_or_update_single_curve_pair(slot_idx, new_sample_idx)

    # Remove curves if desired < current
    while desired_num < len(current_sample_indices):
        slot_idx_to_remove = len(current_sample_indices) - 1
        print(f"Removing GT/Recon curve pair from slot {slot_idx_to_remove}")
        if ps_gt_curve_structures and len(ps_gt_curve_structures) > slot_idx_to_remove and ps_gt_curve_structures[slot_idx_to_remove] and ps.has_curve_network(f"ground_truth_curve_{slot_idx_to_remove}"):
            ps.remove_curve_network(f"ground_truth_curve_{slot_idx_to_remove}")
        if ps_recon_curve_structures and len(ps_recon_curve_structures) > slot_idx_to_remove and ps_recon_curve_structures[slot_idx_to_remove] and ps.has_curve_network(f"reconstructed_curve_{slot_idx_to_remove}"):
            ps.remove_curve_network(f"reconstructed_curve_{slot_idx_to_remove}")
        
        clear_sampled_variations_for_slot(slot_idx_to_remove) # Clear its variations

        current_sample_indices.pop()
        if ps_gt_curve_structures: ps_gt_curve_structures.pop()
        if ps_recon_curve_structures: ps_recon_curve_structures.pop()
        if slot_idx_to_remove in ps_sampled_variation_structures: # Clean up dict entry
            del ps_sampled_variation_structures[slot_idx_to_remove]
        
    # Update printout for current number
    if current_num_on_screen != len(current_sample_indices):
        print(f"Now displaying {len(current_sample_indices)} GT/Recon curve pair(s).")

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
def generate_latent_variations(source_curve_data_cxn, num_variations, sigma_val):
    global vae_model, vae_cfg, device, latent_spatial_dim_global, model_points_num
    if vae_model is None or vae_cfg is None or device is None or latent_spatial_dim_global is None:
        print("VAE model not loaded. Cannot generate variations.")
        return []

    if sigma_val <= 0:
        print("Sigma scale must be positive.")
        return []

    # Input curve_data is expected as (C, N_points) from data_gt_all[idx] (already processed)
    # Add batch dim: (1, C, N_points)
    input_tensor = torch.from_numpy(source_curve_data_cxn).float().unsqueeze(0).to(device)
    
    # Ensure we're using the correct number of points
    expected_points = model_points_num if model_points_num is not None else vae_cfg.model.sample_points_num
    
    generated_curves_list = []
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
        z_to_decode = torch.cat(z_samples_list, dim=0) # (num_variations, latent_C, latent_N_spatial)
        
        # Prepare query points 't' for the decoder - use the model's expected number of points
        t_queries = torch.rand(num_variations, expected_points, device=device)
        t_queries, _ = torch.sort(t_queries, dim=-1)
        
        decoded_curves_batch = vae_model.decode(z_to_decode, t_queries).sample # (num_variations, C, N_out)
        
        for i in range(decoded_curves_batch.shape[0]):
            # Transpose to (N_out, C) for Polyscope
            generated_curves_list.append(decoded_curves_batch[i].cpu().numpy().transpose()) 
            
    print(f"Generated {len(generated_curves_list)} variations with sigma={sigma_val}.")
    return generated_curves_list

# --- Polyscope User Interface Callback --- 
def my_ui_callback():
    global num_curves_to_display_ui, current_sample_indices, data_gt_all
    global ui_source_slot_for_sampling, ui_num_variations, ui_sigma_scale, ps_sampled_variation_structures
    global eval_onthefly, npz_points_num, model_points_num, skip_factor  # Added sampling info
    
    # --- Display current evaluation mode ---
    if eval_onthefly:
        ps.imgui.TextColored((0.2, 0.8, 0.2, 1), "Mode: On-the-fly VAE Reconstruction")
    else:
        ps.imgui.TextColored((0.8, 0.8, 0.2, 1), "Mode: Pre-computed Reconstructions")
    
    # --- Display sampling information ---
    if npz_points_num is not None and model_points_num is not None:
        if skip_factor == 1:
            ps.imgui.TextColored((0.7, 0.7, 0.7, 1), f"Points: {npz_points_num} (NPZ) = {model_points_num} (Model) - No sampling")
        else:
            ps.imgui.TextColored((0.9, 0.7, 0.3, 1), f"Points: {npz_points_num} (NPZ) -> {model_points_num} (Model), Skip: {skip_factor}")
    
    ps.imgui.Separator()
    
    # --- GT/Recon Display Management ---
    ps.imgui.PushItemWidth(100)
    changed_num_display, num_curves_to_display_ui[0] = ps.imgui.InputInt("Num GT/Recon", num_curves_to_display_ui[0], step=1, step_fast=5)
    ps.imgui.PopItemWidth()
    if num_curves_to_display_ui[0] < 0: num_curves_to_display_ui[0] = 0 
    max_displayable_main = min(num_samples_total, 50) 
    if num_curves_to_display_ui[0] > max_displayable_main: num_curves_to_display_ui[0] = max_displayable_main

    if changed_num_display:
        manage_curve_display_count(force_refresh_all=False)

    if ps.imgui.Button("Refresh GT/Recon Curves"):
        if len(current_sample_indices) > 0:
             manage_curve_display_count(force_refresh_all=True)
        else: 
             manage_curve_display_count(force_refresh_all=False)
    
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
                source_curve_data = data_gt_all[data_sample_idx_for_source] # This is (C, N_out)

                print(f"Attempting to generate {ui_num_variations[0]} variations for GT curve in slot {source_slot_idx_val} (data sample {data_sample_idx_for_source}) with sigma {ui_sigma_scale[0]:.2f}")
                
                # Clear old variations for this slot first
                clear_sampled_variations_for_slot(source_slot_idx_val)
                
                newly_sampled_curves = generate_latent_variations(source_curve_data, ui_num_variations[0], ui_sigma_scale[0])
                
                current_variations_for_slot = []
                for i, variation_points_nx_c in enumerate(newly_sampled_curves): # variation_points_nx_c is (N_out, C)
                    var_name = f"sampled_variation_{source_slot_idx_val}_{i}"
                    num_points_var = variation_points_nx_c.shape[0]
                    if num_points_var > 1:
                        edges_var = np.array([[k, k + 1] for k in range(num_points_var - 1)])
                        var_ps_struct = ps.register_curve_network(var_name, variation_points_nx_c, edges_var)
                        var_ps_struct.set_color((0.9, 0.5, 0.1)) # Orange color for variations
                        var_ps_struct.set_radius(0.005) # Slightly smaller radius
                        current_variations_for_slot.append(var_ps_struct)
                ps_sampled_variation_structures[source_slot_idx_val] = current_variations_for_slot
            else:
                print(f"Invalid source slot index: {source_slot_idx_val}")
    else:
        ps.imgui.TextDisabled("No GT/Recon curves displayed to select as source.")

# --- Main Execution ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize VAE Reconstructions and Sample Latent Variations.")
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

    # Initial display based on default num_curves_to_display_ui[0]
    manage_curve_display_count()

    # Show the Polyscope GUI. This is a blocking call.
    ps.show()

