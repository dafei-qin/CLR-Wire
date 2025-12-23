"""
Test script for VAE v4 DC-AE FSQ model.

This script visualizes three pipelines for bspline surfaces:
1. json -> visualize_json_interset (raw JSON visualization)
2. json -> dataset_v2.dataset_compound (canonical space) -> from_canonical (optional) -> visualize (dataset round-trip)
3. json -> dataset_v2.dataset_compound (canonical space) -> vae_v4_dcae_fsq -> from_canonical (optional) -> visualize (VAE reconstruction)

Note: dataset_v2 returns surfaces in canonical space. from_canonical is only applied if config.canonical=True.
"""
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
import copy
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v2 import dataset_compound
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from src.utils.config import load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from utils.surface import visualize_json_interset
import einops

# Global variables for interactive visualization
dataset = None
model = None
current_idx = 0
max_idx = 0
json_files = []
valid_to_original_idx = []  # Mapping from valid idx to original dataset idx
canonical = False  # Whether dataset uses canonical space (loaded from config)

# Polyscope groups
gt_group = None
dataset_roundtrip_group = None
vae_recon_group = None

# Surfaces dictionaries
gt_surfaces = {}
dataset_roundtrip_surfaces = {}
vae_recon_surfaces = {}

# Visibility flags
show_gt = True
show_dataset_roundtrip = True
show_vae_recon = True


def load_json_file(json_path):
    """Load and validate JSON file for bspline surfaces."""
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Filter for bspline surfaces only
    bspline_surfaces = [surf for surf in json_data if surf.get('type') == 'bspline_surface']
    
    if len(bspline_surfaces) == 0:
        return None
    
    return json_data


def process_sample(idx):
    """
    Process a single sample through all three pipelines.
    
    Args:
        idx: Valid index (index in json_files list)
    
    Returns:
        gt_data: Raw JSON data (in original space)
        dataset_roundtrip_data: JSON after dataset parsing (canonical -> original if config.canonical=True)
        vae_recon_data: JSON after VAE reconstruction (canonical -> original if config.canonical=True)
    """
    global dataset, model, valid_to_original_idx, canonical
    
    # Map valid idx to original dataset idx
    dataset_idx = valid_to_original_idx[idx]
    json_path = json_files[idx]
    
    print(f'\n{"="*70}')
    print(f'Processing file: {json_path}')
    print(f'Valid index: {idx}, Dataset index: {dataset_idx}')
    print(f'{"="*70}')
    
    # Load raw JSON
    json_data = load_json_file(json_path)
    if json_data is None:
        print(f"Skipping {json_path}: No bspline surfaces found")
        return None, None, None
    
    # Filter bspline surfaces
    bspline_surfaces = [surf for surf in json_data if surf.get('type') == 'bspline_surface']
    print(f"Found {len(bspline_surfaces)} bspline surfaces")
    
    # Pipeline 1: Raw JSON visualization
    gt_data = bspline_surfaces
    
    # Pipeline 2: Dataset round-trip (dataset returns canonical space, optionally convert back)
    try:
        # Get dataset sample (already in canonical space after dataset processing)
        # Use the mapped dataset index
        sample = dataset[dataset_idx]
        
        # Unpack sample from dataset_v2
        # Format: (params_tensor, types_tensor, mask_tensor, shift, rotation, scale)
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale = sample
        
        # Apply mask to get only valid surfaces
        mask_bool = mask_tensor.bool()
        valid_params = params_tensor[mask_bool]
        valid_types = types_tensor[mask_bool]
        shift = shift[mask_bool.cpu().numpy()]
        rotation = rotation[mask_bool.cpu().numpy()]
        scale = scale[mask_bool.cpu().numpy()]

        # Keep only bspline surfaces (type == 5)
        bspline_mask = (valid_types == 5)
        if bspline_mask.sum() == 0:
            print("No valid bspline surfaces (type==5) in this sample; skipping.")
            return None, None, None

        bspline_mask_np = bspline_mask.cpu().numpy() if hasattr(bspline_mask, "cpu") else bspline_mask
        valid_params = valid_params[bspline_mask]
        valid_types = valid_types[bspline_mask]
        shift = shift[bspline_mask_np]
        rotation = rotation[bspline_mask_np]
        scale = scale[bspline_mask_np]
        
        print(f"Dataset sample shapes:")
        print(f"  params: {valid_params.shape}")
        print(f"  types: {valid_types}")
        print(f"  shift: {shift.shape}")
        print(f"  rotation: {rotation.shape}")
        print(f"  scale: {scale.shape}")
        
        # Recover surfaces using dataset's method (returns surfaces in canonical space)
        dataset_roundtrip_data = []
        for i in range(len(valid_params)):
            recovered_surface = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
            recovered_surface['idx'] = [i, i]
            recovered_surface['orientation'] = 'Forward'
            
            # If canonical=True, apply from_canonical to restore to original space
            if canonical:
                recovered_surface = from_canonical(recovered_surface, shift[i], rotation[i], scale[i])
            
            dataset_roundtrip_data.append(recovered_surface)
        
    except Exception as e:
        print(f"Error in dataset round-trip pipeline: {e}")
        import traceback
        traceback.print_exc()
        dataset_roundtrip_data = []
    
    # Pipeline 3: VAE reconstruction
    try:
        # Extract control points (patches) from params for VAE processing
        # For bspline surfaces, control points are at params[..., 17:17+48] reshaped to (4, 4, 3)
        patches_input = valid_params[..., 17:17+48].reshape(-1, 4, 4, 3)
        patches_input = einops.rearrange(patches_input, 'b h w c -> b c h w')
        patches_input = patches_input.float()
        
        print(f"\nVAE input patches shape: {patches_input.shape}")
        
        # Forward through VAE
        with torch.no_grad():
            patches_input_device = patches_input.to(next(model.parameters()).device)
            x_recon, z_quantized, indices, metrics = model(patches_input_device)
            
            print(f"VAE output shape: {x_recon.shape}")
            print(f"Latent shape: {z_quantized.shape}")
            if indices is not None:
                if indices.ndim == 1:
                    print(f"FSQ indices shape: {indices.shape}, unique codes: {torch.unique(indices).numel()}")
                else:
                    print(f"FSQ indices shape: {indices.shape}")
                    for i in range(indices.shape[1]):
                        print(f"  Codebook {i}: unique codes = {torch.unique(indices[:, i]).numel()}")
            
            # Compute reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(x_recon, patches_input_device)
            print(f"Reconstruction MSE loss: {recon_loss.item():.6f}")
        
        # Convert reconstructed patches back to JSON format
        vae_recon_data = []
        x_recon_cpu = x_recon.cpu()
        x_recon_patches = einops.rearrange(x_recon_cpu, 'b c h w -> b h w c')
        
        # Create reconstructed params by replacing control points in original params
        params_recon = valid_params.clone()
        params_recon[..., 17:17+48] = x_recon_patches.reshape(len(valid_params), -1)
        
        # Recover surfaces from reconstructed params (in canonical space)
        for i in range(len(params_recon)):
            recovered_surface = dataset._recover_surface(params_recon[i].cpu().numpy(), valid_types[i].item())
            recovered_surface['idx'] = [i, i]
            recovered_surface['orientation'] = 'Forward'
            
            # If canonical=True, apply from_canonical to restore to original space
            if canonical:
                recovered_surface = from_canonical(recovered_surface, shift[i], rotation[i], scale[i])
            
            vae_recon_data.append(recovered_surface)
        
    except Exception as e:
        print(f"Error in VAE reconstruction pipeline: {e}")
        import traceback
        traceback.print_exc()
        vae_recon_data = []
    
    return gt_data, dataset_roundtrip_data, vae_recon_data


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, gt_group, dataset_roundtrip_group, vae_recon_group
    global gt_surfaces, dataset_roundtrip_surfaces, vae_recon_surfaces
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    gt_data, dataset_roundtrip_data, vae_recon_data = process_sample(current_idx)
    
    if gt_data is None:
        print("No valid data to visualize")
        return
    
    # Visualize Pipeline 1: Raw JSON
    try:
        gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-5, ps_header='gt')
        for surface_key, surface_data in gt_surfaces.items():
            if 'surface' in surface_data and surface_data['surface'] is not None:
                if 'ps_handler' in surface_data:
                    surface_data['ps_handler'].add_to_group(gt_group)
        print(f"Pipeline 1 (Raw JSON): Visualized {len(gt_surfaces)} surfaces")
    except Exception as e:
        print(f'Error visualizing GT data: {e}')
        gt_surfaces = {}
    
    # Visualize Pipeline 2: Dataset round-trip
    if dataset_roundtrip_data:
        try:
            dataset_roundtrip_surfaces = visualize_json_interset(
                dataset_roundtrip_data, plot=True, plot_gui=False, tol=1e-5, ps_header='dataset_rt'
            )
            for surface_key, surface_data in dataset_roundtrip_surfaces.items():
                if 'surface' in surface_data and surface_data['surface'] is not None:
                    if 'ps_handler' in surface_data:
                        surface_data['ps_handler'].add_to_group(dataset_roundtrip_group)
            print(f"Pipeline 2 (Dataset round-trip): Visualized {len(dataset_roundtrip_surfaces)} surfaces")
        except Exception as e:
            print(f'Error visualizing dataset round-trip data: {e}')
            dataset_roundtrip_surfaces = {}
    
    # Visualize Pipeline 3: VAE reconstruction
    if vae_recon_data:
        try:
            vae_recon_surfaces = visualize_json_interset(
                vae_recon_data, plot=True, plot_gui=False, tol=1e-5, ps_header='vae_recon'
            )
            for surface_key, surface_data in vae_recon_surfaces.items():
                if 'surface' in surface_data and surface_data['surface'] is not None:
                    if 'ps_handler' in surface_data:
                        surface_data['ps_handler'].add_to_group(vae_recon_group)
            print(f"Pipeline 3 (VAE reconstruction): Visualized {len(vae_recon_surfaces)} surfaces")
        except Exception as e:
            print(f'Error visualizing VAE reconstruction data: {e}')
            vae_recon_surfaces = {}
    
    # Configure visibility
    gt_group.set_enabled(show_gt)
    dataset_roundtrip_group.set_enabled(show_dataset_roundtrip)
    vae_recon_group.set_enabled(show_vae_recon)
    
    print(f"\nVisualization complete for index {current_idx}")


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_gt, show_dataset_roundtrip, show_vae_recon
    
    psim.Text("VAE v4 DC-AE FSQ Test")
    psim.Separator()
    
    # Index controls
    slider_changed, slider_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    input_changed, input_idx = psim.InputInt("Go To Index", current_idx)
    if input_changed:
        input_idx = max(0, min(max_idx, input_idx))
        if input_idx != current_idx:
            current_idx = input_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Valid Index: {current_idx}")
    psim.Text(f"Dataset Index: {valid_to_original_idx[current_idx] if current_idx < len(valid_to_original_idx) else 'N/A'}")
    psim.Text(f"Max Valid Index: {max_idx}")
    psim.Text(f"Current File: {json_files[current_idx] if current_idx < len(json_files) else 'N/A'}")
    
    # Group controls
    if gt_group is not None:
        psim.Separator()
        psim.Text("Pipeline Visibility:")
        
        changed, show_gt = psim.Checkbox("Show Pipeline 1 (Raw JSON)", show_gt)
        if changed:
            gt_group.set_enabled(show_gt)
        
        changed, show_dataset_roundtrip = psim.Checkbox("Show Pipeline 2 (Dataset Round-trip)", show_dataset_roundtrip)
        if changed:
            dataset_roundtrip_group.set_enabled(show_dataset_roundtrip)
        
        changed, show_vae_recon = psim.Checkbox("Show Pipeline 3 (VAE Reconstruction)", show_vae_recon)
        if changed:
            vae_recon_group.set_enabled(show_vae_recon)
    
    psim.Separator()
    psim.Text("Legend:")
    psim.Text("  Pipeline 1: Original JSON")
    psim.Text("  Pipeline 2: Dataset parse -> canonical -> recover")
    psim.Text("  Pipeline 3: Dataset -> VAE encode/decode -> recover")


def collect_json_files(dataset):
    """
    Collect all JSON files from dataset that contain bspline surfaces.
    
    Returns:
        json_files_list: List of valid JSON file paths
        idx_mapping: List mapping valid index to original dataset index
    """
    json_files_list = []
    idx_mapping = []
    
    # Try to get json file names from dataset
    if hasattr(dataset, 'json_names'):
        for original_idx, json_path in enumerate(dataset.json_names):
            json_data = load_json_file(json_path)
            if json_data is not None:
                json_files_list.append(json_path)
                idx_mapping.append(original_idx)
    elif hasattr(dataset, 'data_paths'):
        for original_idx, json_path in enumerate(dataset.data_paths):
            json_data = load_json_file(json_path)
            if json_data is not None:
                json_files_list.append(json_path)
                idx_mapping.append(original_idx)
    else:
        # Fallback: iterate through dataset indices
        print("Warning: Could not find json_names or data_paths in dataset")
        for i in range(len(dataset)):
            try:
                # Try to get sample - if valid, assume it has bspline
                sample = dataset[i]
                if sample[-1].item():  # valid flag
                    if i < len(dataset.json_names if hasattr(dataset, 'json_names') else []):
                        json_files_list.append(dataset.json_names[i])
                        idx_mapping.append(i)
            except:
                continue
    
    return json_files_list, idx_mapping


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser(description='Test VAE v4 DC-AE FSQ model')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting index for visualization')
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Load canonical setting from config
    canonical = config.data_val['params'].get('canonical', False)
    print(f"Canonical mode: {canonical}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_from_config(config, section='data_val')
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Collect valid JSON files with bspline surfaces
    print("Collecting JSON files with bspline surfaces...")
    json_files, valid_to_original_idx = collect_json_files(dataset)
    max_idx = len(json_files) - 1
    current_idx = min(args.start_idx, max_idx)
    
    if len(json_files) == 0:
        print("Error: No JSON files with bspline surfaces found!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} files with bspline surfaces")
    print(f"Index mapping created: valid[0-{max_idx}] -> dataset[{min(valid_to_original_idx)}-{max(valid_to_original_idx)}]")
    
    # Load model
    print("Loading model...")
    model = load_model_from_config(config)
    
    # Load checkpoint
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'ema_model' in checkpoint or 'ema' in checkpoint:
            ema_key = 'ema' if 'ema' in checkpoint else 'ema_model'
            ema_model = checkpoint[ema_key]
            ema_model = {k.replace("ema_model.", "").replace("ema.", ""): v for k, v in ema_model.items()}
            model.load_state_dict(ema_model, strict=False)
            print("Loaded EMA model weights.")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Loaded model weights.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded raw model state_dict.")
    
    model.eval()
    
    print(f"\nModel and configuration:")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Canonical mode: {canonical}")
    if hasattr(model, 'codebook_size'):
        print(f"  Codebook size: {model.codebook_size}")
    if hasattr(model, 'num_codebooks'):
        print(f"  Num codebooks: {model.num_codebooks}")
    if hasattr(model, 'latent_dim'):
        print(f"  Latent dim: {model.latent_dim}")
    
    # Initialize polyscope
    ps.init()
    
    gt_group = ps.create_group("Pipeline 1: Raw JSON")
    dataset_roundtrip_group = ps.create_group("Pipeline 2: Dataset Round-trip")
    vae_recon_group = ps.create_group("Pipeline 3: VAE Reconstruction")
    
    ps.set_user_callback(callback)
    
    # Load initial visualization
    print("\nInitializing visualization...")
    update_visualization()
    
    # Show the interface
    print("Starting Polyscope interface...")
    ps.show()

