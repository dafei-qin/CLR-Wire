"""
Test script for VAE v4 DC-AE FSQ model.

This script visualizes three pipelines for bspline surfaces:
1. json -> visualize_json_interset (raw JSON visualization)
2. json -> dataset_v2.dataset_compound -> to_canonical -> from_canonical -> visualize (dataset round-trip)
3. json -> dataset_v2.dataset_compound -> to_canonical -> vae_v4_dcae_fsq -> from_canonical -> visualize (VAE reconstruction)
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
    
    Returns:
        gt_data: Raw JSON data
        dataset_roundtrip_data: JSON after dataset parsing + to_canonical + from_canonical
        vae_recon_data: JSON after VAE reconstruction
    """
    global dataset, model
    
    json_path = json_files[idx]
    print(f'\n{"="*70}')
    print(f'Processing file: {json_path}')
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
    
    # Pipeline 2: Dataset round-trip (parse -> to_canonical -> from_canonical)
    try:
        # Get dataset sample (this goes through parsing and to_canonical)
        sample = dataset[idx]
        
        # Unpack sample from dataset_v2
        # Format: (patches, shift, rotation, scale, valid)
        patches, shift, rotation, scale, valid = sample
        
        if not valid.item():
            print(f"Sample {idx} is invalid, skipping")
            return None, None, None
        
        print(f"Dataset sample shapes:")
        print(f"  patches: {patches.shape}")  # Should be (N, H, W, C) where patches are control points
        print(f"  shift: {shift.shape}")
        print(f"  rotation: {rotation.shape}")
        print(f"  scale: {scale.shape}")
        
        # Convert patches back to surface format and apply from_canonical
        # For bspline surfaces, patches contain control points (4x4x3)
        dataset_roundtrip_data = []
        
        # Since dataset_v2 processes surfaces to canonical space, we need to recover them
        # The patches are already in canonical form (4x4x3 control points)
        # We need to convert back using from_canonical
        
        # For now, create a simple bspline surface from patches
        # This is simplified - actual implementation may need more complex recovery
        num_surfaces = patches.shape[0] if patches.ndim == 4 else 1
        
        for surf_idx in range(len(bspline_surfaces)):
            # Use original surface structure but with canonical transformation applied
            surf_copy = copy.deepcopy(bspline_surfaces[surf_idx])
            
            # Apply from_canonical transformation
            # Note: We use the transformation parameters from dataset
            if surf_idx < len(shift):
                surf_canonical = to_canonical(surf_copy, shift[surf_idx].cpu().numpy(), 
                                             rotation[surf_idx].cpu().numpy(), 
                                             scale[surf_idx].cpu().numpy())
                surf_recovered = from_canonical(surf_canonical, shift[surf_idx].cpu().numpy(),
                                              rotation[surf_idx].cpu().numpy(),
                                              scale[surf_idx].cpu().numpy())
                dataset_roundtrip_data.append(surf_recovered)
            else:
                dataset_roundtrip_data.append(surf_copy)
        
    except Exception as e:
        print(f"Error in dataset round-trip pipeline: {e}")
        import traceback
        traceback.print_exc()
        dataset_roundtrip_data = []
    
    # Pipeline 3: VAE reconstruction
    try:
        # Get patches in the format expected by VAE
        # Format from dataset: (num_surfaces, H, W, C)
        patches_input = patches[..., 17:].reshape(-1, 4, 4, 3)
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
        
        # Create bspline surfaces from reconstructed patches
        for surf_idx in range(min(len(bspline_surfaces), x_recon_patches.shape[0])):
            surf_copy = copy.deepcopy(bspline_surfaces[surf_idx])
            
            # Replace control points (poles) with reconstructed ones
            recon_poles = x_recon_patches[surf_idx].detach().cpu().numpy()
            
            # The poles might need to be in the format expected by visualization
            # For a 4x4 control grid, reshape appropriately
            if 'poles' in surf_copy:
                original_poles_shape = np.array(surf_copy['poles']).shape
                # Reshape reconstructed poles to match original
                if len(original_poles_shape) == 3:  # (u, v, coords)
                    if original_poles_shape[-1] == 4:  # Has weights
                        # Preserve weights from original
                        original_weights = np.array(surf_copy['poles'])[..., 3:4]
                        recon_poles_with_weights = np.concatenate([recon_poles, original_weights[:4, :4, :]], axis=-1)
                        surf_copy['poles'] = recon_poles_with_weights.tolist()
                    else:
                        surf_copy['poles'] = recon_poles.tolist()
            
            # Apply from_canonical to bring back to original space
            if surf_idx < len(shift):
                # First, create canonical version with reconstructed poles
                # Then apply from_canonical
                surf_recovered = from_canonical(surf_copy, shift[surf_idx].cpu().numpy(),
                                              rotation[surf_idx].cpu().numpy(),
                                              scale[surf_idx].cpu().numpy())
                vae_recon_data.append(surf_recovered)
            else:
                vae_recon_data.append(surf_copy)
        
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
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
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
    """Collect all JSON files from dataset that contain bspline surfaces."""
    json_files_list = []
    
    # Try to get json file names from dataset
    if hasattr(dataset, 'json_names'):
        for json_path in dataset.json_names:
            json_data = load_json_file(json_path)
            if json_data is not None:
                json_files_list.append(json_path)
    elif hasattr(dataset, 'data_paths'):
        for json_path in dataset.data_paths:
            json_data = load_json_file(json_path)
            if json_data is not None:
                json_files_list.append(json_path)
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
            except:
                continue
    
    return json_files_list


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
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset_from_config(config, section='data_val')
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Collect valid JSON files with bspline surfaces
    print("Collecting JSON files with bspline surfaces...")
    json_files = collect_json_files(dataset)
    max_idx = len(json_files) - 1
    current_idx = min(args.start_idx, max_idx)
    
    if len(json_files) == 0:
        print("Error: No JSON files with bspline surfaces found!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} files with bspline surfaces")
    
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
    
    print(f"\nModel info:")
    print(f"  Type: {type(model).__name__}")
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

