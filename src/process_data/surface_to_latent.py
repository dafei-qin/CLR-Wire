"""
Script to convert surface data from dataset_v1 to latent representations using VAE.

This script:
1. Loads all data from dataset_v1
2. Encodes surfaces to latent space using vae_v1
3. Saves latent params, rotation, shift, scale, and class to npz files

The output npz files maintain the same relative directory structure as the input JSON files,
but with a different root directory specified by the user.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP
from src.vae.vae_v1 import SurfaceVAE


def load_model(checkpoint_path, device='cpu'):
    """Load the VAE model from checkpoint"""
    # Initialize model with the correct parameter dimensions
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to load model weights
    if 'ema_model' in checkpoint:
        ema_model = checkpoint['ema']
        ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
        model.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")
    
    model.to(device)
    model.eval()
    return model


def process_sample(model, dataset, params_tensor, types_tensor, mask_tensor, 
                   all_shifts, all_rotations, all_scales, device='cpu'):
    """
    Process a single sample and extract latent representations.
    Also decode to check reconstruction accuracy.
    
    Args:
        model: The VAE model
        dataset: The dataset object (for getting valid param masks)
        params_tensor: Surface parameters (max_num_surfaces, max_param_dim)
        types_tensor: Surface types (max_num_surfaces,)
        mask_tensor: Valid surface mask (max_num_surfaces,)
        all_shifts: Shift vectors (max_num_surfaces, 3)
        all_rotations: Rotation matrices (max_num_surfaces, 3, 3)
        all_scales: Scale values (max_num_surfaces,)
        device: Device to run inference on
        
    Returns:
        Dictionary containing:
        - latent_params: (num_valid_surfaces, latent_dim)
        - rotations: (num_valid_surfaces, 6) - first 6 elements of rotation matrix
        - scales: (num_valid_surfaces, 1)
        - shifts: (num_valid_surfaces, 3)
        - classes: (num_valid_surfaces, 1)
        - cls_acc: classification accuracy (float)
        - params_mse: mean squared error of reconstructed parameters (float)
    """
    # Get valid surfaces based on mask
    valid_mask = mask_tensor.bool()
    valid_params = params_tensor[valid_mask]
    valid_types = types_tensor[valid_mask]
    valid_shifts = all_shifts[valid_mask]
    valid_rotations = all_rotations[valid_mask]
    valid_scales = all_scales[valid_mask]
    
    # Ensure all tensors are torch tensors (in case they are numpy arrays)
    if not isinstance(valid_shifts, torch.Tensor):
        valid_shifts = torch.from_numpy(valid_shifts)
    if not isinstance(valid_rotations, torch.Tensor):
        valid_rotations = torch.from_numpy(valid_rotations)
    if not isinstance(valid_scales, torch.Tensor):
        valid_scales = torch.from_numpy(valid_scales)
    
    # Move to device
    valid_params = valid_params.to(device)
    valid_types = valid_types.to(device)
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(valid_params, valid_types)
        # Use mean (mu) as the latent representation for deterministic encoding
        latent = mu
        
        # Classify and decode to check reconstruction accuracy
        type_logits_pred, types_pred = model.classify(latent)
        params_recon, recon_mask = model.decode(latent, valid_types)
        
        # Calculate classification accuracy
        cls_acc = (types_pred == valid_types).float().mean().item()
        
        # Calculate parameter reconstruction MSE (only on valid dimensions)
        # Get the actual valid parameter mask based on surface types
        # This gives us the true parameter dimensions, not just what the decoder outputs
        valid_param_mask = dataset.get_valid_param_mask(valid_types, return_tensor=True).to(device)
        
        # Calculate MSE only on the true valid parameters
        squared_errors = (params_recon - valid_params) ** 2
        masked_errors = squared_errors * valid_param_mask.float()
        params_mse = masked_errors.sum() / valid_param_mask.float().sum()
        params_mse = params_mse.item()
    
    # Extract first 6 elements of rotation matrix (first two rows)
    # Rotation matrix shape: (N, 3, 3) -> (N, 6) by taking [:, :2, :].flatten(1, 2)
    rotations_6d = valid_rotations[:, :2, :].reshape(-1, 6)
    
    # Expand dimensions for scales
    scales_expanded = valid_scales.unsqueeze(-1)
    
    # Expand dimensions for classes
    classes = valid_types.unsqueeze(-1)
    
    # Return as numpy arrays
    return {
        'latent_params': latent.cpu().numpy(),
        'rotations': rotations_6d.cpu().numpy(),
        'scales': scales_expanded.cpu().numpy(),
        'shifts': valid_shifts.cpu().numpy(),
        'classes': classes.cpu().numpy(),
        'cls_acc': cls_acc,
        'params_mse': params_mse
    }


def get_output_path(input_json_path, input_root, output_root):
    """
    Get the output npz file path maintaining the same relative directory structure.
    
    Args:
        input_json_path: Full path to input JSON file
        input_root: Root directory of input JSON files
        output_root: Root directory for output npz files
        
    Returns:
        Full path to output npz file
    """
    input_path = Path(input_json_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    # Get relative path from input root
    relative_path = input_path.relative_to(input_root)
    
    # Change extension from .json to .npz
    relative_path = relative_path.with_suffix('.npz')
    
    # Construct output path
    output_path = output_root / relative_path
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert surface data to latent representations using VAE'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Input directory containing JSON files'
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Path to VAE model checkpoint'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Output directory for npz files (will maintain same subdirectory structure)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    parser.add_argument(
        '--canonical',
        action='store_true',
        help='Use canonical dataset'
    )
    parser.add_argument(
        '--batch_process',
        action='store_true',
        help='Process in batches for better efficiency'
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load dataset
    print(f"Loading dataset from: {args.input_dir}")
    dataset = dataset_compound(args.input_dir, canonical=args.canonical)
    print(f"Found {len(dataset)} samples")
    
    # Load model
    print(f"Loading model from: {args.checkpoint_path}")
    model = load_model(args.checkpoint_path, device=args.device)
    
    # Create output directory
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Process all samples
    print("Processing samples...")
    failed_samples = []
    
    # Statistics tracking
    all_cls_acc = []
    all_params_mse = []
    
    for idx in tqdm(range(len(dataset)), desc="Converting to latent"):
        try:
            # Get data from dataset
            params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales = dataset[idx]
            
            # Check if there are any valid surfaces
            num_valid = mask_tensor.sum().item()
            if num_valid == 0:
                print(f"\nWarning: No valid surfaces in sample {idx} ({dataset.json_names[idx]})")
                continue
            
            # Process sample
            result = process_sample(
                model, dataset, params_tensor, types_tensor, mask_tensor,
                all_shifts, all_rotations, all_scales,
                device=args.device
            )
            
            # Track reconstruction metrics
            all_cls_acc.append(result['cls_acc'])
            all_params_mse.append(result['params_mse'])
            
            # Get output path
            input_json_path = dataset.json_names[idx]
            output_path = get_output_path(input_json_path, args.input_dir, args.output_dir)
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to npz file (excluding metrics)
            np.savez_compressed(
                output_path,
                latent_params=result['latent_params'],
                rotations=result['rotations'],
                scales=result['scales'],
                shifts=result['shifts'],
                classes=result['classes']
            )
            
        except Exception as e:
            print(f"\nError processing sample {idx} ({dataset.json_names[idx]}): {e}")
            failed_samples.append((idx, dataset.json_names[idx], str(e)))
            continue
    
    print("\n" + "="*80)
    print("Processing complete!")
    print(f"Successfully processed: {len(dataset) - len(failed_samples)} samples")
    print(f"Failed: {len(failed_samples)} samples")
    
    # Print reconstruction statistics
    if all_cls_acc:
        print("\n" + "-"*80)
        print("Reconstruction Quality Metrics:")
        print(f"  Average Classification Accuracy: {np.mean(all_cls_acc):.4f} (±{np.std(all_cls_acc):.4f})")
        print(f"  Average Parameters MSE: {np.mean(all_params_mse):.6f} (±{np.std(all_params_mse):.6f})")
        print(f"  Min/Max Classification Accuracy: {np.min(all_cls_acc):.4f} / {np.max(all_cls_acc):.4f}")
        print(f"  Min/Max Parameters MSE: {np.min(all_params_mse):.6f} / {np.max(all_params_mse):.6f}")
        print("-"*80)
    
    if failed_samples:
        print("\nFailed samples:")
        for idx, path, error in failed_samples:
            print(f"  [{idx}] {path}")
            print(f"       Error: {error}")
    
    print(f"\nOutput saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

