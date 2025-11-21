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
from icecream import ic
ic.disable()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP
from src.vae.vae_v1 import SurfaceVAE
from src.tools.sample_simple_surface import sample_surface_uniform


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


def compute_bounding_box(points):
    """
    Compute the bounding box of a point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        
    Returns:
        bbox_min: (3,) array of minimum coordinates [x_min, y_min, z_min]
        bbox_max: (3,) array of maximum coordinates [x_max, y_max, z_max]
    """
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    return bbox_min, bbox_max


def sample_and_compute_bbox(params, surface_type_idx, num_u=8, num_v=8):
    """
    Sample a surface and compute its bounding box.
    
    Args:
        params: Surface parameters
        surface_type_idx: Surface type index
        num_u: Number of samples in u direction
        num_v: Number of samples in v direction
        
    Returns:
        bbox_min: (3,) array of minimum coordinates
        bbox_max: (3,) array of maximum coordinates
    """
    try:
        # Sample the surface
        points = sample_surface_uniform(
            params,
            surface_type_idx,
            num_u=num_u,
            num_v=num_v,
            flatten=True
        )
        
        # Compute bounding box
        bbox_min, bbox_max = compute_bounding_box(points)
        return bbox_min, bbox_max
    
    except Exception as e:
        # If sampling fails, return zero bounding box
        print(f"Warning: Failed to sample surface (type {surface_type_idx}): {e}")
        return np.zeros(3), np.zeros(3)


def sort_by_bbox(latent_params, rotations, scales, shifts, classes, bbox_mins, bbox_maxs):
    """
    Sort all data by bounding box minimum coordinates.
    Priority: x > y > z (first sort by x, then by y, then by z)
    
    Args:
        latent_params: (N, latent_dim) array
        rotations: (N, 6) array
        scales: (N, 1) array
        shifts: (N, 3) array
        classes: (N, 1) array
        bbox_mins: (N, 3) array of bbox minimum coordinates
        bbox_maxs: (N, 3) array of bbox maximum coordinates
        
    Returns:
        All arrays sorted by bbox_mins with priority x > y > z
    """
    # Create sorting key: prioritize x, then y, then z
    # Use lexsort: sorts by last key first, so we reverse the order
    sort_indices = np.lexsort((bbox_mins[:, 2], bbox_mins[:, 1], bbox_mins[:, 0]))
    
    # Sort all arrays
    latent_params_sorted = latent_params[sort_indices]
    rotations_sorted = rotations[sort_indices]
    scales_sorted = scales[sort_indices]
    shifts_sorted = shifts[sort_indices]
    classes_sorted = classes[sort_indices]
    bbox_mins_sorted = bbox_mins[sort_indices]
    bbox_maxs_sorted = bbox_maxs[sort_indices]
    
    return (
        latent_params_sorted,
        rotations_sorted,
        scales_sorted,
        shifts_sorted,
        classes_sorted,
        bbox_mins_sorted,
        bbox_maxs_sorted
    )


def process_sample(model, dataset, params_tensor, types_tensor, mask_tensor, 
                   all_shifts, all_rotations, all_scales, 
                   params_tensor_original=None, device='cpu'):
    """
    Process a single sample and extract latent representations.
    Also decode to check reconstruction accuracy.
    Samples each surface with 8x8 grid to compute bounding boxes.
    Sorts all data by bounding box minimum coordinates (priority: x > y > z).
    
    Args:
        model: The VAE model
        dataset: The dataset object (for getting valid param masks)
        params_tensor: Surface parameters (max_num_surfaces, max_param_dim) - for VAE encoding
        types_tensor: Surface types (max_num_surfaces,)
        mask_tensor: Valid surface mask (max_num_surfaces,)
        all_shifts: Shift vectors (max_num_surfaces, 3)
        all_rotations: Rotation matrices (max_num_surfaces, 3, 3)
        all_scales: Scale values (max_num_surfaces,)
        params_tensor_original: Original space parameters for bbox computation (if None, use params_tensor)
        device: Device to run inference on
        
    Returns:
        Dictionary containing (all sorted by bbox):
        - latent_params: (num_valid_surfaces, latent_dim)
        - rotations: (num_valid_surfaces, 6) - first 6 elements of rotation matrix
        - scales: (num_valid_surfaces, 1)
        - shifts: (num_valid_surfaces, 3)
        - classes: (num_valid_surfaces, 1)
        - bbox_mins: (num_valid_surfaces, 3) - minimum coordinates of bounding boxes (in original space)
        - bbox_maxs: (num_valid_surfaces, 3) - maximum coordinates of bounding boxes (in original space)
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
    
    # Convert to numpy arrays (before sorting)
    latent_params_np = latent.cpu().numpy()
    rotations_np = rotations_6d.cpu().numpy()
    scales_np = scales_expanded.cpu().numpy()
    shifts_np = valid_shifts.cpu().numpy()
    classes_np = classes.cpu().numpy()
    
    # Sample surfaces and compute bounding boxes
    # Use original space parameters if provided, otherwise use canonical parameters
    num_valid_surfaces = len(valid_params)
    bbox_mins = np.zeros((num_valid_surfaces, 3), dtype=np.float32)
    bbox_maxs = np.zeros((num_valid_surfaces, 3), dtype=np.float32)
    
    # Determine which parameters to use for bbox computation
    if params_tensor_original is not None:
        # Use original space parameters for bbox computation
        valid_params_for_bbox = params_tensor_original[mask_tensor.bool()]
        bbox_space = "original"
    else:
        # Use the same parameters (canonical or original depending on dataset setting)
        valid_params_for_bbox = valid_params
        bbox_space = "same as encoding"
    
    # print(f"  Sampling {num_valid_surfaces} surfaces for bounding boxes (space: {bbox_space})...")
    for i in range(num_valid_surfaces):
        bbox_min, bbox_max = sample_and_compute_bbox(
            valid_params_for_bbox[i].cpu().numpy() if torch.is_tensor(valid_params_for_bbox[i]) else valid_params_for_bbox[i],
            valid_types[i].item(),
            num_u=8,
            num_v=8
        )
        bbox_mins[i] = bbox_min
        bbox_maxs[i] = bbox_max
    
    # Sort all data by bounding box (priority: x > y > z)
    (
        latent_params_sorted,
        rotations_sorted,
        scales_sorted,
        shifts_sorted,
        classes_sorted,
        bbox_mins_sorted,
        bbox_maxs_sorted
    ) = sort_by_bbox(
        latent_params_np,
        rotations_np,
        scales_np,
        shifts_np,
        classes_np,
        bbox_mins,
        bbox_maxs
    )
    
    # Return sorted data
    return {
        'latent_params': latent_params_sorted,
        'rotations': rotations_sorted,
        'scales': scales_sorted,
        'shifts': shifts_sorted,
        'classes': classes_sorted,
        'bbox_mins': bbox_mins_sorted,
        'bbox_maxs': bbox_maxs_sorted,
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
    
    # If using canonical space, also create a dataset for original space (for bbox computation)
    dataset_original = None
    if args.canonical:
        print("\n" + "="*80)
        print("CANONICAL MODE ENABLED:")
        print("  - Latent encoding: using canonical space (normalized, centered)")
        print("  - Bbox computation: using original space (actual positions)")
        print("  Loading original (non-canonical) dataset for bbox computation...")
        print("="*80 + "\n")
        dataset_original = dataset_compound(args.input_dir, canonical=False)
    else:
        print("\n" + "="*80)
        print("NON-CANONICAL MODE:")
        print("  - Both encoding and bbox computation use original space")
        print("="*80 + "\n")
    
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
            # Get data from dataset (canonical space if args.canonical=True)
            params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales = dataset[idx]
            
            # Check if there are any valid surfaces
            num_valid = mask_tensor.sum().item()
            if num_valid == 0:
                print(f"\nWarning: No valid surfaces in sample {idx} ({dataset.json_names[idx]})")
                continue
            
            # If using canonical space, also load original space parameters for bbox computation
            params_tensor_original = None
            if dataset_original is not None:
                params_tensor_original, _, _, _, _, _ = dataset_original[idx]
            
            # Process sample
            result = process_sample(
                model, dataset, params_tensor, types_tensor, mask_tensor,
                all_shifts, all_rotations, all_scales,
                params_tensor_original=params_tensor_original,
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
            
            # Save to npz file (excluding metrics, including bounding boxes)
            np.savez_compressed(
                output_path,
                latent_params=result['latent_params'],
                rotations=result['rotations'],
                scales=result['scales'],
                shifts=result['shifts'],
                classes=result['classes'],
                bbox_mins=result['bbox_mins'],
                bbox_maxs=result['bbox_maxs']
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

