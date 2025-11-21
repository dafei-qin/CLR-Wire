"""
Simple non-interactive test for LatentDataset.

This script loads a single sample and prints statistics without GUI.
Useful for quick testing and debugging.
"""

import torch
import numpy as np
import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_latent import LatentDataset
from src.vae.vae_v1 import SurfaceVAE
from src.tools.surface_to_canonical_space import from_canonical


def load_vae_model(checkpoint_path, device='cpu'):
    """Load the VAE model from checkpoint"""
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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


def main():
    parser = argparse.ArgumentParser(description='Simple test for LatentDataset')
    parser.add_argument('npz_dir', type=str, help='Directory containing NPZ files')
    parser.add_argument('checkpoint_path', type=str, help='Path to VAE checkpoint')
    parser.add_argument('--index', type=int, default=0, help='Sample index to test')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    
    args = parser.parse_args()
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print("="*80)
    print("LATENT DATASET SIMPLE TEST")
    print("="*80)
    
    # Load dataset
    print(f"\n1. Loading latent dataset from: {args.npz_dir}")
    dataset = LatentDataset(
        npz_dir=args.npz_dir,
        max_num_surfaces=500,
        latent_dim=args.latent_dim
    )
    print(f"   Dataset size: {len(dataset)} samples")
    
    # Load VAE model
    print(f"\n2. Loading VAE model from: {args.checkpoint_path}")
    model = load_vae_model(args.checkpoint_path, device=device)
    
    # Create dataset helper
    print("\n3. Creating dataset helper...")
    from src.dataset.dataset_v1 import dataset_compound, SURFACE_PARAM_SCHEMAS, build_surface_postpreprocess
    dataset_helper = object.__new__(dataset_compound)
    dataset_helper.postprocess_funcs = {
        k: build_surface_postpreprocess(v) 
        for k, v in SURFACE_PARAM_SCHEMAS.items()
    }
    
    # Load sample
    print(f"\n4. Loading sample {args.index}...")
    (latent_params, rotations, scales, shifts, classes, 
     bbox_mins, bbox_maxs, mask) = dataset[args.index]
    
    valid_mask = mask.bool()
    num_valid = valid_mask.sum().item()
    
    print(f"   Valid surfaces: {num_valid}")
    print(f"   Latent params shape: {latent_params[valid_mask].shape}")
    print(f"   Surface types: {classes[valid_mask].numpy()}")
    
    # Show bbox sorting
    print(f"\n5. Bounding box sorting verification:")
    bbox_mins_valid = bbox_mins[valid_mask].numpy()
    bbox_maxs_valid = bbox_maxs[valid_mask].numpy()
    
    print(f"   First 5 bbox_mins:")
    for i in range(min(5, num_valid)):
        print(f"     [{i}] {bbox_mins_valid[i]}")
    
    print(f"   Last 5 bbox_mins:")
    for i in range(max(0, num_valid - 5), num_valid):
        print(f"     [{i}] {bbox_mins_valid[i]}")
    
    # Check sorting
    is_sorted = True
    for i in range(1, num_valid):
        prev = bbox_mins_valid[i-1]
        curr = bbox_mins_valid[i]
        # Check lexicographic order: x > y > z
        if prev[0] > curr[0] + 1e-6:
            is_sorted = False
            print(f"   WARNING: Not sorted at index {i-1} -> {i}")
        elif abs(prev[0] - curr[0]) < 1e-6 and prev[1] > curr[1] + 1e-6:
            is_sorted = False
            print(f"   WARNING: Not sorted at index {i-1} -> {i}")
        elif abs(prev[0] - curr[0]) < 1e-6 and abs(prev[1] - curr[1]) < 1e-6 and prev[2] > curr[2] + 1e-6:
            is_sorted = False
            print(f"   WARNING: Not sorted at index {i-1} -> {i}")
    
    if is_sorted:
        print(f"   ✓ Surfaces are correctly sorted by bbox (X > Y > Z)")
    
    # Decode using VAE
    print(f"\n6. Decoding latent representations...")
    latent_params_valid = latent_params[valid_mask].to(device)
    classes_valid = classes[valid_mask].to(device)
    
    with torch.no_grad():
        params_decoded, decode_mask = model.decode(latent_params_valid, classes_valid)
    
    print(f"   Decoded params shape: {params_decoded.shape}")
    print(f"   Decode mask sum per surface: {decode_mask.sum(dim=1).cpu().numpy()[:5]}")
    
    # Recover to original space
    print(f"\n7. Recovering to original space...")
    
    params_decoded_np = params_decoded.cpu().numpy()
    classes_np = classes_valid.cpu().numpy()
    rotations_valid = rotations[valid_mask].numpy()
    scales_valid = scales[valid_mask].numpy()
    shifts_valid = shifts[valid_mask].numpy()
    
    recovered_surfaces = []
    for i in range(min(3, num_valid)):  # Test first 3 surfaces
        # Recover canonical space surface
        surface_canonical = dataset_helper._recover_surface(
            params_decoded_np[i],
            classes_np[i]
        )
        
        # Reconstruct rotation matrix
        rotation_6d = rotations_valid[i]
        row1 = rotation_6d[:3]
        row2 = rotation_6d[3:6]
        row3 = np.cross(row1, row2)
        rotation_matrix = np.array([row1, row2, row3], dtype=np.float64)
        
        shift = shifts_valid[i]
        scale = scales_valid[i, 0]
        
        # Transform back
        surface_original = from_canonical(
            surface_canonical,
            shift,
            rotation_matrix,
            scale
        )
        
        recovered_surfaces.append(surface_original)
        
        print(f"   Surface {i}:")
        print(f"     Type: {surface_original['type']}")
        print(f"     Location: {surface_original['location'][0][:3]}")
        print(f"     Bbox: {bbox_mins_valid[i]} to {bbox_maxs_valid[i]}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nSummary:")
    print(f"  - Loaded {num_valid} surfaces")
    print(f"  - Decoded latent representations")
    print(f"  - Recovered to original space")
    print(f"  - Verified bbox sorting: {'✓ PASS' if is_sorted else '✗ FAIL'}")
    print(f"\nTo visualize interactively, run:")
    print(f"  python src/tests/test_dataset_latent.py {args.npz_dir} {args.checkpoint_path} --index {args.index}")


if __name__ == '__main__':
    main()


