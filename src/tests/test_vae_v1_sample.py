"""
Simple script to sample from VAE latent space and visualize reconstructed surfaces.
Randomly samples z from Normal(0, 1) and reconstructs surfaces using the VAE.
"""
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json

sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound
from src.vae.vae_v1 import SurfaceVAE
from myutils.surface import visualize_json_interset


def to_json(params_tensor, types_tensor, mask_tensor, dataset):
    """Convert model outputs to JSON format"""
    json_data = []
    for i in range(len(params_tensor)):
        params = params_tensor[i][mask_tensor[i]]
        recovered_surface = dataset._recover_surface(params, types_tensor[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)
    return json_data


def load_model(checkpoint_path):
    """Load VAE model from checkpoint"""
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
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
    
    model.eval()
    return model


def sample_and_reconstruct(model, dataset, num_samples=5, latent_dim=256):
    """
    Sample random z from Normal(0, 1) and reconstruct surfaces.
    
    Args:
        model: VAE model
        dataset: Dataset object (used for surface recovery)
        num_samples: Number of surfaces to generate
        latent_dim: Dimension of latent space
    
    Returns:
        json_data: List of reconstructed surfaces in JSON format
    """
    with torch.no_grad():
        # Sample random latent vectors from Normal(0, 1)
        z_random = torch.randn(num_samples, latent_dim)
        print(f"Sampled z with shape: {z_random.shape}")
        
        # Classify the latent vectors to get surface types
        type_logits_pred, types_pred = model.classify(z_random)
        print(f"Predicted types: {types_pred.cpu().numpy()}")
        
        # Decode to get surface parameters
        params_pred, mask = model.decode(z_random, types_pred)
        print(f"Decoded params with shape: {params_pred.shape}")
        
        # Convert to JSON format
        json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), 
                           mask.cpu().numpy(), dataset)
        
        return json_data


def visualize_samples(json_data):
    """Visualize reconstructed surfaces using polyscope"""
    surfaces = visualize_json_interset(json_data, plot=True, plot_gui=False, 
                                      tol=1e-5, ps_header='sampled')
    print(f"Visualized {len(surfaces)} surfaces")
    return surfaces


# Global variables for interactive UI
model = None
dataset = None
surfaces = {}
num_samples = 5
latent_dim = 256


def callback():
    """Polyscope callback for interactive controls"""
    global num_samples, latent_dim, surfaces
    
    psim.Text("VAE Random Sampling")
    psim.Separator()
    
    # Number of samples control
    changed_samples, new_samples = psim.SliderInt("Num Samples", num_samples, 1, 20)
    if changed_samples:
        num_samples = new_samples
    
    psim.Text(f"Latent Dim: {latent_dim}")
    psim.Separator()
    
    # Resample button
    if psim.Button("Generate New Samples"):
        # Clear existing structures
        ps.remove_all_structures()
        surfaces = {}
        
        # Sample and visualize
        print(f"\nGenerating {num_samples} new samples...")
        json_data = sample_and_reconstruct(model, dataset, num_samples, latent_dim)
        
        # Save to file
        with open('./assets/temp/test_vae_v1_sample.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        print("Saved to ./assets/temp/test_vae_v1_sample.json")
        
        # Visualize
        surfaces = visualize_samples(json_data)
    
    psim.Separator()
    psim.Text(f"Current surfaces: {len(surfaces)}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sample from VAE latent space and visualize')
    parser.add_argument('--dataset_path', type=str, required=True, 
                       help='Path to the dataset (used for surface recovery)')
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                       help='Path to the VAE checkpoint')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of surfaces to generate')
    parser.add_argument('--latent_dim', type=int, default=256, 
                       help='Dimension of latent space')
    parser.add_argument('--canonical', type=bool, default=False, 
                       help='Whether to use canonical dataset')
    
    args = parser.parse_args()
    
    # Update global variables
    num_samples = args.num_samples
    latent_dim = args.latent_dim
    
    # Load dataset (needed for surface recovery)
    print("Loading dataset...")
    dataset = dataset_compound(args.dataset_path, canonical=args.canonical)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Load model
    print("Loading VAE model...")
    model = load_model(args.checkpoint_path)
    
    # Initialize polyscope
    ps.init()
    ps.set_user_callback(callback)
    
    # Generate initial samples
    print(f"\nGenerating {num_samples} initial samples...")
    json_data = sample_and_reconstruct(model, dataset, num_samples, latent_dim)
    
    # Save to file
    with open('./assets/temp/test_vae_v1_sample.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print("Saved to ./assets/temp/test_vae_v1_sample.json")
    
    # Visualize
    surfaces = visualize_samples(json_data)
    
    # Show interface
    print("\nStarting interactive viewer...")
    ps.show()


