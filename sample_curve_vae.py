import torch
import numpy as np
from argparse import ArgumentParser
import os

from src.vae.vae_curve import AutoencoderKL1D
from src.utils.config import NestedDictToClass, load_config
from einops import rearrange
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

def calculate_latent_spatial_dim(cfg_model):
    initial_length = cfg_model.sample_points_num
    num_downsamples = 0
    if hasattr(cfg_model, 'block_out_channels') and isinstance(cfg_model.block_out_channels, (list, tuple)) and len(cfg_model.block_out_channels) > 1:
        # Typically, number of downsampling stages is len(block_out_channels) - 1
        # as the last down_block in an encoder often doesn't downsample.
        num_downsamples = len(cfg_model.block_out_channels) - 1
    
    latent_spatial_dim = initial_length // (2**num_downsamples)
    latent_spatial_dim = max(1, latent_spatial_dim) # Ensure it's at least 1
    print(f"Calculated latent spatial dimension: {latent_spatial_dim} (from initial length {initial_length} and {num_downsamples} downsamples)")
    return latent_spatial_dim

def main():
    parser = ArgumentParser(description='Sample curves from a trained VAE model.')
    parser.add_argument('--config', type=str, required=True, help='Path to model config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save generated samples (e.g., generated_samples.npz).')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of curves to generate.')
    parser.add_argument('--input_curve_path', type=str, default=None, help='Optional: Path to a .npy file of a single curve (shape N,C) to sample variations from.')
    parser.add_argument('--sigma_scale', type=float, default=0.5, help='Std deviation scale for sampling around an input curve. Used if --input_curve_path is set.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    if args.sigma_scale <= 0:
        raise ValueError(f"sigma_scale must be positive, but got {args.sigma_scale}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Load config
    cfg_dict = load_config(args.config)
    cfg = NestedDictToClass(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = AutoencoderKL1D(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        down_block_types=cfg.model.down_block_types,
        up_block_types=cfg.model.up_block_types,
        block_out_channels=cfg.model.block_out_channels,
        layers_per_block=cfg.model.layers_per_block,
        act_fn=cfg.model.act_fn,
        latent_channels=cfg.model.latent_channels,
        norm_num_groups=cfg.model.norm_num_groups,
        sample_points_num=cfg.model.sample_points_num,
        kl_weight=cfg.model.kl_weight, # Not used in sampling, but part of signature
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")
    
    model.to(device)
    model.eval()

    latent_spatial_dim = calculate_latent_spatial_dim(cfg.model)
    
    z_to_decode = None

    with torch.no_grad():
        if args.input_curve_path:
            print(f"Sampling {args.num_samples} variations near input curve: {args.input_curve_path}")
            try:
                input_curve_np = np.load(args.input_curve_path)
                if input_curve_np.ndim != 2 or input_curve_np.shape[1] != cfg.model.in_channels:
                    raise ValueError(f"Input curve must be a 2D array of shape (N, C), where C={cfg.model.in_channels}. Got {input_curve_np.shape}")
            except Exception as e:
                print(f"Error loading input curve: {e}")
                return

            input_tensor = torch.from_numpy(input_curve_np).float().unsqueeze(0).to(device) # (1, N, C)
            input_tensor = rearrange(input_tensor, 'b n c -> b c n') # (1, C, N) for encoder

            posterior_orig = model.encode(input_tensor).latent_dist
            original_mean = posterior_orig.mean 
            original_logvar = posterior_orig.logvar

            # Calculate new logvar based on sigma_scale: logvar_new = logvar_orig + 2*log(sigma_scale)
            # Ensure sigma_scale is a tensor on the correct device for torch.log
            sigma_scale_tensor = torch.tensor(args.sigma_scale, device=original_logvar.device, dtype=original_logvar.dtype)
            log_sigma_scale_val = torch.log(sigma_scale_tensor)
            
            scaled_logvar = original_logvar + 2 * log_sigma_scale_val
            
            # Create moments for the new distribution: [mean, logvar] concatenated along channel dim
            moments_for_sampling_dist = torch.cat([original_mean, scaled_logvar], dim=1) 
            
            sampling_distribution = DiagonalGaussianDistribution(moments_for_sampling_dist)

            z_samples_list = []
            for _ in range(args.num_samples):
                z_samples_list.append(sampling_distribution.sample()) 
            z_to_decode = torch.cat(z_samples_list, dim=0)
            
            print(f"Generated {args.num_samples} latent codes by sampling from a modified posterior (sigma_scale={args.sigma_scale}) using its .sample() method.")

        else:
            print(f"Randomly sampling {args.num_samples} new curves.")
            z_to_decode = torch.randn(args.num_samples, cfg.model.latent_channels, latent_spatial_dim, device=device)
            print(f"Generated {args.num_samples} random latent codes.")

        # Prepare query points 't' for the decoder
        # These define where along the normalized length of the curve the VAE reconstructs points
        t_queries = torch.rand(args.num_samples, cfg.model.sample_points_num, device=device)
        t_queries, _ = torch.sort(t_queries, dim=-1) # Sort t values for ordered points along the curve
        
        print(f"Decoding {z_to_decode.shape[0]} latent codes with {cfg.model.sample_points_num} query points each...")
        generated_curves_batch = model.decode(z_to_decode, t_queries).sample # (num_samples, C, N_out)
        
        generated_curves_np = generated_curves_batch.cpu().numpy()

        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        np.savez(args.output_path, samples=generated_curves_np)
        print(f"Saved {generated_curves_np.shape[0]} generated samples to {args.output_path}")
        print(f"Shape of saved samples: {generated_curves_np.shape}")

if __name__ == '__main__':
    main() 