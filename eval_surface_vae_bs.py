#!/usr/bin/env python3
"""
Evaluation script for B-spline enhanced surface VAE model.
Usage: python eval_surface_vae_bs.py --config configs/train_surface_vae_bs.yaml --checkpoint checkpoints/surface_vae_bs/model-99.pt
"""

import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from einops import rearrange
from collections import defaultdict
from src.vae.vae_surface import AutoencoderKLBS2D
from src.dataset.dataset import BSplineSurfaceDataset
from src.utils.config import NestedDictToClass, load_config
from tqdm import tqdm

def evaluate_model(model, dataloader, device, num_samples=None):
    """Evaluate model on dataset and compute reconstruction metrics."""
    model.eval()
    
    loss_dict_sum = defaultdict(float)
    num_batches = 0
    
    print(f"Evaluating on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if num_samples and i >= num_samples:
                break
                
            # Prepare data - ensure float32 dtype
            data = batch['data'].to(device, dtype=torch.float32)  # Surface points
            control_points = batch['control_points'].to(device, dtype=torch.float32)  # B-spline control points
            
            # Generate grid coordinates for decoding
            bs, h, w, c = data.shape
            t_1d = torch.linspace(0, 1, h, device=device, dtype=torch.float32)
            t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
            t = t_grid.unsqueeze(0).repeat(bs, 1, 1, 1)
            
            # Forward pass
            # data_input = rearrange(data, "b h w c -> b c h w")
            total_loss, loss_dict = model(
                data=data,
                control_points=control_points,
                t=t,
                return_loss=True,
                sample_posterior=False
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                loss_dict_sum[key] += value.item()

            loss_dict_sum['total_loss'] += total_loss.item()
            num_batches += 1
            # print(loss_dict_sum)
            # if i % 50 == 0:
            #     print(f"  Batch {i}/{len(dataloader)}")

    # Compute average metrics
    avg_loss = {key: value / num_batches for key, value in loss_dict_sum.items()}
    return avg_loss


def main():
    parser = ArgumentParser(description='Evaluate B-spline surface VAE model.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of batches to evaluate (None for all).')
    parser.add_argument('--use_val_data', action='store_true', help='Use validation data instead of test data.')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    config = NestedDictToClass(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = AutoencoderKLBS2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        down_block_types=config.model.down_block_types,
        up_block_types=config.model.up_block_types,
        block_out_channels=config.model.block_out_channels,
        layers_per_block=config.model.layers_per_block,
        act_fn=config.model.act_fn,
        latent_channels=config.model.latent_channels,
        norm_num_groups=config.model.norm_num_groups,
        sample_points_num=config.model.sample_points_num,
        kl_weight=config.model.kl_weight,
        bspline_resolution=getattr(config.model, 'sample_points_num', 32),
        bspline_cp_weight=getattr(config.model, 'bspline_cp_weight', 1.0),
        bspline_surface_weight=getattr(config.model, 'bspline_surface_weight', 1.0),
        mlp_hidden_dim=getattr(config.model, 'mlp_hidden_dim', 256),
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
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
    # Ensure model uses float32
    model = model.float()
    print("Model loaded successfully.")
    
    # Prepare dataset
    if args.use_val_data:
        bs_path = config.data.val_bs_path
        points_path = config.data.val_points_path
        print("Using validation dataset.")
    else:
        # Default to validation if test paths not specified
        bs_path = getattr(config.data, 'test_bs_path', config.data.val_bs_path)
        points_path = getattr(config.data, 'test_points_path', config.data.val_points_path)
        print("Using test/validation dataset.")
    
    dataset = BSplineSurfaceDataset(
        bs_path=bs_path,
        points_path=points_path,
        is_train=False,
        num_samples=config.model.sample_points_num,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    metrics = evaluate_model(model, dataloader, device, args.num_samples)
    
    # Print results
    print("\nEvaluation Results:")
    print("==================")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

if __name__ == "__main__":
    main() 