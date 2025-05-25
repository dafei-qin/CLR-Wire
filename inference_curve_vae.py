import os
import sys
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset

from src.vae.vae_curve import AutoencoderKL1D
from src.dataset.dataset import CurveDataset
from src.utils.config import NestedDictToClass, load_config
from src.utils.torch_tools import calculate_polyline_lengths, interpolate_1d
from einops import rearrange

def evaluate(model, dataloader, device, sample_points_num, save_file_path=None):
    model.eval()
    total_mse_loss = 0.0
    total_relative_mse_loss = 0.0
    total_weighted_mse_loss = 0.0
    count = 0

    all_reconstructions = []
    all_gt_samples = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            data = batch.to(device) # Corrected: CurveDataset returns tensor directly
            bs = data.shape[0]

            # Prepare query points 't' for the decoder
            # For evaluation, we might want to use the same points as in training, or a fixed set
            t = torch.rand(bs, sample_points_num, device=device)
            t, _ = torch.sort(t, dim=-1)
            
            # model forward pass
            reconstructions = model(data, t=t, return_loss=False).sample # (B, C, N_out)
            
            # Ensure reconstructions and ground truth have the same number of points for comparison
            # Original data is (B, N_in, C), reconstructions are (B, C, N_out)
            # We need to interpolate ground truth to match reconstruction points
            
            # Transpose data to (B, C, N_in) for interpolate_1d
            data_transposed = rearrange(data, "b n c -> b c n")
            gt_samples = interpolate_1d(t, data_transposed) # (B, C, N_out)

            # Calculate MSE
            mse = F.mse_loss(reconstructions, gt_samples, reduction='none').mean(dim=[1, 2]) # (B,)
           
            total_mse_loss += mse.sum().item()

            # Calculate weighted MSE as in training for a more comparable metric
            original_data_for_length = rearrange(data, "b n c -> b n c") # (B, N_in, C)
            batch_lengths = calculate_polyline_lengths(original_data_for_length)
            batch_lengths = torch.clamp(batch_lengths, min=2.0, max=torch.pi * 10)
            relative_mse = mse / (batch_lengths + 1e-6) # Added 1e-6 for stability
            total_relative_mse_loss += relative_mse.sum().item()
            weights = torch.log(batch_lengths + 0.2) # (B,)
            
            weighted_mse = (mse * weights).sum().item()
            total_weighted_mse_loss += weighted_mse
            
            if save_file_path:
                all_reconstructions.append(reconstructions.cpu().numpy())
                all_gt_samples.append(gt_samples.cpu().numpy())
            
            count += bs

    if count == 0: # Avoid division by zero if dataloader is empty
        print("No data to evaluate.")
        return 0.0, 0.0, 0.0
        
    avg_mse_loss = total_mse_loss / count
    avg_weighted_mse_loss = total_weighted_mse_loss / count
    avg_relative_mse_loss = total_relative_mse_loss / count
    print(f"Average MSE Loss: {avg_mse_loss:.8f}")
    print(f"Average Weighted MSE Loss: {avg_weighted_mse_loss:.8f}")
    print(f"Average Relative MSE Loss: {avg_relative_mse_loss:.8f}")

    if save_file_path and all_reconstructions and all_gt_samples:
        all_reconstructions_np = np.concatenate(all_reconstructions, axis=0)
        all_gt_samples_np = np.concatenate(all_gt_samples, axis=0)
        np.savez(save_file_path, reconstructions=all_reconstructions_np, ground_truth=all_gt_samples_np)
        print(f"Saved {all_reconstructions_np.shape[0]} reconstructed and ground truth samples to {save_file_path}")
    elif save_file_path:
        print(f"No samples were processed, nothing to save to {save_file_path}")

    return avg_mse_loss, avg_weighted_mse_loss, avg_relative_mse_loss

def main():
    # Arguments
    program_parser = ArgumentParser(description='Inference script for curve VAE model.')
    program_parser.add_argument('--config', type=str, required=True, help='Path to config file used during training.')
    program_parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint file.')
    program_parser.add_argument('--test_set_path', type=str, required=True, help='Path to the test set file.')
    program_parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference.')
    program_parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading.')
    program_parser.add_argument('--num_eval', type=int, default=None, help='Number of random samples to evaluate from the test set. If None or --save_file is set, evaluate all.')
    program_parser.add_argument('--save_file', type=str, default=None, help='Path to save reconstructed curves (e.g., reconstructions.npz). If set, all samples are evaluated and saved.')
    program_parser.add_argument('--scale', type=float, default=1., help='Scale factor for the input curves.')
    args = program_parser.parse_args()

    # Load config
    cfg_dict = load_config(args.config)
    cfg = NestedDictToClass(cfg_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
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
        sample_points_num=cfg.model.sample_points_num, # This is important
        kl_weight=cfg.model.kl_weight, # Not used in inference directly, but part of model signature
    )
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if 'ema_model' in checkpoint:
        model.load_state_dict(checkpoint['ema_model'])
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model weights.")

    model.to(device)

    # Load test dataset
    full_test_dataset = CurveDataset(
        dataset_file_path=args.test_set_path,
        is_train=False, # Important for dataset specific transforms if any
        transform=None
        # replication factor is usually for training, can be omitted or set to 1
    )
    full_test_dataset.data = full_test_dataset.data * args.scale
    if args.save_file and args.num_eval is None:
        print(f"--save_file specified ({args.save_file}). Evaluating all {len(full_test_dataset)} samples from the test set.")
        eval_dataset = full_test_dataset
    elif args.num_eval is not None and args.num_eval > 0 and args.num_eval < len(full_test_dataset):
        print(f"Randomly selecting {args.num_eval} samples from the test set.")
        indices = np.random.choice(len(full_test_dataset), args.num_eval, replace=False)
        eval_dataset = Subset(full_test_dataset, indices)
    else:
        if args.num_eval is not None and args.num_eval >= len(full_test_dataset):
            print(f"num_eval ({args.num_eval}) is >= dataset size ({len(full_test_dataset)}). Evaluating all samples.")
        elif args.num_eval is not None and args.num_eval <= 0:
            print(f"num_eval ({args.num_eval}) is <= 0. Evaluating all samples.")
        eval_dataset = full_test_dataset
        
    if len(eval_dataset) == 0:
        print("No samples selected for evaluation. Exiting.")
        return

    test_dataloader = torch.utils.data.DataLoader(
        eval_dataset, # Use the (potentially subset) dataset
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for evaluation, especially if it's already a random subset
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Evaluating on {len(eval_dataset)} samples.")

    # Evaluate
    avg_mse_loss, avg_weighted_mse_loss, avg_relative_mse_loss = evaluate(model, test_dataloader, device, cfg.model.sample_points_num, save_file_path=args.save_file)

if __name__ == '__main__':
    # If no arguments are provided (only script name), use default example arguments.
    if len(sys.argv) == 1:
        print("No command-line arguments provided. Using default example arguments.")
        # These are example paths/values. Adjust them if your default files are different.
        default_args = [
            '--config', 'config.yaml',          # Example config file name
            '--checkpoint', 'model.pth',        # Example checkpoint file name
            '--test_set_path', 'test_set.npy', # Example test set file name
            '--batch_size', '128',
            '--num_workers', '8',
            '--num_eval', '100',
            '--save_file', 'reconstructions.npz' # Example output file name
        ]
        sys.argv.extend(default_args)
        print(f"Running with: python inference_curve_vae.py {' '.join(default_args)}")
    
    main() 