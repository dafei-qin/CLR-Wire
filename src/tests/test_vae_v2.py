"""
Test script for VAE model with simplified visualization (no transformations).
Compatible with dataset_v2.py
"""
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
import os
import traceback
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v2 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from src.utils.config import NestedDictToClass, load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.utils.surface_tools import params_to_samples_with_rts
from src.utils.rts_tools import RotationCodebook, TranslationCodebook, ScaleCodebook
from myutils.surface import write_to_step
from myutils.surface import visualize_json_interset
from pathlib import Path

# Colors for is_closed visualization
COLOR_BOTH_CLOSED = [0.2, 0.8, 0.2]      # Green - both u and v closed
COLOR_U_CLOSED = [0.2, 0.2, 0.8]         # Blue - only u closed
COLOR_V_CLOSED = [0.8, 0.2, 0.2]         # Red - only v closed
COLOR_NEITHER_CLOSED = [0.7, 0.7, 0.7]   # Gray - neither closed


def get_closed_color(is_u_closed, is_v_closed):
    """Get color based on is_closed status"""
    if is_u_closed and is_v_closed:
        return COLOR_BOTH_CLOSED
    elif is_u_closed:
        return COLOR_U_CLOSED
    elif is_v_closed:
        return COLOR_V_CLOSED
    else:
        return COLOR_NEITHER_CLOSED


def apply_closed_colors_to_surfaces(surfaces_dict, is_u_closed_list, is_v_closed_list):
    """Apply colors to surfaces based on is_closed status"""
    for i, (surface_key, surface_data) in enumerate(surfaces_dict.items()):
        if i < len(is_u_closed_list) and 'ps_handler' in surface_data:
            color = get_closed_color(is_u_closed_list[i], is_v_closed_list[i])
            try:
                surface_data['ps_handler'].set_color(color)
            except Exception as e:
                print(f"Warning: Could not set color for surface {i}: {e}")


def to_json(params_tensor, types_tensor, mask_tensor):
    """Convert model output tensors to JSON format"""
    json_data = []
    for i in range(len(params_tensor)):
        params = params_tensor[i][mask_tensor[i]]
        recovered_surface = dataset._recover_surface(params, types_tensor[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)
    return json_data


# Global variables
dataset = None
model = None
current_idx = 0
max_idx = 0
pending_idx = 0  # Pending index for "Go To Index" input
current_json_path = ""  # Current JSON file path being processed
gt_group = None
recovered_group = None
pipeline_group = None
gt_surfaces = {}
recovered_surfaces = {}
pipeline_surfaces = {}
resampled_surfaces = {}
show_gt = True
show_recovered = True
show_pipeline = True
show_resampled = False

# Surface sampling control
show_gt_samples = False
show_pred_samples = False
show_pipeline_samples = False
num_samples = 8  # Default 8x8 sampling

# Model configuration
pred_is_closed = False
show_closed_colors = True
use_fsq = False
canonical = False

# Sample point clouds
gt_sample_points = None
pred_sample_points = None
pipeline_sample_points = None

# RTS Codebook configuration
tokenize_rts = False  # Whether to use RTS quantization
use_tokenized_rts = False  # UI toggle for using quantized RTS
rotation_codebook = None
translation_codebook = None
scale_codebook = None
rts_quantization_errors = {}  # Store quantization error statistics

# STEP export configuration
export_folder_path = "./assets/temp/step_exports"  # Default export folder path


def process_sample(idx):
    """Process a single sample and return both GT and recovered data"""
    global dataset, model, pred_is_closed, use_fsq, canonical, num_samples
    global tokenize_rts, rotation_codebook, translation_codebook, scale_codebook, rts_quantization_errors
    global current_json_path
    
    # Load data from dataset
    if pred_is_closed:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale, is_u_closed_tensor, is_v_closed_tensor = dataset[idx]
    else:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[idx]
        is_u_closed_tensor = None
        is_v_closed_tensor = None
    
    print('Processing file:', dataset.json_names[idx])
    json_path = dataset.json_names[idx]
    current_json_path = json_path  # Update global variable
    
    mask_bool = mask_tensor.bool()
    mask_np = mask_bool.cpu().numpy().astype(bool)
    
    # Apply mask to get valid surfaces
    valid_params = params_tensor[mask_bool]
    valid_types = types_tensor[mask_bool]
    shift = shift[mask_np]
    rotation = rotation[mask_np]
    scale = scale[mask_np]
    shift = shift.astype(np.float32)
    rotation = rotation.astype(np.float32)
    scale = scale.astype(np.float32)
    
    # RTS Quantization (if enabled)
    shift_quantized = None
    rotation_quantized = None
    scale_quantized = None
    
    if tokenize_rts and rotation_codebook is not None:
        # Debug: print shapes
        debug_rts = False  # Set to True for debugging
        if debug_rts:
            print(f"  rotation shape: {rotation.shape}")
            print(f"  shift shape: {shift.shape}")
            print(f"  scale shape: {scale.shape}")
        
        # Encode RTS to indices
        rot_indices = rotation_codebook.encode(rotation, batch_size=10000, verbose=False)
        trans_indices = translation_codebook.encode(shift, batch_size=10000, verbose=False)
        scale_indices = scale_codebook.encode(scale, batch_size=10000, verbose=False)
        
        if debug_rts:
            print(f"  rot_indices shape: {rot_indices.shape}")
            print(f"  trans_indices shape: {trans_indices.shape}")
            print(f"  scale_indices shape: {scale_indices.shape}")
        
        # Decode back to get quantized RTS
        rotation_quantized = rotation_codebook.decode(rot_indices)
        shift_quantized = translation_codebook.decode(trans_indices)
        scale_quantized = scale_codebook.decode(scale_indices)
        
        # Compute quantization errors
        rot_errors_deg = []
        for i in range(len(rotation)):
            R_diff = rotation[i].T @ rotation_quantized[i]
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            rot_errors_deg.append(np.rad2deg(angle))
        
        trans_errors = np.linalg.norm(shift - shift_quantized, axis=1)
        
        # Scale errors (scalar)
        scale_errors_abs = np.abs(scale - scale_quantized)
        scale_errors_relative = scale_errors_abs / (np.abs(scale) + 1e-8)
        
        rts_quantization_errors = {
            'rotation_mean_deg': np.mean(rot_errors_deg),
            'rotation_max_deg': np.max(rot_errors_deg),
            'translation_mean': np.mean(trans_errors),
            'translation_max': np.max(trans_errors),
            'scale_abs_mean': np.mean(scale_errors_abs),
            'scale_abs_max': np.max(scale_errors_abs),
            'scale_relative_mean': np.mean(scale_errors_relative),
            'scale_relative_max': np.max(scale_errors_relative),
            'num_surfaces': len(rotation),
        }
        
        print(f"RTS Quantization Errors:")
        print(f"  Rotation: mean={rts_quantization_errors['rotation_mean_deg']:.4f}°, "
              f"max={rts_quantization_errors['rotation_max_deg']:.4f}°")
        print(f"  Translation: mean={rts_quantization_errors['translation_mean']:.6f}, "
              f"max={rts_quantization_errors['translation_max']:.6f}")
        print(f"  Scale (scalar): abs mean={rts_quantization_errors['scale_abs_mean']:.6f}, "
              f"relative mean={rts_quantization_errors['scale_relative_mean']:.4f}")
    
    # Sample GT surfaces (before VAE processing)
    # Choose RTS based on tokenization setting
    global use_tokenized_rts
    rotation_to_use = rotation_quantized if (use_tokenized_rts and rotation_quantized is not None) else rotation
    shift_to_use = shift_quantized if (use_tokenized_rts and shift_quantized is not None) else shift
    scale_to_use = scale_quantized if (use_tokenized_rts and scale_quantized is not None) else scale
    
    gt_sampled_points_list = []
    for i in range(len(valid_params)):
        try:
            samples = params_to_samples_with_rts(
                torch.from_numpy(rotation_to_use[i]).float(), 
                torch.tensor(scale_to_use[i]).float(), 
                torch.from_numpy(shift_to_use[i]).float(), 
                valid_params[i].unsqueeze(0), 
                valid_types[i], 
                num_samples, 
                num_samples
            )  # (1, H, W, 3)
            samples = samples.reshape(-1, 3)  # (H*W, 3)
            gt_sampled_points_list.append(samples)
        except Exception as e:
            print(f"Warning: Failed to sample GT surface {i} (type={valid_types[i].item()}): {e}")
    
    if gt_sampled_points_list:
        gt_all_samples = torch.cat(gt_sampled_points_list, dim=0)  # (N_total, 3)
    else:
        gt_all_samples = None
    
    # Handle is_closed data
    gt_is_u_closed = None
    gt_is_v_closed = None
    if pred_is_closed and is_u_closed_tensor is not None:
        gt_is_u_closed = is_u_closed_tensor[mask_bool].cpu().numpy()
        gt_is_v_closed = is_v_closed_tensor[mask_bool].cpu().numpy()
    
    # Load ground truth JSON data
    with open(json_path, 'r') as f:
        gt_json_data = json.load(f)
    
    # Run VAE/FSQ inference
    pred_is_u_closed = None
    pred_is_v_closed = None
    is_closed_accuracy = None
    
    with torch.no_grad():
        # Encode based on model type
        if use_fsq:
            z_quantized, indices = model.encode(valid_params, valid_types)
            z = z_quantized
            print(f'FSQ indices shape: {indices.shape}, unique codes: {torch.unique(indices.flatten() if indices.ndim > 1 else indices).numel()}')
        else:
            mu, logvar = model.encode(valid_params, valid_types)
            z = model.reparameterize(mu, logvar)
            print('Difference between mu and z:', (mu - z).abs().mean())
        
        # Classify
        if pred_is_closed:
            type_logits_pred, types_pred, is_closed_logits, is_closed_pred = model.classify(z)
            pred_is_u_closed = is_closed_pred[:, 0].cpu().numpy()
            pred_is_v_closed = is_closed_pred[:, 1].cpu().numpy()
            
            # Calculate is_closed accuracy
            if gt_is_u_closed is not None:
                is_closed_gt = torch.stack([torch.from_numpy(gt_is_u_closed), 
                                           torch.from_numpy(gt_is_v_closed)], dim=-1).bool()
                is_closed_correct = (is_closed_pred.cpu() == is_closed_gt).float()
                is_closed_accuracy = is_closed_correct.mean().item()
        else:
            type_logits_pred, types_pred = model.classify(z)
        
        # Decode
        params_pred, mask = model.decode(z, types_pred)

        # use_gt_uvs = True
        # if use_gt_uvs:
        #     params_pred[..., 9:17] = valid_params[..., 9:17]
        #     print('Warning, now use gt uv for reconstruction')
        
        # Calculate metrics
        recon_fn = torch.nn.MSELoss()
        recon_loss = (recon_fn(params_pred, valid_params)) * mask.float().mean()
        accuracy = (types_pred == valid_types).float().mean()
        
        # Sample predicted surfaces (use same RTS as GT for fair comparison)
        pred_sampled_points_list = []
        for i in range(len(params_pred)):
            try:
                samples = params_to_samples_with_rts(
                    torch.from_numpy(rotation_to_use[i]).float(), 
                    torch.tensor(scale_to_use[i]).float(), 
                    torch.from_numpy(shift_to_use[i]).float(), 
                    params_pred[i].unsqueeze(0), 
                    types_pred[i], 
                    num_samples, 
                    num_samples
                )  # (1, H, W, 3)
                samples = samples.reshape(-1, 3)  # (H*W, 3)
                pred_sampled_points_list.append(samples)
            except Exception as e:
                print(f"Warning: Failed to sample predicted surface {i} (type={types_pred[i].item()}): {e}")
        
        if pred_sampled_points_list:
            pred_all_samples = torch.cat(pred_sampled_points_list, dim=0)  # (N_total, 3)
        else:
            pred_all_samples = None
        
        # Calculate sample loss
        sample_loss = None
        if gt_all_samples is not None and pred_all_samples is not None:
            if gt_all_samples.shape[0] == pred_all_samples.shape[0]:
                sample_loss = torch.nn.functional.mse_loss(pred_all_samples, gt_all_samples).item()
            else:
                print(f"Warning: GT and pred sample counts differ: {gt_all_samples.shape[0]} vs {pred_all_samples.shape[0]}")
        
        metrics_str = f'Index {idx}: recon_loss: {recon_loss.item():.6f}, accuracy: {accuracy.item():.4f}'
        if sample_loss is not None:
            metrics_str += f', sample_loss: {sample_loss:.6f}'
        if is_closed_accuracy is not None:
            metrics_str += f', is_closed_accuracy: {is_closed_accuracy:.4f}'
        print(metrics_str)
    
    # Convert predictions to JSON format
    recovered_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
    
    if canonical:
        if use_tokenized_rts:
            recovered_json_data = [from_canonical(recovered_json_data[i], shift_quantized[i], rotation_quantized[i], scale_quantized[i]) 
                              for i in range(len(recovered_json_data))]
        else:
            recovered_json_data = [from_canonical(recovered_json_data[i], shift[i], rotation[i], scale[i]) 
                                for i in range(len(recovered_json_data))]
    
    # Convert dataset pipeline surfaces to JSON
    pipeline_json_data = []
    for i in range(len(valid_params)):
        recovered_surface = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        if canonical:
            if use_tokenized_rts:
                recovered_surface = from_canonical(recovered_surface, shift_quantized[i], rotation_quantized[i], scale_quantized[i])
            else:
                recovered_surface = from_canonical(recovered_surface, shift[i], rotation[i], scale[i])
        pipeline_json_data.append(recovered_surface)
    
    # Sample pipeline surfaces (dataset GT after preprocessing)
    pipeline_sampled_points_list = []
    for i in range(len(valid_params)):
        try:
            samples = params_to_samples_with_rts(
                torch.from_numpy(rotation_to_use[i]).float(), 
                torch.tensor(scale_to_use[i]).float(), 
                torch.from_numpy(shift_to_use[i]).float(), 
                valid_params[i].unsqueeze(0), 
                valid_types[i], 
                num_samples, 
                num_samples
            )  # (1, H, W, 3)
            samples = samples.reshape(-1, 3)  # (H*W, 3)
            pipeline_sampled_points_list.append(samples)
        except Exception as e:
            print(f"Warning: Failed to sample pipeline surface {i} (type={valid_types[i].item()}): {e}")
    
    if pipeline_sampled_points_list:
        pipeline_all_samples = torch.cat(pipeline_sampled_points_list, dim=0)  # (N_total, 3)
    else:
        pipeline_all_samples = None
    
    # Prepare is_closed data
    is_closed_data = {
        'gt_is_u_closed': gt_is_u_closed,
        'gt_is_v_closed': gt_is_v_closed,
        'pred_is_u_closed': pred_is_u_closed,
        'pred_is_v_closed': pred_is_v_closed,
    }
    
    # Prepare sample data
    sample_data = {
        'gt_samples': gt_all_samples,
        'pred_samples': pred_all_samples,
        'pipeline_samples': pipeline_all_samples,
        'sample_loss': sample_loss if 'sample_loss' in locals() else None,
    }
    
    return gt_json_data, recovered_json_data, pipeline_json_data, recon_loss.item(), accuracy.item(), is_closed_data, sample_data


def resample_model():
    """Generate new samples from the VAE's latent space"""
    global model, resampled_surfaces, current_idx, pred_is_closed, show_closed_colors, use_fsq, canonical, num_samples
    global tokenize_rts, rotation_codebook, translation_codebook, scale_codebook, use_tokenized_rts
    
    if model is None:
        print("Model not loaded yet!")
        return
    
    # Load data
    if pred_is_closed:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale, is_u_closed_tensor, is_v_closed_tensor = dataset[current_idx]
    else:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[current_idx]
    
    valid_params = params_tensor[mask_tensor.bool()]
    valid_types = types_tensor[mask_tensor.bool()]
    shift = shift[mask_tensor.bool()]
    rotation = rotation[mask_tensor.bool()]
    scale = scale[mask_tensor.bool()]
    
    # Apply RTS quantization if enabled
    if tokenize_rts and use_tokenized_rts and rotation_codebook is not None:
        rot_indices = rotation_codebook.encode(rotation, batch_size=10000, verbose=False)
        trans_indices = translation_codebook.encode(shift, batch_size=10000, verbose=False)
        scale_indices = scale_codebook.encode(scale, batch_size=10000, verbose=False)
        
        rotation = rotation_codebook.decode(rot_indices)
        shift = translation_codebook.decode(trans_indices)
        scale = scale_codebook.decode(scale_indices)
    
    with torch.no_grad():
        # Encode to get latent vectors
        if use_fsq:
            z_quantized, indices = model.encode(valid_params, valid_types)
            z_random = z_quantized
            print(f'Resampled FSQ indices: unique codes = {torch.unique(indices.flatten() if indices.ndim > 1 else indices).numel()}')
        else:
            mu, logvar = model.encode(valid_params, valid_types)
            z_random = model.reparameterize(mu, logvar)
        
        # Classify
        if pred_is_closed:
            type_logits_pred, types_pred, is_closed_logits, is_closed_pred = model.classify(z_random)
            pred_is_u_closed = is_closed_pred[:, 0].cpu().numpy()
            pred_is_v_closed = is_closed_pred[:, 1].cpu().numpy()
        else:
            type_logits_pred, types_pred = model.classify(z_random)
            pred_is_u_closed = None
            pred_is_v_closed = None
        
        # Decode
        params_pred, mask = model.decode(z_random, types_pred)
        
        # Sample resampled surfaces
        resampled_points_list = []
        for i in range(len(params_pred)):
            try:
                samples = params_to_samples_with_rts(
                    torch.from_numpy(rotation[i]).float(), 
                    torch.tensor(scale[i]).float(), 
                    torch.from_numpy(shift[i]).float(), 
                    params_pred[i].unsqueeze(0), 
                    types_pred[i], 
                    num_samples, 
                    num_samples
                )  # (1, H, W, 3)
                samples = samples.reshape(-1, 3)  # (H*W, 3)
                resampled_points_list.append(samples)
            except Exception as e:
                print(f"Warning: Failed to sample resampled surface {i} (type={types_pred[i].item()}): {e}")
        
        if resampled_points_list:
            resampled_all_samples = torch.cat(resampled_points_list, dim=0)  # (N_total, 3)
        else:
            resampled_all_samples = None
        
        # Convert to JSON format
        resampled_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
        if canonical:
            resampled_json_data = [from_canonical(resampled_json_data[i], shift[i], rotation[i], scale[i]) 
                                  for i in range(len(resampled_json_data))]
        
        with open('./assets/temp/test_vae_v2_runtime_resampled.json', 'w') as f:
            json.dump(resampled_json_data, f)
        
        # Visualize resampled surfaces
        resampled_surfaces = visualize_json_interset(resampled_json_data, plot=True, plot_gui=False, 
                                                     tol=1e-5, ps_header='resampled')
        
        # Add to resampled group
        for i, (surface_key, surface_data) in enumerate(resampled_surfaces.items()):
            if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
                surface_data['ps_handler'].add_to_group(resampled_group)
        
        # Visualize resampled sample points
        if resampled_all_samples is not None and show_resampled:
            resampled_pc = ps.register_point_cloud("Resampled_samples", resampled_all_samples.cpu().numpy())
            resampled_pc.set_radius(0.003)
            resampled_pc.set_color([0.8, 0.8, 0.2])  # Yellow
            resampled_pc.add_to_group(resampled_group)
            print(f"Resampled sampled points: {resampled_all_samples.shape[0]}")
        
        # Apply is_closed colors
        if pred_is_closed and show_closed_colors and pred_is_u_closed is not None:
            apply_closed_colors_to_surfaces(resampled_surfaces, pred_is_u_closed, pred_is_v_closed)
        
        return resampled_json_data, resampled_surfaces


def export_surfaces_to_step():
    """Export current GT, dataset GT, and recovered surfaces to STEP files"""
    global gt_surfaces, pipeline_surfaces, recovered_surfaces, export_folder_path, current_idx, dataset
    
    try:
        # Create export folder if it doesn't exist
        export_path = Path(export_folder_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Get the current file name (without extension)
        json_path = Path(dataset.json_names[current_idx])
        file_stem = json_path.stem
        
        # Export GT surfaces
        gt_output = export_path / f"{file_stem}_gt.step"
        success_gt = write_to_step([_['surface'] for _ in gt_surfaces.values()], gt_output)
        
        # Export dataset GT (pipeline) surfaces
        pipeline_output = export_path / f"{file_stem}_dataset_gt.step"
        success_pipeline = write_to_step([_['surface'] for _ in pipeline_surfaces.values()], pipeline_output)
        
        # Export recovered surfaces
        recovered_output = export_path / f"{file_stem}_reconstructed.step"
        success_recovered = write_to_step([_['surface'] for _ in recovered_surfaces.values()], recovered_output)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"STEP Export Summary (Index {current_idx}):")
        print(f"  GT: {'✓' if success_gt else '✗'} {gt_output}")
        print(f"  Dataset GT: {'✓' if success_pipeline else '✗'} {pipeline_output}")
        print(f"  Reconstructed: {'✓' if success_recovered else '✗'} {recovered_output}")
        print(f"{'='*70}\n")
        
        return success_gt and success_pipeline and success_recovered
        
    except Exception as e:
        print(f"Error during STEP export: {e}")
        traceback.print_exc()
        return False


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, pending_idx, gt_group, recovered_group, pipeline_group
    global gt_surfaces, recovered_surfaces, pipeline_surfaces
    global pred_is_closed, show_closed_colors
    global gt_sample_points, pred_sample_points, pipeline_sample_points
    global show_gt_samples, show_pred_samples, show_pipeline_samples
    
    # Sync pending_idx with current_idx
    pending_idx = current_idx
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    gt_data, recovered_data, pipeline_data, recon_loss, accuracy, is_closed_data, sample_data = process_sample(current_idx)
    
    # Store sample points
    gt_sample_points = sample_data['gt_samples']
    pred_sample_points = sample_data['pred_samples']
    pipeline_sample_points = sample_data['pipeline_samples']
    
    # Visualize ground truth surfaces
    try:
        gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-5)
    except ValueError as e:
        print(f'GT has wrong visualization data: {e}')
        return
    
    for i, (surface_key, surface_data) in enumerate(gt_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(gt_group)
    
    # Apply is_closed colors to GT surfaces
    if pred_is_closed and show_closed_colors and is_closed_data['gt_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(gt_surfaces, 
                                        is_closed_data['gt_is_u_closed'], 
                                        is_closed_data['gt_is_v_closed'])
    
    # Visualize dataset pipeline surfaces
    try:
        pipeline_surfaces = visualize_json_interset(pipeline_data, plot=True, plot_gui=False, 
                                                    tol=1e-5, ps_header='dataset_gt')
    except ValueError as e:
        print(f'Pipeline GT has wrong visualization data: {e}')
        return
    
    for i, (surface_key, surface_data) in enumerate(pipeline_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(pipeline_group)
    
    # Apply is_closed colors to pipeline surfaces
    if pred_is_closed and show_closed_colors and is_closed_data['gt_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(pipeline_surfaces, 
                                        is_closed_data['gt_is_u_closed'], 
                                        is_closed_data['gt_is_v_closed'])
    
    # Visualize recovered surfaces
    try:
        recovered_surfaces = visualize_json_interset(recovered_data, plot=True, plot_gui=False, 
                                                     tol=1e-5, ps_header='z_rec')
    except ValueError as e:
        print(f'Error: {e}, Recovered has wrong visualization data!')
        return
    
    for i, (surface_key, surface_data) in enumerate(recovered_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(recovered_group)
    
    # Apply is_closed colors to recovered surfaces
    if pred_is_closed and show_closed_colors and is_closed_data['pred_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(recovered_surfaces, 
                                        is_closed_data['pred_is_u_closed'], 
                                        is_closed_data['pred_is_v_closed'])
    
    # Visualize sample points as point clouds
    if gt_sample_points is not None and show_gt_samples:
        gt_pc = ps.register_point_cloud("GT_samples", gt_sample_points.cpu().numpy())
        gt_pc.set_radius(0.003)
        gt_pc.set_color([0.2, 0.8, 0.2])  # Green
        gt_pc.add_to_group(gt_group)
    
    if pred_sample_points is not None and show_pred_samples:
        pred_pc = ps.register_point_cloud("Pred_samples", pred_sample_points.cpu().numpy())
        pred_pc.set_radius(0.003)
        pred_pc.set_color([0.8, 0.2, 0.2])  # Red
        pred_pc.add_to_group(recovered_group)
    
    if pipeline_sample_points is not None and show_pipeline_samples:
        pipeline_pc = ps.register_point_cloud("Pipeline_samples", pipeline_sample_points.cpu().numpy())
        pipeline_pc.set_radius(0.003)
        pipeline_pc.set_color([0.2, 0.2, 0.8])  # Blue
        pipeline_pc.add_to_group(pipeline_group)
    
    # Configure group visibility
    gt_group.set_enabled(show_gt)
    pipeline_group.set_enabled(show_pipeline)
    recovered_group.set_enabled(show_recovered)
    
    print(f"Visualized {len(gt_surfaces)} GT surfaces, {len(pipeline_surfaces)} dataset GT surfaces, "
          f"and {len(recovered_surfaces)} recovered surfaces")
    
    # Print sample statistics
    if sample_data['sample_loss'] is not None:
        print(f"Sample loss (pred vs GT): {sample_data['sample_loss']:.6f}")
    if gt_sample_points is not None:
        print(f"GT sampled points: {gt_sample_points.shape[0]}")
    if pred_sample_points is not None:
        print(f"Predicted sampled points: {pred_sample_points.shape[0]}")
    if pipeline_sample_points is not None:
        print(f"Pipeline sampled points: {pipeline_sample_points.shape[0]}")
    
    # Print is_closed statistics
    if pred_is_closed and is_closed_data['gt_is_u_closed'] is not None:
        gt_u_closed_count = sum(is_closed_data['gt_is_u_closed'])
        gt_v_closed_count = sum(is_closed_data['gt_is_v_closed'])
        print(f"GT is_closed: u={gt_u_closed_count}/{len(is_closed_data['gt_is_u_closed'])}, "
              f"v={gt_v_closed_count}/{len(is_closed_data['gt_is_v_closed'])}")
    
    if pred_is_closed and is_closed_data['pred_is_u_closed'] is not None:
        pred_u_closed_count = sum(is_closed_data['pred_is_u_closed'])
        pred_v_closed_count = sum(is_closed_data['pred_is_v_closed'])
        print(f"Pred is_closed: u={pred_u_closed_count}/{len(is_closed_data['pred_is_u_closed'])}, "
              f"v={pred_v_closed_count}/{len(is_closed_data['pred_is_v_closed'])}")


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_gt, show_pipeline, show_recovered, show_resampled
    global resampled_surfaces, pred_is_closed, show_closed_colors, pending_idx
    global show_gt_samples, show_pred_samples, show_pipeline_samples, num_samples
    global tokenize_rts, use_tokenized_rts, rts_quantization_errors
    global export_folder_path
    
    psim.Text("VAE V2 Surface Reconstruction Test")
    psim.Separator()
    
    # Display current JSON path
    if current_json_path:
        psim.TextWrapped(f"Current File: {current_json_path}")
        psim.Separator()
    
    # Index controls
    slider_changed, slider_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    # Go To Index with button
    input_changed, input_idx = psim.InputInt("Go To Index", pending_idx)
    if input_changed:
        pending_idx = max(0, min(max_idx, input_idx))
    
    psim.SameLine()
    if psim.Button("Go"):
        if pending_idx != current_idx:
            current_idx = pending_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Resample button
    psim.Separator()
    psim.Text("Model Controls:")
    if psim.Button("Resample Model"):
        ps.remove_all_structures()
        resampled_surfaces = {}
        resample_model()
    
    # STEP export controls
    psim.Separator()
    psim.Text("=== STEP Export ===")
    changed, export_folder_path = psim.InputText("Export Folder", export_folder_path)
    
    if psim.Button("Export to STEP"):
        export_surfaces_to_step()
    
    psim.TextWrapped(f"Will export: *_gt.step, *_dataset_gt.step, *_reconstructed.step")
    
    # Group visibility controls
    if gt_group is not None:
        psim.Separator()
        psim.Text("Group Controls:")
        changed, show_gt = psim.Checkbox("Show Ground Truth", show_gt)
        if changed:
            gt_group.set_enabled(show_gt)
        
        changed, show_pipeline = psim.Checkbox("Show Dataset GT", show_pipeline)
        if changed:
            pipeline_group.set_enabled(show_pipeline)
        
        changed, show_recovered = psim.Checkbox("Show Recovered", show_recovered)
        if changed:
            recovered_group.set_enabled(show_recovered)
        
        changed, show_resampled = psim.Checkbox("Show Resampled", show_resampled)
        if changed:
            resampled_group.set_enabled(show_resampled)
        
        # Surface sampling controls
        psim.Separator()
        psim.Text("=== Surface Sampling ===")
        
        changed, num_samples_input = psim.InputInt("Samples per edge", num_samples)
        if changed and num_samples_input != num_samples and 2 <= num_samples_input <= 32:
            num_samples = num_samples_input
            update_visualization()
        
        changed, show_gt_samples = psim.Checkbox("Show GT Samples", show_gt_samples)
        if changed:
            update_visualization()
        
        changed, show_pred_samples = psim.Checkbox("Show Predicted Samples", show_pred_samples)
        if changed:
            update_visualization()
        
        changed, show_pipeline_samples = psim.Checkbox("Show Pipeline Samples", show_pipeline_samples)
        if changed:
            update_visualization()
        
        # RTS Tokenization controls
        if tokenize_rts:
            psim.Separator()
            psim.Text("=== RTS Tokenization ===")
            changed, use_tokenized_rts = psim.Checkbox("Use Tokenized RTS", use_tokenized_rts)
            if changed:
                update_visualization()
            
            # Display quantization statistics
            if rts_quantization_errors:
                psim.Text(f"Quantization Errors ({rts_quantization_errors['num_surfaces']} surfaces):")
                psim.Text(f"  Rotation: {rts_quantization_errors['rotation_mean_deg']:.3f}° "
                         f"(max {rts_quantization_errors['rotation_max_deg']:.3f}°)")
                psim.Text(f"  Translation: {rts_quantization_errors['translation_mean']:.5f} "
                         f"(max {rts_quantization_errors['translation_max']:.5f})")
                psim.Text(f"  Scale (rel): {rts_quantization_errors['scale_relative_mean']:.4f} "
                         f"(max {rts_quantization_errors['scale_relative_max']:.4f})")
        
        # is_closed visualization controls
        if pred_is_closed:
            psim.Separator()
            psim.Text("=== is_closed Visualization ===")
            changed, show_closed_colors = psim.Checkbox("Show is_closed Colors", show_closed_colors)
            if changed:
                update_visualization()
            
            # Color legend
            psim.Text("Color Legend:")
            psim.TextColored([COLOR_BOTH_CLOSED[0], COLOR_BOTH_CLOSED[1], COLOR_BOTH_CLOSED[2], 1.0], 
                           "  Green: Both U and V closed")
            psim.TextColored([COLOR_U_CLOSED[0], COLOR_U_CLOSED[1], COLOR_U_CLOSED[2], 1.0], 
                           "  Blue: Only U closed")
            psim.TextColored([COLOR_V_CLOSED[0], COLOR_V_CLOSED[1], COLOR_V_CLOSED[2], 1.0], 
                           "  Red: Only V closed")
            psim.TextColored([COLOR_NEITHER_CLOSED[0], COLOR_NEITHER_CLOSED[1], COLOR_NEITHER_CLOSED[2], 1.0], 
                           "  Gray: Neither closed")


if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--tokenize_rts', action='store_true', help='Enable RTS tokenization')
    parser.add_argument('--codebook_dir', type=str, default='', help='Directory containing RTS codebooks')
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    dataset = load_dataset_from_config(config, section='data_val')
    max_idx = len(dataset) - 1
    model = load_model_from_config(config)
    
    pred_is_closed = config.data_val['params']['detect_closed']
    canonical = config.data_val['params'].get('canonical', False)
    use_fsq = 'fsq' in config.model.name.lower()
    
    print(f"Using FSQ: {use_fsq}")
    print(f"Pred is closed: {pred_is_closed}")
    print(f"Canonical: {canonical}")
    
    # Load RTS codebooks if enabled
    tokenize_rts = args.tokenize_rts
    if tokenize_rts:
        if not args.codebook_dir:
            raise ValueError("--codebook_dir must be specified when --tokenize_rts is enabled")
        
        codebook_dir = Path(args.codebook_dir)
        if not codebook_dir.exists():
            raise FileNotFoundError(f"Codebook directory not found: {codebook_dir}")
        
        print(f"\n{'='*70}")
        print("Loading RTS Codebooks...")
        print(f"{'='*70}")
        
        # Load rotation codebook
        rot_cb_path = codebook_dir / 'cb_rotation.pkl'
        if not rot_cb_path.exists():
            raise FileNotFoundError(f"Rotation codebook not found: {rot_cb_path}")
        rotation_codebook = RotationCodebook(codebook_size=0)  # Size will be loaded
        rotation_codebook.load(rot_cb_path)
        
        # Load translation codebook
        trans_cb_path = codebook_dir / 'cb_translation.pkl'
        if not trans_cb_path.exists():
            raise FileNotFoundError(f"Translation codebook not found: {trans_cb_path}")
        translation_codebook = TranslationCodebook(codebook_size=0)
        translation_codebook.load(trans_cb_path)
        
        # Load scale codebook
        scale_cb_path = codebook_dir / 'cb_scale.pkl'
        if not scale_cb_path.exists():
            raise FileNotFoundError(f"Scale codebook not found: {scale_cb_path}")
        scale_codebook = ScaleCodebook(codebook_size=0)
        scale_codebook.load(scale_cb_path)
        
        print(f"\n✓ RTS Codebooks loaded successfully!")
        print(f"  Rotation: {rotation_codebook.codebook_size} entries")
        print(f"  Translation: {translation_codebook.bins_per_dim} bins per dimension → {translation_codebook.actual_codebook_size} total entries ({translation_codebook.bins_per_dim}³)")
        print(f"  Scale (scalar): {scale_codebook.codebook_size} entries")
        print(f"{'='*70}\n")
    else:
        print("\nRTS tokenization disabled (using continuous RTS values)\n")
    
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
    
    # Initialize polyscope
    ps.init()
    resampled_surfaces = {}
    
    gt_group = ps.create_group("Ground Truth Surfaces")
    pipeline_group = ps.create_group("Dataset GT Surfaces")
    recovered_group = ps.create_group("Recovered Surfaces")
    resampled_group = ps.create_group("Resampled Surfaces")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()

