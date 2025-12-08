import sys
import json
import os

from matplotlib.cbook import strip_math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
from typing import Dict, Any, Optional
from types import SimpleNamespace

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf



# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_latent_segment import LatentDataset
from src.dataset.dataset_v1 import dataset_compound
from src.utils.config import NestedDictToClass, load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.flow.surface_flow import ZLDMPipeline, get_new_scheduler
from src.tests.test_vae_v1 import to_json
from src.tools.surface_to_canonical_space import from_canonical
from src.utils.surface_latent_tools import decode_and_sample_with_rts, decode_latent
from utils.surface import visualize_json_interset

# Colors for is_closed visualization
# Format: [R, G, B] in range [0, 1]
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


def register_unit_cube():
    """Register a semi-transparent unit cube for spatial reference."""
    if not _ps_initialized:
        return

    try:
        if hasattr(ps, "has_surface_mesh") and ps.has_surface_mesh(_UNIT_CUBE_NAME):
            return
    except AttributeError:
        pass

    half = 0.5
    cube_vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    cube_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    try:
        cube = ps.register_surface_mesh(
            _UNIT_CUBE_NAME,
            cube_vertices,
            cube_faces,
            color=(0.9, 0.9, 0.9),
            smooth_shade=False,
            transparency=0.2,
        )
        if hasattr(cube, "set_edge_color"):
            cube.set_edge_color((0.2, 0.2, 0.2))
    except Exception as exc:
        print(f"Failed to register unit cube reference: {exc}")


def reset_scene():
    """Clear Polyscope structures while preserving the reference cube."""
    if not _ps_initialized:
        return
    ps.remove_all_structures()
    register_unit_cube()


# Globals for UI
_dataset = None
_dataset_train = None
_dataset_val = None
_use_train_dataset = False  # False = val, True = train
_model = None
_pipe = None
_vae = None
_current_idx = 0
_current_filtered_idx = 0
_max_idx = 0
_gt_group = None
_gen_group = None
_gt_pc_group = None
_gen_pc_group = None
_gt_surfaces = {}
_gen_surfaces = {}
_gt_point_clouds = {}
_gen_point_clouds = {}
_show_gt = True
_show_gen = True
_show_gt_pc = True
_show_gen_pc = True
_filtered_indices = []
_index_metadata = []
_filters = {}
_filter_limits = {
    "num_surfaces": {"min": 0, "max": 0},
}
_ps_initialized = False
_UNIT_CUBE_NAME = "unit_cube_reference"
_rotation_euler = [0.0, 0.0, 0.0]
_apply_rotation = False
_shift_xyz = [0.0, 0.0, 0.0]
_apply_shift = False
_scale_factor = 1.0
_apply_scale = False
_num_inference_steps = 50
# pred_is_closed related variables
_pred_is_closed = False
_show_closed_colors = True  # Toggle for showing is_closed coloring
_vae_model_name = 'vae_v1'  # VAE model name
_dit_model_name = 'dit_v1'
# Custom surface count control
_use_custom_num_surfaces = False  # Whether to use custom surface count
_custom_num_surfaces = 20  # User-specified surface count (15-32)
_default_num_surfaces = 20  # Default value from GT mask
_log_scale = True
_dataset_compound = None

# Surface type mapping
SURFACE_TYPE_MAP_INV = {
    0: 'plane',
    1: 'cylinder',
    2: 'cone',
    3: 'sphere',
    4: 'torus',
    5: 'bspline_surface',
}


def build_index_metadata():
    """Precompute metadata for each dataset sample to enable fast filtering."""
    global _index_metadata, _filter_limits

    if _dataset is None:
        _index_metadata = []
        _filter_limits = {"num_surfaces": {"min": 0, "max": 0}}
        return

    _index_metadata = []
    num_surfaces_values = []

    for idx in range(len(_dataset)):
        try:
            latent_params, rotations, scales, shifts, classes, bbox_mins, bbox_maxs, mask = _dataset[idx]
            num_valid = int(mask.sum().item())
            meta = {
                "valid": num_valid > 0,
                "num_surfaces": num_valid,
            }
            num_surfaces_values.append(num_valid)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            meta = {"valid": False, "num_surfaces": 0}
        
        _index_metadata.append(meta)

    def _limits(values):
        if not values:
            return {"min": 0, "max": 0}
        return {"min": int(min(values)), "max": int(max(values))}

    _filter_limits = {
        "num_surfaces": _limits(num_surfaces_values),
    }


def initialize_filters():
    """Initialize filter defaults using the current dataset statistics."""
    global _filters

    num_limits = _filter_limits.get("num_surfaces", {"min": 0, "max": 0})
    
    _filters = {
        "min_surfaces": num_limits["min"],
        "max_surfaces": num_limits["max"],
    }


def sample_matches_filters(meta: Dict[str, Any]) -> bool:
    """Check whether a metadata entry satisfies the active filters."""
    if not meta.get("valid", False):
        return False

    num_surfaces = meta["num_surfaces"]
    if num_surfaces < _filters.get("min_surfaces", num_surfaces):
        return False
    if num_surfaces > _filters.get("max_surfaces", num_surfaces):
        return False

    return True


def refresh_filtered_indices(preserve_current: bool = True):
    """Rebuild the list of dataset indices that satisfy the active filters."""
    global _filtered_indices, _max_idx, _current_idx, _current_filtered_idx

    if not _index_metadata:
        _filtered_indices = []
        _max_idx = -1
        _current_idx = -1
        _current_filtered_idx = -1
        return

    _filtered_indices = [
        idx for idx, meta in enumerate(_index_metadata) if sample_matches_filters(meta)
    ]
    _max_idx = len(_filtered_indices) - 1

    if not _filtered_indices:
        _current_filtered_idx = -1
        _current_idx = -1
        return

    if preserve_current and _current_idx in _filtered_indices:
        _current_filtered_idx = _filtered_indices.index(_current_idx)
    else:
        _current_filtered_idx = 0

    _current_idx = _filtered_indices[_current_filtered_idx]


def switch_dataset(use_train: bool):
    """Switch between train and val datasets."""
    global _dataset, _dataset_train, _dataset_val, _use_train_dataset, _current_idx
    
    _use_train_dataset = use_train
    
    if _use_train_dataset:
        _dataset = _dataset_train
        print("Switched to TRAIN dataset")
    else:
        _dataset = _dataset_val
        print("Switched to VAL dataset")
    
    # Rebuild metadata and indices for the new dataset
    build_index_metadata()
    initialize_filters()
    refresh_filtered_indices(preserve_current=False)
    
    # Reset current index to 0
    _current_idx = 0 if _filtered_indices else -1
    
    print(f"Dataset has {len(_dataset)} samples, {len(_filtered_indices)} after filtering")


def load_model_and_dataset(
    config_path,
    dit_ckpt_path=None,
    vae_config_path=None,
    vae_checkpoint_path=None,
    device='cuda'
):
    """Load dataset and corresponding model weights using config-based loading."""
    global _dataset, _dataset_train, _dataset_val, _model, _pipe, _max_idx, _vae, _dataset_compound
    global _pred_is_closed, _vae_model_name, _dit_model_name, _log_scale
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Load datasets using config
    _dataset_train = load_dataset_from_config(config, section='data_train')
    _dataset_val = load_dataset_from_config(config, section='data_val')
    
    # Set initial dataset to val
    _dataset = _dataset_val
    
    # Get log_scale from config
    _log_scale = config.data_train.params.get('log_scale', True) if 'data_train' in config else True
    
    # Load VAE config if provided
    if vae_config_path:
        vae_config = OmegaConf.load(vae_config_path)
    elif 'vae' in config and 'config_file' in config.vae:
        vae_config = OmegaConf.load(config.vae.config_file)
    else:
        raise ValueError("VAE config not provided. Use --vae_config or ensure config has vae.config_file")
    
    # Load VAE model
    _vae = load_model_from_config(vae_config, device=device)
    _vae.eval()
    
    # Get VAE model name and pred_is_closed from VAE config
    if 'model' in vae_config:
        model_cfg = vae_config.model
        _vae_model_name = model_cfg.get('name', 'vae_v1')
        if isinstance(_vae_model_name, str) and '.' in _vae_model_name:
            # Extract model name from full path
            _vae_model_name = _vae_model_name.split('.')[-1].lower()
        _pred_is_closed = model_cfg.get('pred_is_closed', False)
    else:
        _vae_model_name = 'vae_v1'
        _pred_is_closed = False
    
    print(f"Loaded VAE: model_name={_vae_model_name}, pred_is_closed={_pred_is_closed}")
    
    # Load dataset_compound for to_json
    _dataset_compound = dataset_compound(json_dir='./', canonical=True, detect_closed=_pred_is_closed)
    
    build_index_metadata()
    initialize_filters()
    refresh_filtered_indices(preserve_current=False)
    
    # Load DiT model
    _dit_model_name = config.model.get('name', 'dit_v1')
    if isinstance(_dit_model_name, str) and '.' in _dit_model_name:
        # Extract model name from full path
        _dit_model_name = _dit_model_name.split('.')[-1]
    
    # Load DiT model using config
    _model = load_model_from_config(config, device=device)
    
    # Override checkpoint if provided
    if dit_ckpt_path:
        checkpoint = torch.load(dit_ckpt_path, map_location=device)
        if 'ema_model' in checkpoint or 'ema' in checkpoint:
            ema_key = 'ema' if 'ema' in checkpoint else 'ema_model'
            ema_model = checkpoint[ema_key]
            ema_model = {k.replace("ema_model.", "").replace("ema.", ""): v for k, v in ema_model.items()}
            _model.load_state_dict(ema_model, strict=False)
            print("Loaded DiT EMA model weights.")
        elif 'model' in checkpoint:
            _model.load_state_dict(checkpoint['model'])
            print("Loaded DiT model weights.")
        else:
            _model.load_state_dict(checkpoint)
            print("Loaded DiT raw model state_dict.")
    
    _model.eval()
    
    # Create pipeline
    prediction_type = config.trainer.get('prediction_type', 'v_prediction')
    num_training_timesteps = config.trainer.get('num_training_timesteps', 1000)
    scheduler = get_new_scheduler(prediction_type, num_training_timesteps)
    _pipe = ZLDMPipeline(_model, scheduler, dtype=torch.float32)
    
    print(f"Loaded DiT model: {_dit_model_name}")


def _compute_loss(output, target, masks):
    loss_raw = torch.nn.functional.mse_loss(output, target, reduction='none')
    loss_others = loss_raw[..., 1:] * masks.float()
    total_valid_surfaces = masks.float().sum()
    loss_shifts = loss_others[..., :3].mean(dim=(2)).sum() / total_valid_surfaces
    loss_rotations = loss_others[..., 3:3+6].mean(dim=(2)).sum() / total_valid_surfaces
    loss_scales = loss_others[..., 3+6:3+6+1].mean(dim=(2)).sum() / total_valid_surfaces
    loss_params = loss_others[..., 3+6+1:].mean(dim=(2)).sum() / total_valid_surfaces
    
    bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    loss_valid = bce_logits_loss(output[..., 0], masks.float().squeeze(-1)).mean()
    
    return loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params


def fix_rts(shifts, rotations, scales):
    shifts = shifts.cpu().numpy()
    rotations = rotations.cpu().numpy()
    scales = scales.cpu().numpy()
    D = rotations[:, :3]
    X = rotations[:, 3:6]
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.cross(D, X)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    X = np.cross(Y, D)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    rotations = np.concatenate([D, X, Y], axis=1).reshape(-1, 3, 3)
    return shifts, rotations, scales


def to_json(params_tensor, types_tensor, mask_tensor):
    global _dataset_compound
    json_data = []
    for i in range(len(params_tensor)):
        params = params_tensor[i][mask_tensor[i]]
        recovered_surface = _dataset_compound._recover_surface(params, types_tensor[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)
    return json_data


def process_index(idx: int, num_inference_steps: int, use_gt_mask=False):
    """Process a single index: load GT, generate point cloud condition, run inference."""
    global _dataset, _pipe, _vae, _pred_is_closed, _use_custom_num_surfaces, _custom_num_surfaces, _log_scale
    
    device = next(_pipe.denoiser.parameters()).device
    
    forward_args = _dataset[idx]
    forward_args = [_.to(device).unsqueeze(0) for _ in forward_args]
    
    params_padded, rotations_padded, scales_padded, shifts_padded, surface_type, bbox_mins, bbox_maxs, masks = forward_args
    
    # Build custom mask if enabled
    if _use_custom_num_surfaces:
        batch_size = masks.shape[0]
        max_num_surfaces = masks.shape[1]
        custom_mask = torch.zeros(batch_size, max_num_surfaces, device=device, dtype=masks.dtype)
        num_surfaces = min(_custom_num_surfaces, max_num_surfaces)
        custom_mask[:, :num_surfaces] = 1
        masks = custom_mask
    
    masks = masks.unsqueeze(-1)
    gt_sample = torch.cat([masks.float(), shifts_padded, rotations_padded, scales_padded, params_padded], dim=-1)
    
    # Generate point cloud condition from GT (same as in trainer)
    with torch.no_grad():
        if masks.dim() == 3:
            masks_2d = masks.squeeze(-1)
        else:
            masks_2d = masks
        masks_bool = masks_2d.bool()
        
        valid_indices = torch.nonzero(masks_bool, as_tuple=False)
        
        if valid_indices.numel() > 0:
            valid_params = params_padded[masks_bool]
            valid_shifts = shifts_padded[masks_bool]
            valid_rotations = rotations_padded[masks_bool]
            valid_scales = scales_padded[masks_bool]
            
            valid_sampled_points = decode_and_sample_with_rts(
                _vae, valid_params, valid_shifts, valid_rotations, valid_scales, log_scale=_log_scale
            )
            valid_sampled_points = valid_sampled_points.reshape(valid_sampled_points.shape[0], -1, 3)
            
            B, num_max_pad = masks_bool.shape
            num_points = valid_sampled_points.shape[1]
            gt_sampled_points = torch.zeros(
                B, num_max_pad, num_points, 3,
                device=valid_sampled_points.device,
                dtype=valid_sampled_points.dtype
            )
            
            batch_indices = valid_indices[:, 0]
            pad_indices = valid_indices[:, 1]
            gt_sampled_points[batch_indices, pad_indices] = valid_sampled_points
        else:
            B, num_max_pad = masks_bool.shape
            gt_sampled_points = torch.zeros(
                B, num_max_pad, 64, 3,
                device=params_padded.device,
                dtype=params_padded.dtype
            )
        
        gt_sampled_points = gt_sampled_points.reshape(B, num_max_pad, -1)
    
    noise = torch.randn_like(gt_sample)
    
    sample = _pipe(noise=noise, pc=gt_sampled_points, num_inference_steps=num_inference_steps, 
                   show_progress=True, tgt_key_padding_mask=~masks.bool().squeeze(-1), 
                   memory_key_padding_mask=~masks.bool().squeeze(-1))
    
    loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = _compute_loss(sample, gt_sample, masks)
    
    # Decode samples
    valid, shifts, rotations, scales, params = decode_latent(sample, log_scale=_log_scale)
    valid_gt, shifts_gt, rotations_gt, scales_gt, params_gt = decode_latent(gt_sample, log_scale=_log_scale)
    
    shifts_gt = shifts_gt[valid_gt].squeeze()
    rotations_gt = rotations_gt[valid_gt].squeeze()
    scales_gt = scales_gt[valid_gt].squeeze()
    params_gt = params_gt[valid_gt].squeeze()
    
    if use_gt_mask:
        valid = masks.squeeze(-1).bool()
    
    shifts = shifts[valid].squeeze()
    rotations = rotations[valid].squeeze()
    scales = scales[valid].squeeze()
    params = params[valid].squeeze()
    
    # Classify with is_closed support
    if _pred_is_closed:
        type_logits_pred, types_pred, is_closed_logits, is_closed_pred = _vae.classify(params)
        pred_is_u_closed = is_closed_pred[:, 0].cpu().numpy()
        pred_is_v_closed = is_closed_pred[:, 1].cpu().numpy()
        
        type_logits_pred_gt, types_pred_gt, is_closed_logits_gt, is_closed_pred_gt = _vae.classify(params_gt)
        gt_is_u_closed = is_closed_pred_gt[:, 0].cpu().numpy()
        gt_is_v_closed = is_closed_pred_gt[:, 1].cpu().numpy()
    else:
        type_logits_pred, types_pred = _vae.classify(params)
        type_logits_pred_gt, types_pred_gt = _vae.classify(params_gt)
        pred_is_u_closed = None
        pred_is_v_closed = None
        gt_is_u_closed = None
        gt_is_v_closed = None
    
    # Decode to get surface parameters for visualization
    params_decoded_gt, mask_gt = _vae.decode(params_gt, types_pred_gt)

    shifts_gt_fixed_np, rotations_fixed_np, scales_gt_fixed_np = fix_rts(shifts_gt, rotations_gt, scales_gt)

    surface_jsons_gt = to_json(params_decoded_gt.cpu().numpy(), types_pred_gt.cpu().numpy(), mask_gt.cpu().numpy())
    surface_jsons_gt = [from_canonical(surface_jsons_gt[i], shifts_gt_fixed_np[i], rotations_fixed_np[i], scales_gt_fixed_np[i]) for i in range(len(surface_jsons_gt))]
    
    params_decoded, mask = _vae.decode(params, types_pred)
    shifts_fixed_np, rotations_fixed_np, scales_fixed_np = fix_rts(shifts, rotations, scales)
    surface_jsons = to_json(params_decoded.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
    surface_jsons = [from_canonical(surface_jsons[i], shifts_fixed_np[i], rotations_fixed_np[i], scales_fixed_np[i]) for i in range(len(surface_jsons))]
    
    # Generate point clouds for visualization
    with torch.no_grad():
        # GT point clouds
        # Becasue the scales are already exp-ed by the decode_latent function
        gt_point_clouds = decode_and_sample_with_rts(_vae, params_gt, shifts_gt, rotations_gt, scales_gt[..., None], log_scale=False)
        gt_point_clouds = gt_point_clouds.cpu().numpy()  # (num_surfaces, H, W, 3)
        
        # Pred point clouds
        pred_point_clouds = decode_and_sample_with_rts(_vae, params, shifts, rotations, scales[..., None], log_scale=False)
        pred_point_clouds = pred_point_clouds.cpu().numpy()  # (num_surfaces, H, W, 3)
    
    loss_samples = np.mean(np.abs((pred_point_clouds**2  - gt_point_clouds**2)))
    print('loss_samples: ', loss_samples)
    # Prepare is_closed data for return
    is_closed_data = {
        'pred_is_u_closed': pred_is_u_closed,
        'pred_is_v_closed': pred_is_v_closed,
        'gt_is_u_closed': gt_is_u_closed,
        'gt_is_v_closed': gt_is_v_closed,
    }
    
    return (surface_jsons, surface_jsons_gt, loss_valid, loss_shifts, loss_rotations, 
            loss_scales, loss_params, is_closed_data, gt_point_clouds, pred_point_clouds)


def update_visualization():
    """Update the visualization with current index"""
    global _current_idx, _gen_group, _gt_group, _gt_pc_group, _gen_pc_group, _num_inference_steps
    global _gt_surfaces, _gen_surfaces, _gt_point_clouds, _gen_point_clouds
    global _show_gt, _show_gen, _show_gt_pc, _show_gen_pc
    global _pred_is_closed, _show_closed_colors, _default_num_surfaces, _dataset
    global _use_custom_num_surfaces, _custom_num_surfaces
    
    if not _ps_initialized:
        return
    
    # Update default num_surfaces from GT mask
    if _dataset is not None:
        try:
            forward_args = _dataset[_current_idx]
            masks = forward_args[7]  # masks is the 8th element (0-indexed)
            gt_mask_count = int(masks.sum().item())
            _default_num_surfaces = gt_mask_count
            if not _use_custom_num_surfaces:
                _custom_num_surfaces = _default_num_surfaces
        except Exception as e:
            print(f"Warning: Could not get GT mask count: {e}")
    
    # Clear existing structures
    reset_scene()
    
    # Process current sample
    print(f"\nProcessing index {_current_idx}...")
    with torch.no_grad():
        result = process_index(_current_idx, _num_inference_steps, use_gt_mask=True)
        (surface_jsons, surface_jsons_gt, loss_valid, loss_shifts, loss_rotations, 
         loss_scales, loss_params, is_closed_data, gt_point_clouds, pred_point_clouds) = result
    
    # Display losses
    print(f"Losses - Valid: {loss_valid.item():.6f}, Shifts: {loss_shifts.item():.6f}, "
          f"Rotations: {loss_rotations.item():.6f}, Scales: {loss_scales.item():.6f}, "
          f"Params: {loss_params.item():.6f}")
    
    print(f"Visualizing {len(surface_jsons)} generated surfaces and {len(surface_jsons_gt)} GT surfaces")
    
    # Visualize generated surfaces (OCC)
    _gen_surfaces = visualize_json_interset(surface_jsons, plot=True, plot_gui=False, tol=1e-5, ps_header=f'sample_{_current_idx}')
    for surface_key, surface_data in _gen_surfaces.items():
        if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
            surface_data['ps_handler'].add_to_group(_gen_group)
    
    # Apply is_closed colors to generated surfaces
    if _pred_is_closed and _show_closed_colors and is_closed_data and is_closed_data['pred_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(_gen_surfaces, 
                                        is_closed_data['pred_is_u_closed'], 
                                        is_closed_data['pred_is_v_closed'])
    
    # Visualize GT surfaces (OCC)
    _gt_surfaces = visualize_json_interset(surface_jsons_gt, plot=True, plot_gui=False, tol=1e-5, ps_header=f'gt_{_current_idx}')
    for surface_key, surface_data in _gt_surfaces.items():
        if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
            surface_data['ps_handler'].add_to_group(_gt_group)
    
    # Apply is_closed colors to GT surfaces
    if _pred_is_closed and _show_closed_colors and is_closed_data and is_closed_data['gt_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(_gt_surfaces, 
                                        is_closed_data['gt_is_u_closed'], 
                                        is_closed_data['gt_is_v_closed'])
    
    # Visualize generated point clouds
    _gen_point_clouds = {}
    for i, pc in enumerate(pred_point_clouds):
        pc_flat = pc.reshape(-1, 3)  # (H*W, 3)
        pc_name = f"pred_pc_{_current_idx}_{i}"
        try:
            pc_handler = ps.register_point_cloud(pc_name, pc_flat, radius=0.003)
            pc_handler.set_color([0.8, 0.2, 0.2])  # Red for predictions
            pc_handler.add_to_group(_gen_pc_group)
            _gen_point_clouds[pc_name] = pc_handler
        except Exception as e:
            print(f"Warning: Could not visualize pred point cloud {i}: {e}")
    
    # Visualize GT point clouds
    _gt_point_clouds = {}
    for i, pc in enumerate(gt_point_clouds):
        pc_flat = pc.reshape(-1, 3)  # (H*W, 3)
        pc_name = f"gt_pc_{_current_idx}_{i}"
        try:
            pc_handler = ps.register_point_cloud(pc_name, pc_flat, radius=0.003)
            pc_handler.set_color([0.2, 0.8, 0.2])  # Green for GT
            pc_handler.add_to_group(_gt_pc_group)
            _gt_point_clouds[pc_name] = pc_handler
        except Exception as e:
            print(f"Warning: Could not visualize GT point cloud {i}: {e}")
    
    # Configure groups with current visibility settings
    _gen_group.set_enabled(_show_gen)
    _gt_group.set_enabled(_show_gt)
    _gen_pc_group.set_enabled(_show_gen_pc)
    _gt_pc_group.set_enabled(_show_gt_pc)
    
    print(f"Visualized {len(_gen_surfaces)} generated surfaces, {len(_gt_surfaces)} GT surfaces")
    print(f"Visualized {len(_gen_point_clouds)} generated point clouds, {len(_gt_point_clouds)} GT point clouds")
    
    # Print is_closed statistics if enabled
    if _pred_is_closed and is_closed_data:
        if is_closed_data['pred_is_u_closed'] is not None:
            pred_u_closed_count = sum(is_closed_data['pred_is_u_closed'])
            pred_v_closed_count = sum(is_closed_data['pred_is_v_closed'])
            print(f"Pred is_closed: u={pred_u_closed_count}/{len(is_closed_data['pred_is_u_closed'])}, v={pred_v_closed_count}/{len(is_closed_data['pred_is_v_closed'])}")
        if is_closed_data['gt_is_u_closed'] is not None:
            gt_u_closed_count = sum(is_closed_data['gt_is_u_closed'])
            gt_v_closed_count = sum(is_closed_data['gt_is_v_closed'])
            print(f"GT is_closed: u={gt_u_closed_count}/{len(is_closed_data['gt_is_u_closed'])}, v={gt_v_closed_count}/{len(is_closed_data['gt_is_v_closed'])}")


def callback():
    """Polyscope callback function for UI controls"""
    global _current_idx, _max_idx, _num_inference_steps
    global _show_gt, _show_gen, _show_gt_pc, _show_gen_pc
    global _gt_group, _gen_group, _gt_pc_group, _gen_pc_group
    global _use_train_dataset, _dataset_train, _dataset_val
    global _show_closed_colors, _use_custom_num_surfaces, _custom_num_surfaces, _default_num_surfaces
    
    psim.Text("DiT Simple Surface Segment Inference")
    psim.Separator()
    
    # Dataset switcher
    psim.Text("Dataset Selection:")
    if _dataset_train is not None and _dataset_val is not None:
        dataset_options = ["Validation", "Train"]
        current_selection = 1 if _use_train_dataset else 0
        
        changed, new_selection = psim.Combo("Dataset", current_selection, dataset_options)
        if changed and new_selection != current_selection:
            use_train = (new_selection == 1)
            switch_dataset(use_train)
            update_visualization()
        
        dataset_name = "TRAIN" if _use_train_dataset else "VAL"
        dataset_size = len(_dataset_train) if _use_train_dataset else len(_dataset_val)
        psim.Text(f"Current: {dataset_name} ({dataset_size} samples)")
    
    psim.Separator()
    
    # Index controls
    psim.Text("Sample Navigation:")
    slider_changed, slider_idx = psim.SliderInt("Test Index", _current_idx, 0, _max_idx)
    if slider_changed and slider_idx != _current_idx:
        _current_idx = slider_idx
        update_visualization()
    
    input_changed, input_idx = psim.InputInt("Go To Index", _current_idx)
    if input_changed:
        input_idx = max(0, min(_max_idx, input_idx))
        if input_idx != _current_idx:
            _current_idx = input_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {_current_idx}")
    psim.Text(f"Max Index: {_max_idx}")
    
    # Inference steps control
    psim.Separator()
    psim.Text("Inference Settings:")
    steps_changed, new_steps = psim.SliderInt("Inference Steps", _num_inference_steps, 1, 1000)
    if steps_changed:
        _num_inference_steps = new_steps
    
    if psim.Button("Regenerate"):
        update_visualization()
    
    psim.Separator()
    psim.Text(f"Inference Steps: {_num_inference_steps}")
    
    # Surface count control
    psim.Separator()
    psim.Text("Surface Count Control:")
    psim.Text(f"Default (GT): {_default_num_surfaces}")
    
    changed_use_custom, _use_custom_num_surfaces = psim.Checkbox("Use Custom Surface Count", _use_custom_num_surfaces)
    
    if _use_custom_num_surfaces:
        input_changed, new_count = psim.InputInt("Surface Count (15-32)", _custom_num_surfaces)
        if input_changed:
            _custom_num_surfaces = max(15, min(32, new_count))
        if input_changed or changed_use_custom:
            update_visualization()
    else:
        if changed_use_custom:
            _custom_num_surfaces = _default_num_surfaces
            update_visualization()
    
    # Group visibility controls
    if _gt_group is not None and _gen_group is not None:
        psim.Separator()
        psim.Text("Visibility Controls:")
        changed_gt, _show_gt = psim.Checkbox("Show Ground Truth Surfaces", _show_gt)
        if changed_gt:
            _gt_group.set_enabled(_show_gt)
        
        changed_gen, _show_gen = psim.Checkbox("Show Generated Surfaces", _show_gen)
        if changed_gen:
            _gen_group.set_enabled(_show_gen)
        
        changed_gt_pc, _show_gt_pc = psim.Checkbox("Show GT Point Clouds", _show_gt_pc)
        if changed_gt_pc:
            _gt_pc_group.set_enabled(_show_gt_pc)
        
        changed_gen_pc, _show_gen_pc = psim.Checkbox("Show Generated Point Clouds", _show_gen_pc)
        if changed_gen_pc:
            _gen_pc_group.set_enabled(_show_gen_pc)
        
        # is_closed color controls
        if _pred_is_closed:
            psim.Separator()
            psim.Text("=== is_closed Visualization ===")
            changed, _show_closed_colors = psim.Checkbox("Show is_closed Colors", _show_closed_colors)
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/dit_segment/1205_init.yaml',
        help='Path to the YAML config file.',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default='',
        help='Path to the DiT model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--vae_config',
        type=str,
        default='',
        help='Path to the VAE YAML config file (overrides config.vae.config_file)'
    )
    parser.add_argument(
        '--vae_ckpt',
        type=str,
        default='',
        help='Path to the VAE checkpoint (overrides config, usually not needed if config has it)'
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda'
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help='Starting index for visualization'
    )
    args = parser.parse_args()

    _num_inference_steps = args.num_inference_steps
    _current_idx = args.start_idx

    load_model_and_dataset(
        config_path=args.config,
        dit_ckpt_path=args.ckpt_path if args.ckpt_path else None,
        vae_config_path=args.vae_config if args.vae_config else None,
        vae_checkpoint_path=args.vae_ckpt if args.vae_ckpt else None,
        device=args.device
    )
    
    # Update max index based on filtered indices
    if _filtered_indices:
        _max_idx = len(_filtered_indices) - 1
        _current_idx = min(_current_idx, _max_idx)
    else:
        _max_idx = len(_dataset) - 1 if _dataset else 0
        _current_idx = min(_current_idx, _max_idx)
    
    # Initialize polyscope
    ps.init()
    _ps_initialized = True
    
    # Create groups
    _gt_group = ps.create_group("Ground Truth Surfaces")
    _gen_group = ps.create_group("Generated Surfaces")
    _gt_pc_group = ps.create_group("Ground Truth Point Clouds")
    _gen_pc_group = ps.create_group("Generated Point Clouds")
    
    # Register reference cube
    register_unit_cube()
    
    # Set user callback
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()
