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
from myutils.surface import visualize_json_interset

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
_use_ddim = False  # Whether to use DDIM scheduler
_ddim_eta = 0.0    # DDIM eta parameter
_config_path = None  # Store config path for scheduler switching

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

        forward_args = _dataset[idx]
        
        # Handle different dataset formats

        sampled_points, shifts, rotations, scales, params, surface_type, masks, masks_valid = forward_args
        
        
        # num_valid = int(masks.sum().item())
        meta = {
            "valid": masks_valid,
            "num_surfaces": masks.sum().item(),
        }
        num_surfaces_values.append(masks.sum().item())

        
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


def is_valid_sample(idx: int) -> bool:
    """Check if a sample is valid by checking its valid_mask."""
    global _dataset
    
    if _dataset is None or idx < 0 or idx >= len(_dataset):
        return False
    
    try:
        forward_args = _dataset[idx]
        valid_mask = forward_args[-1]
        return bool(valid_mask)
    except Exception as e:
        print(f"Error checking validity of index {idx}: {e}")
        return False


def find_next_valid_index(start_idx: int, direction: int = 1) -> int:
    """Find the next valid sample index in the given direction.
    
    Args:
        start_idx: Starting index
        direction: 1 for forward, -1 for backward
    
    Returns:
        Next valid index, or start_idx if no valid sample found
    """
    global _filtered_indices, _max_idx
    
    if not _filtered_indices:
        return start_idx
    
    # Start from the next index
    current_idx = start_idx + direction
    max_attempts = len(_filtered_indices)
    
    for _ in range(max_attempts):
        # Wrap around
        if current_idx > _max_idx:
            current_idx = 0
        elif current_idx < 0:
            current_idx = _max_idx
        
        actual_idx = _filtered_indices[current_idx]
        if is_valid_sample(actual_idx):
            return current_idx
        
        current_idx += direction
    
    # No valid sample found, return start
    print("Warning: No valid samples found in dataset")
    return start_idx


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
    
    # Find first valid sample
    if _current_idx >= 0:
        _current_idx = find_next_valid_index(-1, direction=1)  # Start from -1 to check from 0
    
    print(f"Dataset has {len(_dataset)} samples, {len(_filtered_indices)} after filtering")


def load_model_and_dataset(
    config_path,
    dit_ckpt_path=None,
    vae_config_path=None,
    vae_checkpoint_path=None,
    device='cuda',
    use_ddim=False,
    ddim_eta=0.0,
):
    """Load dataset and corresponding model weights using config-based loading."""
    global _dataset, _dataset_train, _dataset_val, _model, _pipe, _max_idx, _vae, _dataset_compound
    global _pred_is_closed, _vae_model_name, _dit_model_name, _log_scale
    global _use_ddim, _ddim_eta, _config_path
    
    # Store config path for later use
    _config_path = config_path
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Load datasets using config
    _dataset_train = load_dataset_from_config(config, section='data_train')
    _dataset_val = load_dataset_from_config(config, section='data_val')
    
    # Set initial dataset to val
    _dataset = _dataset_val
    
    # Get log_scale from config - check both data_train.params and top-level
    if 'data_train' in config and 'params' in config.data_train:
        _log_scale = config.data_train.params.get('log_scale', True)
    elif 'log_scale' in config:
        _log_scale = config.log_scale
    else:
        _log_scale = True
    
    print(f"Using log_scale: {_log_scale}")
    
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
    _use_ddim = use_ddim
    _ddim_eta = ddim_eta
    scheduler = get_new_scheduler(prediction_type, num_training_timesteps, 
                                   use_ddim=use_ddim, ddim_eta=ddim_eta)
    _pipe = ZLDMPipeline(_model, scheduler, dtype=torch.float32)
    
    scheduler_name = "DDIM" if use_ddim else "DDPM"
    print(f"Loaded DiT model: {_dit_model_name}")
    print(f"Loaded scheduler: {scheduler_name} (prediction_type={prediction_type})")
    if use_ddim:
        print(f"  DDIM eta={ddim_eta} ({'deterministic' if ddim_eta == 0.0 else 'stochastic'})")


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


def encode_params_to_latents(params_padded, surface_type, masks):
    """
    Encode surface parameters to latents using VAE (without gradient).
    
    Args:
        params_padded: (B, num_max_pad, param_dim) Surface parameters
        surface_type: (B, num_max_pad) Surface types
        masks: (B, num_max_pad) Binary mask
        
    Returns:
        latents_padded: (B, num_max_pad, latent_dim) Latent representations
    """
    global _vae
    
    masks_bool = masks.bool()
    valid_indices = torch.nonzero(masks_bool, as_tuple=False)
    
    if valid_indices.numel() > 0:
        # Extract valid surfaces only
        valid_params = params_padded[masks_bool]
        valid_surface_type = surface_type[masks_bool]
        
        # Encode to latents (no gradient)
        mu, var = _vae.encode(valid_params, valid_surface_type)
        valid_latents = mu
        
        # Create padded output
        B, num_max_pad = masks_bool.shape
        latent_dim = valid_latents.shape[-1]
        latents_padded = torch.zeros(
            B, num_max_pad, latent_dim,
            device=valid_latents.device,
            dtype=valid_latents.dtype
        )
        
        # Place valid latents back
        batch_indices = valid_indices[:, 0]
        pad_indices = valid_indices[:, 1]
        latents_padded[batch_indices, pad_indices] = valid_latents
    else:
        B, num_max_pad = masks_bool.shape
        latent_dim = 128  # Default latent dimension
        latents_padded = torch.zeros(
            B, num_max_pad, latent_dim,
            device=params_padded.device,
            dtype=params_padded.dtype
        )
    
    return latents_padded


def process_index(idx: int, num_inference_steps: int, use_gt_mask=False, return_raw_data=False):
    """Process a single index: load GT, generate point cloud condition, run inference.
    
    Args:
        idx: dataset index
        num_inference_steps: number of diffusion steps
        use_gt_mask: whether to use ground truth mask
        return_raw_data: if True, return raw shifts/rotations/scales/params tensors before visualization
    
    Returns:
        None if valid_mask is False, otherwise returns the processing results
    """
    global _dataset, _pipe, _vae, _pred_is_closed, _use_custom_num_surfaces, _custom_num_surfaces, _log_scale
    
    device = next(_pipe.denoiser.parameters()).device
    
    forward_args = _dataset[idx]
    
    # Check valid_mask (last element)
    valid_mask = forward_args[-1]
    if not valid_mask:
        print(f"Skipping index {idx}: valid_mask is False")
        return None
    
    # Convert all tensors to device except the last valid_mask
    forward_args = [_.to(device).unsqueeze(0) for _ in forward_args[:-1]]
    
    # Unpack data: sampled_points, shifts, rotations, scales, params, types, masks
    # Dataset returns: sampled_points, shifts, rotations, scales, params, surface_type, masks, valid_mask
    sampled_points, shifts_padded, rotations_padded, scales_padded, params_padded, surface_type, masks = forward_args
    
    # Build custom mask if enabled
    if _use_custom_num_surfaces:
        batch_size = masks.shape[0]
        max_num_surfaces = masks.shape[1]
        custom_mask = torch.zeros(batch_size, max_num_surfaces, device=device, dtype=masks.dtype)
        num_surfaces = min(_custom_num_surfaces, max_num_surfaces)
        custom_mask[:, :num_surfaces] = 1
        masks = custom_mask
    
    # Ensure masks is 2D for processing
    if masks.dim() == 1:
        masks = masks.unsqueeze(0)
    
    # Generate point cloud condition from GT (same as in trainer)
    with torch.no_grad():
        # Encode params to latents using VAE
        latents_padded = encode_params_to_latents(params_padded, surface_type, masks)
        
        # Use sampled_points if available, otherwise generate them
        
        # Reshape sampled_points from (B, N_Surface, N_points, feat_dim) to (B, N_Surface, N_points*feat_dim)
        B, N_Surface, N_points, feat_dim = sampled_points.shape
        pc_cond = sampled_points.reshape(B, N_Surface, N_points * feat_dim)
        
    
    # Prepare ground truth sample: [mask, shifts, rotations, scales, latents]
    masks_input = masks.unsqueeze(-1)
    gt_sample = torch.cat([
        masks_input.float(), 
        shifts_padded, 
        rotations_padded, 
        scales_padded, 
        latents_padded
    ], dim=-1)
    
    noise = torch.randn_like(gt_sample)
    
    sample = _pipe(noise=noise, pc=pc_cond, num_inference_steps=num_inference_steps, 
                   show_progress=True, tgt_key_padding_mask=~masks_input.bool().squeeze(-1), 
                   memory_key_padding_mask=~masks_input.bool().squeeze(-1))
    
    loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = _compute_loss(sample, gt_sample, masks_input)
    
    # Decode samples - note: now decoding latents, not params
    valid, shifts, rotations, scales, latents = decode_latent(sample, log_scale=_log_scale)
    valid_gt, shifts_gt, rotations_gt, scales_gt, latents_gt = decode_latent(gt_sample, log_scale=_log_scale)
    
    shifts_gt = shifts_gt[valid_gt].squeeze()
    rotations_gt = rotations_gt[valid_gt].squeeze()
    scales_gt = scales_gt[valid_gt].squeeze()
    latents_gt = latents_gt[valid_gt].squeeze()

    
    if use_gt_mask:
        valid = masks_input.squeeze(-1).bool()
    
    shifts = shifts[valid].squeeze()
    rotations = rotations[valid].squeeze()
    scales = scales[valid].squeeze()
    latents = latents[valid].squeeze()
    
    # If return_raw_data is True, return tensors before processing
    if return_raw_data:
        return {
            'shifts': shifts,
            'rotations': rotations,
            'scales': scales,
            'params': latents,  # Return latents as 'params' for compatibility
            'valid': valid,
        }
    
    # Decode latents to params using VAE
    # Note: VAE.decode expects latents, not params
    # with torch.no_grad():
    #     # For GT
    #     if latents_gt.dim() == 1:
    #         latents_gt = latents_gt.unsqueeze(0)
    #     # params_gt = _vae.decode(latents_gt, surface_type)
        
    #     # For predictions
    #     if latents.dim() == 1:
    #         latents = latents.unsqueeze(0)
    #     params = _vae.decode(latents)
    params_gt = latents_gt
    params = latents
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
        # Because the scales are already exp-ed by the decode_latent function
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


def compute_generation_variance(num_runs=10):
    """Generate multiple times with current settings and compute std of outputs.
    
    Args:
        num_runs: number of generations to run (default: 10)
    """
    global _current_idx, _num_inference_steps
    
    print(f"\n{'='*80}")
    print(f"Computing generation variance over {num_runs} runs...")
    print(f"Index: {_current_idx}, Inference steps: {_num_inference_steps}")
    print(f"{'='*80}\n")
    
    # Collect results from multiple runs
    all_shifts = []
    all_rotations = []
    all_scales = []
    all_params = []
    
    with torch.no_grad():
        for run_idx in range(num_runs):
            print(f"Run {run_idx + 1}/{num_runs}...", end=' ')
            
            # Run inference with return_raw_data=True
            result = process_index(_current_idx, _num_inference_steps, 
                                  use_gt_mask=True, return_raw_data=True)
            
            # Check if sample was skipped
            if result is None:
                print(f"Sample {_current_idx} is invalid, cannot compute variance")
                return None
            
            # Extract tensors
            shifts = result['shifts'].cpu().numpy()
            rotations = result['rotations'].cpu().numpy()
            scales = result['scales'].cpu().numpy()
            params = result['params'].cpu().numpy()
            
            # Ensure consistent shape (handle both single surface and multiple surfaces)
            if shifts.ndim == 1:
                shifts = shifts[None, :]
            if rotations.ndim == 1:
                rotations = rotations[None, :]
            if scales.ndim == 0:
                scales = scales[None]
            if params.ndim == 1:
                params = params[None, :]
            
            all_shifts.append(shifts)
            all_rotations.append(rotations)
            all_scales.append(scales)
            all_params.append(params)
            
            print("Done")
    
    # Stack results: shape (num_runs, num_surfaces, feature_dim)
    all_shifts = np.stack(all_shifts, axis=0)  # (num_runs, num_surfaces, 3)
    all_rotations = np.stack(all_rotations, axis=0)  # (num_runs, num_surfaces, 6)
    all_scales = np.stack(all_scales, axis=0)  # (num_runs, num_surfaces)
    all_params = np.stack(all_params, axis=0)  # (num_runs, num_surfaces, param_dim)
    
    # Compute standard deviation across runs (axis=0)
    std_shifts = np.std(all_shifts, axis=0)  # (num_surfaces, 3)
    std_rotations = np.std(all_rotations, axis=0)  # (num_surfaces, 6)
    std_scales = np.std(all_scales, axis=0)  # (num_surfaces,)
    std_params = np.std(all_params, axis=0)  # (num_surfaces, param_dim)
    
    # Compute mean for reference
    mean_shifts = np.mean(all_shifts, axis=0)
    mean_rotations = np.mean(all_rotations, axis=0)
    mean_scales = np.mean(all_scales, axis=0)
    mean_params = np.mean(all_params, axis=0)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"VARIANCE ANALYSIS RESULTS (over {num_runs} runs)")
    print(f"{'='*80}\n")
    
    num_surfaces = std_shifts.shape[0]
    print(f"Number of surfaces: {num_surfaces}\n")
    
    # Summary statistics (average across all surfaces)
    print("=== AVERAGE STD ACROSS ALL SURFACES ===")
    print(f"Shifts STD:    {np.mean(std_shifts):.6f}  (per-dim: {np.mean(std_shifts, axis=0)})")
    print(f"Rotations STD: {np.mean(std_rotations):.6f}  (per-dim mean: {np.mean(std_rotations, axis=0).mean():.6f})")
    print(f"Scales STD:    {np.mean(std_scales):.6f}")
    print(f"Params STD:    {np.mean(std_params):.6f}  (per-dim mean: {np.mean(std_params, axis=0).mean():.6f})\n")
    
    # Per-surface details
    print("=== PER-SURFACE STD ===")
    for i in range(num_surfaces):
        print(f"\nSurface {i}:")
        print(f"  Shifts STD:    mean={np.mean(std_shifts[i]):.6f}, xyz={std_shifts[i]}")
        print(f"  Rotations STD: mean={np.mean(std_rotations[i]):.6f}, range=[{np.min(std_rotations[i]):.6f}, {np.max(std_rotations[i]):.6f}]")
        print(f"  Scales STD:    {std_scales[i]:.6f}")
        print(f"  Params STD:    mean={np.mean(std_params[i]):.6f}, range=[{np.min(std_params[i]):.6f}, {np.max(std_params[i]):.6f}]")
    
    # Relative variance (std / mean)
    print(f"\n{'='*80}")
    print("=== COEFFICIENT OF VARIATION (STD/MEAN) ===")
    print(f"{'='*80}\n")
    
    eps = 1e-8  # prevent division by zero
    cv_shifts = np.mean(std_shifts / (np.abs(mean_shifts) + eps))
    cv_rotations = np.mean(std_rotations / (np.abs(mean_rotations) + eps))
    cv_scales = np.mean(std_scales / (np.abs(mean_scales) + eps))
    cv_params = np.mean(std_params / (np.abs(mean_params) + eps))
    
    print(f"Shifts CV:    {cv_shifts:.6f}")
    print(f"Rotations CV: {cv_rotations:.6f}")
    print(f"Scales CV:    {cv_scales:.6f}")
    print(f"Params CV:    {cv_params:.6f}")
    
    print(f"\n{'='*80}\n")
    
    return {
        'std_shifts': std_shifts,
        'std_rotations': std_rotations,
        'std_scales': std_scales,
        'std_params': std_params,
        'mean_shifts': mean_shifts,
        'mean_rotations': mean_rotations,
        'mean_scales': mean_scales,
        'mean_params': mean_params,
    }


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
            
            # Check valid_mask
            valid_mask = forward_args[-1]
            if not valid_mask:
                print(f"Warning: Sample {_current_idx} has valid_mask=False")
                return
            
            # Dataset format: sampled_points, shifts, rotations, scales, params, surface_type, masks, valid_mask
            masks = forward_args[6]  # masks is the 7th element (index 6)
            
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
        
        # Check if sample was skipped
        if result is None:
            print(f"Sample {_current_idx} is invalid, skipping visualization")
            return
        
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
    global _use_ddim, _ddim_eta, _pipe, _config_path
    
    psim.Text("DiT Sample Surface Segment Inference")
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
    psim.Text(f"Current: {_current_idx} / {_max_idx}")
    psim.Separator()
    
    # Slider for browsing
    slider_changed, slider_idx = psim.SliderInt("Browse (Slider)", _current_idx, 0, _max_idx)
    if slider_changed and slider_idx != _current_idx:
        _current_idx = slider_idx
        update_visualization()
    
    # Input box for direct jump
    psim.PushItemWidth(150)
    input_changed, input_idx = psim.InputInt("Jump to Index", _current_idx)
    psim.PopItemWidth()
    if input_changed:
        input_idx = max(0, min(_max_idx, input_idx))
        if input_idx != _current_idx:
            _current_idx = input_idx
            update_visualization()
    
    psim.SameLine()
    psim.Text(f"(0-{_max_idx})")
    
    # Quick navigation buttons
    psim.PushItemWidth(80)
    if psim.Button("First"):
        if _current_idx != 0:
            new_idx = find_next_valid_index(-1, direction=1)  # Find first valid from start
            if new_idx != _current_idx:
                _current_idx = new_idx
                update_visualization()
    
    psim.SameLine()
    if psim.Button("Prev"):
        if _current_idx > 0:
            new_idx = find_next_valid_index(_current_idx, direction=-1)
            if new_idx != _current_idx:
                _current_idx = new_idx
                update_visualization()
    
    psim.SameLine()
    if psim.Button("Next"):
        if _current_idx < _max_idx:
            new_idx = find_next_valid_index(_current_idx, direction=1)
            if new_idx != _current_idx:
                _current_idx = new_idx
                update_visualization()
    
    psim.SameLine()
    if psim.Button("Last"):
        if _current_idx != _max_idx:
            # Find last valid sample by searching backwards from end
            temp_idx = _max_idx + 1
            new_idx = find_next_valid_index(temp_idx, direction=-1)
            if new_idx != _current_idx:
                _current_idx = new_idx
                update_visualization()
    psim.PopItemWidth()
    
    # Inference steps control
    psim.Separator()
    psim.Text("Inference Settings:")
    steps_changed, new_steps = psim.SliderInt("Inference Steps", _num_inference_steps, 1, 1000)
    if steps_changed:
        _num_inference_steps = new_steps
    
    if psim.Button("Regenerate"):
        update_visualization()
    
    psim.SameLine()
    if psim.Button("Compute Variance (10 runs)"):
        compute_generation_variance(num_runs=10)
    
    psim.Separator()
    psim.Text(f"Inference Steps: {_num_inference_steps}")
    
    # Sampling method control
    psim.Separator()
    psim.Text("Sampling Method:")
    
    changed_ddim, new_use_ddim = psim.Checkbox("Use DDIM (faster)", _use_ddim)
    if changed_ddim:
        _use_ddim = new_use_ddim
        # Recreate scheduler and update pipeline
        if _config_path:
            from omegaconf import OmegaConf
            config = OmegaConf.load(_config_path)
            prediction_type = config.trainer.get('prediction_type', 'v_prediction')
            num_training_timesteps = config.trainer.get('num_training_timesteps', 1000)
            from src.flow.surface_flow import get_new_scheduler
            scheduler = get_new_scheduler(prediction_type, num_training_timesteps,
                                           use_ddim=_use_ddim, ddim_eta=_ddim_eta)
            _pipe.scheduler = scheduler
            _pipe.scheduler_type = type(scheduler).__name__
            print(f"Switched to {'DDIM' if _use_ddim else 'DDPM'} scheduler")
    
    if _use_ddim:
        eta_changed, new_eta = psim.SliderFloat("DDIM eta", _ddim_eta, 0.0, 1.0)
        if eta_changed:
            _ddim_eta = new_eta
            _pipe.scheduler.eta = _ddim_eta
        psim.Text(f"eta={_ddim_eta:.2f} ({'det' if _ddim_eta < 0.01 else 'stoch'})")
    
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
    parser = argparse.ArgumentParser(description='Test DiT Sample Surface Segment model with visualization.')
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
        default=50,
        help='Number of inference steps for diffusion sampling'
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        help='Device to run on (cuda or cpu)'
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help='Starting index for visualization'
    )
    parser.add_argument(
        "--use_ddim",
        action='store_true',
        default=False,
        help='Use DDIM scheduler instead of DDPM'
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help='DDIM eta parameter (0.0=deterministic, 1.0=stochastic)'
    )
    args = parser.parse_args()

    _num_inference_steps = args.num_inference_steps
    _current_idx = args.start_idx

    print("="*80)
    print("DiT Sample Surface Segment Test Script")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt_path if args.ckpt_path else '(from config)'}")
    print(f"VAE Config: {args.vae_config if args.vae_config else '(from config)'}")
    print(f"Device: {args.device}")
    print(f"Inference Steps: {args.num_inference_steps}")
    print(f"Start Index: {args.start_idx}")
    print(f"Scheduler: {'DDIM' if args.use_ddim else 'DDPM'}")
    if args.use_ddim:
        print(f"DDIM eta: {args.ddim_eta}")
    print("="*80 + "\n")

    load_model_and_dataset(
        config_path=args.config,
        dit_ckpt_path=args.ckpt_path if args.ckpt_path else None,
        vae_config_path=args.vae_config if args.vae_config else None,
        vae_checkpoint_path=args.vae_ckpt if args.vae_ckpt else None,
        device=args.device,
        use_ddim=args.use_ddim,
        ddim_eta=args.ddim_eta,
    )
    
    # Update max index based on filtered indices
    if _filtered_indices:
        _max_idx = len(_filtered_indices) - 1
        _current_idx = min(_current_idx, _max_idx)
        
        # Ensure starting index is valid
        if not is_valid_sample(_filtered_indices[_current_idx]):
            print(f"Starting index {_current_idx} is invalid, finding next valid sample...")
            _current_idx = find_next_valid_index(_current_idx - 1, direction=1)
            print(f"Set starting index to: {_current_idx}")
    else:
        _max_idx = len(_dataset) - 1 if _dataset else 0
        _current_idx = min(_current_idx, _max_idx)
    
    print(f"Loaded dataset: {len(_dataset)} samples")
    print(f"Filtered samples: {len(_filtered_indices)}")
    print(f"Max index: {_max_idx}\n")
    
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
    print("Loading initial visualization...")
    update_visualization()
    
    print("\n" + "="*80)
    print("Polyscope GUI is ready!")
    print("Use the UI controls to navigate through samples and adjust settings.")
    print("="*80 + "\n")
    
    # Show the interface
    ps.show()

