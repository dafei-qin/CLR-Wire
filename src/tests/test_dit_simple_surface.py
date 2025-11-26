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



# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_latent import LatentDataset
from src.dataset.dataset_v1 import dataset_compound
from src.utils.config import NestedDictToClass, load_config
from src.flow.surface_flow import ZLDMPipeline, get_new_scheduler
from src.tests.test_vae_v1 import to_json
from src.vae.vae_v1 import SurfaceVAE
from src.tools.surface_to_canonical_space import from_canonical
from utils.surface import visualize_json_interset



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
            transparency=0.7,
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
_model = None
_pipe = None
_current_idx = 0
_current_filtered_idx = 0
_max_idx = 0
_gt_group = None
_gen_group = None
_gt_pc = None
_gen_pc = None
_show_gt = True
_show_gen = True
_show_pc = True
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
            latent_params, rotations, scales, shifts, classes, bbox_mins, bbox_maxs, mask, pc = _dataset[idx]
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


def load_model_and_dataset(
    args,
    device = 'cuda'
):
    """Load dataset and corresponding model weights."""
    global _dataset, _model, _pipe, _max_idx, _vae, _dataset_compound

    _dataset = LatentDataset(
    latent_dir=args.data.train_latent_dir, pc_dir=args.data.train_pc_dir, max_num_surfaces=args.data.max_num_surfaces, 
    latent_dim=args.data.surface_latent_dim, num_data=args.data.val_num,
    log_scale=args.data.log_scale
    )
    
    _dataset_compound = dataset_compound(json_dir='./',canonical=True)
    build_index_metadata()
    initialize_filters()
    refresh_filtered_indices(preserve_current=False)

    # Import and create model
    from src.dit.simple_surface_decoder import SimpleSurfaceDecoder
    
    _model = SimpleSurfaceDecoder(
        input_dim=args.model.input_dim,
        cond_dim=args.model.cond_dim,
        output_dim=args.model.output_dim,
        latent_dim=args.model.latent_dim,
        num_layers=args.model.num_layers,
        num_heads=args.model.num_heads
    )

    _model.to(device)
    # Load checkpoint
    checkpoint = torch.load(args.model.checkpoint_file_name, map_location=device)
    if 'ema_model' in checkpoint or 'ema' in checkpoint:
        ema_key = 'ema' if 'ema' in checkpoint else 'ema_model'
        ema_model = checkpoint[ema_key]
        ema_model = {k.replace("ema_model.", "").replace("ema.", ""): v for k, v in ema_model.items()}
        _model.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        _model.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        _model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")

    
    _vae = SurfaceVAE(args.vae.param_raw_dim)

    checkpoint = torch.load(args.vae.checkpoint_file_name, map_location=device)
    if 'ema_model' in checkpoint or 'ema' in checkpoint:
        ema_key = 'ema' if 'ema' in checkpoint else 'ema_model'
        ema_model = checkpoint[ema_key]
        ema_model = {k.replace("ema_model.", "").replace("ema.", ""): v for k, v in ema_model.items()}
        _vae.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        _vae.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        _vae.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")

    _vae.to(device)
    _vae.eval()

    _model.eval()

    # Create pipeline
    scheduler = get_new_scheduler('v_prediction', 1000)
    _pipe = ZLDMPipeline(_model, scheduler, dtype=torch.float32)



def _compute_loss(output, target, masks):

        loss_raw = torch.nn.functional.mse_loss(output, target, reduction='none')

        loss_others = loss_raw[..., 1:] * (1 - masks.float())
        total_valid_surfaces = masks.float().sum()
        # loss_types = loss_others[..., :self.model.num_surface_types]
        loss_shifts = loss_others[..., :3].mean(dim=(2)).sum() / total_valid_surfaces
        loss_rotations = loss_others[..., 3:3+6].mean(dim=(2)).sum() / total_valid_surfaces
        loss_scales = loss_others[..., 3+6:3+6+1].mean(dim=(2)).sum() / total_valid_surfaces
        loss_params = loss_others[..., 3+6+1:].mean(dim=(2)).sum() / total_valid_surfaces
        
        bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_valid = bce_logits_loss(output[..., 0], masks.float().squeeze(-1)).mean()
        # print('gt_scale: ', target[..., 3+6+1:3+6+2])
        # print('pred_scale: ', output[..., 3+6+1:3+6+2])
        
        return loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params

def decode_sample(sample: torch.Tensor):
    global _log_scale
    valid_tensor = sample[..., 0]
    sample = sample[..., 1:]
    shifts_tensor = sample[..., :3]
    rotations_tensor = sample[..., 3:3+6]
    scales_tensor = sample[..., 3+6:3+6+1]
    if _log_scale:
        scales_tensor = torch.exp(scales_tensor)

    params_tensor = sample[..., 3+6+1:]
    valid = torch.sigmoid(valid_tensor) > 0.5

    return valid, shifts_tensor, rotations_tensor, scales_tensor, params_tensor

def process_index(idx: int, num_inference_steps: int, use_gt_mask=False):
    """Process a single index: load GT, sample point cloud, run inference."""
    global _dataset, _pipe, _gt_pc, _vae
    
    device = next(_pipe.denoiser.parameters()).device
    
    forward_args = _dataset[idx]
    forward_args = [_.to(device).unsqueeze(0) for _ in forward_args]
    params_padded, rotations_padded, scales_padded, shifts_padded, surface_type, bbox_mins, bbox_maxs, masks, pc_cond = forward_args
    masks = masks.unsqueeze(-1)
    print(scales_padded)
    gt_sample = torch.cat([masks.float(), shifts_padded, rotations_padded, scales_padded, params_padded], dim=-1)

    # gt_sample = gt_sample.unsqueeze(0)
    # pc_cond = pc_cond.unsqueeze(0)
    noise = torch.randn_like(gt_sample)

    sample  = _pipe(noise=noise, pc=pc_cond, num_inference_steps=num_inference_steps, show_progress=True)

    loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = _compute_loss(sample, gt_sample, masks)
    # sample = torch.cat([gt_sample[..., :1], sample[..., 1:1+3+6], gt_sample[..., 1+3+6:1+3+6+1], sample[..., -128:]], dim=-1)
    sample = torch.cat([gt_sample[..., :-128], sample[..., -128:]], dim=-1)
    valid, shifts, rotations, scales, params = decode_sample(sample)

    valid_gt, shifts_gt, rotations_gt, scales_gt, params_gt = decode_sample(gt_sample)
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

    type_logits_pred, types_pred = _vae.classify(params)

    type_logits_pred_gt, types_pred_gt = _vae.classify(params_gt)
    
    params_decoded_gt, mask_gt = _vae.decode(params_gt, types_pred_gt)
    shifts_gt, rotations_gt, scales_gt = fix_rts(shifts_gt, rotations_gt, scales_gt)
    surface_jsons_gt = to_json(params_decoded_gt.cpu().numpy(), types_pred_gt.cpu().numpy(), mask_gt.cpu().numpy())
    surface_jsons_gt = [from_canonical(surface_jsons_gt[i], shifts_gt[i], rotations_gt[i], scales_gt[i]) for i in range(len(surface_jsons_gt))]
        
    # Decode to get surface parameters
    params_decoded, mask = _vae.decode(params, types_pred)

    shifts, rotations, scales = fix_rts(shifts, rotations, scales)

    surface_jsons = to_json(params_decoded.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
    surface_jsons = [from_canonical(surface_jsons[i], shifts[i], rotations[i], scales[i]) for i in range(len(surface_jsons))]

    return surface_jsons, surface_jsons_gt, loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params

def fix_rts(shifts, rotations, scales):
    shifts = shifts.cpu().numpy()
    rotations = rotations.cpu().numpy()
    scales = scales.cpu().numpy()
    D = rotations[:, :3]
    X = rotations[:, 3:6]
    D = D / np.linalg.norm(D)
    X = X / np.linalg.norm(X)
    Y = np.cross(D, X)
    Y = Y / np.linalg.norm(Y)
    X = np.cross(Y, D)
    X = X / np.linalg.norm(X)
    rotations = np.concatenate([D, X, Y], axis=1).reshape(-1, 3, 3)
    return shifts, rotations, scales

def to_json(params_tensor, types_tensor, mask_tensor):
    global _dataset_compound
    json_data = []
    # SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    for i in range(len(params_tensor)):
        # (types_tensor[i].item(), mask_tensor[i].sum())
        params = params_tensor[i][mask_tensor[i]]
        # surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        print('surface index: ',i)
        recovered_surface = _dataset_compound._recover_surface(params, types_tensor[i].item())

        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)

    return json_data


def update_visualization():
    """Update the visualization with current index"""
    global _current_idx, _gen_group, _gen_pc, _num_inference_steps
    
    if not _ps_initialized:
        return
    
    # Clear existing structures
    reset_scene()
    
    # Process current sample
    print(f"\nProcessing index {_current_idx}...")
    with torch.no_grad():
        surface_jsons, surface_jsons_gt, loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = process_index(
            _current_idx, _num_inference_steps, use_gt_mask=True
        )
    
    # Display losses
    print(f"Losses - Valid: {loss_valid.item():.6f}, Shifts: {loss_shifts.item():.6f}, "
          f"Rotations: {loss_rotations.item():.6f}, Scales: {loss_scales.item():.6f}, "
          f"Params: {loss_params.item():.6f}")
    
    # Visualize generated surfaces

    try:
        print(f"Visualized {len(surface_jsons)} generated surfaces")
        visualize_json_interset(surface_jsons, plot=True, plot_gui=False, tol=1e-5, ps_header=f'sample_{_current_idx}')
        visualize_json_interset(surface_jsons_gt, plot=True, plot_gui=False, tol=1e-5, ps_header=f'gt_{_current_idx}')

    except Exception as e:
        print(f'Error visualizing surfaces: {e}')


def callback():
    """Polyscope callback function for UI controls"""
    global _current_idx, _max_idx, _num_inference_steps
    
    psim.Text("DiT Simple Surface Inference")
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



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/train_dit_simple_surface_v1.yaml',
        help='Path to the YAML config file.',
    )
    parser.add_argument(
        '--ckpt_path',
        type=str,
        default=''
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

    cfg = load_config(args.config) if args.config else {}
    cfg_args = NestedDictToClass(cfg) if cfg else SimpleNamespace()

    _log_scale = args.data.log_scale
    _num_inference_steps = args.num_inference_steps
    _current_idx = args.start_idx

    if args.ckpt_path != '':
        cfg_args.model.checkpoint_file_name = args.ckpt_path
    load_model_and_dataset(
        cfg_args,
        args.device
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
    
    # Register reference cube
    register_unit_cube()
    
    # Set user callback
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()
