import sys
import json
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.import_tools import load_dataset_from_config, load_model_from_config
from src.utils.surface_latent_tools import decode_and_sample_with_rts
from src.dataset.dataset_v1 import dataset_compound

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from omegaconf import OmegaConf



# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.tests.test_vae_v1 import to_json
from src.tools.surface_to_canonical_space import from_canonical
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




def reset_scene():
    """Clear Polyscope structures while preserving the reference cube."""
    if not _ps_initialized:
        return
    ps.remove_all_structures()



# Globals for UI
_dataset = None
_dataset_train = None
_dataset_val = None
_use_train_dataset = False  # False = val, True = train
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
# Surface groups
_gt_surfaces = {}
_gen_surfaces = {}
# pred_is_closed related variables
_pred_is_closed = False
_show_closed_colors = True  # Toggle for showing is_closed coloring
_vae_model_name = 'vae_v1'  # VAE model name
_dit_model_name = 'dit_v1'
# Custom surface count control
_use_custom_num_surfaces = False  # Whether to use custom surface count
_custom_num_surfaces = 20  # User-specified surface count (15-32)
_default_num_surfaces = 20  # Default value from GT mask

# Surface type mapping
SURFACE_TYPE_MAP_INV = {
    0: 'plane',
    1: 'cylinder',
    2: 'cone',
    3: 'sphere',
    4: 'torus',
    5: 'bspline_surface',
}















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
    global _dataset, _pipe, _gt_pc, _vae, _pred_is_closed, _use_custom_num_surfaces, _custom_num_surfaces
    
    device='cuda'
    
    forward_args = _dataset[idx]
    forward_args = [_.to(device).unsqueeze(0) for _ in forward_args]
    
    # Check if dataset returns is_closed data

    params_padded, rotations_padded, scales_padded, shifts_padded, surface_type, bbox_mins, bbox_maxs, masks, pc_cond = forward_args


    
# decode and visualize surfaces
    masks = masks.unsqueeze(-1)
    gt_sample = torch.cat([masks.float(), shifts_padded, rotations_padded, scales_padded, params_padded], dim=-1).to(device)
    valid_gt, shifts_gt, rotations_gt, scales_gt, params_gt = decode_sample(gt_sample)
    shifts_gt = shifts_gt[valid_gt].squeeze()
    rotations_gt = rotations_gt[valid_gt].squeeze()
    scales_gt = scales_gt[valid_gt].squeeze()
    params_gt = params_gt[valid_gt].squeeze()
    # Classify with is_closed support
    if _pred_is_closed:
        type_logits_pred_gt, types_pred_gt, is_closed_logits_gt, is_closed_pred_gt = _vae.classify(params_gt)
        gt_is_u_closed = is_closed_pred_gt[:, 0].cpu().numpy()
        gt_is_v_closed = is_closed_pred_gt[:, 1].cpu().numpy()
    else:
        type_logits_pred_gt, types_pred_gt = _vae.classify(params_gt)
        gt_is_u_closed = None
        gt_is_v_closed = None
    
    params_decoded_gt, mask_gt = _vae.decode(params_gt, types_pred_gt)
    shifts_gt, rotations_gt, scales_gt = fix_rts(shifts_gt, rotations_gt, scales_gt)
    surface_jsons_gt = to_json(params_decoded_gt.cpu().numpy(), types_pred_gt.cpu().numpy(), mask_gt.cpu().numpy())
    surface_jsons_gt = [from_canonical(surface_jsons_gt[i], shifts_gt[i], rotations_gt[i], scales_gt[i]) for i in range(len(surface_jsons_gt))]
    # Prepare is_closed data for return
    is_closed_data = {
        'gt_is_u_closed': gt_is_u_closed,
        'gt_is_v_closed': gt_is_v_closed,
    }

# decode and sample
    latent_params = params_padded[valid_gt.bool()].to(device)
    shifts_tensor = shifts_padded[valid_gt.bool()].to(device)
    rotations_tensor = rotations_padded[valid_gt.bool()].to(device)
    scales_tensor = scales_padded[valid_gt.bool()].to(device)

    samples = decode_and_sample_with_rts(_vae, latent_params, shifts_tensor, rotations_tensor, scales_tensor, log_scale=_log_scale)

    return samples, surface_jsons_gt,  is_closed_data

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
        # print('surface index: ',i)
        recovered_surface = _dataset_compound._recover_surface(params, types_tensor[i].item())

        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)

    return json_data


def update_visualization():
    """Update the visualization with current index"""
    global _current_idx, _gen_group, _gt_group, _gen_pc, _num_inference_steps
    global _gt_surfaces, _gen_surfaces, _show_gt, _show_gen
    global _pred_is_closed, _show_closed_colors, _default_num_surfaces, _dataset
    global _use_custom_num_surfaces, _custom_num_surfaces
    global _gt_sample_group
    
    if not _ps_initialized:
        return
    
    # Update default num_surfaces from GT mask
    if _dataset is not None:
        try:
            forward_args = _dataset[_current_idx]
            masks = forward_args[7]  # masks is the 8th element (0-indexed)
            gt_mask_count = int(masks.sum().item())
            _default_num_surfaces = gt_mask_count
            # If not using custom, update custom_num_surfaces to match default
            if not _use_custom_num_surfaces:
                _custom_num_surfaces = _default_num_surfaces
        except Exception as e:
            print(f"Warning: Could not get GT mask count: {e}")
    
    # Clear existing structures
    reset_scene()
    
    # Process current sample
    print(f"\nProcessing index {_current_idx}...")
    with torch.no_grad():
        result = process_index(
            _current_idx, _num_inference_steps, use_gt_mask=True
        )
        if len(result) == 3:  # Has is_closed_data
             samples, surface_jsons_gt, is_closed_data = result
        else:
            samples, surface_jsons_gt = result
            is_closed_data = None
    samples = samples.cpu()
    

 


    _gt_surfaces = visualize_json_interset(surface_jsons_gt, plot=True, plot_gui=False, tol=1e-5, ps_header=f'gt_{_current_idx}')
    for surface_key, surface_data in _gt_surfaces.items():
        if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
            surface_data['ps_handler'].add_to_group(_gt_group)
    

    # Configure groups with current visibility settings
    _gt_group.set_enabled(_show_gt)


# Visualize point cloud
    assert len(_gt_surfaces) == len(samples)
    for i in range(len(samples)):
        _pc_group = ps.register_point_cloud(f'sample_{i}_{surface_jsons_gt[i]["type"]}', samples[i].reshape(-1, 3).numpy(), radius=0.003)
        _pc_group.add_to_group(_gt_sample_group)
    
    
    print(f"Visualized  {len(_gt_surfaces)} GT surfaces")

def callback():
    """Polyscope callback function for UI controls"""
    global _current_idx, _max_idx, _num_inference_steps
    global _show_gt, _gt_group, _gen_group
    global _use_train_dataset, _dataset_train, _dataset_val
    global _show_closed_colors, _use_custom_num_surfaces, _custom_num_surfaces, _default_num_surfaces
    
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

    # Group visibility controls
    if _gt_group is not None and _gen_group is not None:
        psim.Separator()
        psim.Text("Visibility Controls:")
        changed_gt, _show_gt = psim.Checkbox("Show Ground Truth", _show_gt)
        if changed_gt:
            _gt_group.set_enabled(_show_gt)
        

        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument(
        '--vae_config',
        type=str,
        default='',
        help='Path to the VAE YAML config file (for model version and pred_is_closed)'
    )
    parser.add_argument(
        '--dataset_config',
        type=str,
        default='',
        help='Path to the dataset config'
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

    
    vae_config = OmegaConf.load(args.vae_config)

    _vae, _ = load_model_from_config(vae_config)
    _vae = _vae.to('cuda')
    _vae = _vae

    _pred_is_closed = vae_config.model.pred_is_closed



    dataset_config = OmegaConf.load(args.dataset_config)
    dataset = load_dataset_from_config(dataset_config)
    _dataset = dataset
    _log_scale = dataset_config.data.params.log_scale

    _dataset_compound = dataset_compound(json_dir='./', canonical=True, detect_closed=_pred_is_closed)


    _current_idx = args.start_idx


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
    
    # Create surface groups
    _gt_group = ps.create_group("Ground Truth Surfaces")
    _gt_sample_group = ps.create_group("Ground Truth Samples")


    
    # Set user callback
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()
