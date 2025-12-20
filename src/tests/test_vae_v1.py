"""
This script test the vae model for simple surface reconstruction.
"""
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
import copy
from scipy.spatial.transform import Rotation as R
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')


from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from src.utils.numpy_tools import orthonormal_basis_from_normal
from src.utils.config import NestedDictToClass, load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config

from utils.surface import visualize_json_interset

# Colors for is_closed visualization
# Format: [R, G, B] in range [0, 1]
COLOR_BOTH_CLOSED = [0.2, 0.8, 0.2]      # Green - both u and v closed
COLOR_U_CLOSED = [0.2, 0.2, 0.8]         # Blue - only u closed
COLOR_V_CLOSED = [0.8, 0.2, 0.2]         # Red - only v closed
COLOR_NEITHER_CLOSED = [0.7, 0.7, 0.7]   # Gray - neither closed

def apply_transformation_to_point(point, scale, rotation_matrix, shift):
    """Apply SRT transformation to a 3D point: Scale -> Rotate -> Translate"""
    try:
        point = np.array(point, dtype=np.float64)
        if point.shape != (3,):
            print(f"Warning: Point has unexpected shape {point.shape}, expected (3,)")
            return point.tolist() if point.size == 3 else list(point.flatten()[:3])
        # Scale
        if _apply_scale:
            point = point * scale
        # Rotate
        if _apply_rotation:
            point = rotation_matrix @ point
        # Translate
        if _apply_shift:
            point = point + np.array(shift, dtype=np.float64)
        return point.tolist()
    except Exception as e:
        print(f"Error in apply_transformation_to_point: {e}")
        return list(point) if hasattr(point, '__iter__') else [0.0, 0.0, 0.0]


def apply_transformation_to_vector(vector, rotation_matrix):
    """Apply rotation to a direction vector (no scale or translation)"""
    try:
        if not _apply_rotation:
            return vector
        vector = np.array(vector, dtype=np.float64)
        if vector.shape != (3,):
            print(f"Warning: Vector has unexpected shape {vector.shape}, expected (3,)")
            return vector.tolist() if vector.size == 3 else list(vector.flatten()[:3])
        result = rotation_matrix @ vector
        return result.tolist()
    except Exception as e:
        print(f"Error in apply_transformation_to_vector: {e}")
        return list(vector) if hasattr(vector, '__iter__') else [0.0, 0.0, 1.0]


def apply_transformation_to_surface(surface_dict):
    """Apply geometric transformations to a surface dictionary"""
    if not (_apply_scale or _apply_rotation or _apply_shift):
        return surface_dict
    
    # Create rotation matrix
    if _apply_rotation:
        rotation = R.from_euler('xyz', _rotation_euler, degrees=True)
        rot_matrix = rotation.as_matrix()
    else:
        rot_matrix = np.eye(3)
    
    surface_type = surface_dict.get('type', '')
    
    # Handle unknown or unsupported surface types
    if surface_type not in ['plane', 'cylinder', 'cone', 'sphere', 'torus', 'bspline_surface']:
        print(f"Warning: Unsupported surface type '{surface_type}' for transformation, skipping.")
        return surface_dict
    
    # Transform based on surface type
    if surface_type == 'plane':
        # Transform location and axis
        if 'scalar' in surface_dict:
            scalar = surface_dict['scalar']
            if len(scalar) >= 6:
                # Location: first 3 elements
                location = apply_transformation_to_point(scalar[:3], _scale_factor, rot_matrix, _shift_xyz)
                # Axis (normal): next 3 elements
                axis = apply_transformation_to_vector(scalar[3:6], rot_matrix)
                remaining = scalar[6:] if len(scalar) > 6 else []
                surface_dict['scalar'] = location + axis + remaining
    
    elif surface_type in ['cylinder', 'cone']:
        # Transform location and axis
        if 'scalar' in surface_dict:
            scalar = surface_dict['scalar']
            if len(scalar) >= 6:
                # Location: first 3 elements
                location = apply_transformation_to_point(scalar[:3], _scale_factor, rot_matrix, _shift_xyz)
                # Axis: next 3 elements
                axis = apply_transformation_to_vector(scalar[3:6], rot_matrix)
                # Radius: scale it if exists
                if len(scalar) >= 7:
                    radius = scalar[6] * (_scale_factor if _apply_scale else 1.0)
                    remaining = scalar[7:] if len(scalar) > 7 else []
                    surface_dict['scalar'] = location + axis + [radius] + remaining
                else:
                    surface_dict['scalar'] = location + axis
    
    elif surface_type == 'sphere':
        # Transform center and radius
        if 'scalar' in surface_dict:
            scalar = surface_dict['scalar']
            if len(scalar) >= 3:
                # Center: first 3 elements
                center = apply_transformation_to_point(scalar[:3], _scale_factor, rot_matrix, _shift_xyz)
                # Radius: scale it if exists
                if len(scalar) >= 4:
                    radius = scalar[3] * (_scale_factor if _apply_scale else 1.0)
                    remaining = scalar[4:] if len(scalar) > 4 else []
                    surface_dict['scalar'] = center + [radius] + remaining
                else:
                    surface_dict['scalar'] = center
    
    elif surface_type == 'torus':
        # Transform location, axis, major and minor radius
        if 'scalar' in surface_dict:
            scalar = surface_dict['scalar']
            if len(scalar) >= 6:
                # Location: first 3 elements
                location = apply_transformation_to_point(scalar[:3], _scale_factor, rot_matrix, _shift_xyz)
                # Axis: next 3 elements
                axis = apply_transformation_to_vector(scalar[3:6], rot_matrix)
                # Major and minor radius: scale them if exist
                if len(scalar) >= 8:
                    major_radius = scalar[6] * (_scale_factor if _apply_scale else 1.0)
                    minor_radius = scalar[7] * (_scale_factor if _apply_scale else 1.0)
                    remaining = scalar[8:] if len(scalar) > 8 else []
                    surface_dict['scalar'] = location + axis + [major_radius, minor_radius] + remaining
                else:
                    surface_dict['scalar'] = location + axis
    
    elif surface_type == 'bspline_surface':
        # Transform control poles
        if 'poles' in surface_dict:
            poles = np.array(surface_dict['poles'])
            original_shape = poles.shape
            
            # Handle both (u, v, 3) and (u, v, 4) cases
            xyz = poles[..., :3]
            xyz_flat = xyz.reshape(-1, 3)
            
            # Apply transformations
            xyz_transformed = []
            for point in xyz_flat:
                transformed_point = apply_transformation_to_point(point, _scale_factor, rot_matrix, _shift_xyz)
                xyz_transformed.append(transformed_point)
            xyz_transformed = np.array(xyz_transformed).reshape(xyz.shape)
            
            # Preserve weights if they exist
            if original_shape[-1] == 4:
                weights = poles[..., 3:4]
                poles_transformed = np.concatenate([xyz_transformed, weights], axis=-1)
            else:
                poles_transformed = xyz_transformed
            
            surface_dict['poles'] = poles_transformed.tolist()
    
    return surface_dict


def to_json(params_tensor, types_tensor, mask_tensor):
    json_data = []
    # SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    for i in range(len(params_tensor)):
        # (types_tensor[i].item(), mask_tensor[i].sum())
        params = params_tensor[i][mask_tensor[i]]
        # surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        # print('surface index: ',i)
        recovered_surface = dataset._recover_surface(params, types_tensor[i].item())

        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)

    return json_data

# Global variables for interactive visualization
dataset = None
model = None
current_idx = 21
max_idx = 0
gt_group = None
recovered_group = None
pipeline_group = None
gt_surfaces = {}
recovered_surfaces = {}
pipeline_surfaces = {}
show_gt = True
show_recovered = True
show_pipeline = True
resampled_surfaces = {}
show_resampled = False

# pred_is_closed related variables
pred_is_closed = False
show_closed_colors = True  # Toggle for showing is_closed coloring
model_name = 'vae_v1'  # Default model name
use_fsq = False  # Whether using FSQ-based model
config_args = None  # Config object for FSQ parameters

# Dataset and checkpoint paths
dataset_path = ''
checkpoint_path = ''
canonical = False

# Transformation variables
_rotation_euler = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] in degrees
_apply_rotation = False
_shift_xyz = [0.0, 0.0, 0.0]  # [x, y, z] translation
_apply_shift = False
_scale_factor = 1.0  # uniform scale factor
_apply_scale = False


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



def process_sample(idx):
    """Process a single sample and return both GT and recovered data"""
    global dataset, model, pred_is_closed, use_fsq
    
    if pred_is_closed:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale, is_u_closed_tensor, is_v_closed_tensor = dataset[idx]
    else:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[idx]
        is_u_closed_tensor = None
        is_v_closed_tensor = None
    
    print('processing file: ', dataset.json_names[idx])
    json_path = dataset.json_names[idx]
    
    mask_bool = mask_tensor.bool()
    mask_np = mask_bool.cpu().numpy().astype(bool)
    # Apply mask to get valid surfaces
    valid_params = params_tensor[mask_bool]
    valid_types = types_tensor[mask_bool]
    shift = shift[mask_np]
    rotation = rotation[mask_np]
    scale = scale[mask_np]
    
    # Handle is_closed data
    gt_is_u_closed = None
    gt_is_v_closed = None
    if pred_is_closed and is_u_closed_tensor is not None:
        gt_is_u_closed = is_u_closed_tensor[mask_bool].cpu().numpy()
        gt_is_v_closed = is_v_closed_tensor[mask_bool].cpu().numpy()
    
    # Apply transformations if any are enabled
    if _apply_scale or _apply_rotation or _apply_shift:
        # Convert params to JSON, apply transformations, convert back to params
        transformed_params_list = []
        for i in range(len(valid_params)):
            try:
                # Recover surface from params
                surface_dict = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
                # Apply transformations
                surface_dict = apply_transformation_to_surface(surface_dict)
                # Parse back to params
                transformed_params = dataset._parse_surface(surface_dict, valid_types[i].item())
                transformed_params_list.append(transformed_params)
            except Exception as e:
                print(f"Warning: Failed to transform surface {i} (type: {valid_types[i].item()}): {e}")
                # Use original params if transformation fails
                transformed_params_list.append(valid_params[i].cpu().numpy())
        valid_params = torch.tensor(np.array(transformed_params_list), dtype=valid_params.dtype, device=valid_params.device)
    
    # Load ground truth JSON data
    with open(json_path, 'r') as f:
        gt_json_data = json.load(f)
    
    # Apply transformations to GT for visualization
    if _apply_scale or _apply_rotation or _apply_shift:
        transformed_gt = []
        for surf_idx, surf in enumerate(gt_json_data):
            try:
                surf_copy = copy.deepcopy(surf)
                transformed_surf = apply_transformation_to_surface(surf_copy)
                transformed_gt.append(transformed_surf)
            except Exception as e:
                print(f"Warning: Failed to transform GT surface {surf_idx} (type: {surf.get('type', 'unknown')}): {e}")
                print(f"Surface scalar length: {len(surf.get('scalar', []))}")
                # Use original surface if transformation fails
                transformed_gt.append(surf)
        gt_json_data = transformed_gt

    # print('-' * 10 + 'gt_json_data' + '-' * 10)
    # print(gt_json_data)
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
            print('Difference between mu and z: ', (mu - z).abs().mean())
            # z = mu
        
        if pred_is_closed:
            type_logits_pred, types_pred, is_closed_logits, is_closed_pred = model.classify(z)
            # is_closed_pred shape: [batch, 2], where [:, 0] is u_closed, [:, 1] is v_closed
            pred_is_u_closed = is_closed_pred[:, 0].cpu().numpy()
            pred_is_v_closed = is_closed_pred[:, 1].cpu().numpy()
            
            # Calculate is_closed accuracy if ground truth is available
            if gt_is_u_closed is not None:
                is_closed_gt = torch.stack([torch.from_numpy(gt_is_u_closed), 
                                           torch.from_numpy(gt_is_v_closed)], dim=-1).bool()
                is_closed_correct = (is_closed_pred.cpu() == is_closed_gt).float()
                is_closed_accuracy = is_closed_correct.mean().item()
        else:
            type_logits_pred, types_pred = model.classify(z)
        
        params_pred, mask = model.decode(z, types_pred)
        
        # Calculate metrics
        recon_fn = torch.nn.MSELoss()
        recon_loss = (recon_fn(params_pred, valid_params)) * mask.float().mean()
        accuracy = (types_pred == valid_types).float().mean()
        
        metrics_str = f'Index {idx}: recon_loss: {recon_loss.item():.6f}, accuracy: {accuracy.item():.4f}'
        if is_closed_accuracy is not None:
            metrics_str += f', is_closed_accuracy: {is_closed_accuracy:.4f}'
        print(metrics_str)
        # print(f'Predicted types: {types_pred.cpu().numpy()}')
        # print(f'Ground truth types: {valid_types.cpu().numpy()}')
    
    # Convert predictions to JSON format
    recovered_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())

    if canonical:
        recovered_json_data = [from_canonical(recovered_json_data[i], shift[i], rotation[i], scale[i]) for i in range(len(recovered_json_data))]

    # Convert dataset pipeline surfaces to JSON (parse -> recover -> optional from canonical)
    pipeline_json_data = []
    for i in range(len(valid_params)):
        recovered_surface = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        if canonical:
            recovered_surface = from_canonical(recovered_surface, shift[i], rotation[i], scale[i])
        pipeline_json_data.append(recovered_surface)

    # Prepare is_closed data for return
    is_closed_data = {
        'gt_is_u_closed': gt_is_u_closed,
        'gt_is_v_closed': gt_is_v_closed,
        'pred_is_u_closed': pred_is_u_closed,
        'pred_is_v_closed': pred_is_v_closed,
    }

    return gt_json_data, recovered_json_data, pipeline_json_data, recon_loss.item(), accuracy.item(), is_closed_data

def resample_model(canonical):
    """Generate new samples from the VAE's latent space"""
    global model, resampled_surfaces, current_idx, pred_is_closed, show_closed_colors, use_fsq
    
    if model is None:
        print("Model not loaded yet!")
        return
    
    if pred_is_closed:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale, is_u_closed_tensor, is_v_closed_tensor = dataset[current_idx]
    else:
        params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[current_idx]
    
    valid_params = params_tensor[mask_tensor.bool()]
    valid_types = types_tensor[mask_tensor.bool()]
    shift = shift[mask_tensor.bool()]
    rotation = rotation[mask_tensor.bool()]
    scale = scale[mask_tensor.bool()]
    
    # Apply transformations to input if any are enabled
    if _apply_scale or _apply_rotation or _apply_shift:
        transformed_params_list = []
        for i in range(len(valid_params)):
            try:
                surface_dict = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
                surface_dict = apply_transformation_to_surface(surface_dict)
                transformed_params = dataset._parse_surface(surface_dict, valid_types[i].item())
                transformed_params_list.append(transformed_params)
            except Exception as e:
                print(f"Warning: Failed to transform surface {i} in resample_model: {e}")
                # Use original params if transformation fails
                transformed_params_list.append(valid_params[i].cpu().numpy())
        valid_params = torch.tensor(np.array(transformed_params_list), dtype=valid_params.dtype, device=valid_params.device)
    
    with torch.no_grad():
        # Encode to get latent vectors
        if use_fsq:
            z_quantized, indices = model.encode(valid_params, valid_types)
            z_random = z_quantized
            print(f'Resampled FSQ indices: unique codes = {torch.unique(indices.flatten() if indices.ndim > 1 else indices).numel()}')
        else:
            mu, logvar = model.encode(valid_params, valid_types)
            z_random = model.reparameterize(mu, logvar)
            # z_random = mu
        
        # Classify the latent vectors to get surface types
        if pred_is_closed:
            type_logits_pred, types_pred, is_closed_logits, is_closed_pred = model.classify(z_random)
            pred_is_u_closed = is_closed_pred[:, 0].cpu().numpy()
            pred_is_v_closed = is_closed_pred[:, 1].cpu().numpy()
        else:
            type_logits_pred, types_pred = model.classify(z_random)
            pred_is_u_closed = None
            pred_is_v_closed = None
        
        # Decode to get surface parameters
        params_pred, mask = model.decode(z_random, types_pred)
        
        # print(f"Generated {len(params_pred)} resampled surfaces")
        # print(f"Resampled types: {types_pred.cpu().numpy()}")
        
        # Convert to JSON format
        resampled_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
        if canonical:
            resampled_json_data = [from_canonical(resampled_json_data[i], shift[i], rotation[i], scale[i]) for i in range(len(resampled_json_data))]
        with open('./assets/temp/test_vae_v1_runtime_resampled.json', 'w') as f:
            json.dump(resampled_json_data, f)
        # Visualize resampled surfaces
        resampled_surfaces = visualize_json_interset(resampled_json_data, plot=True, plot_gui=False, tol=1e-5, ps_header='resampled')
        
        # Add to resampled group if it exists
        for i, (surface_key, surface_data) in enumerate(resampled_surfaces.items()):
            if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
                surface_data['ps_handler'].add_to_group(resampled_group)
        
        # Apply is_closed colors to resampled surfaces
        if pred_is_closed and show_closed_colors and pred_is_u_closed is not None:
            apply_closed_colors_to_surfaces(resampled_surfaces, pred_is_u_closed, pred_is_v_closed)
        
        return resampled_json_data, resampled_surfaces

def update_visualization():
    """Update the visualization with current index"""
    global current_idx, gt_group, recovered_group, pipeline_group
    global gt_surfaces, recovered_surfaces, pipeline_surfaces
    global pred_is_closed, show_closed_colors
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    gt_data, recovered_data, pipeline_data, recon_loss, accuracy, is_closed_data = process_sample(current_idx)

    
    # Create groups

    
    # Visualize ground truth surfaces
    try:
        gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-5)
    except ValueError:
        print('GT has wrong visualization data!')
        return
    # print(gt_surfaces)
    for i, (surface_key, surface_data) in enumerate(gt_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to ground truth group
            surface_data['ps_handler'].add_to_group(gt_group)
    
    # Apply is_closed colors to GT surfaces
    if pred_is_closed and show_closed_colors and is_closed_data['gt_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(gt_surfaces, 
                                        is_closed_data['gt_is_u_closed'], 
                                        is_closed_data['gt_is_v_closed'])
    
    # Visualize dataset pipeline ground truth surfaces
    try:
        pipeline_surfaces = visualize_json_interset(pipeline_data, plot=True, plot_gui=False, tol=1e-5, ps_header='dataset_gt')
    except ValueError:
        print('Pipeline GT has wrong visualization data!')
        return
    for i, (surface_key, surface_data) in enumerate(pipeline_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(pipeline_group)
    
    # Apply is_closed colors to pipeline surfaces (use GT is_closed)
    if pred_is_closed and show_closed_colors and is_closed_data['gt_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(pipeline_surfaces, 
                                        is_closed_data['gt_is_u_closed'], 
                                        is_closed_data['gt_is_v_closed'])
    
    # Visualize recovered surfaces  
    try:
        recovered_surfaces = visualize_json_interset(recovered_data, plot=True, plot_gui=False, tol=1e-5, ps_header='z_rec')
    except ValueError as e:
        print('Error: ', e, 'Recovered has wrong visualization data!')
        return
    for i, (surface_key, surface_data) in enumerate(recovered_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to recovered group
            surface_data['ps_handler'].add_to_group(recovered_group)
    
    # Apply is_closed colors to recovered surfaces (use predicted is_closed)
    if pred_is_closed and show_closed_colors and is_closed_data['pred_is_u_closed'] is not None:
        apply_closed_colors_to_surfaces(recovered_surfaces, 
                                        is_closed_data['pred_is_u_closed'], 
                                        is_closed_data['pred_is_v_closed'])
    
    # Configure groups with current visibility settings
    gt_group.set_enabled(show_gt)
    pipeline_group.set_enabled(show_pipeline)
    recovered_group.set_enabled(show_recovered)

    print(f"Visualized {len(gt_surfaces)} GT surfaces, {len(pipeline_surfaces)} dataset GT surfaces and {len(recovered_surfaces)} recovered surfaces")
    
    # Print is_closed statistics if enabled
    if pred_is_closed and is_closed_data['gt_is_u_closed'] is not None:
        gt_u_closed_count = sum(is_closed_data['gt_is_u_closed'])
        gt_v_closed_count = sum(is_closed_data['gt_is_v_closed'])
        print(f"GT is_closed: u={gt_u_closed_count}/{len(is_closed_data['gt_is_u_closed'])}, v={gt_v_closed_count}/{len(is_closed_data['gt_is_v_closed'])}")
    if pred_is_closed and is_closed_data['pred_is_u_closed'] is not None:
        pred_u_closed_count = sum(is_closed_data['pred_is_u_closed'])
        pred_v_closed_count = sum(is_closed_data['pred_is_v_closed'])
        print(f"Pred is_closed: u={pred_u_closed_count}/{len(is_closed_data['pred_is_u_closed'])}, v={pred_v_closed_count}/{len(is_closed_data['pred_is_v_closed'])}")

def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_gt, show_pipeline, show_recovered, show_resampled, resampled_surfaces
    global _rotation_euler, _apply_rotation, _shift_xyz, _apply_shift, _scale_factor, _apply_scale
    global pred_is_closed, show_closed_colors
    
    psim.Text("VAE Surface Reconstruction Test")
    psim.Separator()
    
    # Transformation controls
    psim.Text("=== Transformation Controls (Test Generalization) ===")
    transform_changed = False
    
    # Scale controls
    psim.Text("Scale Controls")
    changed_apply_scale, _apply_scale = psim.Checkbox("Apply Scale", _apply_scale)
    if changed_apply_scale:
        transform_changed = True
    
    if _apply_scale:
        changed_scale, new_scale = psim.SliderFloat("Scale Factor", _scale_factor, 0.1, 3.0)
        if changed_scale:
            _scale_factor = new_scale
            transform_changed = True
        psim.Text(f"Current Scale: {_scale_factor:.3f}")
        if psim.Button("Reset Scale"):
            _scale_factor = 1.0
            transform_changed = True
    
    psim.Separator()
    
    # Rotation controls
    psim.Text("Rotation Controls")
    changed_apply_rot, _apply_rotation = psim.Checkbox("Apply Rotation", _apply_rotation)
    if changed_apply_rot:
        transform_changed = True
    
    if _apply_rotation:
        changed_roll, new_roll = psim.SliderFloat("Roll (X-axis, deg)", _rotation_euler[0], -180.0, 180.0)
        changed_pitch, new_pitch = psim.SliderFloat("Pitch (Y-axis, deg)", _rotation_euler[1], -180.0, 180.0)
        changed_yaw, new_yaw = psim.SliderFloat("Yaw (Z-axis, deg)", _rotation_euler[2], -180.0, 180.0)
        
        if changed_roll or changed_pitch or changed_yaw:
            _rotation_euler = [new_roll, new_pitch, new_yaw]
            transform_changed = True
        
        psim.Text(f"Current: Roll={_rotation_euler[0]:.1f}°, Pitch={_rotation_euler[1]:.1f}°, Yaw={_rotation_euler[2]:.1f}°")
        
        if psim.Button("Reset Rotation"):
            _rotation_euler = [0.0, 0.0, 0.0]
            transform_changed = True
    
    psim.Separator()
    
    # Shift (translation) controls
    psim.Text("Shift Controls")
    changed_apply_shift, _apply_shift = psim.Checkbox("Apply Shift", _apply_shift)
    if changed_apply_shift:
        transform_changed = True
    
    if _apply_shift:
        changed_x, new_x = psim.SliderFloat("Shift X", _shift_xyz[0], -2.0, 2.0)
        changed_y, new_y = psim.SliderFloat("Shift Y", _shift_xyz[1], -2.0, 2.0)
        changed_z, new_z = psim.SliderFloat("Shift Z", _shift_xyz[2], -2.0, 2.0)
        
        if changed_x or changed_y or changed_z:
            _shift_xyz = [new_x, new_y, new_z]
            transform_changed = True
        
        psim.Text(f"Current: X={_shift_xyz[0]:.3f}, Y={_shift_xyz[1]:.3f}, Z={_shift_xyz[2]:.3f}")
        
        if psim.Button("Reset Shift"):
            _shift_xyz = [0.0, 0.0, 0.0]
            transform_changed = True
    
    psim.Separator()
    
    # Reset all transformations
    if psim.Button("Reset All Transformations"):
        _scale_factor = 1.0
        _rotation_euler = [0.0, 0.0, 0.0]
        _shift_xyz = [0.0, 0.0, 0.0]
        transform_changed = True
    
    if transform_changed:
        update_visualization()
    
    psim.Separator()
    
    # Index controls
    slider_changed, slider_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    input_changed, input_idx = psim.InputInt("Go To Index", current_idx)
    if input_changed:
        input_idx = max(0, min(max_idx, input_idx))
        if input_idx != current_idx:
            current_idx = input_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Resample button
    psim.Separator()
    psim.Text("Model Controls:")
    if psim.Button("Resample Model"):
        #  This may lead to some unexpected crash, temporarily remove all structures instead
        # for surface in resampled_surfaces.values():
        #     surface['ps_handler'].remove()
        ps.remove_all_structures()
        resampled_surfaces = {}
        resampled_json_data, resampled_surfaces = resample_model(canonical)
    
    # Group controls
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
        
        # Resampled group control
        changed, show_resampled = psim.Checkbox("Show Resampled", show_resampled)
        if changed:
            resampled_group.set_enabled(show_resampled)
        
        # is_closed color controls
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
    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--canonical', type=bool, default=False, help='Whether to use canonical dataset')
    parser.add_argument('--config', type=str, default='', help='Path to config file (optional, to read pred_is_closed)')
    parser.add_argument('--pred_is_closed', action='store_true', help='Enable is_closed prediction (overrides config)')
    parser.add_argument(
        '--rotation',
        type=float,
        nargs=3,
        default=None,
        metavar=('ROLL', 'PITCH', 'YAW'),
        help='Initial Euler rotation angles in degrees [roll, pitch, yaw].',
    )
    parser.add_argument(
        '--apply_rotation',
        action='store_true',
        help='Apply rotation by default.',
    )
    parser.add_argument(
        '--shift',
        type=float,
        nargs=3,
        default=None,
        metavar=('X', 'Y', 'Z'),
        help='Initial translation shift [x, y, z].',
    )
    parser.add_argument(
        '--apply_shift',
        action='store_true',
        help='Apply shift by default.',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=None,
        help='Initial uniform scale factor.',
    )
    parser.add_argument(
        '--apply_scale',
        action='store_true',
        help='Apply scale by default.',
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    canonical = args.canonical
    
    # Load pred_is_closed, model_name, and FSQ parameters from config file if provided


    config = OmegaConf.load(args.config)
    # config_args = NestedDictToClass(cfg)
    dataset = load_dataset_from_config(config, section='data_val')
    max_idx = len(dataset) - 1
    model = load_model_from_config(config)

    pred_is_closed = config.data_val['params']['detect_closed']
    use_fsq = 'fsq' in config.model.name.lower()
    print(f"Using FSQ: {use_fsq}")
    print(f"Pred is closed: {pred_is_closed}")


    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        if 'ema_model' in checkpoint or 'ema' in checkpoint:
            ema_key = 'ema' if 'ema' in checkpoint else 'ema_model'
            ema_model = checkpoint[ema_key]
            ema_model = {k.replace("ema_model.", "").replace("ema.", ""): v for k, v in ema_model.items()}
            model.load_state_dict(ema_model, strict=False)
            print("Loaded DiT EMA model weights.")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Loaded DiT model weights.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded DiT raw model state_dict.")
    
    model.eval()


    
    # Command line argument overrides config
    if args.pred_is_closed:
        pred_is_closed = True
        print("pred_is_closed enabled via command line argument")
    
    # Set initial transformations from command line arguments
    if args.scale is not None:
        _scale_factor = args.scale
        print(f"Initial scale set to: {_scale_factor}")
    
    if args.apply_scale:
        _apply_scale = True
        print("Scale is enabled by default.")
    
    if args.rotation is not None:
        _rotation_euler = list(args.rotation)
        print(f"Initial rotation set to: Roll={_rotation_euler[0]}°, Pitch={_rotation_euler[1]}°, Yaw={_rotation_euler[2]}°")
    
    if args.apply_rotation:
        _apply_rotation = True
        print("Rotation is enabled by default.")
    
    if args.shift is not None:
        _shift_xyz = list(args.shift)
        print(f"Initial shift set to: X={_shift_xyz[0]}, Y={_shift_xyz[1]}, Z={_shift_xyz[2]}")
    
    if args.apply_shift:
        _apply_shift = True
        print("Shift is enabled by default.")
    
    # Initialize

    
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
