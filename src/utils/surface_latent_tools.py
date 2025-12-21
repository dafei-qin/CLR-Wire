import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.utils.surface_tools import params_to_samples, recover_surface_from_params


def decode_latent(sample: torch.Tensor, log_scale=True):

    valid_tensor = sample[..., 0]
    sample = sample[..., 1:]
    shifts_tensor = sample[..., :3]
    rotations_tensor = sample[..., 3:3+6]
    scales_tensor = sample[..., 3+6:3+6+1]
    if log_scale:
        scales_tensor = torch.exp(scales_tensor)

    params_tensor = sample[..., 3+6+1:]
    valid = torch.sigmoid(valid_tensor) > 0.5
    return valid, shifts_tensor, rotations_tensor, scales_tensor, params_tensor

    
def decode_and_sample(model, latent_params, num_samples=8):
    """
    Decode latent parameters, sample surfaces per surface type, and return samples in
    the same order as `latent_params`. Per-type processing keeps gradients intact.
    """
    assert not torch.isnan(latent_params).any(), "latent_params contains inf/nan" + str(latent_params)
    class_logits, surface_type_pred, is_closed_logits, is_closed = model.classify(latent_params)

    params_raw_recon, mask = model.decode(latent_params, surface_type_pred)
    assert not torch.isnan(params_raw_recon).any(), "params_raw_recon contains inf/nan" + str(params_raw_recon)
    ordered_samples = None
    for surface_type in surface_type_pred.unique():
        type_mask = surface_type_pred == surface_type
        params_raw_recon_per_type = params_raw_recon[type_mask]
        
        samples = params_to_samples(params_raw_recon_per_type, surface_type, num_samples, num_samples)

        if ordered_samples is None:
            sample_shape = samples.shape[1:]
            ordered_samples = samples.new_zeros((latent_params.shape[0],) + sample_shape)

        ordered_samples[type_mask] = samples

    if ordered_samples is None:
        return torch.empty(0, device=latent_params.device, dtype=latent_params.dtype)

    return ordered_samples

def decode_only(model, latent_params):
    """
    Decode latent parameters, sample surfaces per surface type, and return samples in
    the same order as `latent_params`. Per-type processing keeps gradients intact.
    """
    assert not torch.isnan(latent_params).any(), "latent_params contains inf/nan" + str(latent_params)
    class_logits, surface_type_pred, is_closed_logits, is_closed = model.classify(latent_params)

    params_raw_recon, mask = model.decode(latent_params, surface_type_pred)
    assert not torch.isnan(params_raw_recon).any(), "params_raw_recon contains inf/nan" + str(params_raw_recon)
    ordered_params = None
    # Class of params: Location, Direction, uv, scalar
    # Note: For bspline_surface, scalar has 48 dims (control points), so we need max(2, 48) = 48
    max_scalar_dim = 48  # To accommodate bspline_surface control points
    ordered_locations = torch.zeros((params_raw_recon.shape[0], 3), device=params_raw_recon.device, dtype=params_raw_recon.dtype)
    ordered_directions = torch.zeros((params_raw_recon.shape[0], 3, 3), device=params_raw_recon.device, dtype=params_raw_recon.dtype)
    ordered_uvs = torch.zeros((params_raw_recon.shape[0], 4), device=params_raw_recon.device, dtype=params_raw_recon.dtype)
    ordered_scalars = torch.zeros((params_raw_recon.shape[0], max_scalar_dim), device=params_raw_recon.device, dtype=params_raw_recon.dtype)

    for surface_type in surface_type_pred.unique():
        type_mask = surface_type_pred == surface_type
        params_raw_recon_per_type = params_raw_recon[type_mask]
        
        surface_params = recover_surface_from_params(params_raw_recon_per_type, surface_type)
        # samples = params_to_samples(params_raw_recon_per_type, surface_type, num_samples, num_samples)

        # type_mask = type_mask.unsqueeze(-1)
        # #ordered_locations.masked_scatter(type_mask, surface_params['location'])
        ordered_locations[type_mask] = surface_params['location']

        # ordered_directions.masked_scatter(type_mask.unsqueeze(-1), surface_params['direction'])
        ordered_directions[type_mask] = surface_params['direction']
        # ordered_uvs.masked_scatter(type_mask, surface_params['uv'])
        ordered_uvs[type_mask] = surface_params['uv']
        
        # Handle different surface types
        if surface_type == 0:  # plane: No scalar
            pass
        elif surface_type == 5:  # bspline_surface: 48D control points
            ordered_scalars[type_mask, :48] = surface_params['scalar']
        elif surface_type == 2 or surface_type == 4:  # cone, torus: 2D scalar
            # ordered_scalars.masked_scatter(type_mask, surface_params['scalar'])
            ordered_scalars[type_mask, :2] = surface_params['scalar']
        elif surface_type == 1 or surface_type == 3:  # cylinder, sphere: 1D scalar (pad to 2D)
            surface_params_scalar_padded = torch.cat([
                surface_params['scalar'], 
                torch.zeros((surface_params['scalar'].shape[0], 1), 
                           device=surface_params['scalar'].device, 
                           dtype=surface_params['scalar'].dtype)
            ], dim=-1)
            ordered_scalars[type_mask, :2] = surface_params_scalar_padded


        
    ordered_directions = ordered_directions[..., :2].reshape(ordered_directions.shape[0], -1)
    ordered_params = torch.cat([ordered_locations, ordered_directions, ordered_uvs, ordered_scalars], dim=-1)


    return ordered_params

def safe_normalize(v, dim=-1, eps=1e-6):
    norm = torch.norm(v, dim=dim, keepdim=True)
    norm = torch.maximum(norm, torch.ones_like(norm) * eps)
    return v / norm

def decode_and_sample_with_rts(model, latent_params, shifts, rotations, scales, log_scale=False, num_samples=8):
    
    samples = decode_and_sample(model, latent_params, num_samples=num_samples)
    assert not torch.isnan(samples).any(), 'samples is nan'
    if log_scale:
        scales = torch.exp(scales)
    # Could be here's problem?

    X = safe_normalize(rotations[..., :3])
    Y = safe_normalize(rotations[..., 3:6])
    Z = torch.cross(X, Y, dim=-1)
    Z = safe_normalize(Z)
    assert not torch.isnan(X).any(), 'X is nan'
    assert not torch.isnan(Y).any(), 'Y is nan'
    assert not torch.isnan(Z).any(), 'Z is nan'
    rotation_matrix = torch.stack([X, Y, Z], dim=-1)[:, None, None] # (B, 1, 1, 3, 3)
    # samples = (samples[:, :, :, None] @ rotation_matrix )[:, :, :, 0] * scales[:, None, None] + shifts[:, None, None]
    samples = (rotation_matrix @ samples[..., None]  )[..., 0] * scales[:, None, None] + shifts[:, None, None]
    assert not torch.isnan(samples).any(), 'samples after RTS is nan'

    return samples

if __name__ == "__main__":

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('src/configs/train_vae_v1_canonical_logvar_l2norm_pred_closed.yaml')

    model, _ = load_model_from_config(cfg)
    cfg_dataset = OmegaConf.load('src/configs/dit/dit_simple_surface_1204_large_rts.yaml')
    dataset = load_dataset_from_config(cfg_dataset, section='data_val')
    log_scale = cfg_dataset.params.log_scale
    for idx in range(len(dataset)):
        (
        latent_params_tensor,
            rotations_tensor,
            scales_tensor,
            shifts_tensor,
            classes_tensor,
            bbox_mins_tensor,
            bbox_maxs_tensor,
            mask_tensor,
            pc_tensor
        ) = dataset[idx]

        latent_params = latent_params_tensor[mask_tensor.bool()]
        rotations_tensor = rotations_tensor[mask_tensor.bool()]
        scales_tensor = scales_tensor[mask_tensor.bool()]
        shifts_tensor = shifts_tensor[mask_tensor.bool()]
        samples = decode_and_sample_with_rts(model, latent_params, shifts_tensor, rotations_tensor, scales_tensor, log_scale=log_scale)
        print(samples.shape)

