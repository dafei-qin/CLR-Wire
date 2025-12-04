import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.import_tools import load_model_from_config, load_dataset_from_config
from src.utils.surface_tools import params_to_samples, recover_surface_from_params

def decode_and_sample(model, latent_params):
    """
    Decode latent parameters, sample surfaces per surface type, and return samples in
    the same order as `latent_params`. Per-type processing keeps gradients intact.
    """
    class_logits, surface_type_pred, is_closed_logits, is_closed = model.classify(latent_params)

    params_raw_recon, mask = model.decode(latent_params, surface_type_pred)

    ordered_samples = None
    for surface_type in surface_type_pred.unique():
        type_mask = surface_type_pred == surface_type
        params_raw_recon_per_type = params_raw_recon[type_mask]
        samples = params_to_samples(params_raw_recon_per_type, surface_type, 8, 8)

        if ordered_samples is None:
            sample_shape = samples.shape[1:]
            ordered_samples = samples.new_zeros((latent_params.shape[0],) + sample_shape)

        ordered_samples[type_mask] = samples

    if ordered_samples is None:
        return torch.empty(0, device=latent_params.device, dtype=latent_params.dtype)

    return ordered_samples

def decode_and_sample_with_rts(model, latent_params, shifts, rotations, scales):
    samples = decode_and_sample(model, latent_params)
    X = rotations[..., :3] / (torch.norm(rotations[..., :3], dim=-1, keepdim=True) + 1e-8)
    Y = rotations[..., 3:6] / (torch.norm(rotations[..., 3:6], dim=-1, keepdim=True) + 1e-8)
    Z = torch.cross(X, Y, dim=-1)
    rotation_matrix = torch.stack([X, Y, Z], dim=-1)[:, None, None] # (B, 1, 1, 3, 3)
    samples = (samples[:, :, :, None] @ rotation_matrix)[:, :, :, 0] * scales[:, None, None] + shifts[:, None, None]

    return samples

if __name__ == "__main__":

    from omegaconf import OmegaConf
    cfg = OmegaConf.load('src/configs/train_vae_v1_canonical_logvar_l2norm_pred_closed.yaml')

    model, _ = load_model_from_config(cfg)
    cfg_dataset = OmegaConf.load('src/configs/dit/dit_simple_surface_1204_large_rts.yaml')
    dataset = load_dataset_from_config(cfg_dataset, section='data_val')
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
        samples = decode_and_sample_with_rts(model, latent_params, shifts_tensor, rotations_tensor, scales_tensor)
        print(samples.shape)

