from beartype import beartype
from typing import Callable, Optional
from torch.utils.data import Dataset
from typing import Optional, Union
from diffusers import ConfigMixin, ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import wandb
from einops import rearrange
from functools import partial
from contextlib import nullcontext
import time
from collections import defaultdict

from src.flow.surface_flow import ZLDMPipeline, get_new_scheduler
from src.vae.layers import BSplineSurfaceLayer
from src.trainer.trainer_base import BaseTrainer
from src.utils.helpers import divisible_by, get_current_time, get_lr
from src.utils.surface_latent_tools import decode_and_sample_with_rts, decode_latent, decode_only


def has_nan_or_inf(t):
    return not torch.isfinite(t).all()


# Trainer for DIT with sampled point cloud conditions
class TrainerDITSampleSegment(BaseTrainer):
    @beartype
    def __init__(
        self,
        model: Union[ModelMixin, ConfigMixin] | nn.Module,
        vae: Union[ModelMixin, ConfigMixin] | nn.Module,
        dataset: Dataset, 
        val_dataset: Dataset,
        *,
        batch_size = 16,
        checkpoint_folder: str = './checkpoints',
        checkpoint_every_step: int = 1000,
        checkpoint_file_name: str = 'model.pt',
        ema_update_every = 10,
        ema_decay = 0.995,
        grad_accum_every = 1,
        log_every_step: int = 10,
        learning_rate: float = 2e-4,
        mixed_precision_type = 'fp16',
        max_grad_norm: float = 1.,
        num_workers: int = 1,
        num_train_steps = 100000,
        num_step_per_epoch = 100,
        resume_training = False,
        load_checkpoint_from_file = None,
        from_start = False,
        use_wandb_tracking: bool = False,
        collate_fn: Optional[Callable] = None,
        val_every_step: int = 1000,
        val_num_batches: int = 10,
        accelerator_kwargs: dict = dict(),
        visual_eval_every_step: int = 5000,
        num_visual_samples: int = 4,
        scheduler_type: str = 'ddpm',
        prediction_type: str = 'v_prediction',
        num_training_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        weight_valid = 1.0,
        weight_params = 1.0,
        weight_rotations = 1.0,
        weight_scales = 1.0,
        weight_shifts = 1.0,
        weight_original_sample = 1.0,
        original_sample_start_step = 0,
        weight_sample_edges = 0.0,
        use_weighted_sample_loss=False,
        weight_surf_params = 0.0,
        log_scale=True,
        latent_dim=128,
        **kwargs
    ):
        super().__init__(
            model=model, 
            dataset=dataset,
            val_dataset=val_dataset,
            other_models = [vae],
            batch_size=batch_size,
            checkpoint_folder=checkpoint_folder,
            checkpoint_every_step=checkpoint_every_step,
            checkpoint_file_name=checkpoint_file_name,
            ema_update_every=ema_update_every,
            ema_decay=ema_decay,
            grad_accum_every=grad_accum_every,
            log_every_step=log_every_step,
            learning_rate=learning_rate,
            mixed_precision_type=mixed_precision_type,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            num_train_steps=num_train_steps,
            num_step_per_epoch=num_step_per_epoch,
            resume_training=resume_training,
            from_start=from_start,
            load_checkpoint_from_file=load_checkpoint_from_file,
            collate_fn=collate_fn,
            use_wandb_tracking=use_wandb_tracking,
            accelerator_kwargs=accelerator_kwargs,
            val_every_step=val_every_step,
            val_num_batches=val_num_batches,
            **kwargs
        )
        
        self.num_visual_samples = num_visual_samples
        self.scheduler = get_new_scheduler(prediction_type, num_training_timesteps)
        self.num_inference_timesteps = num_inference_timesteps
        self.pipe = ZLDMPipeline(self.model, self.scheduler, dtype=torch.float32)

        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.vae = self.other_models[0]
        self.vae = self.vae.module if hasattr(self.vae, 'module') else self.vae

        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.weight_valid = weight_valid
        self.weight_params = weight_params
        self.weight_rotations = weight_rotations
        self.weight_scales = weight_scales
        self.weight_shifts = weight_shifts 
        self.weight_original_sample = weight_original_sample
        self.use_weighted_sample_loss = use_weighted_sample_loss
        self.original_sample_start_step = original_sample_start_step
        self.weight_sample_edges = weight_sample_edges
        self.weight_surf_params = weight_surf_params
        
        self.log_scale = log_scale
        self.latent_dim = latent_dim
    
    def encode_params_to_latents(self, params_padded, surface_type, masks):
        """
        Encode surface parameters to latents using VAE (without gradient).
        
        Args:
            params_padded: (B, num_max_pad, param_dim) Surface parameters
            masks: (B, num_max_pad) Binary mask
            
        Returns:
            latents_padded: (B, num_max_pad, latent_dim) Latent representations
        """
        masks_bool = masks.bool()
        valid_indices = torch.nonzero(masks_bool, as_tuple=False)
        
        if valid_indices.numel() > 0:
            # Extract valid surfaces only
            valid_params = params_padded[masks_bool]
            surface_type = surface_type[masks_bool]
            # Encode to latents (no gradient)

            mu, var = self.vae.encode(valid_params, surface_type)
            valid_latents = mu
            # Create padded output
            B, num_max_pad = masks_bool.shape
            latents_padded = torch.zeros(
                B, num_max_pad, self.latent_dim,
                device=valid_latents.device,
                dtype=valid_latents.dtype
            )
            
            # Place valid latents back
            batch_indices = valid_indices[:, 0]
            pad_indices = valid_indices[:, 1]
            latents_padded[batch_indices, pad_indices] = valid_latents
        else:
            B, num_max_pad = masks_bool.shape
            latents_padded = torch.zeros(
                B, num_max_pad, self.latent_dim,
                device=params_padded.device,
                dtype=params_padded.dtype
            )
        
        return latents_padded

    def decode_latents_to_params(self, latents_padded, masks):
        """
        Decode latents to surface parameters using VAE.
        
        Args:
            latents_padded: (B, num_max_pad, latent_dim) Latent representations
            masks: (B, num_max_pad) Binary mask
            
        Returns:
            params_padded: (B, num_max_pad, param_dim) Surface parameters
        """
        masks_bool = masks.bool()
        valid_indices = torch.nonzero(masks_bool, as_tuple=False)
        
        if valid_indices.numel() > 0:
            # Extract valid latents
            valid_latents = latents_padded[masks_bool]
            
            # Decode to params
            valid_params = self.vae.decode(valid_latents)
            
            # Create padded output
            B, num_max_pad = masks_bool.shape
            param_dim = valid_params.shape[-1]
            params_padded = torch.zeros(
                B, num_max_pad, param_dim,
                device=valid_params.device,
                dtype=valid_params.dtype
            )
            
            # Place valid params back
            batch_indices = valid_indices[:, 0]
            pad_indices = valid_indices[:, 1]
            params_padded[batch_indices, pad_indices] = valid_params
        else:
            B, num_max_pad = masks_bool.shape
            # Assuming param_dim is known, use a default or get from vae
            param_dim = 19  # base_dim (17) + max_scalar_dim (2)
            params_padded = torch.zeros(
                B, num_max_pad, param_dim,
                device=latents_padded.device,
                dtype=latents_padded.dtype
            )
        
        return params_padded

    def decode_valid_surfaces_with_padding(
        self,
        params_padded: torch.Tensor,
        shifts_padded: torch.Tensor,
        rotations_padded: torch.Tensor,
        scales_padded: torch.Tensor,
        masks: torch.Tensor,
        num_samples: int = 8,
    ) -> torch.Tensor:
        """
        Decode and sample only valid (non-padded) surfaces based on masks.
        
        Args:
            params_padded: (B, num_max_pad, ...) Padded surface parameters
            shifts_padded: (B, num_max_pad, 3) Padded translation vectors
            rotations_padded: (B, num_max_pad, 6) Padded rotation matrices (6D representation)
            scales_padded: (B, num_max_pad, 1) Padded scale factors
            masks: (B, num_max_pad) or (B, num_max_pad, 1) Binary mask
            num_samples: Number of sample points (H=W)
            
        Returns:
            gt_sampled_points: (B, num_max_pad, H*W, 3) Sampled points with padding
        """
        # Ensure masks is 2D (B, num_max_pad)
        if masks.dim() == 3:
            masks_2d = masks.squeeze(-1)
        else:
            masks_2d = masks
        masks_bool = masks_2d.bool()
        
        valid_indices = torch.nonzero(masks_bool, as_tuple=False)
        
        if valid_indices.numel() > 0:
            # Extract valid surfaces only
            valid_params = params_padded[masks_bool]
            valid_shifts = shifts_padded[masks_bool]
            valid_rotations = rotations_padded[masks_bool]
            valid_scales = scales_padded[masks_bool]
            
            # Decode and sample
            valid_sampled_points = decode_and_sample_with_rts(
                self.vae, valid_params, valid_shifts, valid_rotations, valid_scales, 
                log_scale=self.log_scale, num_samples=num_samples
            )
            valid_sampled_points = valid_sampled_points.reshape(valid_sampled_points.shape[0], -1, 3)
            
            # Create padded output
            B, num_max_pad = masks_bool.shape
            num_points = valid_sampled_points.shape[1]
            gt_sampled_points = torch.zeros(
                B, num_max_pad, num_points, 3,
                device=valid_sampled_points.device,
                dtype=valid_sampled_points.dtype
            )
            
            # Place valid results back
            batch_indices = valid_indices[:, 0]
            pad_indices = valid_indices[:, 1]
            gt_sampled_points[batch_indices, pad_indices] = valid_sampled_points
        else:
            B, num_max_pad = masks_bool.shape
            gt_sampled_points = torch.zeros(
                B, num_max_pad, num_samples * num_samples, 3,
                device=params_padded.device,
                dtype=params_padded.dtype
            )
        
        return gt_sampled_points

    def log_loss(self, total_loss, lr, total_norm, step, loss_dict={}):
        """Enhanced loss logging"""
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
            **loss_dict
        }
        
        self.log(**log_dict)
        
        loss_str = f'loss: {total_loss:.3f}, loss_valid: {loss_dict["loss_valid"]:.3f}, loss_shifts: {loss_dict["loss_shifts"]:.3f}, loss_rotations: {loss_dict["loss_rotations"]:.3f}, loss_scales: {loss_dict["loss_scales"]:.3f}, loss_params: {loss_dict["loss_params"]:.3f}, loss_orig_sample: {loss_dict["loss_orig_sample"]:.3f}, loss_edges: {loss_dict["loss_edges"]:.3f}, loss_surf_params: {loss_dict["loss_surf_params"]:.3f}'

        self.print(get_current_time() + f' {loss_str} lr: {lr:.6f} norm: {total_norm:.3f}')

    def compute_loss(self, output, target, masks):
        loss_raw = torch.nn.functional.mse_loss(output, target, reduction='none')
        loss_valid = loss_raw[..., :1].mean()
        
        loss_others = loss_raw[..., 1:] * masks.float()
        total_valid_surfaces = masks.float().sum()
        loss_shifts = loss_others[..., :3].mean(dim=(2)).sum() / total_valid_surfaces
        loss_rotations = loss_others[..., 3:3+6].mean(dim=(2)).sum() / total_valid_surfaces
        loss_scales = loss_others[..., 3+6:3+6+1].mean(dim=(2)).sum() / total_valid_surfaces
        loss_params = loss_others[..., 3+6+1:].mean(dim=(2)).sum() / total_valid_surfaces
        
        return loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params

    def train(self):
        """Training loop with sampled point cloud conditions"""
        step = self.step.item()
        
        self.start_profiler()

        while step < self.num_train_steps:
            total_loss = 0.

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)
                forward_kwargs = [_[forward_kwargs[-1]] for _ in forward_kwargs[:-1]]
                with self.accelerator.autocast(), maybe_no_sync():
                    # Unpack data: sampled_points, shifts, rotations, scales, params, types, mask
                    sampled_points, shifts_padded, rotations_padded, scales_padded, params_padded, surface_type, masks = forward_kwargs
                    
                    # Reshape sampled_points from (B, N_Surface, N_points, 6) to (B, N_Surface, N_points*6)
                    B, N_Surface, N_points, feat_dim = sampled_points.shape
                    pc_cond = sampled_points.reshape(B, N_Surface, N_points * feat_dim)

                    # Prepare data with no gradient
                    with torch.no_grad():
                        # Encode params to latents using VAE (no gradient)
                        latents_padded = self.encode_params_to_latents(params_padded, surface_type, masks)
                        
                        # Compute ground truth for validation
                        gt_valid_surface_params = decode_only(self.vae, latents_padded[masks.bool()])
                        
                        # Compute sampled points for sample loss
                        gt_sampled_points = self.decode_valid_surfaces_with_padding(
                            latents_padded, shifts_padded, rotations_padded, scales_padded, masks, num_samples=8
                        )
                        
                        # Compute surface weights
                        surface_max = torch.max(gt_sampled_points, dim=(2))[0]
                        surface_min = torch.min(gt_sampled_points, dim=(2))[0]
                        surface_weight = torch.max((surface_max - surface_min), dim=-1)[0]
                        surface_weight = torch.clamp(1 / (surface_weight + 1e-3), min=1.0)[masks.bool()]
                        
                        B, num_max_pad = gt_sampled_points.shape[:2]
                        gt_sampled_points = gt_sampled_points.reshape(B, num_max_pad, -1)
                        
                        gt_sampled_edges = self.decode_valid_surfaces_with_padding(
                            latents_padded, shifts_padded, rotations_padded, scales_padded, masks, num_samples=2
                        )
                        gt_sampled_edges = gt_sampled_edges.reshape(B, num_max_pad, -1)

                    # Prepare ground truth sample: [mask, shifts, rotations, scales, latents]
                    gt_sample = torch.cat([
                        masks.unsqueeze(-1).float(), 
                        shifts_padded, 
                        rotations_padded, 
                        scales_padded, 
                        latents_padded
                    ], dim=-1)
                    
                    # Add noise
                    noise = torch.randn_like(gt_sample)
                    timesteps = torch.randint(
                        0, self.scheduler.config.num_train_timesteps, 
                        (gt_sample.shape[0],), 
                        device=gt_sample.device
                    ).long()

                    noisy_sample = self.scheduler.add_noise(gt_sample, noise, timesteps)

                    if self.scheduler.config.prediction_type == 'v_prediction':
                        target = self.scheduler.get_velocity(gt_sample, noise, timesteps)
                    elif self.scheduler.config.prediction_type == 'sample':
                        target = gt_sample
                    elif self.scheduler.config.prediction_type == 'epsilon':
                        target = noise

                    # Forward pass
                    masks_input = masks.unsqueeze(-1)
                    assert not torch.isnan(noisy_sample).any(), "noisy_sample contains nan"
                    assert not torch.isnan(pc_cond).any(), "pc_cond contains nan"
                    
                    output = self.model(
                        sample=noisy_sample, 
                        timestep=timesteps, 
                        cond=pc_cond, 
                        tgt_key_padding_mask=~masks_input.bool().squeeze(-1), 
                        memory_key_padding_mask=~masks_input.bool().squeeze(-1)
                    )

                    # Compute losses
                    loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = self.compute_loss(
                        output, target, masks_input
                    )
                    
                    # Recover x0 and compute sample losses
                    if self.scheduler.config.prediction_type == 'v_prediction':
                        alpha_prod_t = self.scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
                        beta_prod_t = 1 - alpha_prod_t
                        pred_original_sample = (alpha_prod_t**0.5).unsqueeze(1).unsqueeze(1) * noisy_sample - \
                                              (beta_prod_t**0.5).unsqueeze(1).unsqueeze(1) * output
                    elif self.scheduler.config.prediction_type == 'sample':
                        pred_original_sample = output
                    else:
                        raise ValueError(f'Unsupported prediction type: {self.scheduler.config.prediction_type}')

                    # Compute original sample loss
                    if step >= self.original_sample_start_step:
                        pred_original_sample_valid = pred_original_sample[masks_input.squeeze(-1).bool()]
                        valid, shifts, rotations, scales, latents = decode_latent(
                            pred_original_sample_valid, log_scale=self.log_scale
                        )
                        
                        # Decode latents to params
                        # params = self.vae.decode(latents)
                        params = latents
                        
                        # Sample points from predicted params
                        pred_sampled_points = decode_and_sample_with_rts(
                            self.vae, params, shifts, rotations, scales, 
                            log_scale=False, num_samples=8
                        )
                        pred_sampled_edges = decode_and_sample_with_rts(
                            self.vae, params, shifts, rotations, scales, 
                            log_scale=False, num_samples=2
                        )
                        
                        # Compute sample loss
                        loss_original_sample = torch.nn.functional.mse_loss(
                            pred_sampled_points.reshape(pred_sampled_points.shape[0], -1), 
                            gt_sampled_points[masks_input.squeeze(-1).bool()], 
                            reduction='none'
                        )
                        if self.use_weighted_sample_loss:
                            loss_original_sample = loss_original_sample * surface_weight.unsqueeze(-1)
                        loss_original_sample = loss_original_sample.mean()

                        # Edge loss
                        loss_edges = torch.nn.functional.mse_loss(
                            pred_sampled_edges.reshape(pred_sampled_edges.shape[0], -1), 
                            gt_sampled_edges[masks_input.squeeze(-1).bool()]
                        )

                        # Surface params loss
                        pred_valid_surface_params = decode_only(self.vae, params)
                        loss_surf_params = torch.nn.functional.mse_loss(
                            pred_valid_surface_params, gt_valid_surface_params
                        )
                    else:
                        loss_original_sample = torch.tensor(0.0, device=output.device)
                        loss_edges = torch.tensor(0.0, device=output.device)
                        loss_surf_params = torch.tensor(0.0, device=output.device)

                    # Total loss
                    loss = (loss_valid * self.weight_valid + 
                            loss_shifts * self.weight_shifts + 
                            loss_rotations * self.weight_rotations + 
                            loss_scales * self.weight_scales + 
                            loss_params * self.weight_params + 
                            loss_original_sample * self.weight_original_sample * (step >= self.original_sample_start_step) + 
                            loss_edges * self.weight_sample_edges * (step >= self.original_sample_start_step) + 
                            loss_surf_params * self.weight_surf_params * (step >= self.original_sample_start_step))
                    
                    total_loss += loss.item()
                    loss_dict = {
                        'loss_valid': loss_valid.item(),
                        'loss_shifts': loss_shifts.item(),
                        'loss_rotations': loss_rotations.item(),
                        'loss_scales': loss_scales.item(),
                        'loss_params': loss_params.item(),
                        'loss_orig_sample': loss_original_sample.item() if step >= self.original_sample_start_step else 0.0,
                        'loss_edges': loss_edges.item() if step >= self.original_sample_start_step else 0.0,
                        'loss_surf_params': loss_surf_params.item() if step >= self.original_sample_start_step else 0.0
                    }
                
                self.accelerator.backward(loss)

            # Check for nan/inf gradients
            for name, p in self.raw_model.named_parameters():
                if p.grad is None:
                    continue
                if has_nan_or_inf(p.grad):
                    print(f"[NaN/Inf grad] {name}")
                    print(p.grad)
                    print('loss: ', loss)
                    raise RuntimeError("Invalid grad detected")

            self.optimizer.step()

            # Check for nan/inf parameters
            for name, p in self.raw_model.named_parameters():
                if has_nan_or_inf(p.data):
                    print(f"[NaN/Inf param after step] {name}")
                    raise RuntimeError("Invalid parameter update")

            self.optimizer.zero_grad()

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                self.log_loss(total_loss, cur_lr, total_norm, step, loss_dict)

            step += 1
            self.step.add_(1)
            
            self.profiler_step()
            self.wait()
            
            if self.is_main:
                self.ema.update()

            self.wait()

            # Validation
            if self.is_main and divisible_by(step, self.val_every_step):
                with torch.no_grad():
                    print(f"Validating at step {step}")
                    total_val_loss = 0.
                    loss_dict = defaultdict(float)
                    self.ema.eval()
                    self.pipe.denoiser = self.ema
                    device = next(self.model.parameters()).device
                    num_val_batches = self.val_num_batches * self.grad_accum_every

                    for _ in range(num_val_batches):
                        with self.accelerator.autocast(), torch.no_grad():
                            forward_args = self.next_data_to_forward_kwargs(self.val_dl_iter)
                            forward_args = [_[forward_args[-1]] for _ in forward_args[:-1]]
                            forward_args = [_.to(device) for _ in forward_args]
                            sampled_points, shifts_padded, rotations_padded, scales_padded, params_padded, surface_type, masks = forward_args
                            
                            # Reshape sampled_points
                            B, N_Surface, N_points, feat_dim = sampled_points.shape
                            pc_cond = sampled_points.reshape(B, N_Surface, N_points * feat_dim)
                            
                            # Encode params to latents
                            latents_padded = self.encode_params_to_latents(params_padded, surface_type, masks)
                            
                            # Prepare ground truth
                            gt_valid_surface_params = decode_only(self.vae, latents_padded[masks.bool()])
                            
                            gt_sampled_points = self.decode_valid_surfaces_with_padding(
                                latents_padded, shifts_padded, rotations_padded, scales_padded, masks, num_samples=8
                            )
                            
                            surface_max = torch.max(gt_sampled_points, dim=(2))[0]
                            surface_min = torch.min(gt_sampled_points, dim=(2))[0]
                            surface_weight = torch.max((surface_max - surface_min), dim=-1)[0]
                            surface_weight = torch.clamp(1 / (surface_weight + 1e-3), min=1.0)[masks.bool()]
                            
                            B, num_max_pad = gt_sampled_points.shape[:2]
                            gt_sampled_points = gt_sampled_points.reshape(B, num_max_pad, -1)
                            
                            gt_sampled_edges = self.decode_valid_surfaces_with_padding(
                                latents_padded, shifts_padded, rotations_padded, scales_padded, masks, num_samples=2
                            )
                            gt_sampled_edges = gt_sampled_edges.reshape(B, num_max_pad, -1)

                            masks_input = masks.unsqueeze(-1)
                            gt_sample = torch.cat([
                                masks_input.float(), 
                                shifts_padded, 
                                rotations_padded, 
                                scales_padded, 
                                latents_padded
                            ], dim=-1)
                            
                            noise = torch.randn_like(gt_sample)
                            sample = self.pipe(
                                noise=noise, 
                                pc=pc_cond, 
                                num_inference_steps=self.num_inference_timesteps, 
                                show_progress=True, 
                                tgt_key_padding_mask=~masks_input.bool().squeeze(-1)
                            )

                            loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = self.compute_loss(
                                sample, gt_sample, masks_input
                            )
                            
                            pred_original_sample = sample[masks_input.squeeze(-1).bool()]
                            valid, shifts, rotations, scales, latents = decode_latent(
                                pred_original_sample, log_scale=self.log_scale
                            )
                            
                            # Decode latents to params
                            # params = self.vae.decode(latents)
                            params = latents
                            
                            pred_valid_surface_params = decode_only(self.vae, params)
                            loss_surf_params = torch.nn.functional.mse_loss(
                                pred_valid_surface_params, gt_valid_surface_params
                            )

                            if scales.abs().max().item() > 15.0:
                                print('Warning:', f'val_scales is out of range: {scales.abs().max().item()}')
                            
                            pred_sampled_points = decode_and_sample_with_rts(
                                self.vae, params, shifts, rotations, scales, log_scale=False, num_samples=8
                            )
                            pred_sampled_edges = decode_and_sample_with_rts(
                                self.vae, params, shifts, rotations, scales, log_scale=False, num_samples=2
                            )

                            loss_original_sample = torch.nn.functional.mse_loss(
                                pred_sampled_points.reshape(pred_sampled_points.shape[0], -1), 
                                gt_sampled_points[masks_input.squeeze(-1).bool()], 
                                reduction='none'
                            )
                            if self.use_weighted_sample_loss:
                                loss_original_sample = loss_original_sample * surface_weight.unsqueeze(-1)
                            loss_original_sample = loss_original_sample.mean()

                            loss_edges = torch.nn.functional.mse_loss(
                                pred_sampled_edges.reshape(pred_sampled_edges.shape[0], -1), 
                                gt_sampled_edges[masks_input.squeeze(-1).bool()]
                            )

                            loss = (loss_valid * self.weight_valid + 
                                    loss_shifts + loss_rotations + loss_scales + 
                                    loss_params * self.weight_params + 
                                    loss_original_sample * self.weight_original_sample * (step >= self.original_sample_start_step) + 
                                    loss_edges * self.weight_sample_edges + 
                                    loss_surf_params * self.weight_surf_params * (step >= self.original_sample_start_step))
                            
                            loss_dict['val_loss_valid'] += loss_valid.item() / num_val_batches
                            loss_dict['val_loss_shifts'] += loss_shifts.item() / num_val_batches
                            loss_dict['val_loss_rotations'] += loss_rotations.item() / num_val_batches
                            loss_dict['val_loss_scales'] += loss_scales.item() / num_val_batches
                            loss_dict['val_loss_params'] += loss_params.item() / num_val_batches
                            loss_dict['val_loss_orig_sample'] += loss_original_sample.item() / num_val_batches
                            loss_dict['val_loss_edges'] += loss_edges.item() / num_val_batches
                            loss_dict['val_loss_surf_params'] += loss_surf_params.item() / num_val_batches
                            total_val_loss += (loss / num_val_batches)

                # Print validation losses
                self.print(get_current_time() + f" total loss: {total_val_loss:.3f}, loss_valid: {loss_dict['val_loss_valid']:.3f}, loss_shifts: {loss_dict['val_loss_shifts']:.3f}, loss_rotations: {loss_dict['val_loss_rotations']:.3f}, loss_scales: {loss_dict['val_loss_scales']:.3f}, loss_params: {loss_dict['val_loss_params']:.3f}, loss_orig_sample: {loss_dict['val_loss_orig_sample']:.3f}, loss_edges: {loss_dict['val_loss_edges']:.3f}, loss_surf_params: {loss_dict['val_loss_surf_params']:.3f}")
                
                # Calculate estimated finishing time
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
                
                # Log validation losses
                val_log_dict = {"val_loss": total_val_loss, **loss_dict}
                self.log(**val_log_dict)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')
        
        self.stop_profiler()
        self.print('DIT Sample Segment Training complete')
