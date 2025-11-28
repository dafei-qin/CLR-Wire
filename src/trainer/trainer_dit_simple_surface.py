from beartype import beartype
from typing import Callable, Optional
from torch.utils.data import Dataset
from typing import Optional, Union
# from diffusers.configuration_utils import ConfigMixin
# from diffusers.models.modeling_utils import ModelMixin
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

# B-spline enhanced trainer class
class TrainerFlowSurface(BaseTrainer):
    @beartype
    def __init__(
        self,
        model: Union[ModelMixin, ConfigMixin] | nn.Module,
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
        visual_eval_every_step: int = 5000,  # New parameter for visual evaluation frequency
        num_visual_samples: int = 4,  # Number of samples to visualize
        scheduler_type: str = 'ddpm',
        prediction_type: str = 'v_prediction',
        num_training_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        weight_valid = 1.0,
        weight_params = 1.0,
        **kwargs
    ):
        super().__init__(
            model=model, 
            dataset=dataset,
            val_dataset=val_dataset,
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
        self.pipe = ZLDMPipeline(self.model, self.scheduler, dtype=torch.float32) # Model will be replaced with EMA during 

        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.weight_valid = weight_valid
        self.weight_params = weight_params

    
    def log_loss(self, total_loss, lr, total_norm, step, loss_dict={}):
        """Enhanced loss logging for B-spline model"""
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        # loss_dict = {key: value.item() for key, value in loss_dict.items()}
        log_dict = {
            'loss': total_loss,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
            **loss_dict
            
        }
        
        self.log(**log_dict)
        
        # Print detailed loss information
        loss_str = f'loss: {total_loss:.3f}, loss_valid: {loss_dict["loss_valid"]:.3f}, loss_shifts: {loss_dict["loss_shifts"]:.3f}, loss_rotations: {loss_dict["loss_rotations"]:.3f}, loss_scales: {loss_dict["loss_scales"]:.3f}, loss_params: {loss_dict["loss_params"]:.3f}'


        self.print(get_current_time() + f' {loss_str} lr: {lr:.6f} norm: {total_norm:.3f}')


    def compute_loss(self, output, target, masks):

        loss_raw = torch.nn.functional.mse_loss(output, target, reduction='none')

        loss_others = loss_raw[..., 1:] * masks.float()
        total_valid_surfaces = masks.float().sum()
        loss_shifts = loss_others[..., :3].mean(dim=(2)).sum() / total_valid_surfaces
        loss_rotations = loss_others[..., 3:3+6].mean(dim=(2)).sum() / total_valid_surfaces
        loss_scales = loss_others[..., 3+6:3+6+1].mean(dim=(2)).sum() / total_valid_surfaces
        loss_params = loss_others[..., 3+6+1:].mean(dim=(2)).sum() / total_valid_surfaces
        
        loss_valid = self.bce_logits_loss(output[..., 0], masks.float().squeeze()).mean()
        
        return loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params

    def train(self):
        """Enhanced train method with B-spline specific handling"""
        step = self.step.item()
        
        # Start profiler if enabled
        self.start_profiler()

        while step < self.num_train_steps:
            # print(self.visual_eval_every_step)
            total_loss = 0.

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)
                with self.accelerator.autocast(), maybe_no_sync():
                    params_padded, rotations_padded, scales_padded, shifts_padded, surface_type, bbox_mins, bbox_maxs, masks, pc_cond = forward_kwargs
                    masks = masks.unsqueeze(-1)
                    gt_sample = torch.cat([masks.float(), shifts_padded, rotations_padded, scales_padded, params_padded], dim=-1)
                    
                    noise = torch.randn_like(gt_sample)
                    timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (gt_sample.shape[0],), device=gt_sample.device).long()

                    noisy_sample = self.scheduler.add_noise(gt_sample, noise, timesteps)

                    if self.scheduler.config.prediction_type == 'v_prediction':
                        target = self.scheduler.get_velocity(gt_sample, noise, timesteps)
                    elif self.scheduler.config.prediction_type == 'sample':
                        target = gt_sample
                    elif self.scheduler.config.prediction_type == 'epsilon':
                        # Not tried
                        print('Warning, not tried')
                        target = noise

                    # This is for input gt sanity check.
                    # noisy_sample = gt_sample

                    # forward pass
                    output = self.model(sample=noisy_sample, timestep = timesteps, cond=pc_cond, tgt_key_padding_mask=~masks.bool().squeeze(-1))

                
                    loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = self.compute_loss(output, target, masks)
                    
                    # from tqdm import tqdm
                    # loss_all = []
                    # for _t in tqdm(range(1000)):
                    #     _timesteps = torch.tensor([999-_t, 999-_t]).int().cuda()
                    #     _noisy_sample = self.scheduler.add_noise(gt_sample, noise, _timesteps)
                    #     with torch.no_grad():
                    #         _output = self.model(sample=_noisy_sample, timestep = _timesteps, cond=pc_cond, tgt_key_padding_mask=~masks.bool().squeeze(-1))
                    #         loss_all.append(self.compute_loss(_output, target, masks))
                    # This is for input gt sanity check.
                    # loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = self.compute_loss(output, gt_sample, masks)

                
                    loss = loss_valid * self.weight_valid + loss_shifts + loss_rotations + loss_scales + loss_params * self.weight_params
                    total_loss += loss.item()
                    loss_dict = {
                        'loss_valid': loss_valid.item(),
                        'loss_shifts': loss_shifts.item(),
                        'loss_rotations': loss_rotations.item(),
                        'loss_scales': loss_scales.item(),
                        'loss_params': loss_params.item(),
                    }
                
                self.accelerator.backward(loss)
            

            self.optimizer.step()
            self.optimizer.zero_grad()


            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                                                        
                self.log_loss(total_loss, cur_lr, total_norm, step, loss_dict)
            

            
            step += 1
            self.step.add_(1)
            
            # Notify profiler of step completion
            self.profiler_step()
            
            self.wait()
            
            if self.is_main:
                self.ema.update()            

            self.wait()

            # Visual evaluation with B-spline visualization
        

            if self.is_main and divisible_by(step, self.val_every_step):
                print(f"Validating at step {step}")
                total_val_loss = 0.
                loss_dict = defaultdict(float)
                self.ema.eval()
                self.pipe.denoiser = self.ema
                # self.pipe.denoiser = self.raw_model
                device = next(self.model.parameters()).device
                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():


                        forward_args = self.next_data_to_forward_kwargs(self.val_dl_iter)
                        forward_args = [_.to(device) for _ in forward_args]
                        params_padded, rotations_padded, scales_padded, shifts_padded, surface_type, bbox_mins, bbox_maxs, masks, pc_cond = forward_args
                        masks = masks.unsqueeze(-1)
                        gt_sample = torch.cat([masks.float(), shifts_padded, rotations_padded, scales_padded, params_padded], dim=-1)
                        
                        noise = torch.randn_like(gt_sample)

                        sample  = self.pipe(noise=noise, pc=pc_cond, num_inference_steps=self.num_inference_timesteps, show_progress=True, tgt_key_padding_mask=~masks.bool().squeeze(-1))
                        # This is for input gt 
                        # sample  = self.pipe(noise=gt_sample, pc=pc_cond, num_inference_steps=1, show_progress=True)

                        loss_valid, loss_shifts, loss_rotations, loss_scales, loss_params = self.compute_loss(sample, gt_sample, masks)
                        
                        

                        loss = loss_valid * self.weight_valid + loss_shifts + loss_rotations + loss_scales + loss_params * self.weight_params
                        loss_dict['val_loss_valid'] += loss_valid.item() / num_val_batches
                        loss_dict['val_loss_shifts'] += loss_shifts.item() / num_val_batches
                        loss_dict['val_loss_rotations'] += loss_rotations.item() / num_val_batches
                        loss_dict['val_loss_scales'] += loss_scales.item() / num_val_batches
                        loss_dict['val_loss_params'] += loss_params.item() / num_val_batches
                        total_val_loss += (loss / num_val_batches)


                # Print validation losses
                self.print(get_current_time() + f" total loss: {total_val_loss:.3f}, loss_valid: {loss_dict['val_loss_valid']:.3f}, loss_shifts: {loss_dict['val_loss_shifts']:.3f}, loss_rotations: {loss_dict['val_loss_rotations']:.3f}, loss_scales: {loss_dict['val_loss_scales']:.3f}, loss_params: {loss_dict['val_loss_params']:.3f}")

                
                # Calculate and print estimated finishing time
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
        
        # Stop profiler if enabled
        self.stop_profiler()

        self.print('DIT Simple Surface Training complete') 
