from beartype import beartype
from typing import Callable, Optional
from torch.utils.data import Dataset
from typing import Optional, Union
# from diffusers.configuration_utils import ConfigMixin
# from diffusers.models.modeling_utils import ModelMixin
from diffusers import ConfigMixin, ModelMixin
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import wandb
from einops import rearrange
from functools import partial
from contextlib import nullcontext
import time

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
        **kwargs
    ):
        super().__init__(
            model=model, 
            dataset=dataset,
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
        
        # Visual evaluation parameters
        self.visual_eval_every_step = visual_eval_every_step
        self.num_visual_samples = num_visual_samples
        self.scheduler = get_new_scheduler(prediction_type)
        self.pipe = ZLDMPipeline(self.model, self.scheduler, dtype=torch.float32) # Model will be replaced with EMA during inference
        self.cp2surfaceLayer = BSplineSurfaceLayer(resolution=model.res)

    def create_surface_visualization_bs(self, gt_samples, cross_attention_recon, bspline_recon, control_points, step):
        """Create comprehensive visualization for B-spline enhanced surface VAE"""
        # gt_samples and reconstructions should be (batch, channels, height, width)
        # Convert to (batch, height, width, channels) for plotting
        if gt_samples.dim() == 4 and gt_samples.shape[1] == 3:  # (B, 3, H, W)
            gt_samples = rearrange(gt_samples, 'b c h w -> b h w c')
            # cross_attention_recon = rearrange(cross_attention_recon, 'b c h w -> b h w c')
            bspline_recon = rearrange(bspline_recon, 'b c h w -> b h w c')
        
        batch_size = min(self.num_visual_samples, gt_samples.shape[0])
        
        # Fixed axis limits for consistent scaling across all plots
        axis_limit = 1.5  # Slightly larger than the normalized range [-1, 1]
        
        fig = plt.figure(figsize=(6*batch_size, 24))  # Increased height for 4 rows
        
        for i in range(batch_size):
            gt_surface = gt_samples[i].cpu().numpy()  # (H, W, 3)
            # ca_surface = cross_attention_recon[i].cpu().numpy()  # (H, W, 3)
            bs_surface = bspline_recon[i].cpu().numpy()  # (H, W, 3)
            control_pts = control_points[i].cpu().numpy()  # (16, 3)
            
            # Extract coordinates
            X_gt, Y_gt, Z_gt = gt_surface[:, :, 0], gt_surface[:, :, 1], gt_surface[:, :, 2]
            # X_ca, Y_ca, Z_ca = ca_surface[:, :, 0], ca_surface[:, :, 1], ca_surface[:, :, 2]
            X_bs, Y_bs, Z_bs = bs_surface[:, :, 0], bs_surface[:, :, 1], bs_surface[:, :, 2]
            
            # Row 1: Ground Truth
            ax_gt = fig.add_subplot(4, batch_size, i+1, projection='3d')
            ax_gt.plot_surface(X_gt, Y_gt, Z_gt, 
                             facecolors=plt.cm.viridis((Z_gt - Z_gt.min()) / 
                                                      (Z_gt.max() - Z_gt.min() + 1e-8)),
                             alpha=0.8)
            ax_gt.set_title(f'Ground Truth {i+1}')
            ax_gt.set_xlabel('X')
            ax_gt.set_ylabel('Y')
            ax_gt.set_zlabel('Z')
            ax_gt.set_xlim([-axis_limit, axis_limit])
            ax_gt.set_ylim([-axis_limit, axis_limit])
            ax_gt.set_zlim([-axis_limit, axis_limit])
            
            # # Row 2: Cross-Attention Reconstruction
            # ax_ca = fig.add_subplot(4, batch_size, batch_size+i+1, projection='3d')
            # ax_ca.plot_surface(X_ca, Y_ca, Z_ca,
            #                  facecolors=plt.cm.plasma((Z_ca - Z_ca.min()) / 
            #                                          (Z_ca.max() - Z_ca.min() + 1e-8)),
            #                  alpha=0.8)
            # ax_ca.set_title(f'Cross-Attention Recon {i+1}')
            # ax_ca.set_xlabel('X')
            # ax_ca.set_ylabel('Y')
            # ax_ca.set_zlabel('Z')
            # ax_ca.set_xlim([-axis_limit, axis_limit])
            # ax_ca.set_ylim([-axis_limit, axis_limit])
            # ax_ca.set_zlim([-axis_limit, axis_limit])
            
            # # Row 3: B-spline Reconstruction
            ax_bs = fig.add_subplot(4, batch_size, batch_size+i+1, projection='3d')
            ax_bs.plot_surface(X_bs, Y_bs, Z_bs,
                             facecolors=plt.cm.coolwarm((Z_bs - Z_bs.min()) / 
                                                       (Z_bs.max() - Z_bs.min() + 1e-8)),
                             alpha=0.8)
            ax_bs.set_title(f'B-spline Recon {i+1}')
            ax_bs.set_xlabel('X')
            ax_bs.set_ylabel('Y')
            ax_bs.set_zlabel('Z')
            ax_bs.set_xlim([-axis_limit, axis_limit])
            ax_bs.set_ylim([-axis_limit, axis_limit])
            ax_bs.set_zlim([-axis_limit, axis_limit])
            
            # Row 4: B-spline Surface with Control Points
            ax_ctrl = fig.add_subplot(4, batch_size, 2*batch_size+i+1, projection='3d')
            # Plot the B-spline surface with transparency
            ax_ctrl.plot_surface(X_bs, Y_bs, Z_bs,
                               facecolors=plt.cm.coolwarm((Z_bs - Z_bs.min()) / 
                                                         (Z_bs.max() - Z_bs.min() + 1e-8)),
                               alpha=0.6)
            
            # Plot control points
            ax_ctrl.scatter(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2], 
                          c='red', s=100, alpha=1.0, label='Control Points')
            
            # Connect control points in a grid pattern (4x4 grid)
            control_grid = control_pts.reshape(4, 4, 3)
            
            # Draw grid lines for control points
            for row in range(4):
                for col in range(4):
                    # Horizontal connections
                    if col < 3:
                        ax_ctrl.plot([control_grid[row, col, 0], control_grid[row, col+1, 0]],
                                   [control_grid[row, col, 1], control_grid[row, col+1, 1]],
                                   [control_grid[row, col, 2], control_grid[row, col+1, 2]], 
                                   'r--', alpha=0.7, linewidth=1)
                    # Vertical connections
                    if row < 3:
                        ax_ctrl.plot([control_grid[row, col, 0], control_grid[row+1, col, 0]],
                                   [control_grid[row, col, 1], control_grid[row+1, col, 1]],
                                   [control_grid[row, col, 2], control_grid[row+1, col, 2]], 
                                   'r--', alpha=0.7, linewidth=1)
            
            ax_ctrl.set_title(f'B-spline + Control Points {i+1}')
            ax_ctrl.set_xlabel('X')
            ax_ctrl.set_ylabel('Y')
            ax_ctrl.set_zlabel('Z')
            ax_ctrl.set_xlim([-axis_limit, axis_limit])
            ax_ctrl.set_ylim([-axis_limit, axis_limit])
            ax_ctrl.set_zlim([-axis_limit, axis_limit])
            ax_ctrl.legend()
        
        plt.tight_layout()
        plt.suptitle(f'B-spline Surface VAE - Step {step}', y=1.02)
        
        return fig
    
    def perform_visual_evaluation(self, step):
        """Perform visual evaluation with B-spline branch visualization"""
        if not self.use_wandb_tracking or not self.is_main:
            return
            
        self.ema.eval()
        
        with torch.no_grad():
            # Get a batch of validation data
            forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
            if isinstance(forward_kwargs, dict):
                forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()}
                # data = forward_kwargs.get('data', list(forward_kwargs.values())[0])
            else:
                forward_kwargs = forward_kwargs.to(self.device)
                # data = forward_kwargs
            
            # Get model output with ground truth samples
            # if hasattr(self.ema, 'module'):
            #     model = self.ema.module
            # elif hasattr(self.ema, 'model'):
            #     model = self.ema.model
            # else:
            #     model = self.ema
                
            # Check if model has B-spline capability
            # if not hasattr(model, '_decode_bspline'):
            #     self.print("Model does not have B-spline capability, falling back to standard visualization")
            #     return
                
            # # Forward pass to get reconstructions and ground truth samples
            # if isinstance(forward_kwargs, dict):
            #     output = model(**forward_kwargs, sample_posterior=False, return_loss=True, return_both_branches=True)
            # else:
            #     output = model(forward_kwargs, sample_posterior=False, return_loss=True, return_both_branches=True)
            device = forward_kwargs['data'].device
            #   sample  = self.pipe(pc=forward_kwargs['pc'], num_latents=3, sample=forward_kwargs['data'], sample_mask=forward_kwargs['mask'], num_samples=forward_kwargs['data'].shape[0], device=device, num_inference_steps=50)
            sample  = self.pipe(pc=forward_kwargs['pc'], num_latents=3, sample=None, sample_mask=None, num_samples=forward_kwargs['data'].shape[0], device=device, num_inference_steps=50)
            # loss, loss_dict = self.train_step(forward_kwargs, is_train=False)
            # loss = torch.nn.functional.mse_loss(sample, forward_kwargs['data'], weight=(1-forward_kwargs['mask']))
            # total_val_loss += (loss / num_val_batches)
            control_point_results = sample
            sampled_surface_results = self.cp2surfaceLayer(control_point_results)
            gt_surface_samples = self.cp2surfaceLayer(forward_kwargs['data'])

            
            # if isinstance(output, tuple) and len(output) == 2:
            #     loss, loss_dict = output
            #     # Get both branch reconstructions
            #     if isinstance(forward_kwargs, dict):
            #         control_point_results, sampled_surface_results, ca_result = model(**forward_kwargs, sample_posterior=False, return_loss=False, return_both_branches=True)
            #     else:
            #         control_point_results, sampled_surface_results, ca_result = model(forward_kwargs, sample_posterior=False, return_loss=False, return_both_branches=True)
                
            #     cross_attention_recon = ca_result
            #     bspline_recon = sampled_surface_results
                
            #     # Get control points by encoding and then extracting them
            #     if data.dim() == 4:  # Surface data (B, H, W, C)
            #         data_input = rearrange(data, "b h w c -> b c h w")
            #         bs = data_input.shape[0]
            #         sample_points_num = getattr(model, 'sample_points_num', 16)
                    

            #         t_1d = torch.linspace(0, 1, sample_points_num, device=data.device)
            #         t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
            #         t = t_grid.unsqueeze(0).repeat(bs, 1, 1, 1)
                    
            #         from src.utils.torch_tools import interpolate_2d
            #         gt_samples = interpolate_2d(t, data_input)
                    
                    # Create B-spline visualization
            fig = self.create_surface_visualization_bs(
                gt_surface_samples, None, sampled_surface_results, control_point_results, step
            )
            
            # Log to wandb
            wandb.log({
                "bspline_visual_evaluation": wandb.Image(fig),
                "step": step
            })
            
            plt.close(fig)  # Clean up memory
                    
            #     else:
            #         self.print(f"Unsupported data dimension for B-spline visualization: {data.dim()}")
            #         return
                    
            # else:
            #     self.print("Could not extract reconstructions from B-spline model output")
    
    def log_loss(self, total_loss, lr, total_norm, step):
        """Enhanced loss logging for B-spline model"""
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        # loss_dict = {key: value.item() for key, value in loss_dict.items()}
        log_dict = {
            'loss': total_loss,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,

        }
        
        self.log(**log_dict)
        
        # Print detailed loss information
        loss_str = f'loss: {total_loss:.3f}'
        # for key, value in loss_dict.items():
        #     loss_str += f' {key}: {value:.3f}'

        self.print(get_current_time() + f' {loss_str} lr: {lr:.6f} norm: {total_norm:.3f}')

    def train(self):
        """Enhanced train method with B-spline specific handling"""
        step = self.step.item()

        while step < self.num_train_steps:
            # print(step)
            # print(self.visual_eval_every_step)
            total_loss = 0.

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():

                    model = self.model # use model instead of ema
                    gt_sample = forward_kwargs['data']
                    mask_sample = forward_kwargs['mask'] # mask == 1 means no noise
                    pc_cond = forward_kwargs['pc']

                    noise = torch.randn_like(gt_sample)
                    timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (gt_sample.shape[0],), device=gt_sample.device).long()

                    noisy_sample = self.scheduler.add_noise(gt_sample, noise, timesteps)
                    noisy_sample = noisy_sample * (1 - mask_sample) + gt_sample * mask_sample
                    if self.scheduler.prediction_type == 'v_prediction':
                        target = self.scheduler.get_velocity(gt_sample, noise, timesteps)
                    else:
                        target = gt_sample

                    # forward pass
                    output = model(sample=noisy_sample, t = timesteps, pc_cond=pc_cond)

                    loss = torch.nn.functional.mse_loss(output, target, reduction='none')
                    loss = loss * (1-mask_sample)

                    loss = loss.mean()
                    # loss
                    # loss = self.loss_fn(output, target)
                
                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                
                self.accelerator.backward(loss)
            

            self.optimizer.step()
            self.optimizer.zero_grad()


            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                                                        
                self.log_loss(total_loss, cur_lr, total_norm, step)
            

            
            step += 1
            self.step.add_(1)
            
            self.wait()
            
            if self.is_main:
                self.ema.update()            

            self.wait()

            # Visual evaluation with B-spline visualization
            if self.is_main and self.should_validate and divisible_by(step, self.visual_eval_every_step):
                print(f"Visual evaluating at step {step}")
                self.pipe.denoiser = self.ema
                self.perform_visual_evaluation(step)
                # try:
                #     self.perform_visual_evaluation(step)
                # except Exception as e:
                #     self.print(f"B-spline visual evaluation failed at step {step}: {e}")
            self.wait()

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                print(f"Validating at step {step}")
                total_val_loss = 0.
                
                self.ema.eval()
                self.pipe.denoiser = self.ema

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
                        if isinstance(forward_kwargs, dict):
                            forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()}
                        else:
                            forward_kwargs = forward_kwargs.to(self.device)
                        # sample  = self.pipe(pc=forward_kwargs['pc'], num_latents=3, sample=forward_kwargs['data'], sample_mask=forward_kwargs['mask'], num_samples=forward_kwargs['data'].shape[0], device=self.device, num_inference_steps=50)
                        sample  = self.pipe(pc=forward_kwargs['pc'], num_latents=3, sample=None, sample_mask=None, num_samples=forward_kwargs['data'].shape[0], device=self.device, num_inference_steps=50)
                        # sample  = self.pipe(pc=forward_kwargs['pc'], num_latents=3,  num_samples=forward_kwargs['data'].shape[0], device=self.device)
                        # loss, loss_dict = self.train_step(forward_kwargs, is_train=False)

                        # Here we predict the clean sample directly.
                        loss = torch.nn.functional.mse_loss(sample, forward_kwargs['data'], reduction='none')
                        # loss = torch.nn.functional.mse_loss(output, gt_sample, reduction='none')
                        loss = loss * (1-forward_kwargs['mask'])

                        loss = loss.mean()
                        total_val_loss += (loss / num_val_batches)


                # Print validation losses
                self.print(get_current_time() + f' valid loss: {total_val_loss:.3f}')

                
                # Calculate and print estimated finishing time
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
                
                # Log validation losses
                val_log_dict = {"val_loss": total_val_loss}

                self.log(**val_log_dict)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.print('B-spline enhanced training complete') 
