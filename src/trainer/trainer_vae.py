from beartype import beartype
from typing import Callable, Optional
from torch.utils.data import Dataset
from typing import Optional, Union
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
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

from src.trainer.trainer_base import BaseTrainer
from src.utils.helpers import divisible_by, get_current_time, get_lr

# trainer class
class Trainer(BaseTrainer):
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
        from_start = False,
        use_wandb_tracking: bool = False,
        collate_fn: Optional[Callable] = None,
        val_every_step: int = 1000,
        val_num_batches: int = 10,
        accelerator_kwargs: dict = dict(),
        visual_eval_every_step: int = 5000,  # New parameter for visual evaluation frequency
        num_visual_samples: int = 4,  # Number of samples to visualize
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
        
    def create_curve_visualization(self, gt_samples, reconstructed, step):
        """Create visualization for 1D curve data"""
        # gt_samples and reconstructed should be (batch, channels, num_points)
        # Convert to (batch, num_points, channels) for plotting
        if gt_samples.dim() == 3 and gt_samples.shape[1] == 3:  # (B, 3, N)
            gt_samples = rearrange(gt_samples, 'b c n -> b n c')
            reconstructed = rearrange(reconstructed, 'b c n -> b n c')
        
        batch_size = min(self.num_visual_samples, gt_samples.shape[0])
        
        fig, axes = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
        if batch_size == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(batch_size):
            gt_curve = gt_samples[i].cpu().numpy()  # (N, 3)
            recon_curve = reconstructed[i].cpu().numpy()  # (N, 3)
            
            # Plot ground truth
            ax_gt = axes[0, i]
            if gt_curve.shape[1] == 3:  # 3D curve
                ax_gt = fig.add_subplot(2, batch_size, i+1, projection='3d')
                ax_gt.plot(gt_curve[:, 0], gt_curve[:, 1], gt_curve[:, 2], 'b-', linewidth=2, label='GT')
                ax_gt.set_xlabel('X')
                ax_gt.set_ylabel('Y')
                ax_gt.set_zlabel('Z')
            else:  # 2D curve
                ax_gt.plot(gt_curve[:, 0], gt_curve[:, 1], 'b-', linewidth=2, label='GT')
                ax_gt.set_xlabel('X')
                ax_gt.set_ylabel('Y')
            ax_gt.set_title(f'Ground Truth {i+1}')
            ax_gt.grid(True)
            ax_gt.legend()
            
            # Plot reconstruction
            ax_recon = axes[1, i]
            if recon_curve.shape[1] == 3:  # 3D curve
                ax_recon = fig.add_subplot(2, batch_size, batch_size+i+1, projection='3d')
                ax_recon.plot(recon_curve[:, 0], recon_curve[:, 1], recon_curve[:, 2], 'r-', linewidth=2, label='Recon')
                ax_recon.set_xlabel('X')
                ax_recon.set_ylabel('Y')
                ax_recon.set_zlabel('Z')
            else:  # 2D curve
                ax_recon.plot(recon_curve[:, 0], recon_curve[:, 1], 'r-', linewidth=2, label='Recon')
                ax_recon.set_xlabel('X')
                ax_recon.set_ylabel('Y')
            ax_recon.set_title(f'Reconstruction {i+1}')
            ax_recon.grid(True)
            ax_recon.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Curve VAE - Step {step}', y=1.02)
        
        return fig
    
    def create_surface_visualization(self, gt_samples, reconstructed, step):
        """Create visualization for 2D surface data"""
        # gt_samples and reconstructed should be (batch, channels, height, width)
        # Convert to (batch, height, width, channels) for plotting
        if gt_samples.dim() == 4 and gt_samples.shape[1] == 3:  # (B, 3, H, W)
            gt_samples = rearrange(gt_samples, 'b c h w -> b h w c')
            reconstructed = rearrange(reconstructed, 'b c h w -> b h w c')
        
        batch_size = min(self.num_visual_samples, gt_samples.shape[0])
        
        fig = plt.figure(figsize=(6*batch_size, 12))
        
        for i in range(batch_size):
            gt_surface = gt_samples[i].cpu().numpy()  # (H, W, 3)
            recon_surface = reconstructed[i].cpu().numpy()  # (H, W, 3)
            
            # Create meshgrid for plotting
            h, w = gt_surface.shape[:2]
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            X, Y = np.meshgrid(x, y)
            
            # Plot ground truth surface
            ax_gt = fig.add_subplot(2, batch_size, i+1, projection='3d')
            ax_gt.plot_surface(X, Y, gt_surface[:, :, 2], 
                             facecolors=plt.cm.viridis((gt_surface[:, :, 2] - gt_surface[:, :, 2].min()) / 
                                                      (gt_surface[:, :, 2].max() - gt_surface[:, :, 2].min() + 1e-8)),
                             alpha=0.8)
            ax_gt.set_title(f'Ground Truth Surface {i+1}')
            ax_gt.set_xlabel('X')
            ax_gt.set_ylabel('Y')
            ax_gt.set_zlabel('Z')
            
            # Plot reconstruction surface
            ax_recon = fig.add_subplot(2, batch_size, batch_size+i+1, projection='3d')
            ax_recon.plot_surface(X, Y, recon_surface[:, :, 2],
                                facecolors=plt.cm.plasma((recon_surface[:, :, 2] - recon_surface[:, :, 2].min()) / 
                                                        (recon_surface[:, :, 2].max() - recon_surface[:, :, 2].min() + 1e-8)),
                                alpha=0.8)
            ax_recon.set_title(f'Reconstruction Surface {i+1}')
            ax_recon.set_xlabel('X')
            ax_recon.set_ylabel('Y')
            ax_recon.set_zlabel('Z')
        
        plt.tight_layout()
        plt.suptitle(f'Surface VAE - Step {step}', y=1.02)
        
        return fig
    
    def perform_visual_evaluation(self, step):
        """Perform visual evaluation and log to wandb"""
        if not self.use_wandb_tracking or not self.is_main:
            return
            
        self.ema.eval()
        
        with torch.no_grad():
            # Get a batch of validation data
            forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
            if isinstance(forward_kwargs, dict):
                forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()}
                data = forward_kwargs.get('data', list(forward_kwargs.values())[0])
            else:
                forward_kwargs = forward_kwargs.to(self.device)
                data = forward_kwargs
            
            # Get model output with ground truth samples
            if hasattr(self.ema, 'module'):
                model = self.ema.module
            elif hasattr(self.ema, 'model'):
                model = self.ema.model
            else:
                model = self.ema
                
            # Forward pass to get reconstructions and ground truth samples
            if isinstance(forward_kwargs, dict):
                output = model(**forward_kwargs, sample_posterior=False, return_loss=True)
            else:
                output = model(forward_kwargs, sample_posterior=False, return_loss=True)
            
            if isinstance(output, tuple):
                loss, loss_dict = output
                # Get the reconstruction and ground truth
                if isinstance(forward_kwargs, dict):
                    result = model(**forward_kwargs, sample_posterior=False, return_loss=False)
                else:
                    result = model(forward_kwargs, sample_posterior=False, return_loss=False)
                reconstructed = result.sample
                
                # Get ground truth samples by calling the model's interpolation
                if hasattr(model, 'forward'):
                    # Access the t and gt_samples from the model's forward method
                    # We need to recreate the forward pass to get gt_samples
                    if data.dim() == 3:  # Curve data (B, N, C)
                        data_input = rearrange(data, "b n c -> b c n")
                        bs = data_input.shape[0]
                        sample_points_num = getattr(model, 'sample_points_num', 64)
                        t = torch.rand(bs, sample_points_num, device=data.device)
                        t, _ = torch.sort(t, dim=-1)
                        
                        from src.utils.torch_tools import interpolate_1d
                        gt_samples = interpolate_1d(t, data_input)
                        
                        # Create curve visualization
                        fig = self.create_curve_visualization(gt_samples, reconstructed, step)
                        
                    elif data.dim() == 4:  # Surface data (B, H, W, C)
                        data_input = rearrange(data, "b h w c -> b c h w")
                        bs = data_input.shape[0]
                        sample_points_num = getattr(model, 'sample_points_num', 32)
                        
                        # Generate grid coordinates
                        t_1d = torch.linspace(0, 1, sample_points_num, device=data.device)
                        t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
                        t = t_grid.unsqueeze(0).repeat(bs, 1, 1, 1)
                        
                        from src.utils.torch_tools import interpolate_2d
                        gt_samples = interpolate_2d(t, data_input)
                        
                        # Create surface visualization
                        fig = self.create_surface_visualization(gt_samples, reconstructed, step)
                    
                    else:
                        self.print(f"Unsupported data dimension for visualization: {data.dim()}")
                        return
                    
                    # Log to wandb
                    wandb.log({
                        "visual_evaluation": wandb.Image(fig),
                        "step": step
                    })
                    
                    plt.close(fig)  # Clean up memory
                    
            else:
                self.print("Could not extract reconstruction from model output")
    
    def train(self):
        """Override train method to add visual evaluation"""
        step = self.step.item()

        while step < self.num_train_steps:
                
            total_loss = 0.

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)

                # with torch.autograd.detect_anomaly():
                with self.accelerator.autocast(), maybe_no_sync():

                    loss, loss_dict = self.train_step(forward_kwargs)
                
                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                
                self.accelerator.backward(loss)
            
            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                                                        
                self.log_loss(total_loss, loss_dict, cur_lr, total_norm, step)
                    
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            step += 1
            self.step.add_(1)
            
            self.wait()
            
            if self.is_main:
                self.ema.update()            

            self.wait()

            # Visual evaluation
            if self.is_main and self.should_validate and divisible_by(step, self.visual_eval_every_step):
                try:
                    self.perform_visual_evaluation(step)
                except Exception as e:
                    self.print(f"Visual evaluation failed at step {step}: {e}")

            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):

                total_val_loss = 0.
                self.ema.eval()

                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():

                        forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
                        if isinstance(forward_kwargs, dict):
                            forward_kwargs = {k: v.to(self.device) for k, v in forward_kwargs.items()}
                        else:
                            forward_kwargs = forward_kwargs.to(self.device)

                        loss, loss_dict = self.train_step(forward_kwargs, is_train=False)

                        total_val_loss += (loss / num_val_batches)

                self.print(get_current_time() + f' valid loss: {total_val_loss:.3f}')    
                # Calculate and print estimated finishing time
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
                
                self.log(val_loss = total_val_loss)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.print('training complete')