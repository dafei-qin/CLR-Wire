"""
Trainer for VAE v4: DC-AE + FSQ
Simplified trainer for bspline surfaces only (4x4x3 patches)

Key differences from trainer_vae_v1:
1. No surface type classification (bspline only)
2. No sample loss (4x4x3 patches ARE the samples)
3. No is_closed prediction
4. Simplified loss: only reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Callable
import time
import wandb
from contextlib import nullcontext
from functools import partial
import einops

from src.trainer.trainer_base import BaseTrainer
from src.utils.helpers import divisible_by, get_current_time, get_lr


class Trainer_vae_v4_dcae_fsq(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset, 
        *,
        val_dataset: Dataset,
        batch_size: int = 16,
        checkpoint_folder: str = './checkpoints',
        checkpoint_every_step: int = 1000,
        checkpoint_file_name: str = 'model.pt',
        ema_update_every: int = 10,
        ema_decay: float = 0.995,
        grad_accum_every: int = 1,
        log_every_step: int = 10,
        learning_rate: float = 2e-4,
        mixed_precision_type: str = 'fp16',
        max_grad_norm: float = 1.,
        num_workers: int = 1,
        num_train_steps: int = 100000,
        num_step_per_epoch: int = 100,
        resume_training: bool = False,
        load_checkpoint_from_file: Optional[str] = None,
        from_start: bool = False,
        use_wandb_tracking: bool = False,
        collate_fn: Optional[Callable] = None,
        val_every_step: int = 1000,
        val_num_batches: int = 10,
        val_batch_size: int = 256,
        accelerator_kwargs: dict = None,
        loss_recon_weight: float = 1.0,
        train_sampler = None,
        num_workers_val: int = 8,
        **kwargs
    ):
        if accelerator_kwargs is None:
            accelerator_kwargs = dict()
            
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
            val_batch_size=val_batch_size,
            train_sampler=train_sampler,
            num_workers_val=num_workers_val,
            **kwargs
        )
        
        # Loss functions
        self.loss_recon = nn.MSELoss(reduction='mean')
        self.loss_recon_weight = loss_recon_weight
        
        # Get raw model (unwrap from accelerator if needed)
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Print model info
        print(f"\n{'='*70}")
        print(f"Trainer VAE v4: DC-AE + FSQ")
        print(f"{'='*70}")
        print(f"Model type: DCAE_FSQ_VAE")
        print(f"Codebook size: {self.raw_model.codebook_size}")
        print(f"Num codebooks: {self.raw_model.num_codebooks}")
        print(f"Total capacity: {self.raw_model.total_codebook_capacity:,}")
        print(f"Latent dim: {self.raw_model.latent_dim}")
        print(f"Input size: ({self.raw_model.in_channels}, {self.raw_model.input_size}, {self.raw_model.input_size})")
        print(f"Loss reconstruction weight: {self.loss_recon_weight}")
        print(f"{'='*70}\n")

    def log_loss(
        self, 
        total_loss: float,
        loss_recon: float,
        lr: float,
        total_norm: float,
        step: int,
        time_per_step: float,
        codebook_usage: Optional[float] = None,
        unique_codes: Optional[int] = None
    ):
        """Log training metrics"""
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'loss_recon': loss_recon,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
            'time_per_step': time_per_step,
        }
        
        # Add FSQ-specific metrics
        if codebook_usage is not None:
            log_dict['fsq/codebook_usage'] = codebook_usage
        if unique_codes is not None:
            log_dict['fsq/unique_codes'] = unique_codes
        
        self.log(**log_dict)
        
        # Print statement
        print_msg = (f'{get_current_time()} loss: {total_loss:.4f} '
                    f'recon: {loss_recon:.4f} '
                    f'lr: {lr:.6f} norm: {total_norm:.3f} '
                    f'time: {time_per_step:.3f}s')
        
        if codebook_usage is not None:
            print_msg += f' CB_usage: {codebook_usage:.3f} codes: {unique_codes}'
        
        self.print(print_msg)

    def train(self):
        """Main training loop"""
        step = self.step.item()
        
        tt = time.time()
        while step < self.num_train_steps:
            total_loss = 0.
            t = time.time()
            
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext
                
                # Get data
                # Expected format: patches (B, C, H, W) - e.g., (B, 3, 4, 4)
                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)
                
                with self.accelerator.autocast(), maybe_no_sync():
                    # Unpack data
                    # Assuming forward_kwargs returns (patches,) or just patches
                    # if isinstance(forward_kwargs, tuple):
                    #     patches = forward_kwargs[0]
                    # else:
                    #     patches = forward_kwargs
                    patches = forward_kwargs[0]
                    patches = patches[..., 17:].reshape(-1, 4, 4, 3)
                    patches = einops.rearrange(patches, 'b h w c -> b c h w')
                    
                    # Ensure correct shape: (B, 3, 4, 4)
                    if patches.dim() == 3:
                        patches = patches.unsqueeze(1)  # Add channel dim if missing
                    
                    # Forward pass
                    x_recon, z_quantized, indices, metrics = self.model(patches)
                    
                    # Compute reconstruction loss
                    loss_recon = self.loss_recon(x_recon, patches)
                    
                    # Total loss (only reconstruction for now)
                    loss = loss_recon * self.loss_recon_weight
                    
                    # Gradient accumulation
                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                
                # Backward pass
                self.accelerator.backward(loss)
            
            # Compute time per step
            time_per_step = (time.time() - t) / self.grad_accum_every
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Logging
            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                
                # Compute FSQ-specific metrics
                codebook_usage = None
                unique_codes = None
                if indices is not None:
                    # Handle both single and multiple codebooks
                    if indices.ndim == 1:
                        unique_codes_tensor = torch.unique(indices)
                        unique_codes = unique_codes_tensor.numel()
                        codebook_usage = unique_codes / self.raw_model.codebook_size
                    else:
                        # Multiple codebooks: average usage across all codebooks
                        usage_per_cb = []
                        for i in range(indices.shape[1]):
                            unique_codes_tensor = torch.unique(indices[:, i])
                            usage_per_cb.append(unique_codes_tensor.numel() / self.raw_model.codebook_size)
                        codebook_usage = sum(usage_per_cb) / len(usage_per_cb)
                        unique_codes = int(sum([torch.unique(indices[:, i]).numel() 
                                              for i in range(indices.shape[1])]) / indices.shape[1])
                
                self.log_loss(
                    total_loss, 
                    loss_recon.item(), 
                    cur_lr, 
                    total_norm, 
                    step, 
                    time_per_step,
                    codebook_usage, 
                    unique_codes
                )
            
            # Update step
            step += 1
            self.step.add_(1)
            
            self.wait()
            
            # EMA update
            if self.is_main:
                self.ema.update()
            
            self.wait()
            
            # Validation
            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                print(f"Validating at step {step}")
                self.validate(step)
            
            self.wait()
            
            # Checkpoint saving
            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')
        
        self.print('VAE V4 DC-AE + FSQ training complete')

    def validate(self, step: int):
        """Validation loop"""
        total_val_loss = 0.
        total_val_loss_recon = 0.
        device = next(self.model.parameters()).device
        
        # Collect all indices to compute codebook usage
        all_val_indices = []
        
        self.ema.eval()
        num_val_batches = self.val_num_batches * self.grad_accum_every
        
        for _ in range(num_val_batches):
            with self.accelerator.autocast(), torch.no_grad():
                # Get validation data
                forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
                
                # Unpack data
                patches = forward_kwargs[0]
                patches = patches[..., 17:].reshape(-1, 4, 4, 3)
                patches = einops.rearrange(patches, 'b h w c -> b c h w')
                
                # Ensure correct shape
                if patches.dim() == 3:
                    patches = patches.unsqueeze(1)
                
                patches = patches.to(device)
                
                # Forward pass
                x_recon, z_quantized, indices, metrics = self.model(patches)
                
                # Compute reconstruction loss
                loss_recon = self.loss_recon(x_recon, patches)
                
                # Total loss
                loss = loss_recon * self.loss_recon_weight
                
                total_val_loss += (loss / num_val_batches)
                total_val_loss_recon += (loss_recon / num_val_batches)
                
                # Collect indices
                all_val_indices.append(indices)
        
        # Compute FSQ codebook usage for validation
        val_codebook_usage = None
        val_unique_codes = None
        if len(all_val_indices) > 0:
            all_val_indices = torch.cat(all_val_indices)
            
            # Handle both single and multiple codebooks
            if all_val_indices.ndim == 1:
                val_unique_codes_tensor = torch.unique(all_val_indices)
                val_unique_codes = val_unique_codes_tensor.numel()
                val_codebook_usage = val_unique_codes / self.raw_model.codebook_size
            else:
                # Multiple codebooks: average usage across all codebooks
                usage_per_cb = []
                for i in range(all_val_indices.shape[1]):
                    unique_codes_tensor = torch.unique(all_val_indices[:, i])
                    usage_per_cb.append(unique_codes_tensor.numel() / self.raw_model.codebook_size)
                val_codebook_usage = sum(usage_per_cb) / len(usage_per_cb)
                val_unique_codes = int(sum([torch.unique(all_val_indices[:, i]).numel() 
                                           for i in range(all_val_indices.shape[1])]) / all_val_indices.shape[1])
        
        # Print validation results
        print_msg = (f'{get_current_time()} valid loss: {total_val_loss:.4f} '
                    f'recon: {total_val_loss_recon.item():.4f}')
        
        if val_codebook_usage is not None:
            print_msg += f' CB_usage: {val_codebook_usage:.3f} codes: {val_unique_codes}'
        
        self.print(print_msg)
        
        # Calculate estimated finishing time
        steps_remaining = self.num_train_steps - step
        time_per_step = (time.time() - self.start_time) / (step + 1)
        estimated_time_remaining = steps_remaining * time_per_step
        estimated_finish_time = time.time() + estimated_time_remaining
        self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
        
        # Log to wandb
        val_log_dict = {
            "val_loss": total_val_loss,
            "val_loss_recon": total_val_loss_recon.item(),
        }
        
        # Add FSQ-specific validation metrics
        if val_codebook_usage is not None:
            val_log_dict['val_fsq/codebook_usage'] = val_codebook_usage
        if val_unique_codes is not None:
            val_log_dict['val_fsq/unique_codes'] = val_unique_codes
        
        self.log(**val_log_dict)


if __name__ == "__main__":
    """Simple test of the trainer"""
    print("Trainer VAE v4 DC-AE + FSQ")
    print("This trainer is designed for bspline surface patches (4x4x3)")
    print("\nKey features:")
    print("  - No surface type classification")
    print("  - No sample loss (patches are samples)")
    print("  - No is_closed prediction")
    print("  - Only reconstruction loss")
    print("  - FSQ quantization (no KL loss)")

