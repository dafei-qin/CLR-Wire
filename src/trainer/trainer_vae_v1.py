import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Union, Callable
import time
import wandb
from contextlib import nullcontext
from functools import partial

from src.trainer.trainer_base import BaseTrainer
from src.utils.helpers import divisible_by, get_current_time, get_lr

class Trainer_vae_v1(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
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
        loss_recon_weight: float = 1.0,
        loss_cls_weight: float = 1.0,
        loss_kl_weight: float = 1.0,
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
        
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_recon = nn.MSELoss()
        self.loss_recon_weight = loss_recon_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_kl_weight = loss_kl_weight

    def compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        return correct.mean()

    def log_loss(self, total_loss, accuracy, loss_recon, loss_cls, loss_kl, lr, total_norm, step):
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'accuracy': accuracy,
            'loss_recon': loss_recon,
            'loss_cls': loss_cls,
            'loss_kl': loss_kl,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
        }
        
        self.log(**log_dict)
        
        self.print(get_current_time() + f' loss: {total_loss:.3f} acc: {accuracy:.3f} lr: {lr:.6f} norm: {total_norm:.3f}')

    def train(self):
        step = self.step.item()

        while step < self.num_train_steps:
            total_loss = 0.
            total_accuracy = 0.

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)

                with self.accelerator.autocast(), maybe_no_sync():
                    params_padded, surface_type, masks = forward_kwargs
                    params_padded = params_padded[masks.bool()] 
                    surface_type = surface_type[masks.bool()]
                    
                    # Forward pass
                    params_raw_recon, mask, class_logits, mu, logvar = self.model(params_padded, surface_type)
                    
                    loss_recon = (self.loss_recon(params_raw_recon, params_padded) * mask.float()).mean() * self.loss_recon_weight
                    loss_cls = self.loss_cls(class_logits, surface_type.squeeze()).mean()
                    loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

                    loss = loss_recon * self.loss_recon_weight + loss_cls * self.loss_cls_weight + loss_kl * self.loss_kl_weight
                    
                    accuracy = self.compute_accuracy(class_logits, surface_type.squeeze())


                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                    total_accuracy += accuracy.item() / self.grad_accum_every
                
                self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                                                        
                self.log_loss(total_loss, total_accuracy, loss_recon.item(), loss_cls.item(), loss_kl.item(), cur_lr, total_norm, step)
            
            step += 1
            self.step.add_(1)
            
            self.wait()
            
            if self.is_main:
                self.ema.update()            

            self.wait()

            # Validation
            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                print(f"Validating at step {step}")
                total_val_loss = 0.
                total_val_accuracy = 0.
                
                self.ema.eval()
                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)
                        params_padded, surface_type = forward_kwargs
                        
                        params_padded = params_padded.to(self.device)
                        surface_type = surface_type.to(self.device)

                        params_raw_recon, mask, class_logits, mu, logvar = self.ema(params_padded, surface_type)

                        loss_recon = (self.loss_recon(params_raw_recon, params_padded) * mask.float()).mean()
                        loss_cls = self.loss_cls(class_logits, surface_type.squeeze()).mean()
                        loss_kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                        loss = loss_recon + loss_cls + loss_kl
                        accuracy = self.compute_accuracy(class_logits, surface_type.squeeze())

                        
                        # Use EMA model for validation
                        # logits = self.ema(points)
                        
                        # if logits.dim() == 3:
                        #     logits = logits.mean(dim=1)
                        
                        # loss = self.loss_fn(logits, labels)
                        # accuracy = self.compute_accuracy(logits, labels)
                        
                        total_val_loss += (loss / num_val_batches)
                        total_val_accuracy += (accuracy / num_val_batches)

                self.print(get_current_time() + f' valid loss: {total_val_loss:.3f} valid acc: {total_val_accuracy:.3f} valid loss_recon: {loss_recon.item()} valid loss_cls: {loss_cls.item()} valid loss_kl: {loss_kl.item()}')
                
                # Calculate estimated finishing time
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
                
                val_log_dict = {
                    "val_loss": total_val_loss,
                    "val_accuracy": total_val_accuracy,
                    "val_loss_recon": loss_recon.item(),
                    "val_loss_cls": loss_cls.item(),
                    "val_loss_kl": loss_kl.item()
                }
                self.log(**val_log_dict)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.print('VAE V1 training complete') 