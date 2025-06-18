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

class TrainerAdjacency(BaseTrainer):
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
        
        self.loss_fn_bce = nn.BCEWithLogitsLoss()


    def compute_accuracy(self, logits, labels):
        # For binary classification with BCELoss, use sigmoid + threshold
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = (predictions == labels.float()).float()
        return correct.mean()

    def log_loss(self, total_loss, accuracy, lr, total_norm, step):
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'accuracy': accuracy,
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
                    points, type, bbox, adj_mask, nearby_type, nearby_points, nearby_bbox, padding_mask = forward_kwargs.values()
                    
                    x = torch.cat([points.unsqueeze(1), nearby_points], dim=1)
                    padding_mask = torch.cat([torch.zeros(padding_mask.shape[0], 1).to(padding_mask.device), padding_mask], dim=1)
                    type = torch.cat([type, nearby_type], dim=1)
                    bbox = torch.cat([bbox.unsqueeze(1), nearby_bbox], dim=1)

                    
                    # Forward pass
                    adjacency_logits = self.model(x, padding_mask, type)
                    
                    loss = self.loss_fn_bce(adjacency_logits, adj_mask.float())
                    
                    accuracy = self.compute_accuracy(adjacency_logits, adj_mask)
                    

                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                    total_accuracy += accuracy.item() / self.grad_accum_every

                    # self.print(get_current_time() + f' loss: {loss.item():.3f} acc: {accuracy.item():.3f}')
                
                self.accelerator.backward(loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                                                        
                self.log_loss(total_loss, total_accuracy, cur_lr, total_norm, step)
            
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
                        forward_kwargs = {k: v.to(self.model.device) for k, v in forward_kwargs.items()}
                        points, type, bbox, adj_mask, nearby_type, nearby_points, nearby_bbox, padding_mask = forward_kwargs.values()

                        x = torch.cat([points.unsqueeze(1), nearby_points], dim=1)
                        padding_mask = torch.cat([torch.zeros(padding_mask.shape[0], 1).to(padding_mask.device), padding_mask], dim=1)
                        type = torch.cat([type, nearby_type], dim=1)
                        bbox = torch.cat([bbox.unsqueeze(1), nearby_bbox], dim=1)
                        
                        # Forward pass with EMA model
                        adjacency_logits = self.ema(x, padding_mask, type)
                        
                        loss = self.loss_fn_bce(adjacency_logits, adj_mask.float())

                        accuracy = self.compute_accuracy(adjacency_logits, adj_mask)
                        
                        # Calculate losses


                        # Total loss
                        total_val_loss += (loss / num_val_batches)
                        total_val_accuracy += (accuracy / num_val_batches)

                self.print(get_current_time() + f' valid loss: {total_val_loss:.3f} valid acc: {total_val_accuracy:.3f}')
                
                # Calculate estimated finishing time
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
                
                val_log_dict = {
                    "val_loss": total_val_loss,
                    "val_accuracy": total_val_accuracy
                }
                self.log(**val_log_dict)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.print('Classification training complete') 