from typing import Type
import math
from functools import partial

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from accelerate import Accelerator
import torch

from src.utils.helpers import exists

# constants

ConstantLRScheduler = partial(LambdaLR, lr_lambda = lambda step: 1.)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    
    if num_warmup_steps is None:
        num_warmup_steps = int(num_training_steps * 0.1)
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps)) * (1 - 0.01) + 0.01
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1 - 0.01) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) + 0.01


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_train_steps: int, num_warmup_steps: int = None, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



# optimizer with scheduler

class OptimizerWithScheduler(nn.Module):
    def __init__(
        self,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: Type[_LRScheduler] | None = None,
        scheduler_kwargs: dict = dict(),
        max_grad_norm: float | None = 5.0,
    ):
        super().__init__()
        self.max_grad_norm = max_grad_norm

        if exists(scheduler):
            self.scheduler = scheduler(optimizer, **scheduler_kwargs)
        else:
            self.scheduler = get_cosine_schedule_with_warmup(optimizer, **scheduler_kwargs)

        self.optimizer = optimizer

        self.optimizer, self.scheduler = accelerator.prepare(self.optimizer, self.scheduler)
        self.accelerator = accelerator
        self.total_norm = None

    def state_dict(self):
        return dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
        )

    def load_state_dict(self, pkg):
        self.optimizer.load_state_dict(pkg['optimizer'])
        self.scheduler.load_state_dict(pkg['scheduler'])

    def zero_grad(self, current_step: int = -1):
        self.optimizer.zero_grad()

    def step(self, current_step: int = -1):
        if exists(self.max_grad_norm):
            # for param_group in self.optimizer.param_groups:
            all_params = [p for param_group in self.optimizer.param_groups for p in param_group['params'] if p.grad is not None] # Ensure p.grad is not None
            
            if all_params: # Only proceed if there are parameters with gradients
                # Calculate norm before clipping for logging
                norm_before_clip = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in all_params]), 2.0).item()
                # if self.accelerator.is_main_process: # Log only on main process
                #     print(f"Step {current_step}: Grad norm BEFORE clip: {norm_before_clip:.4f}, max_grad_norm: {self.max_grad_norm}")

                total_norm = self.accelerator.clip_grad_norm_(all_params, self.max_grad_norm)
                
                # if self.accelerator.is_main_process: # Log only on main process
                #     print(f"Step {current_step}: Grad norm AFTER clip: {total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm:.4f}")
                self.total_norm = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm # Store the scalar value
            else:
                self.total_norm = 0.0 # No gradients to clip

        self.optimizer.step()

        if not self.accelerator.optimizer_step_was_skipped:
            self.scheduler.step()
        else:
            if self.accelerator.is_main_process:
                print(f"Step {current_step}: Optimizer step SKIPPED due to inf/NaN gradients")
