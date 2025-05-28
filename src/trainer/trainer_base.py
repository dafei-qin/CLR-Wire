from typing import Callable
from pathlib import Path
from functools import partial
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from contextlib import nullcontext
import time

from beartype import beartype
from beartype.typing import Optional, Type

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ema_pytorch import EMA

from pytorch_custom_utils import (
    get_adam_optimizer,
    add_wandb_tracker_contextmanager
)

from src.utils.helpers import (
    cycle,
    divisible_by,
    get_current_time,
    exists,
    get_lr,
)

from src.utils.torch_tools import fmt

from src.optimizer.optimizer_scheduler import OptimizerWithScheduler

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters = False
)

@add_wandb_tracker_contextmanager()
class BaseTrainer(Module):
    @beartype
    def __init__(
        self,
        model: Module,
        dataset: Dataset,
        *,
        val_dataset: Optional[Dataset] = None,
        accelerator_kwargs: dict = dict(),
        amp = False,
        adam_betas = (0.9, 0.99),
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
        optimizer_kwargs: dict = dict(),
        resume_training = False,
        from_start = False,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        use_wandb_tracking: bool = False,
        weight_decay: float = 0.,
        collate_fn: Optional[Callable] = None,
        val_every_step: int = 1000,
        val_num_batches: int = 10,
        **kwargs  
    ):
        super().__init__()
        self.start_time = time.time()  # Initialize start time for training
        
        # experiment tracker

        self.use_wandb_tracking = use_wandb_tracking

        if self.use_wandb_tracking:
            accelerator_kwargs['log_with'] = 'wandb'

        self.log_every_step = log_every_step

        # accelerator
        
        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]        

        self.accelerator = Accelerator(
            **accelerator_kwargs,
        )

        # model

        self.model = model

        if self.is_main: self.print_params_num()

        # sampling and training hyperparameters

        self.grad_accum_every = grad_accum_every
        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps

        # dataset and dataloader

        self.train_dl = DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle=True,
            # sampler=self.train_sampler,
            pin_memory = False,  # Disable pin_memory to reduce RAM usage
            num_workers=min(num_workers, 8),  # Limit workers to prevent memory explosion
            collate_fn=collate_fn,
            prefetch_factor=1,  # Reduce prefetch to save memory
            persistent_workers=False,  # Disable persistent workers to allow cleanup
        )

        self.train_dl = self.accelerator.prepare(self.train_dl)
        self.train_dl_iter = cycle(self.train_dl)


        self.should_validate = exists(val_dataset)

        if self.should_validate and self.is_main:
            self.val_every_step = val_every_step
            # self.val_every_step = 20
            self.val_num_batches = val_num_batches
            self.val_dl = DataLoader(
                val_dataset, 
                batch_size = batch_size, 
                shuffle = True,
                pin_memory = False,  # Disable pin_memory to reduce RAM usage
                drop_last=True,
                num_workers=min(num_workers, 4),  # Even fewer workers for validation
                collate_fn=collate_fn,
                prefetch_factor=1,  # Reduce prefetch to save memory
                persistent_workers=False,  # Disable persistent workers to allow cleanup
            )
            # self.val_dl = self.accelerator.prepare(self.val_dl)
            self.val_dl_iter = cycle(self.val_dl)


        self.num_step_per_epoch = num_step_per_epoch // dataset.replica

        # optimizer

        optimizer = get_adam_optimizer(
            model.parameters(),
            lr = learning_rate,
            wd = weight_decay,
            filter_by_requires_grad = True, # filter ae model params
            **optimizer_kwargs
        )

        self.optimizer = OptimizerWithScheduler(
            accelerator = self.accelerator,
            optimizer = optimizer,
            scheduler = scheduler,
            scheduler_kwargs = scheduler_kwargs if len(scheduler_kwargs) > 0 else dict(num_train_steps = num_train_steps),
            max_grad_norm = max_grad_norm
        )

        # step counter state

        self.register_buffer('step', torch.tensor(0))     

        # prepare model, dataloader, optimizer with accelerator

        (
            self.model, 
            self.optimizer
        ) = self.accelerator.prepare(
            self.model, 
            self.optimizer
        )
        
        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        
        self.checkpoint_every_step = checkpoint_every_step
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)

        if resume_training:
            print("loading checkpoint from the file: ", checkpoint_file_name)
            self.load(checkpoint_file_name, from_start=from_start)

    def log(self, **data_kwargs):
        self.accelerator.log(data_kwargs, step = self.step.item())

    def print_params_num(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"autoencoder Total parameters: {total_params / 1e6} M")  

        non_trainable_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)        
        print(f"Number of non-trainable parameters: {non_trainable_params/ 1e6}") 

    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def next_data_to_forward_kwargs(self, dl_iter) -> dict:
        data = next(dl_iter)

        if isinstance(data, tuple):
            forward_kwargs = dict(zip(self.data_kwargs, data))
        else:
            forward_kwargs = data

        return forward_kwargs

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = dict(        
            step = self.step.item(),
            model = self.accelerator.get_state_dict(self.model),
            optimizer = self.optimizer.state_dict(),
            ema = self.ema.state_dict(),
            scaler = self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        )

        torch.save(data, str(self.checkpoint_folder / f'model-{milestone}.pt'))

    def load(self, file_name: str, from_start=False):
        accelerator = self.accelerator
        device = accelerator.device

        pkg = torch.load(str(self.checkpoint_folder / file_name), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(pkg['model'], strict=True)

        if not from_start:
            self.step.copy_(pkg['step'])
            self.optimizer.load_state_dict(pkg['optimizer'])
        
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(pkg["ema"], strict=True)

        if 'version' in pkg:
            print(f"loading from version {pkg['version']}")

        if exists(self.accelerator.scaler) and exists(pkg['scaler']):
            self.accelerator.scaler.load_state_dict(pkg['scaler'])
            
        print(f"loaded checkpoint from {self.checkpoint_folder / file_name}")
    
    def train_step(self, forward_kwargs, is_train=True):
        
        if is_train:
            model = self.model
        else:
            model = self.ema
        
        # Get current step for KL annealing
        current_step = self.step.item() if hasattr(self, 'step') else 0
        
        if isinstance(forward_kwargs, dict):
            loss, loss_dict = model(
                **forward_kwargs,
                sample_posterior=True,
                return_loss=True,
                training_step=current_step  # Pass training step for KL annealing
            )
        elif isinstance(forward_kwargs, torch.Tensor):
            loss, loss_dict = model(
                forward_kwargs,
                sample_posterior=True,
                return_loss=True,
                training_step=current_step  # Pass training step for KL annealing
            )   
        else:
            raise ValueError(f'unknown forward_kwargs')
        
        return loss, loss_dict

    
    def log_loss(self, loss, loss_dict=None, cur_lr=None, total_norm=None, step=None):
        log_data = {"total_loss": loss}
        
        if loss_dict is not None:
            log_data.update(loss_dict)
        
        if cur_lr is not None:
            log_data["cur_lr"] = cur_lr
        if total_norm is not None:
            log_data["total_norm"] = total_norm if exists(total_norm) else 0.0
        
        if not self.use_wandb_tracking:
            log_str = f"{step} | " + " | ".join(f"{k}: {fmt(v)}" for k,v in log_data.items())
            print(log_str)
        else:
            self.log(**log_data)
    
    
    def train(self):

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
                
                # Force garbage collection after checkpoint save to prevent memory buildup
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if divisible_by(step, self.num_step_per_epoch):
                if self.is_main:
                    print(get_current_time() + f' {step // self.num_step_per_epoch} epoch at ', step)
                    
                    # Periodic memory cleanup every epoch
                    if (step // self.num_step_per_epoch) % 5 == 0:  # Every 5 epochs
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            self.wait()

        # Make sure that the wandb tracker finishes correctly
        self.accelerator.end_training()

        self.print('training complete')
    
    def forward(self, project: str, run: str | None = None, hps: dict | None = None):
        if self.is_main and self.use_wandb_tracking:
            print('using wandb tracking')
            
            with self.wandb_tracking(project=project, run=run, hps=hps):
                self.train()
        else:
            print('not using wandb tracking')
            
            self.train()