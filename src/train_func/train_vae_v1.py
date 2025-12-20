import os
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from omegaconf import OmegaConf

from src.dataset.dataset_v1 import  dataset_compound, dataset_compound_cache
from src.trainer.trainer_vae_v1 import Trainer_vae_v1 
from src.utils.config import NestedDictToClass, load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config

from icecream import ic
ic.disable()

program_parser = ArgumentParser(description='Train vae_v1 model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
program_parser.add_argument('--resume_lr', type=float, default=None, help='New learning rate to use when resuming from a checkpoint.')
cli_args, unknown = program_parser.parse_known_args()

cfg = load_config(cli_args.config)
args = NestedDictToClass(cfg)
config = OmegaConf.load(cli_args.config)

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = True
    # args.batch_size = 2
    # args.num_workers = 1


train_sampler = None

# Load datasets using dynamic loading from config
# Check if using separate data_train/data_val sections or unified data section


train_dataset = load_dataset_from_config(config, section='data_train')
val_dataset = load_dataset_from_config(config, section='data_val')


# Load model dynamically from config
model_name = getattr(args.model, 'name', 'vae_v1')
print(f'Loading model: {model_name}')

# Detect if using FSQ based on model name or fsq_levels parameter
use_fsq = cfg['use_fsq']

model = load_model_from_config(config)



epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

trainer = Trainer_vae_v1(
    model,
    dataset=train_dataset,
    val_dataset=val_dataset,
    num_train_steps=num_train_steps,
    batch_size=batch_size,
    num_workers=args.num_workers,
    num_step_per_epoch=num_step_per_epoch,
    loss_recon_weight=args.loss.recon_weight,
    loss_cls_weight=args.loss.cls_weight,
    loss_kl_weight=args.loss.kl_weight,
    loss_l2norm_weight=args.loss.l2norm_weight,
    loss_is_closed_weight=args.loss.is_closed_weight,
    pred_is_closed=args.model.pred_is_closed,
    kl_annealing_steps=getattr(args.loss, 'kl_annealing_steps', 0),
    kl_free_bits=getattr(args.loss, 'kl_free_bits', 0.0),
    u_closed_pos_weight=getattr(args.trainer, 'u_closed_pos_weight', 1.0),
    v_closed_pos_weight=getattr(args.trainer, 'v_closed_pos_weight', 1.0),
    grad_accum_every=args.grad_accum_every,
    ema_update_every=args.ema_update_every,
    learning_rate=initial_lr,
    max_grad_norm=args.max_grad_norm,
    accelerator_kwargs=dict(
        cpu=False,
        step_scheduler_with_optimizer=False
    ),
    log_every_step=args.log_every_step,
    use_wandb_tracking=args.use_wandb_tracking,
    checkpoint_folder=args.model.checkpoint_folder,
    checkpoint_every_step=args.save_every_epoch * num_step_per_epoch,
    resume_training=args.resume_training,
    from_start=args.from_start,
    checkpoint_file_name=args.model.checkpoint_file_name,
    val_every_step=int(args.val_every_epoch * num_step_per_epoch),
    use_logvar=args.trainer.use_logvar,
    num_workers_val=args.num_workers_val,
    train_sampler=train_sampler,
    use_fsq=use_fsq  # Add FSQ flag
)

if args.resume_training and cli_args.resume_lr is not None:
    new_lr = cli_args.resume_lr
    if trainer.is_main:
        trainer.print(f"Resuming training. Attempting to set new learning rate to: {new_lr}")

    trainer.wait()

    actual_optimizer = trainer.optimizer.optimizer 
    for param_group in actual_optimizer.param_groups:
        param_group['lr'] = new_lr
        if 'initial_lr' in param_group:
             param_group['initial_lr'] = new_lr

    actual_scheduler = trainer.optimizer.scheduler
    if hasattr(actual_scheduler, 'base_lrs'):
        actual_scheduler.base_lrs = [new_lr] * len(actual_scheduler.base_lrs)

    if trainer.is_main:
        trainer.print(f"Successfully set new learning rate to {new_lr} for optimizer and scheduler.")
        current_lr_after_set = actual_optimizer.param_groups[0]['lr']
        trainer.print(f"Current LR in optimizer after manual set: {current_lr_after_set}")

trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg) 