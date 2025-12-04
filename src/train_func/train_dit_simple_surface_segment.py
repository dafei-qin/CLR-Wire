import os
from pathlib import Path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import shutil

from argparse import ArgumentParser
from omegaconf import OmegaConf


from src.dataset.dataset_latent import LatentDataset
from src.utils.config import NestedDictToClass, load_config
from src.utils.import_tools import load_model_from_config, load_dataset_from_config 
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch

program_parser = ArgumentParser(description='Train bspline model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
program_parser.add_argument('--resume_lr', type=float, default=None, help='New learning rate to use when resuming from a checkpoint.')
cli_args, unknown = program_parser.parse_known_args()

cfg = load_config(cli_args.config)
args = NestedDictToClass(cfg)

use_logvar = (getattr(getattr(args, 'trainer', None), 'use_logvar', False) if hasattr(args, 'trainer') else False)
print(f"Use logvar: {use_logvar}")

model_name = args.model.name


# if model_name == 'dit_v2':
#     from src.dit.simple_surface_decoder_v2 import SimpleSurfaceDecoder as Model
#     print('Use the model: dit_v2')
# else:
#     print('Use the default model: dit_simple')
#     from src.dit.simple_surface_decoder import SimpleSurfaceDecoder as Model
     

if args.model.trainer_name == 'dit_v1':
    print('Use the trainer: dit_v1')
    from src.trainer.trainer_dit_simple_surface_segment import TrainerFlowSurface as TrainerFlowSurface
else:
    print('Use the trainer: dit_v1')
    from src.trainer.trainer_dit_simple_surface_segment import TrainerFlowSurface as TrainerFlowSurface

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = True
    args.batch_size = 2
    args.num_workers = 1

else:
    # Here we back-up the code to the ckpt folder.

    os.makedirs(args.model.checkpoint_folder, exist_ok=True)
    code_folder = project_root / 'src'
    shutil.copytree(code_folder, Path(args.model.checkpoint_folder) / 'code', dirs_exist_ok=True)
    shutil.copyfile(cli_args.config, Path(args.model.checkpoint_folder) / 'config.yaml')

transform = getattr(args.data, 'transform', None)
if transform is None:
    transform = None


# class IndexedDataset(torch.utils.data.Dataset):
#     def __init__(self, base: Dataset):
#         self.base = base
#         # forward important attributes transparently
#         self.replica = getattr(base, 'replica', 1)
#         # dataset_bspline exposes data_names
#         self._base_len = len(getattr(base, 'data_names', [])) if hasattr(base, 'data_names') else max(1, len(base) // self.replica)
#     def __len__(self):
#         return len(self.base)
#     def __getitem__(self, i):
#         sample = self.base[i]
#         # original sample: (..., valid)
#         # use base index so replicas share the same id
#         base_idx = i % self._base_len
#         return (*sample[:-1], torch.tensor(base_idx, dtype=torch.long), sample[-1])

config = OmegaConf.load(args.config)

vae_config = OmegaConf.load(config.vae.config_file)
vae = load_model_from_config(vae_config)

model = load_model_from_config(config.model)

train_dataset_raw = load_dataset_from_config(config.data_train)

val_dataset = load_dataset_from_config(config.data_val)


# train_dataset_raw = LatentDataset(
#     latent_dir=args.data.train_latent_dir, pc_dir=args.data.train_pc_dir, max_num_surfaces=args.data.max_num_surfaces, 
#     latent_dim=args.data.surface_latent_dim, num_data=args.data.train_num,
#     log_scale=args.data.log_scale,
#     replica=args.data.replica
#     )

# val_dataset = LatentDataset(
#     latent_dir=args.data.val_latent_dir, pc_dir=args.data.val_pc_dir, max_num_surfaces=args.data.max_num_surfaces, 
#     latent_dim=args.data.surface_latent_dim, num_data=args.data.val_num,
#     log_scale=args.data.log_scale,
#     replica=args.data.replica_val
#     )

if len(train_dataset_raw.latent_files) == 1:
    print(f'Overfitting with {len(train_dataset_raw.latent_files)} data:\n {train_dataset_raw.latent_files[0]}')
# weighted sampling config (optional)
ws_cfg = getattr(args.data, 'weighted_sampling', None)
ws_enabled = False
if ws_cfg is not None:
    ws_enabled = getattr(ws_cfg, 'enabled', False)

# train_dataset = IndexedDataset(train_dataset_raw) if ws_enabled else train_dataset_raw
train_dataset = train_dataset_raw

# model = Model(
#     input_dim=args.model.input_dim,
#     cond_dim=args.model.cond_dim,
#     output_dim=args.model.output_dim,
#     latent_dim=args.model.latent_dim,
#     num_layers=args.model.num_layers,
#     num_heads=args.model.num_heads
# )

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

trainer = TrainerFlowSurface(
    model,
    vae,
    dataset=train_dataset,
    val_dataset=val_dataset,
    num_train_steps=num_train_steps,
    prediction_type=args.trainer.prediction_type,
    batch_size=batch_size,
    num_workers=args.num_workers,
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
    # weighted sampling options
    weighted_sampling_enabled=ws_enabled,
    ws_warmup_epochs=getattr(ws_cfg, 'warmup_epochs', 5) if ws_cfg is not None else 5,
    ws_alpha=getattr(ws_cfg, 'alpha', 1.0) if ws_cfg is not None else 1.0,
    ws_beta=getattr(ws_cfg, 'beta', 0.9) if ws_cfg is not None else 0.9,
    ws_eps=getattr(ws_cfg, 'eps', 1e-6) if ws_cfg is not None else 1e-6,
    ws_refresh_every=getattr(ws_cfg, 'refresh_every', 1) if ws_cfg is not None else 1,
    weight_valid=args.loss.weight_valid,
    weight_params=args.loss.weight_params,
    weight_rotations=args.loss.weight_rotations,
    weight_scales=args.loss.weight_scales,
    weight_shifts=args.loss.weight_shifts,
    num_inference_timesteps=args.trainer.num_inference_timesteps,
    log_scale=config.data_train.params.log_scale
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