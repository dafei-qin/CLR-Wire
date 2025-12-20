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
    from src.trainer.trainer_dit_sample_segment import TrainerDITSampleSegment as TrainerDITSampleSegment
else:
    print('Use the trainer: dit_v1')
    from src.trainer.trainer_dit_sample_segment import TrainerDITSampleSegment as TrainerDITSampleSegment

isDebug = True if sys.gettrace() else False


# torch.autograd.set_detect_anomaly(True)
# print('WARNING: set detect anomaly')

if isDebug:
    args.use_wandb_tracking = False
    # args.batch_size = 512
    # args.num_workers = 1

else:
    # Here we back-up the code to the ckpt folder.

    os.makedirs(args.model.checkpoint_folder, exist_ok=True)
    code_folder = project_root / 'src'
    shutil.copytree(code_folder, Path(args.model.checkpoint_folder) / 'code', dirs_exist_ok=True)
    shutil.copyfile(cli_args.config, Path(args.model.checkpoint_folder) / 'config.yaml')

# transform = getattr(args.data, 'transform', None)
# if transform is None:
#     transform = None


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

config = OmegaConf.load(cli_args.config)

vae_config = OmegaConf.load(config.vae.config_file)
vae = load_model_from_config(vae_config)

model = load_model_from_config(config)

train_dataset_raw = load_dataset_from_config(config, section='data_train')

val_dataset = load_dataset_from_config(config, section='data_val')


train_dataset = train_dataset_raw


epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

print('Warning, here we "fixed" the memory mask to be reversed.')
trainer = TrainerDITSampleSegment(
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
    weight_valid=args.loss.weight_valid,
    weight_params=args.loss.weight_params,
    weight_rotations=args.loss.weight_rotations,
    weight_scales=args.loss.weight_scales,
    weight_shifts=args.loss.weight_shifts,
    weight_original_sample=args.loss.weight_original_sample,
    original_sample_start_step=args.loss.original_sample_start_step,
    weight_sample_edges=args.loss.weight_sample_edges,
    num_inference_timesteps=args.trainer.num_inference_timesteps,
    use_weighted_sample_loss=args.loss.use_weighted_sample_loss,
    log_scale=config.log_scale
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