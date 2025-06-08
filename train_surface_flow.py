import os
import sys
from argparse import ArgumentParser

from src.flow.surface_flow import ZLDM as MyModel
from src.dataset.dataset_flow_cp import surface_dataset_flow_cp as MyDataset
from src.trainer.trainer_flow_surface import TrainerFlowSurface as MyTrainer
from src.utils.config import NestedDictToClass, load_config
from src.dataset.dataset_fn import surface_control_points_scale_and_rotate, surface_samples_add_noise

# Arguments
program_parser = ArgumentParser(description='Train surface vae model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
program_parser.add_argument('--resume_lr', type=float, default=None, help='New learning rate to use when resuming from a checkpoint.')
cli_args, unknown = program_parser.parse_known_args()

cfg = load_config(cli_args.config)
args = NestedDictToClass(cfg)

isDebug = True if sys.gettrace() else False

if isDebug:
    args.use_wandb_tracking = True
    args.batch_size = 2
    args.num_workers = 1

train_dataset = MyDataset(
    cp_path = args.data.train_set_file_path,
    pc_path = getattr(args.data, 'train_pc_path', None),
    mask_prob=getattr(args.data, 'mask_prob', 0),
    mask_pattern=getattr(args.data, 'mask_pattern', None),
    replication=args.data.replication,
    num_samples=args.model.sample_points_num,
    is_train=True,
    transform_cp=surface_control_points_scale_and_rotate if args.data.transform_cp else None,
    transform_pc=surface_samples_add_noise if args.data.transform_pc else None
)


val_dataset = MyDataset(
    cp_path = args.data.val_set_file_path,
    pc_path = getattr(args.data, 'val_pc_path', None),
    mask_prob=getattr(args.data, 'mask_prob', 0),
    mask_pattern=getattr(args.data, 'mask_pattern', None),
    is_train=False,
    num_samples=args.model.sample_points_num,
)   

model = MyModel(
    depth=args.model.depth,
    dim=args.model.dim,
    latent_dim=args.model.latent_dim,
    heads=args.model.heads,
    pe=args.model.pe,
    res=args.model.sample_points_num,
    block_out_channels=args.model.block_out_channels,
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

# Determine initial learning rate: use resume_lr if provided and resuming, else use config LR
initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

trainer = MyTrainer(
    model,
    dataset = train_dataset,
    val_dataset = val_dataset,
    num_train_steps = num_train_steps,
    batch_size = batch_size,
    num_workers = args.num_workers,
    num_step_per_epoch=num_step_per_epoch,
    grad_accum_every = args.grad_accum_every,
    ema_update_every = args.ema_update_every,
    learning_rate = args.lr,
    max_grad_norm = args.max_grad_norm,
    accelerator_kwargs = dict(
        cpu = False,
        step_scheduler_with_optimizer=False
    ),
    log_every_step = args.log_every_step,
    use_wandb_tracking = args.use_wandb_tracking,
    checkpoint_folder = args.model.checkpoint_folder,
    checkpoint_every_step = args.save_every_epoch * num_step_per_epoch,
    resume_training=args.resume_training,
    from_start=args.from_start,
    checkpoint_file_name=args.model.checkpoint_file_name,
    val_every_step=int(args.val_every_epoch * num_step_per_epoch),
    visual_eval_every_step=getattr(args, 'visual_eval_every_step', 5000),
    num_visual_samples=getattr(args, 'num_visual_samples', 4),
    scheduler_type=args.scheduler_type,
    prediction_type=args.prediction_type,
)

# <<<< Add this section to modify LR after loading checkpoint >>>>
if args.resume_training and cli_args.resume_lr is not None:
    new_lr = cli_args.resume_lr
    if trainer.is_main: # Ensure this runs only on the main process
        trainer.print(f"Resuming training. Attempting to set new learning rate to: {new_lr}")

    # Wait for everyone to ensure checkpoint is loaded if distributed
    trainer.wait()

    actual_optimizer = trainer.optimizer.optimizer 
    for param_group in actual_optimizer.param_groups:
        param_group['lr'] = new_lr
        if 'initial_lr' in param_group: # Good practice to update this too
             param_group['initial_lr'] = new_lr

    actual_scheduler = trainer.optimizer.scheduler
    if hasattr(actual_scheduler, 'base_lrs'):
        actual_scheduler.base_lrs = [new_lr] * len(actual_scheduler.base_lrs)
    
    # If your scheduler has a warmup phase and you're significantly changing LR,
    # you *might* consider re-initializing the scheduler or adjusting its state,
    # but for LambdaLR (cosine), just updating base_lrs is usually sufficient.
    # The warmup phase will have already passed if resuming from step > num_warmup_steps.

    if trainer.is_main:
        trainer.print(f"Successfully set new learning rate to {new_lr} for optimizer and scheduler.")
        current_lr_after_set = actual_optimizer.param_groups[0]['lr']
        trainer.print(f"Current LR in optimizer after manual set: {current_lr_after_set}")
# <<<< End of section >>>>

trainer(project=args.wandb_project_name, run=args.wandb_run_name, hps=cfg)
