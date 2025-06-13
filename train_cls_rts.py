import os
import sys
from argparse import ArgumentParser

from src.flow.surface_flow import Encoder as MyModel
from src.dataset.dataset_cls_rts import SurfaceClassificationAndRegressionDataset as MyDataset
from src.trainer.trainer_cls_rts import TrainerClassificationAndRegression as MyTrainer
from src.utils.config import NestedDictToClass, load_config

program_parser = ArgumentParser(description='Train classification and regression model.')
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

transform = getattr(args.data, 'transform', None)
# if transform is None:
#     transform = None
# elif transform == 'scale_and_jitter':
#     transform = surface_scale_and_jitter
# elif transform == 'rotate':
#     transform = surface_rotate
# elif transform == 'scale_and_jitter_and_rotate':
#     transform = lambda x: surface_scale_and_jitter(surface_rotate(x))

train_dataset = MyDataset(
    data_path=args.data.train_data_path,
    replication=args.data.replication,
    transform=transform,
    is_train=True,
    res=args.model.res
)

val_dataset = MyDataset(
    data_path=args.data.val_data_path,
    is_train=False,
    res=args.model.res
)

model = MyModel(
    in_dim=args.model.in_dim,
    out_dim=args.model.out_dim,
    depth=args.model.depth,
    dim=args.model.dim,
    heads=args.model.heads,
    res=args.model.res
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

trainer = MyTrainer(
    model,
    dataset=train_dataset,
    val_dataset=val_dataset,
    num_train_steps=num_train_steps,
    batch_size=batch_size,
    num_workers=args.num_workers,
    num_step_per_epoch=num_step_per_epoch,
    grad_accum_every=args.grad_accum_every,
    ema_update_every=args.ema_update_every,
    learning_rate=initial_lr,
    rts_loss_weight=args.rts_loss_weight,
    cone_loss_weight=args.cone_loss_weight,
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