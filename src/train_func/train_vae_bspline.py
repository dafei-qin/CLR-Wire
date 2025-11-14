import os
from pathlib import Path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import shutil

from argparse import ArgumentParser


from src.dataset.dataset_bspline import dataset_bspline
from src.utils.config import NestedDictToClass, load_config

program_parser = ArgumentParser(description='Train bspline model.')
program_parser.add_argument('--config', type=str, default='', help='Path to config file.')
program_parser.add_argument('--resume_lr', type=float, default=None, help='New learning rate to use when resuming from a checkpoint.')
cli_args, unknown = program_parser.parse_known_args()

cfg = load_config(cli_args.config)
args = NestedDictToClass(cfg)


model_name = args.model.name

if args.model.name == 'vae_bspline_v1':
    from src.vae.vae_bspline import BSplineVAE as BSplineVAE
    print('Use the model: vae_bspline_v1')
elif args.model.name == 'vae_bspline_v3':
    from src.vae.vae_bspline_v3 import BSplineVAE as BSplineVAE
    print('Use the model: vae_bspline_v3')
elif model_name == "vae_bspline_v4":
    from src.vae.vae_bspline_v4 import BSplineVAE as BSplineVAE
elif model_name == "vae_bspline_v5":
    from src.vae.vae_bspline_v5 import BSplineVAE as BSplineVAE
else:
    print('Use the default model: vae_bspline_v1')
    from src.vae.vae_bspline import BSplineVAE as BSplineVAE


if args.model.trainer_name == 'vae_bspline_v2':
    print('Use the trainer: vae_bspline_v2')
    from src.trainer.trainer_vae_bspline_v2 import Trainer_vae_bspline as Trainer_vae_bspline
else:
    print('Use the trainer: vae_bspline')
    from src.trainer.trainer_vae_bspline import Trainer_vae_bspline as Trainer_vae_bspline

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


train_dataset = dataset_bspline(
    path_file=args.data.train_file, data_dir_override=args.data.train_data_dir_override, num_surfaces=args.data.train_num,
    max_num_u_knots=args.model.max_num_u_knots, max_num_v_knots=args.model.max_num_v_knots, max_num_u_poles=args.model.max_num_u_poles, max_num_v_poles=args.model.max_num_v_poles
    )

val_dataset = dataset_bspline(
    path_file=args.data.val_file, data_dir_override=args.data.val_data_dir_override, num_surfaces=args.data.val_num,
    max_num_u_knots=args.model.max_num_u_knots, max_num_v_knots=args.model.max_num_v_knots, max_num_u_poles=args.model.max_num_u_poles, max_num_v_poles=args.model.max_num_v_poles
    )

model = BSplineVAE(
    max_num_u_knots=args.model.max_num_u_knots,
    max_num_v_knots=args.model.max_num_v_knots,
    max_num_u_poles=args.model.max_num_u_poles,
    max_num_v_poles=args.model.max_num_v_poles,
)

epochs = args.epochs
batch_size = args.batch_size
num_gpus = args.num_gpus
num_step_per_epoch = int(train_dataset.__len__() / (batch_size * num_gpus))
num_train_steps = epochs * num_step_per_epoch

initial_lr = cli_args.resume_lr if args.resume_training and cli_args.resume_lr is not None else args.lr

trainer = Trainer_vae_bspline(
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
    kl_annealing_steps=getattr(args.loss, 'kl_annealing_steps', 0),
    kl_free_bits=getattr(args.loss, 'kl_free_bits', 0.0),
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