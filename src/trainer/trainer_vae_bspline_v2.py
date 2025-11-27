import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from typing import Optional, Union, Callable
import time
import wandb
from contextlib import nullcontext
from functools import partial
from collections import defaultdict
from src.trainer.trainer_base import BaseTrainer
from src.utils.helpers import divisible_by, get_current_time, get_lr, cycle
from src.dataset.dataset_bspline import dataset_bspline

class Trainer_vae_bspline(BaseTrainer):
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
        loss_poles_xyz_weight: float = 10.0,
        loss_cls_weight: float = 1.0,
        loss_kl_weight: float = 1.0,
        kl_annealing_steps: int = 0,
        kl_free_bits: float = 32,
        use_logvar: bool = False,
        # weighted sampling options
        weighted_sampling_enabled: bool = False,
        ws_warmup_epochs: int = 5,
        ws_alpha: float = 1.0,
        ws_beta: float = 0.9,
        ws_eps: float = 1e-6,
        ws_refresh_every: int = 1,
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
        
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')


        self.loss_recon_weight = loss_recon_weight
        self.loss_poles_xyz_weight = loss_poles_xyz_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_kl_weight = loss_kl_weight
        
        # KL annealing and free bits parameters
        self.kl_annealing_steps = kl_annealing_steps
        self.kl_free_bits = kl_free_bits
        self.use_logvar = bool(use_logvar)
        
        # Track abnormal values for logging
        self.abnormal_values_buffer = []
        
        # Preserve original dataloader hyperparams for later rebuilds
        self._train_batch_size = batch_size
        self._train_num_workers = num_workers
        
        # weighted sampling state
        self.ws_enabled = bool(weighted_sampling_enabled)
        self.ws_warmup_epochs = int(ws_warmup_epochs)
        self.ws_alpha = float(ws_alpha)
        self.ws_beta = float(ws_beta)
        self.ws_eps = float(ws_eps)
        self.ws_refresh_every = int(ws_refresh_every)
        if self.ws_enabled:
            ds = self.train_dl.dataset
            self.ws_base_len = int(getattr(ds, '_base_len', len(ds)))
            self.ws_replica = int(getattr(ds, 'replica', 1))
            self.ws_ema = torch.ones(self.ws_base_len, dtype=torch.float32)
        
        # Get the raw model for DDP
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model

    def compute_accuracy(self, deg_logits_u, u_degree, deg_logits_v, v_degree, peri_logits_u, is_u_periodic, peri_logits_v, is_v_periodic, knots_num_logits_u, num_knots_u, knots_num_logits_v, num_knots_v, mults_logits_u, u_mults_list, mults_logits_v, v_mults_list):
        deg_pred_u = torch.argmax(deg_logits_u, dim=-1)
        deg_pred_v = torch.argmax(deg_logits_v, dim=-1)
        peri_pred_u = torch.sigmoid(peri_logits_u) > 0.5
        peri_pred_v = torch.sigmoid(peri_logits_v) > 0.5
        knots_num_pred_u = torch.argmax(knots_num_logits_u, dim=-1)
        knots_num_pred_v = torch.argmax(knots_num_logits_v, dim=-1)
        mults_pred_u = torch.argmax(mults_logits_u, dim=-1)
        mults_pred_v = torch.argmax(mults_logits_v, dim=-1)

        acc_deg_u = (deg_pred_u == u_degree).float().mean()
        acc_deg_v = (deg_pred_v == v_degree).float().mean()
        acc_peri_u = (peri_pred_u == is_u_periodic).float().mean()
        acc_peri_v = (peri_pred_v == is_v_periodic).float().mean()
        acc_knots_num_u = (knots_num_pred_u == num_knots_u).float().mean()
        acc_knots_num_v = (knots_num_pred_v == num_knots_v).float().mean()

        mults_u_mask = torch.arange(self.raw_model.max_num_u_knots, device=u_mults_list.device).unsqueeze(0) < num_knots_u.unsqueeze(1)
        mults_v_mask = torch.arange(self.raw_model.max_num_v_knots, device=v_mults_list.device).unsqueeze(0) < num_knots_v.unsqueeze(1)

        acc_mults_u = ((mults_pred_u == u_mults_list) * mults_u_mask.view(-1)).float().sum() / num_knots_u.sum()
        acc_mults_v = ((mults_pred_v == v_mults_list) * mults_v_mask.view(-1)).float().sum() / num_knots_v.sum()

        return {
            "acc_deg_u": acc_deg_u,
            "acc_deg_v": acc_deg_v,
            "acc_peri_u": acc_peri_u,
            "acc_peri_v": acc_peri_v,
            "acc_knots_num_u": acc_knots_num_u,
            "acc_knots_num_v": acc_knots_num_v,
            "acc_mults_u": acc_mults_u,
            "acc_mults_v": acc_mults_v
        }

    def compute_kl_annealing_beta(self, step):
        """
        Compute KL annealing weight using linear schedule.
        Returns value between 0 and 1 that will be multiplied by loss_kl_weight.
        """
        if self.kl_annealing_steps <= 0:
            return 1.0
        return min(1.0, step / self.kl_annealing_steps)

    def log_loss(self, total_loss, accuracy, loss_recon, loss_cls, loss_kl, lr, total_norm, step, time_per_step, kl_beta=1.0, active_dims=None, additional_log_losses={}):
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'loss_recon': loss_recon,
            'loss_cls': loss_cls,
            'loss_kl': loss_kl,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
            'time_per_step': time_per_step,
            'kl_beta': kl_beta,
            **additional_log_losses,
            **accuracy
        }
        
        if active_dims is not None:
            log_dict['active_latent_dims'] = active_dims
        
        self.log(**log_dict)
        
        self.print(get_current_time() + f' loss: {total_loss:.3f}  lr: {lr:.6f} norm: {total_norm:.3f} kl_beta: {kl_beta:.3f} step: {step} t: {time_per_step:.2f}s')

    def train(self):

        step = self.step.item()
        if self.enable_profiler:
            self.start_profiler()


        while step < self.num_train_steps:
            t = time.time()

            total_loss = 0.
            total_accuracy = defaultdict(float)

            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.raw_model) if not is_last else nullcontext

                forward_args = self.next_data_to_forward_kwargs(self.train_dl_iter)

                valid = forward_args[-1].bool()
                forward_args = [_[valid] for _ in forward_args[:-1]]
                if self.ws_enabled:
                    idx = forward_args[-1].long()
                    forward_args = forward_args[:-1]
                forward_args = [_.unsqueeze(-1) if len(_.shape) == 1 else _ for _ in forward_args]
                forward_args = [_.float() if _.dtype == torch.float64 else _ for _ in forward_args]
                u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles = forward_args
                B = u_degree.shape[0]

                with self.accelerator.autocast(), maybe_no_sync():

  

                    if B == 0:
                        # No valid data in this batch
                        continue
                    u_mults_list = u_mults_list.long()
                    v_mults_list = v_mults_list.long()
                    u_degree -= 1 # Start from 0
                    v_degree -= 1 # Start from 0
                    u_mults_list[u_mults_list > 0] -= 1 # Start from 0
                    v_mults_list[v_mults_list > 0] -= 1 # Start form 0
                    num_knots_u = num_knots_u.long()
                    num_knots_v = num_knots_v.long()
                    num_poles_u = num_poles_u.long()
                    num_poles_v = num_poles_v.long()
                    mu, logvar = self.raw_model.encode(u_knots_list, u_mults_list, v_knots_list, v_mults_list, poles, u_degree, v_degree, is_u_periodic, is_v_periodic, num_knots_u, num_knots_v, num_poles_u, num_poles_v)
                    z = self.raw_model.reparameterize(mu, logvar) if self.use_logvar else mu
                    deg_logits_u, deg_logits_v, peri_logits_u, peri_logits_v, knots_num_logits_u, knots_num_logits_v, pred_knots_u, pred_knots_v, mults_logits_u, mults_logits_v, pred_poles = self.raw_model.decode(z, num_knots_u, num_knots_v, num_poles_u, num_poles_v)
                            
                    # Forward pass
                    loss_deg_u = self.ce_loss(deg_logits_u, u_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
                    loss_deg_v = self.ce_loss(deg_logits_v, v_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
                    loss_peri_u = self.bce_logits_loss(peri_logits_u.squeeze(-1), is_u_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
                    loss_peri_v = self.bce_logits_loss(peri_logits_v.squeeze(-1), is_v_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
                    loss_knots_num_u = self.ce_loss(knots_num_logits_u, num_knots_u.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots
                    loss_knots_num_v = self.ce_loss(knots_num_logits_v, num_knots_v.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots

                    # Need knots mask
                    mask_u_knots = torch.arange(self.raw_model.max_num_u_knots, device=num_knots_u.device).unsqueeze(0).repeat(num_knots_u.shape[0], 1) < num_knots_u # 1 for valid pos, 0 for invalid
                    mask_v_knots = torch.arange(self.raw_model.max_num_v_knots, device=num_knots_v.device).unsqueeze(0).repeat(num_knots_v.shape[0], 1) < num_knots_v # 1 for valid pos, 0 for invalid
                    
                    # Knots loss
                    loss_knots_u = self.mse_loss(pred_knots_u, u_knots_list) # Mean Squared Error Loss for knots
                    loss_knots_v = self.mse_loss(pred_knots_v, v_knots_list) # Mean Squared Error Loss for knots
                    
                    loss_knots_u = (loss_knots_u * mask_u_knots / num_knots_u).sum(dim=-1).mean() # Average over the valid positions
                    loss_knots_v = (loss_knots_v * mask_v_knots / num_knots_v).sum(dim=-1).mean()

                    # Mults loss
                    mults_logits_u = mults_logits_u.view(-1, self.raw_model.max_degree + 1)
                    mults_logits_v = mults_logits_v.view(-1, self.raw_model.max_degree + 1)
                    u_mults_list = u_mults_list.view(-1)
                    v_mults_list = v_mults_list.view(-1)
                    loss_mults_u = self.ce_loss(mults_logits_u, u_mults_list) # CE Loss for mults
                    loss_mults_v = self.ce_loss(mults_logits_v, v_mults_list) # CE Loss for mults

                    loss_mults_u = loss_mults_u * mask_u_knots.view(-1)
                    loss_mults_v = loss_mults_v * mask_v_knots.view(-1)
                    loss_mults_u = loss_mults_u.sum() / mask_u_knots.sum()
                    loss_mults_v = loss_mults_v.sum() / mask_v_knots.sum()



                    # Need poles mask
                    loss_poles = self.mse_loss(pred_poles, poles) # Mean Squared Error Loss for poles
                    mask_poles_u = torch.arange(self.raw_model.max_num_u_poles, device=num_poles_u.device).unsqueeze(0).repeat(num_poles_u.shape[0], 1) < num_poles_u # 1 for valid pos, 0 for invalid
                    mask_poles_v = torch.arange(self.raw_model.max_num_v_poles, device=num_poles_v.device).unsqueeze(0).repeat(num_poles_v.shape[0], 1) < num_poles_v # 1 for valid pos, 0 for invalid
                    
                    mask_poles_2d = mask_poles_u.unsqueeze(-1) & mask_poles_v.unsqueeze(-2)  # (B, H, W)
                    mask_poles_4d = mask_poles_2d.unsqueeze(-1)  # (B, H, W, 1) → broadcast to 4 coords
                    loss_poles_xyz = (loss_poles[..., :3] * mask_poles_4d).sum() / (mask_poles_4d.sum().clamp(min=1))
                    loss_poles_w = (loss_poles[..., 3:] * mask_poles_4d).sum() / (mask_poles_4d.sum().clamp(min=1))
                    loss_poles = loss_poles_xyz * self.loss_poles_xyz_weight + loss_poles_w
                    
                    # per-sample EMA update for weighted sampling
                    if self.ws_enabled:
                        Bv = mask_poles_4d.shape[0]
                        mask_flat = mask_poles_4d.view(Bv, -1, 1)
                        mse_all = self.mse_loss(pred_poles, poles)
                        xyz_flat = mse_all[..., :3].view(Bv, -1, 3)
                        w_flat = mse_all[..., 3:].view(Bv, -1, 1)
                        sum_xyz = (xyz_flat * mask_flat).sum(dim=(1, 2))
                        sum_w = (w_flat * mask_flat).sum(dim=(1, 2))
                        den = mask_poles_4d.view(Bv, -1).sum(dim=1).clamp(min=1)
                        per_sample_poles = (sum_xyz / den) * self.loss_poles_xyz_weight + (sum_w / den)
                        idx_cpu = idx.squeeze(-1).detach().to('cpu')
                        per_cpu = per_sample_poles.detach().to('cpu')
                        self.ws_ema[idx_cpu] = self.ws_beta * self.ws_ema[idx_cpu] + (1.0 - self.ws_beta) * per_cpu
                    loss_recon = loss_knots_u + loss_knots_v + loss_poles
                    loss_cls = loss_deg_u + loss_deg_v + loss_peri_u + loss_peri_v + loss_knots_num_u + loss_knots_num_v + loss_mults_u + loss_mults_v
                    
                    # # Compute KL loss with free bits strategy
                    # # Per-dimension KL: [batch, latent_dim]
                    # kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # # Apply free bits: only penalize KL above the threshold
                    # if self.kl_free_bits > 0:
                    #     kl_per_dim = torch.clamp(kl_per_dim - self.kl_free_bits, min=0.0)
                    
                    # Replace your current KL code with:
                    kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # (B,)
                    if self.kl_free_bits > 0:
                        kl_per_sample = torch.clamp(kl_per_sample - self.kl_free_bits, min=0.0)
                    loss_kl = kl_per_sample.mean()  # scalar
                    
                    # Compute KL annealing beta
                    kl_beta = self.compute_kl_annealing_beta(step)
                    
                    # Final loss with annealing
                    loss = loss_recon * self.loss_recon_weight + loss_cls * self.loss_cls_weight + loss_kl * self.loss_kl_weight * kl_beta
                    

                    # TODO: compute all acc
                    accuracy_kwargs = self.compute_accuracy(deg_logits_u, u_degree.squeeze(-1), deg_logits_v, v_degree.squeeze(-1), peri_logits_u, is_u_periodic.squeeze(-1), peri_logits_v, is_v_periodic.squeeze(-1), knots_num_logits_u, num_knots_u.squeeze(-1), knots_num_logits_v, num_knots_v.squeeze(-1), mults_logits_u, u_mults_list, mults_logits_v, v_mults_list)


                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                    # total_accuracy += accuracy.item() / self.grad_accum_every
                    for key, value in accuracy_kwargs.items():
                        total_accuracy[key] += value.item() / self.grad_accum_every
                
                self.accelerator.backward(loss)
                time_per_step = time.time() - t
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                
                # Compute active dimensions if free bits is enabled
                active_dims = None
                # if self.kl_free_bits > 0:
                #     # Count dimensions with KL above threshold
                #     kl_per_dim_mean = kl_per_dim.mean(dim=0)  # Average over batch
                #     active_dims = (kl_per_dim_mean > 0).float().sum().item()
                additional_log_losses = {
                    "loss_deg_u": loss_deg_u.item(),
                    "loss_deg_v": loss_deg_v.item(),
                    "loss_peri_u": loss_peri_u.item(),
                    "loss_peri_v": loss_peri_v.item(),
                    "loss_knots_num_u": loss_knots_num_u.item(),
                    "loss_knots_num_v": loss_knots_num_v.item(),
                    "loss_knots_u": loss_knots_u.item(),
                    "loss_knots_v": loss_knots_v.item(),
                    "loss_mults_u": loss_mults_u.item(),
                    "loss_mults_v": loss_mults_v.item(),
                    "loss_poles": loss_poles.item(),
                }                                        
                self.log_loss(total_loss, total_accuracy, loss_recon.item(), loss_cls.item(), loss_kl.item(), cur_lr, total_norm, step, time_per_step, kl_beta, active_dims, additional_log_losses=additional_log_losses)
                steps_remaining = self.num_train_steps - step
                time_per_step = (time.time() - self.start_time) / (step + 1)
                estimated_time_remaining = steps_remaining * time_per_step
                estimated_finish_time = time.time() + estimated_time_remaining
                self.print(get_current_time() + f' estimated finish time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(estimated_finish_time))}')
            step += 1
            self.step.add_(1)
            
            self.profiler_step()


            self.wait()
            
            if self.is_main:
                self.ema.update()            

            self.wait()

            # Validation
            if self.is_main and self.should_validate and divisible_by(step, self.val_every_step):
                print(f"Validating at step {step}")
                total_val_loss = 0.
                total_val_accuracy = defaultdict(float)
                device = next(self.model.parameters()).device
                
                self.ema.eval()
                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        forward_args = self.next_data_to_forward_kwargs(self.val_dl_iter)
                        valid = forward_args[-1].bool()
                        forward_args = [_[valid] for _ in forward_args[:-1]]
                        forward_args = [_.unsqueeze(-1) if len(_.shape) == 1 else _ for _ in forward_args]
                        forward_args = [_.float() if _.dtype == torch.float64 else _ for _ in forward_args]
                        forward_args = [_.to(device) for _ in forward_args]
                        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles = forward_args
                        B = u_degree.shape[0]

                        if B == 0:
                        # No valid data in this batch
                            continue
                        u_mults_list = u_mults_list.long()
                        v_mults_list = v_mults_list.long()
                        u_degree -= 1 # Start from 0
                        v_degree -= 1 # Start from 0
                        u_mults_list[u_mults_list > 0] -= 1 # Start from 0
                        v_mults_list[v_mults_list > 0] -= 1 # Start form 0
                        num_knots_u = num_knots_u.long()
                        num_knots_v = num_knots_v.long()
                        num_poles_u = num_poles_u.long()
                        num_poles_v = num_poles_v.long()
                        mu, logvar = self.ema.model.encode(u_knots_list, u_mults_list, v_knots_list, v_mults_list, poles, u_degree, v_degree, is_u_periodic, is_v_periodic, num_knots_u, num_knots_v, num_poles_u, num_poles_v)
                        z = self.ema.model.reparameterize(mu, logvar) if self.use_logvar else mu
                        # TODO: use predicted shapes
                        deg_logits_u, deg_logits_v, peri_logits_u, peri_logits_v, knots_num_logits_u, knots_num_logits_v, pred_knots_u, pred_knots_v, mults_logits_u, mults_logits_v, pred_poles = self.ema.model.decode(z, num_knots_u, num_knots_v, num_poles_u, num_poles_v)


                        loss_deg_u = self.ce_loss(deg_logits_u, u_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
                        loss_deg_v = self.ce_loss(deg_logits_v, v_degree.squeeze(-1)).mean() # Cross Entropy Loss for degree, max_degree - 1
                        loss_peri_u = self.bce_logits_loss(peri_logits_u.squeeze(-1), is_u_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
                        loss_peri_v = self.bce_logits_loss(peri_logits_v.squeeze(-1), is_v_periodic.squeeze(-1).float()).mean() # Binary Cross Entropy Loss for periodic
                        loss_knots_num_u = self.ce_loss(knots_num_logits_u, num_knots_u.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots
                        loss_knots_num_v = self.ce_loss(knots_num_logits_v, num_knots_v.squeeze(-1)).mean() # Mean Squared Error Loss for number of knots

                        # Need knots mask
                        mask_u_knots = torch.arange(self.raw_model.max_num_u_knots, device=num_knots_u.device).unsqueeze(0).repeat(num_knots_u.shape[0], 1) < num_knots_u # 1 for valid pos, 0 for invalid
                        mask_v_knots = torch.arange(self.raw_model.max_num_v_knots, device=num_knots_v.device).unsqueeze(0).repeat(num_knots_v.shape[0], 1) < num_knots_v # 1 for valid pos, 0 for invalid
                        
                        # Knots loss
                        loss_knots_u = self.mse_loss(pred_knots_u, u_knots_list) # Mean Squared Error Loss for knots
                        loss_knots_v = self.mse_loss(pred_knots_v, v_knots_list) # Mean Squared Error Loss for knots
                        
                        loss_knots_u = (loss_knots_u * mask_u_knots / num_knots_u).sum(dim=-1).mean() # Average over the valid positions
                        loss_knots_v = (loss_knots_v * mask_v_knots / num_knots_v).sum(dim=-1).mean()

                        # Mults loss
                        mults_logits_u = mults_logits_u.view(-1, self.raw_model.max_degree + 1)
                        mults_logits_v = mults_logits_v.view(-1, self.raw_model.max_degree + 1)
                        u_mults_list = u_mults_list.view(-1)
                        v_mults_list = v_mults_list.view(-1)
                        loss_mults_u = self.ce_loss(mults_logits_u, u_mults_list) # Mean Squared Error Loss for mults
                        loss_mults_v = self.ce_loss(mults_logits_v, v_mults_list) # Mean Squared Error Loss for mults

                        loss_mults_u = loss_mults_u * mask_u_knots.view(-1)
                        loss_mults_v = loss_mults_v * mask_v_knots.view(-1)
                        loss_mults_u = loss_mults_u.sum() / mask_u_knots.sum()
                        loss_mults_v = loss_mults_v.sum() / mask_v_knots.sum()



                        # Need poles mask
                        loss_poles = self.mse_loss(pred_poles, poles) # Mean Squared Error Loss for poles
                        mask_poles_u = torch.arange(self.raw_model.max_num_u_poles, device=num_poles_u.device).unsqueeze(0).repeat(num_poles_u.shape[0], 1) < num_poles_u # 1 for valid pos, 0 for invalid
                        mask_poles_v = torch.arange(self.raw_model.max_num_v_poles, device=num_poles_v.device).unsqueeze(0).repeat(num_poles_v.shape[0], 1) < num_poles_v # 1 for valid pos, 0 for invalid
                        
                        mask_poles_2d = mask_poles_u.unsqueeze(-1) & mask_poles_v.unsqueeze(-2)  # (B, H, W)
                        mask_poles_4d = mask_poles_2d.unsqueeze(-1)  # (B, H, W, 1) → broadcast to 4 coords
                        loss_poles = (loss_poles * mask_poles_4d).sum() / (mask_poles_4d.sum().clamp(min=1))
                        
                        loss_recon = loss_knots_u + loss_knots_v + loss_poles
                        loss_cls = loss_deg_u + loss_deg_v + loss_peri_u + loss_peri_v + loss_knots_num_u + loss_knots_num_v + loss_mults_u + loss_mults_v
                     
                        
                        # Compute KL loss with free bits (same as training)
                        # kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                        # if self.kl_free_bits > 0:
                        #     kl_per_dim = torch.clamp(kl_per_dim - self.kl_free_bits, min=0.0)
                        # loss_kl = torch.mean(kl_per_dim)
                        kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # (B,)
                        if self.kl_free_bits > 0:
                            kl_per_sample = torch.clamp(kl_per_sample - self.kl_free_bits, min=0.0)
                        loss_kl = kl_per_sample.mean()  # scalar
                        
                        # Use current annealing beta for validation loss
                        val_kl_beta = self.compute_kl_annealing_beta(step)
                        loss = loss_recon * self.loss_recon_weight + loss_cls * self.loss_cls_weight + loss_kl * self.loss_kl_weight * val_kl_beta
                        
                        # TODO compute all accuracy
                        accuracy_kwargs = self.compute_accuracy(deg_logits_u, u_degree.squeeze(-1), deg_logits_v, v_degree.squeeze(-1), peri_logits_u, is_u_periodic.squeeze(-1), peri_logits_v, is_v_periodic.squeeze(-1), knots_num_logits_u, num_knots_u.squeeze(-1), knots_num_logits_v, num_knots_v.squeeze(-1), mults_logits_u, u_mults_list, mults_logits_v, v_mults_list)

                        total_val_loss += (loss / num_val_batches)
                        for key, value in accuracy_kwargs.items():
                            total_val_accuracy[key] += value.item() / num_val_batches

                self.print(get_current_time() + f' valid loss: {total_val_loss:.3f} valid loss_recon: {loss_recon.item()} valid loss_cls: {loss_cls.item()} valid loss_kl: {loss_kl.item()}')
                
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
                    "val_loss_kl": loss_kl.item(),
                    "val_kl_beta": val_kl_beta,
                }
                self.log(**val_log_dict)

            self.wait()
            
            # end-of-epoch hooks and optional weighted sampler refresh
            if divisible_by(step, self.num_step_per_epoch):
                if self.is_main:
                    print(get_current_time() + f' {step // self.num_step_per_epoch} epoch at ', step)
                    if (step // self.num_step_per_epoch) % 5 == 0:
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if self.ws_enabled:
                    current_epoch = step // self.num_step_per_epoch
                    if current_epoch >= self.ws_warmup_epochs and divisible_by(current_epoch + 1, self.ws_refresh_every):
                        with torch.no_grad():
                            # aggregate EMA across ranks to keep identical weights
                            ema_all = self.ws_ema.clone()
                            try:
                                import torch.distributed as dist
                                if dist.is_available() and dist.is_initialized():
                                    device = self.device
                                    ema_all = ema_all.to(device)
                                    dist.all_reduce(ema_all, op=dist.ReduceOp.SUM)
                                    ema_all = (ema_all / dist.get_world_size()).to('cpu')
                            except Exception:
                                # fallback: use local EMA
                                ema_all = self.ws_ema.clone()
                            base_w = (ema_all + self.ws_eps).pow(self.ws_alpha)
                            base_w = base_w / base_w.sum().clamp(min=1e-12)
                            full_len = len(self.train_dl.dataset)
                            if full_len == self.ws_base_len * self.ws_replica:
                                weights_full = base_w.repeat(self.ws_replica)
                            else:
                                idxs = torch.arange(full_len) % self.ws_base_len
                                weights_full = base_w[idxs]
                        sampler = WeightedRandomSampler(weights=weights_full.double(), num_samples=len(self.train_dl.dataset), replacement=True)
                        dl = DataLoader(
                            self.train_dl.dataset,
                            batch_size=self._train_batch_size,
                            shuffle=False,
                            sampler=sampler,
                            pin_memory=False,
                            num_workers=self._train_num_workers,
                            prefetch_factor=1,
                            persistent_workers=False,
                        )
                        dl = self.accelerator.prepare(dl)
                        self.train_dl = dl
                        self.train_dl_iter = cycle(self.train_dl)

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.stop_profiler()

        self.print('VAE Bspline training complete') 