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
from src.dataset.dataset_v1 import get_surface_type

class Trainer_vae_v1(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset, 
        *,
        val_dataset: Dataset,
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
        val_batch_size: int = 256,
        accelerator_kwargs: dict = dict(),
        loss_recon_weight: float = 1.0,
        loss_cls_weight: float = 1.0,
        loss_kl_weight: float = 1.0,
        loss_l2norm_weight: float = 0.0,
        kl_annealing_steps: int = 0,
        kl_free_bits: float = 0.0,
        use_logvar: bool = True,
        train_sampler = None,
        num_workers_val: int = 8,
        pred_is_closed = False,
        is_closed_weight = 1.0,
        u_closed_pos_weight: float = 1.0,
        v_closed_pos_weight: float = 1.0,
        type_weight: Optional[dict] = None,  # Per-type reconstruction weight
        use_fsq: bool = False,  # Whether to use FSQ instead of VAE
        **kwargs
    ):
        super().__init__(
            model=model, 
            dataset=dataset,
            val_dataset=val_dataset,
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
            val_batch_size=val_batch_size,
            train_sampler = train_sampler,
            num_workers_val=num_workers_val,
            **kwargs
        )
        
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_recon = nn.MSELoss(reduction='none')  # Use 'none' for per-sample weighting
        self.loss_recon_weight = loss_recon_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_kl_weight = loss_kl_weight
        self.loss_l2norm_weight = loss_l2norm_weight
        self.loss_is_closed_weight = is_closed_weight
        self.pred_is_closed = pred_is_closed
        self.u_closed_pos_weight = u_closed_pos_weight
        self.v_closed_pos_weight = v_closed_pos_weight

        # Convert type_weight dict to tensor
        # Map: plane(0), cylinder(1), cone(2), sphere(3), torus(4), bspline_surface(5)
        if type_weight is not None:
            type_name_to_idx = {
                'plane': 0,
                'cylinder': 1, 
                'cone': 2,
                'sphere': 3,
                'torus': 4,
                'bspline_surface': 5,
            }
            # Create weight tensor indexed by surface type
            type_weight_tensor = torch.ones(6, dtype=torch.float32)
            # Convert NestedDictToClass to dict if needed
            if hasattr(type_weight, '__dict__'):
                type_weight_dict = vars(type_weight)
            elif isinstance(type_weight, dict):
                type_weight_dict = type_weight
            else:
                type_weight_dict = dict(type_weight)
            
            for type_name, weight in type_weight_dict.items():
                if type_name in type_name_to_idx:
                    type_weight_tensor[type_name_to_idx[type_name]] = weight
            self.type_weight = nn.Parameter(type_weight_tensor.cuda(), requires_grad=False)
            print(f"✓ Per-type reconstruction weights: {type_weight_dict}")
        else:
            self.type_weight = None

        if self.pred_is_closed:
            self.pos_weight = nn.Parameter(torch.tensor([u_closed_pos_weight, v_closed_pos_weight], dtype=torch.float32).unsqueeze(0).cuda(), requires_grad=False
            )
            self.loss_is_closed = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            

        
        # KL annealing and free bits parameters
        self.kl_annealing_steps = kl_annealing_steps
        self.kl_free_bits = kl_free_bits
        
        # Track abnormal values for logging
        self.abnormal_values_buffer = []
        self.raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        self.use_logvar = use_logvar
        self.use_fsq = use_fsq  # FSQ mode flag
        
        # Validate FSQ configuration
        if self.use_fsq:
            if self.loss_l2norm_weight > 0:
                print(f"⚠️  Warning: L2 norm loss (weight={self.loss_l2norm_weight}) is enabled with FSQ. "
                      f"This may not be meaningful for quantized latents. Consider setting loss_l2norm_weight=0.")
            if self.loss_kl_weight > 0:
                print(f"ℹ️  Info: KL loss weight is {self.loss_kl_weight} but will be ignored in FSQ mode.")
            print(f"✓ FSQ mode enabled. Codebook size: {self.raw_model.codebook_size}")

    def compute_accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == labels).float()
        return correct.mean()

    def compute_accuracy_is_closed(self, logits, labels):
        # logits shape: [batch, 2], labels shape: [batch, 2]
        predictions = torch.sigmoid(logits) > 0.5
        correct = (predictions == labels).float()
        return correct.mean()


    def compute_kl_annealing_beta(self, step):
        """
        Compute KL annealing weight using linear schedule.
        Returns value between 0 and 1 that will be multiplied by loss_kl_weight.
        """
        if self.kl_annealing_steps <= 0:
            return 1.0
        return min(1.0, step / self.kl_annealing_steps)

    def log_loss(self, total_loss, accuracy, accuracy_is_closed, loss_recon, loss_cls, loss_kl, loss_l2norm, loss_is_closed, lr, total_norm, step, time_per_step, kl_beta=1.0, active_dims=None, codebook_usage=None, unique_codes=None):
        if not self.use_wandb_tracking or not self.is_main:
            return
        
        log_dict = {
            'loss': total_loss,
            'accuracy': accuracy,
            'accuracy_is_closed': accuracy_is_closed,
            'loss_recon': loss_recon,
            'loss_cls': loss_cls,
            'loss_kl': loss_kl,
            'loss_l2norm': loss_l2norm,
            'loss_is_closed': loss_is_closed,
            'lr': lr,
            'grad_norm': total_norm,
            'step': step,
            'kl_beta': kl_beta,
            'time_per_step': time_per_step,
        }
        
        # Add VAE-specific metrics
        if active_dims is not None:
            log_dict['active_latent_dims'] = active_dims
        
        # Add FSQ-specific metrics
        if codebook_usage is not None:
            log_dict['fsq/codebook_usage'] = codebook_usage
        if unique_codes is not None:
            log_dict['fsq/unique_codes'] = unique_codes
        
        self.log(**log_dict)
        
        # Print statement
        print_msg = (f'{get_current_time()} loss: {total_loss:.3f} acc: {accuracy:.3f} recon: {loss_recon:.3f} '
                    f'acc_closed: {accuracy_is_closed:.3f} lr: {lr:.6f} norm: {total_norm:.3f} '
                    f'kl_beta: {kl_beta:.3f} time: {time_per_step:.3f}')
        
        if self.use_fsq and codebook_usage is not None:
            print_msg += f' CB_usage: {codebook_usage:.3f} codes: {unique_codes}'
        
        self.print(print_msg)

    def train(self):
        step = self.step.item()

        tt = time.time()
        while step < self.num_train_steps:
            total_loss = 0.
            total_accuracy = 0.
            t = time.time()
            for i in range(self.grad_accum_every):
                is_last = i == (self.grad_accum_every - 1)
                maybe_no_sync = partial(self.accelerator.no_sync, self.model) if not is_last else nullcontext
                # print('others time: ', f'{time.time() - tt:.2f}s')
                tt = time.time()

                forward_kwargs = self.next_data_to_forward_kwargs(self.train_dl_iter)
                # print('data loading time: ', f'{time.time() - tt:.2f}s')
                tt = time.time()
                with self.accelerator.autocast(), maybe_no_sync():
                    if self.pred_is_closed:
                        params_padded, surface_type, masks, shifts_padded, rotations_padded, scales_padded, is_u_closed, is_v_closed = forward_kwargs
                    else:
                        params_padded, surface_type, masks, shifts_padded, rotations_padded, scales_padded = forward_kwargs
                    params_padded = params_padded[masks.bool()] 
                    surface_type = surface_type[masks.bool()]
                    shifts_padded = shifts_padded[masks.bool()]
                    rotations_padded = rotations_padded[masks.bool()]
                    scales_padded = scales_padded[masks.bool()]

                    if self.pred_is_closed:
                        is_u_closed = is_u_closed[masks.bool()]
                        is_v_closed = is_v_closed[masks.bool()]
                        is_closed_gt = torch.stack([is_u_closed, is_v_closed], dim=-1)

                    if surface_type.shape[0] == 0:
                        continue
                    
                    # print('data prepare time: ', f'{time.time() - tt:.2f}s')
                    tt = time.time()
                    if self.use_fsq:
                        z_quantized, indices = self.raw_model.encode(params_padded, surface_type)
                        z = z_quantized
                    else:
                        mu, logvar = self.raw_model.encode(params_padded, surface_type)
                        if self.use_logvar:
                            z = self.raw_model.reparameterize(mu, logvar)
                        else:
                            z = mu
                    if self.pred_is_closed:
                        class_logits, surface_type_pred, is_closed_logits, is_closed = self.raw_model.classify(z)
                    else:
                        class_logits, surface_type_pred = self.raw_model.classify(z)


                    params_raw_recon, mask = self.raw_model.decode(z, surface_type)
                    
                    # Compute per-sample reconstruction loss: (B, D) → (B,)
                    recon_loss_per_sample = (self.loss_recon(params_raw_recon, params_padded) * mask.float()).mean(dim=-1)
                    
                    # Apply per-type weighting if configured
                    if self.type_weight is not None:
                        sample_weights = self.type_weight[surface_type]  # (B,)
                        loss_recon = (recon_loss_per_sample * sample_weights).mean()
                    else:
                        loss_recon = recon_loss_per_sample.mean()
                    
                    loss_cls = self.loss_cls(class_logits, surface_type).mean()
                    loss_l2norm = torch.norm(z, p=2, dim=-1).mean() 
                    if self.pred_is_closed:
                        # is_closed_logits shape: [batch, 2], is_closed_gt shape: [batch, 2]
                        # pos_weight is applied element-wise per position (u=0, v=1)
                        loss_is_closed = self.loss_is_closed(is_closed_logits, is_closed_gt.float())
                    else:
                        loss_is_closed = torch.tensor(0.0, device=z.device)

                    if not self.use_fsq:
                        # Compute KL loss with free bits strategy
                        # Per-dimension KL: [batch, latent_dim]

                        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                        
                        # Apply free bits: only penalize KL above the threshold
                        if self.kl_free_bits > 0:
                            kl_per_dim = torch.clamp(kl_per_dim - self.kl_free_bits, min=0.0)
                        
                        # Mean across batch and dimensions
                        loss_kl = torch.mean(kl_per_dim)
                    else:
                        loss_kl = torch.tensor(0.0, device=z.device)
                        kl_per_dim = None  # FSQ doesn't have KL loss
                    

                    # Compute KL annealing beta
                    kl_beta = self.compute_kl_annealing_beta(step)
                    
                    # Final loss with annealing
                    loss = loss_recon * self.loss_recon_weight + loss_cls * self.loss_cls_weight + loss_kl * self.loss_kl_weight * kl_beta + loss_l2norm * self.loss_l2norm_weight  + self.loss_is_closed_weight * loss_is_closed
                    
                    accuracy = self.compute_accuracy(class_logits, surface_type)
                    if self.pred_is_closed:
                        accuracy_is_closed = self.compute_accuracy_is_closed(is_closed_logits, is_closed_gt)
                    else:
                        accuracy_is_closed = torch.tensor(0.0, device=z.device)

                    loss = loss / self.grad_accum_every
                    total_loss += loss.item()
                    total_accuracy += accuracy.item() / self.grad_accum_every

                    # print('model and loss time: ', f'{time.time() - tt:.2f}s')
                    tt = time.time()
                
                self.accelerator.backward(loss)
            time_per_step = (time.time() - t) / self.grad_accum_every
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.is_main and divisible_by(step, self.log_every_step):
                cur_lr = get_lr(self.optimizer.optimizer)
                total_norm = self.optimizer.total_norm
                
                # Compute active dimensions if free bits is enabled (only for VAE, not FSQ)
                active_dims = None
                if not self.use_fsq and self.kl_free_bits > 0 and kl_per_dim is not None:
                    # Count dimensions with KL above threshold
                    kl_per_dim_mean = kl_per_dim.mean(dim=0)  # Average over batch
                    active_dims = (kl_per_dim_mean > 0).float().sum().item()
                
                # Compute FSQ-specific metrics
                codebook_usage = None
                unique_codes = None
                if self.use_fsq and 'indices' in locals():
                    # Handle both single and multiple codebooks
                    # indices can be (B,) or (B, num_codebooks)
                    if indices.ndim == 1:
                        unique_codes_tensor = torch.unique(indices)
                        unique_codes = unique_codes_tensor.numel()
                        codebook_size = self.raw_model.codebook_size
                        codebook_usage = unique_codes / codebook_size
                    else:
                        # Multiple codebooks: average usage across all codebooks
                        usage_per_cb = []
                        for i in range(indices.shape[1]):
                            unique_codes_tensor = torch.unique(indices[:, i])
                            usage_per_cb.append(unique_codes_tensor.numel() / self.raw_model.codebook_size)
                        codebook_usage = sum(usage_per_cb) / len(usage_per_cb)
                        unique_codes = int(sum([torch.unique(indices[:, i]).numel() for i in range(indices.shape[1])]) / indices.shape[1])
                                                        
                self.log_loss(total_loss, total_accuracy, accuracy_is_closed, loss_recon.item(), 
                            loss_cls.item(), loss_kl.item(), loss_l2norm.item(), 
                            loss_is_closed.item(), cur_lr, total_norm, step, time_per_step, 
                            kl_beta, active_dims, codebook_usage, unique_codes)
                
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
                total_val_accuracy_is_closed = 0.
                device = next(self.model.parameters()).device
                
                # For FSQ: collect all indices to compute codebook usage
                all_val_indices = [] if self.use_fsq else None
                
                self.ema.eval()
                num_val_batches = self.val_num_batches * self.grad_accum_every

                for _ in range(num_val_batches):
                    with self.accelerator.autocast(), torch.no_grad():
                        t = time.time()
                        forward_kwargs = self.next_data_to_forward_kwargs(self.val_dl_iter)

                        if self.pred_is_closed:
                            params_padded, surface_type, masks, shifts_padded, rotations_padded, scales_padded, is_u_closed, is_v_closed = forward_kwargs
                        else:
                            params_padded, surface_type, masks, shifts_padded, rotations_padded, scales_padded = forward_kwargs
                        # print('Dataloader time: ', f'{time.time() - t:.2f}s')

                        params_padded = params_padded[masks.bool()] 
                        surface_type = surface_type[masks.bool()]
                        shifts_padded = shifts_padded[masks.bool()]
                        rotations_padded = rotations_padded[masks.bool()]
                        scales_padded = scales_padded[masks.bool()]

                        if self.pred_is_closed:
                            is_u_closed = is_u_closed[masks.bool()]
                            is_v_closed = is_v_closed[masks.bool()]
                            is_closed_gt = torch.stack([is_u_closed, is_v_closed], dim=-1)
                            is_closed_gt = is_closed_gt.to(device)

                        params_padded = params_padded.to(device)
                        surface_type = surface_type.to(device)

                        if surface_type.shape[0] == 0:
                            continue
                        # print(params_padded.abs().max())
                        abnormal_value = params_padded.abs().max().item()
                        t = time.time()
                        
                        # Encode (VAE or FSQ)
                        if self.use_fsq:
                            z_quantized, indices = self.raw_model.encode(params_padded, surface_type)
                            z = z_quantized
                        else:
                            mu, logvar = self.raw_model.encode(params_padded, surface_type)
                            if self.use_logvar:
                                z = self.raw_model.reparameterize(mu, logvar)
                            else:
                                z = mu
                        
                        # Classify
                        if self.pred_is_closed:
                            class_logits, surface_type_pred, is_closed_logits, is_closed = self.raw_model.classify(z)
                        else:
                            class_logits, surface_type_pred = self.raw_model.classify(z)
                        
                        # Decode
                        params_raw_recon, mask = self.raw_model.decode(z, surface_type)
                        
                        # Compute per-sample reconstruction loss: (B, D) → (B,)
                        recon_loss_per_sample = (self.loss_recon(params_raw_recon, params_padded) * mask.float()).mean(dim=-1)
                        
                        # Apply per-type weighting if configured
                        if self.type_weight is not None:
                            sample_weights = self.type_weight[surface_type]  # (B,)
                            loss_recon = (recon_loss_per_sample * sample_weights).mean()
                        else:
                            loss_recon = recon_loss_per_sample.mean()
                        
                        loss_cls = self.loss_cls(class_logits, surface_type).mean()
                        loss_l2norm = torch.norm(z, p=2, dim=-1).mean()
                        
                        if self.pred_is_closed:
                            loss_is_closed = self.loss_is_closed(is_closed_logits, is_closed_gt.float()).mean()
                        else:
                            loss_is_closed = torch.tensor(0.0, device=z.device)
                        
                        # Compute KL loss (only for VAE, not FSQ)
                        if not self.use_fsq:
                            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                            if self.kl_free_bits > 0:
                                kl_per_dim = torch.clamp(kl_per_dim - self.kl_free_bits, min=0.0)
                            loss_kl = torch.mean(kl_per_dim)
                        else:
                            loss_kl = torch.tensor(0.0, device=z.device)
                        
                        # Use current annealing beta for validation loss
                        val_kl_beta = self.compute_kl_annealing_beta(step)
                        loss = (loss_recon * self.loss_recon_weight + 
                               loss_cls * self.loss_cls_weight + 
                               loss_kl * self.loss_kl_weight * val_kl_beta + 
                               loss_l2norm * self.loss_l2norm_weight + 
                               self.loss_is_closed_weight * loss_is_closed)

                        accuracy = self.compute_accuracy(class_logits, surface_type)

                        if self.pred_is_closed:
                            accuracy_is_closed = self.compute_accuracy_is_closed(is_closed_logits, is_closed_gt)
                        else:
                            accuracy_is_closed = torch.tensor(0.0, device=z.device)

                        total_val_loss += (loss / num_val_batches)
                        total_val_accuracy += (accuracy / num_val_batches)
                        total_val_accuracy_is_closed += (accuracy_is_closed / num_val_batches)
                        
                        # Collect FSQ indices
                        if self.use_fsq:
                            all_val_indices.append(indices)
                
                # Compute FSQ codebook usage for validation
                val_codebook_usage = None
                val_unique_codes = None
                if self.use_fsq and len(all_val_indices) > 0:
                    all_val_indices = torch.cat(all_val_indices)
                    
                    # Handle both single and multiple codebooks
                    if all_val_indices.ndim == 1:
                        val_unique_codes_tensor = torch.unique(all_val_indices)
                        val_unique_codes = val_unique_codes_tensor.numel()
                        val_codebook_usage = val_unique_codes / self.raw_model.codebook_size
                    else:
                        # Multiple codebooks: average usage across all codebooks
                        usage_per_cb = []
                        for i in range(all_val_indices.shape[1]):
                            unique_codes_tensor = torch.unique(all_val_indices[:, i])
                            usage_per_cb.append(unique_codes_tensor.numel() / self.raw_model.codebook_size)
                        val_codebook_usage = sum(usage_per_cb) / len(usage_per_cb)
                        val_unique_codes = int(sum([torch.unique(all_val_indices[:, i]).numel() for i in range(all_val_indices.shape[1])]) / all_val_indices.shape[1])

                # Print validation results
                print_msg = (f'{get_current_time()} valid loss: {total_val_loss:.3f} '
                            f'acc: {total_val_accuracy:.3f} acc_closed: {total_val_accuracy_is_closed:.3f} '
                            f'recon: {loss_recon.item():.4f} cls: {loss_cls.item():.4f} kl: {loss_kl.item():.4f}')
                
                if self.use_fsq and val_codebook_usage is not None:
                    print_msg += f' CB_usage: {val_codebook_usage:.3f} codes: {val_unique_codes}'
                
                self.print(print_msg)
                
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
                    "val_accuracy_is_closed": total_val_accuracy_is_closed,
                    "val_loss_l2norm": loss_l2norm.item()
                }
                
                # Add FSQ-specific validation metrics
                if val_codebook_usage is not None:
                    val_log_dict['val_fsq/codebook_usage'] = val_codebook_usage
                if val_unique_codes is not None:
                    val_log_dict['val_fsq/unique_codes'] = val_unique_codes
                
                self.log(**val_log_dict)

            self.wait()

            if self.is_main and (divisible_by(step, self.checkpoint_every_step) or step == self.num_train_steps - 1):
                checkpoint_num = step // self.checkpoint_every_step 
                milestone = str(checkpoint_num).zfill(2)
                self.save(milestone)
                self.print(get_current_time() + f' checkpoint saved at {self.checkpoint_folder / f"model-{milestone}.pt"}')

        self.print('VAE V1 training complete') 