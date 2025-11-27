from typing import Callable
from pathlib import Path
from functools import partial
import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader, Sampler
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

# Import profiler
try:
    from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False

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
        load_checkpoint_from_file = None,
        from_start = False,
        scheduler: Optional[Type[LRScheduler]] = None,
        scheduler_kwargs: dict = dict(),
        use_wandb_tracking: bool = False,
        weight_decay: float = 0.,
        collate_fn: Optional[Callable] = None,
        val_every_step: int = 1000,
        val_num_batches: int = 10,
        val_batch_size: int = 0,
        train_sampler: Optional[Sampler] = None,
        num_workers_val: int = 8,
        # Profiler parameters
        enable_profiler: bool = False,
        profiler_output_dir: str = './profiler_logs',
        profiler_wait_steps: int = 5,
        profiler_warmup_steps: int = 2,
        profiler_active_steps: int = 5,
        profiler_repeat: int = 1,
        profiler_record_shapes: bool = True,
        profiler_profile_memory: bool = True,
        profiler_with_stack: bool = False,
        profiler_with_flops: bool = False,
        profiler_export_chrome_trace: bool = True,  # Export to Chrome Trace (JSON)
        profiler_export_stacks: bool = False,  # Export stack traces
        profiler_use_wandb: bool = False,  # Upload profiler data to WandB
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
        self.train_sampler = train_sampler
        prefetch_factor = 1 if num_workers > 0 else None
        persistent_workers = True if num_workers > 0 else False
        # dataset and dataloader
        if train_sampler is not None:
            self.train_dl = DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle=False,
                sampler=self.train_sampler,
                pin_memory = True,  # Disable pin_memory to reduce RAM usage
                num_workers=min(num_workers, 16),  # Limit workers to prevent memory explosion
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor,  # Reduce prefetch to save memory
                persistent_workers=persistent_workers,  # Disable persistent workers to allow cleanup
            )
        else:
            self.train_dl = DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle=True,
                # sampler=self.train_sampler,
                pin_memory = True,  # Disable pin_memory to reduce RAM usage
                num_workers=min(num_workers, 16),  # Limit workers to prevent memory explosion
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor,  # Reduce prefetch to save memory
                persistent_workers=persistent_workers,  # Disable persistent workers to allow cleanup
            )

        self.train_dl = self.accelerator.prepare(self.train_dl)
        self.train_dl_iter = cycle(self.train_dl)


        self.should_validate = exists(val_dataset)

        if self.should_validate and self.is_main:
            self.val_every_step = val_every_step
            # self.val_every_step = 20
            if val_batch_size == 0:
                val_batch_size = batch_size
            self.val_num_batches = val_num_batches
            self.val_dl = DataLoader(
                val_dataset, 
                batch_size = val_batch_size, 
                shuffle = True,
                pin_memory = False,  # Disable pin_memory to reduce RAM usage
                drop_last=True,
                num_workers=min(num_workers_val, 8),  # Even fewer workers for validation
                collate_fn=collate_fn,
                prefetch_factor=prefetch_factor,  # Reduce prefetch to save memory
                persistent_workers=persistent_workers,  # Disable persistent workers to allow cleanup
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

        if load_checkpoint_from_file is not None:
            print("loading checkpoint from the file: ", load_checkpoint_from_file)
            self.load_model_only(load_checkpoint_from_file, strict=False)

        if resume_training:
            print("loading checkpoint from the file: ", checkpoint_file_name)
            self.load(checkpoint_file_name, from_start=from_start)
        
        # Initialize profiler
        self.enable_profiler = enable_profiler and PROFILER_AVAILABLE
        self.profiler = None
        self.profiler_output_dir = Path(profiler_output_dir)
        self.profiler_export_chrome_trace = profiler_export_chrome_trace
        self.profiler_export_stacks = profiler_export_stacks
        self.profiler_use_wandb = profiler_use_wandb and use_wandb_tracking
        
        if self.enable_profiler:
            if not PROFILER_AVAILABLE:
                self.print("Warning: torch.profiler is not available. Profiling disabled.")
            elif not self.is_main:
                self.print("Profiler enabled but not on main process. Skipping profiler initialization.")
            else:
                self.profiler_output_dir.mkdir(exist_ok=True, parents=True)
                
                # Configure profiler activities
                activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(ProfilerActivity.CUDA)
                
                # Create profiler schedule
                profiler_schedule = schedule(
                    wait=profiler_wait_steps,
                    warmup=profiler_warmup_steps,
                    active=profiler_active_steps,
                    repeat=profiler_repeat
                )
                
                # Create custom trace handler
                def custom_trace_handler(prof):
                    # 1. TensorBoard format (always export)
                    tb_handler = tensorboard_trace_handler(str(self.profiler_output_dir))
                    tb_handler(prof)
                    
                    # 2. Chrome Trace format (JSON - can view in chrome://tracing)
                    if self.profiler_export_chrome_trace:
                        chrome_trace_file = self.profiler_output_dir / f"chrome_trace_step_{prof.step_num}.json"
                        prof.export_chrome_trace(str(chrome_trace_file))
                        self.print(f"Chrome trace saved to: {chrome_trace_file}")
                        self.print(f"View in Chrome: Open chrome://tracing and load the JSON file")
                    
                    # 3. Export stack traces
                    if self.profiler_export_stacks and profiler_with_stack:
                        stacks_file = self.profiler_output_dir / f"stacks_step_{prof.step_num}.txt"
                        prof.export_stacks(str(stacks_file))
                        self.print(f"Stack traces saved to: {stacks_file}")
                    
                    # 4. Generate HTML report
                    self._generate_html_report(prof, prof.step_num)
                    
                    # 5. Upload to WandB
                    if self.profiler_use_wandb and self.use_wandb_tracking:
                        self._upload_profiler_to_wandb(prof, prof.step_num)
                
                # Initialize profiler
                self.profiler = profile(
                    activities=activities,
                    schedule=profiler_schedule,
                    on_trace_ready=custom_trace_handler,
                    record_shapes=profiler_record_shapes,
                    profile_memory=profiler_profile_memory,
                    with_stack=profiler_with_stack,
                    with_flops=profiler_with_flops,
                )
                
                self.print(f"Profiler initialized. Output directory: {self.profiler_output_dir}")
                self.print(f"Profiler schedule: wait={profiler_wait_steps}, warmup={profiler_warmup_steps}, "
                          f"active={profiler_active_steps}, repeat={profiler_repeat}")
                self.print(f"Export formats: TensorBoard=True, ChromeTrace={profiler_export_chrome_trace}, "
                          f"Stacks={profiler_export_stacks}, WandB={self.profiler_use_wandb}")
                if profiler_export_chrome_trace:
                    self.print("Chrome Trace: View JSON files in chrome://tracing (no dependencies needed!)")
                self.print("TensorBoard: Use 'tensorboard --logdir={} --port=6006'".format(
                    self.profiler_output_dir))

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
    
    def start_profiler(self):
        """Start the profiler if enabled"""
        if self.profiler is not None:
            self.profiler.__enter__()
            self.print("Profiler started")
    
    def stop_profiler(self):
        """Stop the profiler if enabled"""
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.print("Profiler stopped")
    
    def profiler_step(self):
        """Notify profiler that a step has completed"""
        if self.profiler is not None:
            self.profiler.step()
    
    def _generate_html_report(self, prof, step_num):
        """Generate an HTML report from profiler data"""
        try:
            import html
            
            html_file = self.profiler_output_dir / f"profiler_report_step_{step_num}.html"
            
            # Get profiler statistics
            key_averages = prof.key_averages()
            
            # Sort by CUDA time (or CPU time if no CUDA)
            if torch.cuda.is_available():
                sorted_stats = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)
            else:
                sorted_stats = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)
            
            # Generate HTML content
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PyTorch Profiler Report - Step {step_num}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .summary-label {{
            font-weight: bold;
            color: #2c3e50;
        }}
        .summary-value {{
            color: #3498db;
            font-size: 1.2em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 14px;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            position: sticky;
            top: 0;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            text-align: right;
            font-family: 'Courier New', monospace;
        }}
        .op-name {{
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .note {{
            background-color: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
        }}
        .chrome-trace-link {{
            display: inline-block;
            background-color: #27ae60;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .chrome-trace-link:hover {{
            background-color: #229954;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔥 PyTorch Profiler Report</h1>
        <p><strong>Step:</strong> {step_num} | <strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="note">
            <strong>💡 Tip:</strong> For interactive visualization, load the corresponding JSON file in <code>chrome://tracing</code>
        </div>
        
        <div class="summary">
            <h2>📊 Summary</h2>
"""
            
            # Calculate summary statistics
            total_cpu_time = sum(item.cpu_time_total for item in key_averages) / 1000  # Convert to ms
            if torch.cuda.is_available():
                total_cuda_time = sum(item.cuda_time_total for item in key_averages) / 1000
                html_content += f"""
            <div class="summary-item">
                <span class="summary-label">Total CUDA Time:</span>
                <span class="summary-value">{total_cuda_time:.2f} ms</span>
            </div>
"""
            
            html_content += f"""
            <div class="summary-item">
                <span class="summary-label">Total CPU Time:</span>
                <span class="summary-value">{total_cpu_time:.2f} ms</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Number of Operations:</span>
                <span class="summary-value">{len(key_averages)}</span>
            </div>
        </div>
        
        <h2>⚡ Top Operations by Time</h2>
        <p>Showing top operations sorted by {'CUDA' if torch.cuda.is_available() else 'CPU'} time</p>
        
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Operation</th>
                    <th>Calls</th>
                    <th>CPU Time (ms)</th>
                    <th>CPU Time %</th>
"""
            
            if torch.cuda.is_available():
                html_content += """
                    <th>CUDA Time (ms)</th>
                    <th>CUDA Time %</th>
"""
            
            html_content += """
                </tr>
            </thead>
            <tbody>
"""
            
            # Add top operations (limit to top 100)
            for rank, item in enumerate(sorted_stats[:100], 1):
                cpu_time = item.cpu_time_total / 1000  # Convert to ms
                cpu_time_pct = (item.cpu_time_total / (total_cpu_time * 1000)) * 100 if total_cpu_time > 0 else 0
                
                op_name = html.escape(item.key)
                
                html_content += f"""
                <tr>
                    <td>{rank}</td>
                    <td class="op-name" title="{op_name}">{op_name}</td>
                    <td class="metric">{item.count}</td>
                    <td class="metric">{cpu_time:.3f}</td>
                    <td class="metric">{cpu_time_pct:.2f}%</td>
"""
                
                if torch.cuda.is_available():
                    cuda_time = item.cuda_time_total / 1000
                    cuda_time_pct = (item.cuda_time_total / (total_cuda_time * 1000)) * 100 if total_cuda_time > 0 else 0
                    html_content += f"""
                    <td class="metric">{cuda_time:.3f}</td>
                    <td class="metric">{cuda_time_pct:.2f}%</td>
"""
                
                html_content += """
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
        
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px;">
            <p>Generated by CLR-Wire Training Framework | PyTorch Profiler</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Write HTML file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.print(f"HTML report saved to: {html_file}")
            self.print(f"Open in browser: file://{html_file.absolute()}")
            
        except Exception as e:
            self.print(f"Warning: Failed to generate HTML report: {e}")
    
    def _upload_profiler_to_wandb(self, prof, step_num):
        """Upload profiler data to WandB"""
        try:
            import wandb
            
            # Check if wandb is initialized
            if not wandb.run:
                self.print("Warning: WandB is not initialized, skipping profiler upload")
                return
            
            # Log Chrome Trace file to WandB
            if self.profiler_export_chrome_trace:
                chrome_trace_file = self.profiler_output_dir / f"chrome_trace_step_{step_num}.json"
                if chrome_trace_file.exists():
                    # Upload as artifact
                    artifact = wandb.Artifact(
                        name=f"profiler_trace_step_{step_num}",
                        type="profiler",
                        description=f"PyTorch Profiler trace for step {step_num}"
                    )
                    artifact.add_file(str(chrome_trace_file))
                    wandb.log_artifact(artifact)
                    self.print(f"Profiler trace uploaded to WandB as artifact")
            
            # Log summary statistics to WandB
            key_averages = prof.key_averages()
            
            # Calculate metrics
            total_cpu_time = sum(item.cpu_time_total for item in key_averages)
            
            wandb_data = {
                "profiler/total_cpu_time_ms": total_cpu_time / 1000,
                "profiler/num_operations": len(key_averages),
                "profiler/step": step_num,
            }
            
            if torch.cuda.is_available():
                total_cuda_time = sum(item.cuda_time_total for item in key_averages)
                wandb_data["profiler/total_cuda_time_ms"] = total_cuda_time / 1000
            
            # Log top operations
            if torch.cuda.is_available():
                top_ops = sorted(key_averages, key=lambda x: x.cuda_time_total, reverse=True)[:10]
            else:
                top_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:10]
            
            for i, op in enumerate(top_ops, 1):
                op_name = op.key.replace('/', '_').replace('.', '_')[:50]  # Shorten name
                if torch.cuda.is_available():
                    wandb_data[f"profiler/top{i}_{op_name}_cuda_ms"] = op.cuda_time_total / 1000
                wandb_data[f"profiler/top{i}_{op_name}_cpu_ms"] = op.cpu_time_total / 1000
            
            wandb.log(wandb_data)
            self.print(f"Profiler statistics logged to WandB")
            
        except ImportError:
            self.print("Warning: wandb is not installed, skipping profiler upload")
        except Exception as e:
            self.print(f"Warning: Failed to upload profiler data to WandB: {e}")

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
            ema_model = self.ema.ema_model.state_dict(),  # Save inner model's state dict without wrapper
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

    def load_model_only(self, file_path: str, strict=True):
        accelerator = self.accelerator
        device = accelerator.device

        pkg = torch.load(str(file_path), map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        if strict == False:
            print('Load checkpoint with strict=False, some keys may not match, please check the output')
            print(model.load_state_dict(pkg['model'], strict=strict))
        else:
            model.load_state_dict(pkg['model'], strict=strict)

        print(f"loaded model from {file_path}")

    
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
        
        # Start profiler if enabled
        self.start_profiler()

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
            
            # Notify profiler of step completion
            self.profiler_step()
            
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
        
        # Stop profiler if enabled
        self.stop_profiler()

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