import tqdm
from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
import einops
from diffusers import ModelMixin, ConfigMixin
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.configuration_utils import register_to_config
from diffusers import DDPMScheduler

from PIL import Image
ic.disable()


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def enforce_zero_terminal_snr(betas):
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    # Shift so last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T
    # Scale so first timestep is back to old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas
    return betas



def get_new_scheduler(num_train_timesteps=1000):
    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        # beta_start=0.00085,
        # beta_end=0.012,
        # beta_schedule="scaled_linear",
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        prediction_type="v_prediction",
        timestep_spacing="linspace",
    )
    betas = (scheduler.betas)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        trained_betas=betas,
        clip_sample=False,
        prediction_type="v_prediction",
        timestep_spacing="linspace",
    )

    return scheduler

def get_noiseless_scheduler():
    scheduler = DDPMScheduler(
        num_train_timesteps=1, 
        beta_start=0,
        beta_end=0,
        prediction_type="sample"
    )
    return scheduler


class Pipeline:
    def __init__(self, denoiser, scheduler: DDPMScheduler, dtype):
        super().__init__()
        # self.position_encoder = position_encoder
        # self.image_encoder = image_encoder
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.dtype = dtype

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(round(num_inference_steps * strength)), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(self, condition=None, use_diffusion=True, cfg=7.5, num_inference_steps=50, generator: torch.Generator = None, num_latents=512, output_dim=128, num_samples=1, sample=None, show_progress=True):
        # device = self.Z.device
        device = self.denoiser.device
        B = condition.shape[0] if condition is not None else num_samples

        # set step values
        if use_diffusion:
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
        else:
            self.scheduler.set_timesteps(1)
            timesteps = [0]


        noise = torch.randn(B, num_latents, output_dim, dtype=self.dtype, device=device, generator=generator)


        sample = noise

        # if gs_gt is not None:
        #     gs_gt = gs_gt.to(device).float()
        for t in tqdm.tqdm(timesteps, "[ ZLDMPipeline.__call__ ]", disable=not show_progress):
            # 1. predict noise model_output
            if isinstance(t, torch.Tensor):
                t = t.item()
            if use_diffusion:
                t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            else:
                t_tensor = torch.randint(1, [1], dtype=torch.long, device=device)

            if use_diffusion:
                get_input = sample
            else:
                get_input = torch.zeros_like(sample) # Same as training, input zeros
            
            

            model_output_uncond = self.denoiser(
                get_input,
                t_tensor.expand(B),
                condition,
            )


            model_output = model_output_uncond
            # 2. compute previous image: x_t -> x_t-1
            
            if use_diffusion:
                sample = self.scheduler.step(
                    model_output[:, None, :, :].permute(0, 3, 1, 2),
                    t,
                    sample[:, None, :, :].permute(0, 3, 1, 2),
                    generator=generator,
                ).prev_sample.permute(0, 2, 3, 1)[:, 0, :, :]
            else:
                # Only one inference then return, no need for loop
                return model_output

        # return model_output_uncond
        if use_diffusion:
            return sample
        else:
            return model_output