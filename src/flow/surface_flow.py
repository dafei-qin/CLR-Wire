
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
from einops import rearrange
from diffusers import ModelMixin, ConfigMixin
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.configuration_utils import register_to_config
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
from src.flow.embedding import PointEmbd2D

def zero_module(module):    
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# def get_new_scheduler():
#     scheduler = DDPMScheduler(
#         num_train_timesteps=1000,
#         # beta_start=0.00085,
#         # beta_end=0.012,
#         # beta_schedule="scaled_linear",
#         beta_schedule="squaredcos_cap_v2",
#         clip_sample=False,
#         prediction_type="v_prediction",
#         timestep_spacing="linspace",
#     )
#     betas = (scheduler.betas)

#     scheduler = DDPMScheduler(
#         num_train_timesteps=1000,
#         trained_betas=betas,
#         clip_sample=False,
#         prediction_type="v_prediction",
#         timestep_spacing="linspace",
#     )

#     return scheduler


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """


    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )


    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

class Encoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_dim, out_dim, depth=24, dim=512, heads=8, res=32):
        super().__init__()

        self.depth = depth
        self.dim = dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.res = res

        self.proj_in = nn.Linear(in_dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, out_dim, bias=False)
        self.pe = PointEmbd2D(dim=dim)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0, activation=F.gelu, batch_first=True, norm_first=True, layer_norm_eps=1e-4),
            depth
        )

    def forward(self, x):
        t_1d = torch.linspace(0, 1, self.res, device=x.device)
        t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
        x_pe = t_grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        x_pe = rearrange(x_pe, 'b h w c -> b (h w) c')
        x_pe = self.pe(x_pe)
        x = rearrange(x, 'b h w c -> b (h w) c', h=self.res, w=self.res)
        x = self.proj_in(x)
        x = x + x_pe
        x = self.layers(x)
        x = self.proj_out(x)
        return x

# class EncoderRTS(Encoder):
#     def __init__(self, in_dim, out_dim, depth=24, dim=512, heads=8, res=32):
#         super().__init__(in_dim, out_dim, depth, dim, heads, res)
      
#     def forward(self, x):
#         x, last = super().forward(x, return_last=True)

#         if self.cone_pred is not None:
#             cone_pred = self.cone_pred(last)
#             x = torch.cat([x, cone_pred], dim=1)

#         if return_last:
#             return x, last, rts
#         else:
#             return x, rts

class ZLDM(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        depth=24,
        dim=512,
        latent_dim=64,
        heads=8,
        pe=True,
        res=32,
        block_out_channels=(32, 64, 128, 256)
    ):
        super().__init__()

        timestep_input_dim = dim // 2
        time_embed_dim = dim
        self.time_proj = Timesteps(timestep_input_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn="silu",
            post_act_fn="silu",
        )
        
        if pe:
            print("Use Local PE")
            self.point_embd = PointEmbd2D(dim=dim)
        else:
            print("Not using Local PE")
            self.point_embd = None

        self.pc_encoder = ControlNetConditioningEmbedding(conditioning_embedding_channels=dim, conditioning_channels=3, block_out_channels=block_out_channels)

        self.pe = pe
        self.proj_in = nn.Linear(latent_dim, dim, bias=False)
        
        self.layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0, activation=F.gelu, batch_first=True, norm_first=True, layer_norm_eps=1e-4),
            depth
        )

        self.proj_out = nn.Linear(dim, latent_dim, bias=False)

        self.res = res

    def forward(self, sample, t, pc_cond=None, \
                tgt_key_padding_mask=None):
        """
        sample: [N, L, C]
        t: [N]
        condition: [N, L_C, D_C]
        return: [N, L, C]
        """
        N, L, C = sample.shape


        
        # Todo: Define positional embedding for control points and latent pc
        # PE should be pre-defined base on the CP order. 
        # PE of point cloud should be versatile of number of points and positions. 
        t_1d = torch.linspace(0, 1, 4, device=sample.device)
        t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
        sample_pe = t_grid.unsqueeze(0).repeat(N, 1, 1, 1)
        sample_pe = rearrange(sample_pe, 'b h w c -> b (h w) c')
        sample_pe = self.point_embd(sample_pe)
        


        

            # img_pe = self.point_embd(img_uv.reshape(N, -1, 2) / 0.03025 - 1).reshape(N, uv_shape, uv_shape, -1).permute(0, 3, 1, 2) # Here we normalize the image_uv to [-1, 1] the same as sample
            # img_pe = self.point_embd((img_uv.reshape(N, -1, 2) - 0.0303) / 0.018).reshape(N, uv_shape, uv_shape, -1).transpose(1,3) # Here we normalize the image_uv to [-1, 1] the same as sample
        pc_cond = rearrange(pc_cond, 'b h w c -> b c h w')
        pc_cond = self.pc_encoder(pc_cond)
        H, W = pc_cond.shape[-2:]
        assert H == W
        pc_cond = rearrange(pc_cond, 'b c h w -> b (h w) c')

        t_1d = torch.linspace(0, 1, H, device=sample.device)
        t_grid = torch.stack(torch.meshgrid(t_1d, t_1d, indexing='ij'), dim=-1)
        pc_pe = t_grid.unsqueeze(0).repeat(N, 1, 1, 1)
        pc_pe = rearrange(pc_pe, 'b h w c -> b (h w) c')
        pc_pe = self.point_embd(pc_pe)

        
        pc_cond = pc_cond + pc_pe
        condition = pc_cond
        # condition = condition.transpose(1, 2) # (N, N_Points + 1, latent)


        
        x = self.proj_in(sample)
        x = x + sample_pe


        time_encoding = self.time_proj(t).to(sample)
        time_embed = self.time_embedding(time_encoding) 
        x_aug = torch.cat([x, time_embed[:, None]], dim=1) # Project the t to latent
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = torch.cat([tgt_key_padding_mask, torch.zeros(N, 1).bool().to(tgt_key_padding_mask.device)], dim=1)

        y = self.layers(x_aug, condition, tgt_key_padding_mask=tgt_key_padding_mask)

        eps = self.proj_out(y[:, :-1])

        return eps



class ZLDMPipeline:
    def __init__(self, denoiser: ZLDM, scheduler: Union[DDPMScheduler], dtype):
        super().__init__()

        self.denoiser = denoiser
        self.scheduler = scheduler
        self.scheduler_type = type(scheduler).__name__
        self.dtype = dtype

    def get_timesteps(self, num_inference_steps):
        # get the original timestep using init_timestep
        init_timestep = min(int(round(num_inference_steps)), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def __call__(self, pc=None, cfg=7.5, num_inference_steps=50, generator: torch.Generator = None, num_latents=512, num_samples=1, sample=None, sample_mask=None,\
                  tgt_key_padding_mask=None, show_progress=True, device='cpu',
                ):
        # device = self.Z.device
        # device = self.denoiser.device
        if pc is not None:
            pc = pc.to(device)
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = tgt_key_padding_mask.to(device)


        B = pc.shape[0]


        # set step values

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps




        noise = torch.randn(num_samples, 16, num_latents, dtype=self.dtype, device=device, generator=generator)

        if sample is not None:

            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps)
            t_start = timesteps[0]
            
            sample_noisy = self.scheduler.add_noise(sample, noise, torch.tensor(t_start).to(sample.device))
            # Here we handle partial noising 
            if sample_mask is not None:
                sample_noisy = sample_noisy * (1 - sample_mask) + sample * sample_mask
            sample = sample_noisy
        else:
            sample = noise

        # if gs_gt is not None:
        #     gs_gt = gs_gt.to(device).float()
        for t in tqdm.tqdm(timesteps, "[ ZLDMPipeline.__call__ ]", disable=not show_progress):
            # 1. predict noise model_output
            if isinstance(t, torch.Tensor):
                t = t.item()

            t_tensor = torch.tensor([t], dtype=torch.long, device=device)


            get_input = sample

            
         

            

            model_output_uncond = self.denoiser(
                get_input,
                t_tensor.expand(num_samples),
                pc,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )


            model_output = model_output_uncond

            # 2. compute previous image: x_t -> x_t-1
            

            sample = self.scheduler.step(
                model_output[:, None, :, :].permute(0, 3, 1, 2),
                t,
                sample[:, None, :, :].permute(0, 3, 1, 2),
                generator=generator,
            ).prev_sample.permute(0, 2, 3, 1)[:, 0, :, :]


        # return model_output_uncond
        return sample
    

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


def get_new_scheduler(type='v_prediction'):
    if type == 'v_prediction':
        print("Using v_prediction scheduler")
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
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
            num_train_timesteps=1000,
            trained_betas=betas,
            clip_sample=False,
            prediction_type="v_prediction",
            timestep_spacing="linspace",
        )
    elif type == 'sample':
        print("Using sample scheduler")
        scheduler = DDPMScheduler(
            num_train_timesteps=1000, 
            prediction_type="sample"
        )
    else:
        raise ValueError(f"Invalid scheduler type: {type}")

    return scheduler

def get_noiseless_scheduler():
    scheduler = DDPMScheduler(
        num_train_timesteps=1, 
        beta_start=0,
        beta_end=0,
        prediction_type="sample"
    )
    return scheduler

