from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Linear
from typing import Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.unets.unet_2d_blocks import get_down_block
from x_transformers.x_transformers import AttentionLayers

from einops import rearrange

from src.vae.modules import AutoencoderKLOutput, RandomFourierEmbed2D, UNetMidBlock2D
from src.vae.layers import BSplineSurfaceLayer
from src.utils.torch_tools import interpolate_2d, calculate_surface_area, sample_surface_points


class Encoder2D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                temb_channels=None,
                downsample_padding=1,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )
        
        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):        
        sample = self.conv_in(x)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)
        
        # middle
        sample = self.mid_block(sample)
        
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class SimpleUpBlock2D(nn.Module):
    """Simplified 2D up block without residual connections."""
    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()
        from diffusers.models.unets.unet_2d_blocks import ResnetBlock2D, UpDecoderBlock2D
        
        self.resnets = nn.ModuleList([
            ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=None),
            ResnetBlock2D(in_channels=in_channels, out_channels=in_channels, temb_channels=None),
            ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, temb_channels=None),
        ])
        if up:
            self.up = UpDecoderBlock2D(out_channels, out_channels)
        else:
            self.up = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.up is not None:
            hidden_states = self.up(hidden_states)
        return hidden_states


class Decoder2D(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=4,
        act_fn="silu",
        norm_type="group",
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
        )

        # up
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_first_block = i == 0
            up_block = SimpleUpBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                up=not is_first_block,
            )
            self.up_blocks.append(up_block)

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], in_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        
        self.conv_act = nn.SiLU()

        self.query_embed = nn.Sequential(
            RandomFourierEmbed2D(block_out_channels[0]),
            Linear(block_out_channels[0] + 2, block_out_channels[0] * 2),  # Increased capacity
            nn.SiLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            Linear(block_out_channels[0] * 2, block_out_channels[0]),
            nn.SiLU()
        )

        self.cross_attend = AttentionLayers(
            dim = block_out_channels[0],
            depth=4,  # Increased depth for better modeling
            heads = 8,
            cross_attend=True,
            rotary_pos_emb=True,
            gate_residual=True,
            use_layerscale=True,
            attn_flash=True,
            only_cross=True,
        )

        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z, queries):
        sample = self.conv_in(z)

        # middle
        sample = self.mid_block(sample) # (3 --> 512), TODO
        
        # up
        for up_block in self.up_blocks:  
            sample = up_block(sample)
        
        # cross-attention
        sample = rearrange(sample, 'b d h w -> b (h w) d')

        queries_embeddings = self.query_embed(queries)
        queries_embeddings = rearrange(queries_embeddings, 'b n m d -> b (n m) d')    
        sample = self.cross_attend(queries_embeddings, context=sample)

        # Fix: output should match query dimensions, not assume square latent features
        n, m = queries.shape[1], queries.shape[2]
        sample = rearrange(sample, 'b (n m) d -> b d n m', n=n, m=m)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AutoencoderKL2D(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D",),
        up_block_types: Tuple[str] = ("UpBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 16,
        kl_weight: float = 1e-6,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder2D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder2D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv =  nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv =  nn.Conv2d(latent_channels, latent_channels, 1)

        self.sample_points_num = sample_points_num
        self.kl_weight = kl_weight

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h) 
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, t)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        decoded = self._decode(z, t).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self,
        data: torch.FloatTensor,
        t: Optional[torch.FloatTensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        training_step: Optional[int] = None,  # Add training step for KL annealing
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Args:
            data: Input surface data
            t: Query points for sampling  
            sample_posterior: Whether to sample from the posterior
            return_dict: Whether to return a DecoderOutput
            generator: Random generator for sampling
            return_loss: Whether to return loss values
            training_step: Current training step for KL annealing
        """
        data = rearrange(data, "b h w c -> b c h w")  # for conv2d input

        bs = data.shape[0]

        posterior = self.encode(data).latent_dist
        
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        
        if t is None:
            # Generate grid-like random points (jittered grid)
            # Create grid cell boundaries
            grid_size = 1.0 / self.sample_points_num
            
            # Generate random offsets within each grid cell
            random_offsets = torch.rand(bs, self.sample_points_num, self.sample_points_num, 2, device=data.device)
            
            # Create base grid coordinates for each cell
            i_coords = torch.arange(self.sample_points_num, device=data.device).float()
            j_coords = torch.arange(self.sample_points_num, device=data.device).float()
            
            # Create meshgrid for base coordinates
            i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
            base_grid = torch.stack([i_grid, j_grid], dim=-1)  # (sample_points_num, sample_points_num, 2)
            
            # Add random jitter within each grid cell
            # t = (base_grid.unsqueeze(0) + random_offsets) * grid_size  # (bs, sample_points_num, sample_points_num, 2)
            t = base_grid.unsqueeze(0) * grid_size
            t = t.repeat(bs, 1, 1, 1)
            # Clamp to ensure values stay in [0, 1]
            t = torch.clamp(t, 0.0, 1.0)
        else:
            assert t.shape[1] == self.sample_points_num and t.shape[2] == self.sample_points_num, \
                   "t should have the same number of self.sample_points_num"

        dec = self.decode(z, t).sample

        if not return_dict:
            return (dec,)
        
        if return_loss:
            # Calculate KL loss with Free-Bits technique to prevent posterior collapse
            kl_loss_per_dim = 0.5 * (
                torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar
            )
            
            # Free-Bits: ensure minimum KL loss per dimension
            free_bits = 0.5  # Minimum bits per latent dimension
            kl_loss_per_dim = torch.clamp(kl_loss_per_dim, min=free_bits)
            kl_loss = torch.sum(kl_loss_per_dim, dim=[1, 2, 3]).mean()

            # KL annealing: gradually increase KL weight during training
            if training_step is not None:
                # Warm up KL loss over first 10k steps
                kl_warmup_steps = 10000
                kl_beta = min(1.0, training_step / kl_warmup_steps)
                kl_loss = kl_beta * kl_loss

            gt_samples = interpolate_2d(t, data)

            data = rearrange(data, 'b c h w -> b h w c')
            batch_areas = calculate_surface_area(data)
            batch_areas = torch.clamp(batch_areas, min=2.0, max=torch.pi * 100)

            weights = torch.log(batch_areas + 0.2)  # reduce influence of large surfaces
            batch_loss = F.mse_loss(dec, gt_samples, reduction='none').mean(dim=[1, 2, 3])
            recon_loss = (batch_loss * weights).mean()
            loss = recon_loss + self.kl_weight * kl_loss

            return loss, dict(
                recon_loss = recon_loss,
                kl_loss = kl_loss
            )

        return DecoderOutput(sample=dec)


class AutoencoderKL2DFastEncode(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.encoder = Encoder2D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.quant_conv =  nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h) 
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def forward(
        self,
        data: torch.FloatTensor,
        return_std: bool = False,
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        data = sample_surface_points(data, 16)  # downsample to 16x16 points
        data = rearrange(data, "b h w c -> b c h w")  # for conv2d input

        posterior = self.encode(data).latent_dist
        z = posterior.mode()

        if return_std:
            return z, posterior.std

        return z


class AutoencoderKL2DFastDecode(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        out_channels: int = 3,
        up_block_types: Tuple[str] = ("UpBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.decoder = Decoder2D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

        self.post_quant_conv =  nn.Conv2d(latent_channels, latent_channels, 1)
  
    def _decode(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, t)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self, 
        z: torch.FloatTensor, 
        t: torch.FloatTensor = None,
        return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        if t is None:
            device = z.device
            bs = z.shape[0]
            t = torch.linspace(0, 1, 16, device=device)
            t = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)
            t = t.repeat(bs, 1, 1, 1)

        decoded = self._decode(z, t)

        return decoded  # DecoderOutput instance, use .sample to get tensor 


class AutoencoderKLBS2D(AutoencoderKL2D):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D",),
        up_block_types: Tuple[str] = ("UpBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_points_num: int = 16,
        kl_weight: float = 1e-6,
        # B-spline branch parameters
        bspline_resolution: int = 32,
        bspline_cp_weight: float = 1.0,
        bspline_surface_weight: float = 1.0,
        mlp_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_points_num=sample_points_num,
            kl_weight=kl_weight,
        )

        # B-spline branch components
        self.bspline_cp_weight = bspline_cp_weight
        self.bspline_surface_weight = bspline_surface_weight
        self.bspline_resolution = bspline_resolution
        
        # MLP to transform latent features to B-spline control points
        # We'll initialize the MLP later after we know the actual latent dimensions
        self.control_point_mlp = None
        self.mlp_hidden_dim = mlp_hidden_dim
        self.act_fn = act_fn
        
        # B-spline surface layer
        self.bspline_layer = BSplineSurfaceLayer(resolution=bspline_resolution)
        self.total_latent_dim = latent_channels * (sample_points_num // 2 ** (len(down_block_types) - 1)) ** 2
        self._init_control_point_mlp(self.total_latent_dim)

    
    def _init_control_point_mlp(self, latent_dim):
        """Initialize the control point MLP with the correct input dimension."""
        if self.control_point_mlp is not None:
            return  # Already initialized
            
        # Get activation function
        if self.act_fn == "silu":
            activation = nn.SiLU()
        elif self.act_fn == "relu":
            activation = nn.ReLU()
        elif self.act_fn == "gelu":
            activation = nn.GELU()
        else:
            activation = nn.SiLU()  # default
        
        self.control_point_mlp = nn.Sequential(
            nn.Linear(latent_dim, self.mlp_hidden_dim),
            activation,
            nn.Dropout(0.1),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            activation,
            nn.Dropout(0.1),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim // 2),
            activation,
            nn.Linear(self.mlp_hidden_dim // 2, 16 * 3),  # 16 control points, each with 3 coordinates
        ).to(next(self.parameters()).device)

    def _decode_bspline(
        self,
        z: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Decode using B-spline branch.
        
        Args:
            z: Latent tensor (B, D, H, W)
            t: Query points (B, N, M, 2)
            return_dict: Whether to return DecoderOutput
            
        Returns:
            Decoded surface points at query locations
        """
        bs = z.shape[0]
        
        # Initialize MLP if not done yet
        if self.control_point_mlp is None:
            latent_dim = z.shape[1] * z.shape[2] * z.shape[3]  # D * H * W
            self._init_control_point_mlp(latent_dim)
        
        # Transform latent codes to control points
        z_flat = z.view(bs, -1)  # Flatten to (B, D*H*W)
        control_points_flat = self.control_point_mlp(z_flat)  # (B, 16*3)
        control_points = control_points_flat.view(bs, 16, 3)  # (B, 16, 3)
        
        # Generate surface points using B-spline
        bspline_surface = self.bspline_layer(control_points)  # (B, resolution, resolution, 3)
        
        # Sample from the B-spline surface at the query points
        # surface_indices = t * (self.bspline_resolution - 1)  # (B, N, M, 2)

        # bspline_dec = self._sample_from_surface(bspline_surface, surface_indices)
        bspline_dec = interpolate_2d(t, rearrange(bspline_surface, 'b h w c -> b c h w'))

        return (control_points, bspline_dec,)


    @apply_forward_hook
    def decode_both_branches(
        self,
        z: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Decode using both cross-attention and B-spline branches.
        
        Returns:
            Tuple of (cross_attention_output, bspline_output)
        """
        # Cross-attention branch
        cross_attention_dec = self._decode(z, t).sample
        
        # B-spline branch
        control_points, sampled_surface = self._decode_bspline(z, t).values()
        # Convert B-spline output from (B, H, W, 3) to (B, 3, H, W) to match cross-attention format
        # bspline_dec = rearrange(bspline_dec, 'b h w c -> b c h w')
        
        # if not return_dict:
        #     return (cross_attention_dec, bspline_dec)
            
        # return DecoderOutput(sample=cross_attention_dec), DecoderOutput(sample=bspline_dec)
        return (control_points, sampled_surface, cross_attention_dec)

    def forward(
        self,
        data: torch.FloatTensor,
        control_points: Optional[torch.FloatTensor] = None,
        t: Optional[torch.FloatTensor] = None,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
        return_loss: bool = False,
        training_step: Optional[int] = None,  # Add training step for KL annealing
        return_both_branches: bool = False,  # Whether to return both branches
        **kwargs,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Args:
            data: Input surface data
            t: Query points for sampling  
            sample_posterior: Whether to sample from the posterior
            return_dict: Whether to return a DecoderOutput
            generator: Random generator for sampling
            return_loss: Whether to return loss values
            training_step: Current training step for KL annealing
            return_both_branches: Whether to return outputs from both branches
        """
        data = rearrange(data, "b h w c -> b c h w")  # for conv2d input

        bs = data.shape[0]

        posterior = self.encode(data).latent_dist
        
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        
        if t is None:
            # Generate grid-like random points (jittered grid)
            # Create grid cell boundaries
            grid_size = 1.0 / self.sample_points_num
            
            # Generate random offsets within each grid cell
            random_offsets = torch.rand(bs, self.sample_points_num, self.sample_points_num, 2, device=data.device)
            
            # Create base grid coordinates for each cell
            i_coords = torch.arange(self.sample_points_num, device=data.device).float()
            j_coords = torch.arange(self.sample_points_num, device=data.device).float()
            
            # Create meshgrid for base coordinates
            i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
            base_grid = torch.stack([i_grid, j_grid], dim=-1)  # (sample_points_num, sample_points_num, 2)
            
            # Add random jitter within each grid cell
            # t = (base_grid.unsqueeze(0) + random_offsets) * grid_size  # (bs, sample_points_num, sample_points_num, 2)
            t = base_grid.unsqueeze(0) * grid_size
            t = t.repeat(bs, 1, 1, 1)
            # Clamp to ensure values stay in [0, 1]
            t = torch.clamp(t, 0.0, 1.0)
        else:
            assert t.shape[1] == self.sample_points_num and t.shape[2] == self.sample_points_num, \
                   "t should have the same number of self.sample_points_num"

        # Cross-attention branch (existing)
        dec = self.decode(z, t).sample

        # B-spline branch (new)
        dec_control_points, dec_sampled_surface = self._decode_bspline(z, t)
        # Convert B-spline output from (B, H, W, 3) to (B, 3, H, W) to match cross-attention format
        # bspline_dec = rearrange(bspline_dec, 'b h w c -> b c h w')

        if not return_dict:
            if return_both_branches:
                return (dec_control_points, dec_sampled_surface, dec)
            return (dec,)
        

        if return_loss:
            assert control_points is not None, "control_points must be provided for loss calculation"
            # Calculate KL loss with Free-Bits technique to prevent posterior collapse
            kl_loss_per_dim = 0.5 * (
                torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar
            )
            
            # Free-Bits: ensure minimum KL loss per dimension
            # free_bits = 0.5  # Minimum bits per latent dimension
            # kl_loss_per_dim = torch.clamp(kl_loss_per_dim, min=free_bits)
            kl_loss = torch.sum(kl_loss_per_dim, dim=[1, 2, 3]).mean()

            # KL annealing: gradually increase KL weight during training
            if training_step is not None:
                # Warm up KL loss over first 10k steps
                kl_warmup_steps = 10000
                kl_beta = min(1.0, training_step / kl_warmup_steps)
                kl_loss = kl_beta * kl_loss

            gt_samples = interpolate_2d(t, data)

            data = rearrange(data, 'b c h w -> b h w c')
            batch_areas = calculate_surface_area(data)
            batch_areas = torch.clamp(batch_areas, min=2.0, max=torch.pi * 100)

            weights = torch.log(batch_areas + 0.2)  # reduce influence of large surfaces
            
            # Cross-attention branch loss
            batch_loss = F.mse_loss(dec, gt_samples, reduction='none').mean(dim=[1, 2, 3])
            recon_loss = (batch_loss * weights).mean()
            
            # B-spline branch loss
            bspline_cp_loss = F.mse_loss(dec_control_points, control_points, reduction='none').mean(dim=[1, 2])
            bspline_surface_loss = F.mse_loss(dec_sampled_surface, gt_samples, reduction='none').mean(dim=[1, 2, 3])
            bspline_cp_loss = (bspline_cp_loss * weights).mean()
            bspline_surface_loss = (bspline_surface_loss * weights).mean()
            # bspline_batch_loss = F.mse_loss(bspline_dec, gt_samples, reduction='none').mean(dim=[1, 2, 3])
            # bspline_recon_loss = (bspline_batch_loss * weights).mean()
            
            # Combined loss
            total_loss = recon_loss + self.bspline_cp_weight * bspline_cp_loss + self.bspline_surface_weight * bspline_surface_loss + self.kl_weight * kl_loss

            return total_loss, dict(
                recon_loss=recon_loss,
                bspline_cp_loss=bspline_cp_loss,
                bspline_surface_loss=bspline_surface_loss,
                kl_loss=kl_loss
            )

        if return_both_branches:
            return dec_control_points, dec_sampled_surface, dec
        
        return dec