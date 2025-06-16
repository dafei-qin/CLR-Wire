import tqdm
import math
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
from src.flow.embedding import PointEmbed




class AdjacencyDecoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, depth=8, heads=8, surface_res=32, num_types=6, num_nearby=20, surface_dim=256, surface_enc_block_out_channels=(32, 64, 128)):
        super().__init__()
        self.depth = depth
        self.heads = heads

        # For individual surface encoder, (surface_res, surface_res, 3) --> (1, surface_dim)
        self.surface_encoder = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=surface_enc_block_out_channels[-1],
            conditioning_channels=3,
            block_out_channels=surface_enc_block_out_channels
        )
        # Calculate output dimension after conv layers
        conv_out_size = surface_res // (2**(len(surface_enc_block_out_channels) - 1))
        self.surface_enc_out = nn.Linear(surface_enc_block_out_channels[-1] * conv_out_size * conv_out_size, surface_dim)

        # For global surface encoder, (num_nearby + 1, surface_dim) --> (num_nearby + 1, surface_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(surface_dim, heads, dim_feedforward=surface_dim * 4, dropout=0.1, activation=F.gelu, batch_first=True, norm_first=True, layer_norm_eps=1e-4),
            depth
        )

        self.type_embedding = nn.Embedding(num_types, surface_dim)
        # self.nearby_embedding = nn.Embedding(num_nearby, dim)

    def forward(self, x, padding_mask, type):
        # x: set of surfaces, (B, num_nearby + 1, surface_res, surface_res, 3)
        # padding_mask: mask for padded surfaces, (B, num_nearby + 1)
        # type: surface types, (B, num_nearby + 1)

        batch_size, num_surfaces = x.shape[:2]
        
        # 1. Encode surfaces - reshape to process all surfaces at once
        x_reshaped = rearrange(x, 'b n h w c -> (b n) h w c')
        x_reshaped = rearrange(x_reshaped, 'bn h w c -> bn c h w')  # Convert to channels first
        
        surface_enc = self.surface_encoder(x_reshaped)  # (B*N, surface_dim, H', W')
        surface_enc = rearrange(surface_enc, 'bn c h w -> bn (c h w)')
        surface_enc = self.surface_enc_out(surface_enc)  # (B*N, surface_dim)
        
        # Reshape back to batch dimension
        surface_enc = rearrange(surface_enc, '(b n) d -> b n d', b=batch_size, n=num_surfaces)  # (B, num_nearby + 1, surface_dim)

        # 2. Add type embedding
        surface_enc = surface_enc + self.type_embedding(type) # (B, num_nearby + 1, surface_dim)
        # 3. Encode global surface
        global_surface_enc = self.encoder(src=surface_enc, src_key_padding_mask=padding_mask) # (B, num_nearby + 1, surface_dim)

        # 4. Calculate adjacency logits
        target_surface_enc = global_surface_enc[torch.arange(surface_enc.shape[0]), 0] # (B, surface_dim)


        # 5. Calculate adjacency logits
        q_vec = target_surface_enc.unsqueeze(1) # (B, 1, surface_dim)
        kv_vec = global_surface_enc # (B, num_nearby + 1, surface_dim)
        adjacency_logits = torch.matmul(q_vec, kv_vec.transpose(-2, -1)) / math.sqrt(q_vec.shape[-1]) # (B, 1, num_nearby + 1)
        adjacency_logits = adjacency_logits.squeeze(1) # (B, num_nearby + 1)
        adjacency_logits = adjacency_logits[:, 1:] # (B, num_nearby)

        # 6. Calculate adjacency loss


        return adjacency_logits
