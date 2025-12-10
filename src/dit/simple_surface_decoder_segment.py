# In this version we use AdaIN to inject the time embedding into the sample

# For the segment version, we use segmented point clouds to server as the conditions each segment has one-to-one mapping to the parameter surface.
# Thus the number of surfaces and surface positions are known as prior.
# We use the same Linear layer to project each sample point to the latent space.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from icecream import ic
from diffusers.models.embeddings import (
    TimestepEmbedding,
    Timesteps,
)
ic.disable()


class AdaLayerNorm(nn.Module):
    """
    Standard AdaLN (FiLM style LayerNorm modulation)
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-4)
        self.modulation = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x, cond):
        """
        x: [B, N, C]
        cond: [B, C]
        """
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None]) + shift[:, None]
        return x

class AdaLNTransformerDecoderLayer(nn.Module):
    """
    Drop-in replacement of TransformerDecoderLayer with AdaLN time modulation
    """
    def __init__(self, latent_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(latent_dim * mlp_ratio, latent_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = AdaLayerNorm(latent_dim, latent_dim)
        self.norm2 = AdaLayerNorm(latent_dim, latent_dim)
        self.norm3 = AdaLayerNorm(latent_dim, latent_dim)

    def forward(self, x, mem, t_emb, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # Self-attention
        h = self.norm1(x, t_emb)
        sa, _ = self.self_attn(h, h, h, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        x = x + sa

        # Cross-attention
        h = self.norm2(x, t_emb)
        ca, _ = self.cross_attn(h, mem, mem, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        x = x + ca

        # Feedforward
        h = self.norm3(x, t_emb)
        x = x + self.mlp(h)

        return x

class SimpleSurfaceDecoder(nn.Module):
    def __init__(
        self,
        input_dim=128,
        cond_dim=768,
        mem_dim=4096,
        output_dim=128,
        latent_dim=256,
        num_layers=4,
        num_heads=8,
        use_diag_memory_mask=False,
        max_num_surfaces=32,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cond_proj = nn.Linear(cond_dim, mem_dim)
        self.output_proj = nn.Linear(latent_dim, output_dim)
        self.use_diag_memory_mask = use_diag_memory_mask
        self.max_num_surfaces=max_num_surfaces
        if use_diag_memory_mask:
            # self.register_buffer('memory_mask', torch.eye(self.max_num_surfaces, dtype=torch.bool))
            self.register_buffer('memory_mask', torch.eye(self.max_num_surfaces, dtype=torch.float32) * 5 - 5) # (diag=0, off-diag = -5)
        else:
            self.memory_mask = None
        self.layers = nn.ModuleList([
            AdaLNTransformerDecoderLayer(
                latent_dim=latent_dim,
                num_heads=num_heads,
                mlp_ratio=4,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])

        timestep_input_dim = latent_dim // 2
        time_embed_dim = latent_dim

        self.time_proj = Timesteps(timestep_input_dim, True, 0)
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn="silu",
            post_act_fn="silu",
        )

    def forward(self, sample, timestep, cond, tgt_key_padding_mask=None, memory_key_padding_mask=None):

        # Project tokens
        x = self.input_proj(sample)
        mem = self.cond_proj(cond)

        # Time embedding
        t_emb = self.time_embedding(self.time_proj(timestep).to(x.device))

        # AdaLN-modulated transformer stack
        for layer in self.layers:
            x = layer(
                x=x,
                mem=mem,
                t_emb=t_emb,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_mask=self.memory_mask,
            )

        x = self.output_proj(x)
        return x