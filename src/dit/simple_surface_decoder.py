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

class SimpleSurfaceDecoder(nn.Module):
    def __init__(self, input_dim=128, cond_dim=768, output_dim=128, latent_dim=256, num_layers=4, num_heads=8):
        super(SimpleSurfaceDecoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cond_proj = nn.Linear(cond_dim, latent_dim)
        layer = nn.TransformerDecoderLayer(latent_dim, num_heads, dim_feedforward=latent_dim * 4, dropout=0.1, activation=F.gelu, batch_first=True, norm_first=True, layer_norm_eps=1e-4)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.output_proj = nn.Linear(latent_dim, output_dim)

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
        sample = self.input_proj(sample)
        cond = self.cond_proj(cond)
        time_embd = self.time_embedding(self.time_proj(timestep).to(sample.device))
        sample = sample + time_embd[:, None]
        # sample = torch.cat([sample, time_embd[:, None]], dim=1)
        if tgt_key_padding_mask is not None:
            sample = self.decoder(sample, cond, tgt_key_padding_mask=tgt_key_padding_mask)
        else:
            sample = self.decoder(sample, cond)
        sample = self.output_proj(sample)
        return sample