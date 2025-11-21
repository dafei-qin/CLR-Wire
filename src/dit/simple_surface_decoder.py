import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from icecream import ic

ic.disable()

class SimpleSurfaceDecoder(nn.Module):
    def __init__(self, output_dim=128, latent_dim=128, num_layers=4, num_heads=8):
        super(SimpleSurfaceDecoder, self).__init__()
        layer = nn.TransformerDecoderLayer(latent_dim, num_heads, dim_feedforward=latent_dim * 4, dropout=0.1, activation=F.gelu, batch_first=True, norm_first=True, layer_norm_eps=1e-4)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
    def forward(self, z):
        z = self.decoder(z)
        z = self.output_proj(z)
        return z