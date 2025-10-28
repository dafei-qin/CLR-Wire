# The first version on Oct. 2025, with coefficients as inputs, and seperate headers of encoders and decoders for mapping coefficients to the latent space with unified length.

# With an additional classifier to predict the class of surface base on the latent code.

# This version does not support B-spline surfaces. 

# It has a total of 5 surface types:

# 1. plane: position x 3 + direction x 3 + XDirection x 3 + UV x 8 = 17

# 2. cylinder: 17 + radius =                        18
# 3. cone: 17 + radius + semi_angle =               19
# 4. torus: 17 + major_radius + minor_radius =      19
# 5. sphere: 17 + radius =                          18


'''

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

ic.disable()
class SurfaceVAE(nn.Module):
    def __init__(self, 
                 param_raw_dim,
                 param_dim=32,       # 每个曲面参数向量长度（补齐后的）
                 latent_dim=128,       # 潜空间维度
                 n_surface_types=5,  # 曲面种类数
                 emb_dim=16):         # embedding维度
        super().__init__()
        assert len(param_raw_dim) == n_surface_types
        self.param_raw_dim = param_raw_dim # Input raw parameter dimension for each surface type
        self.param_dim = param_dim # Output unified parameter dimension for each surface type
        self.max_raw_dim = max(param_raw_dim)
        # 曲面类型 embedding
        self.type_emb = nn.Embedding(n_surface_types, emb_dim)
        self.param_emb_list = nn.ModuleList([nn.Linear(param_raw_dim[i], param_dim) for i in range(n_surface_types)])
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(param_dim + emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        
        # 输出潜空间参数
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, param_dim)  # 输出重建参数
        )

        self.classifier = nn.Linear(latent_dim, n_surface_types)
        # Each head maps unified param embedding to raw param dim of that type
        self.decoder_raw_list = nn.ModuleList([nn.Linear(param_dim, param_raw_dim[i]) for i in range(n_surface_types)])

    # 
    def encode(self, params_raw, surface_type):
        assert params_raw.shape[1] == self.max_raw_dim # Padded to the max dim
        emb = self.type_emb(surface_type)             # (B, emb_dim)
        # Apply type-specific input projection per unique type in batch
        batch_size = params_raw.size(0)
        device = params_raw.device
        param_emb = torch.empty(batch_size, self.param_dim, device=device, dtype=emb.dtype) # Can't support half now
        unique_types = torch.unique(surface_type)
        for t in unique_types:
            idx = (surface_type == t).nonzero(as_tuple=True)[0]
            param_emb[idx] = self.param_emb_list[int(t.item())](params_raw.index_select(0, idx)[:, :self.param_raw_dim[int(t.item())]])
        x = torch.cat([param_emb, emb], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classify(self, z):
        logits = self.classifier(z)
        class_type = logits.argmax(dim=-1)
        return logits, class_type

    def decode(self, z, surface_type):
        
        # Input latent code and surface type, output padded raw parameters

        emb = self.type_emb(surface_type)
        x = torch.cat([z, emb], dim=-1)
        param_emb = self.decoder(x)
        # Apply type-specific output head per unique type in batch
        batch_size = z.size(0)
        device = z.device
        max_dim = self.max_raw_dim
        padded = torch.zeros(batch_size, max_dim, device=device)
        mask = torch.zeros(batch_size, max_dim, dtype=torch.bool, device=device)
        unique_types = torch.unique(surface_type)
        for t in unique_types:
            t_int = int(t.item())
            idx = (surface_type == t).nonzero(as_tuple=True)[0]
            out_t = self.decoder_raw_list[t_int](param_emb.index_select(0, idx))
            dim_t = out_t.size(1)
            padded.index_copy_(0, idx, torch.cat([out_t, torch.zeros(out_t.size(0), max_dim - dim_t, device=device)], dim=1))
            # set mask true for valid positions
            mask[idx, :dim_t] = True
            ic('predicted type', t, 'mask dim', dim_t)
            # mask.index_put_((idx, torch.arange(dim_t, device=device)), torch.ones(out_t.size(0), dim_t, dtype=torch.bool, device=device))
        return padded, mask

    def inference(self, z):
        surface_logits, surface_type = self.classify(z)
        # surface_type = torch.argmax(surface_logits, dim=-1)
        return self.decode(z, surface_type)

    def forward(self, params, surface_type):
        mu, logvar = self.encode(params, surface_type)
        z = self.reparameterize(mu, logvar)
        class_logits, surface_type_pred = self.classify(z)
        recon, mask = self.decode(z, surface_type)
        return recon, mask, class_logits, mu, logvar


if __name__ == "__main__":

    model = SurfaceVAE(param_raw_dim=[10, 11, 12, 12, 11], param_dim=16, latent_dim=32, n_surface_types=5, emb_dim=16)
    