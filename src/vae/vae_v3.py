# The third version on Dec. 2025
# Based on vae_v2.py but using FSQ (Finite Scalar Quantization) instead of VAE's reparameterization
# FSQ provides discrete latent codes without requiring KL divergence loss

# Key differences from v2:
# - Replaces mu/logvar with direct latent encoding + FSQ quantization
# - No reparameterization trick needed
# - Returns quantized latent and codebook indices
# - Simpler training (no KL loss required)

import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
import sys
from pathlib import Path

# Add FSQ-pytorch to path
# Path: src/vae/vae_v3.py -> src/vae -> src -> CLR-Wire -> third_party/FSQ-pytorch
fsq_path = Path(__file__).parent.parent.parent / 'third_party' / 'FSQ-pytorch'
sys.path.insert(0, str(fsq_path))

from quantizers import FSQ

ic.disable()

class SurfaceVAE_FSQ(nn.Module):
    def __init__(self, 
                 param_raw_dim,
                 param_dim=32,           # 每个曲面参数向量长度（补齐后的）
                 latent_dim=128,         # 潜空间维度
                 n_surface_types=5,      # 曲面种类数
                 emb_dim=16,             # embedding维度
                 fsq_levels=[8,5,5,5]):  # FSQ量化级别
        super().__init__()
        assert len(param_raw_dim) == n_surface_types
        self.param_raw_dim = param_raw_dim # Input raw parameter dimension for each surface type
        self.param_dim = param_dim # Output unified parameter dimension for each surface type
        self.max_raw_dim = max(param_raw_dim)
        self.latent_dim = latent_dim
        self.fsq_levels = fsq_levels
        
        # 曲面类型 embedding
        self.type_emb = nn.Embedding(n_surface_types, emb_dim)
        self.param_emb_list = nn.ModuleList([
            nn.Linear(param_raw_dim[i], param_dim) 
            for i in range(n_surface_types)
        ])
        
        # Encoder - 直接输出latent，不需要mu和logvar
        self.encoder = nn.Sequential(
            nn.Linear(param_dim + emb_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # FSQ量化器
        # FSQ会自动处理维度投影：latent_dim -> len(fsq_levels) -> latent_dim
        self.fsq = FSQ(levels=fsq_levels, dim=latent_dim)
        
        # Codebook信息（用于分析和调试）
        self.codebook_size = self.fsq.codebook_size
        ic(f"FSQ codebook size: {self.codebook_size}")
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, param_dim)  # 输出重建参数
        )

        # Classifiers
        self.classifier = nn.Linear(latent_dim, n_surface_types)
        self.is_closed_classifier = nn.Linear(latent_dim, 2)  # [u_closed, v_closed]
        
        # Each head maps unified param embedding to raw param dim of that type
        self.decoder_raw_list = nn.ModuleList([
            nn.Linear(param_dim, param_raw_dim[i]) 
            for i in range(n_surface_types)
        ])

    def encode(self, params_raw, surface_type):
        """
        编码并量化
        
        Args:
            params_raw: (B, max_raw_dim) padded raw parameters
            surface_type: (B,) surface type indices
            
        Returns:
            z_quantized: (B, latent_dim) quantized latent codes
            indices: (B,) codebook indices
        """
        assert params_raw.shape[1] == self.max_raw_dim # Padded to the max dim
        emb = self.type_emb(surface_type)             # (B, emb_dim)
        
        # Apply type-specific input projection per unique type in batch
        batch_size = params_raw.size(0)
        device = params_raw.device
        param_emb = torch.empty(batch_size, self.param_dim, device=device, dtype=emb.dtype)
        
        unique_types = torch.unique(surface_type)
        for t in unique_types:
            idx = (surface_type == t).nonzero(as_tuple=True)[0]
            t_int = int(t.item())
            raw_dim = self.param_raw_dim[t_int]
            param_emb[idx] = self.param_emb_list[t_int](
                params_raw.index_select(0, idx)[:, :raw_dim]
            )
        
        x = torch.cat([param_emb, emb], dim=-1)
        z_continuous = self.encoder(x)  # (B, latent_dim)
        
        # FSQ量化
        # FSQ expects at least 3D input (batch, seq, dim) or 4D for images
        # For 1D latent vectors, we add a dummy sequence dimension
        z_continuous = z_continuous.unsqueeze(1)  # (B, 1, latent_dim)
        z_quantized, indices = self.fsq(z_continuous)  # (B, 1, latent_dim), (B, 1)
        z_quantized = z_quantized.squeeze(1)  # (B, latent_dim)
        indices = indices.squeeze(1)  # (B,)
        
        return z_quantized, indices

    def classify(self, z):
        """
        分类器：预测曲面类型和是否闭合
        
        Args:
            z: (B, latent_dim) latent codes
            
        Returns:
            logits: (B, n_surface_types) surface type logits
            class_type: (B,) predicted surface types
            logits_is_closed: (B, 2) [u_closed, v_closed] logits
            is_closed: (B, 2) boolean predictions
        """
        logits = self.classifier(z)
        logits_is_closed = self.is_closed_classifier(z)
        class_type = logits.argmax(dim=-1)
        is_closed = torch.sigmoid(logits_is_closed) > 0.5
        return logits, class_type, logits_is_closed, is_closed

    def decode(self, z, surface_type):
        """
        解码：从latent code重建参数
        
        Args:
            z: (B, latent_dim) latent codes
            surface_type: (B,) surface type indices
            
        Returns:
            padded: (B, max_raw_dim) reconstructed parameters (padded)
            mask: (B, max_raw_dim) boolean mask for valid dimensions
        """
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
            padded.index_copy_(
                0, idx, 
                torch.cat([out_t, torch.zeros(out_t.size(0), max_dim - dim_t, 
                          device=device)], dim=1)
            )
            # set mask true for valid positions
            mask[idx, :dim_t] = True
            ic('predicted type', t, 'mask dim', dim_t)
        
        return padded, mask

    def inference(self, z):
        """
        推理：从latent code生成曲面参数
        
        Args:
            z: (B, latent_dim) latent codes
            
        Returns:
            padded: (B, max_raw_dim) reconstructed parameters
            mask: (B, max_raw_dim) boolean mask
        """
        surface_logits, surface_type, is_closed_logits, is_closed = self.classify(z)
        return self.decode(z, surface_type)

    def forward(self, params, surface_type):
        """
        完整前向传播
        
        Args:
            params: (B, max_raw_dim) padded raw parameters
            surface_type: (B,) surface type indices
            
        Returns:
            recon: (B, max_raw_dim) reconstructed parameters
            mask: (B, max_raw_dim) valid dimension mask
            class_logits: (B, n_surface_types) surface type logits
            is_closed_logits: (B, 2) [u_closed, v_closed] logits
            z_quantized: (B, latent_dim) quantized latent codes
            indices: (B,) codebook indices
        """
        # 编码+量化
        z_quantized, indices = self.encode(params, surface_type)
        
        # 分类
        class_logits, surface_type_pred, is_closed_logits, is_closed = self.classify(z_quantized)
        
        # 解码
        recon, mask = self.decode(z_quantized, surface_type)
        
        return recon, mask, class_logits, is_closed_logits, z_quantized, indices

    def get_codebook_usage(self, indices):
        """
        计算codebook利用率
        
        Args:
            indices: (B,) or (N,) codebook indices from a batch or dataset
            
        Returns:
            usage_ratio: float, proportion of codebook entries used
            unique_codes: int, number of unique codes used
        """
        unique_codes = torch.unique(indices).numel()
        usage_ratio = unique_codes / self.codebook_size
        return usage_ratio, unique_codes

    def indices_to_latent(self, indices):
        """
        从codebook索引恢复latent code
        
        Args:
            indices: (B,) codebook indices
            
        Returns:
            z: (B, latent_dim) latent codes
        """
        # FSQ的indices_to_codes方法可以从索引恢复codes
        indices = indices.unsqueeze(1)  # (B, 1)
        z = self.fsq.indices_to_codes(indices, project_out=True)  # (B, 1, latent_dim)
        z = z.squeeze(1)  # (B, latent_dim)
        return z


if __name__ == "__main__":
    # 测试不同的FSQ配置
    print("=" * 60)
    print("Testing SurfaceVAE_FSQ with different configurations")
    print("=" * 60)
    
    # 配置1：小codebook (快速实验)
    print("\n[Config 1] Small codebook")
    fsq_levels_small = [8, 6, 5, 4]  # codebook_size = 960
    model1 = SurfaceVAE_FSQ(
        param_raw_dim=[17, 18, 19, 19, 18],  # plane, cylinder, cone, torus, sphere
        param_dim=32, 
        latent_dim=128, 
        n_surface_types=5, 
        emb_dim=16,
        fsq_levels=fsq_levels_small
    )
    print(f"Codebook size: {model1.codebook_size}")
    
    # 配置2：中等codebook (推荐)
    print("\n[Config 2] Medium codebook (recommended)")
    fsq_levels_medium = [8, 8, 8, 5, 5, 5]  # codebook_size = 128,000
    model2 = SurfaceVAE_FSQ(
        param_raw_dim=[17, 18, 19, 19, 18],
        param_dim=32, 
        latent_dim=128, 
        n_surface_types=5, 
        emb_dim=16,
        fsq_levels=fsq_levels_medium
    )
    print(f"Codebook size: {model2.codebook_size}")
    
    # 配置3：大codebook (高精度)
    print("\n[Config 3] Large codebook")
    fsq_levels_large = [8, 8, 8, 8, 5, 5, 5, 5]  # codebook_size = 2,560,000
    model3 = SurfaceVAE_FSQ(
        param_raw_dim=[17, 18, 19, 19, 18],
        param_dim=32, 
        latent_dim=128, 
        n_surface_types=5, 
        emb_dim=16,
        fsq_levels=fsq_levels_large
    )
    print(f"Codebook size: {model3.codebook_size}")
    
    # 测试前向传播
    print("\n" + "=" * 60)
    print("Testing forward pass")
    print("=" * 60)
    
    batch_size = 4
    params = torch.randn(batch_size, 19)  # max_raw_dim = 19
    surface_type = torch.randint(0, 5, (batch_size,))  # random surface types
    
    print(f"\nInput shape: {params.shape}")
    print(f"Surface types: {surface_type}")
    
    with torch.no_grad():
        recon, mask, class_logits, is_closed_logits, z_quantized, indices = model2(params, surface_type)
    
    print(f"\nOutputs:")
    print(f"  recon shape: {recon.shape}")
    print(f"  mask shape: {mask.shape}")
    print(f"  class_logits shape: {class_logits.shape}")
    print(f"  is_closed_logits shape: {is_closed_logits.shape}")
    print(f"  z_quantized shape: {z_quantized.shape}")
    print(f"  indices shape: {indices.shape}")
    print(f"  indices values: {indices}")
    
    # 测试codebook利用率
    usage_ratio, unique_codes = model2.get_codebook_usage(indices)
    print(f"\nCodebook usage: {unique_codes}/{model2.codebook_size} ({usage_ratio*100:.2f}%)")
    
    # 测试从indices恢复latent
    z_recovered = model2.indices_to_latent(indices)
    print(f"\nRecovered latent shape: {z_recovered.shape}")
    print(f"Quantization error: {(z_quantized - z_recovered).abs().max().item():.6f}")
    
    # 测试推理
    print("\n" + "=" * 60)
    print("Testing inference")
    print("=" * 60)
    
    with torch.no_grad():
        recon_inf, mask_inf = model2.inference(z_quantized)
    print(f"Inference output shape: {recon_inf.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
