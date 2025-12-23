"""
VAE v4: DC-AE + FSQ
Version 4 on Dec. 2024

Architecture:
    Input (B, 3, 4, 4)
    ↓ DC-AE Encoder
    Latent (B, 3, 2, 2) → reshape → (B, 12)
    ↓ FSQ Quantization
    Quantized (B, 12) + Indices (B, num_codebooks)
    ↓ reshape → (B, 3, 2, 2)
    ↓ DC-AE Decoder
    Output (B, 3, 4, 4)

Key features:
- Uses DC-AE for spatial encoding (4x4→2x2, 2x compression)
- FSQ for discrete latent codes (no KL loss needed)
- Maintains spatial structure through reshape operations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic

# Add DC-AE module
from src.vae.dc_ae import create_minimal_dcae

# Add FSQ-pytorch to path
fsq_path = Path(__file__).parent.parent.parent / 'third_party' / 'FSQ-pytorch'
sys.path.insert(0, str(fsq_path))
from quantizers import FSQ

ic.disable()


class DCAE_FSQ_VAE(nn.Module):
    """
    DC-AE + FSQ based VAE for 4x4x3 inputs
    
    Args:
        input_size: Input spatial size (default: 4 for 4x4)
        in_channels: Input channels (default: 3 for RGB)
        latent_channels: DC-AE latent channels (default: 3)
        width_base: DC-AE base width (default: 64)
        fsq_levels: FSQ quantization levels (default: [32, 32] for 1024 codes)
        num_codebooks: Number of FSQ codebooks (default: 8)
    """
    
    def __init__(
        self,
        input_size: int = 4,
        in_channels: int = 3,
        latent_channels: int = 3,
        width_base: int = 64,
        fsq_levels: list = None,
        num_codebooks: int = 8,

    ):
        super().__init__()
        
        # Default FSQ levels: [32, 32] gives codebook_size = 1024
        if fsq_levels is None:
            fsq_levels = [32, 32]
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.fsq_levels = fsq_levels
        self.num_codebooks = num_codebooks
        
        # Calculate dimensions
        self.spatial_compression = 2  # DC-AE compresses by 2x
        self.latent_spatial_size = input_size // self.spatial_compression  # 4 → 2
        self.latent_dim = latent_channels * (self.latent_spatial_size ** 2)  # 3 * 2 * 2 = 12
        
        print(f"\n{'='*70}")
        print(f"Initializing DCAE_FSQ_VAE")
        print(f"{'='*70}")
        print(f"Input:  ({in_channels}, {input_size}, {input_size})")
        print(f"Latent: ({latent_channels}, {self.latent_spatial_size}, {self.latent_spatial_size}) "
              f"= {self.latent_dim}D")
        
        # Create DC-AE (4x4x3 → 2x2x3)
        self.dcae = create_minimal_dcae(
            input_size=input_size,
            in_channels=in_channels,
            latent_channels=latent_channels,
            width_base=width_base,
        )
        
        # FSQ configuration
        codebook_dim = len(fsq_levels)
        self.effective_dim = codebook_dim * num_codebooks
        
        print(f"\nFSQ Configuration:")
        print(f"  Levels: {fsq_levels} (dim={codebook_dim})")
        print(f"  Num codebooks: {num_codebooks}")
        print(f"  Effective dimension: {self.effective_dim}")
        print(f"  Latent dimension: {self.latent_dim}")
        
        # Check if projection is needed
        if self.latent_dim != self.effective_dim:
            print(f"  ⚠️  Dimension mismatch: {self.latent_dim} ≠ {self.effective_dim}")
            print(f"  → FSQ will use projection layers")
            self.needs_projection = True
        else:
            print(f"  ✓ Perfect dimension match!")
            self.needs_projection = False
        
        # FSQ quantizer
        self.fsq = FSQ(
            levels=fsq_levels,
            dim=self.latent_dim,
            num_codebooks=num_codebooks
        )
        
        # Codebook information
        self.codebook_size = self.fsq.codebook_size
        self.total_codebook_capacity = self.codebook_size ** num_codebooks
        
        print(f"\nCodebook Information:")
        print(f"  Single codebook size: {self.codebook_size}")
        print(f"  Total capacity: {self.codebook_size}^{num_codebooks} = {self.total_codebook_capacity:,}")
        print(f"  Has projections: {self.fsq.has_projections}")
        print(f"{'='*70}\n")
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to quantized latent codes
        
        Args:
            x: (B, C, H, W) input tensor, e.g., (B, 3, 4, 4)
            
        Returns:
            z_quantized: (B, C_latent, H', W') quantized latent, e.g., (B, 3, 2, 2)
            indices: (B, num_codebooks) codebook indices
        """
        batch_size = x.size(0)
        
        # DC-AE encode: (B, 3, 4, 4) → (B, 3, 2, 2)
        z_continuous = self.dcae.encode(x)
        
        # Reshape to 1D: (B, 3, 2, 2) → (B, 12)
        z_flat = z_continuous.reshape(batch_size, -1)
        
        # FSQ expects at least 3D input: (B, seq, dim)
        z_flat = z_flat.unsqueeze(1)  # (B, 1, 12)
        
        # FSQ quantization
        z_quantized_flat, indices = self.fsq(z_flat)  # (B, 1, 12), (B, 1, num_codebooks)
        
        # Remove sequence dimension
        z_quantized_flat = z_quantized_flat.squeeze(1)  # (B, 12)
        
        # Reshape back to spatial: (B, 12) → (B, 3, 2, 2)
        z_quantized = z_quantized_flat.reshape(
            batch_size, 
            self.latent_channels, 
            self.latent_spatial_size, 
            self.latent_spatial_size
        )
        
        # Handle indices shape
        if self.num_codebooks > 1:
            indices = indices.squeeze(1)  # (B, num_codebooks)
        else:
            indices = indices.squeeze(1)  # (B,)
        
        return z_quantized, indices
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to output
        
        Args:
            z: (B, C_latent, H', W') latent tensor, e.g., (B, 3, 2, 2)
            
        Returns:
            x_recon: (B, C, H, W) reconstructed output, e.g., (B, 3, 4, 4)
        """
        # DC-AE decode: (B, 3, 2, 2) → (B, 3, 4, 4)
        x_recon = self.dcae.decode(z)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Full forward pass: encode → quantize → decode
        
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            x_recon: (B, C, H, W) reconstructed output
            z_quantized: (B, C_latent, H', W') quantized latent
            indices: (B, num_codebooks) codebook indices
            metrics: dict with additional metrics
        """
        # Encode and quantize
        z_quantized, indices = self.encode(x)
        
        # Decode
        x_recon = self.decode(z_quantized)
        
        # Compute metrics
        metrics = {
            'latent_shape': z_quantized.shape,
            'indices_shape': indices.shape,
        }
        
        return x_recon, z_quantized, indices, metrics
    
    def get_codebook_usage(self, indices: torch.Tensor) -> tuple[float, int]:
        """
        Calculate codebook utilization
        
        Args:
            indices: (B,) or (B, num_codebooks) codebook indices
            
        Returns:
            usage_ratio: Proportion of codebook entries used
            unique_codes: Number of unique codes used
        """
        if indices.ndim == 2:
            # Multiple codebooks
            usage_per_codebook = []
            unique_codes_list = []
            for i in range(indices.shape[1]):
                unique_codes = torch.unique(indices[:, i]).numel()
                unique_codes_list.append(unique_codes)
                usage_per_codebook.append(unique_codes / self.codebook_size)
            usage_ratio = sum(usage_per_codebook) / len(usage_per_codebook)
            unique_codes = int(sum(unique_codes_list) / len(unique_codes_list))
        else:
            # Single codebook
            unique_codes = torch.unique(indices).numel()
            usage_ratio = unique_codes / self.codebook_size
        
        return usage_ratio, unique_codes
    
    def indices_to_latent(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Recover latent codes from codebook indices
        
        Args:
            indices: (B,) or (B, num_codebooks) codebook indices
            
        Returns:
            z: (B, C_latent, H', W') latent codes
        """
        batch_size = indices.size(0)
        
        # Ensure indices have correct shape
        if indices.ndim == 1:
            indices = indices.unsqueeze(1)  # (B, 1)
        
        # FSQ indices to codes
        z_flat = self.fsq.indices_to_codes(indices, project_out=True)  # (B, 1, dim)
        z_flat = z_flat.squeeze(1)  # (B, dim)
        
        # Reshape to spatial
        z = z_flat.view(
            batch_size,
            self.latent_channels,
            self.latent_spatial_size,
            self.latent_spatial_size
        )
        
        return z
    
    def generate_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Generate output from codebook indices
        
        Args:
            indices: (B,) or (B, num_codebooks) codebook indices
            
        Returns:
            x_recon: (B, C, H, W) reconstructed output
        """
        z = self.indices_to_latent(indices)
        x_recon = self.decode(z)
        return x_recon
    
    def get_reconstruction_loss(self, x: torch.Tensor, x_recon: torch.Tensor, 
                               reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction loss (MSE or L1)
        
        Args:
            x: (B, C, H, W) original input
            x_recon: (B, C, H, W) reconstructed output
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: Reconstruction loss
        """
        # Using MSE loss (you can switch to L1 if needed)
        loss = F.mse_loss(x_recon, x, reduction=reduction)
        return loss
    
    @torch.no_grad()
    def visualize_latent(self, x: torch.Tensor) -> dict:
        """
        Visualize latent codes for debugging
        
        Args:
            x: (B, C, H, W) input tensor
            
        Returns:
            vis_dict: Dictionary with visualization data
        """
        z_quantized, indices = self.encode(x)
        
        # Flatten latent for analysis
        z_flat = z_quantized.view(x.size(0), -1)
        
        vis_dict = {
            'latent_mean': z_flat.mean(dim=0).cpu(),
            'latent_std': z_flat.std(dim=0).cpu(),
            'latent_min': z_flat.min(dim=0)[0].cpu(),
            'latent_max': z_flat.max(dim=0)[0].cpu(),
            'indices': indices.cpu(),
            'indices_unique': torch.unique(indices).cpu() if indices.ndim == 1 else 
                             [torch.unique(indices[:, i]).cpu() for i in range(indices.shape[1])],
        }
        
        return vis_dict


def create_dcae_fsq_vae(
    input_size: int = 4,
    in_channels: int = 3,
    latent_channels: int = 3,
    width_base: int = 64,
    fsq_levels: list = None,
    num_codebooks: int = 8,
) -> DCAE_FSQ_VAE:
    """
    Convenience function to create DCAE_FSQ_VAE model
    
    Args:
        input_size: Input spatial size (default: 4)
        in_channels: Input channels (default: 3)
        latent_channels: Latent channels (default: 3)
        width_base: DC-AE base width (default: 64)
        fsq_levels: FSQ levels (default: [32, 32] for 1024 codes)
        num_codebooks: Number of codebooks (default: 8)
    
    Returns:
        DCAE_FSQ_VAE model instance
    
    Example:
        >>> model = create_dcae_fsq_vae()
        >>> x = torch.randn(4, 3, 4, 4)
        >>> x_recon, z_quantized, indices, metrics = model(x)
    """
    return DCAE_FSQ_VAE(
        input_size=input_size,
        in_channels=in_channels,
        latent_channels=latent_channels,
        width_base=width_base,
        fsq_levels=fsq_levels,
        num_codebooks=num_codebooks,
    )


if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print(" " * 20 + "DCAE_FSQ_VAE Test Suite")
    print("🚀" * 35 + "\n")
    
    # Test configuration 1: Default (8 codebooks, levels=[32,32])
    print("=" * 70)
    print("Test 1: Default Configuration")
    print("=" * 70)
    
    model1 = create_dcae_fsq_vae(
        input_size=4,
        in_channels=3,
        latent_channels=3,
        width_base=64,
        fsq_levels=[32, 32],  # 1024 codes per codebook
        num_codebooks=8
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 4, 4)
    print(f"\nInput shape: {tuple(x.shape)}")
    
    with torch.no_grad():
        x_recon, z_quantized, indices, metrics = model1(x)
    
    print(f"\nOutputs:")
    print(f"  Reconstructed shape: {tuple(x_recon.shape)}")
    print(f"  Latent shape: {tuple(z_quantized.shape)}")
    print(f"  Indices shape: {tuple(indices.shape)}")
    print(f"  Metrics: {metrics}")
    
    # Reconstruction loss
    recon_loss = model1.get_reconstruction_loss(x, x_recon)
    print(f"\nReconstruction loss (MSE): {recon_loss.item():.6f}")
    
    # Codebook usage
    usage_ratio, unique_codes = model1.get_codebook_usage(indices)
    print(f"\nCodebook usage: {unique_codes}/{model1.codebook_size} ({usage_ratio*100:.2f}%)")
    
    # Test indices to latent
    z_recovered = model1.indices_to_latent(indices)
    print(f"\nRecovered latent shape: {tuple(z_recovered.shape)}")
    quantization_error = (z_quantized - z_recovered).abs().max().item()
    print(f"Quantization error: {quantization_error:.10f}")
    
    # Test generation from indices
    x_generated = model1.generate_from_indices(indices)
    print(f"\nGenerated from indices shape: {tuple(x_generated.shape)}")
    generation_error = (x_recon - x_generated).abs().max().item()
    print(f"Generation error: {generation_error:.10f}")
    
    # Visualize latent
    vis_dict = model1.visualize_latent(x)
    print(f"\nLatent statistics:")
    print(f"  Mean: {vis_dict['latent_mean'].mean():.4f}")
    print(f"  Std:  {vis_dict['latent_std'].mean():.4f}")
    print(f"  Min:  {vis_dict['latent_min'].min():.4f}")
    print(f"  Max:  {vis_dict['latent_max'].max():.4f}")
    
    # Test configuration 2: Different FSQ levels
    print("\n" + "=" * 70)
    print("Test 2: Alternative Configuration (levels=[16,16,16])")
    print("=" * 70)
    
    model2 = create_dcae_fsq_vae(
        input_size=4,
        in_channels=3,
        latent_channels=3,
        fsq_levels=[16, 16, 16],  # 4096 codes per codebook (3D)
        num_codebooks=4
    )
    
    with torch.no_grad():
        x_recon2, z_quantized2, indices2, metrics2 = model2(x)
    
    recon_loss2 = model2.get_reconstruction_loss(x, x_recon2)
    print(f"\nReconstruction loss (MSE): {recon_loss2.item():.6f}")
    
    usage_ratio2, unique_codes2 = model2.get_codebook_usage(indices2)
    print(f"Codebook usage: {unique_codes2}/{model2.codebook_size} ({usage_ratio2*100:.2f}%)")
    
    # Test configuration 3: Perfect dimension match (12 = 12*1)
    print("\n" + "=" * 70)
    print("Test 3: Perfect Dimension Match (num_codebooks=12)")
    print("=" * 70)
    
    model3 = create_dcae_fsq_vae(
        input_size=4,
        in_channels=3,
        latent_channels=3,
        fsq_levels=[32, 32],  # 1024 codes
        num_codebooks=12  # 12 codebooks for 12D latent (perfect fit!)
    )
    
    with torch.no_grad():
        x_recon3, z_quantized3, indices3, metrics3 = model3(x)
    
    recon_loss3 = model3.get_reconstruction_loss(x, x_recon3)
    print(f"\nReconstruction loss (MSE): {recon_loss3.item():.6f}")
    
    # Test different batch sizes
    print("\n" + "=" * 70)
    print("Test 4: Different Batch Sizes")
    print("=" * 70)
    
    for bs in [1, 2, 8, 16]:
        x_test = torch.randn(bs, 3, 4, 4)
        with torch.no_grad():
            x_recon_test, _, indices_test, _ = model1(x_test)
        assert x_recon_test.shape == x_test.shape
        print(f"  Batch size {bs:2d}: ✓ Output shape {tuple(x_recon_test.shape)}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    
    print("\n📋 Model Summary:")
    total_params = sum(p.numel() for p in model1.parameters())
    dcae_params = sum(p.numel() for p in model1.dcae.parameters())
    fsq_params = sum(p.numel() for p in model1.fsq.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  DC-AE parameters: {dcae_params:,} ({dcae_params/total_params*100:.1f}%)")
    print(f"  FSQ parameters: {fsq_params:,} ({fsq_params/total_params*100:.1f}%)")
    print(f"  Codebook capacity: {model1.total_codebook_capacity:,}")
    
    print("\n📖 Usage Example:")
    print("""
from src.vae.vae_v4_dcae_fsq import create_dcae_fsq_vae

# Create model
model = create_dcae_fsq_vae(
    input_size=4,
    in_channels=3,
    latent_channels=3,
    fsq_levels=[32, 32],  # 1024 codes per codebook
    num_codebooks=8
)

# Forward pass
x = torch.randn(batch_size, 3, 4, 4)
x_recon, z_quantized, indices, metrics = model(x)

# Compute loss (no KL loss needed!)
recon_loss = model.get_reconstruction_loss(x, x_recon)
loss = recon_loss  # That's it!

# Generate from indices
x_generated = model.generate_from_indices(indices)
""")
    
    print("\n" + "=" * 70)

