import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

class BSplineSurfaceLayer(nn.Module):
    """
    Ultra-optimized version using pre-computed tensor operations.
    """
    
    def __init__(self, resolution=32, device=None):
        super(BSplineSurfaceLayer, self).__init__()
        self.resolution = resolution
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._precompute_all()
        
    def _precompute_all(self):
        """Pre-compute everything possible for maximum speed"""
        params = torch.linspace(0, 1, self.resolution, device=self.device)
        t = params.unsqueeze(1)
        
        # Compute basis functions
        basis = torch.zeros(self.resolution, 4, device=self.device)
        basis[:, 0] = (1 - t).pow(3).squeeze()
        basis[:, 1] = (3 * t * (1 - t).pow(2)).squeeze()
        basis[:, 2] = (3 * t.pow(2) * (1 - t)).squeeze()
        basis[:, 3] = t.pow(3).squeeze()
        
        # Pre-compute outer product for all combinations
        # Shape: (M, M, 4, 4)
        # Replace einsum with outer product using rearrange
        basis_u = rearrange(basis, 'u i -> u 1 i 1')  # (M, 1, 4, 1)
        basis_v = rearrange(basis, 'v j -> 1 v 1 j')  # (1, M, 1, 4)
        basis_outer = basis_u * basis_v  # Broadcasting: (M, M, 4, 4)
        
        # Reshape to (M*M, 16) for matrix multiplication
        self.register_buffer('basis_matrix', basis_outer.view(self.resolution * self.resolution, 16))
        
    def forward(self, control_points):
        """
        Ultra-optimized forward pass.
        
        Args:
            control_points: (B, 16, 3) tensor of control points
            
        Returns:
            surface_points: (B, M, M, 3) tensor of evaluated surface points
        """
        batch_size = control_points.shape[0]
        
        # Replace einsum with matrix multiplication using rearrange
        # self.basis_matrix: (M*M, 16), control_points: (B, 16, 3)
        # Rearrange control_points to (16, B*3) for matrix multiplication
        control_points_reshaped = rearrange(control_points, 'b i j -> i (b j)')
        
        # Matrix multiplication: (M*M, 16) @ (16, B*3) -> (M*M, B*3)
        surface_flat_reshaped = self.basis_matrix @ control_points_reshaped
        
        # Rearrange back to (B, M*M, 3)
        surface_flat = rearrange(surface_flat_reshaped, '(m_sq) (b j) -> b m_sq j', 
                                b=batch_size, j=3, m_sq=self.resolution * self.resolution)
        
        # Reshape to final form
        return surface_flat.view(batch_size, self.resolution, self.resolution, 3)

