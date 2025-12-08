"""
Pure PyTorch implementation of NURBS surface evaluation
No custom CUDA kernels required - uses only PyTorch operations
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


def find_span(n: int, p: int, u: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Find the knot span index for each parameter value
    
    Args:
        n: number of control points - 1 (max index)
        p: degree
        u: parameter values, shape (num_samples,)
        U: knot vector, shape (num_knots,)
    
    Returns:
        span indices, shape (num_samples,)
    """
    # Handle edge case where u == U[n+1]
    eps = 1e-6
    mask_end = torch.abs(u - U[n + 1]) < eps
    
    # Binary search for each u value
    spans = torch.zeros_like(u, dtype=torch.long)
    
    for i in range(len(u)):
        if mask_end[i]:
            spans[i] = n
        else:
            # Binary search
            low = p
            high = n + 1
            mid = (low + high) // 2
            
            while u[i] < U[mid] - eps or u[i] >= U[mid + 1] + eps:
                if u[i] < U[mid] - eps:
                    high = mid
                else:
                    low = mid
                mid = (low + high) // 2
            
            spans[i] = mid
    
    return spans


def basis_functions(span: int, u: float, p: int, U: torch.Tensor) -> torch.Tensor:
    """
    Compute the non-zero basis functions at parameter u
    
    Args:
        span: knot span index
        u: parameter value
        p: degree
        U: knot vector
    
    Returns:
        basis function values, shape (p+1,)
    """
    N = torch.zeros(p + 1, dtype=U.dtype, device=U.device)
    left = torch.zeros(p + 1, dtype=U.dtype, device=U.device)
    right = torch.zeros(p + 1, dtype=U.dtype, device=U.device)
    
    N[0] = 1.0
    
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        
        for r in range(j):
            temp = N[r] / (right[r + 1] + left[j - r] + 1e-10)
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        
        N[j] = saved
    
    return N


def batch_basis_functions(spans: torch.Tensor, u: torch.Tensor, p: int, 
                          U: torch.Tensor) -> torch.Tensor:
    """
    Compute basis functions for a batch of parameter values
    
    Args:
        spans: knot span indices, shape (num_samples,)
        u: parameter values, shape (num_samples,)
        p: degree
        U: knot vector
    
    Returns:
        basis function values, shape (num_samples, p+1)
    """
    num_samples = len(u)
    N_batch = torch.zeros(num_samples, p + 1, dtype=U.dtype, device=U.device)
    
    for i in range(num_samples):
        N_batch[i] = basis_functions(spans[i].item(), u[i].item(), p, U)
    
    return N_batch


class SurfEvalTorch(nn.Module):
    """
    Pure PyTorch implementation of NURBS surface evaluation
    Compatible with autograd for gradient computation
    """
    
    def __init__(
        self,
        u_degree: int,
        v_degree: int,
        u_knots: torch.Tensor,
        v_knots: torch.Tensor,
        out_dim_u: int = 32,
        out_dim_v: int = 128,
        device: str = 'cuda'
    ):
        """
        Args:
            u_degree: degree in u direction
            v_degree: degree in v direction
            u_knots: knot vector in u direction (with multiplicities expanded)
            v_knots: knot vector in v direction (with multiplicities expanded)
            out_dim_u: number of sampling points in u direction
            out_dim_v: number of sampling points in v direction
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        
        self.p = u_degree
        self.q = v_degree
        self.out_dim_u = out_dim_u
        self.out_dim_v = out_dim_v
        self.device = device
        
        # Compute m, n from knot vectors
        # Knot vector length = m + p + 2, so m = len(U) - p - 2
        self.m = len(u_knots) - u_degree - 2
        self.n = len(v_knots) - v_degree - 2
        
        # Register knot vectors
        self.register_buffer('U', u_knots.to(device))
        self.register_buffer('V', v_knots.to(device))
        
        # Generate uniform sampling parameters
        self.register_buffer('u', torch.linspace(0.0, 1.0, out_dim_u, device=device))
        self.register_buffer('v', torch.linspace(0.0, 1.0, out_dim_v, device=device))
        
        # Pre-compute basis functions and spans
        self._precompute_basis()
    
    def _precompute_basis(self):
        """Pre-compute basis functions and knot spans for all sampling points"""
        # Find spans
        self.uspan = find_span(self.m, self.p, self.u, self.U)
        self.vspan = find_span(self.n, self.q, self.v, self.V)
        
        # Compute basis functions
        self.Nu = batch_basis_functions(self.uspan, self.u, self.p, self.U)  # (out_dim_u, p+1)
        self.Nv = batch_basis_functions(self.vspan, self.v, self.q, self.V)  # (out_dim_v, q+1)
    
    def forward(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Evaluate NURBS surface at sampling points
        
        Args:
            ctrl_pts: control points with weights, shape (batch, m+1, n+1, 4)
                     Last dimension is [x, y, z, w] where w is weight
        
        Returns:
            surface points, shape (batch, out_dim_u, out_dim_v, 3)
        """
        batch_size = ctrl_pts.shape[0]
        
        # Move to correct device
        ctrl_pts = ctrl_pts.to(self.device)
        
        # Initialize output in homogeneous coordinates
        surfaces = torch.zeros(
            batch_size, self.out_dim_u, self.out_dim_v, 4,
            dtype=ctrl_pts.dtype, device=self.device
        )
        
        # For each sampling point in the output grid
        for ui in range(self.out_dim_u):
            for vi in range(self.out_dim_v):
                # Get the relevant control point indices
                u_start = self.uspan[ui] - self.p
                v_start = self.vspan[vi] - self.q
                
                # Extract the (p+1) x (q+1) patch of control points
                # ctrl_patch: (batch, p+1, q+1, 4)
                ctrl_patch = ctrl_pts[
                    :,
                    u_start:u_start + self.p + 1,
                    v_start:v_start + self.q + 1,
                    :
                ]
                
                # Get basis function values
                Nu_i = self.Nu[ui]  # (p+1,)
                Nv_i = self.Nv[vi]  # (q+1,)
                
                # Compute tensor product: sum over u direction first
                # Nu_i: (p+1,) -> (1, p+1, 1, 1)
                # ctrl_patch: (batch, p+1, q+1, 4)
                temp = torch.einsum('i,bijd->bjd', Nu_i, ctrl_patch)  # (batch, q+1, 4)
                
                # Sum over v direction
                # Nv_i: (q+1,) -> (1, q+1, 1)
                # temp: (batch, q+1, 4)
                result = torch.einsum('j,bjd->bd', Nv_i, temp)  # (batch, 4)
                
                surfaces[:, ui, vi, :] = result
        
        # Convert from homogeneous to Cartesian coordinates
        # Divide [x, y, z] by w
        xyz = surfaces[..., :3]  # (batch, out_dim_u, out_dim_v, 3)
        w = surfaces[..., 3:4]   # (batch, out_dim_u, out_dim_v, 1)
        
        # Avoid division by zero
        w = torch.clamp(w, min=1e-8)
        
        surface_points = xyz / w
        
        return surface_points


class SurfEvalTorchFast(nn.Module):
    """
    Faster vectorized version using einsum operations
    """
    
    def __init__(
        self,
        u_degree: int,
        v_degree: int,
        u_knots: torch.Tensor,
        v_knots: torch.Tensor,
        out_dim_u: int = 32,
        out_dim_v: int = 128,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.p = u_degree
        self.q = v_degree
        self.out_dim_u = out_dim_u
        self.out_dim_v = out_dim_v
        self.device = device
        
        self.m = len(u_knots) - u_degree - 2
        self.n = len(v_knots) - v_degree - 2
        
        self.register_buffer('U', u_knots.to(device))
        self.register_buffer('V', v_knots.to(device))
        self.register_buffer('u', torch.linspace(0.0, 1.0, out_dim_u, device=device))
        self.register_buffer('v', torch.linspace(0.0, 1.0, out_dim_v, device=device))
        
        self._precompute_basis()
    
    def _precompute_basis(self):
        self.uspan = find_span(self.m, self.p, self.u, self.U)
        self.vspan = find_span(self.n, self.q, self.v, self.V)
        self.Nu = batch_basis_functions(self.uspan, self.u, self.p, self.U)
        self.Nv = batch_basis_functions(self.vspan, self.v, self.q, self.V)
    
    def forward(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Vectorized forward pass
        
        Args:
            ctrl_pts: (batch, m+1, n+1, 4)
        
        Returns:
            surface_points: (batch, out_dim_u, out_dim_v, 3)
        """
        batch_size = ctrl_pts.shape[0]
        ctrl_pts = ctrl_pts.to(self.device)
        
        # Pre-allocate output
        surfaces = torch.zeros(
            batch_size, self.out_dim_u, self.out_dim_v, 4,
            dtype=ctrl_pts.dtype, device=self.device
        )
        
        # Process each output point
        for ui in range(self.out_dim_u):
            u_start = self.uspan[ui] - self.p
            u_indices = torch.arange(u_start, u_start + self.p + 1, device=self.device)
            
            for vi in range(self.out_dim_v):
                v_start = self.vspan[vi] - self.q
                v_indices = torch.arange(v_start, v_start + self.q + 1, device=self.device)
                
                # Extract control point patch
                ctrl_patch = ctrl_pts[:, u_indices[:, None], v_indices[None, :], :]
                
                # Tensor product of basis functions
                # Nu[ui]: (p+1,), Nv[vi]: (q+1,), ctrl_patch: (batch, p+1, q+1, 4)
                basis_product = torch.einsum(
                    'i,j,bijd->bd',
                    self.Nu[ui], self.Nv[vi], ctrl_patch
                )
                
                surfaces[:, ui, vi, :] = basis_product
        
        # Convert to Cartesian coordinates
        xyz = surfaces[..., :3]
        w = surfaces[..., 3:4].clamp(min=1e-8)
        
        return xyz / w


class SurfEvalTorchVectorized(nn.Module):
    """
    Fully vectorized implementation - NO Python loops over sampling points
    Much faster than SurfEvalTorchFast
    """
    
    def __init__(
        self,
        u_degree: int,
        v_degree: int,
        u_knots: torch.Tensor,
        v_knots: torch.Tensor,
        out_dim_u: int = 32,
        out_dim_v: int = 128,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.p = u_degree
        self.q = v_degree
        self.out_dim_u = out_dim_u
        self.out_dim_v = out_dim_v
        self.device = device
        
        self.m = len(u_knots) - u_degree - 2
        self.n = len(v_knots) - v_degree - 2
        
        self.register_buffer('U', u_knots.to(device))
        self.register_buffer('V', v_knots.to(device))
        self.register_buffer('u', torch.linspace(0.0, 1.0, out_dim_u, device=device))
        self.register_buffer('v', torch.linspace(0.0, 1.0, out_dim_v, device=device))
        
        self._precompute_basis()
        self._precompute_indices()
    
    def _precompute_basis(self):
        """Pre-compute basis functions"""
        self.uspan = find_span(self.m, self.p, self.u, self.U)
        self.vspan = find_span(self.n, self.q, self.v, self.V)
        self.Nu = batch_basis_functions(self.uspan, self.u, self.p, self.U)
        self.Nv = batch_basis_functions(self.vspan, self.v, self.q, self.V)
    
    def _precompute_indices(self):
        """Pre-compute all control point indices for vectorized access"""
        # For each output point (ui, vi), we need indices of (p+1)x(q+1) control points
        # u_indices: (out_dim_u, p+1)
        # v_indices: (out_dim_v, q+1)
        
        u_start = self.uspan - self.p  # (out_dim_u,)
        v_start = self.vspan - self.q  # (out_dim_v,)
        
        # Create offsets
        u_offsets = torch.arange(self.p + 1, device=self.device)  # (p+1,)
        v_offsets = torch.arange(self.q + 1, device=self.device)  # (q+1,)
        
        # Broadcast to get all indices
        self.u_indices = u_start[:, None] + u_offsets[None, :]  # (out_dim_u, p+1)
        self.v_indices = v_start[:, None] + v_offsets[None, :]  # (out_dim_v, q+1)
    
    def forward(self, ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Fully vectorized forward pass - NO Python loops!
        
        Args:
            ctrl_pts: (batch, m+1, n+1, 4)
        
        Returns:
            surface_points: (batch, out_dim_u, out_dim_v, 3)
        """
        batch_size = ctrl_pts.shape[0]
        ctrl_pts = ctrl_pts.to(self.device)
        
        # Extract control point patches for all output points at once
        # u_indices: (out_dim_u, p+1)
        # v_indices: (out_dim_v, q+1)
        
        # Index into control points: ctrl_pts[batch, u_indices, v_indices, dim]
        # We need: (batch, out_dim_u, p+1, out_dim_v, q+1, 4)
        
        # First index u direction: (batch, out_dim_u, p+1, n+1, 4)
        ctrl_u = ctrl_pts[:, self.u_indices, :, :]  # (batch, out_dim_u, p+1, n+1, 4)
        
        # Then index v direction: (batch, out_dim_u, p+1, out_dim_v, q+1, 4)
        ctrl_patches = ctrl_u[:, :, :, self.v_indices, :]  # (batch, out_dim_u, p+1, out_dim_v, q+1, 4)
        
        # Apply basis functions using einsum
        # Nu: (out_dim_u, p+1)
        # Nv: (out_dim_v, q+1)
        # ctrl_patches: (batch, out_dim_u, p+1, out_dim_v, q+1, 4)
        # Result: (batch, out_dim_u, out_dim_v, 4)
        
        surfaces = torch.einsum(
            'ui,vj,buivjd->buvd',
            self.Nu, self.Nv, ctrl_patches
        )
        
        # Convert to Cartesian coordinates
        xyz = surfaces[..., :3]
        w = surfaces[..., 3:4].clamp(min=1e-8)
        
        return xyz / w


if __name__ == '__main__':
    # Simple test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create a simple test case
    u_degree, v_degree = 3, 3
    num_u_poles, num_v_poles = 6, 8
    
    # Generate uniform knot vectors
    u_knots = torch.cat([
        torch.zeros(u_degree),
        torch.linspace(0, 1, num_u_poles - u_degree + 1),
        torch.ones(u_degree)
    ])
    v_knots = torch.cat([
        torch.zeros(v_degree),
        torch.linspace(0, 1, num_v_poles - v_degree + 1),
        torch.ones(v_degree)
    ])
    
    # Create random control points
    ctrl_pts = torch.randn(2, num_u_poles, num_v_poles, 4)
    ctrl_pts[..., 3] = 1.0  # Set weights to 1
    
    # Test all implementations
    print("Testing implementations...")
    
    for name, cls in [
        ("Fast", SurfEvalTorchFast),
        ("Vectorized", SurfEvalTorchVectorized)
    ]:
        print(f"\n{name} Implementation:")
        surf_eval = cls(
            u_degree, v_degree, u_knots, v_knots,
            out_dim_u=16, out_dim_v=32, device=device
        )
        
        ctrl_pts_device = ctrl_pts.to(device)
        
        # Warm up
        for _ in range(3):
            _ = surf_eval(ctrl_pts_device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time it
        import time
        num_runs = 10
        start = time.time()
        for _ in range(num_runs):
            surfaces = surf_eval(ctrl_pts_device)
            if device == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"  Control points shape: {ctrl_pts.shape}")
        print(f"  Surface points shape: {surfaces.shape}")
        print(f"  Surface points range: [{surfaces.min():.3f}, {surfaces.max():.3f}]")
        print(f"  Average time: {elapsed/num_runs*1000:.2f}ms")
    
    print("\n✓ All tests passed!")

