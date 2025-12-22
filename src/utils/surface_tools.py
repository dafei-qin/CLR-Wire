import torch

def cubic_bspline_basis(t):
    """
    Compute cubic B-spline basis functions (Bernstein polynomials) for parameter t.
    For uniform open knot vector [0,0,0,0,1,1,1,1], this gives standard cubic Bezier basis.
    
    Args:
        t: (N,) Parameter values in [0, 1]
    
    Returns:
        basis: (N, 4) Basis function values, each row sums to 1
    """
    # Precompute powers for efficiency
    t2 = t * t
    t3 = t2 * t
    one_minus_t = 1.0 - t
    one_minus_t2 = one_minus_t * one_minus_t
    one_minus_t3 = one_minus_t2 * one_minus_t
    
    # Bernstein basis functions (cubic)
    b0 = one_minus_t3                          # (1-t)^3
    b1 = 3.0 * t * one_minus_t2                # 3t(1-t)^2
    b2 = 3.0 * t2 * one_minus_t                # 3t^2(1-t)
    b3 = t3                                     # t^3
    
    return torch.stack([b0, b1, b2, b3], dim=-1)

def sample_bspline_surface(control_points, num_u, num_v):
    """
    Sample a cubic B-spline surface with 4x4 control points.
    Uses tensor product of cubic Bernstein basis functions.
    
    Args:
        control_points: (..., 4, 4, 3) Control points in any shape
                       Last 3 dims must be (4, 4, 3) for 4x4 grid with xyz coords
        num_u: Number of samples in u direction
        num_v: Number of samples in v direction
    
    Returns:
        points: (..., num_u, num_v, 3) Sampled surface points
    """
    device = control_points.device
    dtype = control_points.dtype
    
    # Generate parameter values in [0, 1]
    u = torch.linspace(0, 1, num_u, device=device, dtype=dtype)
    v = torch.linspace(0, 1, num_v, device=device, dtype=dtype)
    
    # Compute basis functions: (num_u, 4) and (num_v, 4)
    Bu = cubic_bspline_basis(u)  # (num_u, 4)
    Bv = cubic_bspline_basis(v)  # (num_v, 4)
    
    # Tensor product evaluation: points = Bu @ control_points @ Bv^T
    # This computes: sum_i sum_j Bu[u,i] * Bv[v,j] * control_points[i,j,:]
    # 
    # Using einsum for efficient batch computation:
    # 'ui' : (num_u, 4) - u basis
    # '...ijk' : (..., 4, 4, 3) - control points (batch, i_poles, j_poles, xyz)
    # 'vj' : (num_v, 4) - v basis
    # Output: (..., num_u, num_v, 3)
    points = torch.einsum('ui,...ijk,vj->...uvk', Bu, control_points, Bv)
    
    return points

def plane_d0(u, v, location, X, Y, Z):
    """Plane: P(u,v) = Location + u*X + v*Y"""
    return location + u[..., None] * X + v[..., None] * Y

def cylinder_d0(u, v, location, X, Y, Z, radius):
    """Cylinder: P(u,v) = Location + R*(cos(u)*X + sin(u)*Y) + v*Z"""
    r_cos_u = radius * torch.cos(u)
    r_sin_u = radius * torch.sin(u)
    return location + r_cos_u[..., None] * X + r_sin_u[..., None] * Y + v[..., None] * Z

def cone_d0(u, v, location, X, Y, Z, radius, semi_angle):
    """Cone: P(u,v) = Location + (R+v*sin(α))*(cos(u)*X + sin(u)*Y) + v*cos(α)*Z"""
    R = radius + v * torch.sin(semi_angle)
    cos_u, sin_u = torch.cos(u), torch.sin(u)
    return location + (R * cos_u)[..., None] * X + (R * sin_u)[..., None] * Y + (v * torch.cos(semi_angle))[..., None] * Z

def sphere_d0(u, v, location, X, Y, Z, radius):
    """Sphere: P(u,v) = Location + R*cos(v)*(cos(u)*X + sin(u)*Y) + R*sin(v)*Z"""
    R_cos_v = radius * torch.cos(v)
    R_sin_v = radius * torch.sin(v)
    cos_u, sin_u = torch.cos(u), torch.sin(u)
    return location + (R_cos_v * cos_u)[..., None] * X + (R_cos_v * sin_u)[..., None] * Y + R_sin_v[..., None] * Z

def torus_d0(u, v, location, X, Y, Z, major_radius, minor_radius):
    """Torus: P(u,v) = Location + (MajR+MinR*cos(v))*(cos(u)*X + sin(u)*Y) + MinR*sin(v)*Z"""
    R = major_radius + minor_radius * torch.cos(v)
    cos_u, sin_u = torch.cos(u), torch.sin(u)
    return location + (R * cos_u)[..., None] * X + (R * sin_u)[..., None] * Y + (minor_radius * torch.sin(v))[..., None] * Z


SURFACE_TYPE_MAP = {'plane': 0, 'cylinder': 1, 'cone': 2, 'sphere': 3, 'torus': 4, 'bspline_surface': 5}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}


def safe_atan2(y, x, eps=1e-6):
    denom = x**2 + y**2
    scale = torch.sqrt(denom + eps)
    return torch.atan2(y / scale, x / scale)

def safe_asin(x, eps=1e-6):
    return torch.asin(x.clamp(-1 + eps, 1 - eps))

def safe_normalize(v, dim=-1, eps=1e-6):
    norm = torch.norm(v, dim=dim, keepdim=True)
    norm = torch.maximum(norm, torch.ones_like(norm) * eps)
    return v / norm

def recover_surface_from_params(params, surface_type_idx):
    """Recover surface parameters from parameter vector (torch, differentiable)"""
    surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx.item() if isinstance(surface_type_idx, torch.Tensor) else surface_type_idx, 'plane')
    
    # Special handling for bspline_surface
    if surface_type == 'bspline_surface':
        # For bspline, params[17:65] are control points (48 dims)
        # Return them flattened as 'scalar' for consistency
        
        # For bspline, we return control points as 'scalar'
        control_points = params[..., 17:65]  # (batch, 48)
        
        # Dummy UV (not meaningful for bspline in canonical space)
        
        return {
            'location': [],
            'direction': [],
            'uv': torch.stack([0, 1, 0, 1], dim=-1),
            'scalar': control_points.numpy().tolist(),  # 48D control points
            'type': surface_type,
        }
    
    P = params[..., :3]
    D = safe_normalize(params[..., 3:6])
    X = safe_normalize(params[..., 6:9])
    Y = torch.cross(D, X, dim=-1)
    Y = safe_normalize(Y)
    UV = params[..., 9:17]
    scalar_params = params[..., 17:]
    
    if surface_type == 'plane':
        u_min, u_max, v_min, v_max = UV[..., 0], UV[..., 1], UV[..., 2], UV[..., 3]
        scalar = torch.empty(*params.shape[:-1], 0, device=params.device)

    elif surface_type == 'cylinder':
        sin_u_center, cos_u_center, u_half, height = UV[..., 0], UV[..., 1], UV[..., 2], UV[..., 3]
        # u_center = torch.atan2(sin_u_center, cos_u_center)
        u_center = safe_atan2(sin_u_center, cos_u_center)
        u_half = torch.clamp(u_half + 0.5, 0, 1 - 1e-5) * torch.pi
        u_min, u_max = u_center - u_half, u_center + u_half
        v_min, v_max = torch.zeros_like(height), height
        scalar = scalar_params[..., 0:1]
        scalar = torch.exp(scalar)

    elif surface_type == 'cone':
        sin_u_center, cos_u_center, u_half, v_center, v_half = UV[..., 0], UV[..., 1], UV[..., 2], UV[..., 3], UV[..., 4]
        # u_center = torch.atan2(sin_u_center, cos_u_center)
        u_center = safe_atan2(sin_u_center, cos_u_center)
        u_half = torch.clamp(u_half, 0, 1 - 1e-5) * torch.pi
        u_min, u_max = u_center - u_half, u_center + u_half
        v_min, v_max = v_center - v_half, v_center + v_half
        semi_angle = scalar_params[..., 0] * (torch.pi / 2)
        scalar = torch.stack([semi_angle, torch.exp(scalar_params[..., 1])], dim=-1)

    elif surface_type == 'torus':
        sin_u_center, cos_u_center, u_half, sin_v_center, cos_v_center, v_half = UV[..., 0], UV[..., 1], UV[..., 2], UV[..., 3], UV[..., 4], UV[..., 5]
        # u_center = torch.atan2(sin_u_center, cos_u_center)
        u_center = safe_atan2(sin_u_center, cos_u_center)
        u_half = torch.clamp(u_half, 0, 1 - 1e-5) * torch.pi
        u_min, u_max = u_center - u_half, u_center + u_half
        # v_center = torch.atan2(sin_v_center, cos_v_center)
        v_center = safe_atan2(sin_v_center, cos_v_center)
        v_half = torch.clamp(v_half, 0, 1 - 1e-5) * torch.pi
        v_min, v_max = v_center - v_half, v_center + v_half
        scalar = torch.exp(scalar_params[..., :2])
        
    elif surface_type == 'sphere':
        dir_vec = UV[..., :3]
        u_h_norm, v_h_norm = UV[..., 3], UV[..., 4]
        dir_vec = dir_vec / (torch.norm(dir_vec, dim=-1, keepdim=True) + 1e-8)
        x, y, z = dir_vec[..., 0], dir_vec[..., 1], dir_vec[..., 2]
        # u_center = torch.atan2(y, x)
        u_center = safe_atan2(y, x)
        # v_center = torch.asin(torch.clamp(z, -1.0, 1.0))
        v_center = safe_asin(torch.clamp(z, -1.0, 1.0))
        u_half = torch.clamp(u_h_norm, 0.0, 1.0 - 1e-5) * torch.pi
        v_half = torch.clamp(v_h_norm, 0.0, 1.0 - 1e-5) * (torch.pi / 2)
        u_min, u_max = u_center - u_half, u_center + u_half
        v_min, v_max = v_center - v_half, v_center + v_half
        scalar = torch.exp(scalar_params[..., 0:1])
        assert torch.isfinite(UV).all(), "UV contains inf/nan" + str(UV)
        assert torch.isfinite(u_min).all(), "u_min contains inf/nan" + str(u_min)
        assert torch.isfinite(u_max).all(), "u_max contains inf/nan" + str(u_max)
        assert torch.isfinite(v_min).all(), "v_min contains inf/nan" + str(v_min)
        assert torch.isfinite(v_max).all(), "v_max contains inf/nan" + str(v_max)
        assert torch.isfinite(scalar).all(), "scalar contains inf/nan" + str(scalar)
    
    return {
        'location': P,
        'direction': torch.stack([D, X, Y], dim=-2),
        'uv': torch.stack([u_min, u_max, v_min, v_max], dim=-1),
        'scalar': scalar,
        'type': surface_type,
    }

def params_to_samples_with_rts(rotations, scales, shifts, params, surface_type_idx,  num_samples_u, num_samples_v):
    points = batch_params_to_samples(params, surface_type_idx, num_samples_u, num_samples_v)
    if points.shape[0] == 1:
        rotations = rotations.unsqueeze(0)
        scales = scales.unsqueeze(0)
        shifts = shifts.unsqueeze(0)
    
    points = (rotations.transpose(-1, -2)[:, None, None] @ points[..., None])[..., 0] * scales[:, None, None, None] + shifts[:, None, None]
    return points

def params_to_samples(params, surface_type_idx, num_samples_u, num_samples_v):
    """Sample points from surface parameters"""
    assert torch.isfinite(params).all(), "params contains inf/nan" + str(params)
    
    # Get surface type string
    surface_type_idx_scalar = surface_type_idx.item() if isinstance(surface_type_idx, torch.Tensor) else surface_type_idx
    surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx_scalar, 'plane')
    
    # Special handling for bspline_surface
    if surface_type == 'bspline_surface':
        # Extract 4x4x3 control points from params[17:65] (48 dimensions)
        # params structure for bspline: [P(3), D(3), X(3), UV(8), control_points(48)]
        control_points_flat = params[..., 17:65]  # (..., 48)
        
        # Reshape to 4x4x3 grid
        # Handle both single sample (..., 48) and batch (..., N, 48) cases
        original_shape = control_points_flat.shape[:-1]  # Get all dims except last
        control_points = control_points_flat.reshape(*original_shape, 4, 4, 3)
        
        # Sample the B-spline surface
        # Output: (..., num_samples_u, num_samples_v, 3)
        points = sample_bspline_surface(control_points, num_samples_u, num_samples_v)
        
        assert torch.isfinite(points).all(), "bspline points contains inf/nan"
        return points
    
    # Standard parametric surface handling
    surface_params = recover_surface_from_params(params, surface_type_idx)
    
    location = surface_params['location']
    D, X, Y = surface_params['direction'][..., 0, :], surface_params['direction'][..., 1, :], surface_params['direction'][..., 2, :]
    u_min, u_max, v_min, v_max = surface_params['uv'][..., 0], surface_params['uv'][..., 1], surface_params['uv'][..., 2], surface_params['uv'][..., 3]
    scalar = surface_params['scalar']
    # Note, perform exp on all scalars to follow the pre-process function in dataset_v1 L36: SURFACE_PARAM_SCHEMAS
    # scalar = torch.exp(scalar)
    
    assert torch.isfinite(surface_params['uv']).all(), "uv contains inf/nan" + str(surface_params['uv'])
    device = params.device
    u_lin = torch.linspace(0, 1, num_samples_u, device=device)
    v_lin = torch.linspace(0, 1, num_samples_v, device=device)
    u_grid, v_grid = torch.meshgrid(u_lin, v_lin, indexing='ij')
    
    u = u_min[..., None, None] + u_grid * (u_max - u_min)[..., None, None]
    v = v_min[..., None, None] + v_grid * (v_max - v_min)[..., None, None]
    
    loc_exp = location[..., None, None, :]
    X_exp = X[..., None, None, :]
    Y_exp = Y[..., None, None, :]
    D_exp = D[..., None, None, :]

    assert torch.isfinite(u).all(), "u contains inf/nan"
    assert torch.isfinite(v).all(), "v contains inf/nan"
    assert torch.isfinite(scalar).all(), "scalar contains inf/nan"
    # assert torch.isfinite(radius).all(), "radius contains inf/nan"
    assert torch.isfinite(loc_exp).all(), "loc_exp contains inf/nan"
    assert torch.isfinite(X_exp).all(), "X_exp contains inf/nan"
    assert torch.isfinite(Y_exp).all(), "Y_exp contains inf/nan"
    assert torch.isfinite(D_exp).all(), "D_exp contains inf/nan"
    
    if surface_type == 'plane':
        points = plane_d0(u, v, loc_exp, X_exp, Y_exp, D_exp)
    elif surface_type == 'cylinder':
        radius = scalar[..., 0:1, None]
        points = cylinder_d0(u, v, loc_exp, X_exp, Y_exp, D_exp, radius)
    elif surface_type == 'cone':
        semi_angle, radius = scalar[..., 0:1, None], scalar[..., 1:2, None]
        points = cone_d0(u, v, loc_exp, X_exp, Y_exp, D_exp, radius, semi_angle)
    elif surface_type == 'sphere':
        radius = scalar[..., 0:1, None]
        points = sphere_d0(u, v, loc_exp, X_exp, Y_exp, D_exp, radius)
    elif surface_type == 'torus':
        major_radius, minor_radius = scalar[..., 0:1, None], scalar[..., 1:2, None]
        points = torus_d0(u, v, loc_exp, X_exp, Y_exp, D_exp, major_radius, minor_radius)
    assert not torch.isnan(points).any(), "points contains inf/nan, surface_type: " + surface_type + str(surface_params)
    return points


def batch_params_to_samples(params_batch, surface_type_batch, num_samples_u, num_samples_v):
    """
    Batch process surfaces with different types while preserving order.
    Similar to decode_and_sample, processes each surface type separately and
    maintains gradient flow.
    
    Args:
        params_batch: (B, param_dim) Batched surface parameters
        surface_type_batch: (B,) Batched surface type indices
        num_samples_u: Number of samples in u direction
        num_samples_v: Number of samples in v direction
    
    Returns:
        ordered_samples: (B, num_samples_u, num_samples_v, 3) Sampled points in same order as input
    """
    assert torch.isfinite(params_batch).all(), "params_batch contains inf/nan"
    assert params_batch.ndim == 2, f"Expected 2D params_batch, got shape {params_batch.shape}"
    assert surface_type_batch.ndim == 1, f"Expected 1D surface_type_batch, got shape {surface_type_batch.shape}"
    assert params_batch.shape[0] == surface_type_batch.shape[0], \
        f"Batch size mismatch: params {params_batch.shape[0]} vs types {surface_type_batch.shape[0]}"
    
    batch_size = params_batch.shape[0]
    device = params_batch.device
    dtype = params_batch.dtype
    
    # Initialize output tensor (will be filled in per-type)
    ordered_samples = None
    
    # Process each unique surface type separately
    for surface_type in surface_type_batch.unique():
        # Find all surfaces of this type
        type_mask = surface_type_batch == surface_type
        params_per_type = params_batch[type_mask]
        
        # Sample this type (all samples of same type processed together)
        samples = params_to_samples(params_per_type, surface_type, num_samples_u, num_samples_v)
        
        # Initialize output tensor on first iteration
        if ordered_samples is None:
            sample_shape = samples.shape[1:]  # (num_samples_u, num_samples_v, 3)
            ordered_samples = torch.zeros((batch_size,) + sample_shape, device=device, dtype=dtype)
        
        # Place samples back in original order
        ordered_samples[type_mask] = samples
    
    # Handle empty batch case
    if ordered_samples is None:
        return torch.empty((0, num_samples_u, num_samples_v, 3), device=device, dtype=dtype)
    
    assert torch.isfinite(ordered_samples).all(), "ordered_samples contains inf/nan"
    return ordered_samples

