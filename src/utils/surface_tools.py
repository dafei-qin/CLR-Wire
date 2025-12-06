import torch

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


SURFACE_TYPE_MAP = {'plane': 0, 'cylinder': 1, 'cone': 2, 'sphere': 3, 'torus': 4}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}


def safe_atan2(y, x, eps=1e-6):
    denom = x**2 + y**2
    scale = torch.sqrt(denom + eps)
    return torch.atan2(y / scale, x / scale)

def safe_asin(x, eps=1e-6):
    return torch.asin(x.clamp(-1 + eps, 1 - eps))

def recover_surface_from_params(params, surface_type_idx):
    """Recover surface parameters from parameter vector (torch, differentiable)"""
    surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx.item() if isinstance(surface_type_idx, torch.Tensor) else surface_type_idx, 'plane')
    
    P = params[..., :3]
    D = params[..., 3:6] / (torch.norm(params[..., 3:6], dim=-1, keepdim=True) + 1e-6)
    X = params[..., 6:9] / (torch.norm(params[..., 6:9], dim=-1, keepdim=True) + 1e-6)
    Y = torch.cross(D, X, dim=-1)
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

def params_to_samples(params, surface_type_idx, num_samples_u, num_samples_v):
    """Sample points from surface parameters"""
    assert torch.isfinite(params).all(), "params contains inf/nan" + str(params)
    surface_params = recover_surface_from_params(params, surface_type_idx)
    
    location = surface_params['location']
    D, X, Y = surface_params['direction'][..., 0, :], surface_params['direction'][..., 1, :], surface_params['direction'][..., 2, :]
    u_min, u_max, v_min, v_max = surface_params['uv'][..., 0], surface_params['uv'][..., 1], surface_params['uv'][..., 2], surface_params['uv'][..., 3]
    scalar = surface_params['scalar']
    # Note, perform exp on all scalars to follow the pre-process function in dataset_v1 L36: SURFACE_PARAM_SCHEMAS
    # scalar = torch.exp(scalar)
    surface_type = surface_params['type']
    
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

