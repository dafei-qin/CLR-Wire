import torch
from einops import repeat, rearrange
from torchtyping import TensorType

def fmt(v):
    if torch.is_tensor(v) and v.dim() == 0:
        v = v.item()
    if isinstance(v, (float, int)):
        return f"{v:.3f}"
    else:
        return str(v)


def interpolate_1d(
    t: TensorType["bs", "n"],
    data: TensorType["bs", "c", "n"],
):
    """
    Perform 1D linear interpolation on the given data.

    Args:
    t (Tensor): Interpolation coordinates, with values in the range [0, 1], 
                of shape (batch_size, n).
    data (Tensor): Original data to be interpolated, 
                   of shape (batch_size, channels, num_points).

    Returns:
    Tensor: Interpolated data of shape (batch_size, channels, n).
    """
    # Check if input tensors have the expected dimensions
    assert t.dim() == 2, "t should be a 2D tensor with shape (batch_size, n)"
    assert data.dim() == 3, "data should be a 3D tensor with shape (batch_size, channels, num_points)"
    assert (0 <= t).all() and (t <= 1).all(), "t must be within [0, 1]"

    # Map interpolation coordinates from [0, 1] to [0, num_reso - 1]
    num_reso = data.shape[-1]
    t = t * (num_reso - 1)

    left = torch.floor(t).long()
    right = torch.ceil(t).long()
    alpha = t - left

    left = torch.clamp(left, max=num_reso - 1)
    right = torch.clamp(right, max=num_reso - 1)

    c = data.shape[-2]

    left = repeat(left, 'bs n -> bs c n', c=c)
    left_values = torch.gather(data, -1, left)

    right = repeat(right, 'bs n -> bs c n', c=c)
    right_values = torch.gather(data, -1, right)

    alpha = repeat(alpha, 'bs n -> bs c n', c=c)

    interpolated = (1 - alpha) * left_values + alpha * right_values
    
    return interpolated


def calculate_polyline_lengths(points: TensorType['b', 'n', 'c', float]) -> TensorType['b', float]:
    """
    Calculate the lengths of a batch of polylines.

    Args:
    points (torch.Tensor): Tensor of shape (batch_size, num_points, c),
                            where batch_size is the number of polylines,
                            num_points is the number of points per polyline,
                            and c corresponds to the 2D/3D coordinates of each point.

    Returns:
    torch.Tensor: Tensor of shape (batch_size,) representing the total length of each polyline.
    """

    if points.dim() != 3:
        raise ValueError("Input tensor must have shape (batch_size, num_points, c)")

    diffs = points[:, 1:, :] - points[:, :-1, :]
    distances = torch.norm(diffs, dim=2)
    polyline_lengths = distances.sum(dim=1)

    return polyline_lengths

def sample_edge_points(batch_edge_points, num_points=32):
    # example: (batch_size, 256, 3) -> (batch_size, 32, 3)

    t = torch.linspace(0, 1, num_points).to(batch_edge_points.device)
    bs = batch_edge_points.shape[0]
    t = repeat(t, 'n -> b n', b=bs)
    
    batch_edge_points = rearrange(batch_edge_points, 'b n c -> b c n')
    batch_edge_points = interpolate_1d(t, batch_edge_points)
    batch_edge_points = rearrange(batch_edge_points, 'b c n -> b n c')
    
    return batch_edge_points

def interpolate_2d(
    t: TensorType["bs", "h", "w", 2],
    data: TensorType["bs", "c", "h", "w"],
):
    """
    Perform 2D bilinear interpolation on the given data.

    Args:
    t (Tensor): Interpolation coordinates, with values in the range [0, 1], 
                of shape (batch_size, height, width, 2).
    data (Tensor): Original data to be interpolated, 
                   of shape (batch_size, channels, height, width).

    Returns:
    Tensor: Interpolated data of shape (batch_size, channels, height, width).
    """
    # Check if input tensors have the expected dimensions
    assert t.dim() == 4, "t should be a 4D tensor with shape (batch_size, height, width, 2)"
    assert data.dim() == 4, "data should be a 4D tensor with shape (batch_size, channels, height, width)"
    assert (0 <= t).all() and (t <= 1).all(), "t must be within [0, 1]"

    # Map interpolation coordinates from [0, 1] to [0, num_reso - 1]
    h_reso, w_reso = data.shape[-2:]
    t = t * torch.tensor([h_reso - 1, w_reso - 1], device=t.device)

    # Get the four corners of the interpolation cell
    left = torch.floor(t[..., 0]).long()
    right = torch.ceil(t[..., 0]).long()
    top = torch.floor(t[..., 1]).long()
    bottom = torch.ceil(t[..., 1]).long()

    # Clamp indices to valid range
    left = torch.clamp(left, max=h_reso - 1)
    right = torch.clamp(right, max=h_reso - 1)
    top = torch.clamp(top, max=w_reso - 1)
    bottom = torch.clamp(bottom, max=w_reso - 1)

    # Calculate interpolation weights
    alpha = t[..., 0] - left.float()
    beta = t[..., 1] - top.float()

    # Get values at the four corners
    left_top = data[..., left, top]
    right_top = data[..., right, top]
    left_bottom = data[..., left, bottom]
    right_bottom = data[..., right, bottom]

    # Perform bilinear interpolation
    interpolated = (
        (1 - alpha) * (1 - beta) * left_top +
        alpha * (1 - beta) * right_top +
        (1 - alpha) * beta * left_bottom +
        alpha * beta * right_bottom
    )

    return interpolated

def calculate_surface_area(points: TensorType['b', 'h', 'w', 'c', float]) -> TensorType['b', float]:
    """
    Calculate the surface area of a batch of surfaces using triangulation.

    Args:
    points (torch.Tensor): Tensor of shape (batch_size, height, width, c),
                          where batch_size is the number of surfaces,
                          height and width are the grid dimensions,
                          and c corresponds to the 3D coordinates of each point.

    Returns:
    torch.Tensor: Tensor of shape (batch_size,) representing the total area of each surface.
    """
    if points.dim() != 4:
        raise ValueError("Input tensor must have shape (batch_size, height, width, c)")

    # Get the points for each triangle
    p1 = points[:, :-1, :-1, :]  # top-left
    p2 = points[:, :-1, 1:, :]   # top-right
    p3 = points[:, 1:, :-1, :]   # bottom-left
    p4 = points[:, 1:, 1:, :]    # bottom-right

    # Calculate vectors for each triangle
    v1 = p2 - p1
    v2 = p3 - p1
    v3 = p4 - p2
    v4 = p4 - p3

    # Calculate cross products for each triangle
    cross1 = torch.cross(v1, v2, dim=-1)
    cross2 = torch.cross(v3, v4, dim=-1)

    # Calculate areas of each triangle
    area1 = torch.norm(cross1, dim=-1) / 2
    area2 = torch.norm(cross2, dim=-1) / 2

    # Sum up all triangle areas
    total_area = (area1 + area2).sum(dim=[1, 2])

    return total_area

def sample_surface_points(batch_surface_points, num_points=16):
    """
    Sample points from a surface using uniform grid sampling.

    Args:
    batch_surface_points (torch.Tensor): Original surface points of shape (batch_size, height, width, 3)
    num_points (int): Number of points to sample in each dimension

    Returns:
    torch.Tensor: Sampled points of shape (batch_size, num_points, num_points, 3)
    """
    # Generate uniform sampling coordinates
    t = torch.linspace(0, 1, num_points).to(batch_surface_points.device)
    bs = batch_surface_points.shape[0]
    
    # Create 2D grid of sampling coordinates
    t = torch.stack(torch.meshgrid(t, t, indexing='ij'), dim=-1)
    t = repeat(t, 'h w c -> b h w c', b=bs)
    
    # Rearrange input for interpolation
    batch_surface_points = rearrange(batch_surface_points, 'b h w c -> b c h w')
    
    # Perform 2D interpolation
    batch_surface_points = interpolate_2d(t, batch_surface_points)
    
    # Rearrange back to original format
    batch_surface_points = rearrange(batch_surface_points, 'b c h w -> b h w c')
    
    return batch_surface_points