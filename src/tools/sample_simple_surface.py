import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from typing import Tuple, Union

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

# Import surface type mappings and dataset class from dataset_v1
from src.dataset.dataset_v1 import (
    SURFACE_TYPE_MAP,
    SURFACE_TYPE_MAP_INV,
    SURFACE_PARAM_SCHEMAS,
    build_surface_postpreprocess,
    dataset_compound,
)

# Import surface building functions from utils
from myutils.surface import build_plane_face, build_second_order_surface


def _to_numpy(array) -> np.ndarray:
    """Convert torch tensor or other array-like to numpy array."""
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


# Create a singleton instance of dataset_compound for accessing _recover_surface
_dataset_helper = None


def _get_dataset_helper():
    """Get or create a singleton dataset helper instance for _recover_surface."""
    global _dataset_helper
    if _dataset_helper is None:
        # Create a minimal dataset instance without loading any data
        # We only need access to the _recover_surface method
        # Create instance without calling __init__ to avoid needing json_dir
        _dataset_helper = object.__new__(dataset_compound)
        
        # Initialize only the postprocess_funcs needed by _recover_surface
        _dataset_helper.postprocess_funcs = {
            k: build_surface_postpreprocess(v) 
            for k, v in SURFACE_PARAM_SCHEMAS.items()
        }
    return _dataset_helper


def recover_surface_dict(params: np.ndarray, surface_type_idx: int) -> dict:
    """
    Recover surface dictionary from parameter vector using dataset_v1's _recover_surface.
    
    This directly calls the _recover_surface method from dataset_compound to ensure
    consistency with the dataset implementation.
    
    Args:
        params: Parameter vector (after preprocessing with log, etc.)
        surface_type_idx: Integer index of surface type
        
    Returns:
        Dictionary with keys: 'type', 'location', 'direction', 'uv', 'scalar'
    """
    # Use the dataset's _recover_surface method directly
    helper = _get_dataset_helper()
    return helper._recover_surface(params, surface_type_idx)


def build_occ_surface(surface_dict: dict, tol: float = 1e-2):
    """
    Build OCC surface object from surface dictionary using utils/surface.py functions.
    
    Args:
        surface_dict: Dictionary with 'type', 'location', 'direction', 'uv', 'scalar'
        tol: Tolerance for surface construction
        
    Returns:
        OCC Geom_Surface object and (u_min, u_max, v_min, v_max)
    """
    surface_type = surface_dict['type']
    u_min, u_max, v_min, v_max = surface_dict['uv']
    
    # Add orientation if not present (default to Forward)
    if 'orientation' not in surface_dict:
        surface_dict = surface_dict.copy()
        surface_dict['orientation'] = 'Forward'
    
    # Build TopoDS_Face using utils/surface.py functions (without meshify)
    if surface_type == 'plane':
        face_shape, _, _, _ = build_plane_face(surface_dict, tol=tol, meshify=False)
    elif surface_type in ['cylinder', 'cone', 'sphere', 'torus']:
        face_shape, _, _, _ = build_second_order_surface(surface_dict, tol=tol, meshify=False)
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")
    
    # Extract Geom_Surface from TopoDS_Face
    location = TopLoc_Location()
    occ_surface = BRep_Tool.Surface(face_shape, location)
    
    return occ_surface, (u_min, u_max, v_min, v_max)


def sample_surface_uniform(
    params: Union[np.ndarray, torch.Tensor],
    surface_type_idx: int,
    num_u: int = 32,
    num_v: int = 32,
    flatten: bool = True,
) -> np.ndarray:
    """
    Sample a simple surface uniformly in UV parameter space.
    
    Args:
        params: Parameter vector from dataset_v1 (shape: [param_dim])
        surface_type_idx: Integer index of surface type (0-4 for plane/cylinder/cone/sphere/torus)
        num_u: Number of samples in u direction
        num_v: Number of samples in v direction
        flatten: If True, return (N, 3) array; otherwise (num_v, num_u, 3)
        
    Returns:
        Sampled points as numpy array of shape (num_v*num_u, 3) or (num_v, num_u, 3)
    """
    # Convert to numpy if needed
    params = _to_numpy(params)
    
    # Recover surface dictionary
    surface_dict = recover_surface_dict(params, surface_type_idx)
    
    # Build OCC surface
    occ_surface, (u_min, u_max, v_min, v_max) = build_occ_surface(surface_dict)
    
    # Generate uniform UV grid
    u_params = np.linspace(u_min, u_max, num_u, dtype=np.float64)
    v_params = np.linspace(v_min, v_max, num_v, dtype=np.float64)
    
    # Evaluate surface at each UV point
    grid = np.zeros((num_v, num_u, 3), dtype=np.float64)
    for vi, v_val in enumerate(v_params):
        for ui, u_val in enumerate(u_params):
            point = occ_surface.Value(u_val, v_val)
            grid[vi, ui] = [point.X(), point.Y(), point.Z()]
    
    if flatten:
        return grid.reshape(-1, 3)
    return grid


def sample_surface_flexible(
    params: Union[np.ndarray, torch.Tensor],
    surface_type_idx: int,
    num_points: int = 1024,
    sampling_strategy: str = 'stratified',
    flatten: bool = True,
    random_seed: int = None,
) -> tuple:
    """
    Sample a simple surface with flexible sampling strategies.
    
    Args:
        params: Parameter vector from dataset_v1 (shape: [param_dim])
        surface_type_idx: Integer index of surface type (0-4 for plane/cylinder/cone/sphere/torus)
        num_points: Target number of points to sample
        sampling_strategy: Sampling strategy
            - 'uniform': Regular grid sampling (like sample_surface_uniform)
            - 'random': Pure random sampling in UV space
            - 'stratified': Stratified random sampling (grid with jitter)
        flatten: If True, return flattened arrays
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (points, uv_coords):
            - points: (N, 3) or (num_v, num_u, 3) sampled points
            - uv_coords: (N, 2) or (num_v, num_u, 2) UV coordinates
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert to numpy if needed
    params = _to_numpy(params)
    
    # Recover surface dictionary
    surface_dict = recover_surface_dict(params, surface_type_idx)
    
    # Build OCC surface
    occ_surface, (u_min, u_max, v_min, v_max) = build_occ_surface(surface_dict)
    
    # Compute grid size for uniform and stratified sampling
    if sampling_strategy in ['uniform', 'stratified']:
        sqrt_points = int(np.sqrt(num_points))
        num_u = max(2, sqrt_points)
        num_v = max(2, int(np.ceil(num_points / num_u)))
        actual_points = num_u * num_v
    else:
        actual_points = num_points
    
    # Generate UV coordinates based on strategy
    if sampling_strategy == 'uniform':
        # Regular grid sampling
        u_params = np.linspace(u_min, u_max, num_u, dtype=np.float64)
        v_params = np.linspace(v_min, v_max, num_v, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(u_params, v_params, indexing='xy')
        uv_coords = np.stack([u_grid, v_grid], axis=-1)  # (num_v, num_u, 2)
        
    elif sampling_strategy == 'random':
        # Pure random sampling
        u_samples = np.random.uniform(u_min, u_max, num_points)
        v_samples = np.random.uniform(v_min, v_max, num_points)
        uv_coords = np.stack([u_samples, v_samples], axis=-1)  # (N, 2)
        
    elif sampling_strategy == 'stratified':
        # Stratified random sampling (grid with jitter)
        # Divide UV space into grid cells
        u_edges = np.linspace(u_min, u_max, num_u + 1, dtype=np.float64)
        v_edges = np.linspace(v_min, v_max, num_v + 1, dtype=np.float64)
        
        # Sample randomly within each cell
        u_samples = np.zeros((num_v, num_u), dtype=np.float64)
        v_samples = np.zeros((num_v, num_u), dtype=np.float64)
        
        for i in range(num_v):
            for j in range(num_u):
                # Random sample within cell [u_edges[j], u_edges[j+1]] x [v_edges[i], v_edges[i+1]]
                u_samples[i, j] = np.random.uniform(u_edges[j], u_edges[j+1])
                v_samples[i, j] = np.random.uniform(v_edges[i], v_edges[i+1])
        
        uv_coords = np.stack([u_samples, v_samples], axis=-1)  # (num_v, num_u, 2)
    
    else:
        raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}. "
                        f"Choose from 'uniform', 'random', 'stratified'.")
    
    # Evaluate surface at UV coordinates
    if sampling_strategy in ['uniform', 'stratified']:
        # Grid-based sampling
        points = np.zeros((num_v, num_u, 3), dtype=np.float64)
        for i in range(num_v):
            for j in range(num_u):
                u_val, v_val = uv_coords[i, j]
                point = occ_surface.Value(float(u_val), float(v_val))
                points[i, j] = [point.X(), point.Y(), point.Z()]
        
        if flatten:
            points = points.reshape(-1, 3)
            uv_coords = uv_coords.reshape(-1, 2)
    else:
        # Random sampling
        points = np.zeros((num_points, 3), dtype=np.float64)
        for i in range(num_points):
            u_val, v_val = uv_coords[i]
            point = occ_surface.Value(float(u_val), float(v_val))
            points[i] = [point.X(), point.Y(), point.Z()]
    
    return points, uv_coords


def sample_surface_batch(
    params_batch: Union[np.ndarray, torch.Tensor],
    surface_types_batch: Union[np.ndarray, torch.Tensor],
    mask_batch: Union[np.ndarray, torch.Tensor],
    num_u: int = 32,
    num_v: int = 32,
    flatten: bool = True,
) -> np.ndarray:
    """
    Sample multiple surfaces in batch mode.
    
    Args:
        params_batch: Parameter vectors of shape (batch_size, param_dim)
        surface_types_batch: Surface type indices of shape (batch_size,)
        mask_batch: Valid surface mask of shape (batch_size,)
        num_u: Number of samples in u direction
        num_v: Number of samples in v direction
        flatten: If True, return (batch_size, N, 3); otherwise (batch_size, num_v, num_u, 3)
        
    Returns:
        Sampled points array
    """
    params_batch = _to_numpy(params_batch)
    surface_types_batch = _to_numpy(surface_types_batch).astype(np.int32)
    mask_batch = _to_numpy(mask_batch)
    
    batch_size = params_batch.shape[0]
    output_shape = (batch_size, num_v * num_u, 3) if flatten else (batch_size, num_v, num_u, 3)
    results = np.zeros(output_shape, dtype=np.float64)
    
    for i in range(batch_size):
        if mask_batch[i] > 0.5:  # Valid surface
            try:
                sampled = sample_surface_uniform(
                    params_batch[i],
                    surface_types_batch[i],
                    num_u=num_u,
                    num_v=num_v,
                    flatten=flatten,
                )
                results[i] = sampled
            except Exception as e:
                print(f"Warning: Failed to sample surface {i}: {e}")
                # Leave as zeros
    
    return results


def main():
    """Demo: visualize sampled surfaces using polyscope."""
    parser = argparse.ArgumentParser(description="Sample simple surfaces from dataset_v1 parameters")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to JSON directory")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--num_u", type=int, default=32, help="Number of u samples")
    parser.add_argument("--num_v", type=int, default=32, help="Number of v samples")
    args = parser.parse_args()
    
    # Load dataset
    from src.dataset.dataset_v1 import dataset_compound
    dataset = dataset_compound(args.json_dir, max_num_surfaces=500)
    
    print(f"Loading sample {args.index} from dataset...")
    params_batch, types_batch, mask_batch, _, _, _ = dataset[args.index]
    
    # Visualize with polyscope
    import polyscope as ps
    ps.init()
    ps.set_ground_plane_mode("tile")
    
    num_surfaces = int(mask_batch.sum().item())
    print(f"Found {num_surfaces} valid surfaces")
    
    for surf_idx in range(params_batch.shape[0]):
        if mask_batch[surf_idx] < 0.5:
            continue
        
        params = params_batch[surf_idx]
        surf_type_idx = types_batch[surf_idx].item()
        surf_type_name = SURFACE_TYPE_MAP_INV[surf_type_idx]
        
        try:
            # Sample surface
            points = sample_surface_uniform(
                params,
                surf_type_idx,
                num_u=args.num_u,
                num_v=args.num_v,
                flatten=False,
            )
            points_flat = points.reshape(-1, 3)
            
            # Register point cloud
            cloud = ps.register_point_cloud(
                f"surface_{surf_idx:03d}_{surf_type_name}",
                points_flat,
                radius=0.003,
            )
            
            print(f"  [{surf_idx}] {surf_type_name}: {points_flat.shape[0]} points")
            
        except Exception as e:
            print(f"  [{surf_idx}] {surf_type_name}: Failed - {e}")
    
    ps.show()


if __name__ == "__main__":
    main()

