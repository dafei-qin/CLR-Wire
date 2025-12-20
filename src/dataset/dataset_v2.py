# This v2 version add bspline fitting.
# If a surface is bspline:
# 1. convert to canonical space
# 2. fit 4x4x3 poles with weight=1.
# 3. Drop if fit error bigger than threshold.
# 4. Otherwise, keep the surface and return the flattened parameters.

# ---------------------------------------
# This is the dataset for the vae_v1.py

# It is a dataset of surface parameters, with the surface type as the label.

# The surface parameters are padded to the max raw dimension for each surface type.

# The surface type is a one-hot encoded vector.



import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import json
import warnings
from einops import rearrange
from pathlib import Path
from typing import Dict, List, Tuple
from icecream import ic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.tools.surface_to_canonical_space import to_canonical, from_canonical, compute_rotation_matrix

# Import for bspline surface handling
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface
from utils.surface import get_approx_face

from copy import copy

ic.disable()
def safe_exp(x):
    return np.exp(x).clip(1e-6, 1e6)



# 注册每种曲面的参数后处理函数
# 定义每种曲面的参数结构
SURFACE_PARAM_SCHEMAS = {
    "plane": {
        "fields": ["pos", "dir", "xdir", "UV"],
        "dims": [3, 3, 3, 8],
        "transforms": [None, None, None, None]
    },
    "cylinder": {
        "fields": ["pos", "dir", "xdir", "UV", "radius"],
        "dims": [3, 3, 3, 8, 1],
        "transforms": [None, None, None, None, np.log]
    },
    "cone": {
        "fields": ["pos", "dir", "xdir", "UV", "semi_angle", "radius"],
        "dims": [3, 3, 3, 8, 1, 1],
        "transforms": [None, None, None, None, None, np.log]
    },
    "torus": {
        "fields": ["pos", "dir", "xdir", "UV", "major_radius", "minor_radius"],
        "dims": [3, 3, 3, 8, 1, 1],
        "transforms": [None, None, None, None, np.log, np.log]
    },
    "sphere": {
        "fields": ["pos", "dir", "xdir", "UV", "radius"],
        "dims": [3, 3, 3, 8, 1],
        "transforms": [None, None, None, None, np.log]
    },
    "bspline_surface": {
        "fields": ["control_points"],  # 4x4x3 = 48 params
        "dims": [48],
        "transforms": [None]
    },
}


SURFACE_TYPE_MAP = {
        'plane': 0,
        'cylinder': 1,
        'cone': 2,
        'sphere': 3,
        'torus': 4,
        'bspline_surface': 5,
}

    
    # Dimension of scalar parameters for each surface type
SCALAR_DIM_MAP = {
    'plane': 0,      # no scalar parameters
    'cylinder': 1,   # [radius]
    'cone': 2,       # [semi_angle, radius]
    'sphere': 1,     # [radius]
    'torus': 2,      # [major_radius, minor_radius]
    'bspline_surface': 48,  # 4x4x3 control points (flattened)
}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}





type_weights = {
    'plane': 1.0,
    'cylinder': 2.0,
    'cone': 15.0,
    'sphere': 45.0,
    'torus': 13.0,
    'bspline_surface': 1.0,
}


def build_surface_process(schema):
    """返回用于pre-processing的函数"""
    split_indices = torch.cumsum(torch.tensor([0] + schema["dims"]), dim=0)
    transforms = schema["transforms"]

    def fn(raw):
        parts = []
        for i in range(len(transforms)):
            s, e = split_indices[i], split_indices[i+1]
            part = raw[..., s:e]
            if transforms[i] is not None:
                part = transforms[i](part)
            parts.append(part)
        return np.concatenate(parts, axis=-1)
    return fn


def build_surface_postpreprocess(schema):
    """返回用于post-processing的逆变换"""
    split_indices = torch.cumsum(torch.tensor([0] + schema["dims"]), dim=0)
    transforms = schema["transforms"]

    def fn(param):
        parts = []
        for i in range(len(transforms)):
            s, e = split_indices[i], split_indices[i+1]
            part = param[..., s:e]
            if transforms[i] is not None:
                # 找到对应的inverse
                inv = {
                    np.log: safe_exp,
                }.get(transforms[i])
                if inv is not None:
                    part = inv(part)
            parts.append(part)
        return np.concatenate(parts, axis=-1)
    return fn




class V1(Dataset):
    pass

class V1_random(Dataset):
    def __init__(self):
        super().__init__()
        self.replica = 1
        self.surface_types = 5
        self.param_raw_dim = [10, 11, 12, 12, 11]
        self.max_raw_dim = max(self.param_raw_dim)

    def __len__(self):
        return int(1e6)

    def __getitem__(self, idx):
        surface_type = torch.randint(0, self.surface_types, (1,)).long()
        # surface_type_onehot = torch.zeros(1, self.surface_types)
        # surface_type_onehot[0, surface_type] = 1
        params_raw = torch.randn(self.max_raw_dim)
        return params_raw, surface_type


import numpy as np

def normalize_cylinder_with_center(P, D, X_dir, Y_dir, u_min, u_max, v_min, v_max):
    """
    规范化参数，使u_center尽量小（接近0），同时更新方向矩阵以保持几何一致。
    """

    TWO_PI = 2 * np.pi

    # === Step 1: v平移到0 ===
    P = P + D * v_min
    v_max = v_max - v_min
    v_min = 0

    # === Step 2: 计算原始u中心与跨度 ===
    u_center = 0.5 * (u_min + u_max)
    u_diff = u_max - u_min

    # === Step 3: 使u_center尽量靠近0 ===
    # 将u_center映射到[-π, π]范围
    u_center_mod = ((u_center + np.pi) % (2 * np.pi)) - np.pi
    delta_u = -u_center_mod  # 旋转到中心≈0
    # 更新后的u_min/u_max
    u_min_new = u_min + delta_u
    u_max_new = u_max + delta_u

    # === Step 4: 同步旋转局部坐标系 ===
    X_dir = np.array(X_dir, dtype=np.float64)
    Y_dir = np.array(Y_dir, dtype=np.float64)
    D = np.array(D, dtype=np.float64)

    X_new = np.cos(delta_u) * X_dir + np.sin(delta_u) * Y_dir
    Y_new = np.cross(D, X_new)
    Y_new /= np.linalg.norm(Y_new)

    # === Step 5: 计算归一化参数 ===
    u_half = 0.5 * (u_max_new - u_min_new) / np.pi - 0.5  # (-0.5, 0.5)
    sin_u_center, cos_u_center = np.sin(0), np.cos(0)  # 因为中心已归零

    UV = np.array([sin_u_center, cos_u_center, u_half, v_max, 0, 0, 0, 0], dtype=np.float64)

    return {
        "P": P,
        "D": D,
        "X_dir": X_new,
        "Y_dir": Y_new,
        "u_range": (u_min_new, u_max_new),
        "v_range": (v_min, v_max),
        "UV": UV,
        "delta_u": delta_u
    }



def normalize_cone_with_center(P, D, X, u_min, u_max, v_min, v_max, semi_angle, radius, r_min_thresh=1e-2):
    """
    Normalize a conical surface for stable neural network learning.

    Args:
        P: (3,) np.array, apex position or cone base reference.
        D: (3,) np.array, cone axis direction (normalized).
        X: (3,) np.array, cone XDirection.
        u_min, u_max: angular range in radians.
        v_min, v_max: height range along cone axis.
        semi_angle: float, half angle of cone (in radians).
        radius: float, radius at v=0.
        r_min_thresh: minimal allowed radius at v_min (to avoid degeneracy).

    Returns:
        P_new: shifted origin at v_min.
        D, X, Y: updated local frame.
        UV: (8,) np.array, normalized parameters for neural net.
        scalar_params: [semi_angle_norm, radius_norm]
    """

    # --- 1. Compute r_min and adjust if too small ---
    r_min = radius + v_min * np.sin(semi_angle)
    P_min = P + v_min * np.cos(semi_angle) * D

    if r_min < r_min_thresh:
        delta_v = (r_min_thresh - r_min) / np.sin(semi_angle)
        v_min_new = v_min + delta_v
        P_min = P + v_min_new * np.cos(semi_angle) * D
        r_min = radius + v_min_new * np.sin(semi_angle)
    else:
        v_min_new = v_min

    # --- 2. Update geometry baseline ---
    v_min = v_min_new
    v_max = v_max - v_min
    v_min = 0.0
    P = P_min
    radius = max(r_min, r_min_thresh)

    # --- 3. Handle u range ---
    u_center = 0.5 * (u_min + u_max)
    u_diff = u_max - u_min
    if u_diff > 2 * np.pi:
        u_diff -= u_diff // (2 * np.pi) * 2 * np.pi
    u_half = 0.5 * u_diff / np.pi
    v_center = 0.5 * (v_min + v_max)
    v_half = 0.5 * (v_max - v_min)

    # --- 4. Rotate X around D to make u_center -> 0 ---
    def rotate_vector_around_axis(v, axis, angle):
        axis = axis / np.linalg.norm(axis)
        v_par = np.dot(v, axis) * axis
        v_perp = v - v_par
        w = np.cross(axis, v_perp)
        return v_par + np.cos(angle) * v_perp - np.sin(angle) * w

    X = rotate_vector_around_axis(X, D, -u_center)
    Y = np.cross(D, X)
    Y /= np.linalg.norm(Y)

    # --- 5. Center angle normalization for neural input ---
    u_center = 0.0
    sin_u_center, cos_u_center = 0.0, 1.0

    # --- 6. Pack results ---
    UV = np.array([sin_u_center, cos_u_center, u_half, v_center, v_half, 0, 0, 0], dtype=np.float64)
    scalar_params = [semi_angle / (np.pi / 2), radius]

    return P, D, X, Y, UV, scalar_params


class dataset_compound_cache(Dataset):
    def __init__(self, cache_path: str, detect_closed: bool = False):
        super().__init__()
        self.cache_path = cache_path
        self.data = np.load(cache_path)
        self.params = self.data['params']
        self.types = self.data['types']
        self.shifts = self.data['shifts']
        self.rotations = self.data['rotations']
        self.scales = self.data['scales']
        self.replica = 1
        
        # Try to load is_closed data if available
        self.detect_closed = detect_closed
        if detect_closed:
            if 'is_u_closed' in self.data and 'is_v_closed' in self.data:
                self.is_u_closed = self.data['is_u_closed']
                self.is_v_closed = self.data['is_v_closed']
                self.has_closed_data = True
            else:
                print(f"Warning: detect_closed=True but cache file does not contain is_u_closed/is_v_closed data")
                self.has_closed_data = False
        else:
            self.has_closed_data = False
        
    def __len__(self):
        return len(self.params)
    
    def __getitem__(self, idx):
        if self.detect_closed and self.has_closed_data:
            return (torch.from_numpy(self.params[idx]), 
                   torch.from_numpy(np.array(self.types[idx])), 
                   1, 
                   torch.from_numpy(self.shifts[idx]), 
                   torch.from_numpy(self.rotations[idx]), 
                   torch.from_numpy(np.array(self.scales[idx])),
                   torch.from_numpy(np.array(self.is_u_closed[idx])),
                   torch.from_numpy(np.array(self.is_v_closed[idx])))
        else:
            return (torch.from_numpy(self.params[idx]), 
                   torch.from_numpy(np.array(self.types[idx])), 
                   1, 
                   torch.from_numpy(self.shifts[idx]), 
                   torch.from_numpy(self.rotations[idx]), 
                   torch.from_numpy(np.array(self.scales[idx])))
    
class dataset_compound(Dataset):
    """
    Dataset for loading surface data from JSON files in a directory.
    
    Each sample is a complete JSON file containing multiple surfaces.
    Each surface is represented by:
    - P: location (3D point)
    - D: direction[0] (3D direction vector)
    - UV: uv bounds [u_min, u_max, v_min, v_max]
    - Additional parameters (scalar): type-specific parameters
    
    Surface types and their scalar parameters:
    - plane: no scalar parameters (dim=0)
    - cylinder: [radius] (dim=1)
    - cone: [semi_angle, radius] (dim=2)
    - sphere: [radius] (dim=1)
    - torus: [major_radius, minor_radius] (dim=2)
    - bspline_surface: [control_points] (dim=48, 4x4x3 flattened)
    """
    
    # Surface type to index mapping
    
    
    def __init__(self, json_dir: str, max_num_surfaces: int = 500, canonical: bool = False, detect_closed: bool = False, bspline_fit_threshold: float = 1e-5):
        """
        Args:
            json_dir: Path to directory containing JSON files
            max_surfaces_per_file: Maximum number of surfaces per file (None = use actual max)
        """
        super().__init__()
        self.json_dir = Path(json_dir)
        self.max_num_surfaces = max_num_surfaces
        self.canonical = canonical        
        self.bspline_fit_threshold = bspline_fit_threshold
        # Discover all JSON files in directory and subdirectories
        self.json_names = sorted([
            str(p) for p in self.json_dir.rglob("*.json")
        ])
        
        if not self.json_names:
            print(f"No JSON files found in {json_dir}")
        
        # Calculate maximum parameter dimension for padding
        # P(3) + D(3) + UV(4) + scalar(max_dim)
        self.base_dim = 3 + 3 + 3 + 8   # P + D + X + UV = 17
        
        # Scan all JSON files to determine max_scalar_dim and max_num_surfaces
        # self._calculate_dataset_stats()
        self.max_scalar_dim = max(SCALAR_DIM_MAP.values())
        self.max_param_dim = self.base_dim + self.max_scalar_dim # Should be 65 (17 + 48)
        
        self.replica = 1

        self.detect_closed = detect_closed
        
        self.postprocess_funcs = {k: build_surface_postpreprocess(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
        self.preprocess_funcs = {k: build_surface_process(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
        # # Override max_num_surfaces if specified
        # if self.max_surfaces_per_file is not None:
        #     self.max_num_surfaces = self.max_surfaces_per_file

    def __len__(self):
        """Return number of JSON files in the dataset."""
        return len(self.json_names)
    
    def get_valid_param_length(self, surface_type_idx):
        """
        Get the valid parameter length for a given surface type.
        
        Args:
            surface_type_idx: Integer or array of integers representing surface type indices
                             (0=plane, 1=cylinder, 2=cone, 3=sphere, 4=torus, 5=bspline_surface)
        
        Returns:
            Integer or array of integers representing the valid parameter length for each type
        """
        # Map from surface type index to valid parameter length
        # base_dim (17) + scalar_dim
        type_to_length = {
            0: 17,  # plane: base_dim + 0
            1: 18,  # cylinder: base_dim + 1
            2: 19,  # cone: base_dim + 2
            3: 18,  # sphere: base_dim + 1
            4: 19,  # torus: base_dim + 2
            5: 65,  # bspline_surface: base_dim + 48
        }
        
        # Handle both single value and array inputs
        if isinstance(surface_type_idx, (np.ndarray, torch.Tensor)):
            if isinstance(surface_type_idx, torch.Tensor):
                surface_type_idx = surface_type_idx.cpu().numpy()
            return np.array([type_to_length.get(int(t), 0) for t in surface_type_idx])
        else:
            return type_to_length.get(int(surface_type_idx), 0)
    
    def get_valid_param_mask(self, surface_type_idx, return_tensor=False):
        """
        Get a boolean mask indicating valid parameter positions for given surface type(s).
        
        Args:
            surface_type_idx: Integer, numpy array, or torch tensor of surface type indices
                             Shape can be (,) for single type or (N,) for batch
            return_tensor: If True, return torch.Tensor; otherwise return numpy array
        
        Returns:
            Boolean mask of shape (max_param_dim,) or (N, max_param_dim)
            True indicates valid parameter positions, False indicates padding
        """
        is_batch = isinstance(surface_type_idx, (np.ndarray, torch.Tensor))
        
        if isinstance(surface_type_idx, torch.Tensor):
            surface_type_idx_np = surface_type_idx.cpu().numpy()
        elif isinstance(surface_type_idx, np.ndarray):
            surface_type_idx_np = surface_type_idx
        else:
            surface_type_idx_np = np.array([surface_type_idx])
            is_batch = False
        
        # Get valid lengths for all types
        valid_lengths = self.get_valid_param_length(surface_type_idx_np)
        
        # Create mask
        if is_batch:
            batch_size = len(surface_type_idx_np)
            mask = np.zeros((batch_size, self.max_param_dim), dtype=bool)
            for i, length in enumerate(valid_lengths):
                mask[i, :length] = True
        else:
            mask = np.zeros(self.max_param_dim, dtype=bool)
            mask[:valid_lengths] = True
        
        if return_tensor:
            return torch.from_numpy(mask)
        return mask
    
    def _build_occ_bspline_surface(
        self,
        u_degree: int,
        v_degree: int,
        num_poles_u: int,
        num_poles_v: int,
        u_knots: np.ndarray,
        v_knots: np.ndarray,
        u_mults: np.ndarray,
        v_mults: np.ndarray,
        poles: np.ndarray,
        is_u_periodic: bool,
        is_v_periodic: bool,
    ) -> Geom_BSplineSurface:
        """Build OCC BSpline surface from parameters."""
        ctrl_pts = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
        weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
        for i in range(num_poles_u):
            for j in range(num_poles_v):
                x, y, z, w = poles[i, j]
                ctrl_pts.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
                weights.SetValue(i + 1, j + 1, float(max(w, 1e-8)))

        occ_u_knots = TColStd_Array1OfReal(1, len(u_knots))
        for idx, knot in enumerate(u_knots):
            occ_u_knots.SetValue(idx + 1, float(knot))
        occ_v_knots = TColStd_Array1OfReal(1, len(v_knots))
        for idx, knot in enumerate(v_knots):
            occ_v_knots.SetValue(idx + 1, float(knot))

        occ_u_mults = TColStd_Array1OfInteger(1, len(u_mults))
        for idx, mult in enumerate(u_mults):
            occ_u_mults.SetValue(idx + 1, int(mult))
        occ_v_mults = TColStd_Array1OfInteger(1, len(v_mults))
        for idx, mult in enumerate(v_mults):
            occ_v_mults.SetValue(idx + 1, int(mult))

        return Geom_BSplineSurface(
            ctrl_pts,
            weights,
            occ_u_knots,
            occ_v_knots,
            occ_u_mults,
            occ_v_mults,
            int(u_degree),
            int(v_degree),
            bool(is_u_periodic),
            bool(is_v_periodic),
        )

    def _sample_bspline_surface_grid(
        self,
        surface: Geom_BSplineSurface,
        num_u: int = 32,
        num_v: int = 32,
    ) -> np.ndarray:
        """Sample a BSpline surface in a 32x32 grid."""
        face_builder = BRepBuilderAPI_MakeFace(surface, 1e-7)
        surface = face_builder.Face()
        surface_handle = BRep_Tool.Surface(surface)
        surface = GeomAdaptor_Surface(surface_handle)
        u_min = surface.FirstUParameter()
        u_max = surface.LastUParameter()
        v_min = surface.FirstVParameter()
        v_max = surface.LastVParameter()
        
        grid = np.zeros((num_u, num_v, 3), dtype=np.float64)
        for ui in range(num_u):
            u = u_min + (u_max - u_min) * ui / (num_u - 1) if num_u > 1 else u_min
            for vi in range(num_v):
                v = v_min + (v_max - v_min) * vi / (num_v - 1) if num_v > 1 else v_min
                point = surface.Value(float(u), float(v))
                grid[ui, vi] = [point.X(), point.Y(), point.Z()]
        return grid

    def _compute_bspline_centroid_and_normal(
        self,
        surface: Geom_BSplineSurface,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute centroid and normal vector from a BSpline surface."""
        grid = self._sample_bspline_surface_grid(surface, 32, 32)
        centroid = grid.reshape(-1, 3).mean(axis=0)
        
        # Find normal at center
        center_u = grid.shape[0] // 2
        center_v = grid.shape[1] // 2
        
        # Get neighboring points
        prev_u = max(center_u - 1, 0)
        next_u = min(center_u + 1, grid.shape[0] - 1)
        prev_v = max(center_v - 1, 0)
        next_v = min(center_v + 1, grid.shape[1] - 1)
        
        u_vec = grid[next_u, center_v] - grid[prev_u, center_v]
        v_vec = grid[center_u, next_v] - grid[center_u, prev_v]
        
        u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-8)
        v_vec = v_vec / (np.linalg.norm(v_vec) + 1e-8)
        
        normal_vec = np.cross(u_vec, v_vec)
        normal_vec = normal_vec / (np.linalg.norm(normal_vec) + 1e-8)
        
        return centroid, normal_vec, u_vec, v_vec

    def _canonicalize_bspline_poles(
        self,
        poles: np.ndarray,
        surface: Geom_BSplineSurface,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Transform bspline poles to canonical space.
        
        Steps:
        1. Compute centroid and normal from surface
        2. Translate to origin (centroid subtraction)
        3. Rotate to canonical orientation
        4. Scale to fit in [-1, 1] range (max dimension = 2)
        
        Returns:
            poles: Transformed poles in canonical space
            rotation: 3x3 rotation matrix
            centroid: Original centroid position
            scale: Scale factor (original max dimension)
        """
        centroid, normal_vec, u_vec, v_vec = self._compute_bspline_centroid_and_normal(surface)
        rotation = compute_rotation_matrix(normal_vec, u_vec)
        
        # Step 1: Translate to origin
        poles_xyz = poles[..., :3] - centroid
        
        # Step 2: Rotate to canonical orientation
        reshaped = poles_xyz.reshape(-1, 3)
        rotated = reshaped @ rotation.T
        rotated_poles = rotated.reshape(poles_xyz.shape)
        
        # Step 3: Compute scale (max dimension of bounding box)
        poles_min = rotated_poles.min(axis=(0, 1))
        poles_max = rotated_poles.max(axis=(0, 1))
        scale = np.max(poles_max - poles_min)
        scale = scale / 2.0
        
        # Step 4: Normalize to [-1, 1] range
        # # Center the bounding box at origin
        # poles_center = (poles_max + poles_min) / 2.0
        # rotated_poles = rotated_poles - poles_center
        
        # Scale to [-1, 1]
        if scale > 1e-8:
            rotated_poles = rotated_poles / scale
        
        poles[..., :3] = rotated_poles
        
        return poles, rotation, centroid, scale

    def _parse_surface(self, surface_dict: Dict, closed_threshold=1e-3) -> Tuple[np.ndarray, int]:
        """
        Parse a single surface from JSON format to parameter vector.
        
        Args:
            surface_dict: Dictionary containing surface data
            
        Returns:
            params: Numpy array of shape (max_param_dim,) containing:
                    [P(3), D(3), X(3), UV(8), scalar(max_scalar_dim)]
            surface_type: Integer representing surface type
        """
        
        surface_type = surface_dict['type']
        surface_type_idx = SURFACE_TYPE_MAP.get(surface_type, -1)
        
        # Extract P: location (first element of location array)
        if surface_type == 'bspline_surface':
            P = np.zeros(3, dtype=np.float64)
            D = np.zeros(3, dtype=np.float64)
            X = np.zeros(3, dtype=np.float64)
            Y = np.zeros(3, dtype=np.float64)
        
        else:
            P = np.array(surface_dict['location'][0], dtype=np.float64)  # (3,)
            
            # Extract D: direction[0] (first direction vector), X: U direction
            D = np.array(surface_dict['direction'][0], dtype=np.float64)  # (3,)
            X = np.array(surface_dict['direction'][1], dtype=np.float64)  # (3,)
            Y = np.cross(D, X)
        # Extract UV: parameter bounds
        UV = np.array(surface_dict['uv'], dtype=np.float64)  # (4,)
        
        # Extract scalar parameters based on surface type
        scalar_params = []
        u_min, u_max, v_min, v_max = UV
        is_u_closed = False
        is_v_closed = False

        if surface_type == 'plane':
            # No scalar parameters
            centered = P + (u_max + u_min) / 2 * X + (v_max + v_min) / 2 * Y
            u = np.array([u_min, u_max])
            u_new = u - (u_max + u_min) / 2
            v = np.array([v_min, v_max])
            v_new = v - (v_max + v_min) / 2
            u_min = u_new[0]
            u_max = u_new[1]
            v_min = v_new[0]
            v_max = v_new[1]
            P = centered
            UV = np.array([u_min, u_max, v_min, v_max, 0, 0, 0, 0], dtype=np.float64) # (8, )

        elif surface_type == 'cylinder':
            # scalar[0] = radius
            scalar_params = [surface_dict['scalar'][0]]
            if scalar_params[0] < 1e-5: # Radius is too small
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
            P = P + D * v_min
            v_max = v_max - v_min
            v_min = 0

            # 1. guarantee u_min is positive
            if u_min < 0:
                k = (u_min // (2 * np.pi) )
                u_min -= k * 2 * np.pi
                u_max -= k * 2 * np.pi

            # 2. guarantee u_diff < 2 * np.pi
            if u_max - u_min > 2 * np.pi + 1e-4:
                u_max -= (u_max - u_min) // (2 * np.pi) * 2 * np.pi


            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min
            u_half = 0.5 * (u_diff) / np.pi - 0.5 # (0 - pi) --> (-0.5, 0.5)

            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            UV = np.array([sin_u_center, cos_u_center, u_half, v_max, 0, 0, 0, 0], dtype=np.float64) # (8, )
            if u_diff > 2 * np.pi - closed_threshold:
                is_u_closed = True

        elif surface_type == 'cone':

            semi_angle, radius = surface_dict['scalar'][0], surface_dict['scalar'][1]
            if radius < 1e-5:
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
            if not (1e-6 < semi_angle < np.pi/2 - 1e-6):
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
                # raise ValueError(f"Invalid semi-angle: {semi_angle}, should be in (0, pi/2)")


            # Fix the problem that leads to extremely small radius sometime. When radius - v_min * np.sin(semi_angle) ~= 0
            r_min_thresh = 1e-2
            P_min = P + v_min * np.cos(semi_angle) * D
            r_min = radius + v_min * np.sin(semi_angle)
            v_max = v_max - v_min
            v_min = 0

            if r_min < r_min_thresh:
                # Compute how much delta_v we need to increase to make r_min_thresh
                delta_v = (r_min_thresh - r_min) / np.sin(semi_angle)
                v_min_new = v_min + delta_v
                v_max = v_max + delta_v
                P_min = P + v_min_new * np.cos(semi_angle) * D
                r_min = radius + v_min_new * np.sin(semi_angle)

    
            P = P_min
            radius = max(r_min, r_min_thresh)

            # 1. guarantee u_min is positive
            if u_min < 0 - 1e-4:
                
                k = (u_min // (2 * np.pi) )
                u_min -= k * 2 * np.pi
                u_max -= k * 2 * np.pi
            # 2. guarantee u_diff < 2 * np.pi
            if u_max - u_min > 2 * np.pi + 1e-4:
                u_max -= (u_max - u_min) // (2 * np.pi) * 2 * np.pi

            
            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min
                
            u_half = 0.5 * (u_diff) / np.pi  # (0 - pi) --> (0, 1)
            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            v_center = 0.5 * (v_min + v_max)
            v_half = 0.5 * (v_max - v_min)


            UV = [sin_u_center, cos_u_center, u_half, v_center, v_half, 0, 0, 0] # (8, )
            scalar_params = [semi_angle / (np.pi/2), radius]

            if u_diff > 2 * np.pi - closed_threshold:
                is_u_closed = True
            

        elif surface_type == 'sphere':
            PI = np.pi
            HALF_PI = np.pi/2
            TWO_PI = 2*np.pi
            def canonicalize_vc_uc(u_c, v_c):
                # bring v_c into (-pi/2, pi/2], adjusting u_c accordingly, without extra flags
                while v_c > HALF_PI:
                    v_c -= PI
                    u_c += PI
                while v_c <= -HALF_PI:
                    v_c += PI
                    u_c += PI
                # normalize u_c to (-pi, pi] or [0,2pi) if you prefer
                u_c = (u_c + PI) % (2*PI) - PI
                return u_c, v_c
            # scalar[0] = radius

            scalar_params = [surface_dict['scalar'][0]]
            if scalar_params[0] < 1e-5:
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1

            u_center = 0.5 * (u_min + u_max)
            v_center = 0.5 * (v_min + v_max)
            u_diff = u_max - u_min
            v_diff = v_max - v_min
            u_half = 0.5 * (u_diff)
            v_half = 0.5 * (v_diff)
            u_center, v_center = canonicalize_vc_uc(u_center, v_center)

            dir_vec = np.array([
                np.cos(v_center) * np.cos(u_center),
                np.cos(v_center) * np.sin(u_center),
                np.sin(v_center)
            ], dtype=np.float64)
            
            u_h_norm = np.clip(u_half / np.pi, 0, 1 - 1e-5)        # 在 [-1, 1]
            v_h_norm = np.clip(v_half / (PI/2), 0.0, 1.0 - 1e-5)   # 在 [-1, 1]

            UV = np.concatenate([dir_vec, [u_h_norm, v_h_norm, 0, 0, 0]])

            if u_diff > 2 * np.pi - closed_threshold:
                is_u_closed = True

            if v_diff > np.pi - closed_threshold:
                is_v_closed = True


        elif surface_type == 'torus':
            # scalar[0] = major_radius, scalar[1] = minor_radius
            scalar_params = [surface_dict['scalar'][0], surface_dict['scalar'][1]]
            if scalar_params[0] < 1e-5 or scalar_params[1] < 1e-5:
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1

            # 1. guarantee u_min is positive
            if u_min < 0 - 1e-6:
                k = (u_min // (2 * np.pi))
                u_min -= k * 2 * np.pi
                u_max -= k * 2 * np.pi
            # 2. guarantee u_diff < 2 * np.pi
            if u_max - u_min > 2 * np.pi + 1e-4:
                u_max -= (u_max - u_min) // ((2 * np.pi) - 1) * 2 * np.pi

            # 3. guarantee v_min is positive
            if v_min < 0 - 1e-6:
                k = (v_min // (2 * np.pi))
                v_min -= k * 2 * np.pi
                v_max -= k * 2 *np.pi
                ic('v_min < 0, add ', k, 'times 2pi', 'now v_min: ', v_min, 'v_max: ', v_max)
            # 4. guarantee v_diff < 2 * np.pi
            if v_max - v_min > 2 * np.pi + 1e-4:
                v_max -= ((v_max - v_min) // (2 * np.pi) - 1) * 2 * np.pi

            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min
            u_half = 0.5 * (u_diff)
            
            v_center = 0.5 * (v_min + v_max)
            v_diff = v_max - v_min
            v_half = 0.5 * (v_diff)

            # ic('u_center: ', u_center, 'v_center: ', v_center, 'u_diff: ', u_diff, 'v_diff: ', v_diff)
            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            sin_v_center, cos_v_center = np.sin(v_center), np.cos(v_center)

            c, s = np.cos(-u_center), np.sin(-u_center)


            UV = np.array([sin_u_center, cos_u_center, u_half / np.pi, sin_v_center, cos_v_center, v_half / np.pi, 0, 0], dtype=np.float64)

            if u_diff > 2 * np.pi - closed_threshold:
                is_u_closed = True
            if v_diff > np.pi - closed_threshold:
                is_v_closed = True


        elif surface_type == 'bspline_surface':
            # Process bspline surface: convert to canonical space, fit 4x4x3 poles
            # try:
            # Extract bspline surface parameters from JSON
            scalar_data = surface_dict["scalar"]
            u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v = map(int, scalar_data[:6])
            
            # Validate degrees
            if u_degree > 3 or v_degree > 3:
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
            
            # Extract knots and multiplicities
            u_knots_list = np.array(scalar_data[6 : 6 + num_knots_u], dtype=np.float64)
            v_knots_list = np.array(scalar_data[6 + num_knots_u : 6 + num_knots_u + num_knots_v], dtype=np.float64)
            u_mults_list = np.array(scalar_data[6 + num_knots_u + num_knots_v : 6 + num_knots_u + num_knots_v + num_knots_u], dtype=np.int32)
            v_mults_list = np.array(scalar_data[6 + num_knots_u + num_knots_v + num_knots_u :], dtype=np.int32)
            
            # Extract poles
            poles_data = np.array(surface_dict["poles"], dtype=np.float64)
            poles = poles_data.reshape(num_poles_u, num_poles_v, 4)
            
            is_u_periodic = surface_dict.get("u_periodic", False)
            is_v_periodic = surface_dict.get("v_periodic", False)
            
            # Build OCC surface
            surface = self._build_occ_bspline_surface(
                u_degree, v_degree, num_poles_u, num_poles_v,
                u_knots_list, v_knots_list, u_mults_list, v_mults_list,
                poles, is_u_periodic, is_v_periodic
            )
            
            # Transform to canonical space
            poles_canonical, rotation, shift, scale = self._canonicalize_bspline_poles(poles.copy(), surface)
            
            # Rebuild surface with canonical poles (now in [-1, 1] range)
            surface_canonical = self._build_occ_bspline_surface(
                u_degree, v_degree, num_poles_u, num_poles_v,
                u_knots_list, v_knots_list, u_mults_list, v_mults_list,
                poles_canonical, is_u_periodic, is_v_periodic
            )
            
            # Sample 32x32 points from canonical surface
            sampled_points = self._sample_bspline_surface_grid(surface_canonical, 32, 32)
            
            # Fit 4x4x3 control points using get_approx_face
            try:
                fitted_control_points = get_approx_face(sampled_points)
            except AssertionError as e:
                print('Bspline surface too complex, drop')
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
            
            # Compute fitting error
            fitted_poles = np.array(fitted_control_points).reshape(4, 4, 3)
            fitted_poles_with_weights = np.concatenate(
                [fitted_poles, np.ones((4, 4, 1))], axis=-1
            )
            
            # Build fitted surface and compare
            fitted_knots = np.array([0.0, 1.0], dtype=np.float64)
            fitted_mults = np.array([4, 4], dtype=np.int32)
            fitted_surface = self._build_occ_bspline_surface(
                3, 3, 4, 4,
                fitted_knots, fitted_knots, fitted_mults, fitted_mults,
                fitted_poles_with_weights, False, False
            )
            
            # Sample fitted surface
            fitted_samples = self._sample_bspline_surface_grid(fitted_surface, 32, 32)
            
            # Compute MSE (both surfaces are in [-1, 1] canonical space)
            fit_error = np.mean((sampled_points - fitted_samples) ** 2)
            
            if fit_error > self.bspline_fit_threshold:
                print(f'Bspline fitting error: {fit_error:.5f} to large, drop')
                # Fitting error too large, drop this surface
                if self.detect_closed:
                    return None, -1, None, None
                else:
                    return None, -1
            
            # The fitted poles are already in canonical space [-1, 1]
            # Flatten to 48D vector directly
            fitted_poles_flat = fitted_poles.reshape(-1, 3)
            scalar_params = fitted_poles_flat.flatten()
            
            # For bspline, we store transformation info in P, D, X, UV positions
            # This allows us to recover the original surface later
            # P: shift (centroid)
            # D: rotation matrix row 0
            # X: rotation matrix row 1  
            # UV[0:3]: rotation matrix row 2
            # UV[3]: scale
            # UV[4:8]: padding
            P = shift
            D = rotation[0, :]
            X = rotation[1, :]
            UV = np.zeros(8, dtype=np.float64)
            UV[0:3] = rotation[2, :]
            UV[3] = scale
                
                
            # except Exception as e:
            #     # Failed to process bspline, skip
            #     print(f"Failed to process bspline_surface: {e}")
            #     if self.detect_closed:
            #         return None, -1, None, None
            #     else:
            #         return None, -1
        
        scalar_params = np.array(scalar_params, dtype=np.float64)
        
        # Pad scalar parameters to max_scalar_dim
        # if len(scalar_params) < self.max_scalar_dim:
        #     padding = np.zeros(self.max_scalar_dim - len(scalar_params), dtype=np.float64)
        #     scalar_params = np.concatenate([scalar_params, padding])
        
        # Concatenate all parameters: P + D + UV + scalar
        params = np.concatenate([P, D, X, UV, scalar_params])
        # Just control the max value of params to be 10
        if np.max(np.abs(params)) > 10:
            if self.detect_closed:
                return None, -1, None, None
            else:
                return None, -1
        
        # For bspline_surface, skip the preprocess/postprocess validation
        if surface_type != 'bspline_surface':
            assert np.allclose(params, self.postprocess_funcs[surface_type](self.preprocess_funcs[surface_type](params))), f"type: {surface_type}, params: {params}, postprocess: {self.postprocess_funcs[surface_type](self.preprocess_funcs[surface_type](params))}"
            # Do the log processing of radius
            params = self.preprocess_funcs[surface_type](params)
        
        # assert len(params) == self.base_dim + SCALAR_DIM_MAP[surface_type], f"surface {surface_type} params length {len(params)} != expected {self.base_dim + SCALAR_DIM_MAP[surface_type]}"

        if self.detect_closed:
            return params.astype(np.float32), surface_type_idx, is_u_closed, is_v_closed
        else:
            return params.astype(np.float32), surface_type_idx
    

    def _recover_surface(self, params, surface_type_idx):
        """
        Recover a surface from parameter vector.
        """
        params = params.astype(np.float64)
        SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}
        surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx, -1)
        # print(surface_type, params, f'len params: {len(params)}')
        
        # For bspline_surface, skip postprocess and extract control points directly
        if surface_type == 'bspline_surface':
            scalar_params = params[17:]  # 48D control points
        else:
            # Apply exp to radius for other surface types
            params = self.postprocess_funcs[surface_type](params)
            P = params[:3]
            D = params[3:6] / np.linalg.norm(params[3:6])
            X = params[6:9] / np.linalg.norm(params[6:9])
            Y = np.cross(D, X)
            UV = params[9:17]
            scalar_params = params[17:]
        
        if surface_type == 'plane':
            assert len(scalar_params) == 0
            u_min, u_max, v_min, v_max = UV[:4]
            scalar = []
        elif surface_type == 'cylinder':

            sin_u_center, cos_u_center, u_half, height = UV[:4]
            u_center = np.arctan2(sin_u_center, cos_u_center)
            u_half = np.clip((u_half + 0.5), 0, 1 - 1e-5) * np.pi
            
            if np.abs(u_half - np.pi) < 1e-4:
                u_half = np.pi
            u_min, u_max = u_center - u_half, u_center + u_half

            # if np.abs(np.abs(u_max - u_min) - 2 * np.pi) < 1e-4:
            #     # A full loop, make sure distance less than 2pi
            #     u_max = u_min + 2 * np.pi - 1e-4

            v_min = 0.0
            v_max = height

            radius = scalar_params[0]
            scalar = [radius]

        elif surface_type == 'cone':
            sin_u_center, cos_u_center, u_half, v_center, v_half = UV[:5]
            uc = np.arctan2(sin_u_center, cos_u_center)
            u_half = np.clip(u_half, 0, 1 - 1e-5) * np.pi

            if np.abs(u_half - np.pi) < 1e-4:
                u_half = np.pi

            u_min, u_max = uc - u_half, uc + u_half
   

            v_min, v_max = v_center - v_half, v_center + v_half
            v_min, v_max = v_center - v_half, v_center + v_half


            assert len(scalar_params) == 2
            radius = scalar_params[1]
            semi_angle = scalar_params[0]
            semi_angle = semi_angle * (np.pi/2)

            scalar = [semi_angle, radius]
        elif surface_type == 'torus':

            sin_u_center, cos_u_center, u_half, sin_v_center, cos_v_center, v_half = UV[:6]

            u_center = np.arctan2(sin_u_center, cos_u_center)
            u_half = np.clip(u_half, 0, 1 - 1e-5) * np.pi
            if np.abs(u_half - np.pi) < 1e-4:
                u_half = np.pi
            u_min, u_max = u_center - u_half, u_center + u_half

            v_center = np.arctan2(sin_v_center, cos_v_center)
            v_half = np.clip(v_half, 0, 1 - 1e-5) * np.pi
            if np.abs(v_half - np.pi) < 1e-4:
                v_half = np.pi
            v_min, v_max = v_center - v_half, v_center + v_half

            assert len(scalar_params) == 2, f"Wrong scalar number of torus, should be 2 but got {len(scalar_params)}"
            major_radius = scalar_params[0]
            minor_radius = scalar_params[1]
            scalar = [major_radius, minor_radius]
            ic('Torus recovered: u_center: ', u_center, 'v_center: ', v_center, 'u_half: ', u_half, 'v_half: ', v_half)
            ic('u_min: ', u_min, 'u_max: ', u_max, 'v_min: ', v_min, 'v_max: ', v_max)

        elif surface_type == 'sphere':
            dir_vec = UV[:3].astype(np.float64)
            u_h_norm = float(UV[3])
            v_h_norm = float(UV[4])

            if np.linalg.norm(dir_vec) == 0:
                raise ValueError("zero dir vector")
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            x, y, z = dir_vec

            u_c = np.arctan2(y, x)   # in (-pi, pi]
            v_c = np.arcsin(np.clip(z, -1.0, 1.0))  # in [-pi/2, pi/2]

            # recover half widths
            u_h = np.clip(u_h_norm, 0.0, 1.0 - 1e-5) * np.pi
            v_h = np.clip(v_h_norm, 0.0, 1.0 - 1e-5) * (np.pi/2)

            if np.abs(u_h - np.pi) < 1e-4:
                u_h = np.pi
            if np.abs(v_h - np.pi/2) < 1e-4:
                v_h = np.pi/2
                
            u_min, u_max = u_c - u_h, u_c + u_h
            v_min, v_max = v_c - v_h, v_c + v_h



            assert len(scalar_params) == 1, f"Wrong scalar number of sphere, should be 1 but got {len(scalar_params)}"
            radius = scalar_params[0]   
            scalar = [radius]
        elif surface_type == 'bspline_surface':
            # Recover bspline surface from 48D control points
            assert len(scalar_params) == 48, f"Wrong params length for bspline_surface, should be 48 but got {len(scalar_params)}"
            
            # Reshape to 4x4x3 (these are in canonical space [-1, 1])
            # control_points_canonical = scalar_params.reshape(4, 4, 3)
            
            # Optional: Transform back to original space
            # For now, return the canonical space surface
            # If you want original space, apply: 
            # control_points_original = (control_points_canonical * (scale / 2.0)) @ rotation + shift
            
            control_points_flat = scalar_params.reshape(4, 4, 3)
            
            # Build a simple cubic B-spline surface (degree 3, 4x4 control points)
            # Knot vectors: [0, 0, 0, 0, 1, 1, 1, 1] for both u and v
            u_degree = 3
            v_degree = 3
            num_poles_u = 4
            num_poles_v = 4
            num_knots_u = 2
            num_knots_v = 2
            
            u_knots = [0.0, 1.0]
            v_knots = [0.0, 1.0]
            u_mults = [4, 4]
            v_mults = [4, 4]
            
            # Build scalar array for bspline
            scalar = [u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v]
            scalar.extend(u_knots)
            scalar.extend(v_knots)
            scalar.extend(u_mults)
            scalar.extend(v_mults)
            
            # Build poles with weights = 1
            poles = []
            for i in range(4):
                row = []
                for j in range(4):
                    x, y, z = control_points_flat[i, j]
                    row.append([float(x), float(y), float(z), 1.0])
                poles.append(row)
            
            return {
                'type': surface_type,
                'scalar': scalar,
                'poles': poles,
                'u_periodic': False,
                'v_periodic': False,
            }
        else:
            raise NotImplementedError(f"Surface type {surface_type} not implemented")

        return {
            'type': surface_type,
            'location': [P.tolist()],
            'direction': [D.tolist(), X.tolist(), Y.tolist()],
            'uv': [float(u_min), float(u_max), float(v_min), float(v_max)],
            'scalar': [float(s) for s in scalar],
        }

    def _load_approx(self, npz_data, surface_idx):
        # points = npz_data[f'points_{surface_idx}']
        approx = npz_data[f'approx_{surface_idx}']
        return approx
    
    def _load_points(self, npz_data, surface_idx):  
        points = npz_data[f'points_{surface_idx}']
        return points

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all surfaces from a single JSON file.
        
        Args:
            idx: Index of the JSON file
            
        Returns:
            params: Tensor of shape (max_num_surfaces, max_param_dim) containing surface parameters
            surface_types: Tensor of shape (max_num_surfaces,) containing surface type indices
            mask: Tensor of shape (max_num_surfaces,) indicating valid surfaces (1) vs padding (0)
        """
        json_path = self.json_names[idx]
        # print(json_path)
        
        # Load JSON file

        ic('json_path: ', json_path)
        
        # Initialize arrays for all surfaces (padded)
        all_params = np.zeros((self.max_num_surfaces, self.max_param_dim), dtype=np.float64)
        all_types = np.zeros(self.max_num_surfaces, dtype=np.int64)
        all_shifts = np.zeros((self.max_num_surfaces, 3), dtype=np.float64)
        all_rotations = np.zeros((self.max_num_surfaces, 3, 3), dtype=np.float64)
        all_scales = np.zeros(self.max_num_surfaces, dtype=np.float64)
        all_is_u_closed = np.zeros(self.max_num_surfaces, dtype=bool)
        all_is_v_closed = np.zeros(self.max_num_surfaces, dtype=bool)
        mask = np.zeros(self.max_num_surfaces, dtype=np.float64)



        try:
            with open(json_path, 'r') as f:
                surfaces_data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSONDecodeError: {json_path}")
            with open('./assets/abnormal_json.csv', 'w') as f:
                f.write(json_path + '\n')
            return torch.from_numpy(all_params).float(), torch.from_numpy(all_types).long(), torch.from_numpy(mask).float(), torch.from_numpy(all_shifts).float(), torch.from_numpy(all_rotations).float(), torch.from_numpy(all_scales).float()
        
        # When the params are larger than 10, instead of dropping them we use the approx bspline instead.
        # Parse each surface
        for i, surface_dict in enumerate(surfaces_data):
            # print(i)
            if i >= self.max_num_surfaces:
                break

            # try:
            # Catch RuntimeWarnings (overflow, invalid value) and convert to exceptions
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                # print(surface_dict)
                # if i == 6:
                #     print()
                if self.detect_closed:
                    params, surface_type_idx, is_u_closed, is_v_closed = self._parse_surface(surface_dict)
                else:
                    params, surface_type_idx = self._parse_surface(surface_dict)

                # Transform to canonical space.
                
                # For bspline_surface, canonical transform is already done in _parse_surface
                # Extract transformation info from params
                if surface_type_idx == SURFACE_TYPE_MAP.get('bspline_surface', -1):
                    # Extract transformation info from P, D, X, UV positions
                    shift = copy(params[0:3])   
                    rotation_row0 = copy(params[3:6])
                    rotation_row1 = copy(params[6:9])
                    rotation_row2 = copy(params[9:12])
                    rotation = np.array([rotation_row0, rotation_row1, rotation_row2], dtype=np.float64)
                    scale = copy(params[12])
                    params[:17] = 0

                elif self.canonical and surface_type_idx != -1:
                    # For other surface types, use to_canonical
                    surface_str = self._recover_surface(params, surface_type_idx)
                    # print(i, params[9:])
                    surface_canonical, shift, rotation, scale = to_canonical(surface_str)
                    if self.detect_closed:
                        params, surface_type_idx, is_u_closed, is_v_closed = self._parse_surface(surface_canonical)
                    else:
                        params, surface_type_idx = self._parse_surface(surface_canonical)

                else:
                    shift = np.zeros(3, dtype=np.float64)
                    rotation = np.eye(3, dtype=np.float64)
                    scale = 1.0
                    pass

                if surface_type_idx == -1:
                    # Bad surface, skip
                    with open('./assets/abnormal_surfaces.csv', 'a') as f:
                        f.write(json_path + ',' + str(i) + ',' + str(surface_type_idx) + '\n')
                else:
                    all_params[i, :len(params)] = params
                    all_types[i] = surface_type_idx
                    all_shifts[i, :] = shift
                    all_rotations[i, :, :] = rotation
                    all_scales[i] = scale
                    if self.detect_closed:
                        all_is_u_closed[i] = is_u_closed
                        all_is_v_closed[i] = is_v_closed
                    mask[i] = 1.0
            # except (KeyError, IndexError, NotImplementedError, RuntimeWarning) as e:
            #     # Skip invalid surfaces (leave as zeros with mask=0)
            #     # print(f"Warning: Skipping surface {i} in {json_path}: {e}")
            #     continue
            
        
        # Convert to tensors
        params_tensor = torch.from_numpy(all_params).float() 
        types_tensor = torch.from_numpy(all_types).long()
        mask_tensor = torch.from_numpy(mask).float()
        if self.detect_closed:
            is_u_closed_tensor = torch.from_numpy(all_is_u_closed).bool()
            is_v_closed_tensor = torch.from_numpy(all_is_v_closed).bool()
        
        if params_tensor.abs().max() > 10:
            pos = torch.argmax(params_tensor.abs())
            row = (pos // params_tensor.shape[1]).item()
            col = (pos % params_tensor.shape[1]).item()
            surface_type_abl = types_tensor[row]
            surface_type_str = get_surface_type(surface_type_abl.item())
            to_save = [json_path, str(int(row)), surface_type_str, str(int(col)), str(params_tensor.abs().max().item())]
            with open('abnormal_params_1023.csv', 'a') as f:
                f.write(','.join(to_save) + '\n')
        
            mask_tensor[torch.where(params_tensor.abs() > 10)[0].unique()] = 0

        if self.detect_closed:
            return params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales, is_u_closed_tensor, is_v_closed_tensor
        else:
            return params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales
    
    def get_file_info(self, idx: int) -> Dict:
        """
        Get full file information for inspection.
        
        Args:
            idx: Index of the JSON file
            
        Returns:
            Dictionary containing file path, number of surfaces, and parsed data
        """
        json_path = self.json_names[idx]
        
        with open(json_path, 'r') as f:
            surfaces_data = json.load(f)
        
        params_tensor, types_tensor, mask_tensor = self[idx]
        
        return {
            'json_path': json_path,
            'num_surfaces': len(surfaces_data),
            'params': params_tensor,
            'surface_types': types_tensor,
            'mask': mask_tensor,
            'original_data': surfaces_data,
        }

def get_surface_type(surf_type):
    return SURFACE_TYPE_MAP_INV[surf_type]        



if __name__ == '__main__':
    from tqdm import tqdm
    import time
    dataset = dataset_compound(sys.argv[1], canonical=True, detect_closed=True)
    
    total_u_closed = 0
    total_v_closed = 0
    total_surfaces = 0
    
    for i in tqdm(range(len(dataset))):
        params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales, is_u_closed_tensor, is_v_closed_tensor = dataset[i]
        
        # Only count valid surfaces (where mask == 1)
        valid_mask = mask_tensor.bool()
        total_u_closed += is_u_closed_tensor[valid_mask].sum().item()
        total_v_closed += is_v_closed_tensor[valid_mask].sum().item()
        total_surfaces += valid_mask.sum().item()
    
    print(f"\n{'='*50}")
    print(f"Total surfaces: {total_surfaces}")
    print(f"U-closed surfaces: {total_u_closed} ({100 * total_u_closed / total_surfaces:.2f}%)")
    print(f"V-closed surfaces: {total_v_closed} ({100 * total_v_closed / total_surfaces:.2f}%)")
    print(f"{'='*50}")
