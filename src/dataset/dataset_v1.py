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
from src.tools.surface_to_canonical_space import to_canonical, from_canonical


ic.enable()
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
    'bspline_surface': -1,  # variable dimension
}
SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}



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
    """
    
    # Surface type to index mapping
    
    
    def __init__(self, json_dir: str, max_num_surfaces: int = 500, canonical: bool = False):
        """
        Args:
            json_dir: Path to directory containing JSON files
            max_surfaces_per_file: Maximum number of surfaces per file (None = use actual max)
        """
        super().__init__()
        self.json_dir = Path(json_dir)
        self.max_num_surfaces = max_num_surfaces
        self.canonical = canonical        
        # Discover all JSON files in directory and subdirectories
        self.json_names = sorted([
            str(p) for p in self.json_dir.rglob("*.json")
        ])
        
        if not self.json_names:
            raise ValueError(f"No JSON files found in {json_dir}")
        
        # Calculate maximum parameter dimension for padding
        # P(3) + D(3) + UV(4) + scalar(max_dim)
        self.base_dim = 3 + 3 + 3 + 8   # P + D + X + UV = 17
        
        # Scan all JSON files to determine max_scalar_dim and max_num_surfaces
        # self._calculate_dataset_stats()
        self.max_scalar_dim = max(SCALAR_DIM_MAP.values())
        self.max_param_dim = self.base_dim + self.max_scalar_dim # Should be 19
        
        self.replica = 1

        
        
        self.postprocess_funcs = {k: build_surface_postpreprocess(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
        self.preprocess_funcs = {k: build_surface_process(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
        # # Override max_num_surfaces if specified
        # if self.max_surfaces_per_file is not None:
        #     self.max_num_surfaces = self.max_surfaces_per_file

    def __len__(self):
        """Return number of JSON files in the dataset."""
        return len(self.json_names)
    
    def _parse_surface(self, surface_dict: Dict) -> Tuple[np.ndarray, int]:
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

        elif surface_type == 'cone':

            semi_angle, radius = surface_dict['scalar'][0], surface_dict['scalar'][1]
            if radius < 1e-5:
                return None, -1
            if not (1e-6 < semi_angle < np.pi/2 - 1e-6):
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
            else:
                # v_min_new = v_min
                # v_min_new = 0
                pass

            # v_min = v_min_new    
            # v_max = v_max - v_min

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


        elif surface_type == 'torus':
            # scalar[0] = major_radius, scalar[1] = minor_radius
            scalar_params = [surface_dict['scalar'][0], surface_dict['scalar'][1]]
            if scalar_params[0] < 1e-5 or scalar_params[1] < 1e-5:
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

            ic('u_center: ', u_center, 'v_center: ', v_center, 'u_diff: ', u_diff, 'v_diff: ', v_diff)
            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            sin_v_center, cos_v_center = np.sin(v_center), np.cos(v_center)

            c, s = np.cos(-u_center), np.sin(-u_center)

            # Rz = np.array([[c, -s, 0],
            #             [s,  c, 0],
            #             [0,  0, 1]], dtype=np.float64)

            # X_new = Rz @ (X / np.linalg.norm(X))
            # X = X_new


            UV = np.array([sin_u_center, cos_u_center, u_half / np.pi, sin_v_center, cos_v_center, v_half / np.pi, 0, 0], dtype=np.float64)


        elif surface_type == 'bspline_surface':
            # Skip bspline surfaces for now (variable dimension)
            raise NotImplementedError("B-spline surfaces are not supported yet")
        
        scalar_params = np.array(scalar_params, dtype=np.float64)
        
        # Pad scalar parameters to max_scalar_dim
        # if len(scalar_params) < self.max_scalar_dim:
        #     padding = np.zeros(self.max_scalar_dim - len(scalar_params), dtype=np.float64)
        #     scalar_params = np.concatenate([scalar_params, padding])
        
        # Concatenate all parameters: P + D + UV + scalar
        params = np.concatenate([P, D, X, UV, scalar_params])
        # Just control the max value of params to be 10
        if np.max(np.abs(params)) > 10:
            return None, -1
        assert np.allclose(params, self.postprocess_funcs[surface_type](self.preprocess_funcs[surface_type](params))), f"type: {surface_type}, params: {params}, postprocess: {self.postprocess_funcs[surface_type](self.preprocess_funcs[surface_type](params))}"
        # Do the log processing of radius
        params = self.preprocess_funcs[surface_type](params)
        assert len(params) == self.base_dim + SCALAR_DIM_MAP[surface_type], f"surface {surface_type} params length {len(params)} != base_dim {self.base_dim}"
        return params.astype(np.float32), surface_type_idx
    

    def _recover_surface(self, params, surface_type_idx):
        """
        Recover a surface from parameter vector.
        """
        params = params.astype(np.float64)
        SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}
        surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx, -1)
        # print(surface_type, params, f'len params: {len(params)}')
        # Apply exp to radius
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
            
            u_min, u_max = u_center - u_half, u_center + u_half
            if np.abs(np.abs(u_max - u_min) - 2 * np.pi) < 1e-4:
                # A full loop, make sure distance less than 2pi
                u_max = u_min + 2 * np.pi - 1e-4

            v_min = 0.0
            v_max = height

            radius = scalar_params[0]
            scalar = [radius]

        elif surface_type == 'cone':
            sin_u_center, cos_u_center, u_half, v_center, v_half = UV[:5]
            uc = np.arctan2(sin_u_center, cos_u_center)
            u_half = np.clip(u_half, 0, 1 - 1e-5) * np.pi
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
            u_min, u_max = u_center - u_half, u_center + u_half

            v_center = np.arctan2(sin_v_center, cos_v_center)
            v_half = np.clip(v_half, 0, 1 - 1e-5) * np.pi
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

            u_min, u_max = u_c - u_h, u_c + u_h
            v_min, v_max = v_c - v_h, v_c + v_h



            assert len(scalar_params) == 1, f"Wrong scalar number of torus, should be 2 but got {len(scalar_params)}"
            radius = scalar_params[0]   
            scalar = [radius]
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
        mask = np.zeros(self.max_num_surfaces, dtype=np.float64)



        try:
            with open(json_path, 'r') as f:
                surfaces_data = json.load(f)
        except json.JSONDecodeError:
            print(f"JSONDecodeError: {json_path}")
            with open('./assets/abnormal_json.csv', 'w') as f:
                f.write(json_path + '\n')
            return torch.from_numpy(all_params).float(), torch.from_numpy(all_types).long(), torch.from_numpy(mask).float(), torch.from_numpy(all_shifts).float(), torch.from_numpy(all_rotations).float(), torch.from_numpy(all_scales).float()
        
        # Here we load the .npz which stores the bspline approximation and the points data. 
        # npz_data = np.load(json_path.replace('.json', '.npz'))

        # When the params are larger than 10, instead of dropping them we use the approx bspline instead.
        # Parse each surface
        for i, surface_dict in enumerate(surfaces_data):
            # print(i)
            if i >= self.max_num_surfaces:
                break
            
            try:
                # Catch RuntimeWarnings (overflow, invalid value) and convert to exceptions
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    # print(surface_dict)
                    # if i == 6:
                    #     print()
                    params, surface_type_idx = self._parse_surface(surface_dict)

                    # Transform to canonical space.

                    if self.canonical and surface_type_idx != -1:
                        surface_str = self._recover_surface(params, surface_type_idx)
                        # print(i, params[9:])
                        surface_canonical, shift, rotation, scale = to_canonical(surface_str)
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
                        mask[i] = 1.0
            except (KeyError, IndexError, NotImplementedError, RuntimeWarning) as e:
                # Skip invalid surfaces (leave as zeros with mask=0)
                # print(f"Warning: Skipping surface {i} in {json_path}: {e}")
                continue
            
        
        # Convert to tensors
        params_tensor = torch.from_numpy(all_params).float() 
        types_tensor = torch.from_numpy(all_types).long()
        mask_tensor = torch.from_numpy(mask).float()
        
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
    dataset = dataset_compound(sys.argv[1], canonical=True)
    for i in tqdm(range(len(dataset))):
        _ = dataset[i]
        # t = time.time()
        # try:
        #     params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales = dataset[i]
        # except Exception as e:
        #     print(f"Error: {e}")
        #     print(i)
        #     print(dataset.json_names[i])
        #     continue
        # print(f"Time taken: {time.time() - t} seconds")