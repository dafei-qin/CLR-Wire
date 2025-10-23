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
from einops import rearrange
from pathlib import Path
from typing import Dict, List, Tuple



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
    
    
    def __init__(self, json_dir: str, max_num_surfaces: int = 200):
        """
        Args:
            json_dir: Path to directory containing JSON files
            max_surfaces_per_file: Maximum number of surfaces per file (None = use actual max)
        """
        super().__init__()
        self.json_dir = Path(json_dir)
        self.max_num_surfaces = max_num_surfaces
        
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
        P = np.array(surface_dict['location'][0], dtype=np.float32)  # (3,)
        
        # Extract D: direction[0] (first direction vector), X: U direction
        D = np.array(surface_dict['direction'][0], dtype=np.float32)  # (3,)
        X = np.array(surface_dict['direction'][1], dtype=np.float32)  # (3,)
        Y = np.cross(D, X)
        # Extract UV: parameter bounds
        UV = np.array(surface_dict['uv'], dtype=np.float32)  # (4,)
        
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
            UV = np.array([u_min, u_max, v_min, v_max, 0, 0, 0, 0], dtype=np.float32) # (8, )

        elif surface_type == 'cylinder':
            # scalar[0] = radius
            scalar_params = [surface_dict['scalar'][0]]
            P = P + D * v_min
            v_max = v_max - v_min
            v_min = 0




            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min

            while u_diff > 2 * np.pi:
                u_diff -= 2 * np.pi
            u_half = 0.5 * (u_diff) / np.pi - 0.5 # (0 - pi) --> (0, 1)

            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            UV = np.array([sin_u_center, cos_u_center, u_half, v_max, 0, 0, 0, 0], dtype=np.float32) # (8, )

        elif surface_type == 'cone':

            semi_angle, radius = surface_dict['scalar'][0], surface_dict['scalar'][1]


            P_min = P + v_min * np.cos(semi_angle) * D
            r_min = radius + v_min * np.sin(semi_angle)
            v_max = v_max - v_min
            v_min = 0
            P = P_min
            radius = r_min

            
            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min
            while u_diff > 2 * np.pi:
                u_diff -= 2 * np.pi
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


            

            u_center = 0.5 * (u_min + u_max)
            v_center = 0.5 * (v_min + v_max)
            u_diff = u_max - u_min
            v_diff = v_max - v_min
            while u_diff > 2 * np.pi:
                u_diff -= 2 * np.pi
            while v_diff > np.pi:
                v_diff -= np.pi
            
            u_half = 0.5 * (u_diff)
            v_half = 0.5 * (v_diff)
            u_center, v_center = canonicalize_vc_uc(u_center, v_center)

            dir_vec = np.array([
                np.cos(v_center) * np.cos(u_center),
                np.cos(v_center) * np.sin(u_center),
                np.sin(v_center)
            ], dtype=np.float32)
            
            u_h_norm = np.clip(u_half / np.pi, 0, 1)        # 在 [-1, 1]
            v_h_norm = np.clip(v_half / (PI/2), 0.0, 1.0)   # 在 [-1, 1]

            UV = np.concatenate([dir_vec, [u_h_norm, v_h_norm, 0, 0, 0]])


        elif surface_type == 'torus':
            # scalar[0] = major_radius, scalar[1] = minor_radius
            scalar_params = [surface_dict['scalar'][0], surface_dict['scalar'][1]]

            u_center = 0.5 * (u_min + u_max)
            u_diff = u_max - u_min
            while u_diff > 2 * np.pi:
                u_diff -= 2 * np.pi
            u_half = 0.5 * (u_diff)
            v_diff = v_max - v_min
            while v_diff > np.pi:
                v_diff -= np.pi
            v_center = 0.5 * (v_min + v_max)
            v_half = 0.5 * (v_diff)


            sin_u_center, cos_u_center = np.sin(u_center), np.cos(u_center)
            sin_v_center, cos_v_center = np.sin(v_center), np.cos(v_center)

            UV = np.array([sin_u_center, cos_u_center, u_half / np.pi, sin_v_center, cos_v_center, v_half / np.pi, 0, 0], dtype=np.float32)


        elif surface_type == 'bspline_surface':
            # Skip bspline surfaces for now (variable dimension)
            raise NotImplementedError("B-spline surfaces are not supported yet")
        
        scalar_params = np.array(scalar_params, dtype=np.float32)
        
        # Pad scalar parameters to max_scalar_dim
        # if len(scalar_params) < self.max_scalar_dim:
        #     padding = np.zeros(self.max_scalar_dim - len(scalar_params), dtype=np.float32)
        #     scalar_params = np.concatenate([scalar_params, padding])
        
        # Concatenate all parameters: P + D + UV + scalar
        params = np.concatenate([P, D, X, UV, scalar_params])
        assert len(params) == self.base_dim + SCALAR_DIM_MAP[surface_type], f"surface {surface_type} params length {len(params)} != base_dim {self.base_dim}"
        return params, surface_type_idx
    

    def _recover_surface(self, params, surface_type_idx):
        """
        Recover a surface from parameter vector.
        """
        SURFACE_TYPE_MAP_INV = {v: k for k, v in SURFACE_TYPE_MAP.items()}
        surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx, -1)
        print(params.shape)
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
            u_half = (u_half + 0.5) * np.pi
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
            u_half = np.clip(u_half, 0, 1) * np.pi
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
            u_half = np.clip(u_half, 0, 1) * np.pi
            u_min, u_max = u_center - u_half, u_center + u_half

            v_center = np.arctan2(sin_v_center, cos_v_center)
            v_half = np.clip(v_half, 0, 1) * np.pi
            v_min, v_max = v_center - v_half, v_center + v_half

            assert len(scalar_params) == 2
            major_radius = scalar_params[0]
            minor_radius = scalar_params[1]
            scalar = [major_radius, minor_radius]

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
            u_h = np.clip(u_h_norm, 0.0, 1.0) * np.pi
            v_h = np.clip(v_h_norm, 0.0, 1.0) * (np.pi/2)

            u_min, u_max = u_c - u_h, u_c + u_h
            v_min, v_max = v_c - v_h, v_c + v_h



            assert len(scalar_params) == 1
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
        
        # Load JSON file
        with open(json_path, 'r') as f:
            surfaces_data = json.load(f)
        
        num_surfaces = len(surfaces_data)
        
        # Initialize arrays for all surfaces (padded)
        all_params = np.zeros((self.max_num_surfaces, self.max_param_dim), dtype=np.float32)
        all_types = np.zeros(self.max_num_surfaces, dtype=np.int64)
        mask = np.zeros(self.max_num_surfaces, dtype=np.float32)
        
        # Parse each surface
        for i, surface_dict in enumerate(surfaces_data):
            if i >= self.max_num_surfaces:
                break
            
            try:
                params, surface_type_idx = self._parse_surface(surface_dict)
                all_params[i, :len(params)] = params
                all_types[i] = surface_type_idx
                mask[i] = 1.0
            except (KeyError, IndexError, NotImplementedError) as e:
                # Skip invalid surfaces (leave as zeros with mask=0)
                # print(f"Warning: Skipping surface {i} in {json_path}: {e}")
                continue
        
        # Convert to tensors
        params_tensor = torch.from_numpy(all_params).float()
        types_tensor = torch.from_numpy(all_types).long()
        mask_tensor = torch.from_numpy(mask).float()
        
        return params_tensor, types_tensor, mask_tensor
    
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
