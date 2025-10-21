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
        self.base_dim = 3 + 3 + 4  # P + D + UV = 10
        
        # Scan all JSON files to determine max_scalar_dim and max_num_surfaces
        # self._calculate_dataset_stats()
        self.max_scalar_dim = max(SCALAR_DIM_MAP.values())
        self.max_param_dim = self.base_dim + self.max_scalar_dim
        
        # # Override max_num_surfaces if specified
        # if self.max_surfaces_per_file is not None:
        #     self.max_num_surfaces = self.max_surfaces_per_file
    
    # def _calculate_dataset_stats(self):
    #     """Calculate max_scalar_dim and max_num_surfaces across all JSON files."""
    #     max_scalar_dim = 0
    #     max_num_surfaces = 0
        
    #     for json_path in self.json_names:
    #         with open(json_path, 'r') as f:
    #             surfaces_data = json.load(f)
            
    #         # Update max number of surfaces
    #         max_num_surfaces = max(max_num_surfaces, len(surfaces_data))
            
    #         # Update max scalar dimension
    #         for surface in surfaces_data:
    #             surf_type = surface.get('type', '')
    #             scalar_dim = self.SCALAR_DIM_MAP.get(surf_type, 0)
    #             if scalar_dim >= 0:  # Ignore bspline_surface (-1)
    #                 max_scalar_dim = max(max_scalar_dim, scalar_dim)
        
    #     self.max_scalar_dim = max_scalar_dim
    #     self.max_num_surfaces = max_num_surfaces
    
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
                    [P(3), D(3), UV(4), scalar(max_scalar_dim)]
            surface_type: Integer representing surface type
        """
        surface_type = surface_dict['type']
        surface_type_idx = SURFACE_TYPE_MAP.get(surface_type, -1)
        
        # Extract P: location (first element of location array)
        P = np.array(surface_dict['location'][0], dtype=np.float32)  # (3,)
        
        # Extract D: direction[0] (first direction vector)
        D = np.array(surface_dict['direction'][0], dtype=np.float32)  # (3,)
        
        # Extract UV: parameter bounds
        UV = np.array(surface_dict['uv'], dtype=np.float32)  # (4,)
        
        # Extract scalar parameters based on surface type
        scalar_params = []
        if surface_type == 'plane':
            # No scalar parameters
            pass
        elif surface_type == 'cylinder':
            # scalar[0] = radius
            scalar_params = [surface_dict['scalar'][0]]
        elif surface_type == 'cone':
            # scalar[0] = semi_angle, scalar[1] = radius
            scalar_params = [surface_dict['scalar'][0], surface_dict['scalar'][1]]
        elif surface_type == 'sphere':
            # scalar[0] = radius
            scalar_params = [surface_dict['scalar'][0]]
        elif surface_type == 'torus':
            # scalar[0] = major_radius, scalar[1] = minor_radius
            scalar_params = [surface_dict['scalar'][0], surface_dict['scalar'][1]]
        elif surface_type == 'bspline_surface':
            # Skip bspline surfaces for now (variable dimension)
            raise NotImplementedError("B-spline surfaces are not supported yet")
        
        scalar_params = np.array(scalar_params, dtype=np.float32)
        
        # Pad scalar parameters to max_scalar_dim
        if len(scalar_params) < self.max_scalar_dim:
            padding = np.zeros(self.max_scalar_dim - len(scalar_params), dtype=np.float32)
            scalar_params = np.concatenate([scalar_params, padding])
        
        # Concatenate all parameters: P + D + UV + scalar
        params = np.concatenate([P, D, UV, scalar_params])
        
        return params, surface_type_idx
    
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
                all_params[i] = params
                all_types[i] = surface_type_idx
                mask[i] = 1.0
            except (KeyError, IndexError, NotImplementedError) as e:
                # Skip invalid surfaces (leave as zeros with mask=0)
                print(f"Warning: Skipping surface {i} in {json_path}: {e}")
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
