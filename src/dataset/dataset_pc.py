"""
Dataset for loading point cloud samples from NPZ files and surface parameters from JSON files.

This dataset loads:
- NPZ files containing sampled points on surfaces (points, normals, masks, graph_nodes, graph_edges)
- JSON files containing surface parameters (using dataset_v1 logic)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings
import json
import sys

# Import dataset_v1 logic
sys.path.insert(0, str(Path(__file__).parent))
from dataset_v1 import (
    dataset_compound, 
    SURFACE_TYPE_MAP, 
    SURFACE_PARAM_SCHEMAS,
    SCALAR_DIM_MAP
)


class PointCloudSurfaceDataset(Dataset):
    """
    Dataset for loading sampled point clouds from NPZ files and surface parameters from JSON files.
    
    Each sample contains:
    - Sampled points on surfaces (from .npz files)
    - Surface parameters (from .json files using dataset_v1 logic)
    - RTS transformations (rotation, translation, scale)
    """
    
    def __init__(self, dataset_dir: str, max_num_surfaces: int = 500, 
                 max_points_per_surface: int = 1024, canonical: bool = False, replica: int = 1):
        """
        Args:
            dataset_dir: Path to directory containing both .npz and .json files
            max_num_surfaces: Maximum number of surfaces per sample for padding
            max_points_per_surface: Maximum number of points per surface for padding
            canonical: Whether to use canonical transformation
        """
        super().__init__()
        print(f'dataset_dir: {dataset_dir}')
        self.dataset_dir = Path(dataset_dir)
        self.max_num_surfaces = max_num_surfaces
        self.max_points_per_surface = max_points_per_surface
        self.canonical = canonical
        self.replica = replica
        
        # Discover all NPZ files in directory and subdirectories
        self.npz_files = sorted([
            str(p) for p in self.dataset_dir.rglob("*.npz")
        ])
        
        if not self.npz_files:
            raise ValueError(f"No NPZ files found in {dataset_dir}")
        
        print(f"Found {len(self.npz_files)} NPZ files in {dataset_dir}")
        
        # Create a dataset_compound instance to handle JSON parsing
        self.json_dataset = dataset_compound(
            json_dir=str(self.dataset_dir),
            max_num_surfaces=max_num_surfaces,
            canonical=canonical
        )
        
        # Build a mapping from JSON file path to index in json_dataset
        self.json_path_to_idx = {
            json_name: idx 
            for idx, json_name in enumerate(self.json_dataset.json_names)
        }
        
        # Match NPZ files with JSON files and ensure all NPZ files have corresponding JSON
        self.paired_files = []
        self.json_indices = []  # Store the corresponding json_dataset index for each paired file
        
        skipped_files = []
        for npz_path in self.npz_files:
            json_path = npz_path.replace('.npz', '.json')
            
            # Check if JSON file exists and is in json_dataset
            if Path(json_path).exists() and json_path in self.json_path_to_idx:
                self.paired_files.append((npz_path, json_path))
                self.json_indices.append(self.json_path_to_idx[json_path])
            else:
                skipped_files.append(npz_path)
        
        if skipped_files:
            warnings.warn(f"Skipped {len(skipped_files)} NPZ files without matching JSON files")
        
        if not self.paired_files:
            raise ValueError(f"No paired NPZ and JSON files found in {dataset_dir}")
        
        print(f"Found {len(self.paired_files)} paired NPZ-JSON files")
        
        # Calculate max param dimension (from dataset_v1)
        self.base_dim = 3 + 3 + 3 + 8  # P + D + X + UV = 17
        self.max_scalar_dim = max(SCALAR_DIM_MAP.values())
        self.max_param_dim = self.base_dim + self.max_scalar_dim  # Should be 19
    
    def __len__(self):
        """Return number of paired files in the dataset."""
        return len(self.paired_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and return data from paired NPZ and JSON files.
        
        Args:
            idx: Index of the file pair
            
        Returns:
            Tuple containing (in order):
            1. sampled_points: (max_num_surfaces, max_points_per_surface, 6) - padded point clouds with normals
            2. shifts: (max_num_surfaces, 3) - translation vectors (RTS part)
            3. rotations: (max_num_surfaces, 3, 3) - rotation matrices (RTS part)
            4. scales: (max_num_surfaces,) - scale values (RTS part)
            5. params: (max_num_surfaces, max_param_dim) - surface parameters
            6. types: (max_num_surfaces,) - surface type indices
            7. mask: (max_num_surfaces,) - binary mask indicating valid surfaces
            
            Order: sampled_points, RTS (shifts, rotations, scales), surface parameters (params, types), mask
        """
        npz_path, json_path = self.paired_files[idx]
        sample_valid = True
        
        # ===== Step 1: Load NPZ file =====
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            
            # Extract data from NPZ
            points_list = npz_data['points']  # List of arrays, each (N_i, 3)
            normals_list = npz_data['normals']  # List of arrays, each (N_i, 3)
            masks_list = npz_data['masks']  # List of arrays, each (N_i,)
            # Note: graph_nodes and graph_edges are also in NPZ but not used per user requirements
            
            # Process each surface's point cloud
            all_sampled_points = np.zeros((self.max_num_surfaces, self.max_points_per_surface, 6), dtype=np.float32)
            
            num_surfaces = min(len(points_list), self.max_num_surfaces)
            
            for i in range(num_surfaces):
                points = points_list[i]  # (N_i, 3)
                normals = normals_list[i]  # (N_i, 3)
                masks = masks_list[i]  # (N_i,)
                
                # Merge points and normals
                point_normals = np.concatenate([points, normals], axis=-1)  # (N_i, 6)
            
                # Filter by mask (keep only mask=1)
                valid_mask = masks == 1
                valid_points = point_normals[valid_mask]  # (N_valid, 6)
                
                # Pad or truncate to max_points_per_surface
                num_valid = len(valid_points)
                if num_valid > 0:
                    if num_valid >= self.max_points_per_surface:
                        # Randomly sample if too many points
                        indices = np.random.choice(num_valid, self.max_points_per_surface, replace=False)
                        all_sampled_points[i] = valid_points[indices]
                    else:
                        # Pad with zeros if too few points
                        all_sampled_points[i, :num_valid] = valid_points
        
        except Exception as e:
            warnings.warn(f"Error loading NPZ file {npz_path}: {e}")
            sample_valid = False
            all_sampled_points = np.zeros((self.max_num_surfaces, self.max_points_per_surface, 6), dtype=np.float32)
        
        # ===== Step 2: Load JSON file using dataset_v1 logic =====
        # Use pre-built index mapping to get surface parameters
        json_idx = self.json_indices[idx]
        params, types, mask, shifts, rotations, scales = self.json_dataset[json_idx]
        shifts = torch.from_numpy(shifts).float() if not torch.is_tensor(shifts) else shifts.float()
        rotations = torch.from_numpy(rotations).float() if not torch.is_tensor(rotations) else rotations.float()
        scales = torch.from_numpy(scales).float() if not torch.is_tensor(scales) else scales.float()
        
        # Convert to 6d rotations
        rotations = rotations[:, :2].reshape(-1, 6)

        # Apply log to scale
        scales = torch.log(scales + 1e-6)
        
        # Convert sampled points to tensor
        sampled_points_tensor = torch.from_numpy(all_sampled_points).float()
        
        if mask.sum() == 0:
            sample_valid = False
        # Return in the order: sampled_points, RTS (shifts, rotations, scales), surface params (params, types), mask
        return (
            sampled_points_tensor,  # 1. Sampled points (max_num_surfaces, max_points_per_surface, 6)
            shifts,              # 2. RTS - shifts (max_num_surfaces, 3)
            rotations,              # 3. RTS - rotations (max_num_surfaces, 3, 3)
            scales.unsqueeze(-1),                  # 4. RTS - scales (max_num_surfaces,)
            params,                  # 5. Surface parameters (max_num_surfaces, max_param_dim)
            types,                   # 6. Surface type indices (max_num_surfaces,)
            mask,                    # 7. Mask (max_num_surfaces,)
            sample_valid,
        )




if __name__ == '__main__':
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--max_num_surfaces', type=int, default=500)
    parser.add_argument('--max_points_per_surface', type=int, default=1024)
    parser.add_argument('--canonical', action='store_true')
    args = parser.parse_args()
    
    # Test PointCloudSurfaceDataset
    print("="*80)
    print("Testing PointCloudSurfaceDataset:")
    print("="*80)
    dataset = PointCloudSurfaceDataset(
        args.dataset_dir, 
        args.max_num_surfaces, 
        args.max_points_per_surface,
        args.canonical
    )
    print(f"Dataset length: {len(dataset)}")
    
    # Test first sample
    if len(dataset) > 0:
        print("\nTesting first sample:")
        sampled_points, shifts, rotations, scales, params, types, mask = dataset[0]
        print(f"Sampled points shape: {sampled_points.shape}")
        print(f"Shifts shape: {shifts.shape}")
        print(f"Rotations shape: {rotations.shape}")
        print(f"Scales shape: {scales.shape}")
        print(f"Params shape: {params.shape}")
        print(f"Types shape: {types.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of valid surfaces: {mask.sum().item()}")
        
        print("\nIterating through dataset:")
        for i in tqdm(range(len(dataset))):
            sampled_points, shifts, rotations, scales, params, types, mask = dataset[i]
