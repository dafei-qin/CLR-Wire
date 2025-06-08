import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from src.vae.layers import BSplineSurfaceLayer
import os



class surface_dataset_flow_cp(Dataset):
    def __init__(self, cp_path, pc_path=None, mask_prob=0, mask_pattern=None, transform_cp=None, transform_pc=None, is_train=True, replication=1, num_samples=32):
        # Handle multiple dataset paths (both lists and tuples)
        if isinstance(cp_path, (list, tuple)):
            # Load and concatenate multiple datasets
            datasets = []
            shapes = []
            
            for i, path in enumerate(cp_path):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Dataset file not found: {path}")
                    
                data = np.load(path).astype(np.float32)
                datasets.append(data)
                shapes.append(data.shape)
                print(f"Dataset {i+1}: {path} - Shape: {data.shape}")
                
            # Validate that all datasets have compatible shapes (same dimensions except batch size)
            if len(set(shape[1:] for shape in shapes)) > 1:
                raise ValueError(f"All datasets must have the same shape except for the batch dimension. Got shapes: {shapes}")
                
            self.data = np.concatenate(datasets, axis=0)
            print(f"Successfully loaded and concatenated {len(cp_path)} datasets with total {len(self.data)} samples")
            print(f"Final dataset shape: {self.data.shape}")
        else:
            # Single dataset path (backward compatibility)
            if not os.path.exists(cp_path):
                raise FileNotFoundError(f"Dataset file not found: {cp_path}")
            self.data = np.load(cp_path).astype(np.float32)
            print(f"Loaded single dataset with {len(self.data)} samples from {cp_path}")
            
        # Handle multiple point cloud paths (both lists and tuples)
        if pc_path is not None:
            if isinstance(pc_path, (list, tuple)):
                # Validate that we have the same number of PC files as CP files
                if isinstance(cp_path, (list, tuple)) and len(pc_path) != len(cp_path):
                    raise ValueError(f"Number of point cloud files ({len(pc_path)}) must match number of control point files ({len(cp_path)})")
                    
                # Load and concatenate multiple point cloud datasets
                pc_datasets = []
                pc_shapes = []
                
                for i, path in enumerate(pc_path):
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Point cloud file not found: {path}")
                        
                    pc_data = np.load(path)
                    pc_datasets.append(pc_data)
                    pc_shapes.append(pc_data.shape)
                    print(f"Point cloud {i+1}: {path} - Shape: {pc_data.shape}")
                    
                # Validate PC shapes compatibility
                if len(set(shape[1:] for shape in pc_shapes)) > 1:
                    raise ValueError(f"All point cloud datasets must have the same shape except for the batch dimension. Got shapes: {pc_shapes}")
                    
                self.pc = np.concatenate(pc_datasets, axis=0)
                print(f"Successfully loaded and concatenated {len(pc_path)} point cloud datasets")
                print(f"Final point cloud shape: {self.pc.shape}")
                
                # Validate that PC and CP datasets have matching batch sizes
                if len(self.pc) != len(self.data):
                    raise ValueError(f"Point cloud dataset size ({len(self.pc)}) must match control point dataset size ({len(self.data)})")
                    
            else:
                # Single point cloud path
                if not os.path.exists(pc_path):
                    raise FileNotFoundError(f"Point cloud file not found: {pc_path}")
                self.pc = np.load(pc_path)
                
                # Validate that PC and CP datasets have matching batch sizes
                if len(self.pc) != len(self.data):
                    raise ValueError(f"Point cloud dataset size ({len(self.pc)}) must match control point dataset size ({len(self.data)})")
        else:
            self.pc = None
            self.cp2surfaceLayer = BSplineSurfaceLayer(resolution=num_samples, device='cpu')
            
        self.transform_cp = transform_cp
        self.transform_pc = transform_pc
        self.is_train = is_train
        self.replication = replication
        self.num_samples = num_samples
        self.mask_pattern = mask_pattern
        self.mask_prob = mask_prob
        self.replica = replication

    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return int(len(self.data) * self.replica) 

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        cp = self.data[idx]
        if self.transform_cp is not None:
            cp = self.transform_cp(cp)
        if torch.rand(1) < self.mask_prob:
            mask = self.mask_pattern
        else:
            mask = np.zeros_like(cp)[..., 0:1]
        if self.pc is not None:
            pc = self.pc[idx]
        else:
            pc = self.cp2surfaceLayer(torch.from_numpy(cp).float().to(self.cp2surfaceLayer.device).unsqueeze(0)).squeeze(0)
        if self.transform_pc is not None:
            pc = self.transform_pc(pc)

        return {'data': cp, 'mask': mask, 'pc': pc.squeeze()}