import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from einops import rearrange

import math
class SurfaceClassificationAndRegressionDataset(Dataset):
    def __init__(self, data_path, data_dir, replication=1, transform=None, is_train=True, res=32, num_nearby=20):
        """
        Args:
            data_path: Path to the .pkl file containing the list of filenames (e.g., train.pkl)
            data_dir: Directory containing the individual .pkl files
            replication: Number of times to replicate the dataset
            transform: Optional transform to apply to the data
            is_train: Whether this is training data
            res: Resolution for surface points
            num_nearby: Maximum number of nearby surfaces
        """
        super().__init__()
        
        # Load the file list
        with open(data_path, 'rb') as f:
            self.file_names = pickle.load(f)
        
        self.data_dir = data_dir
        self.datasize = len(self.file_names)
        self.num_nearby = num_nearby
        
        self.transform = transform
        self.replica = replication
        self.is_train = is_train
        self.res = res

    def __len__(self):
        return self.datasize * self.replica

    def __getitem__(self, idx):
        idx = idx % self.datasize
        
        # Load individual surface data
        file_path = os.path.join(self.data_dir, self.file_names[idx])
        try:
            with open(file_path, 'rb') as f:
                surface_data = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")
        
        # Extract data from the loaded surface
        type = surface_data['type']
        points = surface_data['points']
        bbox = surface_data['bbox']
        adj_mask = surface_data['adj_mask']
        nearby_list = surface_data['nearby']
        
        points = torch.from_numpy(points).float()
        # Process nearby surfaces
        nearby_type_unpad = torch.from_numpy(np.array([nearby['type'] for nearby in nearby_list], dtype=np.int32))
        nearby_points_unpad = torch.from_numpy(np.array([nearby['points'] for nearby in nearby_list], dtype=np.float32))
        nearby_bbox_unpad = torch.from_numpy(np.array([nearby['bbox'] for nearby in nearby_list], dtype=np.float32))
        
        # Create padding mask and pad nearby data
        padding_mask = torch.zeros(self.num_nearby, dtype=torch.bool)
        padding_mask[len(nearby_list):] = True
        
        nearby_type = torch.zeros(self.num_nearby, dtype=torch.long)
        nearby_type[:len(nearby_list)] = nearby_type_unpad
        
        nearby_points = torch.zeros(self.num_nearby, points.shape[0], points.shape[1], 3, dtype=torch.float)
        nearby_points[:len(nearby_list)] = nearby_points_unpad
        
        nearby_bbox = torch.zeros(self.num_nearby, 2, 3, dtype=torch.float)
        nearby_bbox[:len(nearby_list)] = nearby_bbox_unpad
        
        points_together = torch.cat([points.unsqueeze(0), nearby_points], dim=0)
        # Resize points if necessary
        if points.shape[0] != self.res:
            if points.shape[0] < self.res:
                
                assert not self.res % points_together.shape[1], f'res % points_together.shape[1] != 0, {self.res} % {points_together.shape[0]} != 0'
                factor = self.res // points_together.shape[1]
                factor = math.log(factor, 2)
                assert math.isclose(factor, int(factor)), f'factor is not an integer, {factor}'
                points_together = rearrange(points_together, 'b h w c -> b c h w')
                points_together = torch.nn.functional.interpolate(points_together, scale_factor=factor, mode='bilinear')
                points_together = rearrange(points_together, 'b c h w -> b h w c')
            else:
                assert not points_together.shape[1] % self.res, f'points_together.shape[1] % res != 0, {points_together.shape[1]} % {self.res} != 0'
                points_together = points_together[:, ::points_together.shape[1] // self.res, ::points_together.shape[2] // self.res]

        
        # Apply transform if provided
        if self.transform:
            points_together = self.transform(points_together)

        points = points_together[0].float()
        nearby_points = points_together[1:].float()
        
        return {
            'points': points,
            'type': torch.tensor([type]).long(),
            'bbox': torch.from_numpy(bbox).float(),
            'adj_mask': torch.from_numpy(adj_mask).bool(),
            'nearby_type': nearby_type,
            'nearby_points': nearby_points,
            'nearby_bbox': nearby_bbox,
            'padding_mask': padding_mask,
        }








