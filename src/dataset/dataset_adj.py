import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from einops import rearrange


class SurfaceClassificationAndRegressionDataset(Dataset):
    def __init__(self, data_path, replication=1, transform=None, is_train=True, res=32, num_nearby=20):
        super().__init__()
        self.data = pickle.load(open(data_path, 'rb'))
        self.datasize = len(self.data['points'])
        self.types = self.data['types']
        self.points = self.data['points']
        self.bbox = self.data['bbox']
        self.adj_mask = self.data['adj_mask']
        self.nearby = self.data['nearby']
        self.num_nearby = num_nearby
        
        self.transform = transform
        self.replica = replication
        self.is_train = is_train
        self.res = res

    def __len__(self):
        return self.datasize * self.replica

    def __getitem__(self, idx):
        idx = idx % self.datasize
        type = self.types[idx]  
        points = self.points[idx]
        bbox = self.bbox[idx]
        adj_mask = self.adj_mask[idx]
        nearby_list = self.nearby[idx]
        nearby_type_unpad = torch.from_numpy(np.array([nearby['type'] for nearby in nearby_list], dtype=np.int32))
        nearby_points_unpad = torch.from_numpy(np.array([nearby['points'] for nearby in nearby_list], dtype=np.float32))
        nearby_bbox_unpad = torch.from_numpy(np.array([nearby['bbox'] for nearby in nearby_list], dtype=np.float32))
        padding_mask = torch.zeros(self.num_nearby, dtype=torch.bool)
        padding_mask[len(nearby_list):] = True
        nearby_type = torch.zeros(self.num_nearby, dtype=torch.long)
        nearby_type[:len(nearby_list)] = nearby_type_unpad
        nearby_points = torch.zeros(self.num_nearby, points.shape[0], points.shape[1], 3, dtype=torch.float)
        nearby_points[:len(nearby_list)] = nearby_points_unpad
        nearby_bbox = torch.zeros(self.num_nearby, 2, 3, dtype=torch.float)
        nearby_bbox[:len(nearby_list)] = nearby_bbox_unpad
        
        if points.shape[0] != self.res:
            assert not points.shape[0] % self.res, f'points.shape[0] % res != 0, {points.shape[0]} % {self.res} != 0'
            points = points[::points.shape[0] // self.res, ::points.shape[0] // self.res]
            nearby_points = nearby_points[:, ::nearby_points.shape[0] // self.res, ::nearby_points.shape[0] // self.res]
        if self.transform:
            points = self.transform(points)
        
        return {
            'points': torch.from_numpy(points).float(),
            'type': torch.tensor([type]).long(),
            'bbox': torch.from_numpy(bbox).float(),
            'adj_mask': torch.from_numpy(adj_mask).bool(),
            'nearby_type': nearby_type,
            'nearby_points': nearby_points,
            'nearby_bbox': nearby_bbox,
            'padding_mask': padding_mask,
        }








