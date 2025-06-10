import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange


class SurfaceClassificationDataset(Dataset):
    def __init__(self, points_path, class_label_path, replication=1, transform=None, is_train=True, res=32):
        super().__init__()
        self.points = np.load(points_path)
        self.class_label = np.load(class_label_path)
        self.transform = transform
        self.replica = replication
        self.is_train = is_train
        self.res = res

    def __len__(self):
        return len(self.points) * self.replica

    def __getitem__(self, idx):
        idx = idx % len(self.points)
        points = self.points[idx]
        if points.shape[0] != self.res:
            assert not points.shape[0] % self.res, f'points.shape[0] % res != 0, {points.shape[0]} % {self.res} != 0'
            points = points[::points.shape[0] // self.res, ::points.shape[0] // self.res]
        if self.transform:
            points = self.transform(points)
        
        class_label = self.class_label[idx]
        return torch.from_numpy(points).float(), torch.from_numpy(np.array(class_label)).long()









