import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from einops import rearrange


class SurfaceClassificationAndRegressionDataset(Dataset):
    def __init__(self, data_path, replication=1, transform=None, is_train=True, res=32):
        super().__init__()
        self.data = pickle.load(open(data_path, 'rb'))
        datasize = len(self.data['points'])
        for key, value in self.data.items():
            assert len(value) == datasize, f'data size mismatch, {len(value)} != {datasize} for {key}'
        
        self.points = self.data['points']
        self.class_label = self.data['class_label']
        self.scaling = self.data['scaling']
        self.rotation = self.data['rotation']
        self.translation = self.data['translation']
        self.cone_min_axis = self.data['cone_min_axis'] # TODO: check if this is correct
        self.bspline_control_points = self.data['bspline_control_points'] # TODO: this contains NaN, please check

        
        self.transform = transform
        self.replica = replication
        self.is_train = is_train
        self.res = res

    def __len__(self):
        return len(self.points) * self.replica

    def __getitem__(self, idx):
        idx = idx % len(self.points)
        points = self.points[idx]
        class_label = self.class_label[idx]
        scaling = self.scaling[idx]
        rotation = self.rotation[idx]
        translation = self.translation[idx]
        cone_min_axis = self.cone_min_axis[idx]
        bspline_control_points = self.bspline_control_points[idx]
        
        rts = np.concatenate([scaling, rotation, translation], axis=0)
        rts = torch.from_numpy(rts).float()
        cone_mask = torch.tensor([class_label == 2], dtype=torch.bool)

        if class_label == 0: # plane
            rts_mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool) # z scaling always equals 1
        elif class_label == 1: # cylinder
            rts_mask = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool) # x y scaling always the same
        elif class_label == 2: # cone
            rts_mask = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool) # x y scaling always the same
        elif class_label == 3: # sphere
            rts_mask = torch.tensor([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.bool) # x y z scaling always the same, rotation always [0, 0, 1]
        elif class_label == 4: # torus
            rts_mask = torch.tensor([1, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.bool) # x y z scaling always the same
        elif class_label == 5: # bspline
            rts_mask = torch.zeros(9, dtype=torch.bool) # no rts
        else:
            raise ValueError(f'Invalid class label: {class_label}')
        
        if points.shape[0] != self.res:
            assert not points.shape[0] % self.res, f'points.shape[0] % res != 0, {points.shape[0]} % {self.res} != 0'
            points = points[::points.shape[0] // self.res, ::points.shape[0] // self.res]
        if self.transform:
            points = self.transform(points)
        
        return torch.from_numpy(points).float(), torch.from_numpy(np.array(class_label)).long(), rts, torch.from_numpy(cone_min_axis).float(), torch.from_numpy(bspline_control_points).float(), rts_mask, cone_mask









