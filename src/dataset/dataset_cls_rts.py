import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from einops import rearrange


class SurfaceClassificationAndRegressionDataset(Dataset):
    def __init__(self, data_path, data_dir, replication=1, transform=None, is_train=True, res=32):
        """
        Args:
            data_path: Path to the .pkl file containing the list of filenames (e.g., train.pkl)
            data_dir: Directory containing the individual .pkl files
            replication: Number of times to replicate the dataset
            transform: Optional transform to apply to the data
            is_train: Whether this is training data
            res: Resolution for surface points
        """
        super().__init__()
        
        # Load the file list
        with open(data_path, 'rb') as f:
            self.file_names = pickle.load(f)
        
        self.data_dir = data_dir
        self.datasize = len(self.file_names)
        
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
        points = surface_data['points']
        class_label = surface_data['class_label']
        scaling = surface_data['scaling']
        rotation = surface_data['rotation']
        translation = surface_data['translation']
        cone_min_axis = surface_data['cone_min_axis'] # TODO: check if this is correct
        bspline_control_points = surface_data['bspline_control_points'] # TODO: this contains NaN, please check

        
        # rts = np.concatenate([translation, rotation, scaling], axis=0)
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









