# This is the dataset for the vae_v1.py

# It is a dataset of surface parameters, with the surface type as the label.

# The surface parameters are padded to the max raw dimension for each surface type.

# The surface type is a one-hot encoded vector.


import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from einops import rearrange



class V1(Dataset):
    pass

class V1_random(Dataset):
    def __init__(self):
        super().__init__()
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
