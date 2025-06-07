import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange
from src.vae.layers import BSplineSurfaceLayer

from src.utils.helpers import (
    get_file_list, get_filename_wo_ext, get_file_list_with_extension, 
)
from src.dataset.dataset_fn import (
    scale_and_jitter_pc, scale_and_jitter_wireframe_set, curve_yz_scale,
    random_viewpoint, hidden_point_removal,
    aug_pc_by_idx,
    surface_scale_and_jitter,
    surface_rotate,
)

class surface_dataset_flow_cp(Dataset):
    def __init__(self, cp_path, pc_path=None, mask_prob=0, mask_pattern=None, transform=None, is_train=True, replication=1, num_samples=32):
        self.data = np.load(cp_path).astype(np.float32)
        if pc_path is not None:
            self.pc = np.load(pc_path)
        else:
            self.pc = None
            self.cp2surfaceLayer = BSplineSurfaceLayer(resolution=num_samples, device='cpu')
        self.transform = transform
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
        if self.transform is not None:
            cp = self.transform(cp)
        if torch.rand(1) < self.mask_prob:
            mask = self.mask_pattern
        else:
            mask = np.zeros_like(cp)[..., 0:1]
        if self.pc is not None:
            pc = pc[idx]
        else:
            pc = self.cp2surfaceLayer(torch.from_numpy(cp).float().to(self.cp2surfaceLayer.device).unsqueeze(0)).squeeze(0)

        return {'data': cp, 'mask': mask, 'pc': pc.squeeze()}