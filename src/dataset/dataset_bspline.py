import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
import json
import warnings
from einops import rearrange
from pathlib import Path
from typing import Dict, List, Tuple



class dataset_bspline(Dataset):
    def __init__(self, data_path: str, replica=1):
        """
        Args:
            data_path: Path to directory containing bspline files
        """
        self.data_names = sorted([
            str(p) for p in self.json_dir.rglob("*.npy")
        ])

        self.replica = replica

    def __len__(self):
        return len(self.data_names) * self.replica



    def preprocess(self, data_vec):
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v = map(int, data_vec[:6])
        u_knots_list = data_vec[6 : 6 + num_knots_u]
        v_knots_list = data_vec[6 + num_knots_u : 6 + num_knots_u + num_knots_v]
        u_mults_list = data_vec[6 + num_knots_u + num_knots_v : 6 + num_knots_u + num_knots_v + num_knots_u]
        v_mults_list = data_vec[6 + num_knots_u + num_knots_v + num_knots_u :]
        poles = data_vec[6 + num_knots_u + num_knots_v + num_knots_u + num_knots_v :]
        
        return u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, u_knots_list, v_knots_list, u_mults_list, v_mults_list

    def __getitem__(self, idx):
        idx = idx % len(self.data_names)
        data_path = self.data_names[idx]
        data = np.load(data_path, allow_pickle=False)


        return data