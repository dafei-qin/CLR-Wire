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
    def __init__(self, data_path: str, replica=1, max_degree=3, max_num_u_knots=64, max_num_v_knots=32, max_num_u_poles=64, max_num_v_poles=32):
        """
        Args:
            data_path: Path to directory containing bspline files
        """
        self.data_path = Path(data_path)
        self.data_names = sorted([
            str(p) for p in self.data_path.rglob("*.npy")
        ])

        self.replica = replica
        self.max_degree = max_degree
        self.max_num_u_knots = max_num_u_knots
        self.max_num_v_knots = max_num_v_knots
        self.max_num_u_poles = max_num_u_poles
        self.max_num_v_poles = max_num_v_poles

    def __len__(self):
        return len(self.data_names) * self.replica



    def load_data(self, data_path):
        data_vec = np.load(data_path, allow_pickle=False)
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic = map(int, data_vec[:8])
        u_knots_list = np.array(data_vec[8 : 8 + num_knots_u])
        v_knots_list = np.array(data_vec[8 + num_knots_u : 8 + num_knots_u + num_knots_v])
        u_mults_list = np.array(data_vec[8 + num_knots_u + num_knots_v : 8 + num_knots_u + num_knots_v + num_knots_u])
        v_mults_list = np.array(data_vec[8 + num_knots_u + num_knots_v + num_knots_u : 8 + num_knots_u + num_knots_v + num_knots_u + num_knots_v])
        
        
        poles = np.array(data_vec[8 + num_knots_u + num_knots_v + num_knots_u + num_knots_v :])
        poles = poles.reshape(num_poles_u, num_poles_v, 4)

        valid = True
        if u_degree > self.max_degree:
            valid = False
        if v_degree > self.max_degree:
            valid = False
        if num_poles_u > self.max_num_u_poles:
            valid = False
        if num_poles_v > self.max_num_v_poles:
            valid = False
        if num_knots_u > self.max_num_u_knots:
            valid = False
        if num_knots_v > self.max_num_v_knots:
            valid = False

        return u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid


    def mode_numpy(self, arr):
        vals, counts = np.unique(arr, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

    def normalize_knots(self, knots_list):
        knots_min = min(knots_list)
        knots_max = max(knots_list)
        assert knots_min == knots_list[0]
        assert knots_max == knots_list[-1]
        # return [(i - knots_min) / (knots_max - knots_min) for i in knots_list]
        knots_normalized = (knots_list - knots_min) / (knots_max - knots_min)
        knots_gap = np.diff(knots_normalized)
        knots_gap_mode = self.mode_numpy(knots_gap)
        knots_gap = knots_gap  / knots_gap_mode
        knots_gap = knots_gap / max(knots_gap)
        knots_gap = np.insert(knots_gap, 0, 0)
        return knots_gap

    def normalize_poles(self, poles):
        poles_xyz = poles[..., :3]
        poles_xyz_min = poles_xyz.min(axis=(0, 1))
        poles_xyz_max = poles_xyz.max(axis=(0, 1))
        length = max(poles_xyz_max - poles_xyz_min)
        poles_xyz = (poles_xyz - poles_xyz_min) / length
        poles[..., :3] = poles_xyz
        return poles, length

    def __getitem__(self, idx):
        idx = idx % len(self.data_names)
        data_path = self.data_names[idx]

        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = self.load_data(data_path)

        u_knots_list = self.normalize_knots(u_knots_list) 
        v_knots_list = self.normalize_knots(v_knots_list)
        poles, length = self.normalize_poles(poles)

        u_knots_list_padded = np.zeros(self.max_num_u_knots)
        v_knots_list_padded = np.zeros(self.max_num_v_knots)
        u_mults_list_padded = np.zeros(self.max_num_u_knots)
        v_mults_list_padded = np.zeros(self.max_num_v_knots)
        poles_padded = np.zeros((self.max_num_u_poles, self.max_num_v_poles, 4))
        if valid:
            u_knots_list_padded[:num_knots_u] = u_knots_list
            v_knots_list_padded[:num_knots_v] = v_knots_list
            u_mults_list_padded[:num_knots_u] = u_mults_list
            v_mults_list_padded[:num_knots_v] = v_mults_list
            poles_padded[:num_poles_u, :num_poles_v, :] = poles
        else:
            pass
        return torch.tensor(u_degree), torch.tensor(v_degree), torch.tensor(num_poles_u), torch.tensor(num_poles_v), torch.tensor(num_knots_u), torch.tensor(num_knots_v), torch.tensor(is_u_periodic), torch.tensor(is_v_periodic), torch.tensor(u_knots_list_padded), torch.tensor(v_knots_list_padded), torch.tensor(u_mults_list_padded), torch.tensor(v_mults_list_padded), torch.tensor(poles_padded), torch.tensor(valid).bool()


if __name__ == '__main__':
    dataset = dataset_bspline(data_path='/home/qindafei/CAD/data/logan_bspline/0/')

    def print_knots_mults(index):
        u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset[index]
        u_knots_list = np.insert(u_knots_list, 0, 0)
        v_knots_list = np.insert(v_knots_list, 0, 0)
        U_info = np.stack([u_knots_list, u_mults_list])
        V_info = np.stack([v_knots_list, v_mults_list])
        print(f"U_info: {U_info}")
        print(f"V_info: {V_info}")
    print(len(dataset))

