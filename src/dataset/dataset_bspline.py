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
        self.data_path = Path(data_path)
        self.data_names = sorted([
            str(p) for p in self.data_path.rglob("*.npy")
        ])

        self.replica = replica

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
        
        if u_degree > 3:
            valid = False
        if v_degree > 3:
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
        return knots_gap

    def normalize_poles(self, poles):
        poles_min = poles.min(axis=(0, 1))
        poles_max = poles.max(axis=(0, 1))
        length = max(poles_max - poles_min)
        poles = (poles - poles_min) / length
        return poles, length
    def __getitem__(self, idx):
        idx = idx % len(self.data_names)
        data_path = self.data_names[idx]

        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = self.load_data(data_path)

        u_knots_list = self.normalize_knots(u_knots_list)
        v_knots_list = self.normalize_knots(v_knots_list)
        poles, length = self.normalize_poles(poles)

        return u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid


if __name__ == '__main__':
    dataset = dataset_bspline(data_path='/home/qindafei/CAD/data/logan_bspline/0/')

    def print_knots_mults(index):
        u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset[index]
        U_info = np.stack([u_knots_list, u_mults_list[1:]])
        V_info = np.stack([v_knots_list, v_mults_list[1:]])
        print(f"U_info: {U_info}")
        print(f"V_info: {V_info}")
    print(len(dataset))

