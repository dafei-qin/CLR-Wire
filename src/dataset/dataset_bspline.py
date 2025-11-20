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
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.tools.surface_to_canonical_space import compute_rotation_matrix


class dataset_bspline(Dataset):
    def __init__(
        self,
        path_file: str,
        data_dir_override: str = "",
        num_surfaces: int = -1,
        replica: int = 1,
        max_degree: int = 3,
        max_num_u_knots: int = 64,
        max_num_v_knots: int = 32,
        max_num_u_poles: int = 64,
        max_num_v_poles: int = 32,
        canonical: bool = False,
        canonical_samples_per_interval: int = 4,
    ):
        """
        Args:
            data_path: Path to directory containing bspline files
        """
        if data_dir_override != "":
            self.data_dir = data_dir_override
            self.data_names = sorted([
            str(p) for p in Path(self.data_dir).rglob("*.npy")
        ])
        else:
            self.data_names = open(path_file, 'r').readlines()
        if num_surfaces != -1:
            self.data_names = self.data_names[:num_surfaces]

        self.replica = replica
        self.max_degree = max_degree
        self.max_num_u_knots = max_num_u_knots
        self.max_num_v_knots = max_num_v_knots
        self.max_num_u_poles = max_num_u_poles
        self.max_num_v_poles = max_num_v_poles
        self.canonical = canonical
        self.canonical_samples_per_interval = max(1, int(canonical_samples_per_interval))
        self._transform_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return len(self.data_names) * self.replica



    def load_data(self, data_path):
        data_vec = np.load(data_path.strip(), allow_pickle=False)
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
        if num_poles_u >= self.max_num_u_poles:
            valid = False
        if num_poles_v >= self.max_num_v_poles:
            valid = False
        if num_knots_u >= self.max_num_u_knots:
            valid = False
        if num_knots_v >= self.max_num_v_knots:
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

    def _gaps_to_knots(self, knots_gap: np.ndarray, num_knots: int) -> np.ndarray:
        gaps = np.asarray(knots_gap[:num_knots], dtype=np.float64)
        if gaps.size == 0:
            return gaps
        gaps = np.clip(gaps, 0.0, None)
        knots = np.cumsum(gaps)
        if knots.size > 0 and knots[-1] > 0:
            knots = knots / knots[-1]
        return knots

    def _build_param_samples(self, knots: np.ndarray) -> np.ndarray:
        if knots.size <= 1:
            return knots.copy()
        samples: List[float] = []
        for idx in range(len(knots) - 1):
            start = float(knots[idx])
            end = float(knots[idx + 1])
            if np.isclose(start, end):
                continue
            local = np.linspace(start, end, self.canonical_samples_per_interval, dtype=np.float64)
            samples.extend(local.tolist())
        if not samples:
            samples = knots.tolist()
        return np.asarray(samples, dtype=np.float64)

    def _build_occ_surface(
        self,
        u_degree: int,
        v_degree: int,
        num_poles_u: int,
        num_poles_v: int,
        u_knots: np.ndarray,
        v_knots: np.ndarray,
        u_mults: np.ndarray,
        v_mults: np.ndarray,
        poles: np.ndarray,
        is_u_periodic: bool,
        is_v_periodic: bool,
    ) -> Geom_BSplineSurface:
        ctrl_pts = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
        weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
        for i in range(num_poles_u):
            for j in range(num_poles_v):
                x, y, z, w = poles[i, j]
                ctrl_pts.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
                weights.SetValue(i + 1, j + 1, float(max(w, 1e-8)))

        occ_u_knots = TColStd_Array1OfReal(1, len(u_knots))
        for idx, knot in enumerate(u_knots):
            occ_u_knots.SetValue(idx + 1, float(knot))
        occ_v_knots = TColStd_Array1OfReal(1, len(v_knots))
        for idx, knot in enumerate(v_knots):
            occ_v_knots.SetValue(idx + 1, float(knot))

        occ_u_mults = TColStd_Array1OfInteger(1, len(u_mults))
        for idx, mult in enumerate(u_mults):
            occ_u_mults.SetValue(idx + 1, int(mult))
        occ_v_mults = TColStd_Array1OfInteger(1, len(v_mults))
        for idx, mult in enumerate(v_mults):
            occ_v_mults.SetValue(idx + 1, int(mult))

        return Geom_BSplineSurface(
            ctrl_pts,
            weights,
            occ_u_knots,
            occ_v_knots,
            occ_u_mults,
            occ_v_mults,
            int(u_degree),
            int(v_degree),
            bool(is_u_periodic),
            bool(is_v_periodic),
        )

    def _sample_surface_grid(
        self,
        surface: Geom_BSplineSurface,
        u_params: np.ndarray,
        v_params: np.ndarray,
    ) -> np.ndarray:
        grid = np.zeros((len(v_params), len(u_params), 3), dtype=np.float64)
        for vi, v_val in enumerate(v_params):
            for ui, u_val in enumerate(u_params):
                point = surface.Value(float(u_val), float(v_val))
                grid[vi, ui] = [point.X(), point.Y(), point.Z()]
        return grid

    def _neighbor_indices(self, index: int, length: int):
        if length <= 1:
            return None
        if length == 2:
            return 0, 1
        prev_idx = max(index - 1, 0)
        next_idx = min(index + 1, length - 1)
        if prev_idx == next_idx:
            if next_idx < length - 1:
                next_idx += 1
            elif prev_idx > 0:
                prev_idx -= 1
        return prev_idx, next_idx

    # def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
    #     axis = axis / np.linalg.norm(axis)
    #     x, y, z = axis
    #     c = np.cos(angle)
    #     s = np.sin(angle)
    #     C = 1 - c
    #     return np.array(
    #         [
    #             [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    #             [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    #             [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    #         ],
    #         dtype=np.float64,
    #     )

    # def _rotation_matrix_to_z(self, normal_vec: np.ndarray) -> np.ndarray:
    #     target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    #     source = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
    #     dot = np.clip(np.dot(source, target), -1.0, 1.0)
    #     if np.isclose(dot, 1.0):
    #         return np.eye(3, dtype=np.float64)
    #     if np.isclose(dot, -1.0):
    #         axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    #         if abs(source[0]) > 0.9:
    #             axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    #         axis = axis - axis.dot(source) * source
    #         axis_norm = np.linalg.norm(axis)
    #         if axis_norm < 1e-8:
    #             return np.eye(3, dtype=np.float64)
    #         axis = axis / axis_norm
    #         return self._axis_angle_to_matrix(axis, np.pi)
    #     axis = np.cross(source, target)
    #     s = np.linalg.norm(axis)
    #     axis = axis / (s + 1e-12)
    #     k = np.array(
    #         [
    #             [0.0, -axis[2], axis[1]],
    #             [axis[2], 0.0, -axis[0]],
    #             [-axis[1], axis[0], 0.0],
    #         ],
    #         dtype=np.float64,
    #     )
    #     R = np.eye(3, dtype=np.float64) + k * s + (k @ k) * ((1 - dot) / (s ** 2 + 1e-12))
    #     return R

    def _compute_centroid_and_normal(
        self,
        u_degree,
        v_degree,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_gap,
        v_knots_gap,
        u_mults_list,
        v_mults_list,
        poles,
    ):
        if self.canonical_samples_per_interval < 1:
            return None, None
        u_knots = self._gaps_to_knots(u_knots_gap, num_knots_u)
        v_knots = self._gaps_to_knots(v_knots_gap, num_knots_v)
        if u_knots.size == 0 or v_knots.size == 0:
            return None, None
        surface = self._build_occ_surface(
            u_degree,
            v_degree,
            num_poles_u,
            num_poles_v,
            u_knots,
            v_knots,
            u_mults_list[:num_knots_u],
            v_mults_list[:num_knots_v],
            poles,
            is_u_periodic,
            is_v_periodic,
        )
        u_params = self._build_param_samples(u_knots)
        v_params = self._build_param_samples(v_knots)
        if u_params.size == 0 or v_params.size == 0:
            return None, None
        grid = self._sample_surface_grid(surface, u_params, v_params)
        centroid = grid.reshape(-1, 3).mean(axis=0)

        # Find the normal vector of the surface
        center_v = grid.shape[0] // 2
        center_u = grid.shape[1] // 2
        neighbors_u = self._neighbor_indices(center_u, grid.shape[1])
        neighbors_v = self._neighbor_indices(center_v, grid.shape[0])
        if neighbors_u is None or neighbors_v is None:
            return centroid, np.array([0.0, 0.0, 1.0], dtype=np.float64)
        prev_u, next_u = neighbors_u
        prev_v, next_v = neighbors_v
        u_vec = grid[center_v, next_u] - grid[center_v, prev_u]
        v_vec = grid[next_v, center_u] - grid[prev_v, center_u]
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        normal_vec = np.cross(u_vec, v_vec)
        normal_vec = normal_vec / np.linalg.norm(normal_vec)
        return centroid, normal_vec, u_vec, v_vec

    def _canonicalize_poles(
        self,
        u_degree,
        v_degree,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_list,
        v_mults_list,
        poles,
    ):
        centroid, normal_vec, u_vec, v_vec = self._compute_centroid_and_normal(
            u_degree,
            v_degree,
            num_poles_u,
            num_poles_v,
            num_knots_u,
            num_knots_v,
            is_u_periodic,
            is_v_periodic,
            u_knots_list,
            v_knots_list,
            u_mults_list,
            v_mults_list,
            poles,
        )
        if centroid is None or normal_vec is None:
        # if True:
            rotation = np.eye(3, dtype=np.float64)
            shift = np.zeros(3, dtype=np.float64)
            return poles, rotation, shift
        rotation = compute_rotation_matrix(normal_vec, u_vec)
        # rotation = np.eye(3, dtype=np.float64)

        poles_xyz = poles[..., :3] - centroid

        reshaped = poles_xyz.reshape(-1, 3)
        rotated = reshaped @ rotation.T
        poles[..., :3] = rotated.reshape(poles_xyz.shape)
        return poles, rotation, centroid

    def __getitem__(self, idx):
        idx = idx % len(self.data_names)
        data_path = self.data_names[idx]

        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = self.load_data(data_path)
        poles[..., 3:] = poles[..., 3:] / min(poles[..., 3:].max(), 1)
        u_knots_list = self.normalize_knots(u_knots_list) 
        v_knots_list = self.normalize_knots(v_knots_list)
        poles, length = self.normalize_poles(poles)
        rotation_matrix = np.eye(3, dtype=np.float64)
        shift_vec = np.zeros(3, dtype=np.float64)
        if self.canonical and valid:
            try:
                poles, rotation_matrix, shift_vec = self._canonicalize_poles(
                    u_degree,
                    v_degree,
                    num_poles_u,
                    num_poles_v,
                    num_knots_u,
                    num_knots_v,
                    is_u_periodic,
                    is_v_periodic,
                    u_knots_list,
                    v_knots_list,
                    u_mults_list,
                    v_mults_list,
                    poles,
                )
            except Exception as exc:
                warnings.warn(f"Canonical transform failed for {data_path}: {exc}")
                poles = np.array(poles, copy=False)
                rotation_matrix = np.eye(3, dtype=np.float64)
                shift_vec = np.zeros(3, dtype=np.float64)
        rotation_matrix = rotation_matrix.astype(np.float32)
        shift_vec = shift_vec.astype(np.float32)
        self._transform_cache[idx] = (rotation_matrix, shift_vec, length)

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

    def get_transform(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        base_idx = idx % len(self.data_names)
        transform = self._transform_cache.get(base_idx)
        if transform is None:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1
        rotation, shift, length = transform
        return rotation.copy(), shift.copy(), length


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = dataset_bspline(path_file = "assets/all_bspline_paths_test.txt")

    # def print_knots_mults(index):
    #     u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset[index]
    #     u_knots_list = np.insert(u_knots_list, 0, 0)
    #     v_knots_list = np.insert(v_knots_list, 0, 0)
    #     U_info = np.stack([u_knots_list, u_mults_list])
    #     V_info = np.stack([v_knots_list, v_mults_list])
    #     print(f"U_info: {U_info}")
    #     print(f"V_info: {V_info}")
    print(len(dataset))
    counter = 0
    
    for i in tqdm(range(len(dataset))):
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset[i]
        # print(poles[..., 3].max())
        if poles[..., 3].max() > 1 + 1e-3:
            counter += 1

    print(counter, '/', len(dataset))
