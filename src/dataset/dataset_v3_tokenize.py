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
from icecream import ic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v2 import dataset_compound, dataset_compound_cache, SURFACE_TYPE_MAP, SURFACE_TYPE_MAP_INV


def quantize_to_codebook(x, min_val, max_val, N):
    """
    将数值 x 在 [min_val, max_val] 范围内均匀量化到 N 个码本中心。
    
    Args:
        x: float 或 array-like，待量化数值（支持标量或 NumPy 数组）
        min_val: float，量化范围下界（含）
        max_val: float，量化范围上界（含）
        N: int，码本数量（N >= 1）

    Returns:
        quantized: 同 x 形状的数值，量化后的值（码本中心）
        indices: 同 x 形状的整数，对应码本索引（0 ~ N-1）

    Notes:
        - 码本中心均匀分布在 [min_val, max_val] 区间内
          例如 N=4, [0,1] → centers = [0.125, 0.375, 0.625, 0.875]
        - 边界外的值会被裁剪到最近的码本（即 clamp 后再量化）
    """
    if N < 1:
        raise ValueError("N must be >= 1")
    if min_val >= max_val:
        raise ValueError("min_val must be < max_val")
    
    x = np.asarray(x, dtype=np.float64)
    
    # 裁剪到 [min_val, max_val] 范围内（避免越界导致索引错误）
    x_clipped = np.clip(x, min_val, max_val)
    
    # 均匀划分为 N 个 bin，每个 bin 宽度
    bin_width = (max_val - min_val) / N
    
    # 码本中心位置：第 i 个中心 = min_val + (i + 0.5) * bin_width, i = 0..N-1
    # 对应索引：i = floor((x - min_val) / bin_width)，但需 clamp 到 [0, N-1]
    # 注意：当 x == max_val 时，(x - min_val)/bin_width == N → 应归入 N-1
    # 使用 np.floor 然后 clip 更稳健
    normalized = (x_clipped - min_val) / bin_width
    indices = np.floor(normalized).astype(int)
    indices = np.clip(indices, 0, N - 1)
    
    # 计算量化值 = 码本中心
    quantized = min_val + (indices + 0.5) * bin_width
    
    # 特殊处理 x == max_val 的情况（理论上被 clip 后不会发生，但浮点误差可能）
    # 若 max_val 恰好落在边界，可确保其映射到最后一个码本
    # 下面这行是防御性处理（通常不需要，但更严谨）
    quantized = np.where(x >= max_val, max_val - bin_width / 2, quantized)
    indices = np.where(x >= max_val, N - 1, indices)
    
    # 若输入是标量，返回标量
    if np.ndim(x) == 0:
        return quantized.item(), indices.item()
    return quantized, indices


class dataset_compound_tokenize(Dataset):

    def __init__(self, json_dir: str, max_num_surfaces: int = 500, canonical: bool = False, detect_closed: bool = False, bspline_fit_threshold: float = 1e-5, codebook_size=1024):

        self.dataset_compound = dataset_compound(json_dir, max_num_surfaces, canonical, detect_closed, bspline_fit_threshold)

        self.codebook_size = codebook_size
        self.max_num_surfaces = max_num_surfaces
        self.canonical = canonical
        self.detect_closed = detect_closed
        self.bspline_fit_threshold = bspline_fit_threshold
    
    def __len__(self):
        return len(self.dataset_compound)
        


    def tokenize(self, surface):
        # Assert input raw surface params, no log on scales
        surface_type = surface['type']
        if surface_type == 'bspline_surface':
            return surface, np.zeros(6, dtype=int) - 1 # Total length = 6
        else:
            uv = np.array(surface['uv'])
            if surface_type == 'plane':
                uv_quantized, codes = quantize_to_codebook(uv, -0.5, 0.5, self.codebook_size)
                surface['uv'] = uv_quantized.tolist()
                codes = np.concatenate([codes.flatten(), np.zeros(2, dtype=int) - 1]) 

            elif surface_type == 'cylinder' or surface_type =='cone':
                u = uv[:2]
                v = uv[2:]
                u_quantized, codes_u = quantize_to_codebook(u / 2 / np.pi, -1, 1, self.codebook_size)
                if u_quantized[1] - u_quantized[0] > 1:
                    u_quantized[1] = u_quantized[0] + 1 - 1e-5
                u_quantized = u_quantized * 2 * np.pi

                v_quantized, code_v = quantize_to_codebook(v[1], 0, 1, self.codebook_size)
                v_quantized = [0, v_quantized]
                codes_v = [-1, code_v]
                surface['uv'] = u_quantized.tolist() + v_quantized
                

                if surface_type == 'cylinder':
            
                    scale_quantized, scale_code = quantize_to_codebook(surface['scalar'][0], 0, 1, self.codebook_size)
                    surface['scalar'] = [float(scale_quantized)]
                    codes = np.concatenate([codes_u, codes_v, [scale_code, -1]])

                elif surface_type == 'cone':
                    scale_quantized_0, scale_code_0 = quantize_to_codebook(surface['scalar'][0] * 2 / np.pi, 0, 1, self.codebook_size) 
                    scale_quantized_0 = scale_quantized_0 / 2 * np.pi
                    scale_quantized_1, scale_code_1 = quantize_to_codebook(surface['scalar'][1], 0, 1, self.codebook_size)
                    surface['scalar'] = [scale_quantized_0, scale_quantized_1]
                    codes = np.concatenate([codes_u, codes_v, [scale_code_0, scale_code_1]])

            elif surface_type == 'sphere':
                u = uv[:2]
                v = uv[2:]
                u_quantized, codes_u = quantize_to_codebook(u / 2 / np.pi, -1, 1, self.codebook_size)
                if u_quantized[1] - u_quantized[0] > 1:
                    u_quantized[1] = u_quantized[0] + 1 - 1e-5
                u_quantized = u_quantized * 2 * np.pi
                v_quantized, codes_v = quantize_to_codebook(v / np.pi * 2, -1, 1, self.codebook_size)
                if v_quantized[1] - v_quantized[0] > 1:
                    v_quantized[1] = v_quantized[0] + 1 - 1e-5
                v_quantized = v_quantized * np.pi / 2

                surface['uv'] = u_quantized.tolist() + v_quantized.tolist()
    
                
                codes = np.concatenate([codes_u, codes_v, np.zeros(2, dtype=int) - 1]) # Total length = 6

            elif surface_type == 'torus':
                u = uv[:2]
                v = uv[2:]
                u_quantized, codes_u = quantize_to_codebook(u / 2 / np.pi, -1, 1, self.codebook_size)
                if u_quantized[1] - u_quantized[0] > 1:
                    u_quantized[1] = u_quantized[0] + 1 - 1e-5
                u_quantized = u_quantized * 2 * np.pi
                v_quantized, codes_v = quantize_to_codebook(v / 2 / np.pi, -1, 1, self.codebook_size)
                if v_quantized[1] - v_quantized[0] > 1:
                    v_quantized[1] = v_quantized[0] + 1 - 1e-5
                v_quantized = v_quantized * 2 * np.pi

                surface['uv'] = u_quantized.tolist() + v_quantized.tolist()
                
                scale_2_quantized, scale_2_code = quantize_to_codebook(surface['scalar'][1], 0, 1, self.codebook_size)

                surface['scalar'] = [1.0, float(scale_2_quantized)]
                codes = np.concatenate([codes_u, codes_v, [-1, scale_2_code]])


            return surface, codes

    def de_tokenize(self, surface, code):
        """
        Map discrete codes back to quantized uv / scalar values.

        Args:
            surface: dict describing the surface (will be modified in place)
            code: array-like of length 6 produced by `tokenize`
        """
        surface_type = surface['type']
        if surface_type == 'bspline_surface':
            return surface  # nothing to recover

        code = np.asarray(code).astype(int)

        def decode(idx, min_val, max_val):
            """Recover the codebook center for a given index."""
            if idx < 0:
                return None
            bin_width = (max_val - min_val) / self.codebook_size
            return min_val + (idx + 0.5) * bin_width

        if surface_type == 'plane':
            uv_vals = [decode(c, -0.5, 0.5) for c in code[:4]]
            surface['uv'] = [float(v) for v in uv_vals if v is not None]
            surface['scalar'] = []

        elif surface_type in ('cylinder', 'cone'):
            u = [decode(code[0], -1, 1), decode(code[1], -1, 1)]
            u = [float(v * 2 * np.pi) for v in u]

            v1 = decode(code[3], 0, 1)
            v = [0.0, float(v1)] if v1 is not None else surface.get('uv', [0.0, 0.0])[2:]

            surface['uv'] = u + v

            if surface_type == 'cylinder':
                scale0 = decode(code[4], 0, 1)
                if scale0 is not None:
                    surface['scalar'] = [float(scale0)]

            elif surface_type == 'cone':
                scale0 = decode(code[4], 0, 1)
                scale1 = decode(code[5], 0, 1)
                scalars = []
                if scale0 is not None:
                    scalars.append(float(scale0 / 2 * np.pi))
                if scale1 is not None:
                    scalars.append(float(scale1))
                surface['scalar'] = scalars

        elif surface_type == 'sphere':
            u = [decode(code[0], -1, 1), decode(code[1], -1, 1)]
            u = [float(v * 2 * np.pi) for v in u]

            v = [decode(code[2], -1, 1), decode(code[3], -1, 1)]
            v = [float(val * np.pi / 2) for val in v]

            surface['uv'] = u + v
            surface['scalar'] = [1.0]

        elif surface_type == 'torus':
            u = [decode(code[0], -1, 1), decode(code[1], -1, 1)]
            u = [float(v * 2 * np.pi) for v in u]

            v = [decode(code[2], -1, 1), decode(code[3], -1, 1)]
            v = [float(val * 2 * np.pi) for val in v]

            surface['uv'] = u + v

            scale2 = decode(code[5], 0, 1)
            if scale2 is not None:
                surface['scalar'] = [1.0, float(scale2)]

        return surface

    def __getitem__(self, idx):

        if self.detect_closed:
            params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales, is_u_closed_tensor, is_v_closed_tensor = self.dataset_compound[idx]
        else:
            params_tensor, types_tensor, mask_tensor, all_shifts, all_rotations, all_scales = self.dataset_compound[idx]

        params_tensor = params_tensor[mask_tensor.bool()]
        types_tensor = types_tensor[mask_tensor.bool()]
        all_shifts = all_shifts[mask_tensor.bool()]
        all_rotations = all_rotations[mask_tensor.bool()]
        all_scales = all_scales[mask_tensor.bool()]
        if self.detect_closed:
            is_u_closed_tensor = is_u_closed_tensor[mask_tensor.bool()]
            is_v_closed_tensor = is_v_closed_tensor[mask_tensor.bool()]

        all_codes = np.zeros((params_tensor.shape[0], 6), dtype=int)
        all_recon_surfaces = []
        for i in range(params_tensor.shape[0]):
            recon_surface = self.dataset_compound._recover_surface(params_tensor[i].numpy(), types_tensor[i].item())
            if recon_surface['type'] == 'bspline_surface':
                # For bspline, no tokenization needed
                all_recon_surfaces.append(recon_surface)
                all_codes[i] = np.zeros(6, dtype=int) - 1
            else:
                recon_surface, code = self.tokenize(recon_surface)
                all_recon_surfaces.append(recon_surface)
                all_codes[i] = code

        if self.detect_closed:
            return (all_recon_surfaces, all_codes, types_tensor, 
                    all_shifts, all_rotations, all_scales,
                    is_u_closed_tensor, is_v_closed_tensor)
        else:
            return (all_recon_surfaces, all_codes, types_tensor, 
                    all_shifts, all_rotations, all_scales)


        


        






