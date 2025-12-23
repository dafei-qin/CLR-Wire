import numpy as np
import torch
from tqdm import tqdm
import sys
import os
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v2 import dataset_compound


if __name__ == '__main__':
    dataset = dataset_compound(
        json_dir = '../data/logan_jsons/abc_test/0009',
        canonical = True,
        detect_closed = True,
        bspline_fit_threshold=1e-3
    )

    all_uvs = defaultdict(list)
    all_ps = defaultdict(list)
    all_xs = defaultdict(list)
    all_ns = defaultdict(list)
    all_scalars = defaultdict(list)
    # for idx in tqdm(range(len(dataset))):
    for idx in tqdm(range(4500, 4700)):
        params_tensor, types_tensor, mask_tensor, shifts, rotations, scales, is_u_closed_tensor, is_v_closed_tensor = dataset[idx]
        valid_params = params_tensor[mask_tensor.bool()]
        valid_types = types_tensor[mask_tensor.bool()]
        for _jdx in range(valid_params.shape[0]):
            # print(valid_params[_jdx].numpy(), valid_types[_jdx].numpy())
            recon_surfaces = dataset._recover_surface(valid_params[_jdx].numpy(), valid_types[_jdx].item())
            if 'uv' in recon_surfaces.keys():
                uv = recon_surfaces['uv']
                uv = np.array(uv)
                # print(uv)
                all_uvs[valid_types[_jdx].item()].append(uv)
                assert uv.max() < 2 * np.pi + 1e-3
                assert uv.min() > -2 * np.pi - 1e-3
                # print(dataset.json_names[idx])
                assert np.allclose(np.array(recon_surfaces['location'])[0], np.zeros(3), atol=1e-5), f"type: {valid_types[_jdx].item()}  of P is not [0, 0, 0], got {np.array(recon_surfaces['location'])[0]}"
                assert np.allclose(np.array(recon_surfaces['direction'])[0], np.array([0, 0, 1]), atol=1e-5), f"type: {valid_types[_jdx].item()}  of D is not [0, 0, 1], got {np.array(recon_surfaces['direction'])[0]}"
                assert np.allclose(np.array(recon_surfaces['direction'])[1], np.array([1, 0, 0]), atol=1e-5), f"type: {valid_types[_jdx].item()}  of X is not [1, 0, 0], got {np.array(recon_surfaces['direction'])[1]}"
                all_ps[valid_types[_jdx].item()].append(np.array(recon_surfaces['location'])[0])
                all_xs[valid_types[_jdx].item()].append(np.array(recon_surfaces['direction'])[1])
                all_ns[valid_types[_jdx].item()].append(np.array(recon_surfaces['direction'])[0])
                all_scalars[valid_types[_jdx].item()].append(np.array(recon_surfaces['scalar']))

    all_uvs = {k: np.stack(v) for k, v in all_uvs.items()}
    all_xs = {k: np.stack(v) for k, v in all_xs.items()}
    all_ns = {k: np.stack(v) for k, v in all_ns.items()}
    all_ps = {k: np.stack(v) for k, v in all_ps.items()}
    all_scalars = {k: np.stack(v) for k, v in all_scalars.items()}
    for k in range(5):
        v = all_uvs[k]
        p = all_ps[k]
        x = all_xs[k]
        n = all_ns[k]
        scalars = all_scalars[k]
        if not k == 0:
            v /= np.pi
        print(k, 'uv:', v[:, :2].min(), v[:, :2].max(), v[:, 2:].min(), v[:, 2:].max(), 'u_gap_min:', (v[:, 1] - v[:, 0]).min(), 'u_gap_max: ', (v[:, 1] - v[:, 0]).max(), 'v_gap_min:', (v[:, 3] - v[:, 2]).min(), 'v_gap_max: ', (v[:, 3] - v[:, 2]).max(),   'p: ',  p.mean(axis=0), 'x: ', x.mean(axis=0), 'n: ',   n.mean(axis=0), 'scalars mean: ', scalars.mean(axis=0), 'scalars max: ', scalars.max(axis=0), 'scalars min: ', scalars.min(axis=0))

    print('-' * 60)

    

