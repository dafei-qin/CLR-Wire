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
import networkx as nx
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.dataset.dataset_v3_tokenize import dataset_compound_tokenize
from src.dataset.dataset_v2 import SURFACE_TYPE_MAP_INV
from src.utils.rts_tools import RotationCodebook, TranslationCodebook, ScaleCodebook
from src.tools.surface_to_canonical_space import  from_canonical


surface_template = {
    'plane':
    {
        'type': 'plane',
        'location': [[0, 0, 0]],
        'direction': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        'scalar': [],
        'uv': [0, 0, 0, 0],
    },
    'cylinder':
    {
        'type': 'cylinder',
        'location': [[0, 0, 0]],
        'direction': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        'scalar': [0],
        'uv': [0, 0, 0, 0],
    },
    'cone':
    {
        'type': 'cone',
        'location': [[0, 0, 0]],
        'direction': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        'scalar': [0, 0],
        'uv': [0, 0, 0, 0],
    },
    'sphere':
    {
        'type': 'sphere',
        'location': [[0, 0, 0]],
        'direction': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        'scalar': [1],
        'uv': [0, 0, 0, 0],
    },
    'torus':
    {
        'type': 'torus',
        'location': [[0, 0, 0]],
        'direction': [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        'scalar': [1, 0],
        'uv': [0, 0, 0, 0],
    },
    'bspline_surface':
    {
        'type': 'bspline_surface',
        'location': [[0, 0, 0]],
        'direction': [],
        'scalar': [3, 3, 4, 4, 2, 2, 0.0, 1.0, 0.0, 1.0, 4, 4, 4, 4], # Standard cubic bezier scalars.
        'uv': [0, 1, 0, 1],
        'poles': [],
    }
}

class dataset_compound_tokenize_all(Dataset):

    def __init__(self, json_dir: str, rts_codebook_dir: str, max_num_surfaces: int = 500, canonical: bool = False, detect_closed: bool = False, bspline_fit_threshold: float = 1e-5, codebook_size=1024):

        self.dataset_compound = dataset_compound_tokenize(json_dir, max_num_surfaces, canonical, detect_closed, bspline_fit_threshold)

        self.codebook_size = codebook_size
        self.max_num_surfaces = max_num_surfaces
        self.canonical = canonical
        self.detect_closed = detect_closed
        self.bspline_fit_threshold = bspline_fit_threshold

        self.codebook_dir = rts_codebook_dir
        rot_cb_path = os.path.join(self.codebook_dir, 'cb_rotation.pkl')
        trans_cb_path = os.path.join(self.codebook_dir, 'cb_translation.pkl')
        scale_cb_path = os.path.join(self.codebook_dir, 'cb_scale.pkl')

        self.rotation_codebook = RotationCodebook(codebook_size=0)  # Size will be loaded
        self.rotation_codebook.load(rot_cb_path)

        self.translation_codebook = TranslationCodebook(codebook_size=0)
        self.translation_codebook.load(trans_cb_path)

        self.scale_codebook = ScaleCodebook(codebook_size=0)
        self.scale_codebook.load(scale_cb_path)

    # def rts_augment(self, shifts, rotations, scales):


    def __len__(self):
        return len(self.dataset_compound)


    def detokenize(self, tokens, bspline_poles):
        # Input tokens with bspline placements and bspline list for filling the bspline surface info
        # Output surfaces

        codes = self.unwarp_codes(tokens)
        codes = codes.reshape(-1, 14)
        surface_type = codes[:, 0]
        surface_code = codes[:, 1:7]
        rts_code = codes[:, 7:14]
        shifts = self.translation_codebook.decode(rts_code[:, :3])
        rotations = self.rotation_codebook.decode(rts_code[:, 3:6])
        scales = self.scale_codebook.decode(rts_code[:, 6:7])

        surfaces = []
        bspline_idx = 0

        # First recover surface jsons
        for i in range(len(surface_type)):
            if surface_type[i] == 5:
                # Bspline surface
                surface = surface_template['bspline_surface']
                surface['poles'] = bspline_poles[bspline_idx]
                bspline_idx += 1
            else:
                surface = surface_template[SURFACE_TYPE_MAP_INV[surface_type[i]]]
                surface = self.dataset_compound.de_tokenize(surface, surface_code[i])

            # Then de_canonicalize
            surface = from_canonical(surface, shifts[i], rotations[i], scales[i])
            surfaces.append(surface)


        
        return surfaces




    def unwarp_codes(self, codes):
        return codes[1:-1].reshape(-1, 14)


    def warp_codes(self, codes):
        # Currently we only add the start_code = self.codebook_size, end_code = self.codebook_size+1
        codes = codes.reshape(-1)
        codes = np.concatenate([np.array([self.codebook_size], dtype=int), codes, np.array([self.codebook_size+1], dtype=int)])
        
        return codes
        





    def __getitem__(self, idx):
        # all_recon_surfaces, all_codes, types_tensor, all_shifts, all_rotations, all_scales = self.dataset_compound[idx]
        json_path = self.dataset_compound.dataset_compound.json_names[idx]
        npz_data = np.load(json_path.replace('.json', '.npz'), allow_pickle=True)
        json_data = json.load(open(json_path, 'r'))

        points_list = npz_data['points']  # List of arrays, each (N_i, 3)
        normals_list = npz_data['normals']  # List of arrays, each (N_i, 3)
        masks_list = npz_data['masks']  # List of arrays, each (N_i,)
        nodes = npz_data['graph_nodes']
        edges = npz_data['graph_edges']
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        bfs_nodes = list(nx.bfs_tree(graph, source=0))


        all_recon_surfaces, all_codes, types_tensor, all_shifts, all_rotations, all_scales = self.dataset_compound[idx]


        if len(nodes) != len(all_recon_surfaces):
            # Should be bspline drop
            valid = False
            return [], [], valid
        all_shifts_code = self.translation_codebook.encode(all_shifts)
        all_rotations_code = self.rotation_codebook.encode(all_rotations)
        all_scales_code = self.scale_codebook.encode(all_scales)

        all_tokens = []
        all_bspline_poles = []
        
        
        for node in bfs_nodes:
            surface = all_recon_surfaces[node]
            if surface['type'] == 'bspline_surface':
                # Special marker for bspline for later fsq tokenize
                surface_code = np.zeros(6, dtype=int) - 2
                all_bspline_poles.append(np.array(surface['poles']))
            else:
                surface_code = all_codes[node]
            type_code = types_tensor[node].item()
            shift = all_shifts_code[node]
            rotation = all_rotations_code[node]
            scale = all_scales_code[node]
            rts_code = np.concatenate([shift, rotation, np.array(scale, dtype=int)[None]], axis=0)

            token = np.concatenate([np.array(type_code, dtype=int)[None], surface_code, rts_code], axis=0)
            all_tokens.append(token)

        all_tokens = np.stack(all_tokens, axis=0)
        all_tokens = self.warp_codes(all_tokens)
        if len(all_bspline_poles) > 0:
            return points_list, normals_list, masks_list, all_tokens, np.stack(all_bspline_poles, axis=0), True
        else:
            return points_list, normals_list, masks_list, all_tokens, [], True


if __name__ == '__main__':
    dataset = dataset_compound_tokenize_all(json_dir='../data/abc_step_pc_0009', rts_codebook_dir='./assets/codebook', bspline_fit_threshold=1e-2)
    
    # for data in tqdm(dataset):
    for idx in tqdm(range(len(dataset))):
        points_list, normals_list, masks_list, tokens, bspline_poles, valid = dataset[idx]
        
        if valid:
            surfaces = dataset.detokenize(tokens, bspline_poles)
            print()
    