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
import random
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.dataset.dataset_v3_tokenize import dataset_compound_tokenize
from src.dataset.dataset_v2 import SURFACE_TYPE_MAP_INV
from src.utils.rts_tools import RotationCodebook, TranslationCodebook, ScaleCodebook, rotate_under_axis
from src.tools.surface_to_canonical_space import  from_canonical, to_canonical
from src.utils.surface_tools import params_to_samples

from myutils.surface import get_approx_face


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
        'u_periodic': False,
        'v_periodic': False,
    }
}

    
class dataset_compound_tokenize_all(Dataset):

    # size + 1: start
    # size + 2: end
    # size + 3: pad
    def __init__(self, json_dir: str, rts_codebook_dir: str, max_tokens: int = 1000, canonical: bool = True, detect_closed: bool = False, bspline_fit_threshold: float = 1e-2, codebook_size=1024, replica=1, rotation_augment: bool = False, point_augment: bool = False, point_augment_intensity: float = 0.005, pc_shape: int = 16384):
        
        self.tokens_per_surface = 14

        self.max_num_surfaces = max_tokens // self.tokens_per_surface
        self.dataset_compound = dataset_compound_tokenize(json_dir, 500, canonical, detect_closed, bspline_fit_threshold, return_orig_surfaces=True)

        self.codebook_size = codebook_size
        self.start_id = self.codebook_size
        self.end_id = self.codebook_size + 1
        self.placeholder_id = self.codebook_size + 2 # Known codes with 0 info, could be a mask.
        self.pad_id = self.codebook_size + 3


        self.max_tokens = max_tokens
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

        self.replica = replica
        self.rotation_augment = rotation_augment
        self.point_augment = point_augment
        self.pc_shape = pc_shape
        self.point_augment_intensity = point_augment_intensity
        self.rotation_angles = list(range(0, 360, 15))
        self.rotation_axes = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    # def rts_augment(self, shifts, rotations, scales):


    def __len__(self):
        return int(len(self.dataset_compound )* self.replica)


    
    def detokenize(self, tokens, bspline_poles):
        # Input tokens with bspline placements and bspline list for filling the bspline surface info
        # Output surfaces

        codes = self.unwarp_codes(tokens)
        codes = codes.reshape(-1, 14)
        surface_type = codes[:, 0]

        # Handle invalid surface type
        surface_type[surface_type > 5] = 0
        surface_code = codes[:, 1:7]
        surface_code[surface_code >= self.codebook_size] = 0

        rts_code = codes[:, 7:14]

        # Flag invalid rts codes
        rts_code_invalid = (rts_code >= self.codebook_size)
        rts_code[rts_code_invalid] = 0

        shifts = self.translation_codebook.decode(rts_code[:, :3])
        rotations = self.rotation_codebook.decode(rts_code[:, 3:6])
        scales = self.scale_codebook.decode(rts_code[:, 6:7])

        # Handle invalid rts
        shifts[rts_code_invalid[:, :3]] = 0
        if rts_code_invalid[:, 3:6].any():
            rotations_euler = R.from_matrix(rotations).as_euler('xyz')
            rotations_euler[rts_code_invalid[:, 3:6]] = 0
            rotations = R.from_euler('xyz', rotations_euler).as_matrix()

        scales[rts_code_invalid[:, 6:7]] = 1

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
                # surface['uv'] = np.array(surface['uv'])


            # Then de_canonicalize
            surface = from_canonical(surface, shifts[i], rotations[i], float(scales[i]))
            surface['idx'] = [i, i]
            surface['orientation'] = 'Forward' # We use all forward now...
            surfaces.append(surface)


        
        return surfaces




    def unwarp_codes(self, codes):
        codes = codes[1:-1] # Drop the start and the end token
        codes = codes[:int(len(codes)//14 * 14)] # Make sure the length is divisible by 14, drop tails
        # unwarpped_codes = []
        # idx = 0
        # while idx < len(codes):

        return codes.reshape(-1, 14)


    def warp_codes(self, codes):
        # Currently we only add the start_code = self.codebook_size, end_code = self.codebook_size+1
        codes = codes.reshape(-1)
        codes = np.concatenate([np.array([self.start_id], dtype=int), codes, np.array([self.end_id], dtype=int)])
        codes[codes == -1] = self.placeholder_id # Known codes with 0 info, could be a mask.
        
        return codes
        

    def apply_rotation_augment(self, tokens, points):
        # Unpadded tokens as input
        angle = random.choice(self.rotation_angles)
        axis = random.choice(self.rotation_axes)
        print('choice angle: ', angle, 'axis: ', axis)
        
        assert tokens.shape[-1] == 14

        rtss = tokens[:, 7:14] # rts = 3 + 3 + 1
        shifts = self.translation_codebook.decode(rtss[:, :3])
        rotations = self.rotation_codebook.decode(rtss[:, 3:6])
        # scales = self.scale_codebook.decode(rtss[:, 6:7])

        rotation_to_apply = rotate_under_axis(angle, axis)
        rotations_new = rotations @ rotation_to_apply.as_matrix().T
        shifts_new = rotation_to_apply.as_matrix() @ shifts[..., None]
        shifts_new = shifts_new[..., 0]

        rotations_new_code = self.rotation_codebook.encode(rotations_new)
        shifts_new_code = self.translation_codebook.encode(shifts_new)
        rotations_new_decode = self.rotation_codebook.decode(rotations_new_code)
        shifts_new_decode = self.translation_codebook.decode(shifts_new_code)
        try:
            assert np.abs(rotations_new_decode - rotations_new).mean() < 2e-3
            assert np.abs(shifts_new_decode - shifts_new).mean() < 1e-3
        except AssertionError:
            return None, None, False

        rtss_new = np.concatenate([shifts_new_code, rotations_new_code, rtss[:, 6:7]], axis=1)
        tokens_new = np.concatenate([tokens[:, :7], rtss_new], axis=1)

        points = rotation_to_apply.as_matrix() @ points[..., None]
        points = points[..., 0]
        return tokens_new, points, True

    def apply_pc_augment(self, points, normals):
        # points: (N, 3)
        # augment the points
        # 4. random jitter with gaussian noise
        noise = np.random.normal(0, self.point_augment_intensity, points.shape)
        points = points + noise
        noise = np.random.normal(0, self.point_augment_intensity, points.shape)
        normals = normals + noise
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-6)
        return points, normals

    def downsample_pc(self, points, normals):
        # points: (N, 3)
        # downsample the points
        # 1. random sample self.pc_shape points
        indices = np.random.choice(len(points), self.pc_shape, replace=False)
        points = points[indices]
        normals = normals[indices]
        return points, normals

    def reordering(self, tokens, poles, graph):

        samples = self.samples_from_tokens(self.warp_codes(tokens), poles)
        surface_centers = samples.mean(axis=(1, 2)) # (B, 3)
        first_surface_idx = np.where(surface_centers[:, -1] == surface_centers[:, -1].min())[0][0]

        old_order = list(nx.bfs_tree(graph, source=0))
        new_order = list(nx.bfs_tree(graph, source=first_surface_idx))
        
        # Create mapping from node to position in old_order
        old_to_pos = {node: pos for pos, node in enumerate(old_order)}
        # Create indexing: for each position in new_order, find its position in old_order
        indexing = [old_to_pos[node] for node in new_order]
        
        # Find all bspline surfaces in original tokens (surface_type == 5)
        # tokens shape: (N, 14), first element is surface_type
        surface_types = tokens[:, 0] if len(tokens.shape) > 1 else tokens[0:1]
        bspline_mask_old = (surface_types == 5)
        bspline_poles_indexing = np.where(bspline_mask_old)[0]  # Positions of bspline surfaces in original tokens
        # bspline_poles_indexing[i] is the position in original tokens for poles[i]
        
        # Reorder tokens using indexing
        tokens = np.array(tokens)[indexing]
        
        # Find bspline surfaces in reordered tokens
        surface_types_new = tokens[:, 0] if len(tokens.shape) > 1 else tokens[0:1]
        bspline_mask_new = (surface_types_new == 5)
        bspline_positions_new = np.where(bspline_mask_new)[0]  # Positions of bspline surfaces in reordered tokens
        
        # For each bspline in reordered tokens, find which pole it corresponds to
        # Map: reordered_position -> original_position (via indexing) -> pole_index (via bspline_poles_indexing)
        poles_reordering = []
        for new_pos in bspline_positions_new:
            old_pos = indexing[new_pos]  # Original position of this bspline in reordered tokens
            # Find which pole index corresponds to this original position
            pole_idx = np.where(bspline_poles_indexing == old_pos)[0]
            if len(pole_idx) > 0:
                poles_reordering.append(pole_idx[0])
        
        # Reorder poles according to the new bspline order
        if len(poles_reordering) > 0 and len(poles) > 0:
            if isinstance(poles, np.ndarray):
                poles = poles[poles_reordering]
            else:
                poles = [poles[i] for i in poles_reordering]
        
        return tokens, poles




    def samples_from_tokens(self, tokens, poles):
        surfaces = self.detokenize(tokens, poles)
        samples = []
        for surface in surfaces:
            if surface['type'] == 'bspline_surface':
                surface['poles'] = np.array(surface['poles'])
                sampled_points = params_to_samples(torch.zeros([]), surface['type'], 8, 8, surface).squeeze(0).numpy()
            else:
                surface['location'] = torch.tensor(surface['location'])
                surface['direction'] = torch.tensor(surface['direction'])
                surface['scalar'] =  torch.tensor(surface['scalar'])
                surface['uv'] =  torch.tensor(surface['uv'])
                sampled_points = params_to_samples(torch.zeros([]), surface['type'], 8, 8, surface).squeeze(0).numpy()
            samples.append(sampled_points)
        samples = np.stack(samples, axis=0) # (B, 8, 8, 3)
        
        return samples



    def __getitem__(self, idx):
        # all_recon_surfaces, all_codes, types_tensor, all_shifts, all_rotations, all_scales = self.dataset_compound[idx]

        solid_valid = True
        json_path = self.dataset_compound.dataset_compound.json_names[idx % len(self.dataset_compound.dataset_compound)]
        npz_data = np.load(json_path.replace('.json', '.npz'), allow_pickle=True)
        # json_data = json.load(open(json_path, 'r'))

        points = np.array(npz_data['points'], dtype=np.float32)  # (N_i, 3)
        normals = np.array(npz_data['normals'], dtype=np.float32)  # List of arrays, each (N_i, 3)

        if self.point_augment:
            points, normals = self.apply_pc_augment(points, normals)
        if self.pc_shape != len(points):
            points, normals = self.downsample_pc(points, normals)

        if len(points.shape) != 2:
            solid_valid = False

        # masks_list = npz_data['masks']  # List of arrays, each (N_i,)
        nodes = npz_data['graph_nodes']
        edges = npz_data['graph_edges']
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        bfs_nodes = list(nx.bfs_tree(graph, source=0))


        all_recon_surfaces, all_codes, types_tensor, all_shifts, all_rotations, all_scales, all_orig_surfaces = self.dataset_compound[idx % len(self.dataset_compound)]


        all_tokens_padded = np.zeros((self.max_tokens), dtype=int) + self.pad_id
        all_bspline_poles_padded = np.zeros((self.max_num_surfaces, 4, 4, 4), dtype=np.float32)
        all_bspline_valid_mask = np.zeros((self.max_num_surfaces), dtype=bool)


        if len(nodes) != len(all_recon_surfaces):
            # Should be bspline drop
            solid_valid = False
            return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid, 
        all_shifts_code = self.translation_codebook.encode(all_shifts)
        all_rotations_code = self.rotation_codebook.encode(all_rotations)
        all_scales_code = self.scale_codebook.encode(all_scales)

        all_shifts_diff = np.abs(self.translation_codebook.decode(all_shifts_code) - all_shifts).sum(axis=1)
        all_scales_diff = np.abs(self.scale_codebook.decode(all_scales_code) - all_scales)
        flag_bspline_replacement = (all_scales_diff > 1e-2) | (all_shifts_diff > 1e-2)

        # If params too large, try use bspline fitting.
        for s_idx in range(len(flag_bspline_replacement)):
            if flag_bspline_replacement[s_idx]:
                surface = all_recon_surfaces[s_idx]

                # Try bspline fitting in the original space
                try:
                    surface = from_canonical(surface, all_shifts[s_idx], all_rotations[s_idx], float(all_scales[s_idx]))
                    assert surface['type'] != 'bspline_surface'
                    surface['location'] = torch.tensor(surface['location'])
                    surface['direction'] = torch.tensor(surface['direction'])
                    surface['scalar'] =  torch.tensor(surface['scalar'])
                    surface['uv'] =  torch.tensor(surface['uv'])

                    sampled_points = params_to_samples(torch.zeros([]), surface['type'], 32, 32, surface).squeeze(0).numpy()
                    fitted_poles = np.array(get_approx_face(sampled_points)).reshape(4, 4, 3)

                    fitted_poles_with_weights = np.concatenate(
                        [fitted_poles, np.ones((4, 4, 1))], axis=-1
                    )
                    
                    # Build fitted surface and compare
                    fitted_knots = np.array([0.0, 1.0], dtype=np.float64)
                    fitted_mults = np.array([4, 4], dtype=np.int32)
                    fitted_surface = self.dataset_compound.dataset_compound._build_occ_bspline_surface(
                        3, 3, 4, 4,
                        fitted_knots, fitted_knots, fitted_mults, fitted_mults,
                        fitted_poles_with_weights, False, False
                    )
                    
                    # Sample fitted surface
                    fitted_samples = self.dataset_compound.dataset_compound._sample_bspline_surface_grid(fitted_surface, 32, 32)
                    
                    # Compute MSE (both surfaces are in [-1, 1] canonical space)
                    fit_error = np.mean((sampled_points - fitted_samples) ** 2)

                    if fit_error > self.bspline_fit_threshold:
                        print(f'Over-range Bspline fitting error: {fit_error:.5f} to large, drop solid')
                        solid_valid = False
                    
                    else:
                        # Then we change the type to bspline surface, and re-calculate the rts.

                        poles_canonical, _rotation, _shift, _scale = self.dataset_compound.dataset_compound._canonicalize_bspline_poles(fitted_poles.copy(), fitted_surface)


                        surface = surface_template['bspline_surface']
                        surface['poles'] = np.concatenate(
                        [poles_canonical, np.ones((4, 4, 1))], axis=-1
                    ).tolist()

                        all_recon_surfaces[s_idx] = surface
                        types_tensor[s_idx] = 5
                        all_codes[s_idx] = np.zeros(6, dtype=int) - 2
                        all_shifts[s_idx] = _shift
                        all_rotations[s_idx] = _rotation
                        all_scales[s_idx] = _scale
                except AssertionError:
                    solid_valid = False
                    return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid

        
        # Then we do the tokenization of updated rts.
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

        if len(all_bspline_poles) == 0:
            all_bspline_poles = np.zeros((self.max_num_surfaces, 4, 4, 4), dtype=np.float32)
        else:
            all_bspline_poles = np.stack(all_bspline_poles, axis=0)
            all_bspline_valid_mask[:len(all_bspline_poles)] = True


        # Do the augmentaion if needed
        if self.rotation_augment:
            all_tokens, points, solid_valid = self.apply_rotation_augment(all_tokens, points)
            if not solid_valid:
                return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid
        all_tokens = np.stack(all_tokens, axis=0)
        all_tokens = self.warp_codes(all_tokens)


        if len(all_tokens) > self.max_tokens:
            solid_valid = False
        else:

            all_tokens_padded[:len(all_tokens)] = all_tokens
            all_bspline_poles_padded[:len(all_bspline_poles)] = all_bspline_poles

        return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid


class dataset_compound_tokenize_all_cache(dataset_compound_tokenize_all):
    # TODO: add rotation augmentation
    def __init__(self, cache_file: str, rts_codebook_dir: str, max_tokens: int = 1000, canonical: bool = True, detect_closed: bool = False, bspline_fit_threshold: float = 1e-2, codebook_size=1024, replica=1, rotation_augment: bool = False, point_augment: bool = False, point_augment_intensity: float = 0.005, pc_shape: int = 16384, replace_file_header='', emphasize_long=False):
        super().__init__('', rts_codebook_dir, max_tokens, canonical, detect_closed, bspline_fit_threshold, codebook_size, replica, rotation_augment, point_augment, point_augment_intensity, pc_shape)
        self.cache_file = cache_file
        self.data = pickle.load(open(self.cache_file, 'rb'))
        self.npz_path = self.data['npz_path']
        self.replace_file_header = replace_file_header
        if self.replace_file_header != '':
            self.npz_path = [p.replace('/data/ssd/CAD/data/abc_step_pc', self.replace_file_header) for p in self.npz_path]

        self.tokens = self.data['tokens']
        self.poles = self.data['poles']

        self._npz_path = []
        self._tokens = []
        self._poles = []
        self.emphasize_long = emphasize_long
        print(f"Length of original dataset: {len(self.npz_path)}")
        if self.emphasize_long:
            print(f"Emphasizing long tokens, 1.5 repeat for > 400 tokens, after 100 epochs")
        for i in range(len(self.npz_path)):
            token_length = len(self.tokens[i])
            if token_length < 100:
                repeat = 0

            elif token_length >= 100 and token_length < 200:
                repeat = 1
            elif token_length >= 200 and token_length < 400:
                repeat = 2
            elif token_length >= 400 and token_length < 600:
                repeat = 4
            elif token_length >= 600:
                repeat = 8

            if self.emphasize_long and token_length >= 400:
                repeat = int(repeat * 1.5)

            self._npz_path.extend([self.npz_path[i]] * repeat)
            self._tokens.extend([self.tokens[i]] * repeat)
            self._poles.extend([self.poles[i]] * repeat)
        print(f"Length of augmented dataset: {len(self._npz_path)}")
        self.json_names = [p.replace('.npz', '.json') for p in self.npz_path]
        

    def __len__(self):
        return len(self.npz_path) * int(self.replica)



    def __getitem__(self, idx):

        npz_data = np.load(self.npz_path[idx % len(self.npz_path)], allow_pickle=True)
        json_data = json.load(open(self.npz_path[idx % len(self.npz_path)].replace('.npz', '.json'), 'r'))

        points = np.array(npz_data['points'], dtype=np.float32)  # (N_i, 3)
        normals = np.array(npz_data['normals'], dtype=np.float32)  # List of arrays, each (N_i, 3)

        if self.point_augment:
            points, normals = self.apply_pc_augment(points, normals)
        if self.pc_shape != len(points):
            points, normals = self.downsample_pc(points, normals)

        nodes = npz_data['graph_nodes']
        edges = npz_data['graph_edges']
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        
        
        tokens = self.tokens[idx % len(self.tokens)]

        all_tokens_padded = np.zeros((self.max_tokens), dtype=int) + self.pad_id
        all_bspline_poles_padded = np.zeros((self.max_num_surfaces, 4, 4, 4), dtype=np.float32)
        all_bspline_valid_mask = np.zeros((self.max_num_surfaces), dtype=bool)

        if self.rotation_augment:
            trys = 5
            tokens = self.unwarp_codes(tokens)
            while trys > 0:
                tokens_new, points_new, solid_valid_new = self.apply_rotation_augment(tokens, points)
                if not solid_valid_new:
                    trys -= 1
                    continue
                else:
                    break
            if solid_valid_new:
                tokens = tokens_new
                points = points_new
                solid_valid = solid_valid_new
            else:
                points = np.zeros((16384, 3), dtype=np.float32)
                normals = np.zeros((16384, 3), dtype=np.float32)
                return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid

            tokens = self.warp_codes(tokens)

        
        poles = self.poles[idx % len(self.poles)]

        # Do the reordering base on the rotated version

        tokens, poles = self.reordering(self.unwarp_codes(tokens), poles, graph)
        tokens = self.warp_codes(tokens)



        

        all_tokens_padded[:len(tokens)] = tokens
        all_bspline_poles_padded[:len(poles)] = poles
        all_bspline_valid_mask[:len(poles)] = True

        return points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, True




    



if __name__ == '__main__':
    # dataset = dataset_compound_tokenize_all(json_dir='../data/abc_step_pc_0009', rts_codebook_dir='./assets/codebook', bspline_fit_threshold=1e-2)
    
    # # for data in tqdm(dataset):
    # for idx in tqdm(range(len(dataset))):
    #     points_list, normals_list, masks_list, tokens, bspline_poles, valid = dataset[idx]
        
    #     if valid:
    #         surfaces = dataset.detokenize(tokens, bspline_poles)
    #         print()
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from src.utils.import_tools import load_dataset_from_config
    from omegaconf import OmegaConf

    config = OmegaConf.load('./src/configs/gpt/gpt_0102_michel_A800.yaml')
    dataset = load_dataset_from_config(config, section='data_train')
    num_rot_aug_invalid = 0
    num_total_invalid = 0
    for idx in tqdm(range(1000)):
        print(idx)
        data = dataset[idx]
        if data[-1] == -2:
            num_rot_aug_invalid += 1
        if data[-1] != True:
            num_total_invalid += 1

    print(num_rot_aug_invalid)
    print(num_total_invalid)
