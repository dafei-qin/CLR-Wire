# -*- coding: utf-8 -*-
"""
Minimal inference script for computing latent representations during online training.
"""

import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'third_party' / 'Michelangelo'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent ))
from michelangelo.utils.misc import get_config_from_file, instantiate_from_config


def init_model(config_path, ckpt_path, device='cuda'):
    """Initialize the model for inference
    
    Args:
        config_path (str): Path to model config file
        ckpt_path (str): Path to model checkpoint
        device (str): Device to run on (default: 'cuda')
    
    Returns:
        model: Loaded model in eval mode
    """
    model_config = get_config_from_file(config_path)
    if hasattr(model_config, "model"):
        model_config = model_config.model
    
    model = instantiate_from_config(model_config, ckpt_path=ckpt_path)
    model = model.to(device)
    model = model.eval()
    
    return model


def to_latent(model, points, sample_posterior=True):
    """Convert points to latent representation
    
    Args:
        model: The loaded model from init_model()
        points: Input points, can be:
            - Single array (N, 6): points (xyz) + normals (xyz)
            - Batch of arrays (B, N, 6): batch of points + normals
            - torch.Tensor or np.ndarray
        device (str): Device to run on (default: 'cuda')
        normalize (bool): Whether to normalize the point cloud to [-1, 1] (default: True)
    
    Returns:
        latents (np.ndarray): Latent representation, shape (B, D) or (D,) for single input
        scale (np.ndarray or None): Scale factor(s) used for normalization, shape (B, 1, 1) or (1, 1)
        center (np.ndarray or None): Center point(s) used for normalization, shape (B, 1, 3) or (1, 3)
    """
    # Convert to tensor if numpy array
    
    
    # Add batch dimension if single input (N, 6) -> (1, N, 6)
    single_input = False
    if points.ndim == 2:
        points = points.unsqueeze(0)
        single_input = True
    
    assert points.ndim == 3 and points.shape[-1] == 6, \
        f"Expected shape (B, N, 6) or (N, 6), got {points.shape}"
    
    # Normalize if requested
    

    # Encode to latent
    with torch.no_grad():
        # Encoding pipeline from the model
        shape_embed, shape_latents = model.model.encode_shape_embed(points, return_latents=True)
        shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents, sample_posterior=sample_posterior)

        latents = shape_zq
    
    
    # Remove batch dimension if single input
    if single_input:
        latents = latents[0]  # (D,)
        
    return latents


if __name__ == '__main__':
    from src.utils.mesh_tools import sample_mesh
    import trimesh
    from tqdm import tqdm
    model = init_model('/home/qindafei/CAD/CLR-Wire/third_party/Michelangelo/configs/aligned_shape_latents/shapevae-256.yaml',
     '/home/qindafei/CAD/CLR-Wire/third_party/Michelangelo/checkpoints/aligned_shape_latents/shapevae-256.ckpt',
     device='cuda:0')


    files = Path('../data/abc_objs_full/0').rglob('*obj')
    files = list(files)
    batch_size = 16
    all_points = []
    all_meshes = []
    print('loading meshes...')
    for f in tqdm(files):
        mesh = trimesh.load_mesh(f)
        all_meshes.append(mesh)
    for idx, f in enumerate(tqdm(files)):
        mesh = all_meshes[idx]
        points, faces = mesh.vertices, mesh.faces
        sampled_points = sample_mesh(points, faces, num_points=10240)
        all_points.append(sampled_points)
        if idx % batch_size == batch_size - 1:
            batch_points = torch.FloatTensor(all_points).to(model.device)
            latents = to_latent(model, batch_points, sample_posterior=True)
            all_points = []
        # sampled_points = torch.FloatTensor(sampled_points).to(model.device)
        # latents = to_latent(model, sampled_points, sample_posterior=True)
        # print(latents.shape)
