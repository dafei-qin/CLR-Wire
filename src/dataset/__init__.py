"""
Dataset modules for CAD surface data.
"""

from .dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from .dataset_latent import LatentDataset, LatentDatasetFlat

__all__ = [
    'dataset_compound',
    'SURFACE_TYPE_MAP',
    'SCALAR_DIM_MAP',
    'LatentDataset',
    'LatentDatasetFlat',
]


