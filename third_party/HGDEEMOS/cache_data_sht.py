import glob
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union
import math
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial
import random
import os
from datetime import datetime
import numpy as np
import trimesh
import warnings
import einops
import pickle 
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))
# Add project root to sys.path to import src.utils
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

from src.utils.import_tools import load_dataset_from_config, load_model_from_config
from src.utils.gpt_tools import tokenize_bspline_poles
from omegaconf import OmegaConf
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_file', type=str, default='/data/ssd/CAD/data/abc_step_pc/4')
    parser.add_argument('--max_tokens', type=int, default=1000)
    parser.add_argument('--save_file', type=str, default='/data/ssd/CAD/data/abc_step_pc/cache_4.pkl')
    args = parser.parse_args()
    config = OmegaConf.load("src/configs/gpt/cache_sht.yaml")
    config.data_train.params.json_dir = args.cache_file
    config.data_train.params.max_tokens = args.max_tokens
    dataset = load_dataset_from_config(config, section='data_train')
    
    list_npz_path = []
    list_tokens = []
    list_poles = []

    for idx in tqdm(range(len(dataset))):
    # for idx in tqdm(range(100)):
        points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid = dataset[idx]
        if not solid_valid:
            continue
        all_tokens = all_tokens_padded[~(all_tokens_padded == dataset.pad_id)]
        all_bspline_poles = all_bspline_poles_padded[all_bspline_valid_mask]
        npz_path = dataset.dataset_compound.dataset_compound.json_names[idx % len(dataset.dataset_compound.dataset_compound)].replace('.json', '.npz')

        list_npz_path.append(npz_path)
        list_tokens.append(all_tokens)
        list_poles.append(all_bspline_poles)


    with open(args.save_file, 'wb') as f:
        pickle.dump({'npz_path': list_npz_path, 'tokens': list_tokens, 'poles': list_poles}, f)





        
