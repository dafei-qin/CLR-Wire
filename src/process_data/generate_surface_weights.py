import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.dataset_v1 import type_weights, SURFACE_TYPE_MAP_INV

from argparse import ArgumentParser
import numpy as np

program_parser = ArgumentParser(description='Generate surface weights.')
program_parser.add_argument('--input_cache', type=str, default='', help='Path to cache file.')
program_parser.add_argument('--save_name', type=str, help='path of the weights')
cli_args, unknown = program_parser.parse_known_args()


data = np.load(cli_args.input_cache)

weight_list = []

types = data['types']

for i in range(len(types)):
    weight_list.append(type_weights[SURFACE_TYPE_MAP_INV[types[i]]])

np.save(cli_args.save_name, weight_list)
