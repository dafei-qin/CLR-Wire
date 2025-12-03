import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    input_dirs = list(input_dir.rglob('*.npz'))
    data_all = {}
    data_temp = np.load(input_dirs[0])
    for key in data_temp.keys():
        data_all[key] = []
    for input_dir in tqdm(input_dirs):
        data = np.load(input_dir)
        for key in data.keys():
            data_all[key].append(data[key])
    for key in data_all.keys():
        data_all[key] = np.concatenate(data_all[key], axis=0)
    np.savez_compressed(output_path, **data_all)
    