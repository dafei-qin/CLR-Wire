import argparse
import bisect
from pathlib import Path
import sys
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Tuple

from src.dataset.dataset_bspline import dataset_bspline


def filter_bspline_periodic_surfaces(dataset):
    data_names = []
    for i in tqdm(range(len(dataset))):
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset.load_data(dataset.data_names[i])
        if valid:
            data_names.append(dataset.data_names[i].strip())
    return data_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Filter periodic surfaces')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the filtered dataset')
    args = parser.parse_args()

    dataset = dataset_bspline(path_file=args.dataset, replica=1, only_periodic=True)
    data_names = filter_bspline_periodic_surfaces(dataset)
    print(f'Filtered {len(data_names)} periodic surfaces')

    # print(data_names)
    with open(args.save_path, 'w') as f:
        for data_name in data_names:
            f.write(data_name + '\n')