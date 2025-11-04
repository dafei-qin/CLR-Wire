import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter

from src.dataset.dataset_bspline import dataset_bspline




def check_bspline_relations(dataset):
    counter = 0
    for i in range(len(dataset)):
        u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, u_mults_list, v_mults_list, poles, valid = dataset.load_data(dataset.data_names[i])
        # print(f'Surface {i}: u_degree={u_degree}, v_degree={v_degree}, num_poles_u={num_poles_u}, num_poles_v={num_poles_v}, num_knots_u={num_knots_u}, num_knots_v={num_knots_v}, is_u_periodic={is_u_periodic}, is_v_periodic={is_v_periodic}, u_knots_list={u_knots_list}, v_knots_list={v_knots_list}, u_mults_list={u_mults_list}, v_mults_list={v_mults_list}, poles={poles}, valid={valid}')
        try:
            if is_u_periodic:
                # print('pu, ', np.sum(u_mults_list), u_degree, num_poles_u,  is_u_periodic)
                # assert np.sum(u_mults_list) == u_degree + num_poles_u
                assert len(np.unique(u_mults_list)) == 1
                pass
            else:
                assert np.sum(u_mults_list) == u_degree + 1 + num_poles_u
            if is_v_periodic:
                # assert np.sum(v_mults_list) == v_degree + num_poles_v
                pass
            else:
                assert np.sum(v_mults_list) == v_degree + 1 + num_poles_v
        except AssertionError:
            print(f'Surface {i}: u_degree={u_degree}, v_degree={v_degree}, num_poles_u={num_poles_u}, num_poles_v={num_poles_v}, num_knots_u={num_knots_u}, num_knots_v={num_knots_v}, is_u_periodic={is_u_periodic}, is_v_periodic={is_v_periodic}, u_mults_list={u_mults_list}, v_mults_list={v_mults_list}, valid={valid}, index={i}')
            counter += 1
    print(f'Total {counter}/{len(dataset)} surfaces failed')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Check B-spline relations')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    args = parser.parse_args()

    dataset = dataset_bspline(data_path=args.dataset, replica=1)
    print(f'Loaded {len(dataset)} B-spline surfaces')

    check_bspline_relations(dataset)