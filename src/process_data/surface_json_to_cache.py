import sys
from pathlib import Path
import os
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v1 import dataset_compound

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--canonical', action='store_true', default=False)
    parser.add_argument('--detect_closed', action='store_true', default=False)
    args = parser.parse_args()
    if os.path.exists(args.output_path):
        print('cache already exists, stop')
        exit()
    dataset = dataset_compound(args.input_dir, canonical=args.canonical, detect_closed=args.detect_closed)
    all_params = []
    all_types = []
    all_shifts = []
    all_rotations = []
    all_scales = []
    all_is_u_closed = []
    all_is_v_closed = []

    for i in tqdm(range(len(dataset))):
       if args.detect_closed:
           params_tensor, types_tensor, mask_tensor, shifts, rotations, scales, is_u_closed_tensor, is_v_closed_tensor = dataset[i]
       else:
           params_tensor, types_tensor, mask_tensor, shifts, rotations, scales = dataset[i]
       
       mask_tensor = mask_tensor.bool()
       params_tensor = params_tensor[mask_tensor]
       types_tensor = types_tensor[mask_tensor]
       shifts = shifts[mask_tensor]
       rotations = rotations[mask_tensor]
       scales = scales[mask_tensor]
       
       if args.detect_closed:
           is_u_closed_tensor = is_u_closed_tensor[mask_tensor]
           is_v_closed_tensor = is_v_closed_tensor[mask_tensor]

       if len(params_tensor) == 0:
           continue

       all_params.append(params_tensor.numpy())
       all_types.append(types_tensor.numpy())
       all_shifts.append(shifts)
       all_rotations.append(rotations)
       all_scales.append(scales)
       
       if args.detect_closed:
           all_is_u_closed.append(is_u_closed_tensor.numpy())
           all_is_v_closed.append(is_v_closed_tensor.numpy())

    all_params = np.concatenate(all_params, axis=0)
    all_types = np.concatenate(all_types, axis=0)
    all_shifts = np.concatenate(all_shifts, axis=0)
    all_rotations = np.concatenate(all_rotations, axis=0)
    all_scales = np.concatenate(all_scales, axis=0)

    if args.detect_closed:
        all_is_u_closed = np.concatenate(all_is_u_closed, axis=0)
        all_is_v_closed = np.concatenate(all_is_v_closed, axis=0)
        np.savez_compressed(args.output_path, params=all_params, types=all_types, shifts=all_shifts, 
                          rotations=all_rotations, scales=all_scales, 
                          is_u_closed=all_is_u_closed, is_v_closed=all_is_v_closed)
    else:
        np.savez_compressed(args.output_path, params=all_params, types=all_types, shifts=all_shifts, 
                          rotations=all_rotations, scales=all_scales)

    print(f'Processed {len(dataset)} surfaces')

if __name__ == '__main__':
    main()