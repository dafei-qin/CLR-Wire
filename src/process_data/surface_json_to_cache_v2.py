import sys
from pathlib import Path
import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from collections import Counter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_v2 import dataset_compound, SURFACE_TYPE_MAP_INV

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--canonical', action='store_true', default=False)
    parser.add_argument('--detect_closed', action='store_true', default=False)
    parser.add_argument('--bspline_fit_threshold', type=float, default=1e-3, 
                       help='Fitting error threshold for bspline surfaces')
    args = parser.parse_args()
    if os.path.exists(args.output_path):
        print('cache already exists, stop')
        exit()
    dataset = dataset_compound(args.input_dir, canonical=args.canonical, 
                             detect_closed=args.detect_closed,
                             bspline_fit_threshold=args.bspline_fit_threshold)
    all_params = []
    all_types = []
    all_shifts = []
    all_rotations = []
    all_scales = []
    all_is_u_closed = []
    all_is_v_closed = []
    
    # Statistics
    type_counter = Counter()
    total_jsons = 0
    all_valid_jsons = 0
    total_original_surfaces = 0
    total_valid_surfaces = 0

    for i in tqdm(range(len(dataset))):
       if args.detect_closed:
           params_tensor, types_tensor, mask_tensor, shifts, rotations, scales, is_u_closed_tensor, is_v_closed_tensor = dataset[i]
       else:
           params_tensor, types_tensor, mask_tensor, shifts, rotations, scales = dataset[i]
       
       # Statistics for this json
       total_jsons += 1
       original_mask = mask_tensor.bool()
       
       # Get actual number of surfaces in original JSON file
       json_path = dataset.json_names[i]
       try:
           with open(json_path, 'r') as f:
               surfaces_data = json.load(f)
           actual_surface_count = len(surfaces_data)
       except (json.JSONDecodeError, FileNotFoundError):
           actual_surface_count = 0
       
       # Check if all surfaces in this json are valid (excluding padding)
       valid_surface_count = original_mask.sum().item()
       total_original_surfaces += actual_surface_count
       total_valid_surfaces += valid_surface_count
       
       if actual_surface_count > 0 and valid_surface_count == actual_surface_count:
           all_valid_jsons += 1
       
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
       
       assert params_tensor.shape[0] == types_tensor.shape[0] == shifts.shape[0] == rotations.shape[0] == scales.shape[0] == is_u_closed_tensor.shape[0] == is_v_closed_tensor.shape[0]
       # Count types
       for type_idx in types_tensor.numpy():
           type_counter[int(type_idx)] += 1

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

    # Prepare statistics file path
    stats_path = args.output_path.replace('.npz', '_stats.txt')
    
    # Prepare statistics content
    stats_lines = []
    stats_lines.append('='*70)
    stats_lines.append('PROCESSING SUMMARY')
    stats_lines.append('='*70)
    stats_lines.append('')
    
    # Add configuration info
    stats_lines.append('[Configuration]')
    stats_lines.append(f'Input directory: {args.input_dir}')
    stats_lines.append(f'Output cache: {args.output_path}')
    stats_lines.append(f'Canonical: {args.canonical}')
    stats_lines.append(f'Detect closed: {args.detect_closed}')
    stats_lines.append(f'Bspline fit threshold: {args.bspline_fit_threshold}')
    stats_lines.append('')
    
    # JSON statistics
    stats_lines.append('[JSON Files Statistics]')
    stats_lines.append(f'Total JSON files processed: {total_jsons}')
    stats_lines.append(f'JSON files with all valid surfaces: {all_valid_jsons} ({100*all_valid_jsons/total_jsons:.2f}%)')
    stats_lines.append(f'JSON files with some invalid surfaces: {total_jsons - all_valid_jsons} ({100*(total_jsons-all_valid_jsons)/total_jsons:.2f}%)')
    stats_lines.append('')
    
    # Surface statistics
    total_surfaces = len(all_params)
    stats_lines.append('[Surface Statistics]')
    stats_lines.append(f'Total original surfaces (in JSON files): {total_original_surfaces}')
    stats_lines.append(f'Total valid surfaces (after processing): {total_valid_surfaces}')
    stats_lines.append(f'Total surfaces saved to cache: {total_surfaces}')
    if total_original_surfaces > 0:
        stats_lines.append(f'Success rate: {100*total_valid_surfaces/total_original_surfaces:.2f}%')
        invalid_count = total_original_surfaces - total_valid_surfaces
        stats_lines.append(f'Invalid/dropped surfaces: {invalid_count} ({100*invalid_count/total_original_surfaces:.2f}%)')
    stats_lines.append(f'Parameter dimension: {all_params.shape[1]}')
    stats_lines.append('')
    
    # Type distribution
    stats_lines.append('[Surface Type Distribution]')
    stats_lines.append(f'{"Type":<20} {"Count":<10} {"Percentage":<10}')
    stats_lines.append(f'{"-"*40}')
    
    # Sort by type index for consistent output
    for type_idx in sorted(type_counter.keys()):
        count = type_counter[type_idx]
        percentage = 100 * count / total_surfaces
        type_name = SURFACE_TYPE_MAP_INV.get(type_idx, f'Unknown({type_idx})')
        stats_lines.append(f'{type_name:<20} {count:<10} {percentage:>6.2f}%')
    
    stats_lines.append('='*70)
    
    # Write to file
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    
    # Print to console
    print('\n' + '\n'.join(stats_lines) + '\n')
    print(f'Statistics saved to: {stats_path}')

if __name__ == '__main__':
    main()

