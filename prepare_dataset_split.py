import os
import numpy as np
import glob
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

def main(args):
    """
    Main function to find NPZ files, load, aggregate, split, and save data.
    """
    # Setup
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    if args.json_input_path:
        print(f"Input JSON file: {os.path.abspath(args.json_input_path)}")
    else:
        print(f"Input directory for .npz files: {os.path.abspath(args.input_dir)}")
    print(f"Ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}, Seed={args.seed}")
    if args.keys_to_process:
        print(f"Processing only specified keys: {args.keys_to_process}")
    if args.numerical_keys:
        print(f"Specified numerical keys for validation: {args.numerical_keys}")

    # Convert numerical_keys to a set for efficient lookups
    numerical_keys_as_set = set(args.numerical_keys)

    data_accumulator = defaultdict(list)
    first_file_keys = None
    processed_files_count = 0 # For NPZ, this is file count; for JSON, sample count

    if args.json_input_path:
        # Load from JSON file
        try:
            with open(args.json_input_path, 'r') as f:
                json_data_list = json.load(f)
            if not isinstance(json_data_list, list):
                print(f"Error: JSON file {args.json_input_path} must contain a list of samples. Found {type(json_data_list)}.")
                return
            if not json_data_list:
                print(f"Error: JSON file {args.json_input_path} is empty or contains an empty list.")
                return
            print(f"Loaded {len(json_data_list)} samples from {args.json_input_path}.")

            for i, sample_dict in enumerate(tqdm(json_data_list, desc="Processing JSON samples")):
                if not isinstance(sample_dict, dict):
                    print(f"Warning: Sample at index {i} in JSON is not a dictionary. Skipping this sample.")
                    continue
                
                current_keys = set(sample_dict.keys())
                if not current_keys:
                    print(f"Warning: Sample at index {i} in JSON has no keys. Skipping this sample.")
                    continue

                # Determine and filter keys to process for this sample
                processable_keys_for_sample = current_keys
                if args.keys_to_process:
                    processable_keys_for_sample = current_keys.intersection(set(args.keys_to_process))

                if first_file_keys is None: # First valid sample encountered
                    if processable_keys_for_sample:
                        first_file_keys = processable_keys_for_sample
                        print(f"Discovered keys from the first valid JSON sample (index {i}): {first_file_keys}")
                    elif args.keys_to_process:
                        # This sample didn't have the user-specified keys, try next one
                        print(f"Warning: Sample at index {i} does not contain any of the specified keys_to_process {args.keys_to_process}. Trying next sample.")
                        continue
                    else: # No keys_to_process specified, but sample has no keys (already caught) or processable_keys is empty for other reasons.
                        print(f"Warning: Sample at index {i} resulted in no processable keys. Skipping.")
                        continue 
                elif not processable_keys_for_sample.issuperset(first_file_keys):
                    # Subsequent samples must contain all keys identified as 'first_file_keys'
                    # This also handles the case where keys_to_process was given and first_file_keys was established based on that subset.
                    print(f"Warning: Sample at index {i} (keys: {current_keys}, processable: {processable_keys_for_sample}) does not contain all reference keys {first_file_keys}. Skipping this sample.")
                    continue
                
                if not first_file_keys: # If after some iterations, first_file_keys is still None or empty (e.g. all files skipped so far)
                    if args.keys_to_process and not processable_keys_for_sample:
                        # Still haven't found any specified keys
                        continue
                    elif processable_keys_for_sample: # Found a sample with keys we can use
                        first_file_keys = processable_keys_for_sample
                        print(f"Discovered keys from JSON sample (index {i}): {first_file_keys}")
                    else:
                        # No keys to process from this sample either
                        continue
                
                # Add data for the established first_file_keys from this sample
                temp_data_for_sample = {}
                valid_sample = True
                for key in first_file_keys:
                    if key not in sample_dict: # Should be caught by issuperset check
                        print(f"Critical Warning: Key '{key}' expected but not found in JSON sample index {i}. Skipping this sample.")
                        valid_sample = False
                        break
                    
                    original_json_value = sample_dict[key]
                    try:
                        numpy_value = np.array(original_json_value)
                    except Exception as e:
                        print(f"Warning: Could not convert value for key '{key}' (value: {original_json_value}) in JSON sample index {i} to a numpy array: {e}. Skipping sample.")
                        valid_sample = False
                        break

                    if key in numerical_keys_as_set:
                        if not isinstance(original_json_value, list):
                            print(f"Warning: Numerical key '{key}' in JSON sample index {i} is not a list (type: {type(original_json_value)}). Value: '{str(original_json_value)[:50]}'. Skipping sample.")
                            valid_sample = False
                            break
                        if numpy_value.shape == (): # type: ignore
                            print(f"Warning: Numerical key '{key}' in JSON sample index {i} (original list: '{str(original_json_value)[:50]}') converted to a scalar numpy array (shape {numpy_value.shape}). Skipping sample.")
                            valid_sample = False
                            break
                    
                    temp_data_for_sample[key] = numpy_value

                if valid_sample:
                    for key in first_file_keys:
                        data_accumulator[key].append(temp_data_for_sample[key])
                    processed_files_count += 1
        except FileNotFoundError:
            print(f"Error: JSON file not found at {args.json_input_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {args.json_input_path}: {e}")
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing JSON file {args.json_input_path}: {e}")
            return

    else:
        # 1. Find all .npz files
        glob_path = os.path.join(args.input_dir, '**', '*.npz')
        npz_files = glob.glob(glob_path, recursive=True)
        
        if not npz_files:
            print(f"No .npz files found in '{args.input_dir}' (searched with pattern: '{glob_path}'). Please check the input directory and path, or provide --json_input_path.")
            return
        print(f"Found {len(npz_files)} .npz files.")

        # 2. Load all npz files and aggregate data
        for i, filepath in enumerate(tqdm(npz_files, desc="Processing NPZ files")):
            try:
                with np.load(filepath) as loaded_npz:
                    current_keys = set(loaded_npz.keys())
                    if not current_keys:
                        print(f"Warning: No keys found in {filepath}. Skipping this file.")
                        continue

                    processable_keys_for_file = current_keys
                    if args.keys_to_process:
                        processable_keys_for_file = current_keys.intersection(set(args.keys_to_process))

                    if first_file_keys is None: # First valid file encountered
                        if processable_keys_for_file:
                            first_file_keys = processable_keys_for_file
                            print(f"Discovered keys from the first valid .npz file ({filepath}): {first_file_keys}")
                        elif args.keys_to_process: # This file didn't have the user-specified keys
                            print(f"Warning: File {filepath} does not contain any of the specified keys_to_process {args.keys_to_process}. Will attempt to find them in subsequent files.")
                            continue # Try next file to establish first_file_keys
                        else: # No keys_to_process specified, but file has no processable_keys (e.g. empty intersection)
                            print(f"Warning: File {filepath} resulted in no processable keys. Skipping.")
                            continue
                    elif not processable_keys_for_file.issuperset(first_file_keys):
                        # This file does not contain all the reference keys (which might be filtered by keys_to_process)
                        print(f"Warning: Keys in {filepath} (processable: {processable_keys_for_file}) do not form a superset of reference keys {first_file_keys}. Skipping this file.")
                        continue
                    
                    if not first_file_keys: # If after some iterations, first_file_keys is still None (e.g. all files skipped so far)
                        if args.keys_to_process and not processable_keys_for_file:
                            # Still haven't found any specified keys
                            continue
                        elif processable_keys_for_file:
                             first_file_keys = processable_keys_for_file
                             print(f"Discovered keys from .npz file ({filepath}): {first_file_keys}")
                        else:
                            # No keys to process from this file either
                            continue

                    temp_data_for_file = {}
                    valid_file = True
                    for key in first_file_keys: # Iterate over the established/filtered keys
                        if key not in loaded_npz: # Should be caught by issuperset check above
                            print(f"Critical Warning: Key '{key}' expected (from first_file_keys) but not found in {filepath}. Skipping this file.")
                            valid_file = False
                            break
                        
                        numpy_value = loaded_npz[key]

                        if key in numerical_keys_as_set:
                            # For NPZ, value is already a numpy array. We just check its shape.
                            if not isinstance(numpy_value, np.ndarray):
                                print(f"Internal Warning: Value for numerical key '{key}' from NPZ {filepath} is not a numpy array (type: {type(numpy_value)}). This is unexpected. Skipping file.")
                                valid_file = False
                                break
                            if numpy_value.shape == (): # type: ignore
                                print(f"Warning: Numerical key '{key}' in NPZ file {filepath} is a scalar numpy array (shape {numpy_value.shape}). Skipping file.")
                                valid_file = False
                                break
                        
                        temp_data_for_file[key] = numpy_value

                    if valid_file:
                        for key in first_file_keys:
                            data_accumulator[key].append(temp_data_for_file[key])
                        processed_files_count += 1

            except Exception as e:
                print(f"Error loading or processing {filepath}: {e}. Skipping this file.")
                continue
    
    if not data_accumulator or processed_files_count == 0:
        print("No data successfully loaded and aggregated. Exiting.")
        return
    
    print(f"Successfully processed and aggregated data from {processed_files_count} files.")

    if args.keys_to_process and not first_file_keys:
        print(f"Error: Specified keys_to_process {args.keys_to_process} were not found in any of the processed .npz files. Exiting.")
        return

    if not first_file_keys: 
        print("Error: No keys to process were identified after checking all sources. Exiting.")
        return

    print(f"Final set of keys to be processed from all valid sources: {first_file_keys}")
    if args.numerical_keys: # Check if the user provided the argument
        applicable_numerical_keys = first_file_keys.intersection(numerical_keys_as_set)
        if applicable_numerical_keys:
            print(f"Numerical data checks (list-like for JSON, non-scalar array shape) will be applied to: {applicable_numerical_keys}")
        
        ignored_numerical_keys = numerical_keys_as_set - first_file_keys
        if ignored_numerical_keys:
            print(f"Warning: Specified numerical keys {ignored_numerical_keys} are not among the common/selected keys ({first_file_keys}) and will thus not undergo numerical checks or be processed if not in first_file_keys.")

    if not all(key in data_accumulator and data_accumulator[key] for key in first_file_keys):
        print("Error: Some keys did not accumulate any data. This might be due to all files being skipped for those keys.")
        return

    # aggregated_data will store lists of arrays, not concatenated arrays
    aggregated_data = {}
    for key in first_file_keys:
        aggregated_data[key] = data_accumulator[key]

    if not aggregated_data: # Should not happen if previous checks pass, but as a safeguard
        print("Data aggregation resulted in no data. Exiting.")
        return

    # Determine total samples from the number of items in the list for the first key
    # All keys should have the same number of items (processed files)
    any_key = list(aggregated_data.keys())[0]
    total_samples = len(aggregated_data[any_key]) # Number of files/samples
    
    print(f"Total aggregated samples (files): {total_samples}")
    if total_samples == 0:
        print("No samples to process after aggregation. Exiting.")
        return

    # 4. Generate random shuffle order and split
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    train_end = int(args.train_ratio * total_samples)
    val_end = train_end + int(args.val_ratio * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    print(f"Data split determined: Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}, Test samples: {len(test_indices)}")

    # 5. Store the values based on this split for each key
    for key in aggregated_data:
        list_of_arrays_for_key = aggregated_data[key] # This is a list of np.arrays
        
        # Select arrays for each split based on indices
        train_data = [list_of_arrays_for_key[i] for i in train_indices]
        val_data = [list_of_arrays_for_key[i] for i in val_indices]
        test_data = [list_of_arrays_for_key[i] for i in test_indices]
        
        try:
            # Save as object arrays; requires allow_pickle=True on load
            if train_data: # Avoid error if a split is empty
                np.save(os.path.join(args.output_dir, f"{key}_train.npy"), np.array(train_data, dtype=object))
            else:
                print(f"Note: Train split for key '{key}' is empty.")
            
            if val_data:
                np.save(os.path.join(args.output_dir, f"{key}_val.npy"), np.array(val_data, dtype=object))
            else:
                print(f"Note: Validation split for key '{key}' is empty.")

            if test_data:
                np.save(os.path.join(args.output_dir, f"{key}_test.npy"), np.array(test_data, dtype=object))
            else:
                print(f"Note: Test split for key '{key}' is empty.")

            # Updated print statement
            train_info = f"{len(train_data)} items" + (f", e.g., first item shape: {train_data[0].shape}" if train_data and hasattr(train_data[0], 'shape') else "")
            val_info = f"{len(val_data)} items" + (f", e.g., first item shape: {val_data[0].shape}" if val_data and hasattr(val_data[0], 'shape') else "")
            test_info = f"{len(test_data)} items" + (f", e.g., first item shape: {test_data[0].shape}" if test_data and hasattr(test_data[0], 'shape') else "")
            print(f"Saved splits for key: {key} (Train: [{train_info}], Val: [{val_info}], Test: [{test_info}])")
        except Exception as e:
            print(f"Error saving data for key {key}: {e}")

    # 6. Store the shuffled order
    shuffled_order_path = os.path.join(args.output_dir, "shuffled_order_indices.npy")
    try:
        np.save(shuffled_order_path, indices)
        print(f"Saved shuffled order ({indices.shape}) to {shuffled_order_path}")
    except Exception as e:
        print(f"Error saving shuffled_order_indices.npy: {e}")

    print(f"Data setup script finished. Outputs are in {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process .npz files from an input directory. Each .npz file is treated as a single sample. \n" +
                    "The script assumes all processed .npz files contain the same set of keys, though arrays for these keys \n" +
                    "can have different dimensions within a file and across files for the same key. \n" +
                    "It aggregates these samples (as lists of arrays per key), splits them into train/validation/test sets, \n" +
                    "and saves them. Saved .npy files contain lists of arrays and require 'allow_pickle=True' when loading.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input_dir', type=str, default='curve_wireframe',
                        help="Directory containing .npz files (searched recursively). Default: 'curve_wireframe'. Ignored if --json_input_path is used.")
    parser.add_argument('--output_dir', type=str, default='data_split_output',
                        help="Directory to save the split data and shuffled order. Default: 'data_split_output'")
    
    parser.add_argument('--json_input_path', type=str, default=None,
                        help="Path to a single JSON file containing a list of data samples. If provided, --input_dir is ignored. Default: None")

    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help="Proportion of data for training. Default: 0.8")
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="Proportion of data for validation. Default: 0.1")
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help="Proportion of data for testing. Default: 0.1")
    
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for shuffling. Default: 42")
    
    parser.add_argument('--keys_to_process', type=str, nargs='*', default=None,
                        help="List of keys to process from the .npz files. If not provided, all keys found in the first valid file will be processed. Example: --keys_to_process key1 key2")

    parser.add_argument('--numerical_keys', type=str, nargs='*', default=[],
                        help="List of keys that, if processed, must correspond to list-like data in JSON (or any array in NPZ) "
                             "which converts to a non-scalar numpy array (i.e., array.shape != ()). "
                             "Samples/files failing this check for any such key will be skipped. Example: --numerical_keys points vectors")

    args = parser.parse_args()

    # Validate ratios
    ratios_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios_sum - 1.0) > 1e-7:
        parser.error(
            f"Train, val, and test ratios must sum to 1.0. \n"
            f"Current ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio} (sum={ratios_sum}). \n"
            f"Your original request mentioned '90%, 10%, 10%', which sums to 110%. "
            f"Please adjust the ratios to sum to 1.0 for disjoint sets."
        )
    
    # Validate individual ratios are non-negative
    if not (args.train_ratio >= 0 and args.val_ratio >= 0 and args.test_ratio >= 0):
        parser.error("Train, val, and test ratios must be non-negative.")

    main(args) 