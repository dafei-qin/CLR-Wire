import numpy as np
import os
from collections import defaultdict
if __name__ == '__main__':
    data_dir = '../data/logan_jsons_cache_2/abc_test_updated_rot'
    files = os.listdir(data_dir)
    save_dir = '../data/logan_jsons_cache_2/abc_test_cache_2_updated_rot.npz'
    files = [f for f in files if 'npz' in f]
    print(files)
    total_data = defaultdict(list)
    for f in files:
        data = np.load(os.path.join(data_dir, f))
        for key in data.keys():
            total_data[key].append(data[key])

    total_data = {key: np.concatenate(value, axis=0) for key, value in total_data.items()}

    num_data = len(total_data['params'])
    for value in total_data.values():
        assert num_data == value.shape[0]

    np.savez(save_dir, **total_data)
    print('total samples: ', num_data)
