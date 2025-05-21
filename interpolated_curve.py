import numpy as np
from src.utils.numpy_tools import interpolate_1d

def main():
    data_path = ['/home/qindafei/CAD/abc_split/normalized_points_train.npy', '/home/qindafei/CAD/abc_split/normalized_points_val.npy', '/home/qindafei/CAD/abc_split/normalized_points_test.npy']
    t = np.linspace(0, 1, 64)
    for path in data_path:
        data = np.load(path, allow_pickle=True)
        print(data.shape)
        data_64 = [interpolate_1d(t, data[i]) for i in range(data.shape[0])]
        data_64 = np.array(data_64)
        print(data_64.shape)
        np.save(path.replace('normalized_points', 'normalized_points_64'), data_64)

if __name__ == '__main__':
    main()

    