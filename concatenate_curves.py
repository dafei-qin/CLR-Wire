import numpy as np
import argparse
import os
def main(args):
    input_dir = args.input_dir
    train_file_path = os.path.join(input_dir, "edge_points_train.npy")
    val_file_path = os.path.join(input_dir, "edge_points_val.npy")
    test_file_path = os.path.join(input_dir, "edge_points_test.npy")

    train_data = np.load(train_file_path, allow_pickle=True)
    val_data = np.load(val_file_path, allow_pickle=True)
    test_data = np.load(test_file_path, allow_pickle=True)

    train_data = np.concatenate(train_data, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    test_data = np.concatenate(test_data, axis=0)

    np.save(os.path.join(input_dir, "edge_points_concat_train.npy"), train_data)
    np.save(os.path.join(input_dir, "edge_points_concat_val.npy"), val_data)
    np.save(os.path.join(input_dir, "edge_points_concat_test.npy"), test_data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/home/qindafei/CAD/curve_wireframe_split")
    main(parser.parse_args())