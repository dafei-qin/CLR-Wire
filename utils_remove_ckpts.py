import os
import shutil
import sys
import argparse

def remove_ckpts(root_dir, keep_last=2):
    files = os.listdir(root_dir)
    files_valid = []
    for f in files:
        try:
            int(f.split('-')[-1].split('.')[0])
            files_valid.append(f)
        except:
            pass
    files_valid.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    print(files_valid)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--keep_last", type=int, default=2)
    args = parser.parse_args()
    root_dir = args.root_dir
    keep_last = args.keep_last


    for subfolder in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, subfolder)):
            remove_ckpts(os.path.join(root_dir, subfolder), keep_last)
