import os
import sys
import fpsample
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':
    save_dir = '../data/temp_abc_step_pc_0009_fps'
    os.makedirs(save_dir, exist_ok=True)
    fps_num = 10240
    files = Path('../data/abc_step_pc_0009').rglob('*.npz')
    too_few = 0
    all_processed = 0
    for file in tqdm(files):
        data = np.load(str(file), allow_pickle=True)
        points = data['points']
        masks = data['masks']
        all_points = [p[m.astype(bool)] for p, m in zip(points, masks)]
        all_points = np.concatenate(all_points, axis=0)
        if len(all_points) < fps_num * 5:
            too_few += 1
            continue
        fps = fpsample.bucket_fps_kdtree_sampling(all_points, fps_num)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        o3d.io.write_point_cloud(os.path.join(save_dir, file.name.replace('.npz', '.ply')), pcd)
        pcd.points = o3d.utility.Vector3dVector(all_points[fps])
        o3d.io.write_point_cloud(os.path.join(save_dir, file.name.replace('.npz', '_fps.ply')), pcd)
        all_processed += 1

    print(f'all_processed: {all_processed}')
    print(f'too_few: {too_few}')
    print(f'too_few_ratio: {too_few / all_processed}')



