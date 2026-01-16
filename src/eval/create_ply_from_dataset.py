import open3d as o3d
from pathlib import Path
import numpy as np
from tqdm import tqdm
if __name__ == '__main__':
    dir_name = "/home/qindafei/CAD/data/abc_step_pc_correct_normal/00"
    files = Path(dir_name).glob("*/*.npz")
    for file in tqdm(files):
        data = np.load(file, allow_pickle=True)
        points = data["points"]
        normals = data["normals"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(file.with_suffix(".ply"), pcd)
