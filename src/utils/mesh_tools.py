import trimesh
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm


def sample_mesh(points, faces, num_points=10240):
    mesh = trimesh.Trimesh(points, faces)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    normals = mesh.face_normals[face_indices]

    return np.concatenate([points, normals], axis=1)


if __name__ == '__main__':

    files = Path('../data/abc_objs_full/0').rglob('*obj')
    files = list(files)
    all_bounds = []
    for f in tqdm(files):
        mesh = trimesh.load_mesh(f)
        points, faces = mesh.vertices, mesh.faces
        sampled_points = sample_mesh(points, faces, num_points=10240)
        all_bounds.append(mesh.bounds)
        # print(sampled_points.shape)

    print()