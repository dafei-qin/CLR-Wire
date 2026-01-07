import trimesh
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch


def sample_mesh(points, faces, num_points=10240):
    mesh = trimesh.Trimesh(points, faces)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    normals = mesh.face_normals[face_indices]

    return np.concatenate([points, normals], axis=1)



def sample_points_from_mesh(vertices, faces, num_samples=1024):
    """
    从 mesh 表面随机采样点
    
    Args:
        vertices: (N, 3) numpy array
        faces: (M, 3) numpy array
        num_samples: 采样点数
    
    Returns:
        sampled_points: (num_samples, 3) numpy array
    """
    try:
        # 创建 trimesh 对象
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        # 从表面采样点
        sampled_points, _ = trimesh.sample.sample_surface(mesh, num_samples)
        
        return sampled_points
    except Exception as e:
        # 如果采样失败，直接使用顶点并随机采样
        if len(vertices) >= num_samples:
            indices = np.random.choice(len(vertices), num_samples, replace=False)
            return vertices[indices]
        else:
            # 顶点不够，重复采样
            indices = np.random.choice(len(vertices), num_samples, replace=True)
            return vertices[indices]


def compute_chamfer_distance_fast(pred_points, gt_points):
    """
    快速计算两个点云之间的 Chamfer Distance
    
    Args:
        pred_points: (N, 3) numpy array 或 torch tensor
        gt_points: (M, 3) numpy array 或 torch tensor
    
    Returns:
        chamfer_dist: 双向 Chamfer Distance
    """

    # 转换为 torch tensor（在 GPU 上）
    if not torch.is_tensor(pred_points):
        pred_points = torch.from_numpy(pred_points).float()
    if not torch.is_tensor(gt_points):
        gt_points = torch.from_numpy(gt_points).float()
    
    # 确保在 GPU 上
    if not pred_points.is_cuda:
        pred_points = pred_points.cuda()
    if not gt_points.is_cuda:
        gt_points = gt_points.cuda()
    
    # 计算双向 Chamfer Distance
    # pred -> gt
    dist_matrix = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0), p=2).squeeze(0)  # (N, M)
    min_dist_pred_to_gt = dist_matrix.min(dim=1)[0]  # (N,)
    
    # gt -> pred
    min_dist_gt_to_pred = dist_matrix.min(dim=0)[0]  # (M,)
    
    # Chamfer Distance (双向平均)
    chamfer_dist = (min_dist_pred_to_gt.mean() + min_dist_gt_to_pred.mean()).item()
    
    return chamfer_dist
    
 

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