import torch
import os
import json
from typing import Dict
from pathlib import Path
import numpy as np
import dataclasses
import trimesh
import open3d as o3d
from sft.datasets.data_utils import load_process_mesh, center_vertices, normalize_vertices_scale, process_mesh, process_mesh_xr
from sft.datasets.serializaiton import serialize
from sft.datasets.serializaiton import deserialize
from sft.datasets.data_utils import to_mesh
from utils.common import init_logger, import_module_or_data
from collections import defaultdict

from datetime import datetime
from tqdm import tqdm
import time
import pymeshlab
import open3d as o3d
import igl

import matplotlib.pyplot as plt
logger = init_logger()

SYNSET_DICT_DIR = Path(__file__).resolve().parent  

class DynamicAttributes:
    def __init__(self):
        self._storage = {}

    def __setattr__(self, name, value):
        if name == '_storage':
            super().__setattr__(name, value)
        else:
            self._storage[name] = value

    def __getattr__(self, name):
        if name not in self._storage:
            new_obj = DynamicAttributes()
            self._storage[name] = new_obj
        return self._storage[name]
    
def sample_pc(verts, faces, pc_num, with_normal=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points
    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points[:,[2,0,1]], normals[:,[2,0,1]]], axis=-1, dtype=np.float16)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal


def generate_distinct_colors(n_colors):
    """生成n个有区分度的颜色"""
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
    # 已经包含alpha通道，直接转换为0-255范围的RGBA值
    colors = (colors * 255).astype(np.uint8)
    return colors
    
    


def sample_pc_noTrans(verts, faces, pc_num, with_normal=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points
    points, face_idx = mesh.sample(pc_num, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal

def farthest_point_sampling(points, n_clusters):
    """
    Performs farthest point sampling on a set of points.
    
    Args:
        points (np.ndarray): Input points of shape (N, 3)
        n_clusters (int): Number of points to sample
        
    Returns:
        np.ndarray: Indices of sampled points
    """
    # Convert to numpy array if needed
    points = np.array(points)
    N = points.shape[0]
    
    # Initialize arrays
    selected_indices = np.zeros(n_clusters, dtype=np.int32)
    distances = np.full(N, np.inf)
    
    # Randomly choose first point
    current_idx = np.random.randint(0, N)
    selected_indices[0] = current_idx
    
    # Iteratively select farthest points
    for i in range(1, n_clusters):
        # Update distances to set of selected points
        current_point = points[current_idx]
        dist = np.sum((points - current_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        
        # Select farthest point
        current_idx = np.argmax(distances)
        selected_indices[i] = current_idx
        
    return selected_indices

def line_plane_intersection(p1, p2, plane_point, plane_normal):
    direction = p2 - p1
    dot_normal = np.dot(direction, plane_normal)
    if abs(dot_normal) < 1e-6:
        return None
    t = np.dot(plane_point - p1, plane_normal) / dot_normal
    if t < 0 or t > 1:
        return None
    return p1 + t * direction

def get_normalized_triangle_orientation(v1, v2, v3):
    normal = np.cross(v2 - v1, v3 - v1)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        # 返回全0或全nan，或者raise异常，视你的需求
        return np.zeros_like(normal)
    return normal / norm

def orient_face(face, verts, ori_normal):
    v1, v2, v3 = [verts[i] for i in face]
    normal = get_normalized_triangle_orientation(v1, v2, v3)
    if np.dot(normal, ori_normal) < -1e-8:
        # 交换后两个顶点
        return [face[0], face[2], face[1]]
    return face

def random_selection_mesh(verts, faces, fix=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    verts = mesh.vertices
    faces = mesh.faces

    indices  = np.random.choice(50000, 16384, replace=False)
    pc_global_data       = sample_pc(verts, faces, pc_num=50000, with_normal=True)[indices]
    # mesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/original_mesh_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
    facecnt = len(faces)
    basenum = facecnt / 2000
    # 随机上下浮动
    basenum = np.random.uniform(0.5, 2.5) * basenum
    # int
    basenum = int(basenum)
    if basenum < 1:
        basenum = 1

    if basenum > 30:
        basenum = 30
    cluster_num = basenum + 1 # change back
    indices = np.random.choice(50000, 16384, replace=False)
    pc_global = sample_pc(verts, faces, pc_num=50000, with_normal=False)[indices]
    points = pc_global

    fps_indices = farthest_point_sampling(points, cluster_num)
    fps_points = points[fps_indices]

    # indices = np.random.choice(16384 * cluster_num * 200, 16384 * cluster_num * 100, replace=False)
    # pc_big_normal = sample_pc_noTrans(verts, faces, pc_num=16384 * cluster_num * 200, with_normal=True)[indices]
    # pc_big = pc_big_normal[:, :3]   
    # pc_big_normal = pc_big_normal[:, 3:]
    # 计算 pc_big 到哪个fps_points最近
    # pc_bin_mindis = np.ones(pc_big.shape[0]) * np.inf
    # pc_bin_labels = np.ones(pc_big.shape[0], dtype=np.int32) * -1
    # for i in range(len(fps_points)):
    #     distances = np.linalg.norm(pc_big - fps_points[i], axis=1)
    #     mask = distances < pc_bin_mindis
    #     pc_bin_mindis[mask] = distances[mask]
    #     pc_bin_labels[mask] = i
    # to do ...
    

    # 计算面片三个点的 label
    face_points = verts[faces]
    # print(f"计算{len(face_centers)}个面片中心点到{len(fps_points)}个采样点的测地距离...")

    face_mindis = np.ones((len(face_points),3)) * np.inf
    face_labels = np.ones((len(face_points),3), dtype=np.int32) * -1

    # For each fps point, calculate distances to all face centers
    for i in range(len(fps_points)):
        # Calculate Euclidean distances from current fps point to all face centers
        dis = np.linalg.norm(face_points - fps_points[i], axis=2)  # (num_faces, 3)
        
        # Update labels and distances if current distance is smaller than minimum
        mask = dis < face_mindis
        face_mindis[mask] = dis[mask]
        face_labels[mask] = i




    nface_centers = np.mean(verts[faces], axis=1)
    nface_mindis = np.ones(len(nface_centers)) * np.inf
    nface_labels = np.ones(len(nface_centers), dtype=np.int32) * -1
    
    # For each fps point, calculate distances to all face centers
    for i in range(len(fps_points)):
        # Calculate Euclidean distances from current fps point to all face centers
        dis = np.linalg.norm(nface_centers - fps_points[i], axis=1)  # (num_faces, 3)
        
        # Update labels and distances if current distance is smaller than minimum
        mask = dis < nface_mindis
        nface_mindis[mask] = dis[mask]
        nface_labels[mask] = i

    # 创建类别之间的邻接矩阵，遍历所有面片，记录所有相邻的类别
    # print(f"创建类别之间的邻接矩阵...")
    # 创建类别之间的邻接矩阵
    n_classes = len(fps_points)
    adj_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    
    # 遍历所有面片的邻接关系
    for i in range(len(mesh.face_adjacency)):
        if mesh.face_adjacency[i].shape[0] == 1:
            continue
        if mesh.face_adjacency[i].shape[0] == 2:
            face1 = mesh.face_adjacency[i][0]
            face2 = mesh.face_adjacency[i][1]

            # todo: 处理非流形
        label1 = nface_labels[face1]
        label2 = nface_labels[face2]
        if label1 != label2:
            adj_matrix[label1][label2] = 1
            adj_matrix[label2][label1] = 1

    degree = np.sum(adj_matrix, axis=1)
    # 查找相邻类最多的那一类
    max_neighbors = np.max(np.sum(adj_matrix, axis=1))
    max_neighbors_class = np.argmax(np.sum(adj_matrix, axis=1))
    # print(f"相邻类最多的一类: {max_neighbors_class}, 相邻类数量: {max_neighbors}, 总类数: {n_classes}")
    class_idx = max_neighbors_class

    # 根据 fps_points 从下到上从左到右给 cluster 排序
    # 计算 fps_points 的坐标
    fps_points_coords = fps_points[:, :3]

    # 根据 fps_points 的坐标从下到上（z轴）排序
    fps_points_order = np.lexsort((fps_points_coords[:, 0], fps_points_coords[:, 1], fps_points_coords[:, 2]))
    # 根据 fps_points_order 给 cluster 排序
    class_idx = fps_points_order[0]
    # colors = generate_distinct_colors(cluster_num)  
    # point_colors = colors[pc_bin_labels]

    # # random sample 16384 points
    # indices = np.random.choice(len(pc_big), 16384, replace=False)
    # points_to_sample = pc_big[indices]
    # point_colors_to_sample = point_colors[indices]
    # colored_pc = trimesh.PointCloud(
    #     vertices=points_to_sample,
    #     colors=point_colors_to_sample
    # )
    # # Save colored point cloud
    # colored_pc.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/segmented_points_time{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
    # print(f"Loading new mesh with {cluster_num} segments...")

    # 在进入循环前初始化新的faces列表
    new_faces = []
    cpverts = verts.copy()  # 创建顶点的副本，因为我们会添加新的顶点
    data = []
    used_cluster = []



    visited = set()
    queue = [class_idx]
    # if max_neighbors == 0:
    #     # 筛选所有链接数为0的类
    #     class_idx = np.where(np.sum(adj_matrix, axis=1) == 0)[0][0]
    #     queue = [class_idx]
    reconed_meshs = []
    idx = -1
    while len(visited) < n_classes:
        # print(f"visited: {len(visited)}, queue: {len(queue)}")
        if len(queue) == 0:
            for j in fps_points_order:
                if j not in visited:
                    queue.append(j)
                    break
            # notvisited = [i for i in range(n_classes) if i not in visited]
            # # 寻找联通分支最多的一个，or度数最大的一个
            # max_degree = np.max(degree[notvisited])
            # max_degree_class = np.argmax(degree[notvisited])
            # class_idx = notvisited[max_degree_class]
            # queue = [class_idx]
        cluster = queue.pop(0)
        if cluster in visited:
            continue 
        visited.add(cluster)
        idx += 1
        # 将相邻的类别加入队列
        for j in range(n_classes):
            if adj_matrix[cluster][j] == 1 and j not in visited:
                queue.append(j)


        data_i = {}
        data_i['idx'] = idx
        # pc_big_i = pc_big[pc_bin_labels == cluster]
        # pc_big_normal_i = pc_big_normal[pc_bin_labels == cluster]
        # indices = np.random.choice(pc_big_i.shape[0], 16384, replace=False)
        # pc_big_i = pc_big_i[indices]
        # pc_big_normal_i = pc_big_normal_i[indices]
        # data_i['pc_local'] = torch.cat([torch.tensor(pc_big_i[:,[2,0,1]]), torch.tensor(pc_big_normal_i[:,[2,0,1]])], dim=-1)
        
        # save pc_local 
        # pc_local = trimesh.PointCloud(vertices=pc_big_i)
        # pc_local.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/pc_local_{idx}.ply")
       
        # face_labels is [n,3]
        face_mask = np.all(face_labels == cluster, axis=-1)  # shape: (num_faces,)
        selected_faces = list(faces[face_mask])  # 转换为Python列表
        bd_faces = []
        face2cut_mask = np.any(face_labels == cluster, axis=-1) & ~np.all(face_labels == cluster, axis=-1)
        face2cutlist = np.where(face2cut_mask)[0]
        if len(face2cutlist) == 0 and len(selected_faces) == 0:
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data.append(data_i)
            continue
        data_i['skip'] = False
        for face_idx in face2cutlist:
            labels = face_labels[face_idx]
            unique_labels = np.unique(labels)
            
            if len(unique_labels) == 2:
                # 找出唯一的label和重复的label
                counts = [np.sum(labels == l) for l in unique_labels]
                if counts[0] == 1:
                    unique_label = unique_labels[0]
                    shared_label = unique_labels[1]
                else:
                    unique_label = unique_labels[1]
                    shared_label = unique_labels[0]
                # 找到唯一label的顶点索引
                unique_idx = np.where(labels == unique_label)[0][0]
                shared_idx = np.where(labels == shared_label)[0]
                # 三个顶点
                face_verts = faces[face_idx]
                v0_idx = face_verts[unique_idx]
                v1_idx = face_verts[shared_idx[0]]
                v2_idx = face_verts[shared_idx[1]]
                v0 = cpverts[v0_idx]
                v1 = cpverts[v1_idx]
                v2 = cpverts[v2_idx]
                # 计算面片法向量
                ori_face_normal = get_normalized_triangle_orientation(cpverts[face_verts[0]], cpverts[face_verts[1]], cpverts[face_verts[2]])
                # 计算交点
                sourceP1 = fps_points[unique_label]
                sourceP2 = fps_points[shared_label]
                midpoint = (sourceP1 + sourceP2) / 2
                normal = sourceP2 - sourceP1
                normal = normal / np.linalg.norm(normal)
                p = line_plane_intersection(v0, v1, midpoint, normal)
                q = line_plane_intersection(v0, v2, midpoint, normal)
                if p is not None and q is not None:
                    p_idx = len(cpverts)
                    q_idx = len(cpverts) + 1
                    cpverts = np.vstack([cpverts, p, q])
                    # 三角化
                    tris = [
                        [v0_idx, p_idx, q_idx],
                        [p_idx, v1_idx, v2_idx],
                        [p_idx, v2_idx, q_idx]
                    ]
                    tris = [orient_face(tri, cpverts, ori_face_normal) for tri in tris]
                    new_faces.extend(tris)
                    if unique_label == cluster:
                        selected_faces.append(tris[0])
                        if shared_label in used_cluster:
                            bd_faces.append(tris[1])
                            bd_faces.append(tris[2])
                    else:
                        selected_faces.append(tris[2])
                        selected_faces.append(tris[1])
                        if unique_label in used_cluster:
                            bd_faces.append(tris[0])
            
            elif len(unique_labels) == 3:
                # 处理三个label的情况
                unique_idx = np.where(labels == cluster)[0][0]
                shared_idx = np.where(labels != cluster)[0]
                face_verts = faces[face_idx]
                v0_idx = face_verts[unique_idx]
                v1_idx = face_verts[shared_idx[0]]
                v2_idx = face_verts[shared_idx[1]]
                v0 = cpverts[v0_idx]
                v1 = cpverts[v1_idx]
                v2 = cpverts[v2_idx]
                ori_face_normal = get_normalized_triangle_orientation(cpverts[face_verts[0]], cpverts[face_verts[1]], cpverts[face_verts[2]])
                sourceP1 = fps_points[cluster]
                sourceP2 = fps_points[labels[shared_idx[0]]]
                sourceP3 = fps_points[labels[shared_idx[1]]]
                midpoint1 = (sourceP1 + sourceP2) / 2
                normal1 = sourceP2 - sourceP1
                normal1 = normal1 / np.linalg.norm(normal1)
                p = line_plane_intersection(v0, v1, midpoint1, normal1)
                midpoint2 = (sourceP1 + sourceP3) / 2
                normal2 = sourceP3 - sourceP1
                normal2 = normal2 / np.linalg.norm(normal2)
                q = line_plane_intersection(v0, v2, midpoint2, normal2)
                if p is not None and q is not None:
                    p_idx = len(cpverts)
                    q_idx = len(cpverts) + 1
                    cpverts = np.vstack([cpverts, p, q])
                    tris = [
                        [v0_idx, p_idx, q_idx],
                        [p_idx, v1_idx, v2_idx],
                        [p_idx, v2_idx, q_idx]
                    ]
                    tris = [orient_face(tri, cpverts, ori_face_normal) for tri in tris]
                    selected_faces.append(tris[0])
                    if shared_idx[0] in used_cluster or shared_idx[1] in used_cluster:
                        bd_faces.append(tris[1])
                        bd_faces.append(tris[2])
                    new_faces.extend(tris)

        # 创建临时mesh用于导出，将selected_faces转换回numpy数组
        selected_faces = np.array(selected_faces)
        temp_mesh = trimesh.Trimesh(vertices=cpverts, faces=selected_faces)
        # data_i['selected_verts'] = torch.tensor(temp_mesh.vertices)
        # data_i['selected_faces'] = torch.tensor(temp_mesh.faces)
        selected_verts = temp_mesh.vertices
        selected_faces = temp_mesh.faces
        # temp_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/selected_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        bdd = True 
        # mask1 = np.any(np.isin(face_labels, used_cluster), axis=-1)
        mask = np.all(face_labels != cluster, axis=-1) & np.any(np.isin(face_labels, used_cluster), axis=-1)

        if np.sum(mask) <= 1:
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            bdd = False
        elif len(bd_faces) < 512:

            # 在其他 cluster 中寻找距离当前 cluster 最近的20个面
            other_cluster_faces = np.all(face_labels != cluster, axis=-1) & np.any(np.isin(face_labels, used_cluster), axis=-1)
            other_cluster_faces = faces[other_cluster_faces]
            face_points = verts[other_cluster_faces]
            initial_center = fps_points[cluster]
        # 计算每个面三个点到 initial_center 的最小的那个距离
            # distances = np.min(np.linalg.norm(face_points - initial_center, axis=2), axis=-1)
            distances = np.ones(other_cluster_faces.shape[0]) * 1000000000
            for iddx in range(other_cluster_faces.shape[0]):
                distances[iddx] = np.min(np.linalg.norm(face_points[iddx][0] - selected_verts, axis=-1))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][1] - selected_verts, axis=-1)))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][2] - selected_verts, axis=-1)))
            # 找到距离初始面皮最近并且不属于 selected_faces 的100个面
            boundry_faces = np.argsort(distances)[:512 - len(bd_faces)]
            # 将 bd_faces 添加到 boundry_faces 中
            if len(bd_faces) > 0:
                tmpmesh = trimesh.Trimesh(vertices=cpverts, faces=bd_faces)
                tmpmesh1 = trimesh.Trimesh(vertices=verts, faces=other_cluster_faces[boundry_faces])
                tmpmesh2 = trimesh.Trimesh(vertices=np.concatenate([tmpmesh.vertices, tmpmesh1.vertices], axis=0), faces=np.concatenate([tmpmesh.faces, tmpmesh1.faces + tmpmesh.vertices.shape[0]], axis=0))
                boundry_faces = tmpmesh2.faces
                boundry_verts = tmpmesh2.vertices
            else:
                boundry_faces = other_cluster_faces[boundry_faces]
            
                tmpmesh = trimesh.Trimesh(vertices=verts, faces=boundry_faces)
                boundry_verts = tmpmesh.vertices
                boundry_faces = tmpmesh.faces
            # data_i['bd_verts'] = torch.tensor(boundry_verts)
            # data_i['bd_faces'] = torch.tensor(boundry_faces)
            
        else:
            bdmesh = trimesh.Trimesh(vertices=cpverts, faces=bd_faces)
            face_points = bdmesh.vertices[bdmesh.faces]
            initial_center = fps_points[cluster]
            # 计算每个面三个点到 initial_center 的最小的那个距离
            # distances = np.min(np.linalg.norm(face_points - initial_center, axis=2), axis=-1)
            distances = np.ones(bdmesh.faces.shape[0]) * 1000000000
            for iddx in range(bdmesh.faces.shape[0]):
                distances[iddx] = np.min(np.linalg.norm(face_points[iddx][0] - selected_verts, axis=-1))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][1] - selected_verts, axis=-1)))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][2] - selected_verts, axis=-1)))
            # 找到距离初始面皮最近并且不属于 selected_faces 的100个面
            boundry_faces = np.argsort(distances)[:512]
            # bd_faces = np.array(bd_faces)
            tmpmesh = trimesh.Trimesh(vertices=bdmesh.vertices, faces=bdmesh.faces[boundry_faces])
            boundry_verts = tmpmesh.vertices
            boundry_faces = tmpmesh.faces
            # data_i['bd_verts'] = torch.tensor(boundry_verts)
            # data_i['bd_faces'] = torch.tensor(boundry_faces)
        # data_i['bd_token_length'] = len(boundry_faces)
        seleVL = selected_verts.shape[0]
        if bdd:
            all_verts = np.concatenate([selected_verts, boundry_verts], axis=0)
            # 正确处理 boundary_faces 的索引偏移
            # boundary_faces_offset = boundry_faces + selected_verts.shape[0]  # 对每个元素都加上偏移量
            # all_faces = np.concatenate([selected_faces, boundary_faces_offset], axis=0)
            # tmpmesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/boundry_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        else:
            all_verts = selected_verts
            # all_faces = selected_faces
        if len(all_verts) <= 10 or len(selected_faces) <= 10:
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        all_verts, center, scale = rescale_verts(all_verts)
        data_i['center'] = center
        data_i['scale'] = scale
        localmesh = trimesh.Trimesh(vertices=all_verts[:seleVL], faces=selected_faces)
        try:
            localmesh = process_mesh_xr(localmesh.vertices, localmesh.faces, augment=False)
        except:
            print(f"localmesh.vertices.shape: {localmesh.vertices.shape}, localmesh.faces.shape: {localmesh.faces.shape}")
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        selected_verts = localmesh['vertices']
        selected_faces = localmesh['faces']
        if bdd:
            if len(all_verts[seleVL:]) <= 10 or len(boundry_faces) <= 10:
                boundary_verts = []
                boundary_faces = []
                data_i['bd_token_length'] = 0
                bdd = False
            else:
                localbdmesh = trimesh.Trimesh(vertices=all_verts[seleVL:], faces=boundry_faces)
                # export localbdmesh to ply
                
                try:
                    localbdmesh = process_mesh_xr(localbdmesh.vertices, localbdmesh.faces, augment=False)
                    # localdmesh = trimesh.Trimesh(vertices=selected_verts, faces=selected_faces)
                    # localbddmesh = trimesh.Trimesh(vertices=localbdmesh['vertices'], faces=localbdmesh['faces'])
                    # localdmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/localmesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply")
                    # localbddmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/localbdmesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply")
                except:
                    print(f"localbdmesh.vertices.shape: {localbdmesh.vertices.shape}, localbdmesh.faces.shape: {localbdmesh.faces.shape}")
                    data_i['bd_token_length'] = 0
                    # data_i['bd_verts'] = torch.tensor([])
                    # data_i['bd_faces'] = torch.tensor([])
                    boundary_verts = []
                    boundary_faces = []
                    bdd = False
                else:
                    boundary_verts = localbdmesh['vertices']
                    boundary_faces = localbdmesh['faces']
        else:
            boundary_verts = []
            boundary_faces = []
        token_list = serialize(trimesh.Trimesh(vertices=selected_verts, faces=selected_faces))
        if bdd:
            bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces))
            # print(f"bd_token_list length: {len(bd_token_list)}")
            facetodelete = 32
            while len(bd_token_list) > 2000:
                bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces[:-facetodelete]))
                facetodelete += 32
                # print(f"bd_token_list new length: {len(bd_token_list)}")
            block_size = 8
            offset_size = 16
            patch_size = 4
            special_block_base = block_size**3 + offset_size**3 + patch_size**3
            token_list[0] += special_block_base
            token_list = np.concatenate([[4736], bd_token_list, token_list, [4737]])
            data_i['bd_token_length'] = len(bd_token_list)
        else:
            token_list = np.concatenate([[4736], token_list, [4737]])
        data_i['token_list'] = torch.tensor(token_list, dtype=torch.long)


        if bdd and False:
            codee = bd_token_list.astype(np.int64)
            vertices = deserialize(codee)
            xxfaces = torch.arange(1, len(vertices) + 1).view(-1, 3)
            meshbd = to_mesh(vertices, xxfaces, transpose=False, post_process=False)
            codee = token_list.astype(np.int64)
            vertices = deserialize(codee)
            xxfaces = torch.arange(1, len(vertices) + 1).view(-1, 3)
            meshss = to_mesh(vertices, xxfaces, transpose=False, post_process=False)
            vecs = np.concatenate([meshbd.vertices, meshss.vertices], axis=0)
        
            facess = np.concatenate([meshbd.faces[:,[0,2,1]], meshss.faces + meshbd.vertices.shape[0]], axis=0)
            meshss = trimesh.Trimesh(vertices=vecs, faces=facess)
            meshss.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply") 


        # if len(data_i['token_list']) > 15000:
            # print(f"token_list length > 15000: {len(data_i['token_list'])}")
            # data_i['token_list'] = data_i['token_list'][:15000]
        indices = np.random.choice(50000, 16384, replace=False)
        pc_local = sample_pc(selected_verts, selected_faces, pc_num=50000, with_normal=True)[indices]
        data_i['pc_local'] = torch.tensor(pc_local)
        data_i['pc_global'] = torch.tensor(pc_global_data)
        data_i['cluster'] = fps_points[cluster]
        
        used_cluster.append(cluster)
        # save pc_local and pc_global to ply
        # pc_local_points = pc_local[:, :3]
        # pc_local_normals = pc_local[:, 3:]
        # pc_local_mesh = trimesh.PointCloud(vertices=pc_local_points, colors=pc_local_normals)
        # pc_local_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/pc_local_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        # pc_global_points = pc_global_data[:, :3]
        # pc_global_normals = pc_global_data[:, 3:]
        # pc_global_mesh = trimesh.PointCloud(vertices=pc_global_points, colors=pc_global_normals)
        # pc_global_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/pc_global_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")

        # tmpmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/boundry_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        data.append(data_i)
    return data



def selection_mesh_from_connected_components(verts, faces, components, fix=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    verts = mesh.vertices
    faces = mesh.faces

    indices  = np.random.choice(50000, 16384, replace=False)
    pc_global_data       = sample_pc(verts, faces, pc_num=50000, with_normal=True)[indices]
    # mesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/original_mesh_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
    
    cluster_num = len(components)

    class_idx = -1
    minz = 1000000000
    for i in range(len(components)):
        verts_i = components[i].vertices
        if np.min(verts_i[:, 2]) < minz:
            minz = np.min(verts_i[:, 2])
            class_idx = i

    # 在进入循环前初始化新的faces列表
    new_faces = []
    cpverts = verts.copy()  # 创建顶点的副本，因为我们会添加新的顶点
    data = []
    used_cluster = []



    visited = set()
    queue = [class_idx]
    # if max_neighbors == 0:
    #     # 筛选所有链接数为0的类
    #     class_idx = np.where(np.sum(adj_matrix, axis=1) == 0)[0][0]
    #     queue = [class_idx]
    reconed_meshs = []
    idx = -1
    last_verts = np.zeros((0, 3), dtype=components[0].vertices.dtype)
    last_faces = np.zeros((0, 3), dtype=components[0].faces.dtype)
    while len(visited) < cluster_num:
        # print(f"visited: {len(visited)}, queue: {len(queue)}")
        if len(queue) == 0:
            for j in range(len(components)):
                if j not in visited:
                    queue.append(j)
                    break
            # notvisited = [i for i in range(n_classes) if i not in visited]
            # # 寻找联通分支最多的一个，or度数最大的一个
            # max_degree = np.max(degree[notvisited])
            # max_degree_class = np.argmax(degree[notvisited])
            # class_idx = notvisited[max_degree_class]
            # queue = [class_idx]
        cluster = queue.pop(0)
        if cluster in visited:
            continue 
        visited.add(cluster)
        idx += 1
        # 将相邻的类别加入队列
        for j in range(len(components)):
            if j not in visited:
                verts_j = components[j].vertices
                verts_i = components[cluster].vertices
                # 计算两个点集之间最短的两个点之间的距离
                dists = np.linalg.norm(verts_j[:, None, :] - verts_i[None, :, :], axis=-1)
                dist = np.min(dists)
                if dist < 0.01:
                    queue.append(j)
                    
        data_i = {}
        data_i['idx'] = idx
        # pc_big_i = pc_big[pc_bin_labels == cluster]
        # pc_big_normal_i = pc_big_normal[pc_bin_labels == cluster]
        # indices = np.random.choice(pc_big_i.shape[0], 16384, replace=False)
        # pc_big_i = pc_big_i[indices]
        # pc_big_normal_i = pc_big_normal_i[indices]
        # data_i['pc_local'] = torch.cat([torch.tensor(pc_big_i[:,[2,0,1]]), torch.tensor(pc_big_normal_i[:,[2,0,1]])], dim=-1)
        
        # save pc_local 
        # pc_local = trimesh.PointCloud(vertices=pc_big_i)
        # pc_local.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/pc_local_{idx}.ply")
       
        
        selected_verts = components[cluster].vertices
        selected_faces = components[cluster].faces
        if len(selected_faces) + len(last_faces) <= 1000 and len(visited) < cluster_num and False:
            lenvv = len(last_verts)
            last_verts = np.concatenate([last_verts, selected_verts], axis=0)
            last_faces = np.concatenate([last_faces, selected_faces + lenvv], axis=0)
            continue
        else:
            selected_verts = np.concatenate([last_verts, selected_verts], axis=0)
            selected_faces = np.concatenate([last_faces, selected_faces+len(last_verts)], axis=0)
            last_verts = np.zeros((0, 3), dtype=components[0].vertices.dtype)
            last_faces = np.zeros((0, 3), dtype=components[0].faces.dtype)
            # meshtmp = trimesh.Trimesh(vertices=selected_verts, faces=selected_faces)
            # meshtmp.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/Merge_examples/last_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")

        if len(selected_verts) <= 3 or len(selected_faces) <= 3:
            data_i['skip'] = True
            data_i['bd_token_length'] = 0
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data_i['cluster'] = None
            data.append(data_i)
            continue
        # temp_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/selected_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        if idx == 0:
            bdd = False
        else:
            bdd = True
        if idx == 0:
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            bdd = False
        else:
            boundry_verts = np.zeros((0, 3), dtype=components[cluster].vertices.dtype)
            boundry_faces = np.zeros((0, 3), dtype=components[cluster].faces.dtype)
            for j in visited:
                verts_j = components[j].vertices
                faces_j = components[j].faces
                lenv = len(boundry_verts)
                boundry_verts = np.concatenate([boundry_verts, verts_j], axis=0)
                boundry_faces = np.concatenate([boundry_faces, faces_j + lenv], axis=0)
            bdmesh = trimesh.Trimesh(vertices=boundry_verts, faces=boundry_faces)
            face_points = bdmesh.vertices[bdmesh.faces]
            # 计算每个面三个点到 initial_center 的最小的那个距离
            # distances = np.min(np.linalg.norm(face_points - initial_center, axis=2), axis=-1)
            distances = np.ones(bdmesh.faces.shape[0]) * 1000000000
            for iddx in range(bdmesh.faces.shape[0]):
                distances[iddx] = np.min(np.linalg.norm(face_points[iddx][0] - selected_verts, axis=-1))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][1] - selected_verts, axis=-1)))
                distances[iddx] = min(distances[iddx], np.min(np.linalg.norm(face_points[iddx][2] - selected_verts, axis=-1)))
            # 找到距离初始面皮最近并且不属于 selected_faces 的100个面
            boundry_faces = np.argsort(distances)[:512]
            # bd_faces = np.array(bd_faces)
            tmpmesh = trimesh.Trimesh(vertices=bdmesh.vertices, faces=bdmesh.faces[boundry_faces])
            boundry_verts = tmpmesh.vertices
            boundry_faces = tmpmesh.faces
            # data_i['bd_verts'] = torch.tensor(boundry_verts)
            # data_i['bd_faces'] = torch.tensor(boundry_faces)
        # data_i['bd_token_length'] = len(boundry_faces)
        seleVL = selected_verts.shape[0]
        if bdd:
            all_verts = np.concatenate([selected_verts, boundry_verts], axis=0)
            # 正确处理 boundary_faces 的索引偏移
            # boundary_faces_offset = boundry_faces + selected_verts.shape[0]  # 对每个元素都加上偏移量
            # all_faces = np.concatenate([selected_faces, boundary_faces_offset], axis=0)
            # tmpmesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/boundry_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        else:
            all_verts = selected_verts
            # all_faces = selected_faces
        if len(all_verts) <= 10 or len(selected_faces) <= 10:
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        all_verts, center, scale = rescale_verts(all_verts)
        data_i['center'] = center
        data_i['scale'] = scale
        localmesh = trimesh.Trimesh(vertices=all_verts[:seleVL], faces=selected_faces)
        try:
            localmesh = process_mesh_xr(localmesh.vertices, localmesh.faces, augment=False)
        except:
            print(f"localmesh.vertices.shape: {localmesh.vertices.shape}, localmesh.faces.shape: {localmesh.faces.shape}")
            data_i['bd_token_length'] = 0
            # data_i['bd_verts'] = torch.tensor([])
            # data_i['bd_faces'] = torch.tensor([])
            data_i['token_list'] = torch.tensor([4736, 4737], dtype=torch.long)
            data_i['skip'] = True
            data_i['pc_local'] = torch.tensor([])
            data_i['pc_global'] = torch.tensor([])
            data.append(data_i)
            continue
        selected_verts = localmesh['vertices']
        selected_faces = localmesh['faces']
        if bdd:
            if len(all_verts[seleVL:]) <= 10 or len(boundry_faces) <= 20:
                boundary_verts = []
                boundary_faces = []
                data_i['bd_token_length'] = 0
                bdd = False
            else:
                localbdmesh = trimesh.Trimesh(vertices=all_verts[seleVL:], faces=boundry_faces)
                # export localbdmesh to ply
                
                try:
                    localbdmesh = process_mesh_xr(localbdmesh.vertices, localbdmesh.faces, augment=False)
                    # localdmesh = trimesh.Trimesh(vertices=selected_verts, faces=selected_faces)
                    # localbddmesh = trimesh.Trimesh(vertices=localbdmesh['vertices'], faces=localbdmesh['faces'])
                    # localdmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/localmesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply")
                    # localbddmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/localbdmesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply")
                except:
                    print(f"localbdmesh.vertices.shape: {localbdmesh.vertices.shape}, localbdmesh.faces.shape: {localbdmesh.faces.shape}")
                    data_i['bd_token_length'] = 0
                    # data_i['bd_verts'] = torch.tensor([])
                    # data_i['bd_faces'] = torch.tensor([])
                    boundary_verts = []
                    boundary_faces = []
                    bdd = False
                else:
                    boundary_verts = localbdmesh['vertices']
                    boundary_faces = localbdmesh['faces']
        else:
            boundary_verts = []
            boundary_faces = []
        token_list = serialize(trimesh.Trimesh(vertices=selected_verts, faces=selected_faces))
        if bdd:
            bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces))
            # print(f"bd_token_list length: {len(bd_token_list)}")
            facetodelete = 32
            while len(bd_token_list) > 2000:
                bd_token_list = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces[:-facetodelete]))
                facetodelete += 32
                # print(f"bd_token_list new length: {len(bd_token_list)}")
            block_size = 8
            offset_size = 16
            patch_size = 4
            special_block_base = block_size**3 + offset_size**3 + patch_size**3
            token_list[0] += special_block_base
            token_list = np.concatenate([[4736], bd_token_list, token_list, [4737]])
            data_i['bd_token_length'] = len(bd_token_list)
        else:
            token_list = np.concatenate([[4736], token_list, [4737]])
        data_i['token_list'] = torch.tensor(token_list, dtype=torch.long)


        if bdd and False:
            codee = bd_token_list.astype(np.int64)
            vertices = deserialize(codee)
            xxfaces = torch.arange(1, len(vertices) + 1).view(-1, 3)
            meshbd = to_mesh(vertices, xxfaces, transpose=False, post_process=False)
            codee = token_list.astype(np.int64)
            vertices = deserialize(codee)
            xxfaces = torch.arange(1, len(vertices) + 1).view(-1, 3)
            meshss = to_mesh(vertices, xxfaces, transpose=False, post_process=False)
            vecs = np.concatenate([meshbd.vertices, meshss.vertices], axis=0)
        
            facess = np.concatenate([meshbd.faces[:,[0,2,1]], meshss.faces + meshbd.vertices.shape[0]], axis=0)
            meshss = trimesh.Trimesh(vertices=vecs, faces=facess)
            meshss.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/LTT_examples/mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S%f')}.ply") 


        # if len(data_i['token_list']) > 15000:
            # print(f"token_list length > 15000: {len(data_i['token_list'])}")
            # data_i['token_list'] = data_i['token_list'][:15000]
        indices = np.random.choice(50000, 16384, replace=False)
        pc_local = sample_pc(selected_verts, selected_faces, pc_num=50000, with_normal=True)[indices]
        data_i['pc_local'] = torch.tensor(pc_local)
        data_i['pc_global'] = torch.tensor(pc_global_data)
        data_i['cluster'] = None
        data_i['skip'] = False
        
        used_cluster.append(cluster)
        # save pc_local and pc_global to ply
        # pc_local_points = pc_local[:, :3]
        # pc_local_normals = pc_local[:, 3:]
        # pc_local_mesh = trimesh.PointCloud(vertices=pc_local_points, colors=pc_local_normals)
        # pc_local_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/pc_local_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        # pc_global_points = pc_global_data[:, :3]
        # pc_global_normals = pc_global_data[:, 3:]
        # pc_global_mesh = trimesh.PointCloud(vertices=pc_global_points, colors=pc_global_normals)
        # pc_global_mesh.export(f"/data6/ruixu/code/2025_MeshGen/DeepMesh0621/examples/pc_global_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")

        # tmpmesh.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/boundry_mesh_{idx}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.ply")
        data.append(data_i)
    return data


def rescale_verts(verts):
    # Transpose so that z-axis is vertical.
    verts = verts[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    center = (verts.max(0) + verts.min(0)) / 2
    verts = verts - center

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    scale = np.sqrt(np.sum((verts.max(0) - verts.min(0)) ** 2))
    verts = verts / scale

    return verts, center, scale

@dataclasses.dataclass
class MeshMeta:
    model_id: str
    raw_obj: str = None
    face_info: dict = None
    category: str = None
    category_path_en: str = None
    obj_path: str = None
    model_type: str = None
    face_cnt: int = None

class UnionSet:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: 
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

class Sample_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        quant_bit: int     = 9,
        point_num:int      = 16384,
        path:str           = "",
        uid_list:list      = []
    ) -> None:
        super().__init__()
        self.quant_bit    = quant_bit
        self.point_num    = point_num
        self.path         = path
        # self.s3link = 's3://data-repo-2-1258344700/primexu/3d_datasets/20250425_224244/object_no_repeat_pull_at_20250425_215702_train_samples_0.json'
        # self.s3 = GameaiCosSign("ruiiixu", "GXZRShV1gtpfTGFx4xnzBoj2St9vByGP")

        # self.all_mesh_meta = []
        # need_face_range = [50,32000]
        # self.max_triangles = need_face_range[1]
        # not_need_category = ["gear"]
        # self.deletenum = 0

        # with get_file(self.s3, self.s3link) as f:
        #     metadata = json.load(f)
        # print(f"dataset length: {len(metadata)}")

        # data from LP
        # self.connect_mesh_meta_path = "/workspace/objaverse"
        # # get all obj files in the connect_mesh_meta_path
        # obj_files = os.listdir(self.connect_mesh_meta_path)
        # # get all obj files in the connect_mesh_meta_path
        # obj_files = [f for f in obj_files if f.endswith('.obj')]
        # for obj_file in tqdm(obj_files, desc="Processing connect mesh"):
        #     model_id = obj_file.split('.')[0]
            
        #     mesh_meta = MeshMeta(
        #         model_id=model_id,
        #         model_type="LP",
        #         obj_path=obj_file,
        #     )
        #     self.all_mesh_meta.append(mesh_meta)
        # LP_dataset_length = len(self.all_mesh_meta)
        # print(f"LP dataset length: {LP_dataset_length}")


        # for model_info in tqdm(metadata, desc="Processing models"):
        #     model_id = f"{model_info['model_id']}"
        #     obj_files = model_info['obj_file']
        #     raw_obj = obj_files['obj_url']
        #     # print(f"keys: {model_info.keys()}, face_info: {model_info['face_info'].keys()}")
        #     mesh_meta = MeshMeta(
        #                 model_id=model_id,
        #                 model_type="TX",
        #                 raw_obj=raw_obj,
        #                 face_info=model_info['face_info'],
        #                 category=model_info['category'],
        #                 category_path_en=model_info['category_path_en'],
        #                 face_cnt = model_info['face_info']['face_cnt']
        #                 # point_cnt = model_info['point_info']['point_cnt']
        #             )
        #     triangle_face_cnt = mesh_meta.face_info['face_cnt']

            # if "objaverse" not in model_id:
            #     # 只处理objaverse数据集
            #     continue

        #     if triangle_face_cnt is None:
        #         logger.warning(f"{model_id} has no face_cnt atrribute, mesh_meta is {mesh_meta}")
        #         continue

        #     # 根据 need_face_range 来过滤数据
        #     if need_face_range is not None and (triangle_face_cnt < need_face_range[0] or triangle_face_cnt > need_face_range[1]):
        #         continue
            
        #     # 根据品类来过滤数据
        #     if not_need_category is not None and mesh_meta.category in not_need_category:
        #         continue

        #     self.all_mesh_meta.append(mesh_meta)
        # # print(f"dataset length: {len(self.all_mesh_meta)}, LP dataset length: {LP_dataset_length}, TX dataset length: {len(self.all_mesh_meta) - LP_dataset_length}")
        # print(f"dataset length: {len(self.all_mesh_meta)}")
        # 逆序
        # self.all_mesh_meta = self.all_mesh_meta[::-1]
        # self.all_mesh_meta = self.all_mesh_meta[:3000] # 只取400个

        # sort self.all_mesh_meta by face_cnt
        # self.all_mesh_meta = sorted(self.all_mesh_meta, key=lambda x: x.face_cnt, reverse=True)
        # self.s3 = None 
        name              = os.listdir(path)
        if uid_list == [] or uid_list == [""]:
            self.uid_list     = [i for i in name if len(i.split("."))>1]
        else:
            self.uid_list    = uid_list
        # print("dataset init, dataset length: ", len(self.uid_list))
    
    def __len__(self) -> int:
        return len(self.uid_list)

    def _preprocess_mesh(self, mesh):
        # mesh = self._mesh_filter(mesh)
        
        # 归一化处理
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center, scale = (bbmin + bbmax)*0.5, 2.0 * 0.9 / (bbmax - bbmin).max()
        mesh.vertices = (vertices - center) * scale
        
        # 确保三角形网格
        if mesh.faces.shape[1] == 4:
            mesh.faces = np.vstack([mesh.faces[:, :3], mesh.faces[:, [0,2,3]]])
        
        return mesh

    def _mesh_filter(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """使用PyMeshLab进行网格预处理"""
        ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(ml_mesh, "raw_mesh")
            
        ms.apply_filter('meshing_remove_duplicate_faces')
        ms.apply_filter('meshing_remove_duplicate_vertices')
        ms.apply_filter('meshing_merge_close_vertices', 
                    threshold=pymeshlab.PercentageValue(0.5))
        ms.apply_filter('meshing_remove_unreferenced_vertices')
            
        processed = ms.current_mesh()
        mesh.vertices = processed.vertex_matrix()
        mesh.faces = processed.face_matrix()
            
        return mesh

    def _pairwise_edges(self, face: np.ndarray):
        """
        给定一个面(任意顶点数>=2)，产生一组需要合并的边 (u,v)。
        - 三角面: (v0,v1), (v1,v2), (v2,v0)
        - 多边形：相邻成环合并
        """
        k = face.shape[0]
        if k < 2:
            return []
        return [(int(face[i]), int(face[(i + 1) % k])) for i in range(k)]

    def split_mesh_into_connected_components_union_set(self, mesh_vertices, mesh_faces):
        if mesh_vertices is None or mesh_faces is None:
            print("网格为空: faces or vertices is None")
            return None
        if mesh_vertices.size == 0 or mesh_faces.size == 0:
            print(f"网格为空: faces shape={mesh_faces.shape}, vertices shape={mesh_vertices.shape}")
            return None

        vertices = np.asarray(mesh_vertices)
        faces = np.asarray(mesh_faces)
        if faces.ndim != 2:
            raise ValueError(f"faces 应为二维整型数组，当前 ndim={faces.ndim}")
        if faces.dtype.kind not in ('i', 'u'):
            faces = faces.astype(np.int64)

        N = vertices.shape[0]
        uf = UnionSet(N)

        # 仅对出现在面的顶点进行并查集合并
        used_vertices = np.unique(faces)
        # 简单的安全检查：索引越界
        if used_vertices.size and (used_vertices.min() < 0 or used_vertices.max() >= N):
            raise IndexError("faces 中存在越界顶点索引")

        # 按“面内边相连”进行 union（对三角面即三条边；对多边形为环）
        # 若 faces 为三角形 (M,3)，循环也很快
        for f in faces:
            for u, v in self._pairwise_edges(f):
                uf.union(u, v)

        # 将每个面的“代表元(root)”作为该面所属的连通分支ID
        # 用面第一个顶点的根（理论上同一面内所有顶点根应一致；若数据异常，用第一个即可）
        roots_per_face = np.array([uf.find(int(f[0])) for f in faces], dtype=np.int64)

        # 分组：root -> face indices
        root_to_face_idx = defaultdict(list)
        for fi, r in enumerate(roots_per_face):
            root_to_face_idx[int(r)].append(int(fi))

        # 组装每个连通分支（提取子顶点并重映射面索引到局部）
        components = []
        for r, fidx_list in root_to_face_idx.items():
            fidx = np.array(fidx_list, dtype=np.int64)
            sub_faces_global = faces[fidx]  # (m_i, k)

            # 该分支涉及到的顶点集合（只取被这些面用到的顶点）
            unique_vids = np.unique(sub_faces_global)
            # 全局->局部映射
            g2l = {int(g): i for i, g in enumerate(unique_vids)}
            # 重映射 faces 到局部索引
            # 使用矢量化映射
            # 注意：np.vectorize 返回对象数组，故额外转换 dtype
            remapped = np.vectorize(g2l.__getitem__, otypes=[np.int64])(sub_faces_global)
            remapped = remapped.astype(np.int64)

            if remapped.shape[0] > 5:
                data = DynamicAttributes()
                data.vertices = vertices[unique_vids]            # (n_i, 3)
                data.faces = remapped                            # (m_i, k)
                components.append(data)

        print("debug---------------------------------------------")
        print(len(components))
        return components if len(components) > 0 else None

    def split_mesh_into_connected_components(self, mesh_vertices, mesh_faces):
        """
        将网格拆分为连通体。
        参数:
        mesh_path (str): 网格文件路径。
        返回:
        tuple: 连通体列表（vertices, faces）。
        """
        if (mesh_faces.shape[0] == 0 or mesh_vertices.shape[0] == 0):
            print(mesh_faces.shape)
            return None
        
        # if mesh_faces.shape[0] > self.max_triangles:
        #     return [(mesh_vertices, mesh_faces)], mesh_vertices, mesh_faces
        
        adjacency_matrix = igl.adjacency_matrix(mesh_faces)
        num_components, components, _ = igl.connected_components(adjacency_matrix)
        component_faces = [[] for _ in range(num_components)]

        if components.shape[0]==0:
            return None
            
        # 将每个面分配到对应的连通体
        for i in range(mesh_faces.shape[0]):
            face = mesh_faces[i]
            # 检查面上的所有顶点是否属于同一个连通体
            if components[face[0]] == components[face[1]] == components[face[2]]:
                component_faces[components[face[0]]].append(face)
                
        processed_components = []
        for faces in component_faces:
            if len(faces) > 0:
                faces = np.array(faces)
                unique_vertices = np.unique(faces)
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
                new_faces = np.vectorize(vertex_map.get)(faces)
                new_vertices = mesh_vertices[unique_vertices]

                data = DynamicAttributes()
                data.vertices = new_vertices
                data.faces = new_faces
                if len(new_faces) > 5:
                    processed_components.append(data)
                
        return processed_components
    
    def split_mesh_into_connected_components_trimesh(self, mesh_vertices, mesh_faces):
        """
        使用trimesh库将网格拆分为连通体。
        参数:
        mesh_vertices (np.ndarray): 网格顶点数组
        mesh_faces (np.ndarray): 网格面数组
        返回:
        list: 连通体列表，每个元素为DynamicAttributes对象，包含vertices和faces属性
        """
        if (mesh_faces.shape[0] == 0 or mesh_vertices.shape[0] == 0):
            print(f"网格为空: faces shape={mesh_faces.shape}, vertices shape={mesh_vertices.shape}")
            return None
        
        # 创建trimesh对象
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        
        # 使用trimesh的split方法获取连通分支
        connected_components = mesh.split()
        print("debug---------------------------------------------")
        print(len(connected_components))
        
        processed_components = []
        for component_mesh in connected_components:
            # 检查组件是否有效
            if component_mesh.faces.shape[0] > 5: 
                data = DynamicAttributes()
                data.vertices = component_mesh.vertices
                data.faces = component_mesh.faces
                processed_components.append(data)
        
        return processed_components
    
    def __getitem__(self, idx: int) -> Dict:
        # if self.s3 is None:
        #     self.s3 = GameaiCosSign("ruiiixu", "GXZRShV1gtpfTGFx4xnzBoj2St9vByGP")

        max_retries = 0
        retry_count = 0
        time_limit = 300  # 5分钟超时
        start_time = time.time()
        # Check if elapsed time exceeds limit and try next index if so
        sample_info = self.uid_list[idx]
        # print(f"sample_info: {sample_info.model_id}, {sample_info.model_type}")
    
        while retry_count <= max_retries:
            try:
                # 检查是否超时
                if time.time() - start_time > time_limit:
                    logger.warning(f"Timeout reached for idx {idx} after {time.time() - start_time:.2f} seconds")
                    return self.__getitem__((idx + 1) % len(self))
                    
                # sample_info = self.all_mesh_meta[idx]
                # filename = sample_info.model_id
                # load obj here
                # print(f"DEBUG:{self.path}/{sample_info}")
                raw_obj = trimesh.load(f"{self.path}/{sample_info}", force='mesh', file_type='obj', process=False)
                
                processed_mesh = self._preprocess_mesh(raw_obj)
                if processed_mesh.faces.shape[0] > 10:
                    break
            except Exception as e:
                retry_count += 1
                if retry_count > max_retries:
                    logger.warning(f"Failed to download mesh after {max_retries} retries for idx {idx}, trying next index")
                    return self.__getitem__((idx + 1) % len(self))
                logger.warning(f"Download failed for idx {idx}, attempt {retry_count}/{max_retries}: {str(e)}")
                continue
        
        # print(f"train data model id: {filename} ")
        data = {}
        if True:
            mesh = process_mesh(processed_mesh.vertices, processed_mesh.faces, quantization_bits=8, augment=False)
            # mesh = trimesh.load(file_path, force='mesh', process=False)
            verts, faces = mesh['vertices'], mesh['faces']
            

            # if float(len(verts)) / float(len(faces)) > 0.8:
                # print(f"verts / faces > 0.8, get item again")
                # self.deletenum += 1
                # meshtri = trimesh.Trimesh(vertices=verts, faces=faces)
                # meshtri.export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/mesh_{sample_info.model_id}.obj")
                # return self.__getitem__((idx + 1) % len(self))
            # verts = rescale_verts(verts)
            # selected_verts, selected_faces, boundry_faces, boundry_verts = random_box_selection_mesh(verts, faces)
            # datas = random_selection_mesh(verts, faces, fix=False)
            # if len(datas) == 0:
            #     return self.__getitem__((idx + 1) % len(self))
            # # data['pc_global'] = datas[0]['pc_global']
            # data['name'] = sample_info
            # skip = 0
            # for i in range(len(datas)):
            #     if datas[i]['skip']:    
            #         skip += 1
            #         continue
            #     data['pc_local_'+str(i - skip)] = datas[i]['pc_local']
            #     data['token_list_'+str(i - skip)] = datas[i]['token_list']
            #     data['bd_token_length_'+str(i - skip)] = datas[i]['bd_token_length']
            #     data['skip_'+str(i - skip)] = datas[i]['skip']
            #     data['pc_global'] = datas[i]['pc_global']
            #     data['center_'+str(i - skip)] = datas[i]['center']
            #     data['scale_'+str(i - skip)] = datas[i]['scale']
            #     data['cluster_'+str(i - skip)] = datas[i]['cluster']
            # lendatas = len(datas)
            lendatas = 0
            skip = 0
            # data['len'] = lendatas - skip
            processed_components = self.split_mesh_into_connected_components_union_set(verts, faces)
            if len(processed_components) >= 2:
                datas = selection_mesh_from_connected_components(verts, faces, processed_components, fix=False)
                if len(datas) > 0:
                    data['name'] = sample_info.split('.')[0]
                    for i in range(len(datas)):
                        if datas[i]['skip']:
                            skip += 1
                            continue
                        data['pc_local_'+str(i - skip + lendatas)] = datas[i]['pc_local']
                        data['token_list_'+str(i - skip + lendatas)] = datas[i]['token_list']
                        data['bd_token_length_'+str(i - skip + lendatas)] = datas[i]['bd_token_length']
                        data['skip_'+str(i - skip + lendatas)] = datas[i]['skip']
                        data['pc_global'] = datas[i]['pc_global']
                        data['center_'+str(i - skip + lendatas)] = datas[i]['center']
                        data['scale_'+str(i - skip + lendatas)] = datas[i]['scale']
                        data['cluster_'+str(i - skip + lendatas)] = datas[i]['cluster']
                data['len'] = len(datas) - skip + lendatas
            
            # if selected_verts is None:
            #     return self.__getitem__((idx + 1) % len(self))
            
                
            # bd = False
            # bdd = True
            # if boundary_faces.shape[0] <= 3:
            #     bd = False
            #     bdd = False

                
            # # trimesh.Trimesh(vertices=selected_verts, faces=selected_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/tmp0521/New_selected_mesh_time{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # # trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/MyBDTestResult/New_boundry_mesh_time{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # if bd:  
            #     all_verts = np.concatenate([selected_verts, boundary_verts], axis=0)
            # # 正确处理 boundary_faces 的索引偏移
            #     boundary_faces_offset = boundary_faces + selected_verts.shape[0]  # 对每个元素都加上偏移量
            #     all_faces = np.concatenate([selected_faces, boundary_faces_offset], axis=0)
            # else:
            #     all_verts = selected_verts
            #     all_faces = selected_faces
            # all_verts = rescale_verts(all_verts)
            # localmesh = trimesh.Trimesh(vertices=all_verts[:selected_verts.shape[0]], faces=selected_faces)
            # localmesh = process_mesh_xr(localmesh.vertices, localmesh.faces, augment=False)
            # selected_verts = localmesh['vertices']
            # selected_faces = localmesh['faces']
            # if bdd:
            #     localmesh = trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces)
            #     localmesh = process_mesh_xr(localmesh.vertices, localmesh.faces, augment=False)
            #     boundary_verts = localmesh['vertices']
            #     boundary_faces = localmesh['faces']
            # else:
            #     boundary_verts = []
            #     boundary_faces = []

            # # trimesh.Trimesh(vertices=selected_verts, faces=selected_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/tmp0523/New_selected_mesh_time{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # # trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/tmp0523/New_boundry_mesh_time{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # # save the selected mesh
            # # trimesh.Trimesh(vertices=selected_verts, faces=selected_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/examples/LT_selected_mesh_time{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # # loadmesh = trimesh.load(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/examples/selected_mesh_tmp.obj", force='mesh', process=False)
            # # selected_verts = loadmesh.vertices
            # # selected_faces = loadmesh.faces
            # # trimesh.Trimesh(vertices=boundry_verts, faces=boundry_faces).export(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/examples/boundry_mesh_{file_name}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # indices      = np.random.choice(50000, self.point_num, replace=False)
            # pc_local    = sample_pc(all_verts, all_faces, pc_num=50000, with_normal=True)[indices]
            # data['pc_local'] = torch.tensor(pc_local)
            # if torch.isnan(data['pc_local']).any() or torch.isinf(data['pc_local']).any():
            #     print(f"pc_local is nan or inf, get item again")
            #     return self.__getitem__((idx + 1) % len(self))
            # # save pc_normal as xyz file
            # # np.savetxt(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/tmp0523/pc_local_{file_name}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.xyz", pc_local)
            # # cat selected_verts and boundry_verts
            # # all_verts = np.concatenate([selected_verts, boundry_verts], axis=0)
            # # all_verts = rescale_verts(all_verts)
            # # selected_verts = all_verts[:selected_verts.shape[0]]
            # # boundry_verts = all_verts[selected_verts.shape[0]:]
            # # selected_verts = rescale_verts(selected_verts)
            
            # # trimesh.Trimesh(vertices=selected_verts, faces=selected_faces).export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/New_selected_mesh_{filename}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # # trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces).export(f"/vinowan-cfs/ruixu/code/DeepMesh0527/examples/New_boundry_mesh_{filename}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.obj")
            # token_list   = serialize(trimesh.Trimesh(vertices=selected_verts, faces=selected_faces))
            # if bdd:
            #     bd_token_list   = serialize(trimesh.Trimesh(vertices=boundary_verts, faces=boundary_faces))
            #     data['bd_token'] = torch.tensor(bd_token_list, dtype=torch.long)
            # else:
            #     bd_token_list = torch.ones(300, dtype=torch.long) * 4737
            #     data['bd_token'] = bd_token_list
            # block_size = 8
            # offset_size = 16
            # patch_size = 4
            # special_block_base = block_size**3 + offset_size**3 + patch_size**3
            # if bd:
            #     token_list[0] += special_block_base
            #     token_list = np.concatenate([[4736], bd_token_list, token_list, [4737]])
            # else:
            #     token_list[0] += special_block_base
            #     token_list = np.concatenate([[4736], token_list, [4737]])
            # # token_list = np.concatenate([[4736], token_list, [4737]])
            # # data['mesh_path'] = file_path
            # data['token_list'] = torch.tensor(token_list, dtype=torch.long)
            # if len(token_list) > 9000:
            #     print(f"token_list is longer than 9000, get item again")
            #     return self.__getitem__((idx + 1) % len(self))

            # if torch.isnan(data['token_list']).any() or torch.isinf(data['token_list']).any():
            #     print(f"data is nan or inf, get item again")
            #     return self.__getitem__((idx + 1) % len(self))
            # # save token_list as txt file in int64
            # # np.savetxt(f"/data6/ruixu/code/2025_MeshGen/MyDeepMesh/examples/token_list_{file_name}_date{datetime.now().strftime('%Y%m%d%H%M%S')}.txt", token_list, fmt='%d')
            # # print(f"verts: {selected_verts.shape[0]}, faces: {len(selected_faces)}, bd_verts: {len(boundary_verts)}, bd_faces: {len(boundary_faces)}, bdtoken: {len(bd_token_list)}, token: {len(token_list)}")

            
            # data['bd_token_length'] = len(bd_token_list)
            # indices_local  = np.random.choice(50000, self.point_num, replace=False)
            # pc_global       = sample_pc(verts, faces, pc_num=50000, with_normal=True)[indices_local]
            # data['pc_normal'] = torch.tensor(pc_global)

            # if torch.isnan(data['pc_normal']).any() or torch.isinf(data['pc_normal']).any():
            #     print(f"pc_normal is nan or inf, get item again")
            #     return self.__getitem__((idx + 1) % len(self))




        # elif file_name.split(".")[-1] == "ply":
        #     p             = o3d.io.read_point_cloud(file_path)
        #     pc_normal     = np.concatenate([np.asarray(p.points)[:,[2,0,1]],np.asarray(p.normals)[:,[2,0,1]]],axis=1)
        #     if len(pc_normal)>self.point_num:
        #         indices   = np.random.choice(len(pc_normal), self.point_num, replace=False)
        #         pc_normal = pc_normal[indices]
        #     # For point clouds, we use the same file as mesh path
        #     data['mesh_path'] = file_path
            
        
        return data
