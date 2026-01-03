import dataclasses
import torch
import trimesh
import numpy as np
from einops import rearrange
from typing import Dict, List

from src.custom_pipelines.beam_search import BeamHypotheses
from src.utils.ply_helper import write_ply


FACE_COND_FACTOR = 512


def normalize_mesh(vertices: np.ndarray, scale_range: tuple=(-1.0, 1.0)) -> np.ndarray:
    lower, upper = scale_range
    scale_per_axis = (vertices.max(0) - vertices.min(0)).max()
    center_xyz = 0.5 * (vertices.max(0) + vertices.min(0))
    normalized_xyz = (vertices - center_xyz) / scale_per_axis   # scaled into range (-0.5, 0.5)
    vertices = normalized_xyz * (upper - lower)
    return vertices


def quantize_vertices(vertices, cube_reso):
    """
    对顶点进行特定分辨率的量化。
    Args:
        vertices: shape 为 (n, 3), 数值范围为 [-0.5, 0.5]
    """
    vertices = vertices * 0.5
    # 由于浮点数的精度损失, 某些 vertices 值可能是 -0.5000000000000001，但不影响后面流程，这里使用宽松判定
    assert abs(abs(vertices).max() - 0.5) < 1e-3  # [-0.5, 0.5]
    vertices = (vertices + 0.5) * cube_reso  # [0, num_tokens]
    vertices -= 0.5  # for evenly distributed, [-0.5, num_tokens -0.5] will be round to 0 or num_tokens (-1)
    vertices_quantized_ = np.clip(vertices.round(), 0, cube_reso - 1).astype(int)  # [0, num_tokens -1]
    return vertices_quantized_


def remove_degeneration_elements(mesh: trimesh.Trimesh):
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def sort_mesh(mesh: trimesh.Trimesh):
    # 先将顶点按照zyx顺序排序
    sort_inds = np.lexsort(mesh.vertices.T)
    vertices = mesh.vertices[sort_inds]
    sort_res = np.argsort(sort_inds)
    faces = [sort_res[f] for f in mesh.faces]

    # 再面内顶点排序
    faces = [sorted(sub_arr) for sub_arr in faces]

    # 最后面排序
    faces = sorted(faces, key=lambda face: (face[0], face[1], face[2]))
    return vertices, faces

    
def quantize_mesh(vertices, faces, cube_reso=128):
    vertices_quantized_ = quantize_vertices(vertices, cube_reso)

    # 删除因为量化后产生的退化面点线
    cur_mesh = trimesh.Trimesh(vertices=vertices_quantized_, faces=faces)
    cur_mesh = remove_degeneration_elements(cur_mesh)
    
    # 对 mesh 进行排序
    vertices, faces = sort_mesh(cur_mesh)

    return vertices, faces, cur_mesh.copy()


def resort_input_ids(input_ids, bos_token_id):
    """
    Args:
        input_ids: torch.LongTensor, shape 为 (n,)。
        bos_token_id: tokenizer 中的 bos_token_id。
    """
    assert len(input_ids) % 9 == 0 and len(input_ids) // 9 > 1
    raw_device = input_ids.device
    # 去除 bos token
    bos_ids = None
    if input_ids[0] == bos_token_id:
        bos_ids, input_ids = input_ids[:9], input_ids[9:]
    input_ids = input_ids.cpu()
    
    # 去除尾部重复面
    faces = rearrange(input_ids, '(f c) -> f c', c=9)
    repeat_indices = torch.all(faces == faces[-1], dim=-1)
    if repeat_indices.sum() >= 10:
        faces =faces[repeat_indices.logical_not()]
    
    vertices = rearrange(faces, 'f (v c) -> (f v) c', c=3)
    vertices = torch.flip(vertices, dims=(-1,))  # 将 vertices 还原为 xyz 顺序，方便后面做排序
    faces = torch.arange(vertices.shape[0]).reshape(-1, 3)
    cur_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # NOTE: 这段代码与 quantize_mesh 中逻辑重复，后续验证正确性后进行整合。
    valid_faces = cur_mesh.area_faces > 1e-10
    cur_mesh.update_faces(valid_faces)
    cur_mesh = remove_degeneration_elements(cur_mesh)
    
    # 对 mesh 进行排序
    vertices, faces = sort_mesh(cur_mesh)
    
    # 将点内坐标顺序由 (x, y, z) 转换为 (z, y, x) 以符合排序优先级，方便后续的约束生成
    vertices = torch.from_numpy(vertices).flip(dims=(-1,))
    faces = torch.tensor(faces).long().clip(0)
    final_vertices = vertices[faces]
    sorted_ids = rearrange(final_vertices, 'f v c -> (f v c)', c=3).to(device=raw_device, dtype=torch.long)
    
    if bos_ids is not None:
        sorted_ids = torch.concat([bos_ids, sorted_ids])
    
    return sorted_ids


def y_up_coord_to_z_up(points):
    """
    obj 文件中的坐标系 y 轴正方向为竖直向上方向，要想将物体转换为 z 轴为竖直向上方向，需要将物体绕 x 轴逆时针旋转 theta = 90 度
                Y           
                ^            
                |           
    Z <---------|  
               /     
              X   
    相应旋转矩阵 R 如下，进行 new_p = Rp 矩阵运算即可
    [[1       0          0    ],
     [0 cos(theta) -sin(theta)],
     [0 sin(theta)  cos(theta)]]
    
    Args:
        points: torch.Tensor, shape 为 (..., N, 3)
    """
    rotate_m = torch.tensor([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ], dtype=points.dtype)
    transfer_p = torch.matmul(rotate_m, rearrange(points, '... n c -> ... c n'))
    return rearrange(transfer_p, '... c n -> ... n c')


def z_up_coord_to_y_up(points):
    """
    正常的世界坐标系 z 轴正方向为竖直向上方向，要想将物体转换为 y 轴为竖直向上方向，需要将物体绕 x 轴逆时针旋转 theta = -90 度
        Z           
        ^            
        |           
        |---------> Y
       /     
      X   
    相应旋转矩阵 R 如下，进行 new_p = Rp 矩阵运算即可
    [[1       0          0    ],
     [0 cos(theta) -sin(theta)],
     [0 sin(theta)  cos(theta)]]
    
    Args:
        points: torch.Tensor, shape 为 (..., N, 3)
    """
    rotate_m = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=points.dtype)
    transfer_p = torch.matmul(rotate_m, rearrange(points, '... n c -> ... c n'))
    return rearrange(transfer_p, '... c n -> ... n c')


def post_process_mesh(mesh_coords, filename: str):
    """
    Args:
        mesh_coords: torch.Tensor，shape 为 (f, 3, 3)
    """
    mesh_coords = mesh_coords[~torch.isnan(mesh_coords[:, 0, 0])]   # nvalid_face x 3 x 3
    vertices = mesh_coords.reshape(-1, 3)
    vertices = z_up_coord_to_y_up(vertices)
    vertices_index = np.arange(len(vertices))   # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)
    write_ply(
        np.asarray(vertices.float().cpu()),
        None,
        np.asarray(triangles),
        filename
    )
    return vertices 


@dataclasses.dataclass
class GenerationCtx:
    out: torch.Tensor = None
    # 生成状态部分的记录
    all_current_face: List[tuple] = None
    all_points: List[torch.Tensor] = None
    all_points_count: List[torch.Tensor] = None
    all_faces: List[torch.Tensor] = None
    # beam search 部分的记录
    current_scores: torch.Tensor = None
    end_beam_indices: List = None
    beam_hyps: BeamHypotheses = None
    # burn-in 部分的记录
    last_burn_in_pos: int = None


def penalize_degeneration(scores, gen_ctx: GenerationCtx, repeat_vertex_threshold=None):
    all_current_face, all_points, all_points_count, all_faces = gen_ctx.all_current_face, gen_ctx.all_points, \
        gen_ctx.all_points_count, gen_ctx.all_faces
    
    def penalize_repeat_vertex(half_point, points, points_count, repeat_vertex_threshold, bsz_idx, scores):
        if repeat_vertex_threshold is None:
            return scores
        
        half_point_tensor = torch.tensor(half_point).unsqueeze(0)  # shape: (1, 2)
        half_point_diff = torch.norm((points[:, :2] - half_point_tensor).float(), dim=-1)  # shape: (n,)
        same_half_point_indices = torch.where(half_point_diff == 0)[0]  # shape: (l,)
        for point_idx in same_half_point_indices:
            if points_count[point_idx] > repeat_vertex_threshold:
                dangerous_coord = points[point_idx, 2]
                scores[bsz_idx, dangerous_coord] += float('-inf')
        return scores
    
    bsz, _ = scores.shape
    for idx in range(bsz):
        previous_faces = all_faces[idx]  # shape: (n, 9)
        current_face: tuple = all_current_face[idx]
        previous_points = all_points[idx]  # shape: (n, 3)
        previous_points_count = all_points_count[idx]  # shape: (n,)
        
        if len(current_face) == 2:
            # 面内的每个点不允许与之前点有过多重复
            scores = penalize_repeat_vertex(current_face[0:2], previous_points, previous_points_count, repeat_vertex_threshold,
                                            bsz_idx=idx, scores=scores)
        elif len(current_face) == 5:
            # 面内的第二个点不允许与第一个点重复
            if current_face[3:5] == current_face[0:2]:
                scores[idx, current_face[2]] += float('-inf')
            
            scores = penalize_repeat_vertex(current_face[3:5], previous_points, previous_points_count, repeat_vertex_threshold,
                                            bsz_idx=idx, scores=scores)
        elif len(current_face) == 8:
            # 施加面重复惩罚
            if len(previous_faces) > 0:
                p1, p2 = sorted((current_face[0:3], current_face[3:6]))
                half_face_hash = torch.tensor([p1 + p2 + current_face[6:8]])
                
                face_diff = torch.norm((previous_faces[:, :8] - half_face_hash).float(), dim=-1)  # shape: (n,)
                similar_face_indices = torch.where(face_diff == 0)[0]
                similar_face = previous_faces[similar_face_indices]
                dangerous_coord = similar_face[:, 8]
                for coord in dangerous_coord:
                    scores[idx, coord] += float('-inf')
            
            # 施加面内点重复惩罚
            face_points = [current_face[0:3], current_face[3:6]]
            for coord in [current_face[2], current_face[5]]:
                hypoth_point = tuple(list(current_face[6:8]) + [coord])
                if hypoth_point in face_points:
                    scores[idx, coord] += float('-inf')
            
            scores = penalize_repeat_vertex(current_face[6:8], previous_points, previous_points_count, repeat_vertex_threshold,
                                            bsz_idx=idx, scores=scores)
    return scores
    