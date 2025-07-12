import json
import numpy as np
from scipy.optimize import least_squares
from glob import glob
from collections import defaultdict
from freecad_visualize_json_pythonocc import create_bspline_from_params, sample_points_on_curve, construct_edge
import os
import argparse

def load_json(file_path):
    '''
    load the json and construct the mappings
    wires_to_face: wire_name -> face_index
    wires_mapping: wire_name -> wire_index
    wire_reversed_mapping: edge_index -> [wire_index1, wire_index2, ...] containing the wire that contains the edge
    '''
    with open(file_path, 'r') as f:
        data = json.load(f)
    vertices = np.array(data['vertices'])
    edges = data['edges']
    faces = data['faces']
    wires = [w for face in faces for w in face['wires'] ]
    wires_mapping = {wire2str(wire) :index  for index, wire in enumerate(wires)}
    wires_to_face = {wire2str(w): face_index for face_index, face in enumerate(faces) for w in face['wires']}

    wire_reversed_mapping = defaultdict(list)
    for wire_str, index in wires_mapping.items():
        edge_indexes = wirestr2edge_indexes(wire_str)
        for edge_index in edge_indexes:
            wire_reversed_mapping[edge_index].append(index)


    assert len(wires_mapping) == len(wires)

    return vertices, edges, faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping


def save_json(file_path, vertices, edges, faces):
    data = {
        "vertices": vertices.tolist(),
        "edges": edges,
        "faces": faces
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def wire2str(wire):
    return '_'.join([f"{e['edge_index']:05d}-{e['orientation'][:1]}" for e in wire["ordered_edges"]])

def wirestr2edge_indexes(wire_str):
    return [int(edge_str.split('-')[0]) for edge_str in wire_str.split('_')]


def split_circle_bspline(edge, vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping):

    # Split closed circle into two half edges

    # center = np.array(edge.get('curve_definition').get('center'))
    # normal = np.array(edge.get('curve_definition').get('normal'))
    # radius = edge.get('curve_definition').get('radius')
    edge_index = edge.get('edge_index')
    first_parameter = edge.get('first_parameter')
    last_parameter = edge.get('last_parameter')

    length = edge.get('length')
    if edge.get('curve_type') == 'BSplineCurve':
        curve_type = 'BSplineCurve'
    elif edge.get('curve_type') == 'Circle':
        curve_type = 'Circle'
    else:
        return vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping

    vertex_indices = edge.get('vertices')


    # First, check if the circle is closed
    if len(vertex_indices) != 1:
        return vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping

    # Second, find the other vertex
    vertex_index = vertex_indices[0]
    vertex = vertices[vertex_index]
    new_vertex_index = len(vertices)

    if curve_type == 'Circle':
        center = np.array(edge.get('curve_definition').get('center'))
        vec_to_center = center - vertex
        new_vertex = center + vec_to_center

    elif curve_type == 'BSplineCurve':
        
        assert first_parameter == 0, f"first_parameter != 0, {first_parameter}"
        assert last_parameter == 1, f"last_parameter != 1, {last_parameter}"
        degree = edge.get('curve_definition').get('degree')
        is_periodic = edge.get('curve_definition').get('is_periodic')
        is_closed = len(vertex_indices) == 1
        control_points = edge.get('curve_definition').get('control_points')
        knots = edge.get('curve_definition').get('knots')
        multiplicities = edge.get('curve_definition').get('multiplicities')
        start_vertex = vertices[vertex_index]
        end_vertex = start_vertex
        bspline_arc, sampled_points, is_closed, bspline_curve = create_bspline_from_params(
            degree, is_periodic, is_closed, control_points, knots, multiplicities, 
            start_vertex, end_vertex, return_curve=True
        )
        new_vertex = bspline_curve.Value(0.5)
        new_vertex = np.array([new_vertex.X(), new_vertex.Y(), new_vertex.Z()])

    

    # append the new vertex to the vertices
    vertices = np.concatenate([vertices, new_vertex[np.newaxis, :]], axis=0)
   
    # Third, update current edge
    edge['vertices'] = [vertex_index, new_vertex_index]
    edge['length'] = length / 2
    # edge['first_parameter'] = 0
    edge['last_parameter'] = (edge['last_parameter']  - edge['first_parameter']) / 2 + edge['first_parameter']


    # Third, build new edge
    new_edge_index = len(edges)
    new_edge = edge.copy()
    new_edge['vertices'] = [new_vertex_index, vertex_index]
    new_edge['length'] = length / 2
    new_edge['first_parameter'] = edge['last_parameter']
    new_edge['last_parameter'] = last_parameter
    new_edge['edge_index'] = new_edge_index


    # new_edge = {
    #     "edge_index": new_edge_index,
    #     "curve_type": curve_type,
    #     "length": length / 2,
    #     "vertices": [new_vertex_index, vertex_index],
    #     "first_parameter": edge['last_parameter'],
    #     "last_parameter": edge['last_parameter'] + np.pi,
    #     "curve_definition": edge['curve_definition']
    # }

    edges.append(new_edge)

    # Forth, update wires and faces
    # Loop through all wires that contains the current edge
    for wire_index in wire_reversed_mapping[edge['edge_index']]:
        wire = wires[wire_index]
        wire_old_name = wire2str(wire)
        # A wire is only contained on one face
        face_index = wires_to_face[wire_old_name]
        face = faces[face_index]

        # Insert new edge into the wire
        for e_idx, e in enumerate(wire['ordered_edges']):
            # The closed circle must appear only once in a wire
            if e['edge_index'] == edge['edge_index']:
                new_e = e.copy()
                new_e['edge_index'] = new_edge_index
                # Insert the new edge after the current edge if it is forward, otherwise insert before the current edge
                if e['orientation'] == 'Forward':
                    insert_e_idx = e_idx + 1
                else:
                    insert_e_idx = e_idx

                break
        wire['ordered_edges'].insert(insert_e_idx, new_e)
        wire_new_name = wire2str(wire)
        # Update wire to the face
        for w_idx, w in enumerate(face['wires']):
            if wire2str(w) == wire_old_name:
                face['wires'][w_idx] = wire
                break
        
        # Update mappings
        wires_mapping[wire_new_name] = wires_mapping[wire_old_name]
        wires_mapping.pop(wire_old_name)
        wires_to_face[wire_new_name] = face_index
        wires_to_face.pop(wire_old_name)

    wire_reversed_mapping[new_edge_index] = wire_reversed_mapping[edge['edge_index']]

    print(f"Successfully split {curve_type} on edge {edge['edge_index']}")
    return vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping




def fit_circle_3d(points: np.ndarray):
    """
    从三维空间中的一组点拟合一个最佳拟合圆，并返回详细信息。

    参数:
        points (np.ndarray): 一个 N x 3 的 NumPy 数组，每行代表一个三维点 (x, y, z)。
                             点的数量必须大于等于3。

    返回:
        tuple: 一个包含五个元素的元组 (center, radius, normal, rmse, angles_rad)
            - center (np.ndarray): 拟合圆的三维圆心坐标 (3,)。
            - radius (float): 拟合圆的半径。
            - normal (np.ndarray): 拟合圆所在平面的单位法向量 (3,)。
            - rmse (float): 拟合的均方根误差。
            - angles_rad (np.ndarray): 每个输入点在拟合圆上的弧度值 (N,)，范围 [0, 2π]。
        或者在失败时返回 (None, None, None, None, None)。
    """
    if points.shape[0] < 3:
        print("错误：至少需要3个点来拟合一个圆。")
        return None, None, None, None, None

    # --- 步骤 1 & 2: 拟合平面并投影 ---
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered, full_matrices=False)
    normal = vh[2, :]
    u_axis = vh[0, :]
    v_axis = vh[1, :]
    points_2d = np.column_stack([np.dot(points_centered, u_axis), np.dot(points_centered, v_axis)])

    # --- 步骤 3: 2D 圆拟合 ---
    def circle_residuals(params, points_2d_arr):
        xc, yc, R = params
        return np.sqrt((points_2d_arr[:, 0] - xc)**2 + (points_2d_arr[:, 1] - yc)**2) - R
    
    initial_guess = (0, 0, np.std(points_2d))
    result = least_squares(circle_residuals, initial_guess, args=(points_2d,))
    if not result.success:
        return None, None, None, None, None
    xc_2d, yc_2d, radius = result.x

    # --- 步骤 4: 结果转回3D ---
    center = centroid + xc_2d * u_axis + yc_2d * v_axis
    
    # --- 步骤 5: 计算拟合误差 (RMSE) ---
    squared_errors = []
    for point in points:
        vec_to_center = point - center
        dist_perp = np.abs(np.dot(vec_to_center, normal))
        dist_in_plane = np.linalg.norm(vec_to_center - dist_perp * normal)
        error_sq = (dist_in_plane - radius)**2 + dist_perp**2
        squared_errors.append(error_sq)
    rmse = np.sqrt(np.mean(squared_errors))
    
    # --- 新增步骤 6: 计算每个点的弧度值 ---
    # 计算每个2D投影点相对于2D圆心的向量
    vecs_2d_from_center = points_2d - np.array([xc_2d, yc_2d])
    
    # 使用 arctan2 计算角度，u_axis方向为0弧度
    angles_rad = np.arctan2(vecs_2d_from_center[:, 1], vecs_2d_from_center[:, 0])
    
    # 将角度范围从 [-pi, pi] 调整到 [0, 2*pi]
    angles_rad[angles_rad < 0] += 2 * np.pi
    
    return center, radius, normal, rmse, angles_rad
    

def fit_arc_3d(points: np.ndarray):
    """
    从一个有序的3D点序列拟合一个圆弧，并确保法线和参数化方向
    与点的顺序(points[0] -> points[-1])一致。

    参数:
        points (np.ndarray): 一个 N x 3 的 NumPy 数组，代表一个有序的点序列。

    返回:
        tuple: (center, radius, normal, rmse, angles_rad)
            - center (np.ndarray): 圆心。
            - radius (float): 半径。
            - normal (np.ndarray): **与点序方向通过右手定则确定的法向量**。
            - rmse (float): 拟合的均方根误差。
            - angles_rad (np.ndarray): **从第一个点开始单调递增的弧度值** (第一个点为0)。
    """
    # 1. 首先进行基础的几何拟合
    #    注意：这里得到的 normal 和 angles 只是临时的，方向可能不正确
    temp_center, temp_radius, temp_normal, rmse, _ = fit_circle_3d(points)

    if temp_center is None:
        return None, None, None, None, None

    # 2. 确定正确的基向量和参数化方向
    
    # 2.1 重新计算SVD以获取基向量
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    _, _, vh = np.linalg.svd(points_centered, full_matrices=False)
    
    # 2.2 得到初始的基向量和角度
    u_axis = vh[0, :]
    v_axis = vh[1, :]
    initial_normal = vh[2, :]
    
    points_2d = np.column_stack([np.dot(points_centered, u_axis), np.dot(points_centered, v_axis)])
    center_2d = np.array([np.dot(temp_center - centroid, u_axis), np.dot(temp_center - centroid, v_axis)])
    
    vecs_from_center_2d = points_2d - center_2d
    initial_angles = np.arctan2(vecs_from_center_2d[:, 1], vecs_from_center_2d[:, 0])
    initial_angles = (initial_angles + 2 * np.pi) % (2 * np.pi) # 规范到 [0, 2pi]
    
    # 2.3 检查方向是否与点序一致
    start_angle = initial_angles[0]
    mid_angle = initial_angles[len(points) // 2]
    end_angle = initial_angles[-1]
    
    arc_span = (end_angle - start_angle + 2 * np.pi) % (2 * np.pi)
    mid_span = (mid_angle - start_angle + 2 * np.pi) % (2 * np.pi)
    
    # 如果中点走的是长弧，说明当前参数化方向是反的
    if mid_span > arc_span:
        # 方向错误，需要反转法向量和参数化方向
        final_normal = -initial_normal
        # 反转参数化可以通过交换/反转一个基向量实现，这里我们反转v_axis
        final_v_axis = -v_axis
    else:
        # 方向正确
        final_normal = initial_normal
        final_v_axis = v_axis
        
    # 3. 使用最终确定的方向重新计算角度
    
    # 使用最终的v轴重新计算2D Y坐标
    y_coords_final = np.dot(points_centered, final_v_axis)
    # 2D圆心也需要重新计算Y坐标
    center_y_final = np.dot(temp_center - centroid, final_v_axis)
    
    vecs_y_from_center_final = y_coords_final - center_y_final
    # X坐标不变
    vecs_x_from_center_final = vecs_from_center_2d[:, 0]
    
    final_angles = np.arctan2(vecs_y_from_center_final, vecs_x_from_center_final)
    
    # 4. 将最终角度规范化，使起始点为0，并单调递增
    final_angles = (final_angles + 2 * np.pi) % (2 * np.pi)
    start_angle_offset = final_angles[0]
    final_angles = (final_angles - start_angle_offset + 2 * np.pi) % (2 * np.pi)

    return temp_center, temp_radius, final_normal, rmse, final_angles

def simplify_bspline_curve(curve_def, vertices):
    v_indices = curve_def.get('vertices', [])
    
    curve_def = edge.get('curve_definition', {})
    degree = curve_def.get('degree')
    is_periodic = curve_def.get('is_periodic', False)
    is_closed = len(v_indices) == 1
    control_points = curve_def.get('control_points', [])
    knots = curve_def.get('knots', [])
    multiplicities = curve_def.get('multiplicities', [])
    edge_index = edge.get('edge_index')
    # 验证B样条曲线参数
    if degree is not None and control_points and knots and multiplicities:
        if len(v_indices) >= 2:
            start_vertex = vertices[v_indices[0]]
            end_vertex = vertices[v_indices[-1]]  # 使用最后一个顶点作为终点
        elif len(v_indices) == 1:
            start_vertex = vertices[v_indices[0]]
            end_vertex = vertices[v_indices[0]]
        else:
            # 如果没有足够的顶点索引，使用控制点的首尾作为起终点
            start_vertex = control_points[0]
            end_vertex = control_points[-1]
            
        arc, sampled_points, is_closed = create_bspline_from_params(
            degree, is_periodic, is_closed, control_points, knots, multiplicities, 
            start_vertex, end_vertex
        )
    if degree == 1:
        # This is a line
        new_type = 'Line'
        # print(end_vertex, start_vertex)
        new_length = np.linalg.norm(end_vertex - start_vertex)
        new_curve_def = {
            "start": [start_vertex[0], start_vertex[1], start_vertex[2]],
            "end": [end_vertex[0], end_vertex[1], end_vertex[2]]
        }
        new_fp = 0
        new_lp = 0
        simplify_success = True
        print(f"Successfully fit line on edge {edge_index} with degree={degree}")

    elif degree == 2:
        # Try fit with circle
        center, radius, normal, rmse, angles_rad = fit_arc_3d(sampled_points)
        relative_error = rmse / radius
        if relative_error < 1e-6:
            print(f"Successfully fit circle on edge {edge_index} with degree={degree}")
        
            new_type = 'Circle'
            if np.allclose(start_vertex, end_vertex):
                # Closed
                new_length = np.pi * 2 * radius
                new_fp = 0
                new_lp = 2 * np.pi
            else:
                rad_begin = np.min(angles_rad)
                rad_end = np.max(angles_rad)
                new_fp = rad_begin
                new_lp = rad_end
                new_length = (rad_end - rad_begin) * radius

            new_curve_def = {
                "center": [center[0], center[1], center[2]],
                "normal": [normal[0], normal[1], normal[2]],
                "radius": radius
            }
            simplify_success = True
        else:
            simplify_success = False
    else:
        simplify_success = False

    if simplify_success:
        new_edge = {
                "edge_index": edge_index,
                "curve_type": new_type,
                "length": new_length,
                "vertices": v_indices,
                "first_parameter": new_fp,
                "last_parameter":  new_lp,
                "curve_definition": new_curve_def
            }
    else:
        new_edge = None
    return new_edge

def save_json(vertices, edges, faces, save_path):
    data = {
        "vertices": vertices.tolist(),
        "edges": edges,
        "faces": faces
    }
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

# def update_periodic_face_outer_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping):
#     for wire_idx, wire in enumerate(face['wires']):
#         wire_global_idx = wires_mapping[wire2str(wire)]
#         wire_old_name = wire2str(wire)
#         if not wire['is_outer']:
#             continue
#         edges = wire['ordered_edges']
#         edge_idxes = defaultdict(int)

#         for e_idx, e in enumerate(edges):
#             edge_idxes[e['edge_index']] += 1

#         connected_edge_idx = -1
#         for e_idx, num_occur in edge_idxes.items():
#             if num_occur == 2:
#                 if connected_edge_idx == -1:
#                     connected_edge_idx = e_idx
#                 else:
#                     raise ValueError(f"Found two connected edges for wire {wire_idx} of face {face_idx}")

#         # We only reverse the orientation of the not connected edge.
#         for edge_idx, edge in enumerate(wire['ordered_edges']):
#             if edge['edge_index'] == connected_edge_idx:
#                 continue
#             # Reverse the orientation of the edge
#             if edge['orientation'] == 'Reversed':
#                 edge['orientation'] = 'Forward'
#             else:
#                 edge['orientation'] = 'Reversed'
#             wire['ordered_edges'][edge_idx] = edge
#         # Update the wire to the face
#         face['wires'][wire_idx] = wire
#         # Update mappings
#         wires[wire_global_idx] = wire
#         wires_mapping[wire2str(wire)] = wire_global_idx
#         wires_mapping.pop(wire_old_name)
#         wires_to_face[wire2str(wire)] = face_idx
#         # for e_idx, e in enumerate(wire['ordered_edges']):
#         #     wire_reversed_mapping[e['edge_index']] = wire_idx
#         # print(f"Successfully updated wire {wire_idx} of face {face_idx}")
#     face['orientation'] = 'Forward'
#     faces[face_idx] = face
#     print(f"Successfully updated face {face_idx}")
    
#     return faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping


# def update_non_periodic_face_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping, update_outer=False):
#     # Loop for each wire in the face
#     for wire_idx, wire in enumerate(face['wires']):
#         if wire['is_outer']:
#             if not update_outer:
#                 continue

#         wire_global_idx = wires_mapping[wire2str(wire)]
#         wire_old_name = wire2str(wire)
#         # Loop for each edge in the wire
#         for edge_idx, edge in enumerate(wire['ordered_edges']):
#             # Reverse the orientation of the edge
#             if edge['orientation'] == 'Reversed':
#                 edge['orientation'] = 'Forward'
#             else:
#                 edge['orientation'] = 'Reversed'
#             wire['ordered_edges'][edge_idx] = edge
#         # Update the wire to the face
#         face['wires'][wire_idx] = wire
#         # Update mappings
#         wires[wire_global_idx] = wire # Replace the old wire with the new one
#         wires_mapping[wire2str(wire)] = wire_global_idx # Update the mapping
#         wires_mapping.pop(wire_old_name) # Remove the old wire from the mapping
#         wires_to_face[wire2str(wire)] = face_idx # Update the mapping

#         # No need to update wire_reversed_mapping as the edges are the same and the wire_global_idx is the same
#         # for e_idx, e in enumerate(wire['ordered_edges']):
#         #     wire_reversed_mapping[e['edge_index']] = wire_idx
#         # print(f"Successfully updated wire {wire_idx} of face {face_idx}")
#     face['orientation'] = 'Forward'
#     faces[face_idx] = face
#     print(f"Successfully updated face {face_idx}")

#     return faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping



class Node():
    def __init__(self, vertex_index):
        self.vertex_index = vertex_index
        self.edges_out = []
        self.edges_in = []
        self.edges_in_out = []

    def degree(self):
        return len(self.edges_in_out)
    
    def add_edge(self, edge):
        if edge['vertices'][0] == self.vertex_index:
            self.edges_out.append(edge)
        if edge['vertices'][-1] == self.vertex_index:
            self.edges_in.append(edge)
        self.edges_in_out.append(edge)

class WireGraph:
    """
    一个增强版的WireGraph类，用于表示和查询wire的图结构。
    它假设所有边的方向都是其“自然”方向（即vertices列表中的顺序）。
    """
    def __init__(self, wire, edges):
        all_vertices = []
        for edge in wire['ordered_edges']:
            edge_data = edges[edge['edge_index']]
            all_vertices.extend(edge_data['vertices'])
    
        all_vertices = list(set(all_vertices))
        self.nodes = {}
        self.edge_dict = {}
        self.vertices2edge = {}
        for vertex in all_vertices:
            self.nodes[vertex] = Node(vertex)
        traversed_edges = []
        for edge in wire['ordered_edges']:
            if edge['edge_index'] in traversed_edges:
                continue
            edge_data = edges[edge['edge_index']]
            edge_index = edge['edge_index']
            v1 = edge_data['vertices'][0]
            v2 = v1 if len(edge_data['vertices']) == 1 else edge_data['vertices'][-1]
            self.nodes[v1].add_edge(edge_data)
            self.nodes[v2].add_edge(edge_data)
            self.edge_dict[edge_index] = [v1, v2]
            self.vertices2edge[f"{v1:04d}-{v2:04d}"] = edge_index
            traversed_edges.append(edge_index)

    def get_degree(self, vertex):
        """安全地获取一个顶点的出度。"""
        return self.nodes[vertex].degree()
    
    def get_edge_by_vertices(self, v1, v2):
        """通过有序的顶点对获取边索引。"""
        return self.vertices2edge[f"{v1:04d}-{v2:04d}"]
    
    def get_edge_by_index(self, edge_index):
        """通过边索引获取其有序的顶点对。"""
        return self.edge_dict[edge_index]


def get_polygon_winding_normal_from_vertices(ordered_vertices):
    """
    使用Newell方法计算3D多边形的法线。
    该法线的方向由顶点的环绕顺序（顺时针/逆时针）决定。
    """
    normal = np.array([0.0, 0.0, 0.0])
    num_vertices = len(ordered_vertices)

    for i in range(num_vertices):
        p1 = ordered_vertices[i]
        p2 = ordered_vertices[(i + 1) % num_vertices]
        normal[0] += (p1[1] - p2[1]) * (p1[2] + p2[2])
        normal[1] += (p1[2] - p2[2]) * (p1[0] + p2[0])
        normal[2] += (p1[0] - p2[0]) * (p1[1] + p2[1])

    norm_magnitude = np.linalg.norm(normal)
    if norm_magnitude < 1e-9:
        return np.array([0.0, 0.0, 0.0])
    return normal / norm_magnitude



def fix_wire(face, edges, all_vertices):
    """
    根据面的法线校正其外边界wire中每条边的方向，以满足右手定则。
    """
    for wire_idx, wire in enumerate(face['wires']):
        if wire['is_outer']:
            outer_wire_data = wire
            break

    if not outer_wire_data:
        raise ValueError(f"Face {face.get('face_index')} has no outer wire.")

    wire_graph = WireGraph(outer_wire_data, edges)
    
    degrees = [wire_graph.get_degree(v) for v in wire_graph.nodes.keys()]
    if not degrees:
        raise ValueError(f"Wire for face {face.get('face_index')} is empty or invalid.")
    max_degree = max(degrees)

    # if max_degree > 2:
    #     raise NotImplementedError(f"Wire for face {face.get('face_index')} has branching (max degree > 2), not implemented.")
    if max_degree < 2:
        raise ValueError(f"Wire for face {face.get('face_index')} is disconnected (max degree < 2).")

    # 根据图的复杂度，调用不同的处理函数
    if max_degree == 2:
        # 这是一个简单的闭环
        # face_normal_def = face['surface_definition']
        # face_normal = np.array(face_normal_def.get('normal') or face_normal_def.get('axis'))
        # return _fix_simple_loop(outer_wire_data, wire_graph, face_normal, all_vertices)
        return face
    elif max_degree == 3:
        # 1. Using the axis to determine the top and bottom
        assert face['surface_type'] == 'Cylinder' or face['surface_type'] == 'Cone' or face['surface_type'] == 'Toroid' or face['surface_type'] == 'BSplineSurface'
        if face['surface_type'] == 'Toroid':
            return face
        if face['surface_type'] == 'BSplineSurface':
            return face


        axis = np.array(face['surface_definition']['axis'])
        center = np.array(face['surface_definition']['position'])
        orientation = face['orientation']

        vertex_degree_3 = []
    
        for v in wire_graph.nodes.keys():
            if wire_graph.get_degree(v) == 3:
                vertex_degree_3.append(v)
        vector_center_to_v  = [np.array(all_vertices[vertex_degree_3[i]]) - center for i in range(len(vertex_degree_3))]
        normal_axis = axis / (np.linalg.norm(axis) + 1e-8)
        dot_products = [np.dot(vector_center_to_v[i], normal_axis) for i in range(len(vertex_degree_3))]
        top_vertex_index = vertex_degree_3[np.argmax(dot_products)]
        bottom_vertex_index = vertex_degree_3[np.argmin(dot_products)]

        # 需要通过DFS来确定top edges, bottom edges, vertical edges
        edges_traversed = []
        find_top_edges = False
        while not find_top_edges:
            top_edge_candidates = []
            nodes_traversed = [top_vertex_index]
            curr_v = top_vertex_index
            edge_candidates = wire_graph.nodes[top_vertex_index].edges_in_out
            edge_candidate = [e for e in edge_candidates if e['edge_index'] not in edges_traversed][0]
            if edge_candidate['vertices'][0] == curr_v:
                next_v = edge_candidate['vertices'][-1]
            else:
                next_v = edge_candidate['vertices'][0]
            edges_traversed.append(edge_candidate['edge_index'])
            top_edge_candidates.append(edge_candidate['edge_index'])
            nodes_traversed.append(next_v)
            curr_v = next_v
            while curr_v != bottom_vertex_index and curr_v != top_vertex_index:
                edge_candidates = wire_graph.nodes[curr_v].edges_in_out
                edge_candidates = [e for e in edge_candidates if e['edge_index'] not in edges_traversed]
                if len(edge_candidates) > 1:
                    edge_candidates = [e for e in edge_candidates if e['vertices'][0] not in nodes_traversed[:-1] and e['vertices'][1] not in nodes_traversed[:-1]]
                if len(edge_candidates) > 1:
                    edge_candidates = [e for e in edge_candidates if e['edge_index'] not in edges_traversed]
                edge_candidate = edge_candidates[0]
                if edge_candidate['vertices'][0] == curr_v:
                    next_v = edge_candidate['vertices'][1]
                else:
                    next_v = edge_candidate['vertices'][0]
                edges_traversed.append(edge_candidate['edge_index'])
                nodes_traversed.append(next_v)
                curr_v = next_v
                top_edge_candidates.append(edge_candidate['edge_index'])
            if curr_v == top_vertex_index: # Top loop
                top_edges = top_edge_candidates.copy()
                find_top_edges = True
            else:
                # 部分vertical edges，重新遍历
                continue
        
        # 然后找到bottom edges
        edges_traversed = []
        find_bottom_edges = False
        while not find_bottom_edges:
            bottom_edge_candidates = []
            nodes_traversed = [bottom_vertex_index]
            curr_v = bottom_vertex_index
            edge_candidates = wire_graph.nodes[bottom_vertex_index].edges_in_out
            edge_candidate = [e for e in edge_candidates if e['edge_index'] not in edges_traversed][0]
            if edge_candidate['vertices'][0] == curr_v:
                next_v = edge_candidate['vertices'][-1]
            else:
                next_v = edge_candidate['vertices'][0]
            edges_traversed.append(edge_candidate['edge_index'])
            nodes_traversed.append(next_v)
            bottom_edge_candidates.append(edge_candidate['edge_index'])
            curr_v = next_v
            while curr_v != bottom_vertex_index and curr_v != top_vertex_index:
                edge_candidates = wire_graph.nodes[curr_v].edges_in_out
                edge_candidates = [e for e in edge_candidates if e['edge_index'] not in edges_traversed]
                if len(edge_candidates) > 1:
                    edge_candidates = [e for e in edge_candidates if e['vertices'][0] not in nodes_traversed[:-1] and e['vertices'][1] not in nodes_traversed[:-1]]
                if len(edge_candidates) > 1:
                    edge_candidates = [e for e in edge_candidates if e['edge_index'] not in edges_traversed]
                edge_candidate = edge_candidates[0]
                if edge_candidate['vertices'][0] == curr_v:
                    next_v = edge_candidate['vertices'][1]
                else:
                    next_v = edge_candidate['vertices'][0]
                edges_traversed.append(edge_candidate['edge_index'])
                bottom_edge_candidates.append(edge_candidate['edge_index'])
                nodes_traversed.append(next_v)
                curr_v = next_v
            if curr_v == bottom_vertex_index:
                bottom_edges = bottom_edge_candidates.copy()
                find_bottom_edges = True
            else:
                # 部分vertical edges，重新遍历
                continue
        vertical_edges = [edge_index for edge_index in wire_graph.edge_dict if edge_index not in top_edges and edge_index not in bottom_edges]

        # dot_edges_axis = {edge_index: np.dot(np.array(all_vertices[edge_vertices[1]] - all_vertices[edge_vertices[0]]), normal_axis) for edge_index, edge_vertices in wire_graph.edge_dict.items()}
        # dot_edges_center = {edge_index: np.dot(np.array(all_vertices[edge_vertices[0]] - center), normal_axis) for edge_index, edge_vertices in wire_graph.edge_dict.items()}
        # top_and_bottom_edges = [edge_index for edge_index, dot_product in dot_edges_axis.items() if np.abs(dot_product) < 1e-8] # 垂直于轴的边
        # dot_edges_center_top_bottom = {edge_index: dot_edges_center[edge_index] for edge_index in top_and_bottom_edges}
        # top_dot_value = max(dot_edges_center_top_bottom.values())
        # bottom_dot_value = min(dot_edges_center_top_bottom.values())

        # top_edges = [edge_index for edge_index in top_and_bottom_edges if np.abs(dot_edges_center_top_bottom[edge_index] - top_dot_value) < 1e-8] # 在top_vertex一侧的边
        # bottom_edges = [edge_index for edge_index in top_and_bottom_edges if np.abs(dot_edges_center_top_bottom[edge_index] - bottom_dot_value) < 1e-8] # 在bottom_vertex一侧的边
        # vertical_edges = [edge_index for edge_index in wire_graph.edge_dict if edge_index not in top_edges and edge_index not in bottom_edges]


        new_ordered_edges_top = []
        new_ordered_edges_bottom = []
        new_ordered_edges_vertical = []
        new_ordered_edges_vertical_reversed = []
        traversed_edges = []

        # 遍历顺序，top_edges, vertical edges, bottom_edges, vertical_edges
        # 1. top edges, 右手定则方向和轴方向相反
        if len(top_edges)  == 1: # we encounter a single loop edge, need to handle it separately about the orientation
            edge = edges[top_edges[0]]
            idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge, all_vertices)
            # path_winding_normal = get_polygon_winding_normal(sampled_points, all_vertices)
            path_vertices = sampled_points
            path_edges = [top_edges[0]]
            path_edges_orientation = ["Forward"]
            path_winding_normal = get_polygon_winding_normal_from_vertices(path_vertices)

        else:
            start_vertex_index = top_vertex_index 
            start_edge_index = [i for i in top_edges if wire_graph.edge_dict[i][0] == start_vertex_index][0]
            path_vertices = []
            path_edges = [start_edge_index]
            edge_data = edges[start_edge_index]
            idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge_data, all_vertices)
            path_vertices.extend(sampled_points)
            path_edges_orientation = ["Forward"]

            curr_v = wire_graph.edge_dict[start_edge_index][1]
            while curr_v != start_vertex_index:

                edges_in_out = wire_graph.nodes[curr_v].edges_in_out
                # Only consider edges in top_edges
                edges_out_in_top_edges = [e for e in edges_in_out if e['edge_index'] in top_edges and e['edge_index'] not in path_edges and e['edge_index'] not in traversed_edges]
                assert len(edges_out_in_top_edges) == 1
                edge_out = edges_out_in_top_edges[0]
                if edge_out['vertices'][0] == curr_v:
                    next_v = edge_out['vertices'][1] 
                    path_edges_orientation.append("Forward")
                else:
                    next_v = edge_out['vertices'][0]
                    path_edges_orientation.append("Reversed")
                path_edges.append(edge_out['edge_index'])
                traversed_edges.append(edge_out['edge_index'])
                edge_data = edges[edge_out['edge_index']]
                idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge_data, all_vertices)
                if path_edges_orientation[-1] == "Forward":
                    path_vertices.extend(sampled_points)
                else:
                    path_vertices.extend(sampled_points[::-1])
                curr_v = next_v



            path_winding_normal = get_polygon_winding_normal_from_vertices(path_vertices)

        dot_product = np.dot(path_winding_normal, normal_axis)
        loop_needs_reversal = dot_product > 0 # 如果和轴方向一致，需要反转

        for i in range(len(path_edges)):
            # v_path_start = path_vertices[i]
            # v_path_end = path_vertices[(i + 1) % len(path_vertices)]
            edge_data = edges[path_edges[i]]
            new_ordered_edges_top.append({
                "edge_index": edge_data['edge_index'],
                "orientation": path_edges_orientation[i]
            })

        if loop_needs_reversal:
            new_ordered_edges_top = new_ordered_edges_top[::-1]
            for i in range(len(new_ordered_edges_top)):
                new_ordered_edges_top[i]['orientation'] = "Reversed" if new_ordered_edges_top[i]['orientation'] == "Forward" else "Forward"

        # 2. vertical edges, 从 top vertex 到 bottom vertex, 不允许loop

        start_vertex_index = top_vertex_index
        end_vertex_index = bottom_vertex_index
        start_edge_index = [i for i in vertical_edges if wire_graph.edge_dict[i][0] == start_vertex_index or wire_graph.edge_dict[i][1] == start_vertex_index][0]
        nodes_passed = [start_vertex_index]
        path_edges = [start_edge_index]
        path_edges_orientation = ["Forward" if wire_graph.edge_dict[start_edge_index][0] == start_vertex_index else "Reversed"]

        curr_v = wire_graph.edge_dict[start_edge_index][1] if wire_graph.edge_dict[start_edge_index][0] == start_vertex_index else wire_graph.edge_dict[start_edge_index][0]

        while curr_v != end_vertex_index:
            nodes_passed.append(curr_v)
            edges_in_out = wire_graph.nodes[curr_v].edges_in_out
            edges_in_out_in_vertical_edges = [e for e in edges_in_out if e['edge_index'] in vertical_edges]
            if len(edges_in_out_in_vertical_edges) > 1:

                edges_in_out_in_vertical_edges = [e for e in edges_in_out_in_vertical_edges if e['vertices'][0] not in nodes_passed[:-1] and e['vertices'][1] not in nodes_passed[:-1]] # 边没有走过，且不会形成自环
            if len(edges_in_out_in_vertical_edges) > 1:
                edges_in_out_in_vertical_edges = [e for e in edges_in_out_in_vertical_edges if e['edge_index'] not in traversed_edges]
            # assert len(edges_in_out_in_vertical_edges) == 1
            # assert len(edges_in_out_in_vertical_edges) == 1
            edge_in_out = edges_in_out_in_vertical_edges[0]
            if edge_in_out['vertices'][0] == curr_v:
                next_v = edge_in_out['vertices'][1]
                path_edges_orientation.append("Forward")
            else:
                next_v = edge_in_out['vertices'][0]
                path_edges_orientation.append("Reversed")

            path_edges.append(edge_in_out['edge_index'])
            traversed_edges.append(edge_in_out['edge_index'])
            edge_data = edges[edge_in_out['edge_index']]
            # idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge_data, all_vertices)
            # if path_edges_orientation[-1] == "Forward":
            #     path_vertices.extend(sampled_points)
            # else:
            #     path_vertices.extend(sampled_points[::-1])
            curr_v = next_v



        for i in range(len(path_edges)):
            edge_data = edges[path_edges[i]]
            new_ordered_edges_vertical.append({
                "edge_index": edge_data['edge_index'],
                "orientation": path_edges_orientation[i]
            })
        
        
        # 3. bottom edges, 右手定则方向和轴方向一致
        if len(bottom_edges)  == 1: # we encounter a single loop edge, need to handle it separately about the orientation
            edge = edges[bottom_edges[0]]
            idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge, all_vertices)
            path_vertices = sampled_points
            path_edges = [bottom_edges[0]]
            path_edges_orientation = ["Forward"]
            path_winding_normal = get_polygon_winding_normal_from_vertices(path_vertices)
        else:
            start_vertex_index = bottom_vertex_index
            start_edge_index = [i for i in bottom_edges if wire_graph.edge_dict[i][0] == start_vertex_index][0]
            path_vertices = []
            path_edges = [start_edge_index]
            edge_data = edges[start_edge_index]
            idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge_data, all_vertices)
            path_vertices.extend(sampled_points)
            path_edges_orientation = ["Forward"]
            curr_v = wire_graph.edge_dict[start_edge_index][1]
            while curr_v != start_vertex_index:
                # path_vertices.append(curr_v)

                edges_in_out = wire_graph.nodes[curr_v].edges_in_out
                edges_in_out_in_bottom_edges = [e for e in edges_in_out if e['edge_index'] in bottom_edges and e['edge_index'] not in path_edges and e['edge_index'] not in traversed_edges]
                edge_in_out = edges_in_out_in_bottom_edges[0]
                if edge_in_out['vertices'][0] == curr_v:
                    next_v = edge_in_out['vertices'][1]
                    path_edges_orientation.append("Forward")
                else:
                    next_v = edge_in_out['vertices'][0]
                    path_edges_orientation.append("Reversed")
                path_edges.append(edge_in_out['edge_index'])
                traversed_edges.append(edge_in_out['edge_index'])
                edge_data = edges[edge_in_out['edge_index']]
                idx, curve_type, v_indices, arc, sampled_points, is_closed, curve = construct_edge(edge_data, all_vertices)
                if path_edges_orientation[-1] == "Forward":
                    path_vertices.extend(sampled_points)
                else:
                    path_vertices.extend(sampled_points[::-1])
                curr_v = next_v


        # path_vertices = path_vertices[:-1] # The last vertex is the same as the first vertex
        path_winding_normal = get_polygon_winding_normal_from_vertices(path_vertices)

        dot_product = np.dot(path_winding_normal, normal_axis)
        loop_needs_reversal = dot_product < 0 # 如果和轴方向相反，需要反转

        for i in range(len(path_edges)):
            edge_data = edges[path_edges[i]]
            new_ordered_edges_bottom.append({
                "edge_index": edge_data['edge_index'],
                "orientation": path_edges_orientation[i]
            })

        if loop_needs_reversal:
            new_ordered_edges_bottom = new_ordered_edges_bottom[::-1]
            for i in range(len(new_ordered_edges_bottom)):
                new_ordered_edges_bottom[i]['orientation'] = "Reversed" if new_ordered_edges_bottom[i]['orientation'] == "Forward" else "Forward"

        # 4. vertical edges, 从 bottom vertex 到 top vertex
        start_vertex_index = bottom_vertex_index
        end_vertex_index = top_vertex_index
        start_edge_index = [i for i in vertical_edges if wire_graph.edge_dict[i][0] == start_vertex_index or wire_graph.edge_dict[i][1] == start_vertex_index][0]
        nodes_passed = [start_vertex_index]
        path_edges = [start_edge_index]
        path_edges_orientation = ["Forward" if wire_graph.edge_dict[start_edge_index][0] == start_vertex_index else "Reversed"]
        curr_v = wire_graph.edge_dict[start_edge_index][1] if wire_graph.edge_dict[start_edge_index][0] == start_vertex_index else wire_graph.edge_dict[start_edge_index][0]
        while curr_v != end_vertex_index:
            # path_vertices.append(curr_v)
            nodes_passed.append(curr_v)
            edges_in_out = wire_graph.nodes[curr_v].edges_in_out
            edges_in_out_in_vertical_edges = [e for e in edges_in_out if e['edge_index'] in vertical_edges and e not in path_edges]
            if len(edges_in_out_in_vertical_edges) > 1:
                edges_in_out_in_vertical_edges = [e for e in edges_in_out_in_vertical_edges if e['vertices'][0] not in nodes_passed[:-1] and e['vertices'][1] not in nodes_passed[:-1]]
            if len(edges_in_out_in_vertical_edges) > 1:
                edges_in_out_in_vertical_edges = [e for e in edges_in_out_in_vertical_edges if e['edge_index'] not in traversed_edges]
            assert len(edges_in_out_in_vertical_edges) == 1 # 因为是最后的路径，不会有多余的边
            edge_in_out = edges_in_out_in_vertical_edges[0]
            if edge_in_out['vertices'][0] == curr_v:
                next_v = edge_in_out['vertices'][1]
                path_edges_orientation.append("Forward")
            else:
                next_v = edge_in_out['vertices'][0]
                path_edges_orientation.append("Reversed")
            path_edges.append(edge_in_out['edge_index'])
            traversed_edges.append(edge_in_out['edge_index'])
            curr_v = next_v



        for i in range(len(path_edges)):
            edge_data = edges[path_edges[i]]
            new_ordered_edges_vertical_reversed.append({
                "edge_index": edge_data['edge_index'],
                "orientation": path_edges_orientation[i]
            })

        # if orientation == 'Forward':
        #     new_ordered_edges = new_ordered_edges_top + new_ordered_edges_vertical + new_ordered_edges_bottom + new_ordered_edges_vertical_reversed
        # else:
        #     new_ordered_edges_bottom = new_ordered_edges_bottom[::-1]
        #     for i in range(len(new_ordered_edges_bottom)):
        #         new_ordered_edges_bottom[i]['orientation'] = "Reversed" if new_ordered_edges_bottom[i]['orientation'] == "Forward" else "Forward"
        #     new_ordered_edges_top = new_ordered_edges_top[::-1]
        #     for i in range(len(new_ordered_edges_top)):
        #         new_ordered_edges_top[i]['orientation'] = "Reversed" if new_ordered_edges_top[i]['orientation'] == "Forward" else "Forward"
           
        #     new_ordered_edges = new_ordered_edges_bottom + new_ordered_edges_vertical_reversed + new_ordered_edges_top + new_ordered_edges_vertical
        new_ordered_edges = new_ordered_edges_top + new_ordered_edges_vertical + new_ordered_edges_bottom + new_ordered_edges_vertical_reversed
        outer_wire_data['ordered_edges'] = new_ordered_edges
        face['wires'][wire_idx] = outer_wire_data
        return face



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    root_dir = args.data_path
    to_process_paths = []
    if os.path.isfile(root_dir):
        to_process_paths.append(root_dir)
    else:
        for f in os.listdir(root_dir):
            if f.endswith('.json'):
                to_process_paths.append(os.path.join(root_dir, f))
    for data_path in to_process_paths:
        print(data_path)

    # data_path = r"D:\abc_json\step2json_freecad\00000046_1a67c6032bbd479492910b39_step_000.json"
        vertices, edges, faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping = load_json(data_path)
        # for key in wires_mapping:
        #     print(key)
        for edge_idx, edge in enumerate(edges):
            # edge_index = f"{edge['edge_index']:05d}"
            # all_wire_indexes = wire_reversed_mapping[edge_index]

            # 1. Simplify bsplines to Lines or Circles (if possible)
            if edge['curve_type'] == 'BSplineCurve':
                new_edge = simplify_bspline_curve(edge, vertices)
                if new_edge is not None:
                    edge = new_edge
                    edges[edge_idx] = edge

            # 2. Remove edges with length < 1e-6, remember to update wires, faces

        # 3. Fix face orientation, remember to update wire orientation
        # for face_idx, face in enumerate(faces):
        #     if face['orientation'] == 'Reversed':
        #         # 3.1 For outer_wire, if the face is a periodic surface, detect the two closed circles and the connecting edge. 
        #         if face['surface_type'] == 'Cylinder' or face['surface_type'] == 'Cone':
        #             is_periodic = (face['parameter_range'][1] - face['parameter_range'][0]) >= (2 * np.pi - 1e-4)
        #             if is_periodic:
        #                 # Treat outer and inner wires separately
        #                 faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping = update_periodic_face_outer_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping)
        #                 faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping = update_non_periodic_face_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping, update_outer=False)

        #             else:
        #                 faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping = update_non_periodic_face_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping, update_outer=True)
        #         else:
        #             faces, wires, wires_to_face, wires_mapping, wire_reversed_mapping = update_non_periodic_face_wire(face_idx, face, faces, edges, wires, wires_to_face, wires_mapping, wire_reversed_mapping, update_outer=True)

        # 4. Break closed circles into two half edges, remember to update vertices, wires, edges, faces...

        edge_idx = 0
        while edge_idx < len(edges):
            edge = edges[edge_idx]
            # if edge['curve_type'] == 'Circle' or edge['curve_type'] == 'BSplineCurve' and len(edge['vertices']) == 1:
            if edge['curve_type'] == 'Circle' and len(edge['vertices']) == 1:
                print(f"Trying to split {edge['curve_type']} on edge {edge['edge_index']}")
                vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping = split_circle_bspline(edge, vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping)
            edge_idx += 1

        # 5. Fix wire orientation

        for face_idx, face in enumerate(faces):
            faces[face_idx] = fix_wire(face, edges, vertices)

        save_json(vertices, edges, faces, os.path.join(os.path.dirname(data_path), os.path.basename(data_path).replace('.json', '_fixed.json')))

        # for edge_idx, edge in enumerate(edges):
        #     edge_index = f"{edge['edge_index']:05d}"
        #     all_wire_indexes = wire_reversed_mapping[edge_index]

        #     if len(edge['vertices']) == 1:
        #         # print(edge_index)
        #         try:
        #             assert edge['curve_type'] == 'Circle', f"{edge['curve_type']} != Circle"
        #         except:
        #             # print(edge)
        #             continue
        #         center = np.array(edge.get('curve_definition').get('center'))
        #         normal = np.array(edge.get('curve_definition').get('normal'))
        #         radius = edge.get('curve_definition').get('radius')
        #         length = edge.get('length')
        #         assert np.allclose(np.pi * 2 * radius, length)
        #         first_parameter = edge.get('first_parameter')
        #         last_parameter = edge.get('last_parameter')
        #         assert np.allclose(last_parameter - first_parameter, 2 * np.pi), f"edge_index: {edge_index}, [{first_parameter}, {last_parameter}], Radius: {radius}, Length: {length}"
        #         # assert np.allclose(, last_parameter), f"edge_index: {edge_index}, [{first_parameter}, {last_parameter}], Radius: {radius}, Length: {length}"



