import json
import numpy as np
from scipy.optimize import least_squares
from glob import glob
from collections import defaultdict
from freecad_visualize_json_pythonocc import create_bspline_from_params, sample_points_on_curve
import os
import argparse

def load_json(file_path):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    root_dir = args.data_path
    to_process_paths = []
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


        # 4. Break closed circles into two half edges, remember to update vertices, wires, edges, faces...

        edge_idx = 0
        while edge_idx < len(edges):
            edge = edges[edge_idx]
            # if edge['curve_type'] == 'Circle' or edge['curve_type'] == 'BSplineCurve' and len(edge['vertices']) == 1:
            if edge['curve_type'] == 'Circle' and len(edge['vertices']) == 1:
                print(f"Trying to split {edge['curve_type']} on edge {edge['edge_index']}")
                vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping = split_circle_bspline(edge, vertices, edges, faces, wires_to_face, wires_mapping, wire_reversed_mapping)
            edge_idx += 1

        save_json(vertices, edges, faces, os.path.join(root_dir, os.path.basename(data_path).replace('.json', '_fixed.json')))

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



