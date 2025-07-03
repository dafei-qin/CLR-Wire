import json
import numpy as np
import polyscope as ps
# 【修正】从geomdl导入正确的模块：BSpline用于曲线定义
from geomdl import BSpline
import triangle as tr
from shapely.geometry import Polygon, MultiPolygon
import scipy.optimize as so

def find_closest_parameter(curve, target_point, tolerance=1e-6):
    """
    Find the parameter value on a B-spline curve that gives the closest point to the target point.
    Uses optimization to minimize the distance between curve point and target point.
    
    Args:
        curve: geomdl B-spline curve object
        target_point: numpy array of target coordinates [x, y, z]
        tolerance: optimization tolerance
    
    Returns:
        parameter value (float) that gives the closest point on the curve
    """
    def distance_to_target(u):
        # Evaluate curve at parameter u
        curve_point = np.array(curve.evaluate_single(u[0]))
        # Return squared distance to target point
        return np.sum((curve_point - target_point) ** 2)
    
    # Use optimization to find parameter that minimizes distance
    # Start search at parameter 0.5 (middle of curve)
    result = so.minimize(distance_to_target, [0.5], bounds=[(0, 1)], 
                        method='L-BFGS-B', options={'ftol': tolerance})
    
    return result.x[0]

def sample_circle_arc(edge_definition, start_vertex_pos, end_vertex_pos, num_points=50):
    """
    基于起点和终点顶点，对圆弧进行精确采样。
    """
    curve_def = edge_definition.get("curve_definition")
    if not curve_def: return np.array([])
    
    center = np.array(curve_def['center'])
    normal = np.array(curve_def['normal'])
    radius = curve_def['radius']
    
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-9: return np.array([])
    normal /= norm_val
    
    # 建立圆所在的局部坐标系
    temp_vec = np.array([0., 0., 1.])
    if np.abs(np.dot(normal, temp_vec)) > 0.99: temp_vec = np.array([0., 1., 0.])
    u_dir = np.cross(normal, temp_vec)
    u_dir /= np.linalg.norm(u_dir)
    v_dir = np.cross(normal, u_dir)
    
    start_vec = start_vertex_pos - center
    end_vec = end_vertex_pos - center

    t_start = np.arctan2(np.dot(start_vec, v_dir), np.dot(start_vec, u_dir))
    t_end = np.arctan2(np.dot(end_vec, v_dir), np.dot(end_vec, u_dir))

    original_t_start = edge_definition.get("first_parameter", 0)
    original_t_end = edge_definition.get("last_parameter", 0)
    
    # 调整角度周期，确保沿正确方向
    while t_end < t_start - np.pi: t_end += 2 * np.pi
    while t_end > t_start + np.pi: t_end -= 2 * np.pi
    
    original_span = original_t_end - original_t_start
    if original_span > np.pi and t_end < t_start:
        t_end += 2 * np.pi
    elif original_span < -np.pi and t_end > t_start:
        t_end -= 2 * np.pi
            
    param_range = np.linspace(t_start, t_end, num_points) 
    sampled_points = center + radius * (np.cos(param_range)[:, np.newaxis] * u_dir + np.sin(param_range)[:, np.newaxis] * v_dir)
    
    sampled_points[0] = start_vertex_pos
    sampled_points[-1] = end_vertex_pos
    
    return sampled_points

def sample_bspline_curve(edge_definition, start_vertex_pos, end_vertex_pos, num_points=100):
    """
    【函数已修正】
    基于起点和终点顶点，对B样条曲线进行精确采样。
    """
    curve_def = edge_definition.get("curve_definition")
    if not curve_def: return np.array([])
    print(f"处理第{edge_definition.get('edge_index'):03d}条BSpline曲线")
    if curve_def.get('is_periodic', False):
        print(f"第{edge_definition.get('edge_index'):03d}条BSpline曲线是周期性的，暂时跳过")
        return np.array([])
    
    crv = BSpline.Curve()
    crv.degree = curve_def['degree']
    crv.ctrlpts = curve_def['control_points']
    crv.knotvector = curve_def['knots']

    # --- 【修正】使用自定义函数查找最近点的参数值 ---
    t_start = find_closest_parameter(crv, start_vertex_pos, tolerance=1e-6)
    t_end = find_closest_parameter(crv, end_vertex_pos, tolerance=1e-6)
    # -------------------------------------------------------------

    param_range = np.linspace(t_start, t_end, num_points)
    sampled_points = np.array(crv.evaluate_list(param_range))
    
    sampled_points[0] = start_vertex_pos
    sampled_points[-1] = end_vertex_pos

    return sampled_points
        


# ... (project_points_to_cylinder_uv 和 lift_uv_to_cylinder_3d 函数保持不变) ...
def project_points_to_cylinder_uv(points_3d, cylinder_def):
    center = np.array(cylinder_def["position"])
    axis = np.array(cylinder_def["axis"])
    radius = cylinder_def["radius"]
    axis /= np.linalg.norm(axis) # 极坐标系轴
    z_axis = axis
    # u_dir = np.array([1.,0.,0.]) if np.abs(np.dot(axis, [1,0,0]))<0.9 else np.array([0.,1.,0.])
    if np.allclose(np.abs(z_axis), [1.0, 0.0, 0.0]):
        temp_vec = np.array([0.0, 1.0, 0.0])
    else:
        temp_vec = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(temp_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    # y_axis 已经是单位向量
    vec = points_3d - center
    h = np.dot(vec, z_axis)

    #    在XY平面上的投影分量
    x_prime = np.dot(vec, x_axis)
    y_prime = np.dot(vec, y_axis)

    # 7. 计算角度 theta (U坐标)
    #    使用 atan2 确保360度范围的正确性
    theta = np.arctan2(y_prime, x_prime)
    
    # 将 theta 范围从 [-pi, pi] 调整到 [0, 2*pi]
    # if theta < 0:
    #     theta += 2 * np.pi
    theta[theta < 0] += 2 * np.pi
    # u_dir -= np.dot(u_dir, axis) * axis
    # u_dir /= np.linalg.norm(u_dir)
    # v_dir = np.cross(axis, u_dir)
    # vec_c_p = points_3d - center
    # v_coords = np.dot(vec_c_p, axis)
    # x_local = np.dot(vec_c_p, u_dir)
    # y_local = np.dot(vec_c_p, v_dir)
    # u_coords = np.arctan2(y_local, x_local)
    return np.column_stack((theta, h))

# def lift_uv_to_cylinder_3d(points_uv, cylinder_def):
#     center = np.array(cylinder_def["position"])
#     axis = np.array(cylinder_def["axis"])
#     radius = cylinder_def["radius"]
#     axis /= np.linalg.norm(axis)
#     u_dir = np.array([1.,0.,0.]) if np.abs(np.dot(axis,[1,0,0]))<0.9 else np.array([0.,1.,0.])
#     u_dir -= np.dot(u_dir, axis) * axis
#     u_dir /= np.linalg.norm(u_dir)
#     v_dir = np.cross(axis, u_dir)
#     u = points_uv[:, 0]
#     v = points_uv[:, 1]
#     points_3d = center + radius * (np.cos(u)[:, np.newaxis] * u_dir + np.sin(u)[:, np.newaxis] * v_dir) + v[:, np.newaxis] * axis
#     return points_3d

def lift_uv_to_cylinder_3d(points_uv, cylinder_def):
    cylinder_axis = np.array(cylinder_def["axis"])
    cylinder_center = np.array(cylinder_def["position"])
    cylinder_radius = cylinder_def["radius"]

    z_axis = cylinder_axis / np.linalg.norm(cylinder_axis)
    
    if np.allclose(np.abs(z_axis), [1.0, 0.0, 0.0]):
        temp_vec = np.array([0.0, 1.0, 0.0])
    else:
        temp_vec = np.array([1.0, 0.0, 0.0])
        
    x_axis = np.cross(temp_vec, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    theta = points_uv[:, 0]
    h = points_uv[:, 1]
    radial_vec = cylinder_radius * (np.cos(theta)[:, np.newaxis] * x_axis + np.sin(theta)[:, np.newaxis] * y_axis)
    axial_vec = h[:, np.newaxis] * z_axis
    points_3d = cylinder_center + radial_vec + axial_vec
    return points_3d

def visualize_surfaces(json_file_path, edge_samples=50, curve_thickness=0.002, show_wireframe=True):
    """
    修改版：可视化顶点、所有曲线以及经过几何清理的平面。
    现在曲线采样将由顶点精确控制。
    """
    print("--- 可视化脚本 (顶点控制版) ---")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        cad_data = json.load(f)
    
    vertex_positions = np.array(cad_data.get('vertices', []))
    edges_list = cad_data.get('edges', [])
    faces_list = cad_data.get('faces', [])

    print("正在对所有边进行参数化采样 (由顶点控制)...")
    sampled_edges = {}
    for edge in edges_list:
        # print(edge)
        idx = edge['edge_index']
        curve_type = edge.get('curve_type')
        v_indices = edge.get('vertices')
        
        if not v_indices:
            continue

        # 处理闭合圆（例如只有一个顶点或者没有顶点但参数是完整周期）
        is_closed_circle = (curve_type == 'Circle' and 
                           (len(v_indices) < 2 or 
                           abs(abs(edge.get("last_parameter", 0) - edge.get("first_parameter", 0)) - 2 * np.pi) < 1e-6))
        is_closed_bspline = (curve_type == 'BSplineCurve' and 
                           (len(v_indices) < 2 ))

        if is_closed_circle:
            # 对于闭合圆，我们使用原始的基于参数的方法
            t_start = edge.get("first_parameter", 0)
            t_end = edge.get("last_parameter", 0)
            
            if abs(abs(t_end - t_start) - 2 * np.pi) > 1e-6:
                t_start, t_end = 0, 2 * np.pi

            param_range = np.linspace(t_start, t_end, edge_samples)
            
            curve_def = edge["curve_definition"]
            center, normal, radius = np.array(curve_def['center']), np.array(curve_def['normal']), curve_def['radius']
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-9: continue
            normal /= norm_val
            temp_vec = np.array([0.,0.,1.])
            if np.abs(np.dot(normal, temp_vec)) > 0.99: temp_vec = np.array([0.,1.,0.])
            u_dir = np.cross(normal, temp_vec)
            u_dir /= np.linalg.norm(u_dir)
            v_dir = np.cross(normal, u_dir)
            sampled_edges[idx] = center + radius * (np.cos(param_range)[:,np.newaxis] * u_dir + np.sin(param_range)[:,np.newaxis] * v_dir)
            continue 
        elif is_closed_bspline:
            print(f"第{edge.get('edge_index'):03d}条BSpline曲线是闭合的，暂时跳过")
            sampled_edges[idx] = np.array([])
            continue
        # 对于所有常规的、有起止点的边
        start_v_pos = vertex_positions[v_indices[0]]
        end_v_pos = vertex_positions[v_indices[1]]

        if curve_type in ['Line', 'LineSegment']:
            sampled_edges[idx] = np.array([start_v_pos, end_v_pos])
        elif curve_type == 'Circle':
            sampled_edges[idx] = sample_circle_arc(edge, start_v_pos, end_v_pos, edge_samples)
        elif curve_type == 'BSplineCurve':
            sampled_edges[idx] = sample_bspline_curve(edge, start_v_pos, end_v_pos, edge_samples)

    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_view_projection_mode("orthographic")

    if show_wireframe:
        import colorsys
        registered_curves = 0
        for edge_idx, points in sampled_edges.items():
            if points is not None and len(points) > 1:
                edges_indices = np.array([[j, j + 1] for j in range(len(points) - 1)])
                hue = (edge_idx * 0.618033988749895) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                curve_name = f"Curve_{edge_idx:03d}"
                ps.register_curve_network(curve_name, points, edges_indices, color=rgb, radius=curve_thickness)
                registered_curves += 1
        
        print(f"已注册 {registered_curves} 条单独的曲线。")
    else:
        print("线框显示已禁用。")

    print("正在为所有平面生成三角网格...")
    face_groups = {'Plane': [], 'Cylinder': []}
    for f in faces_list:
        s_type = f.get('surface_type')
        if s_type in face_groups:
            face_groups[s_type].append(f)
    
    for face_type, face_data_list in face_groups.items():
        for face_data in face_data_list:
            face_idx = face_data.get('face_index')
            outer_loop_3d_list, inner_loops_3d_list = [], []
            for wire in face_data.get('wires', []):
                loop_points = []
                ordered_edges_refs = wire.get('ordered_edges', [])
                if not ordered_edges_refs: continue

                for i, edge_ref in enumerate(ordered_edges_refs):
                    points = sampled_edges.get(edge_ref.get('edge_index'))
                    if points is None or len(points) == 0: continue
                    if edge_ref['orientation'] == 'Reversed': points = points[::-1]
                    
                    if i == 0:
                        loop_points.extend(points.tolist())
                    else:
                        if np.allclose(loop_points[-1], points[0], atol=1e-5):
                            loop_points.extend(points[1:].tolist())
                        else:
                            loop_points.extend(points.tolist())
                
                if loop_points:
                    if wire.get('is_outer'):
                        outer_loop_3d_list = loop_points
                    else:
                        inner_loops_3d_list.append(loop_points)
            
            if len(outer_loop_3d_list) < 3: continue
            outer_loop_3d = np.array(outer_loop_3d_list)
            
            surface_def = face_data['surface_definition']

            if face_type == 'Plane':
                normal = np.array(surface_def['normal'])
                origin_3d = outer_loop_3d[0]
                z_axis = normal / np.linalg.norm(normal)
                x_axis = np.array([1.0, 0.0, 0.0]) if np.abs(np.dot(z_axis,[1,0,0]))<0.9 else np.array([0.0, 1.0, 0.0])
                x_axis -= np.dot(x_axis, z_axis) * z_axis
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                transform_matrix = np.column_stack((x_axis, y_axis))
                outer_loop_2d = np.dot(outer_loop_3d - origin_3d, transform_matrix)
                inner_loops_2d = [np.dot(np.array(loop) - origin_3d, transform_matrix) for loop in inner_loops_3d_list]
            elif face_type == 'Cylinder':
                outer_loop_2d = project_points_to_cylinder_uv(outer_loop_3d, face_data['surface_definition'])
                inner_loops_2d = [project_points_to_cylinder_uv(loop, face_data['surface_definition']) for loop in inner_loops_3d_list]
            else:
                continue

            
            poly_shape = Polygon(outer_loop_2d, holes=[h for h in inner_loops_2d if len(h) >= 3])
            cleaned_shape = poly_shape.buffer(0)
            if cleaned_shape.is_empty: continue
            
            shapes_to_triangulate = list(cleaned_shape.geoms) if isinstance(cleaned_shape, MultiPolygon) else [cleaned_shape]
            
            face_vertices, face_triangles, face_vertex_offset = [], [], 0
            for poly in shapes_to_triangulate:
                exterior_coords = np.array(poly.exterior.coords)[:, :2]
                vertices_2d = exterior_coords[:-1].astype(np.float64)
                seg_input = {'vertices': vertices_2d}
                n_outer = len(vertices_2d)
                outer_segments = [[i, (i + 1) % n_outer] for i in range(n_outer)]
                seg_input['segments'] = outer_segments
                
                if poly.interiors:
                    holes_data = []
                    hole_points = []
                    current_offset = n_outer
                    for interior in poly.interiors:
                            hole_coords = np.array(interior.coords)[:-1,:2].astype(np.float64)
                            holes_data.append(hole_coords)
                            n_hole = len(hole_coords)
                            seg_input['segments'].extend([[current_offset + i, current_offset + (i+1)%n_hole] for i in range(n_hole)])
                            hole_points.append(np.mean(hole_coords, axis=0))
                            current_offset += n_hole
                    seg_input['vertices'] = np.vstack([vertices_2d] + holes_data)
                    seg_input['holes'] = hole_points

                t = tr.triangulate(seg_input, 'p')
                

                tri_vertices_2d = np.array(t['vertices'])
                points_on_tangent_plane = origin_3d + tri_vertices_2d[:, 0, np.newaxis] * x_axis + tri_vertices_2d[:, 1, np.newaxis] * y_axis
                
                if face_type == 'Plane':
                    face_vertices.append(points_on_tangent_plane)
                    face_triangles.append(np.array(t['triangles']) + face_vertex_offset)
                    face_vertex_offset += len(points_on_tangent_plane)
                elif face_type == 'Cylinder':
                    face_vertices.append(lift_uv_to_cylinder_3d(points_on_tangent_plane, face_data['surface_definition']))
                    face_triangles.append(np.array(t['triangles']) + face_vertex_offset)
                    face_vertex_offset += len(points_on_tangent_plane)

            if face_vertices:
                final_vertices = np.concatenate(face_vertices)
                final_triangles = np.concatenate(face_triangles)
                
                import colorsys
                hue = (face_idx * 0.618033988749895) % 1.0
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                
                face_name = f"Face_{face_idx:03d}_{face_type}"
                ps.register_surface_mesh(face_name, final_vertices, final_triangles, color=rgb, smooth_shade=True)
                print(f"已注册面 {face_idx} 的网格，顶点数: {len(final_vertices)}, 三角形数: {len(final_triangles)}")

            # except Exception as e:
            #     print(f"错误：处理面 #{face_idx} ({face_type}) 时失败: {e}")
                
    ps.reset_camera_to_home_view()
    print("\n启动Polyscope可视化窗口...")
    ps.show()

if __name__ == '__main__':
    file_to_visualize = r'D:\abc_0000_step_v00\00000030\00000030_ad34a3f60c4a4caa99646600_step_010.json'
    # file_to_visualize = 'C:\\Users\\Dafei Qin\\Feature_reconstruction_data.json'
    
    edge_samples = 50       # 边采样点数
    curve_thickness = 0.002   # 曲线粗细 (可调整: 0.001-0.01)
    show_wireframe = True     # 是否显示线框
    
    print(f"可视化设置:")
    print(f"  文件: {file_to_visualize}")
    print(f"  边采样点数: {edge_samples}")
    print(f"  曲线粗细: {curve_thickness}")
    print(f"  显示线框: {show_wireframe}")
    print()
    
    visualize_surfaces(file_to_visualize, 
                       edge_samples=edge_samples,
                       curve_thickness=curve_thickness, 
                       show_wireframe=show_wireframe)