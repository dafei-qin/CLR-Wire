from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface, Geom_BezierCurve, Geom_BSplineCurve, Geom_Circle 
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopTools import TopTools_HSequenceOfShape
import json
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec, gp_Circ, gp_Lin, gp_Ax1, gp_Ax3, gp_Elips, gp_Hypr, gp_Parab, gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus
from OCC.Core.Geom import Geom_Circle, Geom_Line, Geom_BSplineCurve, Geom_TrimmedCurve, Geom_Ellipse, Geom_Hyperbola, Geom_Parabola, Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment, GC_MakeArcOfEllipse, GC_MakeArcOfHyperbola, GC_MakeArcOfParabola
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace ,BRepBuilderAPI_DisconnectedWire, BRepBuilderAPI_EmptyWire, BRepBuilderAPI_NonManifoldWire, BRepBuilderAPI_WireDone
from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepLib import breplib
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeExtend import ShapeExtend_DONE1
from OCC.Core.BRepCheck import BRepCheck_Analyzer

import polyscope as ps
import polyscope.imgui as psim

from freecad_visualize_json_pythonocc import create_arc_from_params, create_line_from_points, create_bspline_from_params, get_wire_vertices_and_lines, extract_mesh_from_face, create_ellipse_from_params, create_cylindrical_face_mesh, _build_and_mesh_face_robustly

from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_Orientation

ps.init()

def generate_edge_gradient_colors(num_points: int, is_closed: bool = False):
        """
        Generate gradient colors for an edge from light red (start) to dark red (end)
        
        Args:
            num_points (int): Number of points on the edge
            is_closed (bool): Whether the edge is closed (affects how gradient is applied)
            
        Returns:
            numpy.array: Color array with shape (num_points, 3)
        """
        colors = np.zeros((num_points, 3))
        
        if is_closed:
            # For closed curves, create a gradient that loops back
            for i in range(num_points):
                # Use a smooth transition that loops
                t = i / num_points
                # Apply a cosine function to create smooth looping gradient
                fade_factor = (1.0 + np.cos(2 * np.pi * t)) / 2.0
                colors[i] = np.array() * fade_factor + np.array([0.0, 0.0, 1.0]) * (1 - fade_factor)
        else:
            # For open curves, linear gradient from start to end
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 0
                colors[i] = np.array([1.0, 0.0, 0.0]) * (1 - t) + np.array([0.0, 0.0, 1.0]) * t
        
        return colors


def create_planar_face_mesh(face_index, wires, all_edges):



    # --- 2. 重建所有的线框 (Wires) ---


    for wire_info in wires:
        wire_builder = BRepBuilderAPI_MakeWire()
        edges_in_wire = []
        for edge_ref in wire_info["ordered_edges"]:
            edge_index = edge_ref["edge_index"]
            orientation = edge_ref["orientation"]
            
            # 从预创建的边集合中查找对应的边
            if edge_index not in all_edges:
                raise KeyError(f"未在提供的 'all_edges' 集合中找到索引为 {edge_index} 的边。")
            
            edge_to_add = all_edges[edge_index]
            
            # 根据方向要求，可能需要翻转边
            if orientation == "Reversed":
                edge_to_add = edge_to_add.Reversed()
                # edge_to_add = edge_to_add
            wire_builder.Add(edge_to_add)
            print(f'Adding edge {edge_index} to wire, edge close statue is {edge_to_add.Closed()}, Current wire close statue is {wire_builder.Wire().Closed()}')
            
            edges_in_wire.append(edge_index)
            # interm_topo_wires.append(wire_builder.Wire())
            interm_wire_points, interm_wire_lines = get_wire_vertices_and_lines(wire_builder.Wire(), 0.1)

            interm_wire_name = '_'.join([f"{i:04d}" for i in edges_in_wire])
            colors = generate_edge_gradient_colors(interm_wire_points.shape[0], False)
            # curve_net = ps.register_curve_network(interm_wire_name, interm_wire_points, interm_wire_lines, enabled=False)
            curve_net = ps.register_point_cloud(interm_wire_name, interm_wire_points, enabled=False)
            curve_net.add_color_quantity("direction_gradient", colors,  enabled=True)

        if not wire_builder.IsDone():
            # 通常在这里失败意味着边没有正确地首尾相连
            raise RuntimeError(f"构建线框失败，请检查索引为 {face_index:03d} 的面的边连接性。")

def create_cylindrical_face_mesh(face_index: int, position: list, axis: list, radius: float, wires: list, all_edges: dict):
    """
    根据从JSON加载的圆柱面数据及其边界线，重建一个拓扑面并进行网格化。

    :param face_index: 面的索引，用于错误提示。
    :param position: 圆柱轴线上的一点 [x, y, z]。
    :param axis: 圆柱的轴方向 [x, y, z]。
    :param radius: 圆柱的半径。
    :param wires: 包含线框信息（is_outer, ordered_edges）的列表。
    :param all_edges: 一个字典，其键为 edge_index，值为预先创建好的 TopoDS_Edge 对象。
    :return: 一个元组 (final_face, vertices, faces)，包含B-Rep对象和通用网格表示。
    """

    # --- 1. 创建无限高的基准几何圆柱 ---
    # 首先定义圆柱的坐标系
    center_pnt = gp_Pnt(position[0], position[1], position[2])
    axis_dir = gp_Dir(axis[0], axis[1], axis[2])
    cylinder_axis = gp_Ax3(center_pnt, axis_dir) # Z轴是圆柱的轴线

    # 创建几何圆柱对象
    gp_cylinder = gp_Cylinder(cylinder_axis, radius)
    geom_cylinder = Geom_CylindricalSurface(gp_cylinder)
    context_face = BRepBuilderAPI_MakeFace(gp_cylinder).Face() # For shapefix_wire
    # --- 2. 重建所有的线框 (Wires) - 逻辑与平面版本完全相同 ---
    outer_wire = None
    inner_wires = []
    outer_wire_points = None
    outer_wire_lines = None
    outer_wire_name = ''
    inner_wire_points_list = []
    inner_wire_lines_list = []
    inner_wire_name_list = []
    for wire_info in wires:
        wire_builder = BRepBuilderAPI_MakeWire()
        # edges_in_wire = set()
        edge_in_wire = []
        for edge_ref in wire_info["ordered_edges"]:
            edge_index = edge_ref["edge_index"]
            orientation = edge_ref["orientation"]
            
            # if edge_index in edges_in_wire:
            #     print(f"DEBUG INFO: edge_index: {edge_index:03d} already in wire for face {face_index:03d}")
            #     continue
            if edge_index not in all_edges:
                raise KeyError(f"未在提供的 'all_edges' 集合中找到索引为 {edge_index} 的边。")
            
            edge_to_add = all_edges[edge_index]
            
            if orientation == "Reversed":
                edge_to_add = edge_to_add.Reversed()
            

            # wire_builder.Add(edge_to_add)
            # print(f'Adding edge {edge_index} to wire, edge close statue is {edge_to_add.Closed()}, Current wire close statue is {wire_builder.Wire().Closed()}')
            edge_in_wire.append(edge_index)
            interm_wire_points, interm_wire_lines = get_wire_vertices_and_lines(wire_builder.Wire(), 0.1)

            interm_wire_name = '_'.join([f"{i:04d}" for i in edge_in_wire])
            colors = generate_edge_gradient_colors(interm_wire_points.shape[0], False)
            # curve_net = ps.register_curve_network(interm_wire_name, interm_wire_points, interm_wire_lines, enabled=False)
            curve_net = ps.register_point_cloud(interm_wire_name, interm_wire_points, enabled=False)
            curve_net.add_color_quantity("direction_gradient", colors,  enabled=True)
            
           


        
        topo_wire = wire_builder.Wire()
        

        # c. 区分内外边界 (逻辑不变)
        if wire_info["is_outer"]:
            if outer_wire is not None:
                raise ValueError(f"面 {face_index:03d} 有多个外边界，这是不允许的。")
            outer_wire = topo_wire
            outer_wire_points, outer_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            outer_wire_name = '_'.join([f"{i:04d}" for i in edge_in_wire])
        else:
            inner_wires.append(topo_wire)
            inner_wire_points, inner_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            inner_wire_points_list.append(inner_wire_points)
            inner_wire_lines_list.append(inner_wire_lines)
            inner_wire_name_list.append('_'.join([f"{i:04d}" for i in edge_in_wire]))

    if outer_wire is None:
        raise ValueError(f"面 {face_index:03d} 没有找到外边界。")

    # --- 3. 构建拓扑面 (Face)，并添加孔洞 ---
    # 使用基准圆柱和外边界线框创建基础面
    face_builder = BRepBuilderAPI_MakeFace(geom_cylinder, outer_wire)

    
    # 将所有内边界线框作为孔洞添加到面中
    for hole_wire in inner_wires:
        #hole_wire = hole_wire.Reversed()
        hole_wire.Orientation(TopAbs_Orientation.TopAbs_REVERSED)
        face_builder.Add(hole_wire)
        # face_builder.Add(hole_wire)
        
    if not face_builder.IsDone():
        raise RuntimeError(f"创建面 {face_index:03d} 失败。")
        
    final_face = face_builder.Face()
    breplib.UpdateTolerances(final_face, True)
    
    # 2. 这是一个强大的修复工具，可以解决多种拓扑和几何问题
    print(face_builder.Error())
    fixer = ShapeFix_Shape(final_face)
    fixer.Perform()
    healed_face = fixer.Shape()
    step_writer = STEPControl_Writer()
    dd = step_writer.WS().TransferWriter().FinderProcess()
    print('dd', dd)
    Interface_Static_SetCVal("write.step.schema", "AP203")
    step_writer.Transfer(healed_face, STEPControl_AsIs)
    status = step_writer.Write('./cylinder_face.step')
    if status != IFSelect_RetDone:
        raise AssertionError("write failed")
    # breplib.BuildCurves3d(final_face) # 重新构建面上曲线

    # --- 4. 对面进行网格化并提取通用表示 ---
    # mesh_tolerance = 0.2
    # BRepMesh_IncrementalMesh(final_face, mesh_tolerance)
    final_face, vertices, faces = _build_and_mesh_face_robustly(face_builder, linear_deflection=0.1, angular_deflection=0.2)
    
    # 调用您优化过的辅助函数来提取网格数据
    # vertices, faces = extract_mesh_from_face(final_face)
    
    print(f"成功创建、修复并网格化了圆柱面 (face_index: {face_index:03d})。(vertices.shape: {np.array(vertices).shape}, faces.shape: {np.array(faces).shape})")
    
    return final_face, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list

       




if __name__ == "__main__":
    cad_data = json.load(open('d:/abc_json/step2json_freecad_simplified/00000056_005.json', 'r'))
    # cad_data = json.load(open('c:/Users/Dafei Qin/cylinder_cut.json', 'r'))
    # cad_data = json.load(open('c:/Users/Dafei Qin/00000056_005.json', 'r'))
    vertex_positions = np.array(cad_data.get('vertices', []))
    edges_list = cad_data.get('edges', [])
    faces_list = cad_data.get('faces', [])
    all_edges = {}
    all_outer_wires = []
    all_inner_wires = []


    sampled_edges = []
    for edge in edges_list:
        # print(edge)
        idx = edge['edge_index']
        curve_type = edge.get('curve_type')
        v_indices = edge.get('vertices')
        print(f"Fitting edge {idx:03d} of type {curve_type}")
        if edge.get('curve_type') == 'Circle':
            center = np.array(edge.get('curve_definition').get('center'))
            normal = np.array(edge.get('curve_definition').get('normal'))
            radius = edge.get('curve_definition').get('radius')
            if len(v_indices) == 2:
                start_vertex = vertex_positions[v_indices[0]]
                end_vertex = vertex_positions[v_indices[1]]
            else:
                start_vertex = vertex_positions[v_indices[0]]
                end_vertex = vertex_positions[v_indices[0]]
            arc, sampled_points, is_closed = create_arc_from_params(center, normal, radius, start_vertex, end_vertex)
            # sampled_edges.append( {'idx': idx, 'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
        elif edge.get('curve_type') == 'Ellipse':
                curve_def = edge.get('curve_definition', {})
                center = np.array(curve_def.get('center'))
                x_direction = np.array(curve_def.get('major_axis_direction'))
                y_direction = np.array(curve_def.get('minor_axis_direction'))
                normal = np.array(curve_def.get('normal'))
                major_radius = curve_def.get('major_radius')
                minor_radius = curve_def.get('minor_radius')
                if len(v_indices) == 2:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[1]]
                else:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[0]]
                arc, sampled_points, is_closed = create_ellipse_from_params(center, x_direction, y_direction, normal, major_radius, minor_radius, start_vertex, end_vertex)
            
        elif edge.get('curve_type') == 'Line':
            if len(v_indices) >= 2:
                start_vertex = vertex_positions[v_indices[0]]
                end_vertex = vertex_positions[v_indices[1]]
                arc, sampled_points, is_closed = create_line_from_points(start_vertex, end_vertex)
                # sampled_edges.append( {'idx': idx, 'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
            else:
                print(f"Warning: Edge {idx} of type 'Line' has insufficient vertices ({len(v_indices)})")
                
        elif edge.get('curve_type') == 'BSplineCurve':
            curve_def = edge.get('curve_definition', {})
            degree = curve_def.get('degree')
            is_periodic = curve_def.get('is_periodic', False)
            is_closed = len(v_indices) == 1
            control_points = curve_def.get('control_points', [])
            knots = curve_def.get('knots', [])
            multiplicities = curve_def.get('multiplicities', [])
            
            # 验证B样条曲线参数
            if degree is not None and control_points and knots and multiplicities:
                if len(v_indices) >= 2:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[-1]]  # 使用最后一个顶点作为终点
                elif len(v_indices) == 1:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[0]]
                else:
                    # 如果没有足够的顶点索引，使用控制点的首尾作为起终点
                    start_vertex = control_points[0]
                    end_vertex = control_points[-1]
                    
                arc, sampled_points, is_closed = create_bspline_from_params(
                    degree, is_periodic, is_closed, control_points, knots, multiplicities, 
                    start_vertex, end_vertex
                )
            else:
                print(f"Warning: Edge {idx} of type 'BSplineCurve' has incomplete curve definition")
            
        else:
            print(f"Warning: Unsupported curve type '{curve_type}' for edge {idx}")
            continue
        sampled_edges.append( {'idx': idx,  'arc': arc,'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
        all_edges[idx] = arc

    sampled_faces = []
    print('='*50)
    for face in faces_list:

            surface_idx = face['face_index']
            if surface_idx != 8:
                continue
            surface_type = face['surface_type']
            wires = face['wires']
            surface_orientation = face['orientation']
            if surface_orientation == 'Reversed':
                for w in wires:
                    for e in w['ordered_edges']:
                        if e['orientation'] == 'Reversed':
                            e['orientation'] = 'Forward'
                        else:
                            e['orientation'] = 'Reversed'
            if surface_type == 'Cylinder':
                # position = np.array(face['surface_definition']['position'])
                # normal = np.array(face['surface_definition']['normal'])
                # create_planar_face_mesh(surface_idx, wires, all_edges)
                position = np.array(face['surface_definition']['position'])
                axis = np.array(face['surface_definition']['axis'])
                radius = face['surface_definition']['radius']
                face_mesh, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list = create_cylindrical_face_mesh(surface_idx, position, axis, radius, wires, all_edges)
                ps.register_point_cloud(f'cylinder {surface_idx}', np.array(vertices), enabled=True)
                # sampled_faces.append( {'idx': surface_idx, 'surface_type': surface_type, 'face_mesh': face_mesh, 'vertices': vertices, 'faces': faces})


            # for inner_wire_points, inner_wire_lines, inner_wire_name in zip(inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list):
            #     all_inner_wires.append([inner_wire_points, inner_wire_lines, inner_wire_name])
            # sampled_faces.append( {'idx': surface_idx, 'surface_type': surface_type, 'face_mesh': face_mesh, 'vertices': vertices, 'faces': faces})
     
    ps.show()