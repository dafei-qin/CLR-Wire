from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface, Geom_BezierCurve, Geom_BSplineCurve, Geom_Circle 
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.TopTools import TopTools_HSequenceOfShape
import json
import numpy as np

from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec, gp_Circ, gp_Lin, gp_Ax1, gp_Ax3, gp_Elips, gp_Hypr, gp_Parab, gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus, gp_Trsf
from OCC.Core.Geom import Geom_Circle, Geom_Line, Geom_BSplineCurve, Geom_TrimmedCurve, Geom_Ellipse, Geom_Hyperbola, Geom_Parabola, Geom_Plane, Geom_CylindricalSurface, Geom_ConicalSurface, Geom_SphericalSurface
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment, GC_MakeArcOfEllipse, GC_MakeArcOfHyperbola, GC_MakeArcOfParabola
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace ,BRepBuilderAPI_DisconnectedWire, BRepBuilderAPI_EmptyWire, BRepBuilderAPI_NonManifoldWire, BRepBuilderAPI_WireDone, BRepBuilderAPI_Transform
from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepLib import breplib
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeExtend import ShapeExtend_DONE1
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.TopAbs import TopAbs_Orientation
import polyscope as ps
import polyscope.imgui as psim

from freecad_visualize_json_pythonocc import _build_and_mesh_face_robustly_from_topods, extract_mesh_from_face
from itertools import combinations


def build_adjacency_matrix(faces):
    """
    根据给定的面列表构建一个邻接矩阵。
    如果两个面共享至少一条公共边，则认为它们是邻接的。
    矩阵的维度将由最大的face_index确定，以便直接使用原始索引。

    Args:
        faces (list): 从JSON等数据源读取的面属性字典列表。
                      每个字典应包含 'face_index' 和 'wires' 键。
                      'wires' 是一个列表，其中每个元素包含 'ordered_edges'，
                      而 'ordered_edges' 列表中的每个元素都有一个 'edge_index'。

    Returns:
        np.ndarray: NxN 的布尔型邻接矩阵，其中N是最大的face_index + 1。
    """
    # 1. 预处理：提取每个face_index对应的所有edge_index，并找到最大的索引
    if not faces:
        return np.array([[]], dtype=bool)

    face_to_edges = {}
    max_index = -1
    for face in faces:
        face_index = face['face_index']
        if face_index > max_index:
            max_index = face_index
        
        # 使用集合推导式高效地提取所有不重复的edge_index
        edge_set = {
            edge['edge_index'] 
            for wire in face.get('wires', []) 
            for edge in wire.get('ordered_edges', [])
        }
        face_to_edges[face_index] = edge_set

    # 2. 矩阵初始化
    #    矩阵的大小由最大的face_index决定，以直接使用原始索引。
    matrix_size = max_index + 1
    adjacency_matrix = np.full((matrix_size, matrix_size), False, dtype=bool)

    # 3. 邻接判断
    #    使用 itertools.combinations 来高效地获取所有唯一的面索引对
    face_indices = list(face_to_edges.keys())
    for face_idx1, face_idx2 in combinations(face_indices, 2):
        edges1 = face_to_edges[face_idx1]
        edges2 = face_to_edges[face_idx2]
        
        # 检查两个边的集合是否不相交。如果不相交，则它们共享至少一条边。
        # set.isdisjoint(other_set) 比计算完整交集更快。
        if not edges1.isdisjoint(edges2):
            # 直接使用原始的face_index作为矩阵索引
            adjacency_matrix[face_idx1, face_idx2] = True
            adjacency_matrix[face_idx2, face_idx1] = True
            
    return adjacency_matrix


# def build_bspline_surface(face):
#     parameter_range = face['parameter_range']
#     u_min, u_max, v_min, v_max = parameter_range
#     u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v = face['scalar'][:6]
#     # u_degree = face['surface_definition']['u_degree']
#     # is_u_periodic = face['surface_definition']['is_u_periodic']
#     # is_v_periodic = face['surface_definition']['is_v_periodic']
#     # control_points = face['surface_definition']['control_points']

#     u_count = len(control_points)
#     v_count = len(control_points[0]) if u_count > 0 else 0
#     # Create a 2D array of gp_Pnt
#     occ_control_points_array = TColgp_Array2OfPnt(1, u_count, 1, v_count)
#     for i in range(u_count):
#         for j in range(v_count):
#             pt = control_points[i][j]
#             occ_control_points_array.SetValue(i + 1, j + 1, gp_Pnt(pt[0], pt[1], pt[2]))
#     u_knots = face['surface_definition']['u_knots']
#     v_knots = face['surface_definition']['v_knots']
#     u_multiplicities = face['surface_definition']['u_multiplicities']
#     v_multiplicities = face['surface_definition']['v_multiplicities']

#     def compress_knots_and_mults(knots, mults):
#         # If already compressed, do nothing
#         if len(knots) == len(mults):
#             return knots, mults
#         # Otherwise, compress
#         unique_knots = []
#         new_mults = []
#         prev = None
#         count = 0
#         for k in knots:
#             if prev is None or abs(k - prev) > 1e-10:
#                 if prev is not None:
#                     new_mults.append(count)
#                 unique_knots.append(k)
#                 prev = k
#                 count = 1
#             else:
#                 count += 1
#         new_mults.append(count)
#         return unique_knots, new_mults
    
#     # Create and fill knot and multiplicity arrays
#     u_knots_compressed, u_mults_compressed = compress_knots_and_mults(u_knots, u_multiplicities)
#     v_knots_compressed, v_mults_compressed = compress_knots_and_mults(v_knots, v_multiplicities)

#     occ_u_knots = TColStd_Array1OfReal(1, len(u_knots_compressed))
#     for i, val in enumerate(u_knots_compressed):
#         occ_u_knots.SetValue(i + 1, float(val))
#     occ_u_multiplicities = TColStd_Array1OfInteger(1, len(u_mults_compressed))
#     for i, val in enumerate(u_mults_compressed):
#         occ_u_multiplicities.SetValue(i + 1, int(val))

#     # Repeat for v
#     occ_v_knots = TColStd_Array1OfReal(1, len(v_knots_compressed))
#     for i, val in enumerate(v_knots_compressed):
#         occ_v_knots.SetValue(i + 1, float(val))
#     occ_v_multiplicities = TColStd_Array1OfInteger(1, len(v_mults_compressed))
#     for i, val in enumerate(v_mults_compressed):
#         occ_v_multiplicities.SetValue(i + 1, int(val))

#     # print(f"u_degree={u_degree}, v_degree={v_degree}")
#     # print(f"u_knots={u_knots}, v_knots={v_knots}")
#     # print(f"u_mults={u_multiplicities}, v_mults={v_multiplicities}")
#     # print(f"u_count={u_count}, v_count={v_count}")
#     # print(f"sum(u_mults)-u_degree-1={sum(u_multiplicities)-u_degree-1}, expected u_count={u_count}")
#     # print(f"sum(v_mults)-v_degree-1={sum(v_multiplicities)-v_degree-1}, expected v_count={v_count}")
#     occ_bspline_surface = Geom_BSplineSurface(occ_control_points_array, occ_u_knots, occ_v_knots, occ_u_multiplicities, occ_v_multiplicities, u_degree, v_degree, is_u_periodic, is_v_periodic)


def build_bspline_surface(data: dict) -> Geom_BSplineSurface:
    """
    Reconstructs a Geom_BSplineSurface from a dictionary of its properties.

    Args:
        data (dict): A dictionary containing the bspline surface data.

    Returns:
        Geom_BSplineSurface: The reconstructed pythonOCC BSpline surface object.
    """
    # 1. Unpack scalar data
    scalar_data = data["scalar"]
    u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v = map(int, scalar_data[:6])
    is_u_periodic = data["u_periodic"]
    is_v_periodic = data["v_periodic"]
    # 2. Extract knot and multiplicity lists from the scalar data
    u_knots_list = scalar_data[6 : 6 + num_knots_u]
    v_knots_list = scalar_data[6 + num_knots_u : 6 + num_knots_u + num_knots_v]
    u_mults_list = scalar_data[6 + num_knots_u + num_knots_v : 6 + num_knots_u + num_knots_v + num_knots_u]
    v_mults_list = scalar_data[6 + num_knots_u + num_knots_v + num_knots_u :]
    # print('u_degree: ', u_degree, 'v_degree: ', v_degree)
    # print('num_poles_u: ', num_poles_u, 'num_poles_v: ', num_poles_v)
    # print('num_knots_u: ', num_knots_u, 'num_knots_v: ', num_knots_v)
    # print('u_knots_list: ', len(u_knots_list), u_knots_list)
    # print('v_knots_list: ', len(v_knots_list), v_knots_list)
    # print('u_mults_list: ', sum(u_mults_list), u_mults_list)
    # print('v_mults_list: ', sum(v_mults_list), v_mults_list)
    # 3. Create and populate pythonOCC arrays for control points and weights
    # The constructor expects 1-based indexing for these arrays
    occ_control_points = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
    occ_weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
    
    poles_data = data["poles"]
    print('poles_data: ', np.array(poles_data).shape, np.array(poles_data))
    for i in range(num_poles_u):
        for j in range(num_poles_v):
            x, y, z, w = poles_data[i][j]
            # SetValue uses 1-based indexing
            occ_control_points.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))
            occ_weights.SetValue(i + 1, j + 1, w)

    # 4. Create and populate pythonOCC arrays for knots
    occ_u_knots = TColStd_Array1OfReal(1, num_knots_u)
    for i, knot in enumerate(u_knots_list):
        occ_u_knots.SetValue(i + 1, knot)

    occ_v_knots = TColStd_Array1OfReal(1, num_knots_v)
    for i, knot in enumerate(v_knots_list):
        occ_v_knots.SetValue(i + 1, knot)

    # 5. Create and populate pythonOCC arrays for multiplicities
    occ_u_multiplicities = TColStd_Array1OfInteger(1, num_knots_u)
    for i, mult in enumerate(u_mults_list):
        occ_u_multiplicities.SetValue(i + 1, int(mult))

    occ_v_multiplicities = TColStd_Array1OfInteger(1, num_knots_v)
    for i, mult in enumerate(v_mults_list):
        occ_v_multiplicities.SetValue(i + 1, int(mult))
        
    # 6. Recover periodicity information (see explanation below)
    # For a non-periodic curve: NbPoles = NbKnots - Degree - 1
    # For a periodic curve: NbPoles = NbKnots - 1


    # 7. Construct the BSpline Surface
    # The presence of weights indicates a rational surface (NURBS)
    occ_bspline_surface = Geom_BSplineSurface(
        occ_control_points,
        occ_weights,
        occ_u_knots,
        occ_v_knots,
        occ_u_multiplicities,
        occ_v_multiplicities,
        u_degree,
        v_degree,
        is_u_periodic,
        is_v_periodic
    )


    face_builder = BRepBuilderAPI_MakeFace(occ_bspline_surface, 1e-6)
    shape = face_builder.Face()
    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() # 确保执行网格化

    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")

    # --- 步骤 D: 提取网格数据 ---
    # 从修复并网格化后的面中提取数据
    vertices, faces = extract_mesh_from_face(shape)
    return shape, np.array(vertices), np.array(faces)

def build_second_order_surface(face):
    surface_type = face['type']
    parameter_range = face['uv']
    u_min, u_max, v_min, v_max = parameter_range
    position = np.array(face['location'])[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]

    # orientation = face['orientation']
    # 归一化参数范围
    if surface_type == 'cylinder':
        position = position + direction * v_min
        v_max = v_max - v_min
        v_min = 0

    print(position, direction, u_min, u_max, v_min, v_max)
    if surface_type == 'cylinder':
        radius = face['scalar'][0]
    elif surface_type == 'cone':
        radius = face['scalar'][1]
        semi_angle = face['scalar'][0]
    elif surface_type == 'torus':
        major_radius = face['scalar'][0]
        minor_radius = face['scalar'][1]
    elif surface_type == 'sphere':
        radius = face['scalar'][0]
    else:
        raise ValueError(f"Surface type {surface_type} not supported")
    
    occ_position = gp_Pnt(position[0], position[1], position[2])
    occ_direction = gp_Dir(direction[0], direction[1], direction[2])
    occ_XDirection = gp_Dir(XDirection[0], XDirection[1], XDirection[2])
    # occ_YDirection = gp_Dir(YDirection[0], YDirection[1], YDirection[2])
    # 2. 创建定义圆柱位置和方向的坐标系
    cylinder_ax3 = gp_Ax3(occ_position, occ_direction, occ_XDirection)
    if surface_type == 'cylinder':
        occ_surface = gp_Cylinder(cylinder_ax3, radius)
    elif surface_type == 'cone':
        occ_surface = gp_Cone(cylinder_ax3, semi_angle, radius)
    elif surface_type == 'torus':
        occ_surface = gp_Torus(cylinder_ax3, major_radius, minor_radius)
    elif surface_type == 'sphere':
        occ_surface = gp_Sphere(cylinder_ax3, radius)

    face_builder = BRepBuilderAPI_MakeFace(occ_surface, u_min, u_max, v_min, v_max)
    shape = face_builder.Face()

    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() 
    vertices, faces = extract_mesh_from_face(shape)
    vertices = np.array(vertices)
    faces = np.array(faces)
    print(vertices.max(axis=0), vertices.min(axis=0))
    return shape, vertices, faces


def build_plane_face(face):
    position = np.array(face['location'], dtype=np.float64)[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]


    u_min, u_max, v_min, v_max = face['uv']
    occ_position = gp_Pnt(position[0], position[1], position[2])
    occ_direction = gp_Dir(direction[0], direction[1], direction[2])
    occ_XDirection = gp_Dir(XDirection[0], XDirection[1], XDirection[2])
    occ_ax3 = gp_Ax3(occ_position, occ_direction, occ_XDirection)
    occ_plane = gp_Pln(occ_ax3)
    face_builder = BRepBuilderAPI_MakeFace(occ_plane, u_min, u_max, v_min, v_max)
    shape = face_builder.Face()
    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() # 确保执行网格化
    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")
    vertices, faces = extract_mesh_from_face(shape)
    return shape, np.array(vertices), np.array(faces)

def visualize_json_interset(cad_data):

    # vertex_positions = np.array(cad_data.get('vertices', []))
    # edges_list = cad_data.get('edges', [])
    # faces_list = cad_data.get('faces', [])
    # adjacency_matrix = build_adjacency_matrix(faces_list)
    faces_list = cad_data
    all_faces = {}

    ps.init()

    for face in faces_list:
        surface_type = face['type']
        surface_index = face['idx'][0]
        print(f"Processing face {surface_type} with type {surface_index}")
        if surface_type == 'plane':
            # continue
            occ_face, vertices, faces = build_plane_face(face)
        elif surface_type == 'cylinder' or surface_type == 'cone' or surface_type == 'torus' or surface_type == 'sphere':
            occ_face, vertices, faces = build_second_order_surface(face)
        # elif surface_type == "Cone":
        #     face_builder = build_cone_face(face)
        # elif surface_type == 'torus':
        #     face_builder = build_torus_face(face)
        elif surface_type == 'bspline_surface':
            # continue
            occ_face, vertices, faces = build_bspline_surface(face)
        else:
            continue
        ps.register_surface_mesh(f"{surface_index:03d}_{surface_type}", vertices, faces)
        all_faces[surface_index] = {
            'surface': occ_face,
            'vertices': vertices,
            'faces': faces,
            'surface_type': surface_type,
            'surface_index': surface_index,
        }

    # for face_index in all_faces.keys():
    #     face = all_faces[face_index]
    #     surface = face['surface']
    #     intersection_face
    #     adjacent_faces = adjacency_matrix[face_index]
    #     for adjacent_face_index in adjacent_faces:
    #         if adjacent_face_index in all_faces:
    #             adjacent_face = all_faces[adjacent_face_index]
    #             adjacent_surface = adjacent_face['surface']
    #             intersection_face = surface.Intersect(adjacent_surface)
            

    #         ps.register_surface_mesh(f"intersection_face_{face_index:03d}", face['vertices'], face['faces'])


    print(ps.get_bounding_box())
    ps.show()



if __name__ == "__main__":
    import sys
    # data_path = sys.argv[1]
    data_path = r'F:\WORK\CAD\data\examples\032_000.json'
    # with open('Solid_reconstruction_data.json', 'r') as f:
    with open(data_path, 'r') as f:
        cad_data = json.load(f)

    visualize_json_interset(cad_data)







