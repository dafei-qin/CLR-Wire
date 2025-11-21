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
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Compound, TopoDS_Face
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_IntSS
from OCC.Core.GeomInt import GeomInt_IntSS 
from OCC.Core.BRep import BRep_Tool, BRep_Builder
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
# from OCC.Core.BRepAlgo import BRepAlgo_Section
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Section
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepLib import breplib
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge, ShapeFix_ShapeTolerance
from OCC.Core.ShapeExtend import ShapeExtend_DONE1
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_FreeBounds
from OCC.Core.TopAbs import TopAbs_Orientation
from OCC.Core.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
import sys
from OCC.Core.TopOpeBRep import TopOpeBRep_FacesIntersector
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

# from occwl.face import Face
import polyscope as ps
import polyscope.imgui as psim

# from freecad_visualize_json_pythonocc import _build_and_mesh_face_robustly_from_topods, extract_mesh_from_face
from itertools import combinations
from icecream import ic
import traceback

ic.disable()

def Compound(faces):
    # Convert list of faces to a compound
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)

    for face in faces:
        explorer = TopExp_Explorer(face, TopAbs_FACE)
        while explorer.More():
            face = TopoDS_Face(explorer.Current())
            builder.Add(compound, face)
            explorer.Next()

    return compound

def sample_line(line, num_points=32):
    start = line.FirstParameter()
    end = line.LastParameter()
    step = (end - start) / num_points
    points = []
    for i in range(num_points):
        param = start + i * step
        point = line.Value(param)
        points.append([point.X(), point.Y(), point.Z()])
    # isPeriodic = line.IsPeriodic()
    isPeriodic = line.IsClosed()
    if isPeriodic:
        edges = np.stack([np.arange(num_points), np.roll(np.arange(num_points), 1)], axis=1)
    else:
        edges = np.stack([np.arange(num_points)[:-1], np.arange(num_points)[1:]], axis=1)
    return np.array(points), edges

def extract_mesh_from_face(face):
    """
    从一个已经网格化的 TopoDS_Face 对象中提取出通用的顶点和面索引列表。

    :param face: 一个 TopoDS_Face 对象，必须先经过网格化处理。
    :return: 一个元组 (vertices, faces)，其中：
             - vertices: 一个包含 [x, y, z] 坐标的列表。
             - faces: 一个包含 [v1_idx, v2_idx, v3_idx] 的列表，索引是0-based。
    """
    # 获取面的三角化数据，如果面没有被网格化，则返回None
    location = TopLoc_Location()
    # print(type(face))
    triangulation = BRep_Tool.Triangulation(face, location)

    if triangulation is None:
        print("警告: 此面没有有效的三角化网格数据。")
        return [], []

    # --- 提取顶点并应用变换 ---
    # 获取从局部坐标到世界坐标的变换
    trsf = location.Transformation()
    
    # triangulation.Nodes() 返回局部坐标下的顶点数组
    num_nodes = triangulation.NbNodes()
    num_triangles = triangulation.NbTriangles()

    # nodes = TColgp_Array1OfPnt(1, num_nodes)
    # triangles = TColgp_Array1OfPnt(1, num_triangles)
    vertices = []
    for i in range(1, num_nodes + 1):
        # 获取局部坐标点
        local_pnt = triangulation.Node(i)
        # 应用变换得到世界坐标点
        global_pnt = local_pnt.Transformed(trsf)
        vertices.append([global_pnt.X(), global_pnt.Y(), global_pnt.Z()])

    # --- 提取三角面片索引 ---
    # triangulation.Triangles() 返回三角形数组
    # triangles = triangulation.Triangles()
    faces = []
    for i in range(1, num_triangles + 1):
        triangle = triangulation.Triangle(i)
        # 获取1-based的顶点索引
        v1_idx, v2_idx, v3_idx = triangle.Get()
        # 转换为0-based的索引并存储
        faces.append([v1_idx - 1, v2_idx - 1, v3_idx - 1])

    return vertices, faces

def build_plane_face(face, tol=1e-2, meshify=True):
    position = np.array(face['location'], dtype=np.float64)[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]
    YDirection = np.cross(direction, XDirection)
    orientation = face['orientation']

    u_min, u_max, v_min, v_max = face['uv']
    
    # Normalization
    centered = position + (u_max + u_min) / 2 * XDirection + (v_max + v_min) / 2 * YDirection
    u = np.array([u_min, u_max])
    u_new = u - (u_max + u_min) / 2
    v = np.array([v_min, v_max])
    v_new = v - (v_max + v_min) / 2
    u_min = u_new[0]
    u_max = u_new[1]
    v_min = v_new[0]
    v_max = v_new[1]
    position = centered
    
    occ_position = gp_Pnt(position[0], position[1], position[2])
    occ_direction = gp_Dir(direction[0], direction[1], direction[2])
    occ_XDirection = gp_Dir(XDirection[0], XDirection[1], XDirection[2])
    occ_ax3 = gp_Ax3(occ_position, occ_direction, occ_XDirection)
    occ_plane = gp_Pln(occ_ax3)
    face_builder = BRepBuilderAPI_MakeFace(occ_plane, u_min, u_max, v_min, v_max)
    shape = face_builder.Face()
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.SetTolerance(shape, tol)
    if orientation == 'Reversed':
        shape = shape.Reversed()

    # Build attr_str before checking meshify
    attr_str = f"{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}|{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}|{XDirection[0]:.2f},{XDirection[1]:.2f},{XDirection[2]:.2f}"
    attr_str += f"|{u_min:.2f},{u_max:.2f},{v_min:.2f},{v_max:.2f}"

    if not meshify:
        return shape, [], [], attr_str

    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() # 确保执行网格化
    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")
    vertices, faces = extract_mesh_from_face(shape)
    return shape, np.array(vertices), np.array(faces), attr_str



def build_second_order_surface(face, tol=1e-2, meshify=True):

    surface_type = face['type']
    parameter_range = face['uv']
    u_min, u_max, v_min, v_max = parameter_range
    position = np.array(face['location'])[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]
    YDirection = np.cross(direction, XDirection)
    orientation = face['orientation']
    # 归一化参数范围


    if surface_type == 'cylinder':
        radius = face['scalar'][0]
        position = position + direction * v_min
        v_max = v_max - v_min
        v_min = 0

        attr_str = f"{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}|{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}|{XDirection[0]:.2f},{XDirection[1]:.2f},{XDirection[2]:.2f}"
        attr_str += f"|{u_min/np.pi:.2f},{u_max/np.pi:.2f},{v_min:.2f},{v_max:.2f}|{radius:.2f}"
    elif surface_type == 'cone':
        radius = face['scalar'][1]
        semi_angle = face['scalar'][0]

        attr_str = f"{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}|{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}|{XDirection[0]:.2f},{XDirection[1]:.2f},{XDirection[2]:.2f}"
        attr_str += f"|{u_min/np.pi:.2f},{u_max/np.pi:.2f},{v_min:.2f},{v_max:.2f}|{semi_angle:.2f}|{radius:.2f}"
    

    elif surface_type == 'torus':
        major_radius = face['scalar'][0]
        minor_radius = face['scalar'][1]
        attr_str = f"{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}|{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}|{XDirection[0]:.2f},{XDirection[1]:.2f},{XDirection[2]:.2f}"
        attr_str += f"|{u_min/np.pi:.2f},{u_max/np.pi:.2f},{v_min/np.pi:.2f},{v_max/np.pi:.2f}|{major_radius:.2f}|{minor_radius:.2f}"
        
    elif surface_type == 'sphere':
        radius = face['scalar'][0]

        attr_str = f"{position[0]:.2f},{position[1]:.2f},{position[2]:.2f}|{direction[0]:.2f},{direction[1]:.2f},{direction[2]:.2f}|{XDirection[0]:.2f},{XDirection[1]:.2f},{XDirection[2]:.2f}"
        attr_str += f"|{u_min/np.pi:.2f},{u_max/np.pi:.2f},{v_min/np.pi:.2f},{v_max/np.pi:.2f}|{radius:.2f}"
    else:
        raise ValueError(f"Surface type {surface_type} not supported")

    occ_position = gp_Pnt(position[0], position[1], position[2])
    occ_direction = gp_Dir(direction[0], direction[1], direction[2])
    occ_XDirection = gp_Dir(XDirection[0], XDirection[1], XDirection[2])



    cylinder_ax3 = gp_Ax3(occ_position, occ_direction, occ_XDirection)
    try:
        if surface_type == 'cylinder':
            occ_surface = gp_Cylinder(cylinder_ax3, radius)
        elif surface_type == 'cone':
            occ_surface = gp_Cone(cylinder_ax3, semi_angle, radius)
        elif surface_type == 'torus':
            occ_surface = gp_Torus(cylinder_ax3, major_radius, minor_radius)
        elif surface_type == 'sphere':
            occ_surface = gp_Sphere(cylinder_ax3, radius)
    except Exception as e:
        print(face)
        raise e
        

    face_builder = BRepBuilderAPI_MakeFace(occ_surface, u_min, u_max, v_min, v_max)
    shape = face_builder.Face()
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.SetTolerance(shape, tol)
    if orientation == 'Reversed':

        shape = shape.Reversed()

    if not meshify:
        return shape, [], [], attr_str

    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() 
    vertices, faces = extract_mesh_from_face(shape)
    vertices = np.array(vertices)
    faces = np.array(faces)

    attr_str = attr_str.replace('|', '\n')
    return shape, vertices, faces, attr_str


def build_bspline_surface(data: dict, tol=1e-1, normalize_knots=False, normalize_surface=False) -> Geom_BSplineSurface:
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
  
    if normalize_knots:

        def mode_numpy(arr):
                vals, counts = np.unique(arr, return_counts=True)
                index = np.argmax(counts)
                return vals[index]
        u_knots_min = u_knots_list[0]
        u_knots_max = u_knots_list[-1]
        u_knots_list = [(i - u_knots_min) / (u_knots_max - u_knots_min) for i in u_knots_list]
        u_knots_list = np.array(u_knots_list)
        u_knots_diff = np.diff(u_knots_list)
        u_knits_diff_mode = mode_numpy(u_knots_diff)
        u_knots_list = u_knots_list / u_knits_diff_mode
        u_knots_list = u_knots_list / max(u_knots_list)


        v_knots_min = v_knots_list[0]
        v_knots_max = v_knots_list[-1]
        v_knots_list = [(i - v_knots_min) / (v_knots_max - v_knots_min) for i in v_knots_list]

        v_knots_list = np.array(v_knots_list)
        v_knots_diff = np.diff(v_knots_list)
        v_knits_diff_mode = mode_numpy(v_knots_diff)
        v_knots_list = v_knots_list / v_knits_diff_mode
        v_knots_list = v_knots_list / max(v_knots_list)
        
    # 3. Create and populate pythonOCC arrays for control points and weights
    # The constructor expects 1-based indexing for these arrays
    # print(f"u_knots_list: {u_knots_list}, v_knots_list: {v_knots_list}")
    occ_control_points = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
    occ_weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
    
    poles_data = data["poles"]
    # print(np.array(poles_data).shape)
    if normalize_surface:
        # Scale the surface to have maximum bbox = 1, keep the xyz ratio, 
        poles_data = np.array(poles_data)
        poles_list = poles_data.reshape(-1, 4)
        poles_data_min = poles_list.min(axis=0)
        poles_data_min[-1] = 0
        poles_data_max = poles_list.max(axis=0)
        poles_data_max[-1] = 1
        poles_max = max(poles_data_max[:3] - poles_data_min[:3])
        poles_list = (poles_list - poles_data_min) / poles_max
        # poles_list = (poles_list - poles_data_min) / (poles_data_max - poles_data_min)
        poles_data = poles_list.reshape(poles_data.shape[0], poles_data.shape[1], 4).tolist()
    # print('poles_data: ', np.array(poles_data).shape, np.array(poles_data))
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


    face_builder = BRepBuilderAPI_MakeFace(occ_bspline_surface, 1e-7)
    shape = face_builder.Face()
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.SetTolerance(shape, tol)
    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() # 确保执行网格化

    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")

    # --- 步骤 D: 提取网格数据 ---
    # 从修复并网格化后的面中提取数据
    vertices, faces = extract_mesh_from_face(shape)
    return shape, np.array(vertices), np.array(faces), ''

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


def visualize_json_interset(cad_data, plot=True, plot_gui=True,tol=1e-2, ps_header=''):

    faces_list = cad_data
    all_faces = {}
    if plot:
        ps.init()
    else:
        pass

    for face in faces_list:

        surface_type = face['type']
        surface_index = face['idx'][0]
        if len(face['uv']) > 0:
            for i in range(len(face['uv'])):
                if type(face['uv'][i]) == np.float32 or type(face['uv'][i]) == np.float64:
                    face['uv'][i] = face['uv'][i].item()

        if len(face['scalar']) > 0:
            for i in range(len(face['scalar'])):
                if type(face['scalar'][i]) == np.float32 or type(face['scalar'][i]) == np.float64:
                    face['scalar'][i] = face['scalar'][i].item()

        if surface_type != 'bspline_surface':
            ic(f"Processing face {surface_index} with type {surface_type}, uv: {face['uv']}, scalar: {face['scalar']}, D: {face['direction'][0]}, X: {face['direction'][1]}...")
        if surface_type == 'plane':
            # continue
            occ_face, vertices, faces, attr_str = build_plane_face(face, tol=tol)
        elif surface_type == 'cylinder' or surface_type == 'cone' or surface_type == 'torus' or surface_type == 'sphere':
            occ_face, vertices, faces, attr_str = build_second_order_surface(face, tol=tol)
        elif surface_type == 'bspline_surface':
            # continue
            try:
                occ_face, vertices, faces, attr_str = build_bspline_surface(face, tol=tol * 10, normalize_surface=False, normalize_knots=True)
            except Exception as exc:
                header = (ps_header or "").lower()
                if header.startswith("gt"):
                    origin = "Ground Truth"
                elif header.startswith("rec"):
                    origin = "Reconstruction"
                elif header.startswith("resampled"):
                    origin = "Resampled"
                else:
                    origin = "Unknown Origin"
                print(
                    f"[{origin}] Failed to build bspline surface idx {surface_index}: "
                    f"{exc.__class__.__name__}: {exc}"
                )
                traceback.print_exc()
                continue
        else:
            continue
        if plot:    
            # ps_handle = ps.register_surface_mesh(f"{ps_header}_{surface_index:03d}_{surface_type}_{attr_str}", vertices, faces, transparency=0.7)
            ps_handle = ps.register_surface_mesh(f"{ps_header}_{surface_index:03d}_{surface_type}", vertices, faces, transparency=0.7)
        else:
            ps_handle = None
        all_faces[surface_index] = {
            'surface': occ_face,
            'vertices': vertices,
            'faces': faces,
            'surface_type': surface_type,
            'surface_index': surface_index,
            'ps_handler': ps_handle
        }
        ic('done!')

    if plot_gui and plot:
        print(ps.get_bounding_box())
        ps.show()
    return all_faces


