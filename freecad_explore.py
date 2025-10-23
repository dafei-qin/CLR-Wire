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
from OCC.Core.Geom2d import Geom2d_Curve
from OCC.Core.GC import GC_MakeArcOfCircle, GC_MakeSegment, GC_MakeArcOfEllipse, GC_MakeArcOfHyperbola, GC_MakeArcOfParabola
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace ,BRepBuilderAPI_DisconnectedWire, BRepBuilderAPI_EmptyWire, BRepBuilderAPI_NonManifoldWire, BRepBuilderAPI_WireDone, BRepBuilderAPI_Transform
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.Poly import Poly_Triangulation
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnCurve, GeomAPI_IntSS
from OCC.Core.GeomInt import GeomInt_IntSS 
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
#  OCC.Core.BRepAlgo import BRepAlgo_Section
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepLib import breplib
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeExtend import ShapeExtend_DONE1
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
from OCC.Core.TopAbs import TopAbs_Orientation
from OCC.Core.GeomAbs import GeomAbs_CurveType, GeomAbs_SurfaceType
import sys
from OCC.Core.TopOpeBRep import TopOpeBRep_FacesIntersector
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Section
from OCC.Core.ShapeConstruct import ShapeConstruct_ProjectCurveOnSurface

import FreeCAD as App

def json_to_face(entry):
    raise NotImplementedError("json_to_face is copied from CADDreamer and is not used yet")
    typ = entry["type"]
    loc = np.array(entry["location"][0])
    R = np.array(entry["direction"])  # 3 basis vectors (x,y,z)
    uv = entry.get("uv", None)
    u_min, u_max, v_min, v_max = uv
    ori = entry.get("orientation", "Forward")
    # helpers
    def orient(face):
        # return face if ori == "Forward" else face.Reversed()
        return face
    if typ == "plane":
        # normal = R[:,2], point = loc
        # normal = App.Vector(float(R[2,0]), float(R[2,1]), float(R[2,2]))
        normal = App.Vector(float(R[0,0]), float(R[0,1]), float(R[0,2]))
        point  = App.Vector(float(loc[0]), float(loc[1]), float(loc[2]))
        # make a bounded plane patch; size取自uv范围或给个默认
        size_u = u_max - u_min
        size_v = v_max - v_min
        plane = Part.makePlane(size_u, size_v, point, normal)
        return orient(plane.Faces[0])

    if typ == "cylinder":
        # axis dir = R[:,0] or R[:,2] 依你的约定；这里取R[:,0]
        axis_dir = App.Vector(float(R[0,0]), float(R[0,1]), float(R[0,2]))
        center   = App.Vector(float(loc[0]), float(loc[1]), float(loc[2]))
        radius   = float(entry["scalar"][0])

        center = center + axis_dir * v_min
        v_max = v_max - v_min
        v_min = 0
        height   = v_max - v_min  # 给一个包络高度，后续会被裁剪
        cyl_solid = Part.makeCylinder(radius, height, center, axis_dir)
        # 取圆柱侧面
        face = [f for f in cyl_solid.Faces if f.Surface.TypeId == "Part::GeomCylinder"][0]
        return orient(face)

    if typ == "sphere":
        center = App.Vector(float(loc[0]), float(loc[1]), float(loc[2]))
        radius = float(entry["scalar"][0])
        sph_solid = Part.makeSphere(radius, center)
        # 取一个代表面（整体是封闭体，也可直接参与布尔）
        face = [f for f in sph_solid.Faces if f.Surface.TypeId == "Part::GeomSphere"][0]
        return orient(face)

    if typ == "cone":
        # scalar: [apex_angle, height_or_radius?] 你的JSON里第一个是角度（弧度）
        angle = float(entry["scalar"][0])
        axis_dir = App.Vector(float(R[0,0]), float(R[0,1]), float(R[0,2]))
        apex     = App.Vector(float(loc[0]), float(loc[1]), float(loc[2]))
        h = 3.0
        r1, r2 = 0.0, abs(np.tan(angle) * h)
        cone_solid = Part.makeCone(r1, r2, h, apex, axis_dir)
        face = [f for f in cone_solid.Faces if f.Surface.TypeId == "Part::GeomCone"][0]
        return orient(face)

    if typ == "torus":
        axis_dir = App.Vector(float(R[0,0]), float(R[0,1]), float(R[0,2]))
        center   = App.Vector(float(loc[0]), float(loc[1]), float(loc[2]))
        rlarge   = float(entry["scalar"][0])
        rsmall   = float(entry["scalar"][1]) if len(entry["scalar"])>1 else 0.1
        torus = Part.makeTorus(rlarge, rsmall, center, axis_dir)
        face = [f for f in torus.Faces if f.Surface.TypeId == "Part::GeomToroid"][0]
        return orient(face)

    raise NotImplementedError(typ)



def build_plane_face(face):
    position = np.array(face['location'], dtype=np.float64)[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]
    orientation = face['orientation']

    u_min, u_max, v_min, v_max = face['uv']
    occ_position = gp_Pnt(position[0], position[1], position[2])
    occ_direction = gp_Dir(direction[0], direction[1], direction[2])
    occ_XDirection = gp_Dir(XDirection[0], XDirection[1], XDirection[2])
    occ_ax3 = gp_Ax3(occ_position, occ_direction, occ_XDirection)
    occ_plane = gp_Pln(occ_ax3)
    face_builder = BRepBuilderAPI_MakeFace(occ_plane, u_min, u_max, v_min, v_max)
    shape = face_builder.Face()

    if orientation == 'Reversed':
        shape = shape.Reversed()

    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() # 确保执行网格化
    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")
    vertices, faces = extract_mesh_from_face(shape)
    return shape, np.array(vertices), np.array(faces)



def build_second_order_surface(face):
    surface_type = face['type']
    parameter_range = face['uv']
    u_min, u_max, v_min, v_max = parameter_range
    position = np.array(face['location'])[0]
    direction = np.array(face['direction'], dtype=np.float64)[0]
    XDirection = np.array(face['direction'], dtype=np.float64)[1]

    orientation = face['orientation']
    # 归一化参数范围
    if surface_type == 'cylinder':
        position = position + direction * v_min
        v_max = v_max - v_min
        v_min = 0

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

    if orientation == 'Reversed':
        #shape = Face(shape).reversed_face().topods_shape()
        shape = shape.Reversed()


    mesher = BRepMesh_IncrementalMesh(shape, 0.1, True, 0.2)
    mesher.Perform() 
    vertices, faces = extract_mesh_from_face(shape)
    vertices = np.array(vertices)
    faces = np.array(faces)
    # print(vertices.max(axis=0), vertices.min(axis=0))
    return shape, vertices, faces


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
  
    # 3. Create and populate pythonOCC arrays for control points and weights
    # The constructor expects 1-based indexing for these arrays
    occ_control_points = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
    occ_weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
    
    poles_data = data["poles"]
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



def load_json_to_faces(cad_data):


    faces_list = cad_data
    all_faces = {}


    for face in faces_list:
        surface_type = face['type']
        surface_index = face['idx'][0]
        # print(f"Processing face {surface_type} with type {surface_index}")
        if surface_type == 'plane':
            # continue
            occ_face, vertices, faces = build_plane_face(face)
        elif surface_type == 'cylinder' or surface_type == 'cone' or surface_type == 'torus' or surface_type == 'sphere':
            occ_face, vertices, faces = build_second_order_surface(face)
        elif surface_type == 'bspline_surface':
            # continue
            occ_face, vertices, faces = build_bspline_surface(face)
        else:
            continue

        tol_fixer = ShapeFix_ShapeTolerance()
        tol_fixer.SetTolerance(occ_face, 1e-4)
        all_faces[surface_index] = {
            'surface': occ_face,
            'vertices': vertices,
            'faces': faces,
            'surface_type': surface_type,
            'surface_index': surface_index,
        }

    return all_faces



# entrys = json.load(open(r"F:\WORK\CAD\data\logan_jsons\0042\00420339\00420339_2323ceb4edb990b5e01e553b_step_000\index_000.json", "r"))
entrys = json.load(open(r"C:\drivers\CAD\CLR-Wire\assets\abnormal_cases\index_003.json", "r"))
# entrys = [e for e in entrys if e["type"] != "bspline_surface"]
all_faces = load_json_to_faces(entrys)




all_edges = {}

for idx_m in range(len(all_faces)):
    # if idx_m != 8:
    #     continue
    all_edges[idx_m] = []

    face = all_faces[idx_m]
    # geom_face = BRepAdaptor_Surface(face['surface'])
    print(type(face['surface']))
    freecad_face = Part.__fromPythonOCC__(face['surface'])
    freecad_face_show = Part.show(freecad_face, f"face_{idx_m:03d}")

    face_m = all_faces[idx_m]['surface']
    geom_face_m = BRep_Tool.Surface(face_m)

    projector = ShapeConstruct_ProjectCurveOnSurface()
    projector.Init(geom_face_m, 1e-4)

    for idx_n in range(len(all_faces)):
        print('\n', f'Intersecting face with index: {idx_m:03d}-{idx_n:03d}')
        face_n = all_faces[idx_n]['surface']

        geom_face_n = BRep_Tool.Surface(face_n)

# Section Intersector
        section = BRepAlgoAPI_Section(face_m, face_n)
        section.ComputePCurveOn1(True)
        section_shape = section.Shape()
        # print(f'Section shape: ', type(section_shape))
        # vertices, faces = extract_mesh_from_face(section_shape)


        edges = []
        exp = TopExp_Explorer(section_shape, TopAbs_EDGE)
        while exp.More():
            print(type(exp.Current()))
            # edge = TopoDS_Edge(exp.Current())
            edges.append(exp.Current())
            all_edges[idx_m].append(exp.Current())
            exp.Next()

        for edge_idx, edge in enumerate(edges):
            # geom_edge = BRep_Tool.Curve(edge)
            # edge_2d = Geom2d_Curve()
            # projector.Perform(geom_edge, 0, 1, edge_2d)
            freecad_edge = Part.__fromPythonOCC__(edge)
            freecad_edge_show = Part.show(freecad_edge, f"edge_{idx_m:03d}_{idx_n:03d}_edge_{edge_idx:03d}")

        # for idx_line, edge in enumerate(edges):
        #     a_curve = BRepAdaptor_Curve(edge)
        #     line_type = a_curve.GetType()
        #     # points, edges = sample_line(a_curve)
        #     # psEdge = ps.register_curve_network(f"{idx_m:03d}_{idx_n:03d}_line_{idx_line:03d}_{GeomAbs_CurveType(line_type).name}", points, edges, radius=0.001)
        #     # psEdge.add_to_group(new_group)
    



