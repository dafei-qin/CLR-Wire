# -*- coding: utf-8 -*-
import FreeCAD as App
import FreeCADGui as Gui
import Part
import time
import os
import json
from collections import deque
from datetime import datetime

App.Console.PrintMessage("="*50 + "\n")
App.Console.PrintMessage("开始执行“完整CAD拓扑数据序列化为JSON”脚本 (v3.0 - 新增椭圆支持)...\n")

# --- 1. 获取 FreeCAD 对象 ---
try:
    obj = Gui.Selection.getSelection()[0]
    shape = obj.Shape
except IndexError:
    App.Console.PrintError("错误：请先在模型树中选择一个对象。\n")
    raise SystemExit("请选择一个对象后重试")

App.Console.PrintMessage(f"模型加载成功: '{obj.Name}'.\n")

# --- 2. 核心数据提取 ---
App.Console.PrintMessage("步骤1/3: 正在提取顶点、边和面的完整定义...\n")

# 提取所有顶点
all_vertices = shape.Vertexes
vertex_map = {v.hashCode(): i for i, v in enumerate(all_vertices)}
json_vertices = [[v.X, v.Y, v.Z] for v in all_vertices]

# 提取所有边
json_edges = []
edge_map = {e.hashCode(): i for i, e in enumerate(shape.Edges)}
for i, edge in enumerate(shape.Edges):
    try:
        curve = edge.Curve
    except TypeError:
        print(f"Edge {i} has no curve")
        assert edge.Length < 1e-4, f"Edge {i} has valid length {edge.Length}, can't drop."
        print(f"Edge {i} has no curve and invalid length {edge.Length}, make it invalid.")
        edge_data = {
            "edge_index": i,
            "curve_type": "Invalid",
            "length": edge.Length,
            "vertices": [vertex_map.get(v.hashCode()) for v in edge.Vertexes],
            "first_parameter": edge.FirstParameter,
            "last_parameter": edge.LastParameter
        }
        json_edges.append(edge_data)
        continue
    edge_data = {
        "edge_index": i,
        "curve_type": type(curve).__name__,
        "length": edge.Length,
        "vertices": [vertex_map.get(v.hashCode()) for v in edge.Vertexes],
        # 【本次修正的核心】导出边的起始和终止参数
        "first_parameter": edge.FirstParameter,
        "last_parameter": edge.LastParameter
    }
    
    # 提取详细的几何定义
    if isinstance(curve, Part.Line) or isinstance(curve, Part.LineSegment):
        if len(edge.Vertexes) >= 2:
            edge_data["curve_definition"] = {
                "start": [edge.Vertexes[0].X, edge.Vertexes[0].Y, edge.Vertexes[0].Z],
                "end": [edge.Vertexes[1].X, edge.Vertexes[1].Y, edge.Vertexes[1].Z]
            }
        else:
            raise ValueError(f"Line segment has less than 2 vertices: {edge.Vertexes}")
    elif isinstance(curve, Part.Circle):
        edge_data["curve_definition"] = {"center": [curve.Center.x, curve.Center.y, curve.Center.z],
                                         "normal": [curve.Axis.x, curve.Axis.y, curve.Axis.z],
                                         "radius": curve.Radius}
    # --- START: 新增对 Part.Ellipse 的支持 ---
    elif isinstance(curve, Part.Ellipse):
        # We cannot get the axis directions directly from the Part.Ellipse object.
        # The reliable method is to convert the edge's geometry to a B-spline
        # and use its control points to determine the orientation.
        
        # This creates a B-spline representation of the curve
        bspline = curve.toBSpline()
        poles = bspline.getPoles()

        # For a full, untrimmed ellipse, the control points (poles) are positioned
        # in a way that allows us to find the axis directions. We find the vector
        # from the center to the first control point. This will be along an axis.
        first_pole = App.Vector(poles[0])
        center_vec = curve.Center
        
        # Vector from center to the first control point. This defines the direction
        # of one of the axes.
        axis_vec = first_pole - center_vec
        
        # Check if this vector's length corresponds to the Major or Minor radius.
        # This tells us if our vector is aligned with the Major or Minor axis.
        # We compare squared lengths to avoid unnecessary sqrt operations.
        major_axis_dir = App.Vector(0,0,0)
        
        if abs(axis_vec.Length**2 - curve.MajorRadius**2) < 1e-7: # Using a tolerance for float comparison
             major_axis_dir = axis_vec.normalize()
        else:
             # If the first pole wasn't on the major axis, we can find the major axis
             # direction using the cross product of the normal and this first axis vector.
             # The result will be a vector perpendicular to both, which is the other axis direction.
             normal_vec = curve.Axis
             second_axis_vec = normal_vec.cross(axis_vec)
             major_axis_dir = second_axis_vec.normalize()

        # The minor axis is perpendicular to both the normal and the major axis.
        minor_axis_dir = curve.Axis.cross(major_axis_dir).normalize()

        edge_data["curve_definition"] = {
            "center": [curve.Center.x, curve.Center.y, curve.Center.z],
            "major_radius": curve.MajorRadius,
            "minor_radius": curve.MinorRadius,
            "normal": [curve.Axis.x, curve.Axis.y, curve.Axis.z],
            # --- ADDED FOR COMPLETE DEFINITION (Corrected Method) ---
            "major_axis_direction": [major_axis_dir.x, major_axis_dir.y, major_axis_dir.z],
            "minor_axis_direction": [minor_axis_dir.x, minor_axis_dir.y, minor_axis_dir.z]
        }
    elif isinstance(curve, Part.Hyperbola):
        # 根据双曲线的数学定义，当参数u=0时，曲线上的点即为顶点。
        # 我们可以利用顶点和中心点来精确计算出主轴的方向。
        center_vec = curve.Center
        
        # 获取顶点 (vertex)
        vertex_point = curve.value(0)
        
        # 计算主轴方向向量 (从中心指向顶点)
        major_axis_dir = (vertex_point - center_vec).normalize()

        # 法线方向即为曲线的Axis属性
        normal_dir = curve.Axis.normalize()

        # 短轴方向可以通过主轴和法线的叉乘得到，确保右手坐标系
        minor_axis_dir = normal_dir.cross(major_axis_dir).normalize()

        edge_data["curve_definition"] = {
            "center": [center_vec.x, center_vec.y, center_vec.z],
            "major_radius": curve.MajorRadius,
            "minor_radius": curve.MinorRadius,
            "normal": [normal_dir.x, normal_dir.y, normal_dir.z],
            "major_axis_direction": [major_axis_dir.x, major_axis_dir.y, major_axis_dir.z]
            # "minor_axis_direction" is implicitly defined, but can be exported for convenience
            # "minor_axis_direction": [minor_axis_dir.x, minor_axis_dir.y, minor_axis_dir.z]
        }
    # --- END: 新增对 Part.Hyperbola 的支持 ---
    elif isinstance(curve, Part.Parabola):
        # 抛物线的定义由其顶点、焦距和坐标系决定。
        #
        # 【【【 核心修正 】】】
        # .FocalLength 属性不可靠，我们通过计算顶点和焦点间的距离来获得焦距。
        # 这是一个更健壮、基于几何定义的方法。
        
        vertex = curve.Location
        focus = curve.Focus
        
        # 1. 动态计算焦距
        focal_length_calculated = (focus - vertex).Length
        
        # 2. 计算对称轴方向 (从顶点指向焦点)
        axis_of_symmetry = (focus - vertex).normalize()
        
        # 3. 获取法线方向
        normal_dir = curve.Axis.normalize()

        edge_data["curve_definition"] = {
            "vertex": [vertex.x, vertex.y, vertex.z],
            "focal_length": focal_length_calculated, # <-- 使用我们计算出的值
            "normal": [normal_dir.x, normal_dir.y, normal_dir.z],
            "axis_of_symmetry": [axis_of_symmetry.x, axis_of_symmetry.y, axis_of_symmetry.z]
        }
    elif isinstance(curve, Part.BSplineCurve):
        knots = curve.getKnots()
        mults = curve.getMultiplicities()
        full_knot_vector = []
        for k, m in zip(knots, mults):
            full_knot_vector.extend([k] * m)
            
        edge_data["curve_definition"] = {
            "degree": curve.Degree, 
            "is_periodic": curve.isPeriodic(),
            "control_points": [[p.x, p.y, p.z] for p in curve.getPoles()],
            "knots": full_knot_vector,
            "multiplicities": list(mults)
        }
    json_edges.append(edge_data)

# 提取所有面
json_faces = []
for i, face in enumerate(shape.Faces):
    surface = face.Surface
    face_data = {
        "face_index": i,
        "surface_type": type(surface).__name__,
        "area": face.Area,
        "is_planar": isinstance(surface, Part.Plane),
        "orientation": face.Orientation,
        # 【新增】导出面的UV参数范围，为重建提供关键信息
        "parameter_range": face.ParameterRange,
        
        "wires": []
    }

    surface_def = {}
    if isinstance(surface, Part.Plane):
        surface_def = {"position": [surface.Position.x, surface.Position.y, surface.Position.z],
                       "normal": [surface.Axis.x, surface.Axis.y, surface.Axis.z]}
    elif isinstance(surface, Part.Cylinder):
        surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                       "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z],
                       "radius": surface.Radius}
    elif isinstance(surface, Part.Cone):
        surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                       "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z],
                       "radius": surface.Radius, 
                       "semi_angle": surface.SemiAngle}
    elif isinstance(surface, Part.Sphere):
        surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                       "radius": surface.Radius}
    elif isinstance(surface, Part.Toroid):
        surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                       "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z],
                       "major_radius": surface.MajorRadius,
                       "minor_radius": surface.MinorRadius}
    elif isinstance(surface, Part.BSplineSurface):
        # 【修正】为U和V方向都构建完整的节点向量
        u_knots, v_knots = surface.getUKnots(), surface.getVKnots()
        u_mults, v_mults = surface.getUMultiplicities(), surface.getVMultiplicities()
        
        full_u_knots = []
        for k, m in zip(u_knots, u_mults):
            full_u_knots.extend([k] * m)

        full_v_knots = []
        for k, m in zip(v_knots, v_mults):
            full_v_knots.extend([k] * m)

        surface_def = {
            "u_degree": surface.UDegree,
            "v_degree": surface.VDegree,
            "is_u_periodic": surface.isUPeriodic(),
            "is_v_periodic": surface.isVPeriodic(),
            "control_points": [[[p.x, p.y, p.z] for p in row] for row in surface.getPoles()],
            "u_knots": full_u_knots,
            "v_knots": full_v_knots,
            "u_multiplicities": list(u_mults),
            "v_multiplicities": list(v_mults)
        }
    
    face_data["surface_definition"] = surface_def
    
    for wire in face.Wires:
        is_outer_wire = wire.isSame(face.OuterWire)
        wire_data = {
            "is_outer": is_outer_wire,
            "ordered_edges": []
        }
        for edge in wire.OrderedEdges:
            edge_ref = {
                "edge_index": edge_map.get(edge.hashCode()),
                "orientation": edge.Orientation
            }
            wire_data["ordered_edges"].append(edge_ref)
        face_data["wires"].append(wire_data)
    json_faces.append(face_data)

App.Console.PrintMessage("数据提取完成。\n")

# --- 3. 组合并导出为JSON文件 ---
App.Console.PrintMessage("步骤2/3: 正在组合数据为最终JSON对象...\n")
reconstruction_data = {
    "metadata": {
        "source_object": obj.FullName,
        "export_time_utc": datetime.utcnow().isoformat() + "Z",
        "units": "mm", 
        "schema_version": "3.0-reconstruction-final" # 版本号可更新
    },
    "vertices": json_vertices,
    "edges": json_edges,
    "faces": json_faces
}
App.Console.PrintMessage("数据组合完成。\n")

App.Console.PrintMessage("步骤3/3: 正在将数据导出为可重建的JSON文件...\n")
try:
    doc_path = ""
    if hasattr(obj.Document, "FileName") and obj.Document.FileName:
        doc_path = os.path.dirname(obj.Document.FileName)
    if not doc_path:
        doc_path = os.path.expanduser("~")
        App.Console.PrintWarning(f"警告：当前FreeCAD文档尚未保存。JSON文件将被保存在您的用户主目录中: {doc_path}\n")
    
    json_file_path = os.path.join(doc_path, f"cylinder_cut_tube.json")

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(reconstruction_data, f, ensure_ascii=False, indent=2)

    App.Console.PrintMessage(f"成功！可重建的蓝图数据已保存到: {json_file_path}\n")

except Exception as e:
    App.Console.PrintError(f"错误：无法写入JSON文件: {e}\n")
finally:
    App.Console.PrintMessage("="*50 + "\n")