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
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.TopAbs import TopAbs_Orientation
import polyscope as ps
import polyscope.imgui as psim






def create_arc_from_params(center: gp_Pnt, normal: gp_Dir, radius: float, start_vertex: gp_Pnt, end_vertex: gp_Pnt, num_points: int = 32, return_curve=False):
    """
    根据中心点、法线、半径、起始顶点和结束顶点创建圆弧，并在圆弧上均匀采样点。

    Args:
        center (gp_Pnt): 圆弧的中心点。
        normal (gp_Dir): 圆弧所在平面的法线方向。
        radius (float): 圆弧的半径。
        start_vertex (gp_Pnt): 圆弧的起始顶点。
        end_vertex (gp_Pnt): 圆弧的结束顶点。
        num_points (int): 在圆弧上采样的点数量，默认为10。

    Returns:
        tuple: (TopoDS_Edge, list) 圆弧对象和采样点列表，每个点为(x, y, z)元组。
    """
    # 简化的圆弧创建方法：直接使用三点法或圆心+两点法

    # 方法1：使用圆心和两个端点创建圆弧
    center = gp_Pnt(center[0], center[1], center[2])
    normal = gp_Dir(normal[0], normal[1], normal[2])
    
    axis = gp_Ax2(center, normal)
    circle = gp_Circ(axis, radius)
    
    # 检查是否为闭合圆（完整圆）
    if np.allclose(start_vertex, end_vertex):
        # 处理完整圆的情况
        geom_arc = Geom_Circle(axis, radius)
        
        # 创建完整圆的拓扑边
        edge_maker = BRepBuilderAPI_MakeEdge(geom_arc)
        if not edge_maker.IsDone():
            raise RuntimeError("Failed to make edge from circle.")
            
        arc_edge = edge_maker.Edge()
        
        # 在完整圆上均匀采样点
        sampled_points = sample_points_on_curve(geom_arc, num_points, is_closed=True)
        is_closed = True
        
    else:
        # 处理圆弧的情况
        start_vertex = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
        end_vertex = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])
        arc_maker = GC_MakeArcOfCircle(circle, start_vertex, end_vertex, True)
        
        if not arc_maker.IsDone():
            raise RuntimeError("Failed to create arc of circle.")
            
        geom_arc = arc_maker.Value()
        
        # 创建拓扑边
        edge_maker = BRepBuilderAPI_MakeEdge(geom_arc)
        if not edge_maker.IsDone():
            raise RuntimeError("Failed to make edge from arc.")
            
        arc_edge = edge_maker.Edge()
        
        # 在圆弧上均匀采样点
        sampled_points = sample_points_on_curve(geom_arc, num_points, is_closed=False)
        is_closed = False
    if return_curve:
        return arc_edge, sampled_points, is_closed, geom_arc
    return arc_edge, sampled_points, is_closed


def create_line_from_points(start_vertex, end_vertex, num_points: int = 32, return_curve=False):
    """
    根据起始点和结束点创建直线段，并在直线上均匀采样点。

    Args:
        start_vertex: 起始顶点坐标 [x, y, z]
        end_vertex: 结束顶点坐标 [x, y, z]
        num_points (int): 在直线上采样的点数量，默认为32。

    Returns:
        tuple: (TopoDS_Edge, numpy.array, bool) 直线边对象、采样点数组和闭合标志（始终为False）。
    """
    # 转换输入参数为OpenCASCADE对象
    start_pnt = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
    end_pnt = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])
    
    # 使用GC_MakeSegment创建线段
    segment_maker = GC_MakeSegment(start_pnt, end_pnt)
    
    if not segment_maker.IsDone():
        raise RuntimeError("Failed to create line segment.")
        
    geom_line = segment_maker.Value()
    
    # 创建拓扑边
    edge_maker = BRepBuilderAPI_MakeEdge(geom_line)
    if not edge_maker.IsDone():
        raise RuntimeError("Failed to make edge from line segment.")
        
    line_edge = edge_maker.Edge()
    
    # 在直线上均匀采样点
    sampled_points = sample_points_on_curve(geom_line, num_points, is_closed=False)
    if return_curve:
        return line_edge, sampled_points, False, geom_line
    return line_edge, sampled_points, False



def create_ellipse_from_params(center: list, x_direction: list, y_direction: list, normal: list, 
                               major_radius: float, minor_radius: float, 
                               start_vertex: list, end_vertex: list, num_points: int = 64, return_curve=False):
    """
    根据椭圆的几何定义和起始、结束顶点创建椭圆弧，并在其上均匀采样点。

    Args:
        center (list): 椭圆的中心点 [x, y, z]。
        x_direction (list): 椭圆主轴的方向向量 [dx, dy, dz]。
        y_direction (list): 椭圆次轴的方向向量 [dx, dy, dz]。
        normal (list): 椭圆所在平面的法线方向 [nx, ny, nz]。
        major_radius (float): 椭圆的长半轴长度。
        minor_radius (float): 椭圆的短半轴长度。
        start_vertex (list): 椭圆弧的起始顶点 [x, y, z]。
        end_vertex (list): 椭圆弧的结束顶点 [x, y, z]。
        num_points (int): 在椭圆弧上采样的点数量，默认为64。

    Returns:
        tuple: (TopoDS_Edge, numpy.array, bool) 椭圆弧对象、采样点数组和闭合标志。
    """
    # 转换输入参数为OpenCASCADE对象
    center_pnt = gp_Pnt(center[0], center[1], center[2])
    x_dir = gp_Dir(x_direction[0], x_direction[1], x_direction[2])
    y_dir = gp_Dir(y_direction[0], y_direction[1], y_direction[2])
    normal_dir = gp_Dir(normal[0], normal[1], normal[2])

    
   
    ellipse_axis = gp_Ax2(center_pnt, normal_dir, x_dir)
    
    is_closed_curve = False
    if np.allclose(start_vertex, end_vertex):
        # 视为完整椭圆
        print(start_vertex, end_vertex)
        geom_ellipse = Geom_Ellipse(ellipse_axis, major_radius, minor_radius)
        is_closed_curve = True
    else:
        ellipse = Geom_Ellipse(ellipse_axis, major_radius, minor_radius)
        # 椭圆弧的情况
        start_pnt = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
        end_pnt = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])
        # print(start_vertex, end_vertex, np.array((center_pnt.X(), center_pnt.Y(), center_pnt.Z())), np.array((normal_dir.X(), normal_dir.Y(), normal_dir.Z())), np.array((x_dir.X(), x_dir.Y(), x_dir.Z())))
        # 使用投影找到起始点和结束点在椭圆上的参数
        start_projector = GeomAPI_ProjectPointOnCurve(start_pnt, ellipse)
        end_projector = GeomAPI_ProjectPointOnCurve(end_pnt, ellipse)
        geom_ellipse = gp_Elips(ellipse_axis, major_radius, minor_radius)
        if start_projector.NbPoints() == 0 or end_projector.NbPoints() == 0:
            raise RuntimeError("Failed to project start/end points onto ellipse. Check if points are on the ellipse.")
        
        start_param = start_projector.LowerDistanceParameter()
        end_param = end_projector.LowerDistanceParameter()

     
        # print(start_param, end_param)
        arc_maker = GC_MakeArcOfEllipse(geom_ellipse, start_param, end_param, True) # True for counter-clockwise
        
        if not arc_maker.IsDone():
            raise RuntimeError("Failed to create arc of ellipse.")
            
        geom_ellipse = arc_maker.Value()

    # 创建拓扑边
    edge_maker = BRepBuilderAPI_MakeEdge(geom_ellipse)
    if not edge_maker.IsDone():
        raise RuntimeError("Failed to make edge from ellipse/arc.")
        
    ellipse_edge = edge_maker.Edge()
    
    # 在椭圆弧上均匀采样点
    sampled_points = sample_points_on_curve(geom_ellipse, num_points, is_closed=is_closed_curve)
    
    if return_curve:
        return ellipse_edge, sampled_points, is_closed_curve, geom_ellipse
    return ellipse_edge, sampled_points, is_closed_curve


def create_hyperbola_from_params(center: list, x_direction: list, normal: list, major_radius: float, minor_radius: float, start_vertex: list, end_vertex: list, num_points: int = 64, return_curve=False):
    """
    根据双曲线的几何定义和起始、结束顶点创建双曲弧，并在其上均匀采样点。
    """

    center_pnt = gp_Pnt(center[0], center[1], center[2])
    x_dir = gp_Dir(x_direction[0], x_direction[1], x_direction[2])

    normal_dir = gp_Dir(normal[0], normal[1], normal[2])
    hyperbola_axis = gp_Ax2(center_pnt, normal_dir, x_dir)

    is_closed_curve = False
    if np.allclose(start_vertex, end_vertex):
        # 视为完整椭圆
        print(start_vertex, end_vertex)
        geom_hypoerbola = Geom_Hyperbola(hyperbola_axis, major_radius, minor_radius)
        is_closed_curve = True
    else:
        hyperbola = Geom_Hyperbola(hyperbola_axis, major_radius, minor_radius)
        # 椭圆弧的情况
        start_pnt = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
        end_pnt = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])
        # print(start_vertex, end_vertex, np.array((center_pnt.X(), center_pnt.Y(), center_pnt.Z())), np.array((normal_dir.X(), normal_dir.Y(), normal_dir.Z())), np.array((x_dir.X(), x_dir.Y(), x_dir.Z())))
        # 使用投影找到起始点和结束点在椭圆上的参数
        start_projector = GeomAPI_ProjectPointOnCurve(start_pnt, hyperbola)
        end_projector = GeomAPI_ProjectPointOnCurve(end_pnt, hyperbola)
        geom_hyperbola = gp_Hypr(hyperbola_axis, major_radius, minor_radius)

        if start_projector.NbPoints() == 0 or end_projector.NbPoints() == 0:
            raise RuntimeError("Failed to project start/end points onto ellipse. Check if points are on the ellipse.")
        
        start_param = start_projector.LowerDistanceParameter()
        end_param = end_projector.LowerDistanceParameter()

     
        # print(start_param, end_param)
        arc_maker = GC_MakeArcOfHyperbola(geom_hyperbola, start_param, end_param, True) # True for counter-clockwise
        
        if not arc_maker.IsDone():
            raise RuntimeError("Failed to create arc of ellipse.")
            
        geom_hyperbola = arc_maker.Value()

    # 创建拓扑边
    edge_maker = BRepBuilderAPI_MakeEdge(geom_hyperbola)
    if not edge_maker.IsDone():
        raise RuntimeError("Failed to make edge from hyperbola/arc.")
        
    hyperbola_edge = edge_maker.Edge()
    
    # 在椭圆弧上均匀采样点
    sampled_points = sample_points_on_curve(geom_hyperbola, num_points, is_closed=is_closed_curve)
    if return_curve:
        return hyperbola_edge, sampled_points, is_closed_curve, geom_hyperbola
    return hyperbola_edge, sampled_points, is_closed_curve

def create_parabola_from_params(vertex: list, axis_of_symmetry: list, normal: list, focal_length: float, start_vertex: list, end_vertex: list, num_points: int = 64, return_curve=False):
    vertex_pnt = gp_Pnt(vertex[0], vertex[1], vertex[2])
    x_dir = gp_Dir(axis_of_symmetry[0], axis_of_symmetry[1], axis_of_symmetry[2])
    normal_dir = gp_Dir(normal[0], normal[1], normal[2])

    parabola_axis = gp_Ax2(vertex_pnt, normal_dir, x_dir)
    
    is_closed_curve = False
    if np.allclose(start_vertex, end_vertex):
        # 视为完整椭圆
        print(start_vertex, end_vertex)
        geom_parabola = Geom_Parabola(parabola_axis, focal_length)
        is_closed_curve = True
    else:
        parabola = Geom_Parabola(parabola_axis, focal_length)
        # 椭圆弧的情况
        start_pnt = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
        end_pnt = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])

        # 使用投影找到起始点和结束点在椭圆上的参数
        start_projector = GeomAPI_ProjectPointOnCurve(start_pnt, parabola)
        end_projector = GeomAPI_ProjectPointOnCurve(end_pnt, parabola)
        geom_parabola = gp_Parab(parabola_axis, focal_length)

        if start_projector.NbPoints() == 0 or end_projector.NbPoints() == 0:
            raise RuntimeError("Failed to project start/end points onto ellipse. Check if points are on the ellipse.")
        
        start_param = start_projector.LowerDistanceParameter()
        end_param = end_projector.LowerDistanceParameter()

     
        # print(start_param, end_param)
        arc_maker = GC_MakeArcOfParabola(geom_parabola, start_param, end_param, True) # True for counter-clockwise
        
        if not arc_maker.IsDone():
            raise RuntimeError("Failed to create arc of ellipse.")
            
        geom_parabola = arc_maker.Value()


    edge_maker = BRepBuilderAPI_MakeEdge(geom_parabola)
    if not edge_maker.IsDone():
        raise RuntimeError("Failed to make edge from parabola/arc.")
        
    parabola_edge = edge_maker.Edge()
    
    # 在椭圆弧上均匀采样点
    sampled_points = sample_points_on_curve(geom_parabola, num_points, is_closed=is_closed_curve)
    
    if return_curve:
        return parabola_edge, sampled_points, is_closed_curve, geom_parabola
    return parabola_edge, sampled_points, is_closed_curve


def create_bspline_from_params(degree, is_periodic, is_closed, control_points, knots, multiplicities, start_vertex, end_vertex, num_points: int = 128, return_curve=False):
    """
    根据B样条曲线参数创建B样条曲线，并在曲线上均匀采样点。

    Args:
        degree (int): B样条曲线的度数
        is_periodic (bool): 是否为周期性B样条曲线
        control_points (list): 控制点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        knots (list): 节点向量
        multiplicities (list): 节点重数
        start_vertex: 起始顶点坐标 [x, y, z]
        end_vertex: 结束顶点坐标 [x, y, z]
        num_points (int): 在曲线上采样的点数量，默认为32。

    Returns:
        tuple: (TopoDS_Edge, numpy.array, bool) B样条曲线边对象、采样点数组和闭合标志。
    """
    # 创建控制点数组
    control_points = np.array(control_points)
    num_control_points = len(control_points)
    
    # 创建OpenCASCADE控制点数组
    control_points_array = TColgp_Array1OfPnt(1, num_control_points)
    for i, cp in enumerate(control_points):
        control_points_array.SetValue(i + 1, gp_Pnt(cp[0], cp[1], cp[2]))
    
    # 处理节点向量：JSON中可能包含重复的节点值，需要提取唯一值
    # 如果knots长度与multiplicities长度不匹配，说明knots已经按重数展开了
    if len(knots) != len(multiplicities):
        # 从重复的knots中提取唯一值
        unique_knots = []
        current_knot = None
        for knot in knots:
            if current_knot is None or abs(knot - current_knot) > 1e-10:
                unique_knots.append(knot)
                current_knot = knot
        actual_knots = unique_knots
        actual_multiplicities = multiplicities
    else:
        # knots已经是唯一值
        actual_knots = knots
        actual_multiplicities = multiplicities
    
    # print(f"Unique knots: {len(actual_knots)}, Multiplicities: {len(actual_multiplicities)}")
    # print(f"Knots: {actual_knots}")
    # print(f"Multiplicities: {actual_multiplicities}")
    
    # 创建节点向量数组
    knots_array = TColStd_Array1OfReal(1, len(actual_knots))
    for i, knot in enumerate(actual_knots):
        knots_array.SetValue(i + 1, float(knot))
    
    # 创建重数数组
    multiplicities_array = TColStd_Array1OfInteger(1, len(actual_multiplicities))
    for i, mult in enumerate(actual_multiplicities):
        multiplicities_array.SetValue(i + 1, int(mult))
    
    # 创建B样条曲线
    bspline_curve = Geom_BSplineCurve(
        control_points_array,
        knots_array,
        multiplicities_array,
        degree,
        is_periodic
    )
    
    # 创建拓扑边
    edge_maker = BRepBuilderAPI_MakeEdge(bspline_curve)
    if not edge_maker.IsDone():
        raise RuntimeError("Failed to make edge from B-spline curve.")
        
    bspline_edge = edge_maker.Edge()
    
    # 检查是否为闭合曲线
    # is_closed = is_periodic or np.allclose(start_vertex, end_vertex)
    
    # 在B样条曲线上根据起终点进行采样
    sampled_points = sample_bspline_between_points(bspline_curve, start_vertex, end_vertex, num_points, is_closed, is_periodic)
    if return_curve:
        return bspline_edge, sampled_points, is_closed, bspline_curve
    return bspline_edge, sampled_points, is_closed
        



def sample_points_on_curve(geom_curve, num_points: int, is_closed: bool = False):
    """
    在几何曲线上均匀采样指定数量的点。
    
    Args:
        geom_curve: OpenCASCADE几何曲线对象（圆弧或圆）
        num_points (int): 采样点数量
        is_closed (bool): 是否为闭合曲线（完整圆）
        
    Returns:
        list: 采样点列表，每个点为(x, y, z)元组
    """
    if num_points < (3 if is_closed else 2):
        raise ValueError(f"采样点数量必须至少为{3 if is_closed else 2}")
    
    # 获取曲线的参数范围
    first_param = geom_curve.FirstParameter()
    last_param = geom_curve.LastParameter()
    
    sampled_points = []
    
    if is_closed:
        # 对于闭合曲线，避免重复起点和终点
        param_step = (last_param - first_param) / num_points
        for i in range(num_points):
            param = first_param + i * param_step
            point = geom_curve.Value(param)
            sampled_points.append((point.X(), point.Y(), point.Z()))
    else:
        # 对于开放曲线（圆弧），包含起点和终点
        param_step = (last_param - first_param) / (num_points - 1)
        for i in range(num_points):
            param = first_param + i * param_step
            point = geom_curve.Value(param)
            sampled_points.append((point.X(), point.Y(), point.Z()))
    
    return np.array(sampled_points)


def sample_bspline_between_points(bspline_curve, start_vertex, end_vertex, num_points: int, is_closed: bool = False, is_periodic: bool = False):
    """
    在B样条曲线上根据起点和终点确定参数范围进行采样。
    使用OpenCASCADE的GeomAPI_ProjectPointOnCurve进行精确的参数查找。
    
    Args:
        bspline_curve: OpenCASCADE B样条几何曲线对象
        start_vertex: 起始点坐标 [x, y, z]
        end_vertex: 结束点坐标 [x, y, z]
        num_points (int): 采样点数量
        is_closed (bool): 是否为闭合曲线
        is_periodic (bool): 是否为周期曲线
    Returns:
        numpy.array: 采样点数组，每个点为(x, y, z)
    """
    if num_points < (3 if is_closed else 2):
        raise ValueError(f"采样点数量必须至少为{3 if is_closed else 2}")
    
    # 使用OpenCASCADE的投影功能找到参数值
    start_pnt = gp_Pnt(start_vertex[0], start_vertex[1], start_vertex[2])
    end_pnt = gp_Pnt(end_vertex[0], end_vertex[1], end_vertex[2])
    
    # 创建投影器并找到参数
    start_projector = GeomAPI_ProjectPointOnCurve(start_pnt, bspline_curve)
    end_projector = GeomAPI_ProjectPointOnCurve(end_pnt, bspline_curve)
    
    if start_projector.NbPoints() == 0 or end_projector.NbPoints() == 0:
        print("Warning: Cannot project points onto B-spline curve, using full curve range")
        # 回退到完整曲线采样
        return sample_points_on_curve(bspline_curve, num_points, is_closed)
    
    # 获取最近投影点的参数值
    start_param = start_projector.LowerDistanceParameter()
    end_param = end_projector.LowerDistanceParameter()
    print(start_param, end_param)
    # 确保参数顺序正确
    if start_param > end_param and not is_closed:
        if is_periodic and np.allclose(end_param, 0):
            end_param = 1 - end_param
        else:
            start_param, end_param = end_param, start_param
    
    sampled_points = []
    
    if is_closed and abs(start_param - end_param) < 1e-6:
        # 完整的闭合曲线，采样整个参数范围
        curve_first_param = bspline_curve.FirstParameter()
        curve_last_param = bspline_curve.LastParameter()
        param_step = (curve_last_param - curve_first_param) / num_points
        for i in range(num_points):
            param = curve_first_param + i * param_step
            point = bspline_curve.Value(param)
            sampled_points.append((point.X(), point.Y(), point.Z()))
    else:
        # 采样起终点之间的曲线段
        if is_closed:
            param_step = (end_param - start_param) / num_points
            for i in range(num_points):
                param = start_param + i * param_step
                point = bspline_curve.Value(param)
                sampled_points.append((point.X(), point.Y(), point.Z()))
        else:
            param_step = (end_param - start_param) / (num_points - 1)
            for i in range(num_points):
                param = start_param + i * param_step
                point = bspline_curve.Value(param)
                sampled_points.append((point.X(), point.Y(), point.Z()))
    
    return np.array(sampled_points)

def generate_edges_from_points(points, is_closed=False):
    edges = np.arange(len(points))
        
    edges = np.stack([edges[:-1], edges[1:]],axis=1)
    if is_closed:
        edges = np.concatenate([edges, np.array([len(points) - 1, 0])[np.newaxis, :]], axis=0)
    return edges


# ===============================================
# Wire reconstruction
# ===============================================

from collections import defaultdict
def split_psudowire_to_two_cycles(wire):
    # This function maps the outer wire of typical cylinder, cone, etc surfaces to two cycles of the bottom and the top
    # For using in BRepOffsetAPI_ThruSections

    # 1. Find the repeated edges, the repeated one means it is the edge connecting the bottom and the top
    edge_occurance = defaultdict(int)
    for edge in wire['ordered_edges']:
        edge_occurance[edge['edge_index']] += 1

    edge_repeated = -1
    for edge, occurance in edge_occurance.items():
        if occurance  == 2:
            edge_repeated = edge
            break
    wire_drop_repeated = []
    for edge in wire['ordered_edges']:
        if not edge['edge_index'] == edge_repeated:
            wire_drop_repeated.append(edge)
            
    wire_1_index = ()

    wire_1_vertices = set(*wire_drop_repeated[0]['vertices'])
    wire_1_index.add(0)
    for wire_idx, edge in enumerate(wire_drop_repeated):
        if len(edge['vertices']) == 2:
            if edge['vertices'][0] in wire_1_vertices:
                wire_1_index.add()
                wire_1_vertices.add(edge['vertices'][1])
            elif edge['vertices'][1] in wire_1_vertices:
                wire_1_index.add(wire_idx)
                wire_1_vertices.add(edge['vertices'][0])
    
    wire_1 = []
    wire_2 = []
    for edge_idx, edge in enumerate(wire_drop_repeated):
        if edge_idx in wire_1_index:
            wire_1.append(edge)
        else:
            wire_2.append(edge)
            
    



    
    
    

    



from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.GCPnts import GCPnts_UniformAbscissa
from OCC.Core.TopoDS import TopoDS_Wire

def sample_wire_by_distance(wire: TopoDS_Wire, distance: float) -> list[gp_Pnt]:
    """
    [修正版] 通过在Wire上按大致固定的空间距离来采样点。
    """
    if distance <= 0:
        raise ValueError("采样距离必须是正数。")

    sampled_points = []
    explorer = BRepTools_WireExplorer(wire)

    is_first_edge = True
    while explorer.More():
        edge = explorer.Current()
        adaptor = BRepAdaptor_Curve(edge)
        sampler = GCPnts_UniformAbscissa(adaptor, distance)
        
        if sampler.IsDone():
            num_points_on_edge = sampler.NbPoints()
            
            # 确定循环的起始索引，以避免边与边之间的重复点
            start_index = 1
            if is_first_edge:
                is_first_edge = False
            elif num_points_on_edge > 1:
                start_index = 2

            for i in range(start_index, num_points_on_edge + 1):
                # --- 这是修正的核心逻辑 ---
                # 1. 从 sampler 获取第 i 个点的参数(u值)
                parameter = sampler.Parameter(i)
                # 2. 将参数传递给 adaptor 来计算实际的3D点
                point = adaptor.Value(parameter)
                # --- 修正结束 ---
                sampled_points.append(point)

        explorer.Next()
        
    if wire.Closed() and len(sampled_points) > 1:
        if sampled_points[0].Distance(sampled_points[-1]) < 1e-7: # 使用稍大的容差
            sampled_points.pop()

    return sampled_points

def get_wire_vertices_and_lines(
    wire: TopoDS_Wire, 
    sampling_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    [修正版] 对一个 TopoDS_Wire 进行采样，并返回NumPy格式的顶点和线段连接数组。
    """
    # 1. 使用修正后的采样函数
    sampled_points = sample_wire_by_distance(wire, sampling_distance)
    
    num_vertices = len(sampled_points)
    
    if num_vertices < 2:
        return np.empty((0, 3), dtype=float), np.empty((0, 2), dtype=int)

    # 2. 创建 vertices 数组 (N, 3)
    vertices = np.array([pt.Coord() for pt in sampled_points], dtype=np.float64)

    # 3. 创建 lines 数组 (M, 2)
    indices = np.arange(num_vertices)
    
    if wire.Closed():
        lines = np.vstack([indices, np.roll(indices, -1)]).T
    else:
        lines = np.vstack([indices[:-1], indices[1:]]).T

    return vertices, lines

# ===============================================
# Surface reconstruction
# ===============================================


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

def _build_and_mesh_face_robustly(face_builder: BRepBuilderAPI_MakeFace, linear_deflection: float = 0.1, angular_deflection: float = 0.5):
    """
    一个健壮的函数，用于从BRepBuilderAPI_MakeFace构建面，
    并对其进行修复和高质量的网格化。

    :param face_builder: 已经添加了所有内外边界的 BRepBuilderAPI_MakeFace 对象。
    :param linear_deflection: 网格的线性挠度。
    :param angular_deflection: 网格的角度挠度（弧度）。
    :return: 一个元组 (final_face, vertices, faces)。
    """
    if not face_builder.IsDone():
        raise RuntimeError("Face Builder failed before healing and meshing.")
        
    # --- 步骤 A: 构建基础面 ---
    face = face_builder.Face()

    # --- 步骤 B: “几何修复”过程 ---
    # 1. 强制更新公差，确保所有子形状的公差一致
    breplib.UpdateTolerances(face, True)
    
    # 2. 这是一个强大的修复工具，可以解决多种拓扑和几何问题
    print(face_builder.Error())
    fixer = ShapeFix_Shape(face)
    fixer.Perform()
    healed_face = fixer.Shape()
    # healed_face = face

    # --- 步骤 C: 精细化网格 ---
    # 使用修复后的面进行网格化，并同时指定线性和角度挠度
    # 第二个布尔参数 True 表示启用角度挠度控制
    mesher = BRepMesh_IncrementalMesh(healed_face, linear_deflection, True, angular_deflection)
    mesher.Perform() # 确保执行网格化

    if not mesher.IsDone():
         print("警告: BRepMesh_IncrementalMesh 执行后报告未完成，网格可能无效。")

    # --- 步骤 D: 提取网格数据 ---
    # 从修复并网格化后的面中提取数据
    vertices, faces = extract_mesh_from_face(healed_face)
    
    return healed_face, vertices, faces


def create_planar_face_mesh(face_index, pos, norm, wires, all_edges, all_curves):
    """
    根据从JSON加载的面数据（特别是平面）及其边界线，重建一个带孔的拓扑面并进行网格化。

    :param face_data: 包含单个面定义信息的字典。
    :param all_edges: 一个字典，其键为 edge_index，值为预先创建好的 TopoDS_Edge 对象。
    :return: 一个可供显示的 TopoDS_Face 对象，其内部已包含网格化数据。
    """

    
    gp_plane = gp_Pln(gp_Pnt(pos[0], pos[1], pos[2]), gp_Dir(norm[0], norm[1], norm[2]))
    geom_plane = Geom_Plane(gp_plane)

    # --- 2. 重建所有的线框 (Wires) ---
    outer_wire = None
    inner_wires = []
    outer_wire_points = None
    outer_wire_lines = None
    inner_wire_points_list = []
    inner_wire_lines_list = []
    outer_wire_name = ''
    inner_wire_name_list = []

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
            # print(type(all_curves[edge_index]), type(geom_plane))
            # edge_to_add = BRepBuilderAPI_MakeEdge(all_curves[edge_index], geom_plane)
            # 根据方向要求，可能需要翻转边
            if orientation == "Reversed":
                # .Reversed() 返回一个新的翻转后的边，不改变原始对象
                if not edge_index == 46:
                    edge_to_add = edge_to_add.Reversed()
                # edge_to_add = edge_to_add
            
            wire_builder.Add(edge_to_add)
            edges_in_wire.append(edge_index)
        if not wire_builder.IsDone():
            # 通常在这里失败意味着边没有正确地首尾相连
            raise RuntimeError(f"构建线框失败，请检查索引为 {face_index:03d} 的面的边连接性。")
        
        topo_wire = wire_builder.Wire()

        # 根据 is_outer 标志，区分外边界和内孔
        if wire_info["is_outer"]:
            if outer_wire is not None:
                raise ValueError(f"面 {face_index:03d} 有多个外边界，这是不允许的。")
            outer_wire = topo_wire
            outer_wire_points, outer_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            outer_wire_name = '_'.join([f"{i:04d}" for i in edges_in_wire])
        else:
            # pass
            inner_wires.append(topo_wire)
            inner_wire_points, inner_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            inner_wire_points_list.append(inner_wire_points)
            inner_wire_lines_list.append(inner_wire_lines)
            inner_wire_name_list.append('_'.join([f"{i:04d}" for i in edges_in_wire]))


    if outer_wire is None:
        raise ValueError(f"面 {face_index:03d} 没有找到外边界。")

    # --- 3. 构建拓扑面 (Face)，并添加孔洞 ---
    # 首先用基准平面和外边界线框创建基础面
    face_builder = BRepBuilderAPI_MakeFace(geom_plane, outer_wire)
    

    # 将所有内边界线框作为孔洞添加到面中
    for hole_wire in inner_wires:
        face_builder.Add(hole_wire)
        
    if not face_builder.IsDone():
        raise RuntimeError(f"创建面 {face_index:03d} 失败。")
        
    final_face = face_builder.Face()

    # --- 4. 对面进行网格化 ---
    # BRepMesh_IncrementalMesh 会计算三角网格并将其存储在 TopoDS_Face 对象内部
    # 0.1 是线性挠度(linear deflection)值，控制网格精度，值越小网格越密、越精确
    mesh_tolerance = 0.1
    BRepMesh_IncrementalMesh(final_face, mesh_tolerance)
    vertices, faces = extract_mesh_from_face(final_face)
    print(f"成功创建并网格化了平面 (face_index: {face_index:03d})。")
    
    return final_face, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list



def create_thrusections_face_mesh(face_index: int, wires: list, all_edges: dict):
    """
    根据放样线框和孔洞数据重建一个拓扑面并进行网格化。
    此版本使用 BRepOffsetAPI_ThruSections，通过两条 is_outer=True 的线框来创建基础曲面。

    :param face_index: 面的索引，用于错误提示。
    :param wires: 包含线框信息（is_outer, ordered_edges）的列表。
    :param all_edges: 一个字典，其键为 edge_index，值为预先创建好的 TopoDS_Edge 对象。
    :return: 一个元组，包含B-Rep对象、通用网格表示以及用于可视化的线框数据。
    """
    
    # --- 1. 重建并分类所有的线框 (Wires) ---
    lofting_wires = []  # 用于放样的线框 (is_outer: True)
    inner_wires = []    # 作为孔洞的线框 (is_outer: False)

    # 用于可视化的数据
    lofting_wires_data = []
    inner_wires_data = []

    for wire_info in wires:
        wire_builder = BRepBuilderAPI_MakeWire()
        edge_indices_in_wire = []

        for edge_ref in wire_info["ordered_edges"]:
            edge_index = edge_ref["edge_index"]
            orientation = edge_ref["orientation"]

            if edge_index not in all_edges:
                raise KeyError(f"面 {face_index:03d}: 未在提供的 'all_edges' 集合中找到索引为 {edge_index} 的边。")

            edge_to_add = all_edges[edge_index]
            if orientation == "Reversed":
                edge_to_add = edge_to_add.Reversed()

            wire_builder.Add(edge_to_add)
            edge_indices_in_wire.append(edge_index)
        
        # 检查并修复线框（如果需要）
        if not wire_builder.IsDone():
            print(f"警告: 面 {face_index:03d} 的线框构建不完整 (Error: {wire_builder.Error()})。尝试修复...")
            # 注意：ShapeFix_Wire 需要一个基础曲面来投影，这里我们无法预知最终曲面
            # 暂时只能进行基础的连接性修复，效果可能有限
            temp_wire = wire_builder.Wire()
            fixer = ShapeFix_Wire()
            fixer.Load(temp_wire)
            fixer.FixReorder()
            fixer.FixConnected()
            fixer.Perform()
            topo_wire = fixer.Wire()
            if not BRepBuilderAPI_MakeWire(topo_wire).IsDone():
                 raise RuntimeError(f"面 {face_index:03d}: 线框修复失败。")
        else:
            topo_wire = wire_builder.Wire()

        # 分类线框
        wire_name = '_'.join([f"{i:04d}" for i in edge_indices_in_wire])
        # wire_points, wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1) # 假设此函数存在

        if wire_info["is_outer"]:
            lofting_wires.append(topo_wire)
            # lofting_wires_data.append({"name": wire_name, "points": wire_points, "lines": wire_lines})
        else:
            inner_wires.append(topo_wire)
            # inner_wires_data.append({"name": wire_name, "points": wire_points, "lines": wire_lines})

    # --- 2. 验证输入并使用 ThruSections 构建基础曲面 ---
    if len(lofting_wires) != 2:
        raise ValueError(f"面 {face_index:03d}: 需要正好两个 is_outer=True 的线框来进行放样，但找到了 {len(lofting_wires)} 个。")

    # isSolid=False: 创建一个表面而不是实体
    # isRuled=True: 创建直纹面（在截面之间线性插值），对于类似圆柱的形状更稳定
    thru_sections = BRepOffsetAPI_ThruSections(False, True)
    thru_sections.AddWire(lofting_wires[0])
    thru_sections.AddWire(lofting_wires[1])
    thru_sections.Build()

    if not thru_sections.IsDone():
        raise RuntimeError(f"面 {face_index:03d}: BRepOffsetAPI_ThruSections 放样失败。")

    # ThruSections 的结果是一个 Shape，通常是一个 Face 或 Shell。
    # 我们将其作为构建最终带孔曲面的基础。
    base_surface_shape = thru_sections.Shape()

    # --- 3. 构建拓扑面 (Face)，并添加孔洞 ---
    # 使用由 ThruSections 生成的曲面作为基础来构建最终的 Face
    # BRepBuilderAPI_MakeFace 可以接受一个现有的 Face 进行修改（例如添加孔洞）
    face_builder = BRepBuilderAPI_MakeFace(topods_Face(base_surface_shape))
    
    # 将所有内边界线框作为孔洞添加到面中
    for hole_wire in inner_wires:
        face_builder.Add(hole_wire)
        
    if not face_builder.IsDone():
        raise RuntimeError(f"面 {face_index:03d}: 创建带孔洞的面失败 (Error: {face_builder.Error()})。")
        
    final_face = face_builder.Face()

    # --- 4. 对面进行网格化并提取通用表示 ---
    # 这里可以复用您之前的网格化和数据提取逻辑
    # 假设 _build_and_mesh_face_robustly 或类似函数存在
    try:
        # 假设此函数存在并返回 (face, vertices, triangles)
        # final_face, vertices, faces = _build_and_mesh_face_robustly(face_builder, linear_deflection=0.1, angular_deflection=0.2)
        
        # 如果没有辅助函数，这里是基本的网格化逻辑：
        BRepMesh_IncrementalMesh(final_face, 0.1, False, 0.2)
        # vertices, faces = extract_mesh_from_face(final_face) # 假设此函数存在
        vertices, faces = [], [] # Placeholder
        print(f"成功创建、放样并网格化了曲面 (face_index: {face_index:03d})")

    except Exception as e:
        print(f"错误: 面 {face_index:03d} 网格化失败: {e}")
        return None, None, None, None, None

    # 注意：返回的数据结构需要您根据实际情况调整
    return final_face, vertices, faces, lofting_wires_data, inner_wires_data


def create_cylindrical_face_mesh(face_index: int, position: list, axis: list, radius: float, wires: list, all_edges: dict, all_curves: dict):
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
            

            wire_builder.Add(edge_to_add)
            # current_wire = wire_builder.Wire()
            # wire_analyzer = ShapeAnalysis_Wire(current_wire)

            
            if not wire_builder.IsDone():
                print(f"线框构建失败... 尝试修复........... {edge_in_wire} {wire_builder.Error()}")
                # Try fix
                wire_fixer = ShapeFix_Wire(new_wire, context_face, 1e-6)
                # wire_fixer.Perform()

                wire_fixer.FixReorder()
                # wire_fixer.FixConnected()
                # wire_fixer.FixClosed()
                wire_builder = BRepBuilderAPI_MakeWire(wire_fixer.Wire())

                # print(wire_builder.Error(), edge_in_wire)

            new_wire = wire_builder.Wire()
            wire_analyzer = ShapeAnalysis_Wire(new_wire, context_face, 1e-6)
            print(f'Adding edge {edge_index} to wire, edge close status is {edge_to_add.Closed()},  Current wire close status is {new_wire.Closed()} Shape_Analysis_Wire.Closed() is {wire_analyzer.CheckClosed()}')
            edge_in_wire.append(edge_index)
            # edges_in_wire.add(edge_index)


        
        topo_wire = wire_builder.Wire()
        
        wire_fixer = ShapeFix_Wire(topo_wire, context_face, 1e-6)
        wire_fixer.FixReorder()
        topo_wire = wire_fixer.Wire()

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
        hole_wire.Orientation(TopAbs_Orientation.TopAbs_REVERSED)
        face_builder.Add(hole_wire)
        
    if not face_builder.IsDone():
        raise RuntimeError(f"创建面 {face_index:03d} 失败。")
        
    final_face = face_builder.Face()

    # breplib.BuildCurves3d(final_face) # 重新构建面上曲线

    # --- 4. 对面进行网格化并提取通用表示 ---
    # mesh_tolerance = 0.2
    # BRepMesh_IncrementalMesh(final_face, mesh_tolerance)
    final_face, vertices, faces = _build_and_mesh_face_robustly(face_builder, linear_deflection=0.1, angular_deflection=0.2)
    
    # 调用您优化过的辅助函数来提取网格数据
    # vertices, faces = extract_mesh_from_face(final_face)
    
    print(f"成功创建、修复并网格化了圆柱面 (face_index: {face_index:03d})。(vertices.shape: {np.array(vertices).shape}, faces.shape: {np.array(faces).shape})")
    
    return final_face, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list

def create_conical_face_mesh(face_index: int, position: list, axis: list, radius: float, semi_angle: float, wires: list, all_edges: dict, all_curves: dict):
    """
    根据从JSON加载的圆锥面数据及其边界线，重建一个拓扑面并进行网格化。

    :param face_index: 面的索引，用于错误提示。
    :param position: 圆锥顶点(Apex)在轴线上的投影点 [x, y, z]。
    :param axis: 圆锥的轴方向 [x, y, z]。
    :param radius: 圆锥在 position 点所在平面的半径。
    :param semi_angle: 圆锥的半角（弧度）。
    :param wires: 包含线框信息（is_outer, ordered_edges）的列表。
    :param all_edges: 一个字典，其键为 edge_index，值为预先创建好的 TopoDS_Edge 对象。
    :return: 一个元组 (final_face, vertices, faces)，包含B-Rep对象和通用网格表示。
    """

    # --- 1. 创建无限高的基准几何圆锥 ---
    # 首先定义圆锥的坐标系
    center_pnt = gp_Pnt(position[0], position[1], position[2])
    axis_dir = gp_Dir(axis[0], axis[1], axis[2])
    conical_axis = gp_Ax3(center_pnt, axis_dir)  # Z轴是圆锥的轴线

    # 创建几何圆锥基元
    gp_cone = gp_Cone(conical_axis, semi_angle, radius)
    # 用基元初始化可以进行拓扑操作的几何曲面
    geom_conical_surface = Geom_ConicalSurface(gp_cone)

    # --- 2. 重建所有的线框 (Wires) - 逻辑与圆柱/平面版本完全相同 ---
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
        edges_in_wire = []
        for edge_ref in wire_info["ordered_edges"]:
            edge_index = edge_ref["edge_index"]
            orientation = edge_ref["orientation"]
            
            if edge_index not in all_edges:
                raise KeyError(f"未在提供的 'all_edges' 集合中找到索引为 {edge_index} 的边。")
            
            edge_to_add = all_edges[edge_index]
            edges_in_wire.append(edge_index)
            if orientation == "Reversed":
                edge_to_add = edge_to_add.Reversed()
            
            wire_builder.Add(edge_to_add)

        if not wire_builder.IsDone():
            raise RuntimeError(f"构建线框失败，请检查索引为 {face_index:03d} 的面的边连接性。")
        
        topo_wire = wire_builder.Wire()

        if wire_info["is_outer"]:
            if outer_wire is not None:
                raise ValueError(f"面 {face_index:03d} 有多个外边界，这是不允许的。")
            outer_wire = topo_wire
            outer_wire_points, outer_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            outer_wire_name = '_'.join([f"{i:04d}" for i in edges_in_wire])
        else:
            inner_wires.append(topo_wire)
            inner_wire_points, inner_wire_lines = get_wire_vertices_and_lines(topo_wire, 0.1)
            inner_wire_points_list.append(inner_wire_points)
            inner_wire_lines_list.append(inner_wire_lines)
            inner_wire_name_list.append('_'.join([f"{i:04d}" for i in edges_in_wire]))
    if outer_wire is None:
        raise ValueError(f"面 {face_index:03d} 没有找到外边界。")

    # --- 3. 准备 Face Builder ---
    face_builder = BRepBuilderAPI_MakeFace(geom_conical_surface, outer_wire)
    
    for hole_wire in inner_wires:
        face_builder.Add(hole_wire)

    # --- 4. 调用健壮的构建和网格化函数 ---

    final_face, vertices, faces = _build_and_mesh_face_robustly(face_builder, linear_deflection=0.1, angular_deflection=0.2)
    print(f"成功创建、修复并网格化了圆锥面 (face_index: {face_index:03d})。(vertices.shape: {np.array(vertices).shape}, faces.shape: {np.array(faces).shape})")
    return final_face, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list

    

class DistributedSurfaceVisualizer:
    def __init__(self):
        # Data storage
        self.loader = None
        self.available_uids = []
        self.current_uid_idx = 0
        self.current_surface_idx = 0
        self.current_uid_data = None
        
        # Current UID data
        self.surface_cp_list = None
        self.surface_points_list = None
        self.curve_cp_lists = None
        self.curve_points_lists = None
        self.surface_indices = None
        self.surface_types = None
        self.uids = None
        self.file_paths = None
        
        # Counts
        self.total_uids = 0
        self.surfaces_in_current_uid = 0
        
        # Visualization settings
        self.surface_resolution = 32
        self.curve_resolution = 64
        self.show_control_points = True
        self.show_wireframe = False
        self.surface_transparency = 0.8
        self.control_point_size = 0.02
        self.curve_radius = 0.005  # Thickness of curve lines
        self.use_bspline_mode = True  # Toggle between B-spline and direct points mode
        
        # Colors
        self.surface_color = [0.2, 0.8, 0.2]  # Default Green
        self.curve_color = [1.0, 0.0, 0.0]    # Red (not used when gradient is enabled)
        self.control_point_color = [0.0, 0.0, 1.0]  # Blue
        self.outer_wire_color = [1.0, 1.0, 0.0]  # Default yellow (not used when gradient is enabled)
        self.inner_wire_color = [0.5, 0.5, 0.5]  # Default gray (not used when gradient is enabled)
        
        # Curve gradient colors
        self.curve_gradient_start = [1.0, 0.5, 0.5]  # Light red
        self.curve_gradient_end = [0.8, 0.0, 0.0]    # Dark red
        self.use_curve_gradient = True  # Toggle for gradient vs solid color
        
        # Wire gradient colors
        self.outer_wire_gradient_start = [1.0, 1.0, 0.7]  # Light yellow
        self.outer_wire_gradient_end = [0.8, 0.8, 0.0]    # Dark yellow
        self.inner_wire_gradient_start = [0.8, 0.8, 0.8]  # Light gray
        self.inner_wire_gradient_end = [0.3, 0.3, 0.3]    # Dark gray
        self.use_wire_gradient = True  # Toggle for wire gradient vs solid color
        
        # Surface type color mapping
        self.surface_type_colors = {
            "plane": [0.8, 0.2, 0.2],        # Red
            "cylinder": [0.2, 0.8, 0.2],     # Green  
            "cone": [0.2, 0.2, 0.8],         # Blue
            "sphere": [0.8, 0.8, 0.2],       # Yellow
            "torus": [0.8, 0.2, 0.8],        # Magenta
            "bezier_surface": [0.2, 0.8, 0.8], # Cyan
            "bspline_surface": [0.8, 0.5, 0.2], # Orange
            "unknown": [0.5, 0.5, 0.5],      # Gray
        }
        
        # Object tracking
        self.surface_objects = []
        self.curve_objects = []
        self.control_point_objects = []

        self.outer_wire_objects = []
        self.inner_wire_objects = []
        
        # Visibility states for batch operations
        self.show_surfaces = True
        self.show_curves = True
        self.show_outer_wires = True
        self.show_inner_wires = True

    def generate_edge_gradient_colors(self, num_points: int, is_closed: bool = False):
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
                colors[i] = np.array(self.curve_gradient_start) * fade_factor + np.array(self.curve_gradient_end) * (1 - fade_factor)
        else:
            # For open curves, linear gradient from start to end
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 0
                colors[i] = np.array(self.curve_gradient_start) * (1 - t) + np.array(self.curve_gradient_end) * t
        
        return colors

    def generate_wire_gradient_colors(self, num_points: int, is_outer_wire: bool = True, is_closed: bool = False):
        """
        Generate gradient colors for wire objects
        
        Args:
            num_points (int): Number of points on the wire
            is_outer_wire (bool): True for outer wire (yellow gradient), False for inner wire (gray gradient)
            is_closed (bool): Whether the wire is closed (affects how gradient is applied)
            
        Returns:
            numpy.array: Color array with shape (num_points, 3)
        """
        colors = np.zeros((num_points, 3))
        
        if is_outer_wire:
            start_color = np.array(self.outer_wire_gradient_start)
            end_color = np.array(self.outer_wire_gradient_end)
        else:
            start_color = np.array(self.inner_wire_gradient_start)
            end_color = np.array(self.inner_wire_gradient_end)
        
        if is_closed:
            # For closed wires, create a gradient that loops back
            for i in range(num_points):
                # Use a smooth transition that loops
                t = i / num_points
                # Apply a cosine function to create smooth looping gradient
                fade_factor = (1.0 + np.cos(2 * np.pi * t)) / 2.0
                colors[i] = start_color * fade_factor + end_color * (1 - fade_factor)
        else:
            # For open wires, linear gradient from start to end
            for i in range(num_points):
                t = i / (num_points - 1) if num_points > 1 else 0
                colors[i] = start_color * (1 - t) + end_color * t
        
        return colors

    def update_curve_colors(self):
        """Update colors for all curve objects based on current gradient settings"""
        if not hasattr(self, 'edge_data'):
            return
            
        for i, edge in enumerate(self.edge_data):
            curve_name = f"Curve {edge['idx']:03d} {edge['curve_type']}"
            if curve_name in self.curve_objects:
                try:
                    curve_net = ps.get_curve_network(curve_name)
                    if self.use_curve_gradient:
                        # Generate new gradient colors
                        num_points = len(edge['sampled_points'])
                        colors = self.generate_edge_gradient_colors(num_points, edge['is_closed'])
                        curve_net.add_color_quantity("direction_gradient", colors, defined_on='nodes', enabled=True)
                    else:
                        # Use solid color
                        curve_net.set_color(self.curve_color)
                except:
                    pass  # Skip if curve object doesn't exist

    def update_wire_colors(self):
        """Update colors for all wire objects based on current gradient settings"""
        if not hasattr(self, 'wire_data'):
            return
            
        for wire_info in self.wire_data:
            wire_name = wire_info['name']
            is_outer = wire_info['is_outer']
            num_points = wire_info['num_points']
            is_closed = wire_info.get('is_closed', False)
            
            if wire_name in (self.outer_wire_objects if is_outer else self.inner_wire_objects):
                try:
                    wire_net = ps.get_curve_network(wire_name)
                    if self.use_wire_gradient:
                        # Generate new gradient colors
                        colors = self.generate_wire_gradient_colors(num_points, is_outer, is_closed)
                        wire_net.add_color_quantity("wire_gradient", colors, defined_on='nodes', enabled=True)
                    else:
                        # Use solid color
                        if is_outer:
                            wire_net.set_color(self.outer_wire_color)
                        else:
                            wire_net.set_color(self.inner_wire_color)
                except:
                    pass  # Skip if wire object doesn't exist

    def create_gui_callback(self):
        """Create GUI callback for polyscope"""
        def gui_callback():
            # Batch visibility controls
            psim.Text("Batch Visibility Controls")
            psim.Separator()
            
            # Surface objects control
            if psim.Button(f"{'Hide' if self.show_surfaces else 'Show'} All Surfaces ({len(self.surface_objects)})"):
                self.show_surfaces = not self.show_surfaces
                for obj_name in self.surface_objects:
                    try:
                        ps.get_surface_mesh(obj_name).set_enabled(self.show_surfaces)
                    except:
                        pass  # Skip if object doesn't exist
                    try:
                        ps.get_point_cloud(obj_name).set_enabled(self.show_surfaces)
                    except:
                        pass
            
            # Curve objects control
            if psim.Button(f"{'Hide' if self.show_curves else 'Show'} All Curves ({len(self.curve_objects)})"):
                self.show_curves = not self.show_curves
                for obj_name in self.curve_objects:
                    try:
                        ps.get_curve_network(obj_name).set_enabled(self.show_curves)
                    except:
                        pass  # Skip if object doesn't exist
            
            # Outer wire objects control
            if psim.Button(f"{'Hide' if self.show_outer_wires else 'Show'} All Outer Wires ({len(self.outer_wire_objects)})"):
                self.show_outer_wires = not self.show_outer_wires
                for obj_name in self.outer_wire_objects:
                    try:
                        # ps.get_curve_network(obj_name).set_enabled(self.show_outer_wires)
                        ps.get_point_cloud(obj_name).set_enabled(self.show_outer_wires)
                    except:
                        pass  # Skip if object doesn't exist
            
            # Inner wire objects control
            if psim.Button(f"{'Hide' if self.show_inner_wires else 'Show'} All Inner Wires ({len(self.inner_wire_objects)})"):
                self.show_inner_wires = not self.show_inner_wires
                for obj_name in self.inner_wire_objects:
                    try:
                        # ps.get_curve_network(obj_name).set_enabled(self.show_inner_wires)
                        ps.get_point_cloud(obj_name).set_enabled(self.show_inner_wires)
                    except:
                        pass  # Skip if object doesn't exist
            
            psim.Separator()
            
            # Visualization settings
            psim.Text("Visualization Settings")
            
            # # Visualization mode toggle
            # changed, self.use_bspline_mode = psim.Checkbox("Use B-spline Mode", self.use_bspline_mode)
            # if changed:
            #     self.visualize_current_surface()
            
            # psim.SameLine()
            # psim.Text("(Unchecked = Direct Points Mode)")
            
            # changed, self.show_control_points = psim.Checkbox("Show Control Points", self.show_control_points)
            # if changed:
            #     self.visualize_current_surface()
            
            # changed, self.show_wireframe = psim.Checkbox("Show Wireframe", self.show_wireframe)
            # if changed:
            #     self.visualize_current_surface()
            
            # changed, self.surface_transparency = psim.SliderFloat("Surface Transparency", self.surface_transparency, 0.0, 1.0)
            # if changed:
            #     if self.surface_object:
            #         ps.get_surface_mesh(self.surface_object).set_transparency(self.surface_transparency)
            
            # changed, self.control_point_size = psim.SliderFloat("Control Point Size", self.control_point_size, 0.001, 0.1)
            # if changed:
            #     for obj_name in self.control_point_objects:
            #         ps.get_point_cloud(obj_name).set_radius(self.control_point_size)
            
            changed, self.curve_radius = psim.SliderFloat("Curve Thickness", self.curve_radius, 0.001, 0.05)
            if changed:
                for obj_name in self.curve_objects:
                    ps.get_curve_network(obj_name).set_radius(self.curve_radius)
            
            # Color gradient controls
            psim.Text("Edge Direction Gradient")
            changed, self.use_curve_gradient = psim.Checkbox("Enable Direction Gradient", self.use_curve_gradient)
            if changed:
                self.update_curve_colors()
            
            # Gradient start color (light red)
            changed, self.curve_gradient_start = psim.ColorEdit3("Start Color (Light)", self.curve_gradient_start)
            if changed:
                self.update_curve_colors()
                
            # Gradient end color (dark red)  
            changed, self.curve_gradient_end = psim.ColorEdit3("End Color (Dark)", self.curve_gradient_end)
            if changed:
                self.update_curve_colors()
            
            psim.Separator()
            
            # Wire gradient controls
            psim.Text("Wire Gradient Colors")
            changed, self.use_wire_gradient = psim.Checkbox("Enable Wire Gradient", self.use_wire_gradient)
            if changed:
                self.update_wire_colors()
            
            # Outer wire gradient colors (yellow)
            psim.Text("Outer Wire (Yellow Gradient):")
            changed, self.outer_wire_gradient_start = psim.ColorEdit3("Light Yellow", self.outer_wire_gradient_start)
            if changed:
                self.update_wire_colors()
                
            changed, self.outer_wire_gradient_end = psim.ColorEdit3("Dark Yellow", self.outer_wire_gradient_end)
            if changed:
                self.update_wire_colors()
            
            # Inner wire gradient colors (gray)
            psim.Text("Inner Wire (Gray Gradient):")
            changed, self.inner_wire_gradient_start = psim.ColorEdit3("Light Gray", self.inner_wire_gradient_start)
            if changed:
                self.update_wire_colors()
                
            changed, self.inner_wire_gradient_end = psim.ColorEdit3("Dark Gray", self.inner_wire_gradient_end)
            if changed:
                self.update_wire_colors()
            

        return gui_callback

    def process_json(self, cad_data):
        vertex_positions = np.array(cad_data.get('vertices', []))
        edges_list = cad_data.get('edges', [])
        faces_list = cad_data.get('faces', [])
        all_edges = {}
        all_outer_wires = []
        all_inner_wires = []
        all_curves = {}

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
                arc, sampled_points, is_closed, curve = create_arc_from_params(center, normal, radius, start_vertex, end_vertex, return_curve=True)
                # sampled_edges.append( {'idx': idx, 'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
                
            elif edge.get('curve_type') == 'Line':
                if len(v_indices) >= 2:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[1]]
                    arc, sampled_points, is_closed, curve = create_line_from_points(start_vertex, end_vertex, return_curve=True)
                    # sampled_edges.append( {'idx': idx, 'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
                else:
                    print(f"Warning: Edge {idx} of type 'Line' has insufficient vertices ({len(v_indices)})")
                    
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
                arc, sampled_points, is_closed, curve = create_ellipse_from_params(center, x_direction, y_direction, normal, major_radius, minor_radius, start_vertex, end_vertex, return_curve=True)
            elif edge.get('curve_type') == 'Hyperbola':
                curve_def = edge.get('curve_definition', {})
                center = np.array(curve_def.get('center'))
                x_direction = np.array(curve_def.get('major_axis_direction'))
                normal = np.array(curve_def.get('normal'))
                major_radius = curve_def.get('major_radius')
                minor_radius = curve_def.get('minor_radius')
                if len(v_indices) == 2:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[1]]
                else:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[0]]
                arc, sampled_points, is_closed, curve = create_hyperbola_from_params(center, x_direction, normal, major_radius, minor_radius, start_vertex, end_vertex, return_curve=True)
            elif edge.get('curve_type') == 'Parabola':
                curve_def = edge.get('curve_definition', {})
                vertex = np.array(curve_def.get('vertex'))
                axis_of_symmetry = np.array(curve_def.get('axis_of_symmetry'))
                normal = np.array(curve_def.get('normal'))
                focal_length = curve_def.get('focal_length')
                if len(v_indices) == 2:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[1]]
                else:
                    start_vertex = vertex_positions[v_indices[0]]
                    end_vertex = vertex_positions[v_indices[0]]
                arc, sampled_points, is_closed, curve = create_parabola_from_params(vertex, axis_of_symmetry, normal, focal_length, start_vertex, end_vertex, return_curve=True)
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
                        
                    arc, sampled_points, is_closed, curve = create_bspline_from_params(
                        degree, is_periodic, is_closed, control_points, knots, multiplicities, 
                        start_vertex, end_vertex, return_curve=True
                    )
                else:
                    print(f"Warning: Edge {idx} of type 'BSplineCurve' has incomplete curve definition")
                
            else:
                print(f"Warning: Unsupported curve type '{curve_type}' for edge {idx}")
            
            sampled_edges.append( {'idx': idx,  'arc': arc,'sampled_points': sampled_points, 'is_closed': is_closed, 'curve_type': curve_type})
            all_edges[idx] = arc
            all_curves[idx] = curve
            
        # Store edge data for later use in color updates
        self.edge_data = sampled_edges

        for edge in sampled_edges:
            curve_name = f"Curve {edge['idx']:03d} {edge['curve_type']}"
            
            # if edge['curve_type'] == 'Line':
                # Generate gradient colors for this edge
            num_points = len(edge['sampled_points'])
            colors = self.generate_edge_gradient_colors(num_points, edge['is_closed'])
            
            # Register curve network with gradient colors for all curve types

            curve_net = ps.register_curve_network(curve_name, edge['sampled_points'], generate_edges_from_points(edge['sampled_points'], edge['is_closed']))
            # Apply gradient colors to vertices
            curve_net.add_color_quantity("direction_gradient", colors, defined_on='nodes', enabled=True)
            # curve_net.set_color_quantity("direction_gradient")
            curve_net.set_radius(self.curve_radius)
            self.curve_objects.append(curve_name)

        print(f"Total edges processed: {len(sampled_edges):03d}/{len(edges_list):03d}")
        print(f"Begin surface processing...")

        sampled_faces = []
        for face in faces_list:

            surface_idx = face['face_index']
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
            if surface_type == 'Plane':
                position = np.array(face['surface_definition']['position'])
                normal = np.array(face['surface_definition']['normal'])
                face_mesh, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list = create_planar_face_mesh(surface_idx, position, normal, wires, all_edges, all_curves)
                # sampled_faces.append( {'idx': surface_idx, 'surface_type': surface_type, 'face_mesh': face_mesh, 'vertices': vertices, 'faces': faces})
            elif surface_type == 'Cylinder':
                position = np.array(face['surface_definition']['position'])
                axis = np.array(face['surface_definition']['axis'])
                radius = face['surface_definition']['radius']
                face_mesh, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list = create_cylindrical_face_mesh(surface_idx, position, axis, radius, wires, all_edges, all_curves)
                # sampled_faces.append( {'idx': surface_idx, 'surface_type': surface_type, 'face_mesh': face_mesh, 'vertices': vertices, 'faces': faces})
            elif surface_type == 'Cone':
                position = np.array(face['surface_definition']['position'])
                axis = np.array(face['surface_definition']['axis'])
                radius = face['surface_definition']['radius']
                semi_angle = face['surface_definition']['semi_angle']
                face_mesh, vertices, faces, outer_wire_points, outer_wire_lines, outer_wire_name, inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list = create_conical_face_mesh(surface_idx, position, axis, radius, semi_angle, wires, all_edges, all_curves)
            else:
                print(f"Surface type {surface_type} is not supported yet")
                continue
            all_outer_wires.append([outer_wire_points, outer_wire_lines, outer_wire_name])
            for inner_wire_points, inner_wire_lines, inner_wire_name in zip(inner_wire_points_list, inner_wire_lines_list, inner_wire_name_list):
                all_inner_wires.append([inner_wire_points, inner_wire_lines, inner_wire_name])
            sampled_faces.append( {'idx': surface_idx, 'surface_type': surface_type, 'face_mesh': face_mesh, 'vertices': vertices, 'faces': faces})



        for face in sampled_faces:
            face_name = f"Face {face['idx']:03d} {face['surface_type']}"
            try:
                ps.register_surface_mesh(face_name, np.array(face['vertices']), np.array(face['faces']))
                ps.register_point_cloud(face_name, np.array(face['vertices']))
            except Exception as e:
                print(f"Error registering surface mesh {face_name}: {e}")
                continue
            self.surface_objects.append(face_name)

        # Store wire data for later color updates
        self.wire_data = []
        
        for outer_wire_points, outer_wire_lines, outer_wire_name in all_outer_wires:
            outer_wire_name = "Outer Wire " + outer_wire_name
            num_points = len(outer_wire_points)
            
            if self.use_wire_gradient:
                # Generate gradient colors for outer wire (yellow gradient)
                colors = self.generate_wire_gradient_colors(num_points, is_outer_wire=True, is_closed=False)
                # wire_net = ps.register_curve_network(outer_wire_name, outer_wire_points, outer_wire_lines)
                wire_net = ps.register_point_cloud(outer_wire_name, outer_wire_points)
                # wire_net.add_color_quantity("wire_gradient", colors, defined_on='nodes', enabled=True)
                wire_net.add_color_quantity("wire_gradient", colors,  enabled=True)
            else:
                # ps.register_curve_network(outer_wire_name, outer_wire_points, outer_wire_lines, color=self.outer_wire_color)
                ps.register_point_cloud(outer_wire_name, outer_wire_points, color=self.outer_wire_color)
                
            self.outer_wire_objects.append(outer_wire_name)
            self.wire_data.append({
                'name': outer_wire_name,
                'is_outer': True,
                'num_points': num_points,
                'is_closed': False
            })
            
        for inner_wire_points, inner_wire_lines, inner_wire_name in all_inner_wires:
            inner_wire_name = "Inner Wire " + inner_wire_name
            num_points = len(inner_wire_points)
            
            if self.use_wire_gradient:
                # Generate gradient colors for inner wire (gray gradient)
                colors = self.generate_wire_gradient_colors(num_points, is_outer_wire=False, is_closed=False)
                # wire_net = ps.register_curve_network(inner_wire_name, inner_wire_points, inner_wire_lines)
                wire_net = ps.register_point_cloud(inner_wire_name, inner_wire_points)
                # wire_net.add_color_quantity("wire_gradient", colors, defined_on='nodes', enabled=True)
                wire_net.add_color_quantity("wire_gradient", colors,  enabled=True)
            else:
                # ps.register_curve_network(inner_wire_name, inner_wire_points, inner_wire_lines, color=self.inner_wire_color)
                ps.register_point_cloud(inner_wire_name, inner_wire_points, color=self.inner_wire_color)
                
            self.inner_wire_objects.append(inner_wire_name)
            self.wire_data.append({
                'name': inner_wire_name,
                'is_outer': False,
                'num_points': num_points,
                'is_closed': False
            })


    def run(self, cad_data):
        ps.init()

        self.process_json(cad_data)
        ps.set_user_callback(self.create_gui_callback())
        ps.show()




if __name__ == "__main__":
    import sys
    data_path = sys.argv[1]
    # with open('Solid_reconstruction_data.json', 'r') as f:
    with open(data_path, 'r') as f:
        cad_data = json.load(f)

    visualizer = DistributedSurfaceVisualizer()
    visualizer.run(cad_data)