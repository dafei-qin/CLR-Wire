import FreeCAD as App
import FreeCADGui as Gui
import Part
import time
import os
import json
from collections import deque
from datetime import datetime


from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, 
                                      BRepBuilderAPI_MakeWire, 
                                      BRepBuilderAPI_MakeFace)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL, TopAbs_EXTERNAL
from OCC.Core.TopoDS import topods, TopoDS_Wire
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_Line, Geom_Circle, Geom_Ellipse, Geom_BSplineCurve, Geom_BezierCurve
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire

def get_orientation_string(orientation_enum):
    """将 TopAbs 方向枚举转换为字符串"""
    if orientation_enum == TopAbs_FORWARD:
        return "FORWARD"
    if orientation_enum == TopAbs_REVERSED:
        return "REVERSED"
    if orientation_enum == TopAbs_INTERNAL:
        return "INTERNAL"
    if orientation_enum == TopAbs_EXTERNAL:
        return "EXTERNAL"
    return "UNKNOWN"

def get_curve_type_string(edge):
    """获取边的底层几何类型"""
    # BRep_Tool.Curve 返回 (Geom_Curve, u_min, u_max)
    curve, _, _ = BRep_Tool.Curve(edge)
    
    if curve is None:
        return "No Curve"
    if isinstance(curve, Geom_Line):
        return "Line"
    if isinstance(curve, Geom_Circle):
        return "Circle"
    if isinstance(curve, Geom_Ellipse):
        return "Ellipse"
    if isinstance(curve, Geom_BSplineCurve):
        return "BSplineCurve"
    if isinstance(curve, Geom_BezierCurve):
        return "BezierCurve"
    return "Other (" + curve.DynamicType().Name() + ")"

App.Console.PrintMessage("="*50 + "\n")
App.Console.PrintMessage("开始执行“完整CAD拓扑数据序列化为JSON”脚本 (v3.0 - 新增椭圆支持)...\n")

# --- 1. 获取 FreeCAD 对象 ---
try:
    obj = Gui.Selection.getSelection()[0]
    shape = obj.Shape
except IndexError:
    App.Console.PrintError("错误：请先在模型树中选择一个对象。\n")
    raise SystemExit("请选择一个对象后重试")



face = shape.Faces[5]
face_occ = Part.__toPythonOCC__(face)

fix_shape = ShapeFix_Shape(face_occ)
fix_shape.Perform()
fixed_face_occ = fix_shape.Shape()
all_wires = []
wires_and_their_edges = []

print("--- 开始遍历面上的所有线框 ---")

# a. 使用 TopExp_Explorer 查找面上的所有 Wire
wire_explorer = TopExp_Explorer(fixed_face_occ, TopAbs_WIRE)

wire_index = 1
while wire_explorer.More():
    # --- 信息分析 (与之前相同) ---
    original_wire_shape = wire_explorer.Current()
    fixer = ShapeFix_Wire(original_wire_shape, fixed_face_occ, 1e-5) 
    fix_status = fixer.Perform()
    if fix_status:
        print("  -> ShapeFix_Wire 执行成功。")
        repaired_wire = fixer.Wire()
    else:
        print("  -> ShapeFix_Wire 执行失败。")
        repaired_wire = original_wire_shape
    original_wire = topods.Wire(repaired_wire)
    wire_orientation = original_wire_shape.Orientation()
    
    if wire_orientation == TopAbs_FORWARD: wire_classification = "Outer Boundary"
    elif wire_orientation == TopAbs_REVERSED: wire_classification = "Inner Boundary (Hole)"
    else: wire_classification = "Unclassified"

    print(f"\n[Wire {wire_index}] Classification: {wire_classification}")
    
    edges_in_current_wire = []
    edge_explorer = TopExp_Explorer(original_wire, TopAbs_EDGE)
    while edge_explorer.More():
        edges_in_current_wire.append(topods.Edge(edge_explorer.Current()))
        edge_explorer.Next()
        
    print(f"  - 找到了 {len(edges_in_current_wire)} 条边。")

    # --- 新增：尝试从收集到的边重建线框 ---
    print("  -> 正在尝试从以上边重建线框...")
    
    wire_builder = BRepBuilderAPI_MakeWire()
    
    # 将所有提取出的边添加到新的构建器中
    for edge in edges_in_current_wire:
        wire_builder.Add(edge)
        
    if wire_builder.IsDone():
        rebuilt_wire = wire_builder.Wire()
        print("  -> 重建状态: 成功!") # 绿色表示成功
        
        # 验证重建的线框是否与原始线框相同
        if rebuilt_wire.IsSame(original_wire):
            print("  -> 验证结果: 重建的线框与原始线框在拓扑上完全相同。")
        else:
            print("  -> 警告: 重建的线框与原始线框不同。") # 黄色表示警告

        # 检查闭合状态是否一致
        if rebuilt_wire.Closed() == original_wire.Closed():
            print(f"  -> 闭合状态验证: 一致 (Closed: {rebuilt_wire.Closed()})。")
        else:
            print(f"  -> 警告: 闭合状态不一致。")

    else:
        print("  -> 重建状态: 失败!") # 红色表示失败
        # 如果失败，可以获取错误代码（虽然代码本身是枚举，不易读）
        error_status = wire_builder.Error()
        print(f"  -> 错误代码: {error_status}")
        
    wire_index += 1
    wire_explorer.Next()

print("\n--- 所有操作完成 ---")