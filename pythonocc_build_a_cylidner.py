import sys
# 确保导入了 gp_Vec
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, 
                                     BRepBuilderAPI_MakeWire, 
                                     BRepBuilderAPI_MakeFace,
                                     BRepBuilderAPI_MakeSolid)
from OCC.Core.BRep import BRep_Builder
from OCC.Core.Geom import Geom_Circle
from OCC.Core.TopoDS import TopoDS_Shell, TopoDS_Solid
from OCC.Display.SimpleGui import init_display

def create_closed_cylinder(radius: float, height: float) -> TopoDS_Solid:
    """
    利用 pythonocc 构建一个闭合的圆柱体。

    :param radius: 圆柱体的半径。
    :param height: 圆柱体的高度。
    :return: 一个闭合的圆柱体实体 (TopoDS_Solid)。
    """
    position = gp_Pnt(0, 0, 0)
    direction = gp_Dir(0, 0, 1)
    axis = gp_Ax2(position, direction)

    cylinder_wall_shape = BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()

    bottom_circle_geom = Geom_Circle(axis, radius)
    bottom_edge = BRepBuilderAPI_MakeEdge(bottom_circle_geom).Edge()
    bottom_wire = BRepBuilderAPI_MakeWire(bottom_edge).Wire()
    bottom_face = BRepBuilderAPI_MakeFace(bottom_wire).Face()

    # --- 代码修正处 ---
    # 创建一个平移矢量，它有方向 (direction) 和大小 (height)
    translation_vector = gp_Vec(direction) * height
    # 使用这个矢量来计算顶部圆心的位置
    top_position = position.Translated(translation_vector)
    # --- 修正结束 ---
    
    top_axis = gp_Ax2(top_position, direction)
    top_circle_geom = Geom_Circle(top_axis, radius)
    top_edge = BRepBuilderAPI_MakeEdge(top_circle_geom).Edge()
    top_wire = BRepBuilderAPI_MakeWire(top_edge).Wire()
    top_face = BRepBuilderAPI_MakeFace(top_wire).Face()

    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    builder.Add(shell, cylinder_wall_shape)
    builder.Add(shell, bottom_face)
    builder.Add(shell, top_face)

    solid_maker = BRepBuilderAPI_MakeSolid(shell)
    solid_maker.Build()
    if not solid_maker.IsDone():
        raise SystemError("Failed to create solid from shell.")
    
    closed_cylinder = solid_maker.Solid()
    
    return closed_cylinder

if __name__ == '__main__':
    display, start_display, add_menu, _ = init_display()

    cylinder_radius = 20.0
    cylinder_height = 50.0

    my_closed_cylinder = create_closed_cylinder(cylinder_radius, cylinder_height)

    display.DisplayShape(my_closed_cylinder, update=True, color="Green", transparency=0.3)
    display.Display_AIS_trihedron()
    start_display()