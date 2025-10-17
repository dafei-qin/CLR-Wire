from tkinter import Pack
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge

import FreeCAD as App
import Part
# ---------------------
# 1. 基本参数
# ---------------------
degree = 3
poles_py = [
    gp_Pnt(0, 0, 0),
    gp_Pnt(1, 2, 0),
    gp_Pnt(2, 3, 0),
    gp_Pnt(4, 2, 0),
    gp_Pnt(5, 0, 0),
    gp_Pnt(6, -1, 0)
]

knots_py = [0.0, 0.25, 0.5, 0.75, 1.0]
multiplicities_py = [4, 1, 1, 1, 3]

# ---------------------
# 2. 转换为 OCC 数组类型
# ---------------------
# 控制点数组
poles = TColgp_Array1OfPnt(1, len(poles_py))
for i, p in enumerate(poles_py):
    poles.SetValue(i + 1, p)

# 节点数组
knots = TColStd_Array1OfReal(1, len(knots_py))
for i, k in enumerate(knots_py):
    knots.SetValue(i + 1, k)

# 重数数组
multiplicities = TColStd_Array1OfInteger(1, len(multiplicities_py))
for i, m in enumerate(multiplicities_py):
    multiplicities.SetValue(i + 1, m)

# ---------------------
# 3. 创建曲线对象
# ---------------------
curve = Geom_BSplineCurve(poles, knots, multiplicities, degree, False)
print("✅ B-Spline curve created successfully!")

# ---------------------
# 4. 转换为 TopoDS_Shape
# ---------------------
# 使用 BRepBuilderAPI_MakeEdge 从曲线创建边
edge_builder = BRepBuilderAPI_MakeEdge(curve)
topods_shape = edge_builder.Shape()
print("✅ TopoDS_Shape created successfully!")
print(f"Shape type: {topods_shape.ShapeType()}")

# 可选：如果需要在 FreeCAD 中显示
curve_freecad = Part.__fromPythonOCC__(topods_shape)
Part.show(curve_freecad, 'curve')