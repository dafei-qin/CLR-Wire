import FreeCAD as App
import FreeCADGui as Gui
import Part
import math

# --- 0. 准备工作：确保有一个活动的文档 ---
doc = App.ActiveDocument
if not doc:
    doc = App.newDocument("Test_Document")
    App.Console.PrintMessage("已创建新文档。\n")

# --- 1. 定义几何参数和姿态 ---
# 注意：我们无法通过此方法指定焦距，所以将使用FreeCAD的默认值
vertex_pos = App.Vector(-40, 50, 10) # 顶点的最终位置

# 定义一个非标准的姿态 (位置和旋转)
rotation_axis = App.Vector(2, -1, 1).normalize()
rotation_angle = 30.0

# --- 2. 创建有限的抛物线边 ---

# a. 【【【 核心修正 】】】
#    直接创建一个具有默认焦距的标准抛物线。
#    不再尝试设置 .FocalLength 属性。
base_parabola_curve = Part.Parabola()
# App.Console.PrintMessage(f"已创建默认抛物线，其默认焦距为: {base_parabola_curve.FocalLength}\n")


# b. 根据无限曲线创建有限的边
#    对于默认抛物线，我们仍然可以截取其中一段来显示
u_min = -5.0
u_max = 5.0
finite_parabola_edge = Part.Edge(base_parabola_curve, u_min, u_max)

# c. 创建一个Placement对象来定义最终的位置和姿态
rotation = App.Rotation(rotation_axis, rotation_angle)
placement = App.Placement(vertex_pos, rotation)

# d. 对有限的“边”进行变换
transformed_edge = finite_parabola_edge.transformShape(placement.Matrix)

# --- 3. 将最终的“边”添加到文档中并使其可见 ---

OBJECT_NAME = "Test_Parabola_Edge"

# a. 检查并删除同名旧对象
if doc.getObject(OBJECT_NAME):
    doc.removeObject(OBJECT_NAME)
    doc.recompute()

# b. 在文档中显示这个有限的、可渲染的形状
Part.show(transformed_edge, OBJECT_NAME)

# c. 刷新文档并选中
doc.recompute()
Gui.Selection.clearSelection()
if doc.getObject(OBJECT_NAME):
    Gui.Selection.addSelection(doc.getObject(OBJECT_NAME))
    App.Console.PrintMessage(f"对象 '{OBJECT_NAME}' 已创建并选中。\n")

# --- 4. 调整相机视角 ---
try:
    view = Gui.activeDocument().activeView()
    view.fitAll()
    App.Console.PrintMessage("相机视角已自动调整。\n")
except Exception as e:
    App.Console.PrintWarning(f"无法自动调整相机视角: {e}\n")

App.Console.PrintMessage("="*30 + "\n")
App.Console.PrintMessage("抛物线测试脚本执行完毕。\n")
App.Console.PrintMessage("="*30 + "\n")