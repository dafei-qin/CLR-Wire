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
center_pos = App.Vector(50, 30, -20)
major_radius = 100
minor_radius = 40
rotation_axis = App.Vector(1, 1, 2).normalize()
rotation_angle = 45.0

# --- 2. 创建有限的双曲线边 ---

# a. 在原点创建一个无限的、标准姿态的双曲线几何体
base_hyperbola_curve = Part.Hyperbola(App.Vector(0, 0, 0), major_radius, minor_radius)

# b. 【【【 核心修正：根据无限曲线创建有限的边 】】】
#    我们定义一个参数范围 (u_min, u_max) 来截取曲线的一部分。
#    u=0 是双曲线的顶点，我们取其周围的一段。
u_min = -2.0
u_max = 2.0
finite_hyperbola_edge = Part.Edge(base_hyperbola_curve, u_min, u_max)

# c. 创建一个Placement对象来定义最终的位置和姿态
rotation = App.Rotation(rotation_axis, rotation_angle)
placement = App.Placement(center_pos, rotation)

# d. 对有限的“边”进行变换
transformed_edge = finite_hyperbola_edge.transformShape(placement.Matrix)

# --- 3. 将最终的“边”添加到文档中并使其可见 ---

OBJECT_NAME = "Test_Hyperbola_Edge"

# a. 检查并删除同名旧对象
if doc.getObject(OBJECT_NAME):
    doc.removeObject(OBJECT_NAME)
    doc.recompute()

# b. 在文档中显示这个有限的、可渲染的形状
Part.show(transformed_edge, OBJECT_NAME)

# c. 刷新文档
doc.recompute()

# d. 自动选中对象
Gui.runCommand('Std_SelectAll',0)
Gui.Selection.clearSelection()
if doc.getObject(OBJECT_NAME):
    Gui.Selection.addSelection(doc.getObject(OBJECT_NAME))
    App.Console.PrintMessage(f"对象 '{OBJECT_NAME}' 已创建并选中。\n")
else:
    App.Console.PrintError(f"错误：无法找到新创建的对象 '{OBJECT_NAME}'。\n")

# --- 4. 调整相机视角 ---
try:
    view = Gui.activeDocument().activeView()
    view.fitAll() # 缩放视图以适应所有对象
    App.Console.PrintMessage("相机视角已自动调整。\n")
except Exception as e:
    App.Console.PrintWarning(f"无法自动调整相机视角: {e}\n")

App.Console.PrintMessage("="*30 + "\n")
App.Console.PrintMessage("脚本执行完毕。这次应该真的可以了！\n")
App.Console.PrintMessage("="*30 + "\n")