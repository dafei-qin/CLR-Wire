import math
from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Trsf, gp_Vec # <-- 确保导入了 gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_Transform,
)
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods_Face, topods_Wire
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location

def create_lofted_shape_mesh(height=30.0, rotation_angle_deg=45.0):
    """
    创建一个从底部矩形放样到顶部倾斜三角形的闭合实体，并返回其网格。
    这个例子同时展示了“不规则形状”和“不共面”。
    """
    # === 步骤 1: 创建不规则、不共面的线框 (Wires) ===

    # 1a. 创建底部的矩形线框 (位于 XY 平面)
    p1 = gp_Pnt(-10, -10, 0)
    p2 = gp_Pnt(10, -10, 0)
    p3 = gp_Pnt(10, 10, 0)
    p4 = gp_Pnt(-10, 10, 0)
    
    e1 = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
    e2 = BRepBuilderAPI_MakeEdge(p2, p3).Edge()
    e3 = BRepBuilderAPI_MakeEdge(p3, p4).Edge()
    e4 = BRepBuilderAPI_MakeEdge(p4, p1).Edge()
    
    bottom_wire_builder = BRepBuilderAPI_MakeWire()
    bottom_wire_builder.Add(e1)
    bottom_wire_builder.Add(e2)
    bottom_wire_builder.Add(e3)
    bottom_wire_builder.Add(e4)
    bottom_wire = bottom_wire_builder.Wire()

    # 1b. 创建顶部的三角形线框 (初始时也在 XY 平面)
    p5 = gp_Pnt(0, -8, 0)
    p6 = gp_Pnt(8, 8, 0)
    p7 = gp_Pnt(-8, 8, 0)
    
    e5 = BRepBuilderAPI_MakeEdge(p5, p6).Edge()
    e6 = BRepBuilderAPI_MakeEdge(p6, p7).Edge()
    e7 = BRepBuilderAPI_MakeEdge(p7, p5).Edge()

    top_wire_builder = BRepBuilderAPI_MakeWire()
    top_wire_builder.Add(e5)
    top_wire_builder.Add(e6)
    top_wire_builder.Add(e7)
    initial_top_wire = top_wire_builder.Wire()

    # 1c. 对顶部三角形进行变换，使其不共面
    transform = gp_Trsf()
    # 绕 Y 轴旋转
    rotation_axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 1, 0)) # 绕Y轴
    transform.SetRotation(rotation_axis, math.radians(rotation_angle_deg))
    
    # ##################################################################
    # ##                        这里是修正的地方                        ##
    # ##################################################################
    # 错误: transform.SetTranslationPart(gp_Pnt(0, 0, height))
    # 正确:
    transform.SetTranslationPart(gp_Vec(0, 0, height))
    # ##################################################################
    
    top_wire_builder = BRepBuilderAPI_Transform(initial_top_wire, transform)
    top_wire = topods_Wire(top_wire_builder.Shape())

    # === 步骤 2: 放样、封盖并缝合 ===

    # 2a. 创建侧面
    side_face_builder = BRepOffsetAPI_ThruSections(False, True)
    side_face_builder.AddWire(bottom_wire)
    side_face_builder.AddWire(top_wire)
    side_face_builder.Build()
    side_face = side_face_builder.Shape()

    # 2b. 创建盖子
    bottom_face = BRepBuilderAPI_MakeFace(bottom_wire).Face()
    top_face = BRepBuilderAPI_MakeFace(top_wire).Face()

    # 2c. 缝合成一个封闭的壳
    sewing = BRepBuilderAPI_Sewing()
    sewing.Add(side_face)
    sewing.Add(bottom_face)
    sewing.Add(top_face)
    sewing.Perform()
    closed_shape = sewing.SewedShape()

    # === 步骤 3: 网格化并提取数据 (这部分逻辑与上个例子完全相同) ===
    mesh = BRepMesh_IncrementalMesh(closed_shape, 0.5, False, 0.5)
    mesh.Perform()
    
    vertices = []
    triangles = []
    vertex_map = {}
    next_vertex_index = 0
    
    face_explorer = TopExp_Explorer(closed_shape, TopAbs_FACE)
    while face_explorer.More():
        current_face = topods_Face(face_explorer.Current())
        location = TopLoc_Location()
        poly_triangulation = BRep_Tool.Triangulation(current_face, location)

        if poly_triangulation is None:
            face_explorer.Next()
            continue



        for i in range(1, poly_triangulation.NbTriangles() + 1):
            tri_indices = []
            poly_tri =  poly_triangulation.Triangle(i)
            v1_idx, v2_idx, v3_idx = poly_tri.Get()
            for local_idx in [v1_idx, v2_idx, v3_idx]:
                point = poly_triangulation.Node(local_idx).Transformed(location.Transformation())
                pt_coords = (point.X(), point.Y(), point.Z())
                
                if pt_coords not in vertex_map:
                    vertices.append(pt_coords)
                    vertex_map[pt_coords] = next_vertex_index
                    tri_indices.append(next_vertex_index)
                    next_vertex_index += 1
                else:
                    tri_indices.append(vertex_map[pt_coords])
            
            triangles.append(tuple(tri_indices))
            
        face_explorer.Next()

    return vertices, triangles, closed_shape

if __name__ == '__main__':
    verts, tris, final_shape = create_lofted_shape_mesh()

    if verts and tris:
        print(f"成功生成放样实体网格！")
        print(f"总顶点数: {len(verts)}")
        print(f"总三角形数: {len(tris)}")
        
        # 可选：可视化
        try:
            from OCC.Display.SimpleGui import init_display
            display, start_display, _, _ = init_display()
            display.DisplayShape(final_shape, update=True, color="cyan")
            display.FitAll()
            start_display()
        except ImportError:
            print("\n提示: 未找到 SimpleGui，无法进行可视化。")
