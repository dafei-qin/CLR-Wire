# -*- coding: utf-8 -*-
import FreeCAD as App
import FreeCADGui as Gui
import time
from collections import deque

# --- 1. 获取 FreeCAD 对象 ---
try:
    obj = Gui.Selection.getSelection()[0]
    shape = obj.Shape
    all_faces_geom = shape.Faces
    num_faces = len(all_faces_geom)
    App.Console.PrintMessage(f"模型加载成功，共找到 {num_faces} 个面。\n")
except IndexError:
    App.Console.PrintError("错误：请先在模型树中选择一个对象。\n")
    raise SystemExit("请选择一个对象后重Test")

# --- 2. 预处理：手动构建 边 -> 面 的对应地图 ---
# 这是本次修正的核心：我们自己建立这个关系，不再依赖任何可能不存在的函数。
App.Console.PrintMessage("正在预处理，构建边-面拓扑地图...\n")
edge_to_faces_map = {}
for i, face in enumerate(all_faces_geom):
    for edge in face.Edges:
        # setdefault会确保键存在，如果不存在则创建一个空列表
        edge_to_faces_map.setdefault(edge.hashCode(), []).append(i)
App.Console.PrintMessage("拓扑地图构建完成。\n")


# --- 3. 确定唯一的根节点 ---
App.Console.PrintMessage("正在确定根节点...\n")
index_to_geom_map = {i: face for i, face in enumerate(all_faces_geom)}
root_index = min(range(num_faces), key=lambda i: (
    index_to_geom_map[i].CenterOfMass.x,
    index_to_geom_map[i].CenterOfMass.y,
    index_to_geom_map[i].CenterOfMass.z
))
root_face_name = f"Face{root_index + 1}"
App.Console.PrintMessage(f"根节点确定为: {root_face_name} (索引: {root_index})\n")


# --- 4. 构建有向图并进行遍历排序 ---
App.Console.PrintMessage("开始构建有向图并进行拓扑遍历...\n")
ordered_face_indices = []
visited = set()
queue = deque([root_index])
visited.add(root_index)

while queue:
    current_index = queue.popleft()
    ordered_face_indices.append(current_index)
    
    current_face_geom = index_to_geom_map[current_index]
    
    try:
        # 注意：这里我们用 .Edges 而不是 .OuterWire.OrderedEdges
        # 因为一个面可能由多个环（比如有孔的洞），我们需要检查所有的边
        edges_of_current_face = current_face_geom.Edges
    except Exception:
        continue

    for edge in edges_of_current_face:
        # 【最终修正】: 直接从我们手动构建的地图中查询
        sharing_faces_indices = edge_to_faces_map.get(edge.hashCode(), [])

        for neighbor_index in sharing_faces_indices:
            if neighbor_index != current_index:
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append(neighbor_index)
                # 找到一个邻居就够了，跳出内层循环
                break

App.Console.PrintMessage("图构建与遍历完成！\n")


# --- 5. 验证和输出结果 ---
App.Console.PrintMessage("--- 有向图拓扑遍历排序结果 ---\n")
for i, face_index in enumerate(ordered_face_indices):
    face_name_str = f"Face{face_index + 1}"
    print(f"新顺序 #{i+1}: {face_name_str}")
    
    Gui.Selection.clearSelection()
    Gui.Selection.addSelection(obj, face_name_str)
    Gui.updateGui() 
    time.sleep(0.1)

App.Console.PrintMessage(f"--- 验证完成 --- (共找到 {len(ordered_face_indices)} 个连接的面)\n")