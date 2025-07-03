import json
import math
import numpy as np
def find_special_cylinder_faces(json_path):
    """
    读取指定 JSON 文件，返回所有 outer wire
    由两个完整圆和一条被重复利用的线（Line 或 degree=1 BSpline）构成的 faces。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 将所有 edges 建立索引
    edge_map = {e["edge_index"]: e for e in data["edges"]}
    vertices = np.array(data["vertices"])
    def is_closed_circle(edge):
        # 精确判断从 0 到 2π
        return (
            edge.get("curve_type") == "Circle" and
            len(edge.get("vertices")) == 1
        )

    def is_linear(edge):
        # Line 或 degree=1 的 BSplineCurve
        if edge.get("curve_type") == "Line":
            return True
        if edge.get("curve_type") == "BSplineCurve":
            return edge.get("curve_definition", {}).get("degree") == 1
        return False

    special_faces = []

    for face in data.get("faces", []):
        # 找到 outer wire
        outer = None
        for w in face.get("wires", []):
            if w.get("is_outer"):
                outer = w["ordered_edges"]
                break
        if not outer or len(outer) != 4:
            continue

        # 统计
        circle_count = 0
        linear_count = 0
        used_ids = [e["edge_index"] for e in outer]

        used_edges = [edge_map[eid] for eid in used_ids]
        for eid in used_ids:
            edge = edge_map.get(eid)
            if not edge:
                break
            if is_closed_circle(edge):
                circle_count += 1
            if is_linear(edge):
                # 对重复的线也只计一次
                linear_count += 1
        else:
            # 计算被重复使用的 edge id
            reused = [eid for eid in set(used_ids) if used_ids.count(eid) > 1]
            # 条件判断
            if circle_count == 2 and linear_count >= 1 and len(reused) == 1:
                special_faces.append({
                    "face_index":    face["face_index"],
                    "face_type":     face["surface_type"],
                    "edge_sequence": used_ids,
                    "reused_edge":   reused[0],
                    "edges": used_edges
                })

    return special_faces


if __name__ == "__main__":
    json_file = r"C:\Users\Dafei Qin\0056_reconstructed_data.json"
    matches = find_special_cylinder_faces(json_file)
    print("符合条件的 faces：")
    for m in matches:
        print(f"  face_index={m['face_index']}, face_type={m['face_type']}, edges={m['edge_sequence']}, repeated={m['reused_edge']}")
        for edge in m['edges']:
            print(f"  \t edge_index={edge['edge_index']}, edge_type= {edge['curve_type']:>15}, vertices={edge['vertices']}")
            if edge['curve_type'] == 'Circle':
                print(f"  \t \t circle_center={edge['curve_definition']['center']}, circle_radius={edge['curve_definition']['radius']}")
        print()
