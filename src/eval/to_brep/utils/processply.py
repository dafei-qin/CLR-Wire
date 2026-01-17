import numpy as np
import math
import random
import matplotlib.cm as cm
from scipy.spatial import cKDTree

# OCCT Imports
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Vec, gp_Dir
from OCC.Core.GeomLProp import GeomLProp_SLProps

import asyncio
import platform

is_windows = platform.system() == "Windows"


# ==============================================================================
# 1. 辅助函数 (保持上一版逻辑，用于法向校验)
# ==============================================================================
debug__=True

if debug__:
    def mydebug(*args, **kwargs):
        print(*args, **kwargs,flush=True)
        pass
else:
    def mydebug(*args, **kwargs):
        pass

def estimate_point_cloud_normals(cloud_np, k=10):
    """(保持不变) 使用 PCA 估算点云法向"""
    tree = cKDTree(cloud_np)
    dists, indices = tree.query(cloud_np, k=k, workers=-1)
    neighbors = cloud_np[indices]
    means = np.mean(neighbors, axis=1, keepdims=True)
    centered = neighbors - means
    covariances = np.matmul(centered.transpose(0, 2, 1), centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    normals = eigenvectors[:, :, 0]
    return normals

def sample_probes_with_normals(face, num_points, deflection=0.1):
    """(保持不变) 采样探针及其法向"""
    BRepMesh_IncrementalMesh(face, deflection)
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)
    if not triangulation: return np.empty((0, 3)), np.empty((0, 3))

    trsf = loc.Transformation()
    adaptor = BRepAdaptor_Surface(face)
    has_uv = triangulation.HasUVNodes()
    
    triangles_data = []
    total_area = 0.0
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        p1 = triangulation.Node(n1).Transformed(trsf)
        p2 = triangulation.Node(n2).Transformed(trsf)
        p3 = triangulation.Node(n3).Transformed(trsf)
        area = 0.5 * gp_Vec(p1, p2).Crossed(gp_Vec(p1, p3)).Magnitude()
        if area > 1e-12:
            total_area += area
            triangles_data.append((area, n1, n2, n3))

    if total_area == 0: return np.empty((0, 3)), np.empty((0, 3))

    areas = np.array([t[0] for t in triangles_data])
    probs = areas / total_area
    selected_indices = np.random.choice(len(triangles_data), size=num_points, p=probs)
    
    points, normals = [], []
    for idx in selected_indices:
        _, n1, n2, n3 = triangles_data[idx]
        r1, r2 = random.random(), random.random()
        sqrt_r1 = math.sqrt(r1)
        w1, w2, w3 = 1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2
        
        if has_uv:
            uv1, uv2, uv3 = triangulation.UVNode(n1), triangulation.UVNode(n2), triangulation.UVNode(n3)
            u = w1 * uv1.X() + w2 * uv2.X() + w3 * uv3.X()
            v = w1 * uv1.Y() + w2 * uv2.Y() + w3 * uv3.Y()
            props = GeomLProp_SLProps(adaptor.Surface().Surface(), u, v, 1, 1e-6)
            if props.IsNormalDefined():
                p_world = props.Value().Transformed(trsf)
                n_world = props.Normal().Transformed(trsf)
                points.append([p_world.X(), p_world.Y(), p_world.Z()])
                normals.append([n_world.X(), n_world.Y(), n_world.Z()])
        else:
            p1 = triangulation.Node(n1).Transformed(trsf)
            p2 = triangulation.Node(n2).Transformed(trsf)
            p3 = triangulation.Node(n3).Transformed(trsf)
            x = w1*p1.X() + w2*p2.X() + w3*p3.X()
            y = w1*p1.Y() + w2*p2.Y() + w3*p3.Y()
            z = w1*p1.Z() + w2*p2.Z() + w3*p3.Z()
            n_geo = gp_Vec(p1, p2).Crossed(gp_Vec(p1, p3))
            if n_geo.Magnitude() > 1e-9:
                n_geo.Normalize()
                points.append([x, y, z])
                normals.append([n_geo.X(), n_geo.Y(), n_geo.Z()])
                
    return np.array(points), np.array(normals)

def save_result_to_ply(faces, face_values, filename, colormap_name='jet'):
    """(保持不变) PLY 导出"""
    mydebug(f"正在导出 PLY: {filename} ...")
    cmap = cm.get_cmap(colormap_name)
    values_clamped = np.clip(face_values, 0.0, 1.0)
    colors_rgba = cmap(values_clamped)
    
    all_vertices, all_faces_indices, all_vertex_colors = [], [], []
    vertex_offset = 0
    
    for i, face in enumerate(faces):
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if not triangulation: continue
        trsf = loc.Transformation()
        r, g, b = int(colors_rgba[i][0]*255), int(colors_rgba[i][1]*255), int(colors_rgba[i][2]*255)
        
        face_nodes = []
        for j in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(j).Transformed(trsf)
            face_nodes.append((p.X(), p.Y(), p.Z()))
            all_vertex_colors.append((r,g,b))
        all_vertices.extend(face_nodes)
        
        for j in range(1, triangulation.NbTriangles() + 1):
            tri = triangulation.Triangle(j)
            n1, n2, n3 = tri.Get()
            all_faces_indices.append((n1-1+vertex_offset, n2-1+vertex_offset, n3-1+vertex_offset))
        vertex_offset += len(face_nodes)

    with open(filename, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(all_vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(all_faces_indices)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v, c in zip(all_vertices, all_vertex_colors):
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        for tri in all_faces_indices:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

# ==============================================================================
# 2. 核心函数：相对阈值筛选
# ==============================================================================

def estimate_optimal_radius(point_cloud, sample_size=2000, multiplier=2.5):
    """(保持不变)"""
    cloud_np = point_cloud
    if len(cloud_np) < 2: return 1.0
    
    sample_points = cloud_np if len(cloud_np) <= sample_size else cloud_np[np.random.choice(len(cloud_np), sample_size, replace=False)]
    tree = cKDTree(cloud_np)
    dists, _ = tree.query(sample_points, k=2, workers=-1)
    valid_dists = dists[:, 1][dists[:, 1] > 1e-6]
    
    if len(valid_dists) == 0: return 1.0
    suggested_radius = np.median(valid_dists) * multiplier
    mydebug(f"[Auto Radius] Median: {np.median(valid_dists):.4f}, Multiplier: {multiplier}, Result: {suggested_radius:.4f}")
    return suggested_radius

def filter_faces_by_coverage(faces, cloud_normals,tree, n_samples=100, coverage_radius=5.0, min_coverage_ratio=0.8, save_path=None):
    # 存储所有面的计算结果 (face_object, score)
    face_scores = [] 
    raw_ratios = [] # 用于导出 PLY
    
    mydebug(f"开始计算覆盖率 (Radius={coverage_radius:.2f}, 相对系数 k={min_coverage_ratio})...")
    normal_threshold_cos = math.cos(math.radians(30)) # 法向夹角容差
    
    # 2. 遍历计算所有分数
    for i, face in enumerate(faces):
        probes_pos, probes_norm = sample_probes_with_normals(face, n_samples)
        treei = cKDTree(probes_pos)

        if len(probes_pos) == 0:
            face_scores.append((face, 0.0))
            raw_ratios.append(0.0)
            continue
            
        # 距离查询
        indices_list = treei.query_ball_tree(tree, coverage_radius)
        hit_count = 0
        
        for p_idx, neighbors_indices in enumerate(indices_list):
            if not neighbors_indices: continue
            
            # 法向校验
            neighbors_normals = cloud_normals[neighbors_indices]
            current_probe_normal = probes_norm[p_idx]
            dots = np.dot(neighbors_normals, current_probe_normal)
            
            if np.any(np.abs(dots) > normal_threshold_cos):
                hit_count += 1
                
        ratio = hit_count / len(probes_pos)
        face_scores.append((face, ratio))
        raw_ratios.append(ratio)

    # 3. 执行相对筛选策略
    if not face_scores:
        return []
        
    # 找到最大分数
    max_score = max(score for _, score in face_scores)
    
    # 计算动态阈值
    # 如果最大分数太低（例如全是 0），说明计算失败或无重叠，需要保护
    if max_score < 1e-6:
        mydebug("警告: 最大覆盖率为 0，未找到有效面。")
        dynamic_threshold = 0.0
    else:
        dynamic_threshold = max_score * min_coverage_ratio
        
    passed_faces = [f for f, score in face_scores if score >= dynamic_threshold and score > 0]
    passed_faces_score = [score for f, score in face_scores if score >= dynamic_threshold and score > 0]
    
    mydebug(f"统计: Max Score={max_score:.4f}, Threshold={dynamic_threshold:.4f} (k={min_coverage_ratio})")
    mydebug(f"筛选结果: {len(passed_faces)} / {len(faces)} 通过。")
    
    # 4. 导出结果 (使用原始分数，以便观察分布)
    if save_path:
        save_result_to_ply(faces, raw_ratios, save_path, colormap_name='jet')
        
    return passed_faces,passed_faces_score

import math
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt

def get_point_key(p,tol):
    # 简单的空间哈希
    inv_tol = 1.0 / tol
    ix = int(round(p.X() * inv_tol))
    iy = int(round(p.Y() * inv_tol))
    iz = int(round(p.Z() * inv_tol))
    return (ix, iy, iz)

def get_edge_key(edge,tol):
    
    try:
        curve, u_min, u_max = BRep_Tool.Curve(edge)
    except Exception as e:
        return f"null_curve_edge {topods.Edge(edge).TShape().__hash__()}"
    
    # 如果 Edge 没有几何曲线（例如退化边），则返回特定的标识
    if not curve:
        return f"null_curve_edge {topods.Edge(edge).TShape().__hash__()}"
    
    # 3. 计算4个采样点的参数 (按参数递增顺序)
    # Start (0%), Mid1 (33.3%), Mid2 (66.6%), End (100%)
    param_range = u_max - u_min
    u_params = [
        u_min,                          # 起点
        u_min + param_range * (1.0/3.0), # 中间点 1
        u_min + param_range * (2.0/3.0), # 中间点 2
        u_max                           # 终点
    ]
    
    # 4. 获取对应的 3D 点并格式化字符串
    pnt = [get_point_key(curve.Value(u),tol) for u in u_params]

    if pnt[0]<pnt[3]:
        return (pnt[0],pnt[1],pnt[2],pnt[3])
    else: 
        return (pnt[3],pnt[2],pnt[1],pnt[0])

from OCC.Core.TopoDS import TopoDS_Compound, topods
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRepTools import breptools
import os
import threading
import subprocess

import os
import signal
import subprocess
import time

def wait_for_file(file_path, timeout=60):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time >= timeout:
            return False
        time.sleep(0.5)  # 缩短检查间隔，提高响应速度
    return True
 
def run_cmd(cmd_string, timeout=60):
    print("Running：" + cmd_string)
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)

    if not wait_for_file("temp_brep_for_unify.res.brep", timeout):
        print(f"⚠ 进程超时 ({timeout}秒)，正在强制终止...")
        if is_windows:
            os.system(f"taskkill /PID {p.pid} /F /T")
            print(f"Killed process {p.pid}")
        else:
            try:
                pgid = os.getpgid(p.pid)
                os.killpg(pgid, signal.SIGKILL)
                print(f"Killed process group {pgid}")
            except ProcessLookupError:
                print(f"Process {p.pid} already terminated")
        
        # 确保进程被终止
        try:
            p.kill()
            p.wait(timeout=2)
        except:
            pass
    else:
        p.wait()
        print(f"Done {p.pid}")
    return 

def UnifySameDomain(shape,timeout=60):
    if os.path.exists("temp_brep_for_unify.res.brep"):
        os.remove("temp_brep_for_unify.res.brep")
    
    breptools.Write(shape,"temp_brep_for_unify.brep")

    run_cmd("python unify_face.py",timeout)

    if os.path.exists("temp_brep_for_unify.res.brep"):
        breptools.Read(shape,"temp_brep_for_unify.res.brep",BRep_Builder())
    if os.path.exists("temp_brep_for_unify.brep"):
        os.remove("temp_brep_for_unify.brep")
    return shape


def merge_faces_on_same_surface(faces, tol=1e-5, sewing_timeout=30):
    """
    尝试合并属于同一个 Surface 的相邻面。
    
    处理流程:
    1. 缝合 (Sewing): 根据顶点距离 tol，将独立的面的重合边在拓扑上连接起来。
    2. 统一 (Unify): 移除位于同一曲面上的面之间的内部边，实现合并。
    
    参数:
    faces: list[TopoDS_Face]
    tol: float, 判定重合的容差 (默认 1e-5)
    sewing_timeout: float, Sewing 操作超时时间(秒)，默认30秒
    
    返回:
    list[TopoDS_Face]: 合并后的面列表
    """
    if not faces:
        return []

    mydebug(f"正在尝试合并 {len(faces)} 个面 (Tolerance={tol})...")

    # ==========================================================================
    # 第一步：缝合 (Sewing) - 带超时保护
    # 目的：将几何上重合的边（根据顶点距离）转化为拓扑上共享的边。
    # ==========================================================================
    sewing_algo = BRepBuilderAPI_Sewing(tol)
    
    # 添加所有面到缝合算法中
    for face in faces:
        sewing_algo.Add(face)
    
    # 使用线程 + timeout 保护 Sewing 操作
    sewing_result = [None]  # 用列表存储结果，便于线程间传递
    sewing_exception = [None]
    
    def perform_sewing():
        try:
            sewing_algo.Perform()
            sewing_result[0] = sewing_algo.SewedShape()
        except Exception as e:
            sewing_exception[0] = e
    
    sewing_thread = threading.Thread(target=perform_sewing, daemon=True)
    sewing_thread.start()
    sewing_thread.join(timeout=sewing_timeout)
    
    if sewing_thread.is_alive():
        mydebug(f"⚠ Sewing 操作超时 ({sewing_timeout}秒)，跳过合并，返回原始面列表")
        return faces
    
    if sewing_exception[0]:
        mydebug(f"⚠ Sewing 操作失败: {sewing_exception[0]}，返回原始面列表")
        return faces
    
    sewed_shape = sewing_result[0]
    if not sewed_shape:
        mydebug(f"⚠ Sewing 未返回有效形状，返回原始面列表")
        return faces
    
    mydebug(f"Sewing done...")
    # 检查缝合结果
    # 如果缝合没有改变任何东西，sewed_shape 可能结构比较松散，但这不影响下一步
    
    # ==========================================================================
    # 第二步：统一同域 (Unify Same Domain)
    # 目的：检测共享边两侧的面是否属于同一个 Surface，如果是，则移除边并合并面。
    # ==========================================================================
    # 参数说明: Shape, UnifyEdges, UnifyFaces, ConcatBSplines
 
    try:
        result_shape = UnifySameDomain(sewed_shape,15)
    except Exception  as e:
        mydebug(f"合并失败: {e}")
        # 如果失败，尝试返回缝合后的面
        result_shape = sewed_shape

    
    mydebug(f"UnifySameDomain done...")
    # ==========================================================================
    # 第三步：提取结果
    # ==========================================================================
    merged_faces = []
    exp = TopExp_Explorer(result_shape, TopAbs_FACE)
    while exp.More():
        merged_faces.append(topods.Face(exp.Current()))
        exp.Next()
        
    mydebug(f"合并完成: {len(faces)} -> {len(merged_faces)} 个面。")
    
    return merged_faces

import math
from collections import defaultdict
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCC.Core.BRep import BRep_Tool
from OCC.Core.gp import gp_Pnt
import pulp

def solve_maximum_score_manifold(faces, scores, tol=1e-4):
    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    for i in faces:
        builder.Add(new_compound, i)
    #breptools.Write(new_compound, f"D:/all.brep")
    """
    从中选取一个子集，在满足流形约束(每条边最多2个面)的情况下最大化总得分。
    
    参数:
    faces: list[TopoDS_Face]
    scores: list[float], 对应的分数
    tol: float, 顶点重合容差
    method: str, 'auto', 'exact' (需pulp), 'greedy'
    
    返回:
    list[TopoDS_Face]: 选中的面列表
    """
    if len(faces) != len(scores):
        raise ValueError("faces 和 scores 的长度必须一致")
    
    # ==========================================================================
    # 1. 构建拓扑关系图 (Edge -> Face Indices)
    # ==========================================================================


    # 映射：边ID -> [拥有该边的面索引列表]
    edge_to_face_indices = defaultdict(list)
    
    mydebug(f"正在分析 {len(faces)} 个面的拓扑连接...")
    for i, face in enumerate(faces):
        exp = TopExp_Explorer(face, TopAbs_EDGE)
        while exp.More():
            edge = exp.Current()
            key = get_edge_key(edge,tol)
            if key:
                edge_to_face_indices[key].append(i)

            exp.Next()

    # 找出所有涉及冲突的边 (即连接了超过2个面的边)
    # 只有这些边需要建立约束
    conflict_edges = [f_indices for f_indices in edge_to_face_indices.values() if len(f_indices) > 2]
    link_edges = [f_indices for f_indices in edge_to_face_indices.values() if len(f_indices) == 2]
    border_edges = [f_indices[0] for f_indices in edge_to_face_indices.values() if len(f_indices) == 1]
    
    mydebug(f"发现 {len(conflict_edges)} 处潜在的非流形冲突(>2面共边)。")
    mydebug(f"发现 {len(link_edges)} 处流形边(=2面共边)。")
    mydebug(f"发现 {len(border_edges)} 处边界边(=1面边)。")


    selected_indices = []

    # ==========================================================================
    # 精确解 (Integer Linear Programming)
    # ==========================================================================

    mydebug("使用 ILP (整数规划) 求解全局最优...")
    
    # 1. 定义问题：最大化
    prob = pulp.LpProblem("MaxScoreManifold", pulp.LpMaximize)
    
    # 2. 定义变量：每个面是一个 0/1 变量
    # cat='Binary' 确保变量只能取 0 或 1
    vars = [pulp.LpVariable(f"x_{i}", cat='Binary') for i in range(len(faces))]
    evars = [pulp.LpVariable(f"e_{i}", cat='Binary') for i in range(len(conflict_edges))]
    lvars = [pulp.LpVariable(f"l_{i}", cat='Binary') for i in range(len(link_edges))]
    bvars = [pulp.LpVariable(f"b_{i}", cat='Binary') for i in range(len(border_edges))]
    
    # 3. 定义目标函数：Sum(score * x)
    prob += pulp.lpSum([scores[i] * vars[i] for i in range(len(faces))]) - pulp.lpSum(bvars)*2 + pulp.lpSum(lvars)+ pulp.lpSum(evars)
    
    # 4. 定义约束：对于每条冲突边，Sum(x) <= 2
    for ci in range(len(conflict_edges)):
        f_indices=conflict_edges[ci]
        prob +=  pulp.lpSum([vars[i] for i in f_indices])-2*evars[ci]==0
    
    for ci in range(len(link_edges)):
        f_indices=link_edges[ci]
        prob += pulp.lpSum([vars[i] for i in f_indices])-2*lvars[ci]==0

    for ci in range(len(border_edges)):
        f_indice=border_edges[ci]
        prob += vars[f_indice]-bvars[ci]==0

    # 5. 求解
    # msg=False 关闭求解器日志输出
    solver = pulp.COPT()
    if not solver.available():
        solver = pulp.PULP_CBC_CMD()
    solver.msg=False
    solver.timeLimit=60
    prob.solve(solver)
    
    status = pulp.LpStatus[prob.status]
    mydebug(f"求解器状态: {status}")
    
    if status == 'Optimal':
        conflict_faces=set()
        for i in conflict_edges:
            for j in i:
                conflict_faces.add(j)
        res = [int(pulp.value(var)+0.1) for var in vars]
        for i in range(len(faces)):
            if res[i]: # 浮点误差处理
                selected_indices.append(i)
        mydebug("selected:", [(i,selected_indices[i]) for i in range(len(selected_indices))])
        for ci in range(len(conflict_edges)):
            f_indices=conflict_edges[ci]
            builder = BRep_Builder()
            new_compound = TopoDS_Compound()
            builder.MakeCompound(new_compound)
            # for i in f_indices:
            #     builder.Add(new_compound, faces[i])
            #     breptools.Write(faces[i], f"D:/g-{ci}-{i}.brep")
            #breptools.Write(new_compound, f"D:/g-{ci}.brep")
            if sum([res[i] for i in f_indices])not in [0,2]:
                mydebug("conflict_edges: ", f_indices)
                mydebug([res[i] for i in f_indices])
        for ci in range(len(link_edges)):
            f_indices=link_edges[ci]
            if sum([res[i] for i in f_indices])not in [0,2]:
                mydebug("link_edges: ", f_indices)
                mydebug([res[i] for i in f_indices])
        for ci in range(len(border_edges)):
            f_indices=border_edges[ci]
            if res[f_indices]>0:
                mydebug("border_edges: ", f_indices)
                mydebug(f_indices)
                mydebug(res[f_indices])
    else:
        mydebug("求解失败，返回空集。")

   

    # ==========================================================================
    # 3. 构建结果
    # ==========================================================================
    result_faces = [faces[i] for i in selected_indices]
    total_score = sum([scores[i] for i in selected_indices])
    
    mydebug(f"筛选完成: 选中 {len(result_faces)} / {len(faces)} 个面。")
    mydebug(f"总得分: {total_score:.4f}")
    
    return result_faces
