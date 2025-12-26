import os
# Set environment variable to suppress warnings in all processes
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import glob
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp
import random
import sys
import argparse
import signal
import warnings
from functools import wraps
from pathlib import Path
import numpy as np
import json
import networkx as nx
import fpsample

# Suppress warnings BEFORE importing any OCC/occwl modules
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent.parent))

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Extend.DataExchange import write_obj_file
# from occwl import Solid
from occwl.solid import Solid
from occwl.uvgrid import uvgrid, ugrid
from occwl.graph import face_adjacency
from occwl.io import load_step, save_step
from occwl.compound import Compound
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepClass import BRepClass_FaceClassifier
# from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, 
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, 
    GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, 
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from icecream import ic
from logan_process_brep_data import BRepDataProcessor

ic.disable()

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Function execution timed out")


def sample_face_uv(face, num_samples=1000, debug=False):
    """
    Sample points on a face in UV space using random sampling and check validity
    
    Args:
        face: TopoDS_Face object
        num_samples: number of random samples to generate
        debug: whether to print debug information
    
    Returns:
        points: numpy array of shape (N, 3) with valid 3D points
        normals: numpy array of shape (N, 3) with corresponding normals
        masks: numpy array of shape (N,) with validity flags
    """
    points = []
    normals = []
    masks = []
    
    if debug:
        ic("Starting sample_face_uv with random sampling")
    
    # Get the surface from the face
    u_min, u_max, v_min, v_max = [face.uv_bounds().min_point()[0], face.uv_bounds().max_point()[0], face.uv_bounds().min_point()[1], face.uv_bounds().max_point()[1]]
    surface = BRep_Tool.Surface(face.topods_shape())

    # Generate random UV samples
    u_values = np.random.uniform(u_min, u_max, num_samples)
    v_values = np.random.uniform(v_min, v_max, num_samples)
    
    valid_count = 0
    invalid_count = 0
    w_max = 1e-17
    
    # Sample points
    for i in range(num_samples):
        u = u_values[i]
        v = v_values[i]
        
        # Create 2D point in UV space
        uv_pnt = gp_Pnt2d(u, v)
        
        # Check if point is valid using BRepClass_FaceClassifier
        classifier = BRepClass_FaceClassifier(face.topods_shape(), uv_pnt, 1e-6)
        state = classifier.State()
        
        props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)

        pnt = props.Value()
        point = np.array([pnt.X(), pnt.Y(), pnt.Z()])
        
        # Get normal vector
        if props.IsNormalDefined():
            normal = props.Normal()
            normal_vec = np.array([normal.X(), normal.Y(), normal.Z()])
            norm = np.linalg.norm(normal_vec)
            normal_vec = normal_vec / (norm + 1e-6)
        else:
            normal_vec = np.array([0, 0, 0])
        
        # Only keep points that are IN or ON the face
        if state == TopAbs_IN or state == TopAbs_ON:
            # Evaluate surface at (u, v)
            props = GeomLProp_SLProps(surface, u, v, 1, 1e-6)
            
            if props.IsNormalDefined():
                # Get 3D point
                pnt = props.Value()
                point = np.array([pnt.X(), pnt.Y(), pnt.Z()])
                
                # Get normal vector
                normal = props.Normal()
                normal_vec = np.array([normal.X(), normal.Y(), normal.Z()])
                
                # Normalize the normal vector
                norm = np.linalg.norm(normal_vec)
                if norm > 1e-10:
                    normal_vec = normal_vec / norm
                    
                    du = props.D1U()
                    dv = props.D1V()
                    jacobian = du.Crossed(dv).Magnitude()
                    w_max = max(w_max, jacobian)
                    valid = random.random() < jacobian / w_max

                    if valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
                    valid = False
                    if debug and invalid_count <= 5:
                        ic(f"Invalid normal at u={u:.4f}, v={v:.4f}, norm={norm}")
            else:
                invalid_count += 1
                valid = False
                if debug and invalid_count <= 5:
                    ic(f"Normal not defined at u={u:.4f}, v={v:.4f}")
        else:
            invalid_count += 1
            valid = False

        points.append(point)
        normals.append(normal_vec)
        masks.append(valid)
    
    if debug:
        ic(f"Valid points: {valid_count}, Invalid points: {invalid_count}")
    
    if len(points) == 0:
        if debug:
            ic("WARNING: No valid points sampled!")
        return np.array([]), np.array([]), np.array([])
    
    return np.array(points), np.array(normals), np.array(masks)


def save_ply(filename, points, normals):
    """
    Save points and normals to PLY file
    
    Args:
        filename: output PLY filename
        points: numpy array of shape (N, 3)
        normals: numpy array of shape (N, 3)
    """
    if len(points) == 0:
        print(f"Warning: No points to save for {filename}")
        return
    
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        
        # Write data
        for point, normal in zip(points, normals):
            f.write(f"{point[0]} {point[1]} {point[2]} ")
            f.write(f"{normal[0]} {normal[1]} {normal[2]}\n")
    
    print(f"Saved {len(points)} points to {filename}")


def strip_features_and_make_undirected(G: nx.DiGraph) -> nx.Graph:
    """
    删除所有节点和边的 attributes，只保留纯拓扑结构，
    并将 DiGraph 转为 Graph
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return H


def get_surface_type_name(surface_type):
    """
    将曲面类型枚举值转换为可读的名称
    
    Args:
        surface_type: GeomAbs_SurfaceType枚举值
    
    Returns:
        str: 曲面类型名称
    """
    type_mapping = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "BezierSurface",
        GeomAbs_BSplineSurface: "BSplineSurface",
        GeomAbs_SurfaceOfRevolution: "SurfaceOfRevolution",
        GeomAbs_SurfaceOfExtrusion: "SurfaceOfExtrusion",
        GeomAbs_OffsetSurface: "OffsetSurface",
        GeomAbs_OtherSurface: "OtherSurface"
    }
    return type_mapping.get(surface_type, f"Unknown({surface_type})")


def get_solid_surface_statistics(solid):
    """
    获取solid的曲面统计信息
    
    Args:
        solid: Solid对象（occwl.Solid）
    
    Returns:
        dict: 包含曲面类型统计的字典，格式为 {surface_type: count}
    """
    surface_stats = {}
    
    # 遍历solid的所有面
    for face in solid.faces():
        # 创建表面适配器
        adaptor = BRepAdaptor_Surface(face.topods_shape())
        # 获取表面类型
        surface_type = adaptor.GetType()
        # 转换为可读名称
        type_name = get_surface_type_name(surface_type)
        
        # 统计数量
        if type_name in surface_stats:
            surface_stats[type_name] += 1
        else:
            surface_stats[type_name] = 1
    
    return surface_stats


def are_solids_identical(solid1, solid2, verbose=False):
    """
    判断两个solid是否一致
    
    判断标准：
    1. 两个solid的曲面数量一致
    2. 两个solid的每种不同种类曲面的数量一致
    
    Args:
        solid1: 第一个Solid对象（occwl.Solid）
        solid2: 第二个Solid对象（occwl.Solid）
        verbose: 是否输出详细信息
    
    Returns:
        bool: 如果两个solid一致返回True，否则返回False
    """
    # 获取两个solid的曲面统计信息
    stats1 = get_solid_surface_statistics(solid1)
    stats2 = get_solid_surface_statistics(solid2)
    
    # 计算总曲面数量
    total_faces1 = sum(stats1.values())
    total_faces2 = sum(stats2.values())
    
    if verbose:
        print(f"Solid 1: {total_faces1} faces, distribution: {stats1}")
        print(f"Solid 2: {total_faces2} faces, distribution: {stats2}")
    
    # 检查曲面总数是否相同
    if total_faces1 != total_faces2:
        if verbose:
            print(f"Different number of faces: {total_faces1} vs {total_faces2}")
        return False
    
    # 检查每种曲面类型的数量是否相同
    if stats1 != stats2:
        if verbose:
            print(f"Different surface type distribution")
            # 找出差异
            all_types = set(stats1.keys()) | set(stats2.keys())
            for stype in sorted(all_types):
                count1 = stats1.get(stype, 0)
                count2 = stats2.get(stype, 0)
                if count1 != count2:
                    print(f"  {stype}: {count1} vs {count2}")
        return False
    
    if verbose:
        print("Solids are identical")
    
    return True


def filter_unique_solids(solids, verbose=False):
    """
    从solids列表中过滤出唯一的solid
    
    Args:
        solids: Solid对象列表
        verbose: 是否输出详细信息
    
    Returns:
        list: 包含唯一solid的列表
        list: 每个唯一solid对应的原始索引列表
    """
    if len(solids) == 0:
        return [], []
    
    unique_solids = []
    unique_indices = []
    duplicate_groups = []  # 记录每个unique solid对应的重复索引组
    
    for i, solid in enumerate(solids):
        is_duplicate = False
        
        # 与已有的unique solids比较
        for j, unique_solid in enumerate(unique_solids):
            if are_solids_identical(solid, unique_solid, verbose=False):
                is_duplicate = True
                duplicate_groups[j].append(i)
                if verbose:
                    print(f"Solid {i} is identical to solid {unique_indices[j]}")
                break
        
        # 如果不是重复的，添加到unique列表
        if not is_duplicate:
            unique_solids.append(solid)
            unique_indices.append(i)
            duplicate_groups.append([i])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Deduplication Summary:")
        print(f"{'='*60}")
        print(f"Original solids count: {len(solids)}")
        print(f"Unique solids count:   {len(unique_solids)}")
        print(f"Duplicates removed:    {len(solids) - len(unique_solids)}")
        print(f"{'='*60}")
        
        for i, (idx, group) in enumerate(zip(unique_indices, duplicate_groups)):
            if len(group) > 1:
                print(f"Unique solid {i} (original index {idx}) has {len(group)-1} duplicate(s): {group[1:]}")
        print(f"{'='*60}\n")
    
    return unique_solids, unique_indices


def step_to_pointcloud(step_filename, ply_filename, num_samples=1000, debug=False, fps=True, num_fps=81920):
    """
    Convert STEP file to point cloud with normals
    
    Args:
        step_filename: input STEP file
        ply_filename: output NPZ file (base name)
        num_samples: number of random samples per face
        debug: whether to print debug information
    """
    # Suppress warnings in this function too
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    print(f"\n{'='*60}")
    print(f"Processing: {step_filename}")
    print(f"{'='*60}\n")
    
    # Try to load STEP file with error handling
    # OCC library may crash (segfault) on corrupted files, which can't be caught
    # But we can try to catch Python exceptions that might be raised
    try:
        solids, attributes = Compound.load_step_with_attributes(step_filename)
        solids = list(solids.solids())
    except Exception as e:
        error_msg = f"Failed to load STEP file {step_filename}: {e}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    if len(solids) == 0:
        error_msg = f"No solids found in STEP file {step_filename}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    print(f"Number of solids: {len(solids)}")

    processor = BRepDataProcessor()
    
    # Pre-filter solids with too many faces before deduplication
    solids_filtered = []
    for i, solid in enumerate(solids):
        num_faces = len(list(solid.faces()))
        if num_faces > 500:
            print(f"[SKIP] Solid {i} has too many faces ({num_faces} > 500), skipping...")
        else:
            solids_filtered.append(solid)
    
    solids = solids_filtered
    print(f"Number of solids after face count filtering: {len(solids)}")
    
    # Filter out duplicate solids
    unique_solids, unique_indices = filter_unique_solids(solids, verbose=True)
    
    # Only process unique solids
    for idx, (original_index, solid) in enumerate(zip(unique_indices, unique_solids)):
        print(f"\n--- Processing unique solid {idx} (original index: {original_index}) ---")
        
        num_faces = len(list(solid.faces()))
        print(f"Number of faces: {num_faces}")
        
        solid = solid.topods_shape()
        solid = Solid(solid)
        
        # Scale to unit box
        print("Scaling to unit box...")
        solid = solid.scale_to_unit_box()

        try:
            print("Building face adjacency graph...")
            graph = face_adjacency(solid, self_loops=True)
            print(f"Graph nodes: {len(graph.nodes())}, edges: {len(graph.edges())}")
        except Exception as e:
            print(f"Face adjacency failed: {e}")
            raise ValueError("Face adjacency failed. The solid may be invalid.")

        jsons_data = processor.tokenize_cad_data_preload(graph)
        
        # Collect points and normals from all faces
        all_points = []
        all_normals = []
        all_masks = []
        
        for face_idx in graph.nodes():
            face = graph.nodes[face_idx]["face"]
            
            # 累积采样当前 face 的有效点，避免每次重采样丢弃之前的结果
            accumulated_points = []
            accumulated_normals = []
            accumulated_valid = 0

            tried_runs = 0
            num_samples_current = num_samples
            while tried_runs < 5 and accumulated_valid < num_samples:
                points_batch, normals_batch, masks_batch = sample_face_uv(
                    face, num_samples=num_samples_current, debug=debug
                )
                masks_batch = masks_batch.astype(bool)

                if masks_batch.any():
                    accumulated_points.append(points_batch[masks_batch])
                    accumulated_normals.append(normals_batch[masks_batch])
                    accumulated_valid += masks_batch.sum()

                if accumulated_valid < num_samples:
                    num_samples_current = num_samples_current * 2
                    tried_runs += 1
                else:
                    break

            if accumulated_points:
                points = np.concatenate(accumulated_points, axis=0)
                normals = np.concatenate(accumulated_normals, axis=0)
                masks = np.ones(points.shape[0], dtype=bool)
            else:
                # 该面完全没有有效点
                points = np.zeros((0, 3), dtype=np.float32)
                normals = np.zeros((0, 3), dtype=np.float32)
                masks = np.zeros((0,), dtype=bool)
            
            all_points.append(points.astype(np.float32))
            all_normals.append(normals.astype(np.float32))
            all_masks.append(masks.astype(bool))

        graph_undirected = strip_features_and_make_undirected(graph)

        if fps:
            all_points_valid = [p[m.astype(bool)] for p, m in zip(all_points, all_masks)]
            all_normals_valid = [n[m.astype(bool)] for n, m in zip(all_normals, all_masks)]
            all_points_valid = np.concatenate(all_points_valid, axis=0)
            all_normals_valid = np.concatenate(all_normals_valid, axis=0)
            print('Num valid points total: ', len(all_points_valid))
            # 如果有效点太少，直接跳过该 solid 的 FPS 采样，避免异常
            if len(all_points_valid) < num_fps * 2:
                print('[WARN] Num valid points too few, skipping FPS sampling for this solid')
                raise ValueError('Num valid points too few, skipping FPS sampling for this solid')
            else:
                fps_idx = fpsample.bucket_fps_kdtree_sampling(all_points_valid, num_fps)
                all_points = all_points_valid[fps_idx]
                all_normals = all_normals_valid[fps_idx]
                all_masks = np.ones_like(all_points, dtype=bool)
        
        # Save to NPZ file with graph data (use idx for unique solid indexing)
        save_name = ply_filename.replace('.npz', f'_{idx:03d}.npz')
        np.savez(save_name, 
                 points=np.array(all_points, dtype=object), 
                 normals=np.array(all_normals, dtype=object), 
                 masks=np.array(all_masks, dtype=object), 
                 graph_nodes=list(graph_undirected.nodes()), 
                 graph_edges=list(graph_undirected.edges()))
        
        # Save JSON data
        json_name = save_name.replace('.npz', '.json')
        json.dump(jsons_data, open(json_name, 'w'), ensure_ascii=False, indent=2)
        
        print(f"\n✓ Successfully saved {len(all_points)} surfaces to {save_name}")
        print(f"✓ Successfully saved JSON data to {json_name}")


def process_file(args):
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    """Process a single STEP file with timeout protection"""
    stepfile, input_folder, output_dir, timeout_seconds, num_samples, fps, num_fps = args
    
    # Wrap everything in try-except to catch any unexpected errors
    # This is important because OCC library crashes (segfaults) cannot be caught,
    # but we can at least handle Python exceptions gracefully
    try:
        # 计算相对于输入文件夹的相对路径
        rel_path = os.path.relpath(stepfile, input_folder)
        # 在输出目录下创建相同的子目录结构
        npzfile = os.path.join(output_dir, rel_path.replace('.step', '.npz'))
        
        # 获取输出文件的最小子文件夹路径
        output_subdir = os.path.dirname(npzfile)
        
        # 检查输出文件夹是否已存在且包含处理结果
        if os.path.exists(output_subdir):
            # 检查文件夹中是否有 .npz 文件（表示已经处理过）
            existing_npz_files = glob.glob(os.path.join(output_subdir, '*.npz'))
            if existing_npz_files:
                print(f"[SKIP] {stepfile} - output folder already exists with {len(existing_npz_files)} npz file(s)")
                return {'status': 'skipped', 'file': stepfile}
        
        # 创建输出目录
        os.makedirs(output_subdir, exist_ok=True)
        
        # 设置超时信号（仅在 Unix 系统上工作）
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            print(f"[START] Processing {stepfile} (timeout: {timeout_seconds}s)")
            step_to_pointcloud(stepfile, npzfile, num_samples=num_samples, debug=False, fps=fps, num_fps=num_fps)
            
            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            print(f"[SUCCESS] {stepfile} -> {npzfile}")
            return {'status': 'success', 'file': stepfile}
            
        except TimeoutError:
            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            print(f"[TIMEOUT] {stepfile} (exceeded {timeout_seconds}s)")
            return {'status': 'timeout', 'file': stepfile}
            
        except Exception as e:
            # 取消超时
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            error_msg = str(e)
            print(f"[ERROR] {stepfile}: {error_msg}")
            return {'status': 'error', 'file': stepfile, 'error': error_msg}
            
    except Exception as e:
        # Catch any unexpected errors at the outer level
        error_msg = f"Unexpected error in process_file: {e}"
        print(f"[CRITICAL ERROR] {stepfile}: {error_msg}")
        return {'status': 'error', 'file': stepfile, 'error': error_msg}


def _child_process_wrapper(args, result_queue):
    """
    Wrapper function to run process_file in a child process.
    This allows complete isolation - if the child process crashes (segfault),
    it won't affect other processes.
    """
    try:
        result = process_file(args)
        result_queue.put(result)
    except Exception as e:
        stepfile = args[0]
        result_queue.put({
            'status': 'error',
            'file': stepfile,
            'error': f"Exception in child process: {str(e)}"
        })


def process_file_with_isolation(args):
    """
    Process a single file using an isolated subprocess.
    This ensures that if OCC library causes a segfault, it only affects this one task.
    
    Args:
        args: tuple of (stepfile, input_folder, output_dir, timeout_seconds, num_samples, fps, num_fps)
    
    Returns:
        dict: Result dictionary with status and file info
    """
    stepfile, input_folder, output_dir, timeout_seconds, num_samples, fps, num_fps = args
    
    # Create a queue to receive results from child process
    result_queue = mp.Queue()
    
    # Create and start child process
    proc = mp.Process(target=_child_process_wrapper, args=(args, result_queue))
    proc.start()
    
    # Wait for process to complete (with timeout)
    # Use a slightly longer timeout to account for process startup overhead
    proc.join(timeout=timeout_seconds + 10)
    
    # Check if process is still alive (timeout occurred)
    if proc.is_alive():
        # Process timed out, kill it
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {
            'status': 'timeout',
            'file': stepfile,
            'error': f'Process timeout after {timeout_seconds}s'
        }
    
    # Check exit code
    if proc.exitcode is None:
        return {
            'status': 'error',
            'file': stepfile,
            'error': 'Child process exitcode is None'
        }
    
    if proc.exitcode != 0:
        # Process crashed (likely segfault from OCC library)
        # Try to get error message from queue
        try:
            result = result_queue.get(timeout=1)
            return result
        except:
            return {
                'status': 'error',
                'file': stepfile,
                'error': f'Child process crashed with exitcode {proc.exitcode} (likely segfault from OCC library)'
            }
    
    # Normal exit - get result from queue
    try:
        result = result_queue.get(timeout=5)
        return result
    except Exception as e:
        return {
            'status': 'error',
            'file': stepfile,
            'error': f'No result from child process: {str(e)}'
        }


def main(input_folder, output_dir, num_workers=4, timeout_seconds=300, num_samples=1000, fps=True, num_fps=81920):
    """
    Main function to process STEP files with timeout protection
    
    Args:
        input_folder: Input directory containing STEP files
        output_dir: Output directory for NPZ files
        num_workers: Number of parallel worker processes
        timeout_seconds: Timeout in seconds for each file processing
        num_samples: Number of random samples per face
    """
    # Set multiprocessing start method for better compatibility
    # 'fork' is faster on Unix systems, 'spawn' is safer but slower
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass
    
    print('processing ', input_folder)
    stepfiles = glob.glob(os.path.join(input_folder, "*.step"))
    if len(stepfiles) == 0:
        stepfiles = glob.glob(os.path.join(input_folder, "*", "*.step"))
    stepfiles = sorted(stepfiles)
    
    if not stepfiles:
        print("No STEP files found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting conversion with {num_workers} workers")
    print(f"Timeout per file: {timeout_seconds}s")
    print(f"Random samples per face: {num_samples}")
    print(f"Total files to process: {len(stepfiles)}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = [(stepfile, input_folder, output_dir, timeout_seconds, num_samples, fps, num_fps) for stepfile in stepfiles]
    
    # 统计信息
    stats = {
        'success': 0,
        'error': 0,
        'timeout': 0,
        'skipped': 0,
        'timeout_files': [],
        'error_files': []
    }
    
    # 使用 ThreadPoolExecutor + 独立子进程处理文件
    # 这种方式可以完全隔离每个任务：如果一个进程崩溃（segfault），不会影响其他任务
    # 每个任务在独立的子进程中运行，通过 ThreadPoolExecutor 来管理并发
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务（每个任务内部会创建独立的子进程）
        future_to_file = {executor.submit(process_file_with_isolation, args): args[0] for args in args_list}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_file):
            stepfile = future_to_file[future]
            try:
                result = future.result()
                if result:
                    status = result['status']
                    stats[status] += 1
                    
                    if status == 'timeout':
                        stats['timeout_files'].append(result['file'])
                    elif status == 'error':
                        stats['error_files'].append(result['file'])
                        
            except Exception as e:
                print(f"[CRITICAL ERROR] Unexpected error for {stepfile}: {e}")
                stats['error'] += 1
                stats['error_files'].append(stepfile)
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"{'='*60}")
    print(f"Total files:       {len(stepfiles)}")
    print(f"✓ Success:         {stats['success']}")
    print(f"⊘ Skipped:         {stats['skipped']}")
    print(f"⏱ Timeout:          {stats['timeout']}")
    print(f"✗ Error:           {stats['error']}")
    print(f"{'='*60}\n")
    
    # 保存超时和错误文件列表
    if stats['timeout_files']:
        timeout_log = os.path.join(output_dir, 'timeout_files.txt')
        with open(timeout_log, 'w') as f:
            f.write('\n'.join(stats['timeout_files']))
        print(f"Timeout files saved to: {timeout_log}")
    
    if stats['error_files']:
        error_log = os.path.join(output_dir, 'error_files.txt')
        with open(error_log, 'w') as f:
            f.write('\n'.join(stats['error_files']))
        print(f"Error files saved to: {error_log}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert STEP files to point clouds (NPZ) with random UV sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_folder', type=str, 
                        help='Input folder containing STEP files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for NPZ files (default: same as input)')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel worker processes')
    parser.add_argument('--timeout', type=int, default=300, 
                        help='Timeout in seconds for each file (default: 300s = 5 minutes)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of random samples per face')
    parser.add_argument('--fps', type=bool, default=True,
                        help='Whether to use FPS sampling')
    parser.add_argument('--num_fps', type=int, default=81920,
                        help='Number of FPS samples')
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，则使用 input_folder
    output_dir = args.output_dir if args.output_dir else args.input_folder
    
    main(args.input_folder, output_dir, num_workers=args.workers, 
         timeout_seconds=args.timeout, num_samples=args.num_samples, fps=args.fps, num_fps=args.num_fps)
