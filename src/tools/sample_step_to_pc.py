import os
# Set environment variable to suppress warnings in all processes
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import glob
import concurrent.futures
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


def step_to_pointcloud(step_filename, ply_filename, num_samples=1000, debug=False):
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
    
    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())
    print(f"Number of solids: {len(solids)}")

    processor = BRepDataProcessor()

    for index, solid in enumerate(solids):
        print(f"\n--- Processing solid {index} ---")
        
        num_faces = len(list(solid.faces()))
        print(f"Number of faces: {num_faces}")
        
        if num_faces > 500:
            print(f'Too many faces in the solid: {num_faces}')
            raise ValueError("Too many faces in the solid.")
        
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
            
            # Retry mechanism to ensure enough valid samples
            tried_runs = 0
            num_samples_current = num_samples
            while tried_runs < 10:
                points, normals, masks = sample_face_uv(face, num_samples=num_samples_current, debug=debug)
                if masks.sum() < num_samples:
                    num_samples_current = num_samples_current * 2
                    tried_runs += 1
                else:
                    break
            
            all_points.append(points.astype(np.float32))
            all_normals.append(normals.astype(np.float32))
            all_masks.append(masks.astype(bool))

        graph_undirected = strip_features_and_make_undirected(graph)
        
        # Save to NPZ file with graph data
        save_name = ply_filename.replace('.npz', f'_{index:03d}.npz')
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
    stepfile, input_folder, output_dir, timeout_seconds, num_samples = args
    
    # 计算相对于输入文件夹的相对路径
    rel_path = os.path.relpath(stepfile, input_folder)
    # 在输出目录下创建相同的子目录结构
    npzfile = os.path.join(output_dir, rel_path.replace('.step', '.npz'))
    
    # 检查文件是否已存在
    if os.path.exists(npzfile):
        print(f"[SKIP] {stepfile} already converted")
        return {'status': 'skipped', 'file': stepfile}
    
    # 创建输出目录
    dir = os.path.dirname(npzfile)
    os.makedirs(dir, exist_ok=True)
    
    # 设置超时信号（仅在 Unix 系统上工作）
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        print(f"[START] Processing {stepfile} (timeout: {timeout_seconds}s)")
        step_to_pointcloud(stepfile, npzfile, num_samples=num_samples, debug=False)
        
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
        print(f"[ERROR] {stepfile}: {e}")
        return {'status': 'error', 'file': stepfile, 'error': str(e)}

def main(input_folder, output_dir, num_workers=4, timeout_seconds=300, num_samples=1000):
    """
    Main function to process STEP files with timeout protection
    
    Args:
        input_folder: Input directory containing STEP files
        output_dir: Output directory for NPZ files
        num_workers: Number of parallel worker processes
        timeout_seconds: Timeout in seconds for each file processing
        num_samples: Number of random samples per face
    """
    stepfiles = glob.glob(os.path.join(input_folder, "*/*.step"))
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
    args_list = [(stepfile, input_folder, output_dir, timeout_seconds, num_samples) for stepfile in stepfiles]
    
    # 统计信息
    stats = {
        'success': 0,
        'error': 0,
        'timeout': 0,
        'skipped': 0,
        'timeout_files': [],
        'error_files': []
    }
    
    # 使用 ProcessPoolExecutor 处理文件
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_file, args): args[0] for args in args_list}
        
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_file):
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
                stepfile = future_to_file[future]
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
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，则使用 input_folder
    output_dir = args.output_dir if args.output_dir else args.input_folder
    
    main(args.input_folder, output_dir, num_workers=args.workers, 
         timeout_seconds=args.timeout, num_samples=args.num_samples)
