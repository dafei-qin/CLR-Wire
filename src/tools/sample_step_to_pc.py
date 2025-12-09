import os
import glob
import concurrent.futures
import random
import sys
import argparse
import signal
from functools import wraps
import numpy as np

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
from OCC.Core.BRepTools import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON

from icecream import ic

ic.disable()

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Function execution timed out")


def sample_face_uv(face, nu=50, nv=50):
    """
    Sample points on a face in UV space and check validity
    
    Args:
        face: TopoDS_Face object
        nu: number of samples in u direction
        nv: number of samples in v direction
    
    Returns:
        points: numpy array of shape (N, 3) with valid 3D points
        normals: numpy array of shape (N, 3) with corresponding normals
    """
    points = []
    normals = []
    
    # Get the surface from the face
    surf_adaptor = BRep_Tool(face)
    surface = surf_adaptor.Surface()
    
    # Get UV bounds
    u_min = surf_adaptor.FirstUParameter()
    u_max = surf_adaptor.LastUParameter()
    v_min = surf_adaptor.FirstVParameter()
    v_max = surf_adaptor.LastVParameter()
    
    # Create UV grid
    u_values = np.linspace(u_min, u_max, nu)
    v_values = np.linspace(v_min, v_max, nv)
    
    # Sample points
    for u in u_values:
        for v in v_values:
            # Create 2D point in UV space
            uv_pnt = gp_Pnt2d(u, v)
            
            # Check if point is valid using BRepClass_FaceClassifier
            classifier = BRepClass_FaceClassifier(face, uv_pnt, 1e-6)
            state = classifier.State()
            
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
                        
                        points.append(point)
                        normals.append(normal_vec)
    
    if len(points) == 0:
        return np.array([]), np.array([])
    
    return np.array(points), np.array(normals)


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


def step_to_pointcloud(step_filename, ply_filename, nu=50, nv=50):
    """
    Convert STEP file to point cloud with normals
    
    Args:
        step_filename: input STEP file
        ply_filename: output PLY file (base name)
        nu: number of UV samples in u direction
        nv: number of UV samples in v direction
    """
    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())

    for index, solid in enumerate(solids):
        ic(f'Processing solid {index:02d}...')
        
        if len(list(solid.faces())) > 500:
            ic(f'Too many faces in the solid: {len(list(solid.faces()))}')
            raise ValueError("Too many faces in the solid.")
        
        solid = solid.topods_shape()
        solid = Solid(solid)
        
        # Scale to unit box
        solid = solid.scale_to_unit_box()

        try:
            graph = face_adjacency(solid, self_loops=True)
        except:
            raise ValueError("Face adjacency failed. The solid may be invalid.")
        
        # Collect points and normals from all faces
        all_points = []
        all_normals = []
        
        for face in solid.faces():
            points, normals = sample_face_uv(face, nu=nu, nv=nv)
            if len(points) > 0:
                all_points.append(points)
                all_normals.append(normals)
        
        if len(all_points) == 0:
            print(f"Warning: No valid points sampled for solid {index}")
            continue
        
        # Concatenate all points and normals
        all_points = np.vstack(all_points)
        all_normals = np.vstack(all_normals)
        
        # Save to PLY file
        ply_filename_with_index = f"{ply_filename[:-4]}_{index}.ply"
        save_ply(ply_filename_with_index, all_points, all_normals)
        print(f"Saved {len(all_points)} points to {ply_filename_with_index}")


def process_file(args):
    """Process a single STEP file with timeout protection"""
    stepfile, input_folder, output_dir, timeout_seconds, nu, nv = args
    
    # 计算相对于输入文件夹的相对路径
    rel_path = os.path.relpath(stepfile, input_folder)
    # 在输出目录下创建相同的子目录结构
    plyfile = os.path.join(output_dir, rel_path.replace('.step', '.ply'))
    
    # 检查文件是否已存在
    if os.path.exists(plyfile):
        print(f"[SKIP] {stepfile} already converted")
        return {'status': 'skipped', 'file': stepfile}
    
    # 创建输出目录
    dir = os.path.dirname(plyfile)
    os.makedirs(dir, exist_ok=True)
    
    # 设置超时信号（仅在 Unix 系统上工作）
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        print(f"[START] Processing {stepfile} (timeout: {timeout_seconds}s)")
        step_to_pointcloud(stepfile, plyfile, nu=nu, nv=nv)
        
        # 取消超时
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        print(f"[SUCCESS] {stepfile} -> {plyfile}")
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

def main(input_folder, output_dir, num_workers=4, timeout_seconds=300, nu=50, nv=50):
    """
    Main function to process STEP files with timeout protection
    
    Args:
        input_folder: Input directory containing STEP files
        output_dir: Output directory for PLY files
        num_workers: Number of parallel worker processes
        timeout_seconds: Timeout in seconds for each file processing
        nu: Number of UV samples in u direction
        nv: Number of UV samples in v direction
    """
    stepfiles = glob.glob(os.path.join(input_folder, "*/*.step"))
    stepfiles = sorted(stepfiles)
    
    if not stepfiles:
        print("No STEP files found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting conversion with {num_workers} workers")
    print(f"Timeout per file: {timeout_seconds}s")
    print(f"UV sampling: {nu} x {nv}")
    print(f"Total files to process: {len(stepfiles)}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = [(stepfile, input_folder, output_dir, timeout_seconds, nu, nv) for stepfile in stepfiles]
    
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
        description='Convert STEP files to point clouds (PLY) with UV sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_folder', type=str, 
                        help='Input folder containing STEP files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for PLY files (default: same as input)')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel worker processes')
    parser.add_argument('--timeout', type=int, default=300, 
                        help='Timeout in seconds for each file (default: 300s = 5 minutes)')
    parser.add_argument('--nu', type=int, default=50,
                        help='Number of UV samples in u direction')
    parser.add_argument('--nv', type=int, default=50,
                        help='Number of UV samples in v direction')
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，则使用 input_folder
    output_dir = args.output_dir if args.output_dir else args.input_folder
    
    main(args.input_folder, output_dir, num_workers=args.workers, 
         timeout_seconds=args.timeout, nu=args.nu, nv=args.nv)
