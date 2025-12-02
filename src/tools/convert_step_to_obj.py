import os
import glob
import concurrent.futures
import random
import sys
import argparse
import signal
from functools import wraps

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

from icecream import ic

ic.disable()

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Function execution timed out")


def step_to_obj(step_filename, obj_filename):

    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())
   

    datas = []

    for index, solid in enumerate(solids):
        # try:
        ic(f'Processing solid {index:02d}...')
    # solid = solids[0]
        if len(list(solid.faces())) > 500:
            ic(f'Too many faces in the solid: {len(list(solid.faces()))}')
            raise ValueError("Too many faces in the solid.")
        solid = solid.topods_shape()
        solid = Solid(solid)
        
        solid = solid.scale_to_unit_box()

        try:
            graph = face_adjacency(solid, self_loops=True)
        except:
            raise ValueError("Face adjacency failed. The solid may be invalid.")
        data = []

                
        # except Exception as e:
        #     ic(f'Error processing solid {__idx:02d}: {e}')
        #     continue

        #     datas.append(data)


        shape = solid.topods_shape()
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        obj_filename_with_index = f"{obj_filename[:-4]}_{index}.obj"
        write_obj_file(shape, obj_filename_with_index)
        # print(f"OBJ file written to {obj_filename_with_index}")
        index += 1


def process_file(args):
    """Process a single STEP file with timeout protection"""
    stepfile, input_folder, output_dir, timeout_seconds = args
    
    # 计算相对于输入文件夹的相对路径
    rel_path = os.path.relpath(stepfile, input_folder)
    # 在输出目录下创建相同的子目录结构
    objfile = os.path.join(output_dir, rel_path.replace('.step', '.obj'))
    
    # 检查文件是否已存在
    if os.path.exists(objfile):
        print(f"[SKIP] {stepfile} already converted")
        return {'status': 'skipped', 'file': stepfile}
    
    # 创建输出目录
    dir = os.path.dirname(objfile)
    os.makedirs(dir, exist_ok=True)
    
    # 设置超时信号（仅在 Unix 系统上工作）
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        print(f"[START] Processing {stepfile} (timeout: {timeout_seconds}s)")
        step_to_obj(stepfile, objfile)
        
        # 取消超时
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        print(f"[SUCCESS] {stepfile} -> {objfile}")
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

def main(input_folder, output_dir, num_workers=4, timeout_seconds=300):
    """
    Main function to process STEP files with timeout protection
    
    Args:
        input_folder: Input directory containing STEP files
        output_dir: Output directory for OBJ files
        num_workers: Number of parallel worker processes
        timeout_seconds: Timeout in seconds for each file processing
    """
    stepfiles = glob.glob(os.path.join(input_folder, "*/*.step"))
    stepfiles = sorted(stepfiles)
    
    if not stepfiles:
        print("No STEP files found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Starting conversion with {num_workers} workers")
    print(f"Timeout per file: {timeout_seconds}s")
    print(f"Total files to process: {len(stepfiles)}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = [(stepfile, input_folder, output_dir, timeout_seconds) for stepfile in stepfiles]
    
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
        description='Convert STEP files to OBJ files with timeout protection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_folder', type=str, 
                        help='Input folder containing STEP files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for OBJ files (default: same as input)')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of parallel worker processes')
    parser.add_argument('--timeout', type=int, default=300, 
                        help='Timeout in seconds for each file (default: 300s = 5 minutes)')
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，则使用 input_folder
    output_dir = args.output_dir if args.output_dir else args.input_folder
    
    main(args.input_folder, output_dir, num_workers=args.workers, timeout_seconds=args.timeout)
