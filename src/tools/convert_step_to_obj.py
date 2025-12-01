import os
import glob
import concurrent.futures
import random
import sys
import argparse

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
        print(f"OBJ file written to {obj_filename_with_index}")
        index += 1


def process_file(args):
    stepfile, input_folder, output_dir = args
    # 计算相对于输入文件夹的相对路径
    rel_path = os.path.relpath(stepfile, input_folder)
    # 在输出目录下创建相同的子目录结构
    objfile = os.path.join(output_dir, rel_path.replace('.step', '.obj'))
    
    print(f"Processing {stepfile} to {objfile}")
    if os.path.exists(objfile):
        return
    dir = os.path.dirname(objfile)
    os.makedirs(dir, exist_ok=True)
    # try:
    step_to_obj(stepfile, objfile)
    # except Exception as e:
    #     print(f"Error processing {stepfile}: {e}")

def main(input_folder, output_dir):
    stepfiles = glob.glob(os.path.join(input_folder, "*/*.step"))
    stepfiles = sorted(stepfiles)
    
    if not stepfiles:
        print("No STEP files found.")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数列表
    args_list = [(stepfile, input_folder, output_dir) for stepfile in stepfiles]
    
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        executor.map(process_file, args_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert STEP files to OBJ files')
    parser.add_argument('input_folder', type=str, help='Input folder containing STEP files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for OBJ files (default: same as input)')
    args = parser.parse_args()
    
    # 如果没有指定 output_dir，则使用 input_folder
    output_dir = args.output_dir if args.output_dir else args.input_folder
    
    main(args.input_folder, output_dir)
