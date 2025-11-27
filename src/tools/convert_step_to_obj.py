import os
import glob
import concurrent.futures
import random
import sys

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Extend.DataExchange import write_obj_file
# from occwl import Solid
from occwl.io import load_step


def step_to_obj(step_filename, obj_filename):
    solids = load_step(step_filename)
    for index, solid in enumerate(solids):
        shape = solid.topods_shape()

        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        obj_filename_with_index = f"{obj_filename[:-4]}_{index}.obj"
        write_obj_file(shape, obj_filename_with_index)
        print(f"OBJ file written to {obj_filename_with_index}")
        index += 1


def process_file(stepfile):
    objfile = stepfile.replace('.step', '.obj')
    print(f"Processing {stepfile} to {objfile}")
    if os.path.exists(objfile):
        return
    dir = os.path.dirname(objfile)
    os.makedirs(dir, exist_ok=True)
    try:
        step_to_obj(stepfile, objfile)
    except Exception as e:
        print(f"Error processing {stepfile}: {e}")

def main(folder):
    stepfiles = glob.glob(os.path.join(folder, "*/*.step"))
    stepfiles = sorted(stepfiles)
    
    if not stepfiles:
        print("No STEP files found.")
        return
    with concurrent.futures.ProcessPoolExecutor(4) as executor:
        executor.map(process_file, stepfiles)

if __name__ == '__main__':
    main(sys.argv[1])
