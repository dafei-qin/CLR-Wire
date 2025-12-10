import os
import glob
import concurrent.futures
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import signal
from functools import wraps

import trimesh
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Iterator
from OCC.Extend.DataExchange import write_obj_file
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
# from occwl import Solid
from occwl.solid import Solid
from occwl.uvgrid import uvgrid, ugrid
from occwl.graph import face_adjacency
from occwl.io import load_step, save_step
from occwl.compound import Compound
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box

from icecream import ic

from utils.surface import extract_mesh_from_face


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


        shape = solid.topods_shape()
        mesh = BRepMesh_IncrementalMesh(shape, 0.1)
        mesh.Perform()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        tris = []
        faces = []
        counter = 0
        while explorer.More():
            print('processing face', counter)
            face = explorer.Current()
            faces.append(face)
            # loc = TopLoc_Location()
            # triangulation = BRep_Tool.Triangulation(face, loc)
            vertices, triangles = extract_mesh_from_face(face)
            tris.append([vertices, triangles])
            
            counter += 1
            explorer.Next()
        obj_filename_with_index = f"{obj_filename[:-4]}_{index}.obj"
        write_obj_file(shape, obj_filename_with_index)

        for idx in range(len(tris)):
            vertices, triangles = tris[idx]
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(obj_filename_with_index.replace('.obj', f'_surface_{idx:03d}.obj'))
        # print(f"OBJ file written to {obj_filename_with_index}")
        index += 1


if __name__ == '__main__':
    step_filename = r"F:\WORK\CAD\CLR-Wire\assets\examples\00000084\00000084_9a051df11afa4fc5a1d667cd_step_000.step"
    obj_filename = r"F:\WORK\CAD\CLR-Wire\assets\test_export_obj_per_surface\00000084_9a051df11afa4fc5a1d667cd_step_000"
    os.makedirs(os.path.dirname(obj_filename), exist_ok=True)
    step_to_obj(step_filename, obj_filename)