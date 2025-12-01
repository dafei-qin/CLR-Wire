import os
import glob
import concurrent.futures
import sys
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.DataExchange import write_ply_file
from occwl.compound import Compound
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import trimesh
import open3d as o3d
import numpy as np

def convert_step_to_ply(step_file, ply_file):
    # Read the STEP file
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(step_file)
    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()

    stl_file = ply_file.replace('.ply', '.stl')
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, stl_file)

    mesh = trimesh.load_mesh(stl_file)
    points, face_indices = trimesh.sample.sample_surface(mesh, 409600)
    normals = mesh.face_normals[face_indices]

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(ply_file, point_cloud)

def convert_step_to_numpy(step_file, numpy_file, normalize=True, num_points=409600):
    step_reader = STEPControl_Reader()
    step_reader.ReadFile(step_file)
    step_reader.TransferRoots()
    shape = step_reader.OneShape()

    mesh = BRepMesh_IncrementalMesh(shape, 0.1)
    mesh.Perform()

    stl_file = numpy_file.replace('.npz', '.stl')
    stl_writer = StlAPI_Writer()
    stl_writer.Write(shape, stl_file)

    mesh = trimesh.load_mesh(stl_file)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    points = np.array(points)

    if normalize:
        max_corner = np.max(points, axis=0)
        min_corner = np.min(points, axis=0)
        center = (max_corner + min_corner) / 2
        points = (points - center) / np.max((max_corner - min_corner))
        points = points * 2 
    normals = mesh.face_normals[face_indices]
    normals = np.array(normals)
    # print(type(points))
    np.savez(numpy_file, points=points, normals=normals, loc=np.array([0, 0, 0], dtype=np.float32), scale=np.float32(1.0))
    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.normals = o3d.utility.Vector3dVector(normals)

    o3d.io.write_point_cloud(numpy_file.replace('.npz', '.ply'), point_cloud)

def process_file(stepfile):
    objfile = stepfile.replace('/abc/', '/ply/').replace('.step', '.ply')
    if os.path.exists(objfile):
        return
    dir = os.path.dirname(objfile)
    os.makedirs(dir, exist_ok=True)
    try:
        convert_step_to_ply(stepfile, objfile)
        # convert_step_to_numpy(stepfile, objfile.replace('.ply', '.npz'))
    except Exception as e:
        print(f"Error processing {stepfile}: {e}")

def main(folder):
    stepfiles = sorted(glob.glob(os.path.join(folder, "*/*.step")))

    if not stepfiles:
        print("No STEP files found.")
        return
        
    with concurrent.futures.ThreadPoolExecutor(100) as executor:
        executor.map(process_file, stepfiles)

if __name__ == '__main__':
    main(sys.argv[1])
