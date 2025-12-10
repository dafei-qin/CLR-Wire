import os
# Set environment variable to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import argparse
import numpy as np
import sys
import warnings
from pathlib import Path

# Suppress warnings BEFORE importing occwl
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent.parent))

from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.compound import Compound
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.BRepClass import BRepClass_FaceClassifier
import json
from icecream import ic
from logan_process_brep_data import BRepDataProcessor
import random
import networkx as nx
# Enable icecream for debugging
ic.enable()



def sample_face_uv(face, num_samples=2000, debug=True):
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

def step_to_pointcloud(step_filename, ply_filename, num_samples=2000, debug=True):
    """
    Convert STEP file to point cloud with normals (single-threaded debug version)
    
    Args:
        step_filename: input STEP file
        ply_filename: output PLY file (base name)
        num_samples: number of random samples per face
        debug: whether to print debug information
    """
    # Suppress warnings in this function too
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    print(f"\n{'='*60}")
    print(f"Processing: {step_filename}")
    print(f"{'='*60}\n")
    
    ic("Loading STEP file...")
    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())
    ic(f"Number of solids: {len(solids)}")

    processor = BRepDataProcessor()


    for index, solid in enumerate(solids):
        print(f"\n--- Processing solid {index} ---")
        ic(f'Solid {index:02d}')
        
        num_faces = len(list(solid.faces()))
        ic(f"Number of faces: {num_faces}")
        
        if num_faces > 500:
            ic(f'Too many faces in the solid: {num_faces}')
            raise ValueError("Too many faces in the solid.")
        
        solid = solid.topods_shape()
        solid = Solid(solid)
        
        # Scale to unit box
        print("Scaling to unit box...")
        solid = solid.scale_to_unit_box()

        try:
            print("Building face adjacency graph...")
            graph = face_adjacency(solid, self_loops=True)
            ic(f"Graph nodes: {len(graph.nodes())}, edges: {len(graph.edges())}")
        except Exception as e:
            ic(f"Face adjacency failed: {e}")
            raise ValueError("Face adjacency failed. The solid may be invalid.")

        jsons_data = processor.tokenize_cad_data_preload(graph)
        # Collect points and normals from all faces
        all_points = []
        all_normals = []
        all_masks = []
        # faces_list = list(solid.faces())
        for face_idx in graph.nodes():

            # print(f"\n  Face {face_idx}/{len(faces_list)-1}")

            face = graph.nodes[face_idx]["face"]
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
        save_name = ply_filename.replace('.npz', f'_{index:03d}.npz')
        np.savez(save_name, points=all_points, normals=all_normals, masks=all_masks, graph_nodes=list(graph_undirected.nodes()), graph_edges=list(graph_undirected.edges()))
        json.dump(jsons_data, open(save_name.replace('.npz', '.json'), 'w'), ensure_ascii=False, indent=2)
        print(f"\n✓ Successfully saved {len(all_points)} points to {save_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Debug version: Convert single STEP file to point cloud (PLY) with UV sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('step_file', type=str, 
                        help='Input STEP file to process')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output PLY file (default: same name as input)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of random samples per face')
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable debug output')
    
    args = parser.parse_args()
    
    # Set output filename
    if args.output is None:
        output_file = args.step_file.replace('.step', '.npz')
    else:
        output_file = args.output
    
    # Ensure output has .ply extension
    if not output_file.endswith('.npz'):
        output_file += '.npz'

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DEBUG MODE - Single Threaded Conversion")
    print(f"{'='*60}")
    print(f"Input:  {args.step_file}")
    print(f"Output: {output_file}")
    print(f"Random samples per face: {args.num_samples}")
    print(f"Debug output: {'disabled' if args.no_debug else 'enabled'}")
    print(f"{'='*60}\n")
    

    step_to_pointcloud(args.step_file, output_file, 
                        num_samples=args.num_samples, 
                        debug=not args.no_debug)




if __name__ == '__main__':
    exit(main())

