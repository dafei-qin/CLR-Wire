import os
import argparse
import numpy as np

from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.compound import Compound
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.BRepClass import BRepClass_FaceClassifier

from icecream import ic

# Enable icecream for debugging
ic.enable()



def sample_face_uv(face, nu=50, nv=50, debug=True):
    """
    Sample points on a face in UV space and check validity
    
    Args:
        face: TopoDS_Face object
        nu: number of samples in u direction
        nv: number of samples in v direction
        debug: whether to print debug information
    
    Returns:
        points: numpy array of shape (N, 3) with valid 3D points
        normals: numpy array of shape (N, 3) with corresponding normals
    """
    points = []
    normals = []
    
    if debug:
        ic("Starting sample_face_uv")
    
    # Get the surface from the face
    u_min, u_max, v_min, v_max = [face.uv_bounds().min_point()[0], face.uv_bounds().max_point()[0], face.uv_bounds().min_point()[1], face.uv_bounds().max_point()[1]]
    surface = BRep_Tool.Surface(face.topods_shape())
    # surface = surf_adaptor.Surface()
    
    # # Get UV bounds
    # u_min = surf_adaptor.FirstUParameter()
    # u_max = surf_adaptor.LastUParameter()
    # v_min = surf_adaptor.FirstVParameter()
    # v_max = surf_adaptor.LastVParameter()
    
    if debug:
        ic(u_min, u_max, v_min, v_max)
    
    # Create UV grid
    u_values = np.linspace(u_min, u_max, nu)
    v_values = np.linspace(v_min, v_max, nv)
    
    valid_count = 0
    invalid_count = 0
    
    # Sample points
    for i, u in enumerate(u_values):
        for j, v in enumerate(v_values):
            # Create 2D point in UV space
            uv_pnt = gp_Pnt2d(u, v)
            
            # Check if point is valid using BRepClass_FaceClassifier
            classifier = BRepClass_FaceClassifier(face.topods_shape(), uv_pnt, 1e-6)
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
                        valid_count += 1

                        du = props.D1U()
                        dv = props.D1V()
                        jacobian = du.Crossed(dv).Magnitude()
                    else:
                        invalid_count += 1
                        if debug and invalid_count <= 5:
                            ic(f"Invalid normal at u={u:.4f}, v={v:.4f}, norm={norm}")
                else:
                    invalid_count += 1
                    if debug and invalid_count <= 5:
                        ic(f"Normal not defined at u={u:.4f}, v={v:.4f}")
            else:
                invalid_count += 1
    
    if debug:
        ic(f"Valid points: {valid_count}, Invalid points: {invalid_count}")
    
    if len(points) == 0:
        if debug:
            ic("WARNING: No valid points sampled!")
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
    
    print(f"Saved {len(points)} points to {filename}")


def step_to_pointcloud(step_filename, ply_filename, nu=50, nv=50, debug=True):
    """
    Convert STEP file to point cloud with normals (single-threaded debug version)
    
    Args:
        step_filename: input STEP file
        ply_filename: output PLY file (base name)
        nu: number of UV samples in u direction
        nv: number of UV samples in v direction
        debug: whether to print debug information
    """
    print(f"\n{'='*60}")
    print(f"Processing: {step_filename}")
    print(f"{'='*60}\n")
    
    ic("Loading STEP file...")
    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())
    ic(f"Number of solids: {len(solids)}")

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
        
        # Collect points and normals from all faces
        all_points = []
        all_normals = []
        
        faces_list = list(solid.faces())
        for face_idx, face in enumerate(faces_list):
            print(f"\n  Face {face_idx}/{len(faces_list)-1}")

            points, normals = sample_face_uv(face, nu=nu, nv=nv, debug=debug)
            if len(points) > 0:
                all_points.append(points)
                all_normals.append(normals)
                print(f"  -> Sampled {len(points)} points")
            else:
                print(f"  -> No valid points sampled")

        if len(all_points) == 0:
            print(f"\nWarning: No valid points sampled for solid {index}")
            continue
        
        # Concatenate all points and normals
        all_points = np.vstack(all_points)
        all_normals = np.vstack(all_normals)
        
        # Save to PLY file
        ply_filename_with_index = f"{ply_filename[:-4]}_{index}.ply"
        save_ply(ply_filename_with_index, all_points, all_normals)
        print(f"\n✓ Successfully saved {len(all_points)} points to {ply_filename_with_index}")


def main():
    parser = argparse.ArgumentParser(
        description='Debug version: Convert single STEP file to point cloud (PLY) with UV sampling',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('step_file', type=str, 
                        help='Input STEP file to process')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output PLY file (default: same name as input)')
    parser.add_argument('--nu', type=int, default=50,
                        help='Number of UV samples in u direction')
    parser.add_argument('--nv', type=int, default=50,
                        help='Number of UV samples in v direction')
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable debug output')
    
    args = parser.parse_args()
    
    # Set output filename
    if args.output is None:
        output_file = args.step_file.replace('.step', '.ply')
    else:
        output_file = args.output
    
    # Ensure output has .ply extension
    if not output_file.endswith('.ply'):
        output_file += '.ply'
    
    print(f"\n{'='*60}")
    print(f"DEBUG MODE - Single Threaded Conversion")
    print(f"{'='*60}")
    print(f"Input:  {args.step_file}")
    print(f"Output: {output_file}")
    print(f"UV sampling: {args.nu} x {args.nv}")
    print(f"Debug output: {'disabled' if args.no_debug else 'enabled'}")
    print(f"{'='*60}\n")
    

    step_to_pointcloud(args.step_file, output_file, 
                        nu=args.nu, nv=args.nv, 
                        debug=not args.no_debug)




if __name__ == '__main__':
    exit(main())

