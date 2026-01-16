import os
# Set environment variable to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

import argparse
import numpy as np
import sys
import warnings
from pathlib import Path
import json
import random
import networkx as nx
import fpsample
from copy import deepcopy

# Suppress warnings BEFORE importing occwl
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', DeprecationWarning)

sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent.parent))

from occwl.solid import Solid
from occwl.graph import face_adjacency
from occwl.compound import Compound
from occwl.io import save_step
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.gp import gp_Pnt2d
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
    GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion, GeomAbs_OffsetSurface, GeomAbs_OtherSurface,
)
from icecream import ic
from logan_process_brep_data import BRepDataProcessor

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


def get_surface_type_name(surface_type):
    """
    将曲面类型枚举值转换为可读的名称
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
        GeomAbs_OtherSurface: "OtherSurface",
    }
    return type_mapping.get(surface_type, f"Unknown({surface_type})")


def get_solid_surface_statistics(solid):
    """
    获取solid的曲面统计信息
    """
    surface_stats = {}
    for face in solid.faces():
        adaptor = BRepAdaptor_Surface(face.topods_shape())
        surface_type = adaptor.GetType()
        type_name = get_surface_type_name(surface_type)
        surface_stats[type_name] = surface_stats.get(type_name, 0) + 1
    return surface_stats


def are_solids_identical(solid1, solid2, verbose=False):
    """
    判断两个solid是否一致（面数量+各类型分布）
    """
    stats1 = get_solid_surface_statistics(solid1)
    stats2 = get_solid_surface_statistics(solid2)

    total_faces1 = sum(stats1.values())
    total_faces2 = sum(stats2.values())

    if verbose:
        print(f"Solid 1: {total_faces1} faces, distribution: {stats1}")
        print(f"Solid 2: {total_faces2} faces, distribution: {stats2}")

    if total_faces1 != total_faces2:
        if verbose:
            print(f"Different number of faces: {total_faces1} vs {total_faces2}")
        return False

    if stats1 != stats2:
        if verbose:
            print("Different surface type distribution")
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
    从solids列表中过滤出唯一的solid（与正式脚本保持一致的去重逻辑）
    """
    if len(solids) == 0:
        return [], []

    unique_solids = []
    unique_indices = []
    duplicate_groups = []

    for i, solid in enumerate(solids):
        is_duplicate = False
        for j, unique_solid in enumerate(unique_solids):
            if are_solids_identical(solid, unique_solid, verbose=False):
                is_duplicate = True
                duplicate_groups[j].append(i)
                if verbose:
                    print(f"Solid {i} is identical to solid {unique_indices[j]}")
                break
        if not is_duplicate:
            unique_solids.append(solid)
            unique_indices.append(i)
            duplicate_groups.append([i])

    if verbose:
        print(f"\n{'='*60}")
        print("Deduplication Summary:")
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


def step_to_pointcloud(step_filename, ply_filename, num_samples=1000, debug=True, fps=True, num_fps=81920, save_step_file=False):
    """
    Debug 版本：逻辑与 `sample_step_to_pc.step_to_pointcloud` 保持同步，
    但单线程、带更多打印信息。
    
    Args:
        step_filename: 输入STEP文件路径
        ply_filename: 输出NPZ文件路径
        num_samples: 每个面的采样点数
        debug: 是否输出调试信息
        fps: 是否使用FPS采样
        num_fps: FPS采样点数
        save_step_file: 是否保存处理后的solid为STEP文件
    """
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print(f"\n{'='*60}")
    print(f"[DEBUG] Processing: {step_filename}")
    print(f"{'='*60}\n")

    ic("Loading STEP file...")
    solids, attributes = Compound.load_step_with_attributes(step_filename)
    solids = list(solids.solids())
    ic(f"Number of solids: {len(solids)}")

    processor = BRepDataProcessor()

    # 预过滤：面数过多的 solid
    solids_filtered = []
    for i, solid in enumerate(solids):
        num_faces = len(list(solid.faces()))
        if num_faces > 500:
            print(f"[SKIP] Solid {i} has too many faces ({num_faces} > 500), skipping...")
        else:
            solids_filtered.append(solid)

    solids = solids_filtered
    print(f"[DEBUG] Number of solids after face count filtering: {len(solids)}")

    # 去重逻辑与正式脚本一致
    unique_solids, unique_indices = filter_unique_solids(solids, verbose=True)

    for idx, (original_index, solid) in enumerate(zip(unique_indices, unique_solids)):
        print(f"\n--- [DEBUG] Processing unique solid {idx} (original index: {original_index}) ---")

        num_faces = len(list(solid.faces()))
        ic(f"Number of faces: {num_faces}")

        solid = solid.topods_shape()
        solid = Solid(solid)

        print("Scaling to unit box...")
        solid = solid.scale_to_unit_box()

        try:
            print("Building face adjacency graph...")
            graph = face_adjacency(solid, self_loops=True)
            ic(f"Graph nodes: {len(graph.nodes())}, edges: {len(graph.edges())}")
        except Exception as e:
            ic(f"Face adjacency failed: {e}")
            # raise ValueError("Face adjacency failed. The solid may be invalid.")
            continue

        jsons_data = processor.tokenize_cad_data_preload(graph)
        # Collect points and normals from all faces
        all_points = []
        all_normals = []
        all_masks = []

        for face_idx in graph.nodes():
            face = graph.nodes[face_idx]["face"]
            face_json = jsons_data[face_idx]

            # 逐步累积采样结果，而不是每次重采样都丢弃之前的点，
            # 这样可以减少需要放大的采样次数
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

                # 如果仍然不够，则增加采样数并继续累积
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


            # Do the normal orientation adjustment.
            if face_json['orientation'] == 'Reversed':
                normals = -normals
                

            all_points.append(points.astype(np.float32))
            all_normals.append(normals.astype(np.float32))
            all_masks.append(masks.astype(bool))

        graph_undirected = strip_features_and_make_undirected(graph)


        all_points_per_face = deepcopy(all_points)
        all_normals_per_face = deepcopy(all_normals)
        all_masks_per_face = deepcopy(all_masks)

        # 与正式脚本同步：可选 FPS
        if fps:
            all_points_valid = [p[m.astype(bool)] for p, m in zip(all_points, all_masks)]
            all_normals_valid = [n[m.astype(bool)] for n, m in zip(all_normals, all_masks)]
            all_points_valid = np.concatenate(all_points_valid, axis=0)
            all_normals_valid = np.concatenate(all_normals_valid, axis=0)
            print('[DEBUG] Num valid points total: ', len(all_points_valid))
            if len(all_points_valid) < num_fps * 2:
                print('[DEBUG] Num valid points too few, skipping FPS sampling')
                continue
            fps_idx = fpsample.bucket_fps_kdtree_sampling(all_points_valid, num_fps)
            all_points = all_points_valid[fps_idx]
            all_normals = all_normals_valid[fps_idx]
            all_masks = np.ones_like(all_points, dtype=bool)

        save_name = ply_filename.replace('.npz', f'_{idx:03d}.npz')

        if debug and not fps:
            for p, n, m in zip(all_points, all_normals, all_masks):
                print("[DEBUG] face arrays shapes:", p.shape, n.shape, m.shape)

        np.savez(
            save_name,
            points=np.array(all_points, dtype=object),
            normals=np.array(all_normals, dtype=object),
            masks=np.array(all_masks, dtype=object),
            graph_nodes=list(graph_undirected.nodes()),
            graph_edges=list(graph_undirected.edges()),
        )
        np.savez(save_name.replace('.npz', '_per_face.npz'),
            points=np.array(all_points_per_face, dtype=object),
            normals=np.array(all_normals_per_face, dtype=object),
            masks=np.array(all_masks_per_face, dtype=object),
        )
        json.dump(jsons_data, open(save_name.replace('.npz', '.json'), 'w'), ensure_ascii=False, indent=2)
        
        # Save the solid as STEP file if requested
        if save_step_file:
            step_save_path = save_name.replace('.npz', '.step')
            try:
                save_step([solid], step_save_path)
                print(f"✓ [DEBUG] Successfully saved solid to {step_save_path}")
            except Exception as e:
                print(f"✗ [WARNING] Failed to save STEP file {step_save_path}: {e}")
        
        print(f"\n✓ [DEBUG] Successfully saved {len(all_points)} surfaces/points to {save_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Debug version: Convert STEP folder to NPZ with UV sampling (sequential, preserves relative paths)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input', type=str,
                        help='Input folder containing STEP files (searched recursively with rglob) OR single STEP file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output root directory for NPZ files (default: same as input_folder)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of random samples per face')
    parser.add_argument('--fps', type=bool, default=True,
                        help='Whether to use FPS sampling (same logic as main script)')
    parser.add_argument('--num_fps', type=int, default=81920,
                        help='Number of FPS samples (same default as main script)')
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable debug output')
    parser.add_argument('--single-file', action='store_true',
                        help='Process input as a single STEP file (for bash parallel usage)')
    parser.add_argument('--save_step', action='store_true',
                        help='Save the processed solid as STEP file (same name as .npz, default: False)')
    
    args = parser.parse_args()

    input_path = Path(args.input)
    
    # Single file mode (for bash parallel)
    if args.single_file or input_path.is_file():
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            return 1
        
        if not input_path.suffix.lower() == '.step':
            print(f"Error: Not a STEP file: {input_path}")
            return 1
        
        # Determine output path
        if args.output_dir:
            output_root = Path(args.output_dir)
            # If output_dir is specified, use it as base
            # For single file, preserve relative structure if possible
            npz_path = output_root / input_path.with_suffix('.npz').name
        else:
            # Default: same directory as input file
            npz_path = input_path.with_suffix('.npz')
        
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[DEBUG] Processing single file: {input_path}")
        print(f"[DEBUG] Output npz: {npz_path}")
        
        try:
            step_to_pointcloud(
                str(input_path),
                str(npz_path),
                num_samples=args.num_samples,
                debug=not args.no_debug,
                fps=args.fps,
                num_fps=args.num_fps,
                save_step_file=args.save_step,
            )
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to process {input_path}: {e}")
            return 1
    
    # Folder mode (original behavior)
    input_folder = input_path
    output_root = Path(args.output_dir) if args.output_dir is not None else input_folder

    # Collect all STEP files recursively
    stepfiles = sorted(input_folder.rglob("*.step"))

    if not stepfiles:
        print(f"No STEP files found under {input_folder}")
        return 1

    print(f"\n{'='*60}")
    print(f"DEBUG MODE - Sequential Folder Conversion")
    print(f"{'='*60}")
    print(f"Input folder:  {input_folder}")
    print(f"Output root:   {output_root}")
    print(f"Random samples per face: {args.num_samples}")
    print(f"FPS enabled: {args.fps}, num_fps: {args.num_fps}")
    print(f"Debug output: {'disabled' if args.no_debug else 'enabled'}")
    print(f"Save STEP files: {'enabled' if args.save_step else 'disabled'}")
    print(f"Total STEP files found (rglob): {len(stepfiles)}")
    print(f"{'='*60}\n")

    for i, stepfile in enumerate(stepfiles):
        # Relative path inside input folder, preserve structure under output_root
        rel_path = stepfile.relative_to(input_folder)
        npz_path = output_root / rel_path.with_suffix(".npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[DEBUG] ({i+1}/{len(stepfiles)}) Processing file: {stepfile}")
        print(f"[DEBUG] Output npz: {npz_path}")

        try:
            step_to_pointcloud(
                str(stepfile),
                str(npz_path),
                num_samples=args.num_samples,
                debug=not args.no_debug,
                fps=args.fps,
                num_fps=args.num_fps,
                save_step_file=args.save_step,
            )
        except Exception as e:
            print(f"[ERROR] Failed to process {stepfile}: {e}")
            continue
    
    return 0




if __name__ == '__main__':
    exit(main() or 0)

