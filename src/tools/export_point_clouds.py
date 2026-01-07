#!/usr/bin/env python3
"""
Export point clouds from NPZ files to PLY format.
Scans input directory recursively for all .npz files and exports them as .ply files.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import open3d as o3d


def export_npz_to_ply(npz_path: str) -> bool:
    """
    Export a single NPZ file to PLY format.
    
    Args:
        npz_path: Path to the NPZ file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load NPZ file
        data = np.load(npz_path, allow_pickle=True)
        points = data['points']
        normals = data['normals']
        masks = data['masks']
        graph_nodes = data.get('graph_nodes', None)
        graph_edges = data.get('graph_edges', None)
        
        # Handle different data formats
        # If points is a list of arrays, concatenate them
        # if isinstance(points, (list, np.ndarray)) and len(points) > 0:
        #     if isinstance(points[0], np.ndarray):
        #         # List of arrays - concatenate all surfaces
        #         points = np.concatenate(points, axis=0)
        #         normals = np.concatenate(normals, axis=0)
        #         if isinstance(masks, (list, np.ndarray)) and len(masks) > 0:
        #             masks = np.concatenate(masks, axis=0)
        
        # Ensure points and normals are numpy arrays
        points = np.asarray(points)
        normals = np.asarray(normals)
        
        # Apply mask if available and valid
        if masks is not None:
            masks = np.asarray(masks)
            if masks.dtype == bool or masks.dtype == np.int32 or masks.dtype == np.int64:
                # Filter points where mask is True or 1
                valid_mask = masks == 1 if masks.dtype != bool else masks
                if valid_mask.ndim == 1 and len(valid_mask) == len(points):
                    points = points[valid_mask]
                    normals = normals[valid_mask]
        
        # Validate data
        if len(points) == 0:
            print(f"Warning: No points found in {npz_path}")
            return False
        
        if points.shape[1] != 3:
            print(f"Warning: Points should be (N, 3), got shape {points.shape} in {npz_path}")
            return False
        
        if normals.shape[1] != 3:
            print(f"Warning: Normals should be (N, 3), got shape {normals.shape} in {npz_path}")
            return False
        
        if len(points) != len(normals):
            print(f"Warning: Mismatch between points ({len(points)}) and normals ({len(normals)}) in {npz_path}")
            # Use minimum length
            min_len = min(len(points), len(normals))
            points = points[:min_len]
            normals = normals[:min_len]
        
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
        
        # Generate output PLY path (same directory, same name, .ply extension)
        ply_path = str(Path(npz_path).with_suffix('.ply'))
        
        # Save point cloud
        success = o3d.io.write_point_cloud(ply_path, point_cloud)
        
        if success:
            print(f"Exported: {npz_path} -> {ply_path} ({len(points)} points)")
        else:
            print(f"Error: Failed to write {ply_path}")
        
        return success
        
    except KeyError as e:
        print(f"Error: Missing key in {npz_path}: {e}")
        return False
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def scan_and_export(input_dir: str):
    """
    Scan directory recursively for NPZ files and export them to PLY.
    
    Args:
        input_dir: Root directory to scan
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # Find all NPZ files recursively
    npz_files = list(input_path.rglob("*.npz"))
    
    if len(npz_files) == 0:
        print(f"No NPZ files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ file(s)")
    print("-" * 60)
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for npz_file in npz_files:
        if export_npz_to_ply(str(npz_file)):
            success_count += 1
        else:
            fail_count += 1
    
    print("-" * 60)
    print(f"Export complete: {success_count} succeeded, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Export point clouds from NPZ files to PLY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_point_clouds.py --input_dir ./data
  python export_point_clouds.py --input_dir /path/to/npz/files
        """
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory to scan for NPZ files (searched recursively)'
    )
    
    args = parser.parse_args()
    
    scan_and_export(args.input_dir)


if __name__ == '__main__':
    main()

