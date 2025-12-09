"""
Test script for LatentDataset.

This script:
1. Loads latent representations from NPZ files
2. Decodes them using VAE to get canonical space parameters
3. Transforms back to original space using from_canonical
4. Loads corresponding point cloud data from NPY files
5. Visualizes surfaces with gradient colors and point clouds side by side
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
from pathlib import Path
import argparse
import colorsys
import trimesh
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_latent import LatentDataset
from src.vae.vae_v1 import SurfaceVAE
from src.tools.surface_to_canonical_space import from_canonical
from utils.surface import visualize_json_interset
from src.utils.surface_latent_tools import decode_and_sample_with_rts
from src.utils.import_tools import load_model_from_config

from omegaconf import OmegaConf

def generate_gradient_colors(n, colormap='rainbow'):
    """
    Generate n colors forming a gradient.
    
    Args:
        n: Number of colors to generate
        colormap: 'rainbow', 'viridis', 'cool_to_warm', 'red_to_blue'
    
    Returns:
        List of RGB colors (each color is [r, g, b] with values in [0, 1])
    """
    colors = []
    
    if colormap == 'rainbow':
        for i in range(n):
            hue = i / max(n - 1, 1)  # 0 to 1
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(list(rgb))
    
    elif colormap == 'viridis':
        # Approximate viridis colormap
        for i in range(n):
            t = i / max(n - 1, 1)
            r = 0.267 + t * (0.993 - 0.267)
            g = 0.005 + t * (0.906 - 0.005)
            b = 0.329 + t * (0.144 - 0.329)
            colors.append([r, g, b])
    
    elif colormap == 'cool_to_warm':
        for i in range(n):
            t = i / max(n - 1, 1)
            r = t
            g = 0.5
            b = 1.0 - t
            colors.append([r, g, b])
    
    elif colormap == 'red_to_blue':
        for i in range(n):
            t = i / max(n - 1, 1)
            r = 1.0 - t
            g = 0.0
            b = t
            colors.append([r, g, b])
    
    else:
        # Default to rainbow
        return generate_gradient_colors(n, 'rainbow')
    
    return colors


def load_corresponding_pointcloud(npz_dir, npz_path, pointcloud_dir):
    """
    Load the corresponding point cloud NPY file for a given NPZ file.
    
    Supports two formats:
    - (N, 3): positions only [x, y, z]
    - (N, 6): positions + normals [x, y, z, nx, ny, nz]
    
    Args:
        npz_path: Path to the NPZ file
        pointcloud_dir: Root directory containing point cloud NPY files
        
    Returns:
        Tuple of (positions (N, 3), normals (N, 3) or None, npy_path) 
        or (None, None, None) if not found
    """
    # npz_path = Path(npz_path)
    pointcloud_dir = Path(npz_path.replace(str(npz_dir), pointcloud_dir))
    
    npy_file = str(pointcloud_dir).replace('.npz', '.npy')
    print(npy_file)
    
    try:
        data = np.load(npy_file)
        
        if data.shape[1] == 3:
            # Only positions
            return data, None, str(npy_file)
        elif data.shape[1] == 6:
            # Positions + normals
            positions = data[:, :3]
            normals = data[:, 3:]
            return positions, normals, str(npy_file)
        else:
            print(f"Warning: Unexpected data shape {data.shape} in {npy_file}")
            return None, None, None
    except Exception as e:
        print(f"Warning: Could not load NPY file {npy_file}: {e}")
        return None, None, None


def decode_and_recover(model, latent_params, rotations, scales, shifts, classes, dataset_helper, device='cpu'):
    """
    Decode latent representations and recover to original space.
    
    Args:
        model: VAE model
        latent_params: (N, latent_dim) tensor
        rotations: (N, 6) tensor - rotation matrices (first 6 elements)
        scales: (N, 1) tensor
        shifts: (N, 3) tensor
        classes: (N,) tensor
        dataset_helper: Dataset instance for _recover_surface
        device: Device to run on
        
    Returns:
        List of surface dictionaries in original space
    """
    # Move to device
    latent_params = latent_params.to(device)
    classes = classes.to(device)
    
    # Decode using VAE
    with torch.no_grad():
        params_decoded, mask = model.decode(latent_params, classes)
    
    # Convert to numpy
    params_decoded_np = params_decoded.cpu().numpy()
    classes_np = classes.cpu().numpy()
    rotations_np = rotations.cpu().numpy() if torch.is_tensor(rotations) else rotations
    scales_np = scales.cpu().numpy() if torch.is_tensor(scales) else scales
    shifts_np = shifts.cpu().numpy() if torch.is_tensor(shifts) else shifts
    
    # Recover surfaces
    recovered_surfaces = []
    
    for i in range(len(latent_params)):
        # Recover canonical space surface from params
        surface_canonical = dataset_helper._recover_surface(
            params_decoded_np[i],
            classes_np[i]
        )
        
        # Reconstruct full 3x3 rotation matrix from first 6 elements
        # rotation_6d contains [r11, r12, r13, r21, r22, r23]
        # Third row can be computed as cross product of first two rows
        rotation_6d = rotations_np[i]
        row1 = rotation_6d[:3]
        row2 = rotation_6d[3:6]
        row3 = np.cross(row1, row2)
        rotation_matrix = np.array([row1, row2, row3], dtype=np.float64)
        
        # Get shift and scale
        shift = shifts_np[i]
        scale = scales_np[i, 0] if scales_np.ndim > 1 else scales_np[i]
        
        # Transform back to original space
        surface_original = from_canonical(
            surface_canonical,
            shift,
            rotation_matrix,
            scale
        )
        
        # Add metadata
        surface_original['idx'] = [i, i]
        surface_original['orientation'] = 'Forward'
        
        recovered_surfaces.append(surface_original)
    
    return recovered_surfaces


# Global variables for interactive visualization
current_idx = 0
max_idx = 0
latent_dataset = None
vae_model = None
dataset_helper = None
device = 'cpu'
surfaces_dict = {}
pointcloud_dir = None
current_pointcloud = None
current_normals = None
current_npz_path = None
current_npy_path = None
show_surfaces = True
show_pointcloud = True
show_normals = True
colormap_options = ['rainbow', 'viridis', 'cool_to_warm', 'red_to_blue']
current_colormap = 0
# OBJ file visualization
obj_path_input = ""
loaded_obj_mesh = None
loaded_obj_path = None
show_obj_mesh = True
obj_dir = None  # Root directory for OBJ files
auto_load_obj = True  # Whether to auto-load corresponding OBJ files
obj_original_vertices = None  # Store original vertices for rotation
obj_rotation_euler = [0.0, 0.0, 0.0]  # Euler angles in degrees [rx, ry, rz]
# Surface group
surfaces_group = None
# Sampled points from latent
show_sampled_points = True
sampled_points = None  # Store sampled points (B, H, W, 3)


def convert_npz_path_to_obj_path(npz_path, latent_dir, obj_dir):
    """
    Convert NPZ path to corresponding OBJ path.
    
    Example mapping:
    NPZ: latent_dir/abc/0/0000/00000008/00000008_9b3d6a97e8de4aa193b81000_step_000/index_000.npz
    OBJ: obj_dir/abc/s/0/0000/00000008/00000008_9b3d6a97e8de4aa193b81000_step_000_0.obj
    
    Args:
        npz_path: Full path to NPZ file
        latent_dir: Root directory for latent files
        obj_dir: Root directory for OBJ files
        
    Returns:
        Path to corresponding OBJ file, or None if conversion fails
    """
    try:
        npz_path = Path(npz_path)
        latent_dir = Path(latent_dir)
        obj_dir = Path(obj_dir)
        
        # Get relative path from latent_dir
        rel_path = npz_path.relative_to(latent_dir)
        parts = list(rel_path.parts)
        
        if len(parts) < 2:
            return None
        
        # Extract components
        # parts = ['abc', '0', '0000', '00000008', '00000008_9b3d6a97e8de4aa193b81000_step_000', 'index_000.npz']
        
        # Insert 's' after first directory
        first_dir = parts[0]
        middle_dirs = parts[1:-2]  # ['0', '0000', '00000008']
        parent_dir = parts[-2]  # '00000008_9b3d6a97e8de4aa193b81000_step_000'
        filename = parts[-1]  # 'index_000.npz'
        
        # Extract index number from filename (index_000.npz -> 0)
        # Remove 'index_' prefix and '.npz' suffix, then convert to int to remove leading zeros
        if filename.startswith('index_') and filename.endswith('.npz'):
            index_str = filename[6:-4]  # '000'
            try:
                index_num = int(index_str)  # 0
            except ValueError:
                index_num = 0
        else:
            index_num = 0
        
        # Construct OBJ filename: parent_dir + '_' + index + '.obj'
        obj_filename = f"{parent_dir}_{index_num}.obj"
        
        # Build OBJ path: obj_dir/first_dir/s/middle_dirs.../obj_filename
        obj_path = obj_dir / first_dir  / Path(*middle_dirs) / obj_filename
        
        return str(obj_path)
        
    except Exception as e:
        print(f"Error converting NPZ path to OBJ path: {e}")
        return None


def apply_rotation_to_obj(euler_angles):
    """
    Apply rotation to OBJ mesh using Euler angles.
    
    Args:
        euler_angles: List of [rx, ry, rz] in degrees
    """
    global loaded_obj_mesh, obj_original_vertices, show_obj_mesh
    
    if loaded_obj_mesh is None or obj_original_vertices is None:
        return
    
    try:
        # Create rotation from Euler angles (in degrees, XYZ order)
        rotation = R.from_euler('xyz', euler_angles, degrees=True)
        rotation_matrix = rotation.as_matrix()
        
        # Apply rotation to original vertices
        rotated_vertices = (rotation_matrix @ obj_original_vertices.T).T
        
        # Update the mesh in polyscope
        faces = np.array(loaded_obj_mesh.faces, dtype=np.int32)
        
        # Remove and re-register
        if ps.has_surface_mesh("obj_mesh"):
            ps.remove_surface_mesh("obj_mesh")
        
        ps_mesh = ps.register_surface_mesh("obj_mesh", rotated_vertices, faces)
        ps_mesh.set_enabled(show_obj_mesh)
        ps_mesh.set_color([0.2, 0.8, 0.4])
        ps_mesh.set_transparency(0.7)
        
    except Exception as e:
        print(f"Error applying rotation: {e}")
        import traceback
        traceback.print_exc()


def load_obj_file(obj_path):
    """
    Load an OBJ file using trimesh and visualize it in polyscope.
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        True if successful, False otherwise
    """
    global loaded_obj_mesh, loaded_obj_path, show_obj_mesh, obj_original_vertices, obj_rotation_euler
    
    try:
        print(f"Loading OBJ file: {obj_path}")
        
        # Load mesh using trimesh
        mesh = trimesh.load(obj_path, force='mesh')
        
        # If it's a Scene, get the first geometry
        if isinstance(mesh, trimesh.Scene):
            # Combine all geometries in the scene
            mesh = trimesh.util.concatenate(
                [geom for geom in mesh.geometry.values() 
                 if isinstance(geom, trimesh.Trimesh)]
            )
        
        # Store the loaded mesh
        loaded_obj_mesh = mesh
        loaded_obj_path = obj_path
        
        # Store original vertices
        obj_original_vertices = np.array(mesh.vertices, dtype=np.float64)
        
        # Reset rotation to identity
        obj_rotation_euler = [90.0, 0.0, 0.0]
        
        # Register mesh in polyscope
        vertices = obj_original_vertices.copy()
        faces = np.array(mesh.faces, dtype=np.int32)
        
        print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
        
        # Remove old obj mesh if exists
        if ps.has_surface_mesh("obj_mesh"):
            ps.remove_surface_mesh("obj_mesh")
        
        # Register the mesh
        ps_mesh = ps.register_surface_mesh("obj_mesh", vertices, faces)
        ps_mesh.set_enabled(show_obj_mesh)
        ps_mesh.set_color([0.2, 0.8, 0.4])  # Green color for OBJ mesh
        ps_mesh.set_transparency(0.7)
        apply_rotation_to_obj(obj_rotation_euler)
        print(f"Successfully visualized OBJ file: {Path(obj_path).name}")
        return True
        
    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, latent_dataset, vae_model, dataset_helper, device
    global surfaces_dict, current_colormap, pointcloud_dir, current_pointcloud
    global current_npz_path, current_npy_path, current_normals, show_normals
    global loaded_obj_mesh, show_obj_mesh, obj_dir, auto_load_obj, loaded_obj_path
    global surfaces_group
    
    # Clear existing structures (don't preserve OBJ mesh - will reload if available)
    ps.remove_all_structures()
    
    # Load data from latent dataset
    (latent_params, rotations, scales, shifts, classes, 
     bbox_mins, bbox_maxs, mask, pc) = latent_dataset[current_idx]
    
    # Store current NPZ path
    current_npz_path = latent_dataset.latent_files[current_idx]
    
    # Load corresponding point cloud if pointcloud_dir is provided
    current_pointcloud = None
    current_normals = None
    current_npy_path = None
    if pointcloud_dir is not None:
        current_pointcloud, current_normals, current_npy_path = load_corresponding_pointcloud(
            latent_dataset.latent_dir, current_npz_path, pointcloud_dir
        )
    
    # Auto-load corresponding OBJ file if obj_dir is provided
    loaded_obj_mesh = None
    loaded_obj_path = None
    if obj_dir is not None and auto_load_obj:
        obj_path = convert_npz_path_to_obj_path(
            current_npz_path, 
            latent_dataset.latent_dir, 
            obj_dir
        )
        if obj_path is not None and Path(obj_path).exists():
            print(f"Auto-loading OBJ: {Path(obj_path).name}")
            load_obj_file(obj_path)
        else:
            if obj_path is not None:
                print(f"OBJ file not found: {obj_path}")
    
    # Get valid surfaces
    valid_mask = mask.bool()
    num_valid = valid_mask.sum().item()
    
    if num_valid == 0:
        print(f"No valid surfaces in sample {current_idx}")
        return
    
    print(f"\nProcessing sample {current_idx} with {num_valid} surfaces")
    
    # Extract valid data
    latent_params_valid = latent_params[valid_mask]
    rotations_valid = rotations[valid_mask]
    scales_valid = scales[valid_mask]
    shifts_valid = shifts[valid_mask]
    classes_valid = classes[valid_mask]
    bbox_mins_valid = bbox_mins[valid_mask]
    bbox_maxs_valid = bbox_maxs[valid_mask]
    
    # Decode and recover surfaces
    recovered_surfaces = decode_and_recover(
        vae_model,
        latent_params_valid,
        rotations_valid,
        scales_valid,
        shifts_valid,
        classes_valid,
        dataset_helper,
        device=device
    )
    
    # Generate gradient colors based on sorting order
    colors = generate_gradient_colors(num_valid, colormap_options[current_colormap])
    
    # Create a group for all surfaces
    if surfaces_group is not None:
        ps.remove_group(surfaces_group)
    surfaces_group = ps.create_group("latent_surfaces")
    
    # Visualize surfaces with gradient colors
    try:
        surfaces_dict = visualize_json_interset(
            recovered_surfaces,
            plot=True,
            plot_gui=False,
            tol=1e-5,
            ps_header='latent_surfaces/surface'
        )
        
        # Apply gradient colors to surfaces
        for i, (surface_key, surface_data) in enumerate(surfaces_dict.items()):
            if 'ps_handler' in surface_data and surface_data['ps_handler'] is not None:
                # Set color for this surface
                try:
                    surface_data['ps_handler'].add_color_quantity(
                        "gradient",
                        np.tile(colors[i], (surface_data['ps_handler'].n_vertices(), 1)),
                        enabled=True
                    )
                except:
                    # Fallback: just set a uniform color if possible
                    pass
                surface_data['ps_handler'].add_to_group(surfaces_group)
        
        print(f"Visualized {len(surfaces_dict)} surfaces with {colormap_options[current_colormap]} colormap")
        print(f"First surface bbox: {bbox_mins_valid[0].numpy()}")
        print(f"Last surface bbox: {bbox_mins_valid[-1].numpy()}")
        print(f"Surface types: {classes_valid.numpy()}")
        
    except Exception as e:
        print(f"Error visualizing surfaces: {e}")
        import traceback
        traceback.print_exc()
    
    # Sample points from latent using decode_and_sample_with_rts
    global sampled_points, show_sampled_points
    sampled_points = None
    try:
        print("Sampling points from latent...")
        with torch.no_grad():
            # Sample points using the latent tool
            samples = decode_and_sample_with_rts(
                vae_model,
                latent_params_valid.to(device),
                shifts_valid.to(device),
                rotations_valid.to(device),
                scales_valid.to(device),
                log_scale=False
            )
            
            # samples shape: (num_surfaces, H, W, 3)
            sampled_points = samples.cpu().numpy()
            print(f"Sampled points shape: {sampled_points.shape}")
            sampled_points = sampled_points.reshape(-1, 3)
            ps_samples = ps.register_point_cloud('point_cloud', sampled_points)
            # Visualize sampled points for each surface
            # for i in range(len(sampled_points)):
            #     surface_samples = sampled_points[i]  # (H, W, 3)
            #     # Flatten to point cloud
            #     points = surface_samples.reshape(-1, 3)
                
            #     # Register point cloud for this surface
            #     pc_name = f"sampled_surface_{i}"
            #     ps_samples = ps.register_point_cloud(pc_name, points)
            #     ps_samples.set_enabled(show_sampled_points)
            #     ps_samples.set_color(colors[i])  # Use same gradient color
            #     ps_samples.set_radius(0.002, relative=False)
            
            print(f"Visualized sampled points for {len(sampled_points)} surfaces")
            
    except Exception as e:
        print(f"Error sampling points from latent: {e}")
        import traceback
        traceback.print_exc()
    
    # Visualize point cloud if available
    if current_pointcloud is not None:
        try:
            print(f"Visualizing point cloud with {len(current_pointcloud)} points")
            pc_cloud = ps.register_point_cloud("point_cloud", current_pointcloud)
            pc_cloud.set_enabled(show_pointcloud)
            # Set point cloud color to gray for contrast with colored surfaces
            pc_cloud.set_color([0.5, 0.5, 0.5])
            pc_cloud.set_radius(0.003, relative=False)
            
            # Add normals if available
            if current_normals is not None:
                print(f"Adding normal vectors to point cloud")
                pc_cloud.add_vector_quantity(
                    "normals",
                    current_normals,
                    enabled=show_normals,
                    vectortype='standard',
                    length=0.02,  # Adjust this for better visualization
                    radius=0.002,
                    color=[1.0, 0.3, 0.3]  # Red color for normals
                )
                print(f"Normal vector stats: mean magnitude = {np.linalg.norm(current_normals, axis=1).mean():.4f}")
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
            import traceback
            traceback.print_exc()


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_surfaces, surfaces_dict, current_colormap
    global show_pointcloud, show_normals, current_pointcloud, current_normals
    global current_npz_path, current_npy_path, latent_dataset, pointcloud_dir
    global obj_path_input, loaded_obj_mesh, loaded_obj_path, show_obj_mesh
    global obj_dir, auto_load_obj, surfaces_group
    global obj_rotation_euler, obj_original_vertices
    global show_sampled_points, sampled_points
    
    psim.Text("Latent Dataset Visualization")
    psim.Separator()
    
    # Display file paths
    if current_npz_path is not None:
        npz_path_obj = Path(current_npz_path)
        npz_dir_obj = Path(latent_dataset.latent_dir)
        try:
            npz_relative = npz_path_obj.relative_to(npz_dir_obj)
            psim.TextColored((0.3, 0.7, 1.0, 1.0), f"NPZ: {npz_relative}")
        except:
            psim.TextColored((0.3, 0.7, 1.0, 1.0), f"NPZ: {npz_path_obj.name}")
    
    if current_npy_path is not None and pointcloud_dir is not None:
        npy_path_obj = Path(current_npy_path)
        pc_dir_obj = Path(pointcloud_dir)
        try:
            npy_relative = npy_path_obj.relative_to(pc_dir_obj)
            psim.TextColored((0.3, 1.0, 0.7, 1.0), f"NPY: {npy_relative}")
        except:
            psim.TextColored((0.3, 1.0, 0.7, 1.0), f"NPY: {npy_path_obj.name}")
    elif pointcloud_dir is not None:
        psim.TextColored((1.0, 0.5, 0.3, 1.0), "NPY: Not found")
    
    psim.Separator()
    
    # OBJ file loading controls
    psim.Text("=== Load OBJ File ===")
    
    # Auto-load toggle (only show if obj_dir is set)
    if obj_dir is not None:
        changed, auto_load_obj = psim.Checkbox("Auto-load OBJ", auto_load_obj)
        if changed:
            update_visualization()
        psim.TextColored((0.6, 0.6, 0.6, 1.0), "(Auto-loads from obj_dir)")
    
    # Manual loading controls
    changed, obj_path_input = psim.InputText("OBJ Path", obj_path_input)
    
    if psim.Button("Load OBJ"):
        if obj_path_input.strip():
            load_obj_file(obj_path_input.strip())
    
    psim.SameLine()
    if psim.Button("Clear OBJ"):
        if ps.has_surface_mesh("obj_mesh"):
            ps.remove_surface_mesh("obj_mesh")
            loaded_obj_mesh = None
            loaded_obj_path = None
            print("Cleared OBJ mesh")
    
    if loaded_obj_path is not None:
        psim.TextColored((0.2, 0.8, 0.4, 1.0), f"Loaded: {Path(loaded_obj_path).name}")
        
        # Show/hide control for OBJ mesh
        changed, show_obj_mesh = psim.Checkbox("Show OBJ Mesh", show_obj_mesh)
        if changed and ps.has_surface_mesh("obj_mesh"):
            obj_mesh = ps.get_surface_mesh("obj_mesh")
            obj_mesh.set_enabled(show_obj_mesh)
        
        if loaded_obj_mesh is not None:
            psim.Text(f"Vertices: {len(loaded_obj_mesh.vertices)}")
            psim.Text(f"Faces: {len(loaded_obj_mesh.faces)}")
            
            # Rotation controls
            psim.Text("Rotation (Euler XYZ, degrees):")
            
            # Individual sliders for each axis
            changed_x, obj_rotation_euler[0] = psim.SliderFloat("Rot X", obj_rotation_euler[0], -180.0, 180.0)
            changed_y, obj_rotation_euler[1] = psim.SliderFloat("Rot Y", obj_rotation_euler[1], -180.0, 180.0)
            changed_z, obj_rotation_euler[2] = psim.SliderFloat("Rot Z", obj_rotation_euler[2], -180.0, 180.0)
            
            if changed_x or changed_y or changed_z:
                apply_rotation_to_obj(obj_rotation_euler)
            
            # Reset button
            if psim.Button("Reset Rotation"):
                obj_rotation_euler = [0.0, 0.0, 0.0]
                apply_rotation_to_obj(obj_rotation_euler)
            
    elif obj_dir is not None and auto_load_obj:
        psim.TextColored((1.0, 0.5, 0.3, 1.0), "OBJ: Not found")
    
    psim.Separator()
    
    # Index controls
    slider_changed, slider_idx = psim.SliderInt("Sample Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    input_changed, input_idx = psim.InputInt("Go To Index", current_idx)
    if input_changed:
        input_idx = max(0, min(max_idx, input_idx))
        if input_idx != current_idx:
            current_idx = input_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    psim.Text(f"Surfaces: {len(surfaces_dict)}")
    # if sampled_points is not None:
    #     total_sampled = sampled_points.shape[0] * sampled_points.shape[1] * sampled_points.shape[2]
    #     psim.TextColored((1.0, 0.8, 0.2, 1.0), f"Sampled Points: {total_sampled}")
    #     psim.Text(f"  ({sampled_points.shape[0]} surfaces × {sampled_points.shape[1]}×{sampled_points.shape[2]})")
    if current_pointcloud is not None:
        psim.Text(f"GT Points: {len(current_pointcloud)}")
        if current_normals is not None:
            psim.TextColored((0.3, 1.0, 0.3, 1.0), "Normals: Available")
        else:
            psim.Text("Normals: N/A")
    else:
        psim.Text("GT Points: N/A")
        psim.Text("Normals: N/A")
    
    # Colormap selection
    psim.Separator()
    psim.Text("Colormap:")
    for i, colormap_name in enumerate(colormap_options):
        if psim.RadioButton(colormap_name, current_colormap == i):
            if current_colormap != i:
                current_colormap = i
                update_visualization()
    
    # Show/hide controls
    psim.Separator()
    changed, show_surfaces = psim.Checkbox("Show Surfaces", show_surfaces)
    if changed:
        if surfaces_group is not None:
            # Use group to enable/disable all surfaces at once
            surfaces_group.set_enabled(show_surfaces)
        else:
            # Fallback: enable/disable individual surfaces
            for surface_data in surfaces_dict.values():
                if 'ps_handler' in surface_data and surface_data['ps_handler'] is not None:
                    surface_data['ps_handler'].set_enabled(show_surfaces)
    
    # Sampled points show/hide control
    if sampled_points is not None:
        changed, show_sampled_points = psim.Checkbox("Show Sampled Points", show_sampled_points)
        if changed:
            # Enable/disable all sampled point clouds
            for i in range(len(sampled_points)):
                pc_name = f"sampled_surface_{i}"
                try:
                    pc_structure = ps.get_point_cloud(pc_name, error_if_absent=False)
                    if pc_structure is not None:
                        pc_structure.set_enabled(show_sampled_points)
                except:
                    pass
    
    # Point cloud show/hide control
    if current_pointcloud is not None:
        changed, show_pointcloud = psim.Checkbox("Show Point Cloud", show_pointcloud)
        if changed:
            pc_structure = ps.get_point_cloud("point_cloud", error_if_absent=False)
            if pc_structure is not None:
                pc_structure.set_enabled(show_pointcloud)
    
    # Normals show/hide control
    if current_normals is not None:
        changed, show_normals = psim.Checkbox("Show Normals", show_normals)
        if changed:
            pc_structure = ps.get_point_cloud("point_cloud", error_if_absent=False)
            if pc_structure is not None:
                try:
                    pc_structure.set_vector_quantity_enabled("normals", show_normals)
                except:
                    pass
    
    # Navigation buttons
    psim.Separator()
    if psim.Button("Previous (←)") or psim.IsKeyPressed(psim.ImGuiKey_LeftArrow):
        if current_idx > 0:
            current_idx -= 1
            update_visualization()
    
    psim.SameLine()
    if psim.Button("Next (→)") or psim.IsKeyPressed(psim.ImGuiKey_RightArrow):
        if current_idx < max_idx:
            current_idx += 1
            update_visualization()
    
    # Info display
    psim.Separator()
    psim.Text("=== Info ===")
    psim.Text("Colors show sorting order:")
    psim.Text("  Red/Start → surfaces with small X")
    psim.Text("  Blue/End → surfaces with large X")
    psim.Text("Sorting priority: X > Y > Z")
    if sampled_points is not None:
        psim.Text("")
        psim.Text("Sampled Points (colored):")
        psim.Text("  Points sampled from latent")
        psim.Text("  Using decode_and_sample_with_rts")
        psim.Text("  Same gradient color as surfaces")
    if current_pointcloud is not None:
        psim.Text("")
        psim.Text("GT Point Cloud (gray):")
        psim.Text("  Original ground truth points")
        if current_normals is not None:
            psim.Text("")
            psim.Text("Normals (red vectors):")
            psim.Text("  Surface normal directions")
            psim.Text("  Format: (N, 6) [x,y,z,nx,ny,nz]")


def main():
    global current_idx, max_idx, latent_dataset, vae_model, dataset_helper, device, pointcloud_dir, obj_dir
    
    parser = argparse.ArgumentParser(description='Test and visualize LatentDataset with point clouds')
    parser.add_argument('npz_dir', type=str, help='Directory containing NPZ files')
    parser.add_argument('checkpoint_path', type=str, help='Path to VAE checkpoint')
    parser.add_argument('--pointcloud_dir', type=str, default=None,
                       help='Directory containing corresponding NPY point cloud files')
    parser.add_argument('--obj_dir', type=str, default=None,
                       help='Directory containing corresponding OBJ mesh files (auto-loads if provided)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on')
    parser.add_argument('--index', type=int, default=0, help='Initial sample index')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    parser.add_argument('--vae_config', type=str, default='configs/vae_v1.yaml', help='VAE config file')
    
    args = parser.parse_args()
    
    # Set point cloud directory
    pointcloud_dir = args.pointcloud_dir
    
    # Set OBJ directory
    obj_dir = args.obj_dir
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load latent dataset
    print(f"Loading latent dataset from: {args.npz_dir}")
    if args.pointcloud_dir is None:
        pointcloud_dir = ''
    else:
        pointcloud_dir = args.pointcloud_dir
    latent_dataset = LatentDataset(
        latent_dir=args.npz_dir,
        pc_dir=pointcloud_dir,
        max_num_surfaces=500,
        latent_dim=args.latent_dim
    )
    max_idx = len(latent_dataset) - 1
    current_idx = min(args.index, max_idx)
    
    # Load VAE model
    print(f"Loading VAE model from: {args.checkpoint_path}")

    vae_config = OmegaConf.load(args.vae_config)
    vae_model = load_model_from_config(vae_config)
    
    # Create dataset helper for _recover_surface
    print("Creating dataset helper...")
    from src.dataset.dataset_v1 import dataset_compound
    # Create a minimal instance just for the helper methods
    dataset_helper = object.__new__(dataset_compound)
    from src.dataset.dataset_v1 import SURFACE_PARAM_SCHEMAS, build_surface_postpreprocess
    dataset_helper.postprocess_funcs = {
        k: build_surface_postpreprocess(v) 
        for k, v in SURFACE_PARAM_SCHEMAS.items()
    }
    
    # Initialize polyscope
    print("Initializing visualization...")
    ps.init()
    ps.set_ground_plane_mode("shadow_only")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    print("\n" + "="*80)
    print("Visualization Controls:")
    print("  - Use slider or input box to change sample index")
    print("  - Use arrow keys (← →) to navigate between samples")
    print("  - Use colormap radio buttons to change gradient colors")
    print("  - Colors show sorting order (red=start, blue=end)")
    print("  - Surfaces are sorted by bounding box (X > Y > Z)")
    print("\nSampled Points:")
    print("  - Points sampled from latent using decode_and_sample_with_rts")
    print("  - Each surface gets colored points matching its gradient color")
    print("  - Toggle 'Show Sampled Points' to show/hide sampled points")
    print("\nOBJ File Loading:")
    if obj_dir:
        print("  - Auto-load enabled: OBJ files loaded automatically from obj_dir")
        print("  - Toggle 'Auto-load OBJ' checkbox to enable/disable")
        print(f"  - OBJ directory: {obj_dir}")
    print("  - Manual loading: Enter path in 'OBJ Path' and click 'Load OBJ'")
    print("  - OBJ mesh appears in green color with transparency")
    print("  - Rotation controls: Rot X/Y/Z sliders to rotate OBJ")
    print("  - Click 'Clear OBJ' to remove the loaded mesh")
    print("  - Toggle 'Show OBJ Mesh' to show/hide the mesh")
    if pointcloud_dir:
        print("\nGround Truth Point Cloud:")
        print("  - Point clouds (gray) are loaded from corresponding NPY files")
        print("  - Toggle 'Show Point Cloud' to show/hide point clouds")
        print("  - Supported formats:")
        print("    * (N, 3): positions only [x, y, z]")
        print("    * (N, 6): positions + normals [x, y, z, nx, ny, nz]")
        print("  - Normal vectors (red) show surface orientation when available")
        print("  - Toggle 'Show Normals' to show/hide normal vectors")
    print("="*80 + "\n")
    
    ps.show()


if __name__ == '__main__':
    main()


