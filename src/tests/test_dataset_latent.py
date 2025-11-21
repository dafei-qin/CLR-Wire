"""
Test script for LatentDataset.

This script:
1. Loads latent representations from NPZ files
2. Decodes them using VAE to get canonical space parameters
3. Transforms back to original space using from_canonical
4. Loads corresponding point cloud data from NPY files
5. Visualizes surfaces with gradient colors and point clouds side by side
"""

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
from pathlib import Path
import argparse
import colorsys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dataset.dataset_latent import LatentDataset
from src.vae.vae_v1 import SurfaceVAE
from src.tools.surface_to_canonical_space import from_canonical
from utils.surface import visualize_json_interset


def load_vae_model(checkpoint_path, device='cpu'):
    """Load the VAE model from checkpoint"""
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'ema_model' in checkpoint:
        ema_model = checkpoint['ema']
        ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
        model.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights.")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict.")
    
    model.to(device)
    model.eval()
    return model


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


def load_corresponding_pointcloud(npz_path, pointcloud_dir):
    """
    Load the corresponding point cloud NPY file for a given NPZ file.
    
    Args:
        npz_path: Path to the NPZ file
        pointcloud_dir: Root directory containing point cloud NPY files
        
    Returns:
        Tuple of (point cloud array (N, 3), npy_path) or (None, None) if not found
    """
    npz_path = Path(npz_path)
    pointcloud_dir = Path(pointcloud_dir)
    
    # Get the NPZ file name (without extension)
    npz_stem = npz_path.stem
    
    # Try to find the corresponding NPY file
    # Method 1: Same relative path structure
    try:
        # Get relative path from NPZ directory
        npz_relative = npz_path.relative_to(npz_path.parent.parent)
        npy_path = pointcloud_dir / npz_relative.parent / f"{npz_stem}.npy"
        if npy_path.exists():
            return np.load(npy_path), str(npy_path)
    except:
        pass
    
    # Method 2: Direct file name match in pointcloud_dir
    npy_path = pointcloud_dir / f"{npz_stem}.npy"
    if npy_path.exists():
        return np.load(npy_path), str(npy_path)
    
    # Method 3: Search recursively in pointcloud_dir
    for npy_file in pointcloud_dir.rglob(f"{npz_stem}.npy"):
        return np.load(npy_file), str(npy_file)
    
    print(f"Warning: Could not find corresponding NPY file for {npz_path.name}")
    return None, None


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
current_npz_path = None
current_npy_path = None
show_surfaces = True
show_pointcloud = True
colormap_options = ['rainbow', 'viridis', 'cool_to_warm', 'red_to_blue']
current_colormap = 0


def update_visualization():
    """Update the visualization with current index"""
    global current_idx, latent_dataset, vae_model, dataset_helper, device
    global surfaces_dict, current_colormap, pointcloud_dir, current_pointcloud
    global current_npz_path, current_npy_path
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Load data from latent dataset
    (latent_params, rotations, scales, shifts, classes, 
     bbox_mins, bbox_maxs, mask) = latent_dataset[current_idx]
    
    # Store current NPZ path
    current_npz_path = latent_dataset.npz_files[current_idx]
    
    # Load corresponding point cloud if pointcloud_dir is provided
    current_pointcloud = None
    current_npy_path = None
    if pointcloud_dir is not None:
        current_pointcloud, current_npy_path = load_corresponding_pointcloud(current_npz_path, pointcloud_dir)
    
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
    
    # Visualize surfaces with gradient colors
    try:
        surfaces_dict = visualize_json_interset(
            recovered_surfaces,
            plot=True,
            plot_gui=False,
            tol=1e-5,
            ps_header='surface'
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
        
        print(f"Visualized {len(surfaces_dict)} surfaces with {colormap_options[current_colormap]} colormap")
        print(f"First surface bbox: {bbox_mins_valid[0].numpy()}")
        print(f"Last surface bbox: {bbox_mins_valid[-1].numpy()}")
        print(f"Surface types: {classes_valid.numpy()}")
        
    except Exception as e:
        print(f"Error visualizing surfaces: {e}")
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
        except Exception as e:
            print(f"Error visualizing point cloud: {e}")
            import traceback
            traceback.print_exc()


def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_surfaces, surfaces_dict, current_colormap
    global show_pointcloud, current_pointcloud, current_npz_path, current_npy_path
    global latent_dataset, pointcloud_dir
    
    psim.Text("Latent Dataset Visualization")
    psim.Separator()
    
    # Display file paths
    if current_npz_path is not None:
        npz_path_obj = Path(current_npz_path)
        npz_dir_obj = Path(latent_dataset.npz_dir)
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
    if current_pointcloud is not None:
        psim.Text(f"Points: {len(current_pointcloud)}")
    else:
        psim.Text("Points: N/A")
    
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
        for surface_data in surfaces_dict.values():
            if 'ps_handler' in surface_data and surface_data['ps_handler'] is not None:
                surface_data['ps_handler'].set_enabled(show_surfaces)
    
    # Point cloud show/hide control
    if current_pointcloud is not None:
        changed, show_pointcloud = psim.Checkbox("Show Point Cloud", show_pointcloud)
        if changed:
            pc_structure = ps.get_point_cloud("point_cloud", error_if_absent=False)
            if pc_structure is not None:
                pc_structure.set_enabled(show_pointcloud)
    
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
    if current_pointcloud is not None:
        psim.Text("")
        psim.Text("Point Cloud (gray):")
        psim.Text("  Original sampled points")


def main():
    global current_idx, max_idx, latent_dataset, vae_model, dataset_helper, device, pointcloud_dir
    
    parser = argparse.ArgumentParser(description='Test and visualize LatentDataset with point clouds')
    parser.add_argument('npz_dir', type=str, help='Directory containing NPZ files')
    parser.add_argument('checkpoint_path', type=str, help='Path to VAE checkpoint')
    parser.add_argument('--pointcloud_dir', type=str, default=None,
                       help='Directory containing corresponding NPY point cloud files')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on')
    parser.add_argument('--index', type=int, default=0, help='Initial sample index')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension')
    
    args = parser.parse_args()
    
    # Set point cloud directory
    pointcloud_dir = args.pointcloud_dir
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # Load latent dataset
    print(f"Loading latent dataset from: {args.npz_dir}")
    latent_dataset = LatentDataset(
        npz_dir=args.npz_dir,
        max_num_surfaces=500,
        latent_dim=args.latent_dim
    )
    max_idx = len(latent_dataset) - 1
    current_idx = min(args.index, max_idx)
    
    # Load VAE model
    print(f"Loading VAE model from: {args.checkpoint_path}")
    vae_model = load_vae_model(args.checkpoint_path, device=device)
    
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
    if pointcloud_dir:
        print("  - Point clouds (gray) are loaded from corresponding NPY files")
        print("  - Toggle 'Show Point Cloud' to show/hide point clouds")
    print("="*80 + "\n")
    
    ps.show()


if __name__ == '__main__':
    main()


