"""
Batch render orig_surfaces, detok_surfaces, and points on headless server.
Uses xvfb and polyscope screenshot mode to generate JPG images and concatenate them horizontally.

Usage:
  python src/tests/test_tokenize_detokenize_server.py \
      --json_dir path/to/json_folder \
      --rts_codebook_dir path/to/codebook_folder \
      --output_dir path/to/output_folder \
      [--start_index 0] \
      [--end_index 100] \
      [--width 1920] \
      [--height 1080]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import torch
from PIL import Image
from pyvirtualdisplay import Display

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.dataset.dataset_v4_tokenize_all import dataset_compound_tokenize_all
from utils.surface import visualize_json_interset


def load_raw_json(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


def visualize_points(points_list, radius=0.002):
    """Visualize point clouds from points_list."""
    # Concatenate all points
    all_points = np.concatenate(points_list, axis=0)
    if len(all_points) == 0:
        return None
    
    # Register point cloud
    cloud = ps.register_point_cloud("points", all_points, radius=radius)
    return cloud


def render_single_view(surfaces, points_list, view_name, output_path, width=1920, height=1080):
    """Render a single view and save screenshot."""
    # Clear previous structures
    ps.remove_all_structures()
    
    # Visualize surfaces (plot=False to avoid re-initializing polyscope)
    if surfaces:
        vis_data = visualize_json_interset(surfaces, plot=False, plot_gui=False, tol=1e-5, ps_header=view_name)
        # Manually register surfaces since plot=False
        for surface_data in vis_data.values():
            if 'vertices' in surface_data and 'faces' in surface_data:
                vertices = surface_data['vertices']
                faces = surface_data['faces']
                surface_index = surface_data.get('surface_index', 0)
                surface_type = surface_data.get('surface_type', 'unknown')
                ps.register_surface_mesh(
                    f"{view_name}_{surface_index:03d}_{surface_type}",
                    vertices,
                    faces,
                    transparency=0.7
                )
    
    # Visualize points
    if points_list and len(points_list) > 0:
        visualize_points(points_list, radius=0.002)
    
    # Set viewport size and camera
    ps.set_view_projection_mode("perspective")
    ps.reset_camera_to_home_view()
    
    # Take screenshot
    ps.screenshot(output_path, transparent_bg=False)
    
    return output_path


def concatenate_images(image_paths, output_path):
    """Concatenate images horizontally."""
    images = [Image.open(path) for path in image_paths]
    
    # Calculate total width and max height
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    # Create combined image
    combined_image = Image.new('RGB', (total_width, max_height))
    
    # Paste images horizontally
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Save combined image
    combined_image.save(output_path, quality=95)
    return output_path


def process_sample(dataset, index, output_dir, width=1920, height=1080):
    """Process a single sample: render orig, detok, points and concatenate."""
    print(f"Processing index {index}...")
    
    # Load data
    json_path = Path(dataset.dataset_compound.dataset_compound.json_names[index])
    raw_surfaces = load_raw_json(json_path)
    
    # Tokenize and detokenize
    points_list, normals_list, masks_list, tokens, bspline_poles, valid = dataset[index]
    if not valid:
        print(f"  Sample {index} marked invalid, skipping...")
        return False
    
    detok_surfaces = dataset.detokenize(tokens, bspline_poles)
    
    # Create output directory for this sample
    sample_output_dir = output_dir / f"sample_{index:06d}"
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporary screenshot paths
    orig_path = sample_output_dir / "orig_surfaces.jpg"
    detok_path = sample_output_dir / "detok_surfaces.jpg"
    points_path = sample_output_dir / "points.jpg"
    combined_path = sample_output_dir / "combined.jpg"
    
    screenshot_paths = []
    
    try:
        # Render original surfaces
        print(f"  Rendering orig_surfaces...")
        render_single_view(raw_surfaces, None, "orig", str(orig_path), width, height)
        screenshot_paths.append(orig_path)
        
        # Render detokenized surfaces
        print(f"  Rendering detok_surfaces...")
        render_single_view(detok_surfaces, None, "detok", str(detok_path), width, height)
        screenshot_paths.append(detok_path)
        
        # Render points
        print(f"  Rendering points...")
        render_single_view(None, points_list, "points", str(points_path), width, height)
        screenshot_paths.append(points_path)
        
        # Concatenate images
        print(f"  Concatenating images...")
        concatenate_images(screenshot_paths, str(combined_path))
        
        # Clean up individual screenshots (optional, comment out if you want to keep them)
        # for path in screenshot_paths:
        #     path.unlink()
        
        print(f"  ✓ Completed index {index}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing index {index}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Folder containing JSON + NPZ pairs")
    parser.add_argument("--rts_codebook_dir", type=str, required=True, help="Folder containing cb_rotation.pkl, cb_translation.pkl, cb_scale.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered images")
    parser.add_argument("--start_index", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive, None means all)")
    parser.add_argument("--width", type=int, default=1920, help="Screenshot width")
    parser.add_argument("--height", type=int, default=1080, help="Screenshot height")
    args = parser.parse_args()
    
    json_dir = Path(args.json_dir)
    rts_codebook_dir = Path(args.rts_codebook_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataset
    print("Building dataset...")
    dataset = dataset_compound_tokenize_all(
        json_dir=str(json_dir),
        rts_codebook_dir=str(rts_codebook_dir),
        canonical=False,
        detect_closed=False,
    )
    
    # Determine index range
    total_samples = len(dataset)
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else total_samples
    
    if start_idx < 0 or start_idx >= total_samples:
        raise IndexError(f"start_index {start_idx} out of range (len={total_samples})")
    if end_idx < start_idx or end_idx > total_samples:
        raise IndexError(f"end_index {end_idx} out of range (start={start_idx}, len={total_samples})")
    
    print(f"Processing samples {start_idx} to {end_idx-1} (total: {end_idx - start_idx})")
    
    # Initialize virtual display
    print("Initializing virtual display...")
    display = Display(visible=False, size=(args.width, args.height))
    display.start()
    
    try:
        # Initialize polyscope
        ps.init()
        ps.set_verbosity(0)
        ps.set_ground_plane_mode("shadow_only")
        
        # Process each sample
        success_count = 0
        for idx in range(start_idx, end_idx):
            success = process_sample(dataset, idx, output_dir, args.width, args.height)
            if success:
                success_count += 1
        
        print(f"\nCompleted: {success_count}/{end_idx - start_idx} samples processed successfully")
        
    finally:
        # Clean up
        display.stop()
        ps.shutdown()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

