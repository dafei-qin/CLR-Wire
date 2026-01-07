"""
Visualize raw JSON surfaces vs. surfaces reconstructed through the full
tokenization→detokenization pipeline (v2→v3→v4).

Usage:
  python src/tests/test_tokenize_detokenize_visualize.py \
      --json_dir path/to/json_folder \
      --rts_codebook_dir path/to/codebook_folder \
      --index 0
"""
import argparse
import json
from pathlib import Path

import polyscope as ps
import polyscope.imgui as psim
import torch
import os
import numpy as np
import einops
from omegaconf import OmegaConf
import open3d as o3d

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from src.dataset.dataset_v4_tokenize_all import dataset_compound_tokenize_all
from src.utils.import_tools import load_model_from_config
from myutils.surface import visualize_json_interset, write_to_step


def load_raw_json(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


# === Global state (for interactive polyscope UI & VAE FSQ) ===
dataset = None
current_idx = 0
max_idx = 0
pending_idx = 0

raw_group = None
detok_group = None

raw_surfaces_vis = {}
detok_surfaces_vis = {}

current_json_path = ""

vae_model = None  # VAE v4 DC-AE FSQ model (for bspline poles)

# Save functionality
save_folder_path = ""  # User-specified folder path for saving
current_all_points = None  # Store current points for saving
current_all_normals = None  # Store current normals for saving


def build_dataset(json_dir: str, rts_codebook_dir: str):
    """Create the v4 tokenize dataset (v2→v3→v4 pipeline)."""
    global dataset, max_idx

    dataset = dataset_compound_tokenize_all(
        json_dir=json_dir,
        rts_codebook_dir=rts_codebook_dir,
        canonical=True,
        detect_closed=False,
    )
    max_idx = len(dataset) - 1
    print(f"Loaded dataset with {len(dataset)} samples")


def process_sample(idx: int):
    """Load raw JSON and detokenized surfaces for a single index."""
    global dataset, current_json_path, vae_model

    # Raw surfaces directly from JSON
    json_path = Path(dataset.dataset_compound.dataset_compound.json_names[idx])
    current_json_path = str(json_path)

    raw_surfaces = load_raw_json(json_path)

    # Tokenize via dataset_v4 and detokenize back
    all_points, all_normals, tokens, bspline_poles, valid = dataset[idx]
    # points_list = list(points_list)
    # normals_list = list(normals_list)

    if not valid:
        print(f"Sample {idx} marked invalid (likely bspline drop)")
        return None, None, None, None

    # If we have a VAE FSQ model and bspline poles, run discrete code processing on patches
    bspline_poles_for_detok = bspline_poles

    assert vae_model is not None, "Should provide the bspline checkpoint for bspline tokenization."

    if (
        vae_model is not None
        and isinstance(bspline_poles, np.ndarray)
        and bspline_poles.size > 0
    ):

        # bspline_poles: (B, 4, 4, 4) where last dim is [x, y, z, w]
        # We take (4, 4, 3) as patches for DCAE-FSQ
        patches_np = bspline_poles[..., :3]  # (B, 4, 4, 3)
        patches = torch.from_numpy(patches_np).float()  # to torch
        patches = einops.rearrange(patches, "b h w c -> b c h w")  # (B, 3, 4, 4)

        with torch.no_grad():
            device = next(vae_model.parameters()).device
            patches_device = patches.to(device)
            # x_recon, z_quantized, indices, metrics = vae_model(patches_device)
            z_quantized, indices = vae_model.encode(patches_device)
            
            # Inject bspline tokens
            tokens_unwarp = dataset.unwarp_codes(tokens)
            tokens_unwarp[tokens_unwarp==-2] = indices.reshape(-1)
            tokens = dataset.warp_codes(tokens_unwarp)
            
            # Extract bspline tokens

            tokens_unwarp = dataset.unwarp_codes(tokens)
            bspline_tokens_recovered = []
            for token in tokens_unwarp:
                if token[0] == 5:  # Bspline surface
                    bspline_tokens_recovered.append(token[1:7]) # six tokens starting from the second index
            

            # Then decode via the bspline vae
            bspline_tokens_recovered = torch.tensor(bspline_tokens_recovered, dtype=torch.int32)
            z_quantized_recover = vae_model.indices_to_latent(bspline_tokens_recovered)
            x_recon = vae_model.decode(z_quantized_recover)


        # Back to (B, 4, 4, 3)
        x_recon_np = einops.rearrange(
            x_recon.cpu(), "b c h w -> b h w c"
        ).numpy()

        # Rebuild poles: (x, y, z) from recon, w=1
        new_poles = np.array(bspline_poles, copy=True)
        new_poles[..., :3] = x_recon_np
        new_poles[..., 3] = 1.0

        bspline_poles_for_detok = new_poles
        print(
            f"Applied VAE FSQ quantization on {bspline_poles.shape[0]} bspline surfaces"
        )


    # Detokenize using (potentially) quantized bspline poles
    detok_surfaces = dataset.detokenize(tokens, bspline_poles_for_detok)

    print(f"Loaded index {idx}")
    print(f"JSON path: {json_path}")
    print(f"Raw surfaces: {len(raw_surfaces)}, Detokenized surfaces: {len(detok_surfaces)}")

    return raw_surfaces, detok_surfaces, all_points, all_normals


def update_visualization():
    """Update polyscope visualization for the current index."""
    global current_idx, pending_idx
    global raw_group, detok_group
    global raw_surfaces_vis, detok_surfaces_vis, raw_surfaces, detok_surfaces
    global current_all_points, current_all_normals

    # Keep pending in sync with current index
    pending_idx = current_idx

    # Clear existing structures (groups remain)
    ps.remove_all_structures()

    # Re-create groups if needed (after remove_all_structures)
    if raw_group is None:
        raw_group = ps.create_group("Raw JSON Surfaces")
    if detok_group is None:
        detok_group = ps.create_group("Detokenized Surfaces")

    # Process current sample
    raw_surfaces, detok_surfaces, all_points, all_normals = process_sample(current_idx)
    if raw_surfaces is None:
        print(f"No valid data for index {current_idx}")
        return

    # Store points/normals for saving
    current_all_points = all_points
    current_all_normals = all_normals

    # Visualize pcd (if available)
    if all_points is not None and all_points.shape[0] > 0:
        points = ps.register_point_cloud("all_points", all_points)
        if all_normals is not None and all_normals.shape[0] == all_points.shape[0]:
            points.add_vector_quantity("normals", all_normals)
        points.set_color([0.5, 0.5, 0.5])
        points.set_radius(0.003)

    # Visualize raw
    try:
        raw_surfaces_vis = visualize_json_interset(
            raw_surfaces,
            plot=True,
            plot_gui=False,
            tol=1e-5,
            ps_header="raw",
        )
    except Exception as e:
        print(f"Error visualizing raw surfaces: {e}")
        raw_surfaces_vis = {}

    for surface_data in raw_surfaces_vis.values():
        if "ps_handler" in surface_data and surface_data.get("surface") is not None:
            surface_data["ps_handler"].add_to_group(raw_group)

    # Visualize detokenized
    try:
        detok_surfaces_vis = visualize_json_interset(
            detok_surfaces,
            plot=True,
            plot_gui=False,
            tol=1e-5,
            ps_header="detok",
        )
    except Exception as e:
        print(f"Error visualizing detokenized surfaces: {e}")
        detok_surfaces_vis = {}

    for surface_data in detok_surfaces_vis.values():
        if "ps_handler" in surface_data and surface_data.get("surface") is not None:
            surface_data["ps_handler"].add_to_group(detok_group)

    print(f"Visualized {len(raw_surfaces_vis)} raw surfaces and {len(detok_surfaces_vis)} detokenized surfaces")


def save_current_sample():
    """Save raw faces, detokenized faces, and point cloud to the specified folder."""
    global save_folder_path, raw_surfaces_vis, detok_surfaces_vis, raw_surfaces, detok_surfaces
    global current_all_points, current_all_normals, current_idx

    if not save_folder_path:
        print("Error: Please specify a save folder path first")
        return

    save_dir = Path(save_folder_path)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory {save_dir}: {e}")
        return

    # Extract TopoDS_Face objects from visualization dictionaries
    raw_faces = []
    detok_faces = []

    for surface_data in raw_surfaces_vis.values():
        if "surface" in surface_data and surface_data["surface"] is not None:
            raw_faces.append(surface_data["surface"])

    for surface_data in detok_surfaces_vis.values():
        if "surface" in surface_data and surface_data["surface"] is not None:
            detok_faces.append(surface_data["surface"])

    # Save raw faces to STEP
    if raw_faces:
        raw_step_path = save_dir / f"raw_surfaces_idx_{current_idx:04d}.step"
        try:
            write_to_step(raw_faces, str(raw_step_path))
            open(save_dir / f"raw_surfaces_idx_{current_idx:04d}.json", "w").write(json.dumps(raw_surfaces, indent=2))
            print(f"Saved {len(raw_faces)} raw faces to {raw_step_path}")
        except Exception as e:
            print(f"Error saving raw faces to STEP: {e}")
    else:
        print("Warning: No raw faces to save")

    # Save detokenized faces to STEP
    if detok_faces:
        detok_step_path = save_dir / f"detokenized_surfaces_idx_{current_idx:04d}.step"
        try:
            write_to_step(detok_faces, str(detok_step_path))
            open(save_dir / f"detokenized_surfaces_idx_{current_idx:04d}.json", "w").write(json.dumps(detok_surfaces, indent=2))
            print(f"Saved {len(detok_faces)} detokenized faces to {detok_step_path}")
        except Exception as e:
            print(f"Error saving detokenized faces to STEP: {e}")
    else:
        print("Warning: No detokenized faces to save")

    # Save point cloud with normals to PLY using open3d
    if current_all_points is not None and current_all_points.shape[0] > 0:
        ply_path = save_dir / f"pointcloud_idx_{current_idx:04d}.ply"
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(current_all_points)

            if (
                current_all_normals is not None
                and current_all_normals.shape[0] == current_all_points.shape[0]
            ):
                pcd.normals = o3d.utility.Vector3dVector(current_all_normals)

            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"Saved point cloud ({current_all_points.shape[0]} points) to {ply_path}")
        except Exception as e:
            print(f"Error saving point cloud to PLY: {e}")
    else:
        print("Warning: No point cloud data to save")

    print(f"Save complete for index {current_idx}")


def callback():
    """Polyscope UI: slider + input box to control index."""
    global current_idx, max_idx, pending_idx, current_json_path
    global save_folder_path

    psim.Text("Tokenize → Detokenize Visualization (v2→v3→v4)")
    psim.Separator()

    # Show current JSON path
    if current_json_path:
        psim.TextWrapped(f"Current File: {current_json_path}")
        psim.Separator()

    psim.Text("=== Index Controls ===")

    # Slider
    slider_changed, slider_idx = psim.SliderInt("Sample Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()

    # Go To Index with input box + button (same pattern as v3 script)
    input_changed, input_idx = psim.InputInt("Go To Index", pending_idx)
    if input_changed:
        # Clamp into valid range
        pending_idx = max(0, min(max_idx, input_idx))

    psim.SameLine()
    if psim.Button("Go"):
        if pending_idx != current_idx:
            current_idx = pending_idx
            update_visualization()

    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")

    if psim.Button("Refresh Current Sample"):
        update_visualization()

    # Save controls
    psim.Separator()
    psim.Text("=== Save Controls ===")
    
    # Input box for save folder path (same pattern as test_vae_v2.py)
    changed, save_folder_path = psim.InputText("Save Folder Path", save_folder_path)
    
    if save_folder_path:
        psim.TextWrapped(f"Save to: {save_folder_path}")
    
    psim.Separator()
    if psim.Button("Save Current Sample"):
        save_current_sample()


def main():
    global current_idx, vae_model
    global raw_group, detok_group

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dir",
        type=str,
        required=True,
        help="Folder containing JSON + NPZ pairs",
    )
    parser.add_argument(
        "--rts_codebook_dir",
        type=str,
        required=True,
        help="Folder containing cb_rotation.pkl, cb_translation.pkl, cb_scale.pkl",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Initial dataset index to visualize (can change via UI)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to VAE v4 DC-AE FSQ config (for bspline poles quantization)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional explicit checkpoint for VAE model (overrides config if provided)",
    )
    args = parser.parse_args()

    # Load VAE FSQ model (for bspline poles)
    print(f"Loading VAE FSQ config from: {args.config}")
    cfg = OmegaConf.load(args.config)
    vae_model = load_model_from_config(cfg)

    # Optionally override checkpoint
    if args.checkpoint_path:
        print(f"Overriding model checkpoint from: {args.checkpoint_path}")
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and ("ema_model" in ckpt or "ema" in ckpt):
            ema_key = "ema" if "ema" in ckpt else "ema_model"
            ema_model_state = ckpt[ema_key]
            ema_model_state = {
                k.replace("ema_model.", "").replace("ema.", ""): v
                for k, v in ema_model_state.items()
            }
            vae_model.load_state_dict(ema_model_state, strict=False)
            print("Loaded EMA model weights.")
        elif isinstance(ckpt, dict) and "model" in ckpt:
            vae_model.load_state_dict(ckpt["model"])
            print("Loaded model weights from 'model' key.")
        else:
            vae_model.load_state_dict(ckpt)
            print("Loaded raw model state_dict.")

    vae_model.eval()

    # Build dataset
    build_dataset(json_dir=str(Path(args.json_dir)), rts_codebook_dir=str(Path(args.rts_codebook_dir)))

    # Clamp initial index
    if args.index < 0 or args.index > max_idx:
        print(f"Warning: index {args.index} out of range [0, {max_idx}], clamping.")
        current_idx = max(0, min(max_idx, args.index))
    else:
        current_idx = args.index

    # Initialize polyscope
    ps.init()
    raw_group = ps.create_group("Raw JSON Surfaces")
    detok_group = ps.create_group("Detokenized Surfaces")

    # Initial visualization
    update_visualization()

    # Attach UI
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

