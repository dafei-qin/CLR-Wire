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
import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_v4_tokenize_all import dataset_compound_tokenize_all
from utils.surface import visualize_json_interset


def load_raw_json(json_path: Path):
    with open(json_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, required=True, help="Folder containing JSON + NPZ pairs")
    parser.add_argument("--rts_codebook_dir", type=str, required=True, help="Folder containing cb_rotation.pkl, cb_translation.pkl, cb_scale.pkl")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    rts_codebook_dir = Path(args.rts_codebook_dir)

    # Build dataset (uses v2 parsing, v3 tokenization, v4 RTS codebooks)
    dataset = dataset_compound_tokenize_all(
        json_dir=str(json_dir),
        rts_codebook_dir=str(rts_codebook_dir),
        canonical=False,
        detect_closed=False,
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} out of range (len={len(dataset)})")

    # Raw surfaces directly from JSON
    json_path = Path(dataset.dataset_compound.dataset_compound.json_names[args.index])
    raw_surfaces = load_raw_json(json_path)

    # Tokenize via dataset_v4 (this calls v3 tokenization internally) and detokenize back
    points_list, normals_list, masks_list, tokens, bspline_poles, valid = dataset[args.index]
    if not valid:
        raise RuntimeError(f"Sample {args.index} marked invalid (likely bspline drop)")

    detok_surfaces = dataset.detokenize(tokens, bspline_poles)

    # Initialize polyscope
    ps.init()
    raw_group = ps.create_group("Raw JSON Surfaces")
    detok_group = ps.create_group("Detokenized Surfaces")

    # Visualize raw
    raw_vis = visualize_json_interset(raw_surfaces, plot=True, plot_gui=False, tol=1e-5, ps_header="raw")
    for surface_data in raw_vis.values():
        if "ps_handler" in surface_data:
            surface_data["ps_handler"].add_to_group(raw_group)

    # Visualize detokenized
    detok_vis = visualize_json_interset(detok_surfaces, plot=True, plot_gui=False, tol=1e-5, ps_header="detok")
    for surface_data in detok_vis.values():
        if "ps_handler" in surface_data:
            surface_data["ps_handler"].add_to_group(detok_group)

    print(f"Loaded index {args.index}")
    print(f"JSON path: {json_path}")
    print(f"Raw surfaces: {len(raw_surfaces)}, Detokenized surfaces: {len(detok_surfaces)}")

    ps.show()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

