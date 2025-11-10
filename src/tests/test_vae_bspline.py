import sys
import json
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_bspline import dataset_bspline
from src.vae.vae_bspline import BSplineVAE
from utils.surface import visualize_json_interset


def to_python_int(x: torch.Tensor) -> int:
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def to_python_bool(x: torch.Tensor) -> bool:
    if isinstance(x, torch.Tensor):
        # handle shapes (1,), (1,1) gracefully
        return bool(x.squeeze().item())
    return bool(x)


def gaps_to_knots(knots_gap_vec: np.ndarray, num_knots: int) -> np.ndarray:
    """
    Convert our gap-encoded knots (starts with 0, then normalized gaps) to cumulative knots (monotonic nondecreasing).
    """
    gaps = np.asarray(knots_gap_vec[:num_knots], dtype=np.float64)
    # Ensure nonnegative gaps
    gaps = np.clip(gaps, a_min=0.0, a_max=None)
    # Convert to cumulative positions
    knots = np.cumsum(gaps)
    if knots[-1] > 0:
        knots = knots / knots[-1]
    return knots


def build_bspline_json(
    u_degree: int,
    v_degree: int,
    num_poles_u: int,
    num_poles_v: int,
    num_knots_u: int,
    num_knots_v: int,
    is_u_periodic: bool,
    is_v_periodic: bool,
    u_knots_gap: np.ndarray,
    v_knots_gap: np.ndarray,
    u_mults: np.ndarray,
    v_mults: np.ndarray,
    poles_padded: np.ndarray,
    idx: int,
) -> Dict[str, Any]:
    """
    Compose a bspline surface dict consumable by utils.surface.build_bspline_surface via visualize_json_interset.
    Converts gap-encoded knots to cumulative normalized knot vectors.
    Crops multiplicities and poles to their valid extents.
    """
    u_knots = gaps_to_knots(u_knots_gap, num_knots_u)
    v_knots = gaps_to_knots(v_knots_gap, num_knots_v)

    u_mults_crop = np.asarray(u_mults[:num_knots_u], dtype=np.int32)
    v_mults_crop = np.asarray(v_mults[:num_knots_v], dtype=np.int32)
    poles_crop = np.asarray(poles_padded[:num_poles_u, :num_poles_v, :], dtype=np.float64)

    scalar = [
        int(u_degree),
        int(v_degree),
        int(num_poles_u),
        int(num_poles_v),
        int(num_knots_u),
        int(num_knots_v),
    ]
    scalar = scalar + u_knots.tolist() + v_knots.tolist() + u_mults_crop.tolist() + v_mults_crop.tolist()

    face = {
        "type": "bspline_surface",
        "idx": [idx, idx],
        "orientation": "Forward",
        "scalar": scalar,
        "u_periodic": bool(is_u_periodic),
        "v_periodic": bool(is_v_periodic),
        "poles": poles_crop.tolist(),
    }
    return face


# Globals for UI
_dataset = None
_model = None
_current_idx = 0
_max_idx = 0
_gt_group = None
_rec_group = None
_gt_surfaces = {}
_rec_surfaces = {}
_show_gt = True
_show_rec = True


def load_model_and_dataset(path_file: str, model_path: str, num_surfaces=1000):
    global _dataset, _model, _max_idx
    _dataset = dataset_bspline(path_file=path_file, num_surfaces=num_surfaces)
    _max_idx = len(_dataset) - 1
    _model = BSplineVAE()
    _model.load_state_dict(torch.load(model_path))
    _model.eval()


def sample_to_batch_tensors(sample):
    """
    Prepare a batch of size 1 following the preprocessing used in vae_bspline.__main__.
    """
    (
        u_degree,
        v_degree,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_list,
        v_mults_list,
        poles,
        valid,
    ) = sample

    if not valid:
        return None

    # Batch dim
    u_degree = u_degree.unsqueeze(0).unsqueeze(-1).long()
    v_degree = v_degree.unsqueeze(0).unsqueeze(-1).long()
    num_poles_u = num_poles_u.unsqueeze(0).unsqueeze(-1).long()
    num_poles_v = num_poles_v.unsqueeze(0).unsqueeze(-1).long()
    num_knots_u = num_knots_u.unsqueeze(0).unsqueeze(-1).long()
    num_knots_v = num_knots_v.unsqueeze(0).unsqueeze(-1).long()
    is_u_periodic = is_u_periodic.unsqueeze(0).unsqueeze(-1).long()
    is_v_periodic = is_v_periodic.unsqueeze(0).unsqueeze(-1).long()

    u_knots_list = u_knots_list.unsqueeze(0).float()
    v_knots_list = v_knots_list.unsqueeze(0).float()
    u_mults_list = u_mults_list.unsqueeze(0).long()
    v_mults_list = v_mults_list.unsqueeze(0).long()
    poles = poles.unsqueeze(0).float()

    # 0-based targets for embeddings where needed
    u_degree_input = u_degree.clone() - 1
    v_degree_input = v_degree.clone() - 1
    u_mults_input = u_mults_list.clone()
    v_mults_input = v_mults_list.clone()
    u_mults_input[u_mults_input > 0] -= 1
    v_mults_input[v_mults_input > 0] -= 1

    return (
        u_degree_input,
        v_degree_input,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_input,
        v_mults_input,
        poles,
        # Also return originals for GT JSON
        u_degree,
        v_degree,
        u_mults_list,
        v_mults_list,
    )


def process_index(idx: int):
    """
    Returns (gt_json_list, rec_json_list)
    """
    global _dataset, _model
    sample = _dataset[idx]
    if not sample[-1]:  # 'valid' flag
        print(f"Index {idx} is invalid; skipping.")
        return [], []

    (
        u_degree_in,
        v_degree_in,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_in,
        v_mults_in,
        poles,
        u_degree_gt,
        v_degree_gt,
        u_mults_gt,
        v_mults_gt,
    ) = sample_to_batch_tensors(sample)

    # Encode → z
    with torch.no_grad():
        mu, logvar = _model.encode(
            u_knots_list,
            u_mults_in,
            v_knots_list,
            v_mults_in,
            poles,
            u_degree_in,
            v_degree_in,
            is_u_periodic,
            is_v_periodic,
            num_knots_u.squeeze(-1),
            num_knots_v.squeeze(-1),
            num_poles_u.squeeze(-1),
            num_poles_v.squeeze(-1),
        )
        z = _model.reparameterize(mu, logvar)
        (
            pred_degree_u,
            pred_degree_v,
            pred_periodic_u,
            pred_periodic_v,
            pred_knots_num_u,
            pred_knots_num_v,
            pred_mults_u,
            pred_mults_v,
            pred_num_poles_u,
            pred_num_poles_v,
            pred_knots_u,
            pred_knots_v,
            pred_poles,
        ) = _model.inference(z)

    # Build GT JSON (batch size 1)
    gt_face = build_bspline_json(
        u_degree=to_python_int(u_degree_gt[0, 0]),
        v_degree=to_python_int(v_degree_gt[0, 0]),
        num_poles_u=to_python_int(num_poles_u[0, 0]),
        num_poles_v=to_python_int(num_poles_v[0, 0]),
        num_knots_u=to_python_int(num_knots_u[0, 0]),
        num_knots_v=to_python_int(num_knots_v[0, 0]),
        is_u_periodic=bool(to_python_int(is_u_periodic[0, 0])),
        is_v_periodic=bool(to_python_int(is_v_periodic[0, 0])),
        u_knots_gap=u_knots_list[0].cpu().numpy(),
        v_knots_gap=v_knots_list[0].cpu().numpy(),
        u_mults=u_mults_gt[0].cpu().numpy().astype(np.int32),
        v_mults=v_mults_gt[0].cpu().numpy().astype(np.int32),
        poles_padded=poles[0].cpu().numpy(),
        idx=idx,
    )

    # Build reconstructed JSON (batch size 1)
    # Use predicted counts/mults/knots/poles
    pred_u_deg = to_python_int(pred_degree_u[0])
    pred_v_deg = to_python_int(pred_degree_v[0])
    pred_u_per = to_python_bool(pred_periodic_u[0])
    pred_v_per = to_python_bool(pred_periodic_v[0])
    pred_u_kn_n = to_python_int(pred_knots_num_u[0])
    pred_v_kn_n = to_python_int(pred_knots_num_v[0])
    pred_u_np_n = to_python_int(pred_num_poles_u[0])
    pred_v_np_n = to_python_int(pred_num_poles_v[0])

    rec_face = build_bspline_json(
        u_degree=pred_u_deg,
        v_degree=pred_v_deg,
        num_poles_u=pred_u_np_n,
        num_poles_v=pred_v_np_n,
        num_knots_u=pred_u_kn_n,
        num_knots_v=pred_v_kn_n,
        is_u_periodic=pred_u_per,
        is_v_periodic=pred_v_per,
        u_knots_gap=pred_knots_u[0].cpu().numpy(),
        v_knots_gap=pred_knots_v[0].cpu().numpy(),
        u_mults=np.argmax(pred_mults_u[0].cpu().numpy(), axis=-1) + 1,
        v_mults=np.argmax(pred_mults_v[0].cpu().numpy(), axis=-1) + 1,
        poles_padded=pred_poles[0].cpu().numpy(),
        idx=idx,
    )

    return [gt_face], [rec_face]


def update_visualization():
    global _current_idx, _gt_group, _rec_group, _gt_surfaces, _rec_surfaces, _show_gt, _show_rec
    ps.remove_all_structures()
    gt_list, rec_list = process_index(_current_idx)
    if not gt_list:
        return
    try:
        _gt_surfaces = visualize_json_interset(gt_list, plot=True, plot_gui=False, tol=1e-5, ps_header="gt")
    except ValueError:
        print("GT visualization failed.")
        return
    for _, s in _gt_surfaces.items():
        if "surface" in s and s["surface"] is not None:
            s["ps_handler"].add_to_group(_gt_group)

    try:
        _rec_surfaces = visualize_json_interset(rec_list, plot=True, plot_gui=False, tol=1e-5, ps_header="rec")
    except ValueError:
        print("Reconstruction visualization failed.")
        return
    for _, s in _rec_surfaces.items():
        if "surface" in s and s["surface"] is not None:
            s["ps_handler"].add_to_group(_rec_group)

    _gt_group.set_enabled(_show_gt)
    _rec_group.set_enabled(_show_rec)


def callback():
    global _current_idx, _max_idx, _show_gt, _show_rec
    psim.Text("BSplineVAE Reconstruction Viewer")
    psim.Separator()
    changed, new_idx = psim.SliderInt("Index", _current_idx, 0, _max_idx)
    if changed:
        _current_idx = new_idx
        update_visualization()

    psim.Separator()
    psim.Text(f"Current Index: {_current_idx}")
    psim.Text(f"Max Index: {_max_idx}")

    if _gt_group is not None:
        psim.Separator()
        psim.Text("Visibility")
        changed, _show_gt = psim.Checkbox("Show GT", _show_gt)
        if changed:
            _gt_group.set_enabled(_show_gt)
        changed, _show_rec = psim.Checkbox("Show Reconstructed", _show_rec)
        if changed:
            _rec_group.set_enabled(_show_rec)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/tests/test_vae_bspline.py <path_file_list>  <ckpt_path> <num_surfaces>")
        sys.exit(1)

    load_model_and_dataset(sys.argv[1], sys.argv[2], int(sys.argv[3]))

    ps.init()
    _gt_group = ps.create_group("GT Surfaces")
    _rec_group = ps.create_group("Reconstructed Surfaces")
    ps.set_user_callback(callback)
    update_visualization()
    ps.show()


