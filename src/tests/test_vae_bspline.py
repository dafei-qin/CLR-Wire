import sys
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from types import SimpleNamespace

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_bspline import dataset_bspline
from src.utils.config import NestedDictToClass, load_config




from utils.surface import visualize_json_interset


def normalize_path(path_value: Optional[str]) -> str:
    """
    Convert a potentially relative path to an absolute string path.
    Returns empty string if the input is None or empty.
    """
    if path_value is None:
        return ""
    path_str = str(path_value).strip()
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return str(path)


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


def build_face_from_prediction(
    idx: int,
    pred_degree_u: torch.Tensor,
    pred_degree_v: torch.Tensor,
    pred_periodic_u: torch.Tensor,
    pred_periodic_v: torch.Tensor,
    pred_knots_num_u: torch.Tensor,
    pred_knots_num_v: torch.Tensor,
    pred_mults_u: torch.Tensor,
    pred_mults_v: torch.Tensor,
    pred_num_poles_u: torch.Tensor,
    pred_num_poles_v: torch.Tensor,
    pred_knots_u: torch.Tensor,
    pred_knots_v: torch.Tensor,
    pred_poles: torch.Tensor,
):
    """Convert decoder predictions (batch size = 1) to a bspline face dict."""
    return build_bspline_json(
        u_degree=to_python_int(pred_degree_u[0]),
        v_degree=to_python_int(pred_degree_v[0]),
        num_poles_u=to_python_int(pred_num_poles_u[0]),
        num_poles_v=to_python_int(pred_num_poles_v[0]),
        num_knots_u=to_python_int(pred_knots_num_u[0]),
        num_knots_v=to_python_int(pred_knots_num_v[0]),
        is_u_periodic=to_python_bool(pred_periodic_u[0]),
        is_v_periodic=to_python_bool(pred_periodic_v[0]),
        u_knots_gap=pred_knots_u[0].detach().cpu().numpy(),
        v_knots_gap=pred_knots_v[0].detach().cpu().numpy(),
        u_mults=pred_mults_u[0].detach().cpu().numpy(),
        v_mults=pred_mults_v[0].detach().cpu().numpy(),
        poles_padded=pred_poles[0].detach().cpu().numpy(),
        idx=idx,
    )


def register_unit_cube():
    """Register a semi-transparent unit cube for spatial reference."""
    if not _ps_initialized:
        return

    try:
        if hasattr(ps, "has_surface_mesh") and ps.has_surface_mesh(_UNIT_CUBE_NAME):
            return
    except AttributeError:
        # Older Polyscope versions may not expose has_surface_mesh; rely on exception handling below.
        pass

    half = 0.5
    cube_vertices = np.array(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    cube_faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    try:
        cube = ps.register_surface_mesh(
            _UNIT_CUBE_NAME,
            cube_vertices,
            cube_faces,
            color=(0.9, 0.9, 0.9),
            smooth_shade=False,
            transparency=0.7,
        )
        if hasattr(cube, "set_edge_color"):
            cube.set_edge_color((0.2, 0.2, 0.2))
    except Exception as exc:
        print(f"Failed to register unit cube reference: {exc}")


def reset_scene():
    """Clear Polyscope structures while preserving the reference cube."""
    if not _ps_initialized:
        return
    ps.remove_all_structures()
    register_unit_cube()


# Globals for UI
_dataset = None
_model = None
_current_idx = 0
_current_filtered_idx = 0
_max_idx = 0
_gt_group = None
_rec_group = None
_resampled_group = None
_gt_surfaces = {}
_rec_surfaces = {}
_resampled_surfaces = {}
_show_gt = True
_show_rec = True
_show_resampled = False
_gt_face = None
_rec_face = None
_resampled_face = None
_filtered_indices = []
_index_metadata = []
_filters = {}
_filter_limits = {
    "num_poles_u": {"min": 0, "max": 0},
    "num_poles_v": {"min": 0, "max": 0},
}
_ps_initialized = False
_UNIT_CUBE_NAME = "unit_cube_reference"


def build_index_metadata():
    """
    Precompute metadata (periodicity & pole counts) for each dataset sample
    to enable fast filtering inside the visualization UI.
    """
    global _index_metadata, _filter_limits

    if _dataset is None:
        _index_metadata = []
        _filter_limits = {
            "num_poles_u": {"min": 0, "max": 0},
            "num_poles_v": {"min": 0, "max": 0},
        }
        return

    _index_metadata = []
    u_poles_values = []
    v_poles_values = []

    for idx in range(len(_dataset)):
        sample = _dataset[idx]
        valid = bool(sample[-1])
        meta = {"valid": valid}

        if valid:
            num_poles_u = to_python_int(sample[2])
            num_poles_v = to_python_int(sample[3])
            is_u_periodic = to_python_bool(sample[6])
            is_v_periodic = to_python_bool(sample[7])

            meta.update(
                {
                    "num_poles_u": num_poles_u,
                    "num_poles_v": num_poles_v,
                    "is_u_periodic": is_u_periodic,
                    "is_v_periodic": is_v_periodic,
                }
            )

            u_poles_values.append(num_poles_u)
            v_poles_values.append(num_poles_v)

        _index_metadata.append(meta)

    def _limits(values):
        if not values:
            return {"min": 0, "max": 0}
        return {"min": int(min(values)), "max": int(max(values))}

    _filter_limits = {
        "num_poles_u": _limits(u_poles_values),
        "num_poles_v": _limits(v_poles_values),
    }


def initialize_filters():
    """Initialize filter defaults using the current dataset statistics."""
    global _filters

    u_limits = _filter_limits.get("num_poles_u", {"min": 0, "max": 0})
    v_limits = _filter_limits.get("num_poles_v", {"min": 0, "max": 0})
    u_min_limit = min(u_limits["min"], u_limits["max"])
    u_max_limit = max(u_limits["min"], u_limits["max"])
    v_min_limit = min(v_limits["min"], v_limits["max"])
    v_max_limit = max(v_limits["min"], v_limits["max"])

    _filters = {
        "include_u_periodic": True,
        "include_u_non_periodic": True,
        "include_v_periodic": True,
        "include_v_non_periodic": True,
        "min_poles_u": u_limits["min"],
        "max_poles_u": u_limits["max"],
        "min_poles_v": v_limits["min"],
        "max_poles_v": v_limits["max"],
    }


def sample_matches_filters(meta: Dict[str, Any]) -> bool:
    """Check whether a metadata entry satisfies the active filters."""
    if not meta.get("valid", False):
        return False

    if meta["is_u_periodic"]:
        if not _filters.get("include_u_periodic", True):
            return False
    else:
        if not _filters.get("include_u_non_periodic", True):
            return False

    if meta["is_v_periodic"]:
        if not _filters.get("include_v_periodic", True):
            return False
    else:
        if not _filters.get("include_v_non_periodic", True):
            return False

    num_poles_u = meta["num_poles_u"]
    num_poles_v = meta["num_poles_v"]

    if num_poles_u < _filters.get("min_poles_u", num_poles_u):
        return False
    if num_poles_u > _filters.get("max_poles_u", num_poles_u):
        return False
    if num_poles_v < _filters.get("min_poles_v", num_poles_v):
        return False
    if num_poles_v > _filters.get("max_poles_v", num_poles_v):
        return False

    return True


def refresh_filtered_indices(preserve_current: bool = True):
    """
    Rebuild the list of dataset indices that satisfy the active filters and
    update the current slider bounds.
    """
    global _filtered_indices, _max_idx, _current_idx, _current_filtered_idx

    if not _index_metadata:
        _filtered_indices = []
        _max_idx = -1
        _current_idx = -1
        _current_filtered_idx = -1
        return

    _filtered_indices = [
        idx for idx, meta in enumerate(_index_metadata) if sample_matches_filters(meta)
    ]
    _max_idx = len(_filtered_indices) - 1

    if not _filtered_indices:
        _current_filtered_idx = -1
        _current_idx = -1
        return

    if preserve_current and _current_idx in _filtered_indices:
        _current_filtered_idx = _filtered_indices.index(_current_idx)
    else:
        _current_filtered_idx = 0

    _current_idx = _filtered_indices[_current_filtered_idx]


def load_model_and_dataset(
    path_file: str,
    model_path: str,
    num_surfaces: int = 1000,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Load dataset and corresponding model weights. When dataset_kwargs/model_kwargs are provided,
    they fully control initialization; otherwise fall back to legacy path-based loading.
    """
    global _dataset, _model, _max_idx

    if dataset_kwargs is not None:
        _dataset = dataset_bspline(**dataset_kwargs)
    else:
        # Legacy behavior using direct path arguments.
        if os.path.isdir(path_file):
            _dataset = dataset_bspline(path_file="", data_dir_override=path_file, num_surfaces=num_surfaces)
        else:
            _dataset = dataset_bspline(path_file=path_file, num_surfaces=num_surfaces)

    build_index_metadata()
    initialize_filters()
    refresh_filtered_indices(preserve_current=False)

    model_kwargs = model_kwargs or {}
    print(model_kwargs)

    _model = BSplineVAE(**model_kwargs)
    checkpoint = torch.load(model_path, map_location="cpu")
    if 'ema_model' in checkpoint:
        ema_model = checkpoint['ema']
        ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
        _model.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights for classification.")
    elif 'model' in checkpoint:
        _model.load_state_dict(checkpoint['model'])
        print("Loaded model weights for classification.")
    else:
        _model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict for classification.")
    
    # _model.load_state_dict(torch.load(model_path, map_location='cpu'))
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
        # z = _model.reparameterize(mu, logvar)
        z = mu
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

    rec_face = build_face_from_prediction(
        idx,
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
    )
    print(f"gt poles: {gt_face['poles']}")
    print(f"rec poles: {rec_face['poles']}")
    if np.array(gt_face['poles']).shape == np.array(rec_face['poles']).shape:
        print(f"poles mse: {np.mean(np.square(np.array(gt_face['poles']) - np.array(rec_face['poles'])))}")
    else:
        print("poles shapes do not match")
    return [gt_face], [rec_face]


def visualize_generated_face(face_data, header: str = "resampled"):
    """
    Helper to visualize generated faces (either resampled-from-data or random latent).
    """
    global _resampled_face, _resampled_surfaces, _resampled_group, _show_resampled

    if face_data is None:
        return [], {}

    _resampled_face = face_data
    resampled_json_data = [face_data]

    try:
        _resampled_surfaces = visualize_json_interset(
            resampled_json_data, plot=True, plot_gui=False, tol=1e-5, ps_header=header
        )
    except ValueError:
        print(f"{header.capitalize()} visualization failed.")
        return [], {}

    if _resampled_group is not None:
        for _, surface in _resampled_surfaces.items():
            if "surface" in surface and surface["surface"] is not None and surface["ps_handler"] is not None:
                surface["ps_handler"].add_to_group(_resampled_group)

    # Visualize poles
    if _resampled_face is not None:
        poles = np.array(_resampled_face["poles"])[..., :3]
        poles_flat = poles.reshape(-1, 3)
        poles_cloud = ps.register_point_cloud(f"{header}_poles", poles_flat, radius=0.005)
        if _resampled_group is not None:
            poles_cloud.add_to_group(_resampled_group)
        poles_cloud.set_color((0.0, 0.0, 1.0))

    if _resampled_group is not None:
        _resampled_group.set_enabled(_show_resampled)

    return resampled_json_data, _resampled_surfaces


def resample_model():
    """Generate new samples from the VAE's latent space"""
    global _model, _resampled_surfaces, _current_idx, _dataset, _resampled_face, _resampled_group, _show_resampled
    
    if _model is None:
        print("Model not loaded yet!")
        return [], {}

    if _dataset is None or _current_idx < 0 or _current_idx >= len(_dataset):
        print("No valid sample selected for resampling.")
        return [], {}
    
    sample = _dataset[_current_idx]
    if not sample[-1]:  # 'valid' flag
        print(f"Index {_current_idx} is invalid; skipping resample.")
        return [], {}
    
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
    
    with torch.no_grad():
        # Encode to get mu and logvar
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
        # Reparameterize to sample from latent space
        z_random = _model.reparameterize(mu, logvar)
        
        # Run inference to get predictions
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
        ) = _model.inference(z_random)
    
    # Build resampled JSON
    pred_u_deg = to_python_int(pred_degree_u[0])
    pred_v_deg = to_python_int(pred_degree_v[0])
    pred_u_per = to_python_bool(pred_periodic_u[0])
    pred_v_per = to_python_bool(pred_periodic_v[0])
    pred_u_kn_n = to_python_int(pred_knots_num_u[0])
    pred_v_kn_n = to_python_int(pred_knots_num_v[0])
    pred_u_np_n = to_python_int(pred_num_poles_u[0])
    pred_v_np_n = to_python_int(pred_num_poles_v[0])
    
    resampled_face = build_face_from_prediction(
        _current_idx,
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
    )

    return visualize_generated_face(resampled_face, header="resampled")


def random_sample_model():
    """Sample a fresh latent vector z ~ N(0, 1) and decode without conditioning on data."""
    global _model

    if _model is None:
        print("Model not loaded yet!")
        return [], {}

    device = next(_model.parameters()).device
    dtype = next(_model.parameters()).dtype
    latent_dim = getattr(_model, "embd_dim", None)
    mean_latent = getattr(_model, "mean_latent", True)
    if latent_dim is None:
        latent_dim = _model.fc_mu.out_features

    with torch.no_grad():
        if mean_latent:
            z = torch.randn(1, latent_dim, device=device, dtype=dtype)
        else:
            query_dim = _model.latent_queries.shape[0]
            z = torch.randn(1, query_dim // 8 * latent_dim, device=device, dtype=dtype)
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
    print('Randomly sampled a z from N(0, 1)', 'pred_degree_u: ', pred_degree_u, 'pred_degree_v: ', pred_degree_v, 'pred_periodic_u: ', pred_periodic_u, 'pred_periodic_v: ', pred_periodic_v, 'pred_knots_num_u: ', pred_knots_num_u, 'pred_knots_num_v: ', pred_knots_num_v, 'pred_num_poles_u: ', pred_num_poles_u, 'pred_num_poles_v: ', pred_num_poles_v)
    print('pred_mults_u: ', pred_mults_u, 'pred_mults_v: ', pred_mults_v, 'pred_knots_u: ', pred_knots_u, 'pred_knots_v: ', pred_knots_v, 'pred_poles: ', pred_poles)
    random_face = build_face_from_prediction(
        -1,
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
    )

    return visualize_generated_face(random_face, header="random")


def update_visualization():
    global _current_idx, _gt_group, _rec_group, _gt_surfaces, _rec_surfaces, _show_gt, _show_rec, _gt_face, _rec_face

    if _dataset is None or _current_idx < 0 or _current_idx >= len(_dataset):
        print("No valid index selected; skipping visualization.")
        return

    reset_scene()
    gt_list, rec_list = process_index(_current_idx)
    if not gt_list:
        return
    
    # Store the face data for UI display
    _gt_face = gt_list[0] if gt_list else None
    _rec_face = rec_list[0] if rec_list else None
    
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

    # Visualize control poles as point clouds
    if _gt_face is not None:
        gt_poles = np.array(_gt_face["poles"])[..., :3]
        print(gt_poles.shape)
        # Reshape poles from (num_poles_u, num_poles_v, 3) to (num_poles_u * num_poles_v, 3)
        gt_poles_flat = gt_poles.reshape(-1, 3)
        gt_poles_cloud = ps.register_point_cloud("gt_poles", gt_poles_flat, radius=0.005)
        gt_poles_cloud.add_to_group(_gt_group)
        gt_poles_cloud.set_color((0.0, 1.0, 0.0))  # Green for GT poles
    
    if _rec_face is not None:
        rec_poles = np.array(_rec_face["poles"])[..., :3]
        # Reshape poles from (num_poles_u, num_poles_v, 3) to (num_poles_u * num_poles_v, 3)
        rec_poles_flat = rec_poles.reshape(-1, 3)
        rec_poles_cloud = ps.register_point_cloud("rec_poles", rec_poles_flat, radius=0.005)
        rec_poles_cloud.add_to_group(_rec_group)
        rec_poles_cloud.set_color((1.0, 0.0, 0.0))  # Red for reconstructed poles

    _gt_group.set_enabled(_show_gt)
    _rec_group.set_enabled(_show_rec)


def render_filter_controls() -> bool:
    """Render filter UI elements and return True when filters changed."""
    global _filters

    if not _filters:
        return False

    filters_changed = False

    psim.Separator()
    psim.Text("Filters")
    total_valid = sum(1 for meta in _index_metadata if meta.get("valid", False))
    psim.Text(f"Matching Surfaces: {len(_filtered_indices)} / {total_valid}")

    changed, val = psim.Checkbox("Include U Periodic", _filters["include_u_periodic"])
    if changed:
        _filters["include_u_periodic"] = val
        filters_changed = True

    changed, val = psim.Checkbox(
        "Include U Non-Periodic", _filters["include_u_non_periodic"]
    )
    if changed:
        _filters["include_u_non_periodic"] = val
        filters_changed = True

    changed, val = psim.Checkbox("Include V Periodic", _filters["include_v_periodic"])
    if changed:
        _filters["include_v_periodic"] = val
        filters_changed = True

    changed, val = psim.Checkbox(
        "Include V Non-Periodic", _filters["include_v_non_periodic"]
    )
    if changed:
        _filters["include_v_non_periodic"] = val
        filters_changed = True

    u_limits = _filter_limits.get("num_poles_u", {"min": 0, "max": 0})
    v_limits = _filter_limits.get("num_poles_v", {"min": 0, "max": 0})
    u_min_limit = min(u_limits["min"], u_limits["max"])
    u_max_limit = max(u_limits["min"], u_limits["max"])
    v_min_limit = min(v_limits["min"], v_limits["max"])
    v_max_limit = max(v_limits["min"], v_limits["max"])

    min_u_prev = _filters["min_poles_u"]
    max_u_prev = _filters["max_poles_u"]
    min_v_prev = _filters["min_poles_v"]
    max_v_prev = _filters["max_poles_v"]

    changed_min_u, min_u = psim.SliderInt(
        "Min # Poles U", min_u_prev, u_min_limit, u_max_limit
    )
    changed_max_u, max_u = psim.SliderInt(
        "Max # Poles U", max_u_prev, u_min_limit, u_max_limit
    )

    if min_u > max_u:
        if changed_min_u and not changed_max_u:
            max_u = min_u
        else:
            min_u = max_u

    if changed_min_u or changed_max_u:
        if min_u != min_u_prev or max_u != max_u_prev:
            _filters["min_poles_u"] = min_u
            _filters["max_poles_u"] = max_u
            filters_changed = True

    changed_min_v, min_v = psim.SliderInt(
        "Min # Poles V", min_v_prev, v_min_limit, v_max_limit
    )
    changed_max_v, max_v = psim.SliderInt(
        "Max # Poles V", max_v_prev, v_min_limit, v_max_limit
    )

    if min_v > max_v:
        if changed_min_v and not changed_max_v:
            max_v = min_v
        else:
            min_v = max_v

    if changed_min_v or changed_max_v:
        if min_v != min_v_prev or max_v != max_v_prev:
            _filters["min_poles_v"] = min_v
            _filters["max_poles_v"] = max_v
            filters_changed = True

    return filters_changed


def callback():
    global _current_idx, _current_filtered_idx, _filtered_indices, _max_idx
    global _show_gt, _show_rec, _show_resampled
    global _gt_face, _rec_face, _resampled_surfaces
    psim.Text("BSplineVAE Reconstruction Viewer")
    psim.Separator()

    filters_changed = render_filter_controls()
    if filters_changed:
        previous_idx = _current_idx
        refresh_filtered_indices(preserve_current=True)
        reset_scene()
        if _current_idx >= 0:
            update_visualization()
        elif previous_idx != -1:
            print("No samples match current filters.")

    psim.Separator()
    if _max_idx >= 0:
        psim.Text(f"Current Dataset Index: {_current_idx}")
        psim.Text(
            f"Filtered Position: {_current_filtered_idx + 1} / {len(_filtered_indices)}"
        )
        changed, new_idx = psim.SliderInt(
            "Filtered Index", _current_filtered_idx, 0, _max_idx
        )
        if changed:
            _current_filtered_idx = new_idx
            _current_idx = _filtered_indices[_current_filtered_idx]
            update_visualization()
    else:
        psim.Text("No samples match the current filters.")
        return

    # Resample button
    psim.Separator()
    psim.Text("Model Controls:")
    if psim.Button("Resample Model"):
        if _current_idx < 0:
            print("No valid sample selected for resampling.")
        else:
            # Remove all structures before resampling
            reset_scene()
            _resampled_surfaces = {}
            resampled_json_data, _resampled_surfaces = resample_model()
    if psim.Button("Randomly Sample"):
        if _model is None:
            print("Model not loaded yet!")
        else:
            reset_scene()
            _resampled_surfaces = {}
            random_json_data, _resampled_surfaces = random_sample_model()

    if _gt_group is not None:
        psim.Separator()
        psim.Text("Visibility")
        changed, _show_gt = psim.Checkbox("Show GT", _show_gt)
        if changed:
            _gt_group.set_enabled(_show_gt)
        changed, _show_rec = psim.Checkbox("Show Reconstructed", _show_rec)
        if changed:
            _rec_group.set_enabled(_show_rec)
        changed, _show_resampled = psim.Checkbox("Show Resampled", _show_resampled)
        if changed:
            if _resampled_group is not None:
                _resampled_group.set_enabled(_show_resampled)
    
    # Display surface details
    if _gt_face is not None and _rec_face is not None:
        psim.Separator()
        psim.Text("=== Surface Details ===")
        
        # Extract scalar information from the face dict
        # scalar format: [u_deg, v_deg, num_poles_u, num_poles_v, num_knots_u, num_knots_v, 
        #                 u_knots..., v_knots..., u_mults..., v_mults...]
        gt_scalar = _gt_face["scalar"]
        rec_scalar = _rec_face["scalar"]
        
        gt_u_deg, gt_v_deg = gt_scalar[0], gt_scalar[1]
        gt_num_poles_u, gt_num_poles_v = gt_scalar[2], gt_scalar[3]
        gt_num_knots_u, gt_num_knots_v = gt_scalar[4], gt_scalar[5]
        
        rec_u_deg, rec_v_deg = rec_scalar[0], rec_scalar[1]
        rec_num_poles_u, rec_num_poles_v = rec_scalar[2], rec_scalar[3]
        rec_num_knots_u, rec_num_knots_v = rec_scalar[4], rec_scalar[5]
        
        # Extract knots and mults
        gt_u_knots_start = 6
        gt_v_knots_start = gt_u_knots_start + gt_num_knots_u
        gt_u_mults_start = gt_v_knots_start + gt_num_knots_v
        gt_v_mults_start = gt_u_mults_start + gt_num_knots_u
        
        rec_u_knots_start = 6
        rec_v_knots_start = rec_u_knots_start + rec_num_knots_u
        rec_u_mults_start = rec_v_knots_start + rec_num_knots_v
        rec_v_mults_start = rec_u_mults_start + rec_num_knots_u
        
        gt_u_mults = gt_scalar[gt_u_mults_start:gt_u_mults_start + gt_num_knots_u]
        gt_v_mults = gt_scalar[gt_v_mults_start:gt_v_mults_start + gt_num_knots_v]
        rec_u_mults = rec_scalar[rec_u_mults_start:rec_u_mults_start + rec_num_knots_u]
        rec_v_mults = rec_scalar[rec_v_mults_start:rec_v_mults_start + rec_num_knots_v]
        
        gt_sum_u_mults = sum(gt_u_mults)
        gt_sum_v_mults = sum(gt_v_mults)
        rec_sum_u_mults = sum(rec_u_mults)
        rec_sum_v_mults = sum(rec_v_mults)
        
        # Display GT info
        psim.Separator()
        psim.Text("--- Ground Truth ---")
        psim.Text(f"U Degree: {gt_u_deg}, V Degree: {gt_v_deg}")
        psim.Text(f"U Periodic: {_gt_face['u_periodic']}, V Periodic: {_gt_face['v_periodic']}")
        psim.Text(f"Num Knots U: {gt_num_knots_u}, Num Knots V: {gt_num_knots_v}")
        psim.Text(f"Num Poles U: {gt_num_poles_u}, Num Poles V: {gt_num_poles_v}")
        psim.Text(f"Sum U Mults: {gt_sum_u_mults}, Sum V Mults: {gt_sum_v_mults}")
        psim.Text(f"U Mults: {gt_u_mults}")
        psim.Text(f"V Mults: {gt_v_mults}")
        
        # Display Reconstructed info
        psim.Separator()
        psim.Text("--- Reconstructed ---")
        psim.Text(f"U Degree: {rec_u_deg}, V Degree: {rec_v_deg}")
        psim.Text(f"U Periodic: {_rec_face['u_periodic']}, V Periodic: {_rec_face['v_periodic']}")
        psim.Text(f"Num Knots U: {rec_num_knots_u}, Num Knots V: {rec_num_knots_v}")
        psim.Text(f"Num Poles U: {rec_num_poles_u}, Num Poles V: {rec_num_poles_v}")
        psim.Text(f"Sum U Mults: {rec_sum_u_mults}, Sum V Mults: {rec_sum_v_mults}")
        psim.Text(f"U Mults: {rec_u_mults}")
        psim.Text(f"V Mults: {rec_v_mults}")
        
        # Compare poles
        psim.Separator()
        psim.Text("--- Poles Comparison ---")
        gt_poles = np.array(_gt_face["poles"])
        rec_poles = np.array(_rec_face["poles"])
        
        psim.Text(f"GT Poles Shape: {gt_poles.shape}")
        psim.Text(f"Rec Poles Shape: {rec_poles.shape}")
        
        if gt_poles.shape == rec_poles.shape:
            mse = np.mean((gt_poles - rec_poles) ** 2)
            psim.Text(f"Poles MSE: {mse:.6e}")
        else:
            psim.Text("Poles shapes do not match - cannot compute MSE")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/train_vae_bspline_v1_full.yaml',
        help='Path to the YAML config file (defaults to train_vae_bspline_v1_full).',
    )
    parser.add_argument('--model_name', type=str, default=None, help='Override model name from config.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Override checkpoint file path.')
    parser.add_argument('--num_surfaces', type=int, default=None, help='Override number of surfaces to visualize.')
    parser.add_argument(
        '--path_file_or_dir',
        type=str,
        default=None,
        help='Override dataset source (text file of paths or directory containing .npy files).',
    )
    args = parser.parse_args()
    # if len(sys.argv) != 4:
    #     print("Usage: python src/tests/test_vae_bspline.py <path_file_or_dir> <ckpt_path> <num_surfaces>")
    #     sys.exit(1)

    cfg = load_config(args.config) if args.config else {}
    cfg_args = NestedDictToClass(cfg) if cfg else SimpleNamespace()

    data_cfg = getattr(cfg_args, "data", SimpleNamespace())
    model_cfg = getattr(cfg_args, "model", SimpleNamespace())

    model_name =  getattr(model_cfg, "name", "vae_bspline_v1")
    if model_name == 'vae_bspline_v1':
        from src.vae.vae_bspline import BSplineVAE as BSplineVAE
        print('Use the model: vae_bspline_v1')
    elif model_name == 'vae_bspline_v3':
        from src.vae.vae_bspline_v3 import BSplineVAE as BSplineVAE
        print('Use the model: vae_bspline_v3')
    elif model_name == "vae_bspline_v4":
        from src.vae.vae_bspline_v4 import BSplineVAE as BSplineVAE
    elif model_name == "vae_bspline_v5":
        from src.vae.vae_bspline_v5 import BSplineVAE as BSplineVAE
    elif model_name == "vae_bspline_v6":
        from src.vae.vae_bspline_v6 import BSplineVAE as BSplineVAE
        print('Use the model: vae_bspline_v6')
    else:
        print('Use the default model: vae_bspline_v1')
        from src.vae.vae_bspline import BSplineVAE as BSplineVAE
        

    def _getattr(obj, attr, default=None):
        return getattr(obj, attr, default) if obj is not None else default

    num_surfaces_cfg = _getattr(data_cfg, "val_num", _getattr(data_cfg, "train_num", -1))
    num_surfaces = args.num_surfaces if args.num_surfaces is not None else (num_surfaces_cfg if num_surfaces_cfg is not None else -1)

    default_path_file = _getattr(data_cfg, "val_file", _getattr(data_cfg, "train_file", ""))
    default_data_dir = _getattr(data_cfg, "val_data_dir_override", _getattr(data_cfg, "train_data_dir_override", ""))

    dataset_path_override = args.path_file_or_dir
    dataset_path_file = ""
    dataset_data_dir = ""

    if dataset_path_override:
        resolved_override = normalize_path(dataset_path_override)
        if os.path.isdir(resolved_override):
            dataset_data_dir = resolved_override
        else:
            dataset_path_file = resolved_override
    else:
        default_dir_resolved = normalize_path(default_data_dir)
        default_path_resolved = normalize_path(default_path_file)
        if default_dir_resolved:
            dataset_data_dir = default_dir_resolved
        elif default_path_resolved:
            dataset_path_file = default_path_resolved

    if not dataset_path_file and not dataset_data_dir:
        raise ValueError("Unable to determine dataset source. Provide it via the config file or --path_file_or_dir.")

    dataset_kwargs = {
        "path_file": dataset_path_file,
        "data_dir_override": dataset_data_dir,
        "num_surfaces": num_surfaces,
        "max_num_u_knots": _getattr(model_cfg, "max_num_u_knots", 64),
        "max_num_v_knots": _getattr(model_cfg, "max_num_v_knots", 32),
        "max_num_u_poles": _getattr(model_cfg, "max_num_u_poles", 64),
        "max_num_v_poles": _getattr(model_cfg, "max_num_v_poles", 32),
        "max_degree": _getattr(model_cfg, "max_degree", 3),
    }

    model_kwarg_keys = [
        "max_degree",
        "embd_dim",
        "num_query",
        "mults_dim",
        "max_num_u_knots",
        "max_num_v_knots",
        "max_num_u_poles",
        "max_num_v_poles",
    ]
    model_kwargs = {key: getattr(model_cfg, key) for key in model_kwarg_keys if hasattr(model_cfg, key)}

    # Ensure the geometric limits align with the dataset padding assumptions.
    for dim_key in ["max_degree", "max_num_u_knots", "max_num_v_knots", "max_num_u_poles", "max_num_v_poles"]:
        if dim_key not in model_kwargs and dim_key in dataset_kwargs:
            model_kwargs[dim_key] = dataset_kwargs[dim_key]

    ckpt_path = normalize_path(args.ckpt_path) if args.ckpt_path else ""
    if not ckpt_path:
        ckpt_folder = _getattr(model_cfg, "checkpoint_folder", "")
        ckpt_file_name = _getattr(model_cfg, "checkpoint_file_name", "")
        if ckpt_folder or ckpt_file_name:
            combined_ckpt = os.path.join(str(ckpt_folder), ckpt_file_name) if ckpt_folder and ckpt_file_name else (ckpt_folder or ckpt_file_name)
            ckpt_path = normalize_path(combined_ckpt)

    if not ckpt_path:
        raise ValueError("Checkpoint path must be provided via the config file or --ckpt_path.")

    load_model_and_dataset(
        dataset_kwargs["path_file"],
        ckpt_path,
        num_surfaces=num_surfaces,
        dataset_kwargs=dataset_kwargs,
        model_kwargs=model_kwargs,
    )

    ps.init()
    _ps_initialized = True
    register_unit_cube()
    _gt_group = ps.create_group("GT Surfaces")
    _rec_group = ps.create_group("Reconstructed Surfaces")
    _resampled_group = ps.create_group("Resampled Surfaces")
    ps.set_user_callback(callback)
    update_visualization()
    ps.show()


