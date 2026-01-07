import argparse
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.dataset.dataset_bspline import dataset_bspline  # noqa: E402

from myutils.surface import build_bspline_surface, get_approx_face  # noqa: E402
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GeomAdaptor import GeomAdaptor_Surface

_dataset_raw = None
_dataset_canonical = None
_dataset_size = 0
_current_idx = 0
_status_message = ""

_FACE_COLORS = {
    "original": (0.2, 0.7, 0.3),
    "canonical": (0.2, 0.4, 0.8),
    "reconstructed": (0.9, 0.4, 0.2),
    "approx_fitted": (0.9, 0.2, 0.9),
}


def normalize_path(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def gaps_to_knots(knots_gap_vec: np.ndarray, num_knots: int) -> np.ndarray:
    gaps = np.asarray(knots_gap_vec[:num_knots], dtype=np.float64)
    gaps = np.clip(gaps, 0.0, None)
    knots = np.cumsum(gaps)
    if knots.size > 0 and knots[-1] > 0:
        knots = knots / knots[-1]
    return knots


def tensor_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def extract_sample(sample) -> Dict[str, Any]:
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

    data = {
        "u_degree": int(u_degree.item()),
        "v_degree": int(v_degree.item()),
        "num_poles_u": int(num_poles_u.item()),
        "num_poles_v": int(num_poles_v.item()),
        "num_knots_u": int(num_knots_u.item()),
        "num_knots_v": int(num_knots_v.item()),
        "is_u_periodic": bool(is_u_periodic.item()),
        "is_v_periodic": bool(is_v_periodic.item()),
        "u_knots": tensor_to_numpy(u_knots_list),
        "v_knots": tensor_to_numpy(v_knots_list),
        "u_mults": tensor_to_numpy(u_mults_list),
        "v_mults": tensor_to_numpy(v_mults_list),
        "poles": tensor_to_numpy(poles),
        "valid": bool(valid.item()),
    }
    return data


def build_bspline_face(data: Dict[str, Any], idx: int, poles_override: np.ndarray = None) -> Dict[str, Any]:
    # if not data["valid"]:
    #     raise ValueError(f"Sample {idx} is invalid.")

    u_knots = gaps_to_knots(data["u_knots"], data["num_knots_u"])
    v_knots = gaps_to_knots(data["v_knots"], data["num_knots_v"])
    u_mults = np.asarray(data["u_mults"][: data["num_knots_u"]], dtype=np.int32)
    v_mults = np.asarray(data["v_mults"][: data["num_knots_v"]], dtype=np.int32)
    poles = data["poles"][: data["num_poles_u"], : data["num_poles_v"], :]
    if poles_override is not None:
        poles = poles_override

    scalar = [
        data["u_degree"],
        data["v_degree"],
        data["num_poles_u"],
        data["num_poles_v"],
        data["num_knots_u"],
        data["num_knots_v"],
    ]
    scalar = scalar + u_knots.tolist() + v_knots.tolist() + u_mults.tolist() + v_mults.tolist()

    face = {
        "type": "bspline_surface",
        "idx": [idx, idx],
        "orientation": "Forward",
        "scalar": scalar,
        "u_periodic": data["is_u_periodic"],
        "v_periodic": data["is_v_periodic"],
        "poles": poles.tolist(),
        "valid": data["valid"],
    }
    return face


def invert_canonical_poles(
    canonical_poles: np.ndarray,
    rotation: np.ndarray,
    shift: np.ndarray,
    length: float,
) -> np.ndarray:
    coords = canonical_poles[..., :3].reshape(-1, 3)
    restored = coords @ rotation + shift
    # restored = restored * length
    result = canonical_poles.copy()
    result[..., :3] = restored.reshape(canonical_poles.shape[0], canonical_poles.shape[1], 3)
    return result


def sample_bspline_surface_32x32(face_dict: Dict[str, Any]) -> np.ndarray:
    """Sample 32x32 points from a bspline surface."""
    # Build the surface first
    occ_face, _, _, _ = build_bspline_surface(
        face_dict,
        tol=1e-2,
        normalize_surface=False,
        normalize_knots=False,
    )
    
    # Get the surface from the face
    surface_handle = BRep_Tool.Surface(occ_face)
    adaptor = GeomAdaptor_Surface(surface_handle)
    
    # Get parameter bounds
    u_min = adaptor.FirstUParameter()
    u_max = adaptor.LastUParameter()
    v_min = adaptor.FirstVParameter()
    v_max = adaptor.LastVParameter()
    
    # Sample 32x32 points
    points = np.zeros((32, 32, 3), dtype=np.float64)
    for u_idx in range(32):
        u = u_min + (u_max - u_min) * u_idx / 31.0
        for v_idx in range(32):
            v = v_min + (v_max - v_min) * v_idx / 31.0
            pnt = adaptor.Value(u, v)
            points[u_idx, v_idx, :] = [pnt.X(), pnt.Y(), pnt.Z()]
    
    return points


def _set_status(message: str):
    global _status_message
    _status_message = message


def _register_faces(idx: int, faces: List[Tuple[str, Dict[str, Any]]]):
    ps.remove_all_structures()
    for label, face in faces:
        _, vertices, faces_idx, _ = build_bspline_surface(
            face,
            tol=1e-2,
            normalize_surface=False,
            normalize_knots=False,
        )
        mesh_name = f"{label}_{idx:05d}"
        mesh = ps.register_surface_mesh(mesh_name, vertices, faces_idx, transparency=0.7)
        mesh.set_color(_FACE_COLORS.get(label, (0.7, 0.7, 0.7)))



def register_unit_cube():
    """Register a semi-transparent unit cube for spatial reference."""

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
            'unit_cube',
            cube_vertices,
            cube_faces,
            color=(0.9, 0.9, 0.9),
            smooth_shade=False,
            transparency=0.1,
        )
        if hasattr(cube, "set_edge_color"):
            cube.set_edge_color((0.2, 0.2, 0.2))
    except Exception as exc:
        print(f"Failed to register unit cube reference: {exc}")


def _update_visualization(idx: int):
    global _current_idx
    if idx < 0 or idx >= _dataset_size:
        _set_status(f"Index {idx} out of range [0, {_dataset_size - 1}]")
        return

    raw_sample = _dataset_raw[idx]
    raw_data = extract_sample(raw_sample)
    if not raw_data["valid"]:
        ps.remove_all_structures()
        _set_status(f"Index {idx} is invalid; visualization skipped.")
        _current_idx = idx
        return

    canonical_sample = _dataset_canonical[idx]
    canonical_data = extract_sample(canonical_sample)
    if not canonical_data["valid"]:
        ps.remove_all_structures()
        _set_status(f"Canonical sample index {idx} invalid; visualization skipped.")
        _current_idx = idx
        return

    rotation, shift, length = _dataset_canonical.get_transform(idx)
    rotation = rotation.astype(np.float64)
    shift = shift.astype(np.float64)
    length = length.astype(np.float64)
    canonical_poles = canonical_data["poles"][: canonical_data["num_poles_u"], : canonical_data["num_poles_v"], :]
    restored_poles = invert_canonical_poles(canonical_poles, rotation, shift, length)

    recon_error = np.mean(
        (restored_poles[..., :3] - raw_data["poles"][: raw_data["num_poles_u"], : raw_data["num_poles_v"], :3]) ** 2
    )

    # Sample canonical surface and fit with get_approx_face
    canonical_face_dict = build_bspline_face(canonical_data, idx + 1000)
    try:
        sampled_points = sample_bspline_surface_32x32(canonical_face_dict)
        fitted_control_points = get_approx_face(sampled_points)
        
        # Build fitted poles array (4x4x4 with weights=1.0)
        fitted_poles = np.array(fitted_control_points).reshape(4, 4, 3)
        fitted_poles_with_weights = np.concatenate(
            [fitted_poles, np.ones((4, 4, 1))], axis=-1
        )
        
        # Create face dict for fitted surface
        fitted_data = {
            "u_degree": 3,
            "v_degree": 3,
            "num_poles_u": 4,
            "num_poles_v": 4,
            "num_knots_u": 2,
            "num_knots_v": 2,
            "is_u_periodic": False,
            "is_v_periodic": False,
            "u_knots": np.array([0.0, 1.0]),
            "v_knots": np.array([0.0, 1.0]),
            "u_mults": np.array([4, 4]),
            "v_mults": np.array([4, 4]),
            "poles": fitted_poles_with_weights,
            "valid": True,
        }
        fitted_face_dict = build_bspline_face(fitted_data, idx + 3000)
        
        # Compute approximation error by sampling both surfaces
        canonical_samples = sample_bspline_surface_32x32(canonical_face_dict)
        fitted_samples = sample_bspline_surface_32x32(fitted_face_dict)
        approx_error = np.mean((canonical_samples - fitted_samples) ** 2)
        
        faces = [
            ("original", build_bspline_face(raw_data, idx)),
            ("canonical", canonical_face_dict),
            ("reconstructed", build_bspline_face(canonical_data, idx + 2000, poles_override=restored_poles)),
            ("approx_fitted", fitted_face_dict),
        ]
        
        status_msg = f"Index {idx}: valid · Reconstruction MSE {recon_error:.6e} · Approx MSE {approx_error:.6e}"
    except Exception as exc:
        # If approximation fails, fall back to original visualization
        print(f"Approximation failed for index {idx}: {exc}")
        faces = [
            ("original", build_bspline_face(raw_data, idx)),
            ("canonical", canonical_face_dict),
            ("reconstructed", build_bspline_face(canonical_data, idx + 2000, poles_override=restored_poles)),
        ]
        status_msg = f"Index {idx}: valid · Reconstruction MSE {recon_error:.6e} · Approx failed"
    
    # print(raw_data['poles'] - restored_poles)
    # print(np.linalg.norm(raw_data['poles'] - restored_poles))
    try:
        _register_faces(idx, faces)

        register_unit_cube()

        _set_status(status_msg)
    except ValueError as exc:
        ps.remove_all_structures()
        _set_status(f"Visualization failed for index {idx}: {exc}")
        return

    _current_idx = idx


def _viewer_callback():
    global _current_idx
    if _dataset_size == 0:
        psim.Text("Dataset is empty.")
        return
    psim.Text(f"Dataset size: {_dataset_size}")
    psim.Separator()

    changed_input, new_idx = psim.InputInt("Dataset Index", _current_idx)
    if changed_input:
        new_idx = max(0, min(new_idx, _dataset_size - 1))
        _update_visualization(new_idx)

    changed_slider, slider_idx = psim.SliderInt("Index Slider", _current_idx, 0, _dataset_size - 1)
    if changed_slider:
        _update_visualization(slider_idx)

    psim.Separator()
    psim.TextWrapped(_status_message or "Ready.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize dataset_bspline canonical transform.")
    parser.add_argument("--path_file", type=str, default="assets/all_bspline_paths_test.txt")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_surfaces", type=int, default=-1)
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--max_num_u_knots", type=int, default=64)
    parser.add_argument("--max_num_v_knots", type=int, default=32)
    parser.add_argument("--max_num_u_poles", type=int, default=64)
    parser.add_argument("--max_num_v_poles", type=int, default=32)
    parser.add_argument("--canonical_samples", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_kwargs = {
        "path_file": normalize_path(args.path_file) if args.path_file else "",
        "data_dir_override": normalize_path(args.data_dir) if args.data_dir else "",
        "num_surfaces": args.num_surfaces,
        "max_degree": args.max_degree,
        "max_num_u_knots": args.max_num_u_knots,
        "max_num_v_knots": args.max_num_v_knots,
        "max_num_u_poles": args.max_num_u_poles,
        "max_num_v_poles": args.max_num_v_poles,
    }

    dataset_raw = dataset_bspline(**dataset_kwargs, canonical=False)
    dataset_canonical = dataset_bspline(
        **dataset_kwargs,
        canonical=True,
        canonical_samples_per_interval=args.canonical_samples,
    )

    if len(dataset_raw) == 0:
        raise ValueError("Dataset is empty.")

    global _dataset_raw, _dataset_canonical, _dataset_size, _current_idx
    _dataset_raw = dataset_raw
    _dataset_canonical = dataset_canonical
    _dataset_size = len(dataset_raw)

    ps.init()
    ps.set_ground_plane_mode("tile")
    initial_idx = args.index % _dataset_size
    _update_visualization(initial_idx)
    _current_idx = initial_idx

    ps.set_user_callback(_viewer_callback)
    ps.show()


if __name__ == "__main__":
    main()

