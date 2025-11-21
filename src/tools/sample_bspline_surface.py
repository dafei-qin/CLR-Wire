import argparse
import bisect
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Tuple

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import (
    TColStd_Array1OfInteger,
    TColStd_Array1OfReal,
    TColStd_Array2OfReal,
)
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance

from utils.surface import extract_mesh_from_face


_dataset = None
_valid_indices = []
_index_to_valid_pos = {}
_current_valid_pos = 0
_pending_dataset_idx = 0
_samples_per_interval = 4
_linear_deflection = 0.05
_angular_deflection = 0.1
_status_message = ""


def _to_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.detach().cpu().item())
    return int(value)


def _to_bool(value) -> bool:
    if torch.is_tensor(value):
        return bool(value.detach().cpu().item())
    return bool(value)


def _to_numpy(array) -> np.ndarray:
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def gaps_to_knots(knots_gap_vec: np.ndarray, num_knots: int) -> np.ndarray:
    """
    Convert the dataset's gap-encoded knot vector back to cumulative form.
    """
    gaps = np.asarray(knots_gap_vec[:num_knots], dtype=np.float64)
    gaps = np.clip(gaps, a_min=0.0, a_max=None)
    knots = np.cumsum(gaps)
    if knots[-1] > 0:
        knots = knots / knots[-1]
    return knots


def _build_occ_surface_from_sample(sample) -> Tuple[Geom_BSplineSurface, np.ndarray, np.ndarray]:
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

    if not _to_bool(valid):
        raise ValueError("Invalid sample provided by dataset.")

    u_degree = _to_int(u_degree)
    v_degree = _to_int(v_degree)
    num_poles_u = _to_int(num_poles_u)
    num_poles_v = _to_int(num_poles_v)
    num_knots_u = _to_int(num_knots_u)
    num_knots_v = _to_int(num_knots_v)
    is_u_periodic = _to_bool(is_u_periodic)
    is_v_periodic = _to_bool(is_v_periodic)

    u_knots_gap = _to_numpy(u_knots_list)[:num_knots_u]
    v_knots_gap = _to_numpy(v_knots_list)[:num_knots_v]
    u_knots = gaps_to_knots(u_knots_gap, num_knots_u)
    v_knots = gaps_to_knots(v_knots_gap, num_knots_v)

    u_mults = _to_numpy(u_mults_list)[:num_knots_u].astype(np.int32)
    v_mults = _to_numpy(v_mults_list)[:num_knots_v].astype(np.int32)
    poles_np = _to_numpy(poles)[:num_poles_u, :num_poles_v, :]

    ctrl_pts = TColgp_Array2OfPnt(1, num_poles_u, 1, num_poles_v)
    weights = TColStd_Array2OfReal(1, num_poles_u, 1, num_poles_v)
    for i in range(num_poles_u):
        for j in range(num_poles_v):
            x, y, z, w = poles_np[i, j]
            ctrl_pts.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))
            weights.SetValue(i + 1, j + 1, float(max(w, 1e-6)))

    occ_u_knots = TColStd_Array1OfReal(1, num_knots_u)
    occ_v_knots = TColStd_Array1OfReal(1, num_knots_v)
    for idx, knot in enumerate(u_knots):
        occ_u_knots.SetValue(idx + 1, float(knot))
    for idx, knot in enumerate(v_knots):
        occ_v_knots.SetValue(idx + 1, float(knot))

    occ_u_mults = TColStd_Array1OfInteger(1, num_knots_u)
    occ_v_mults = TColStd_Array1OfInteger(1, num_knots_v)
    for idx, mult in enumerate(u_mults):
        occ_u_mults.SetValue(idx + 1, int(mult))
    for idx, mult in enumerate(v_mults):
        occ_v_mults.SetValue(idx + 1, int(mult))

    surface = Geom_BSplineSurface(
        ctrl_pts,
        weights,
        occ_u_knots,
        occ_v_knots,
        occ_u_mults,
        occ_v_mults,
        u_degree,
        v_degree,
        is_u_periodic,
        is_v_periodic,
    )
    return surface, u_knots, v_knots


def _build_param_samples(knots: np.ndarray, samples_per_interval: int) -> np.ndarray:
    if samples_per_interval < 1:
        raise ValueError("samples_per_interval must be >= 1.")
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("At least two knots are required to define intervals.")

    params = []
    for idx in range(len(knots) - 1):
        start = float(knots[idx])
        end = float(knots[idx + 1])
        if np.isclose(end, start):
            params.extend([start] * samples_per_interval)
            continue
        local = np.linspace(start, end, samples_per_interval, dtype=np.float64)
        params.extend(local.tolist())
    return np.asarray(params, dtype=np.float64)


def sample_bspline_surface_from_dataset(
    sample,
    samples_per_interval: int = 4,
    flatten: bool = True,
) -> np.ndarray:
    """
    Sample a dataset-provided bspline surface using OCC evaluation.

    Args:
        sample: One element returned by dataset_bspline.__getitem__.
        samples_per_interval: Number of evaluation points per knot span (default 4).
        flatten: When True, returns (N, 3), otherwise returns (Nv, Nu, 3).

    Returns:
        numpy.ndarray containing the sampled XYZ coordinates.
    """
    surface, u_knots, v_knots = _build_occ_surface_from_sample(sample)
    u_params = _build_param_samples(u_knots, samples_per_interval)
    v_params = _build_param_samples(v_knots, samples_per_interval)

    grid = np.zeros((len(v_params), len(u_params), 3), dtype=np.float64)
    for vi, v_val in enumerate(v_params):
        for ui, u_val in enumerate(u_params):
            point = surface.Value(u_val, v_val)
            grid[vi, ui] = [point.X(), point.Y(), point.Z()]

    if flatten:
        return grid.reshape(-1, 3)
    return grid


def build_triangulated_mesh_from_sample(
    sample,
    linear_deflection: float = 0.05,
    angular_deflection: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a triangulated mesh from a dataset sample using pythonOCC meshing utilities.
    """
    surface, _, _ = _build_occ_surface_from_sample(sample)
    face_builder = BRepBuilderAPI_MakeFace(surface, 1e-7)
    face = face_builder.Face()
    tol_fixer = ShapeFix_ShapeTolerance()
    tol_fixer.SetTolerance(face, 1e-6)
    mesher = BRepMesh_IncrementalMesh(face, linear_deflection, True, angular_deflection)
    mesher.Perform()
    vertices, faces = extract_mesh_from_face(face)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Failed to generate a valid mesh for the bspline surface.")
    return vertices, faces


def _set_status(message: str):
    global _status_message
    _status_message = message.strip()


def _snap_to_valid_position(target_idx: int) -> int:
    if not _valid_indices:
        raise ValueError("No valid surfaces available.")
    total = len(_dataset)
    clamped = max(0, min(target_idx, total - 1))
    pos = bisect.bisect_left(_valid_indices, clamped)
    if pos >= len(_valid_indices):
        pos = len(_valid_indices) - 1
    return pos


def _refresh_scene(status_prefix: str = ""):
    if not _valid_indices:
        _set_status("No valid surfaces to visualize.")
        return

    sample_idx = _valid_indices[_current_valid_pos]
    sample = _dataset[sample_idx]
    try:
        vertices, faces = build_triangulated_mesh_from_sample(
            sample,
            linear_deflection=_linear_deflection,
            angular_deflection=_angular_deflection,
        )
        grid_points = sample_bspline_surface_from_dataset(
            sample,
            samples_per_interval=_samples_per_interval,
            flatten=False,
        )
        sample_points = grid_points.reshape(-1, 3)
        centroid = sample_points.mean(axis=0)

        nv, nu, _ = grid_points.shape
        center_v = nv // 2
        center_u = nu // 2

        def _neighbor_indices(index: int, length: int):
            if length <= 1:
                return None
            if length == 2:
                return 0, 1
            prev_idx = max(index - 1, 0)
            next_idx = min(index + 1, length - 1)
            if prev_idx == next_idx:
                if next_idx < length - 1:
                    next_idx += 1
                elif prev_idx > 0:
                    prev_idx -= 1
            return prev_idx, next_idx

        u_vec = np.zeros(3, dtype=np.float64)
        neighbors_u = _neighbor_indices(center_u, nu)
        if neighbors_u is not None:
            prev_u, next_u = neighbors_u
            u_vec = grid_points[center_v, next_u] - grid_points[center_v, prev_u]

        v_vec = np.zeros(3, dtype=np.float64)
        neighbors_v = _neighbor_indices(center_v, nv)
        if neighbors_v is not None:
            prev_v, next_v = neighbors_v
            v_vec = grid_points[next_v, center_u] - grid_points[prev_v, center_u]

        normal_vec = np.cross(u_vec, v_vec)
        norm_val = np.linalg.norm(normal_vec)
        if norm_val > 1e-8:
            normal_vec = normal_vec / norm_val
        else:
            normal_vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    except Exception as exc:
        _set_status(
            (status_prefix + "\n" if status_prefix else "")
            + f"Failed to visualize index {sample_idx}: {exc}"
        )
        return

    ps.remove_all_structures()
    surface_name = f"bspline_surface_{sample_idx:05d}"
    samples_name = f"bspline_samples_{sample_idx:05d}"
    surface_mesh = ps.register_surface_mesh(surface_name, vertices, faces, transparency=0.6)
    if faces.size > 0:
        normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        areas = np.linalg.norm(normals, axis=1) * 0.5
        surface_mesh.add_scalar_quantity("face_area", areas, defined_on="faces")
    sample_cloud = ps.register_point_cloud(samples_name, sample_points, radius=0.004, color=(0.1, 0.6, 0.9))
    # sample_cloud.set_color((0.1, 0.6, 0.9))
    centroid_name = f"bspline_centroid_{sample_idx:05d}"
    centroid_cloud = ps.register_point_cloud(centroid_name, centroid[np.newaxis, :], radius=0.008)
    centroid_cloud.set_color((1.0, 0.85, 0.2))
    normal_quantity = centroid_cloud.add_vector_quantity(
        f"centroid_normal_{sample_idx:05d}",
        normal_vec[np.newaxis, :],
        enabled=True,
        color=(0.95, 0.3, 0.2),
    )
    # normal_quantity.set_color((0.95, 0.3, 0.2))
    # ps.set_window_title(f"BSpline Surface Viewer — index {sample_idx}")

    base_message = (
        f"Showing dataset index {sample_idx} "
        f"({_current_valid_pos + 1} / {len(_valid_indices)} valid surfaces)."
    )
    full_message = base_message if not status_prefix else f"{status_prefix}\n{base_message}"
    _set_status(full_message)


def _set_current_valid_position(pos: int, status_prefix: str = ""):
    global _current_valid_pos, _pending_dataset_idx
    if not _valid_indices:
        _set_status("No valid surfaces to visualize.")
        return
    pos = max(0, min(pos, len(_valid_indices) - 1))
    _current_valid_pos = pos
    _pending_dataset_idx = _valid_indices[pos]
    _refresh_scene(status_prefix=status_prefix)


def _show_dataset_index(request_idx: int, announce_skip: bool = False):
    if not _valid_indices:
        _set_status("No valid surfaces to visualize.")
        return
    total = len(_dataset)
    if total == 0:
        _set_status("Dataset is empty.")
        return
    clamped = max(0, min(request_idx, total - 1))
    if clamped in _index_to_valid_pos:
        pos = _index_to_valid_pos[clamped]
        status_prefix = ""
    else:
        pos = _snap_to_valid_position(clamped)
        snapped_idx = _valid_indices[pos]
        status_prefix = (
            f"Index {clamped} is invalid; showing {snapped_idx} instead."
            if announce_skip
            else ""
        )
    _set_current_valid_position(pos, status_prefix=status_prefix)


def _polyscope_callback():
    if _dataset is None:
        psim.Text("Dataset not loaded.")
        return

    total = len(_dataset)
    valid_count = len(_valid_indices)
    psim.Text(f"Dataset size: {total}")
    psim.Text(f"Valid surfaces: {valid_count}")
    psim.Text(f"Current valid rank: {_current_valid_pos + 1} / {max(valid_count, 1)}")
    psim.Separator()

    changed_input, input_idx = psim.InputInt("Dataset Index", _pending_dataset_idx)
    if changed_input:
        _show_dataset_index(input_idx, announce_skip=True)

    slider_max = max(0, total - 1)
    changed_slider, slider_idx = psim.SliderInt("Index Slider", _pending_dataset_idx, 0, slider_max)
    if changed_slider:
        _show_dataset_index(slider_idx, announce_skip=True)

    psim.Separator()
    psim.TextWrapped(_status_message or "Ready.")


def _resolve_path(path_value: str, project_root: Path) -> str:
    if not path_value:
        return ""
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return str(candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize BSpline samples from dataset_bspline.")
    parser.add_argument("--path_file", type=str, default="assets/all_bspline_paths_test.txt", help="Text file with npy paths.")
    parser.add_argument("--data_dir", type=str, default="", help="Directory override containing npy files.")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize.")
    parser.add_argument("--num_surfaces", type=int, default=-1, help="Limit number of surfaces loaded from dataset.")
    parser.add_argument("--samples_per_interval", type=int, default=4, help="Samples per knot interval (default matches spec).")
    parser.add_argument("--linear_deflection", type=float, default=0.05, help="Linear deflection for OCC mesher.")
    parser.add_argument("--angular_deflection", type=float, default=0.1, help="Angular deflection for OCC mesher.")
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--max_num_u_knots", type=int, default=64)
    parser.add_argument("--max_num_v_knots", type=int, default=32)
    parser.add_argument("--max_num_u_poles", type=int, default=64)
    parser.add_argument("--max_num_v_poles", type=int, default=32)
    parser.add_argument("--replica", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.dataset.dataset_bspline import dataset_bspline

    dataset_kwargs = {
        "path_file": _resolve_path(args.path_file, project_root) if args.path_file else "",
        "data_dir_override": _resolve_path(args.data_dir, project_root) if args.data_dir else "",
        "num_surfaces": args.num_surfaces,
        "replica": args.replica,
        "max_degree": args.max_degree,
        "max_num_u_knots": args.max_num_u_knots,
        "max_num_v_knots": args.max_num_v_knots,
        "max_num_u_poles": args.max_num_u_poles,
        "max_num_v_poles": args.max_num_v_poles,
    }

    if not dataset_kwargs["path_file"] and not dataset_kwargs["data_dir_override"]:
        raise ValueError("Provide --path_file or --data_dir.")

    dataset = dataset_bspline(**dataset_kwargs)
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is empty. Check path_file/data_dir arguments.")

    if args.samples_per_interval < 1:
        raise ValueError("samples_per_interval must be >= 1.")

    print("Scanning dataset for valid bspline surfaces...")
    valid_indices = []
    for idx in range(dataset_size):
        sample = dataset[idx]
        if bool(sample[-1]):
            valid_indices.append(idx)

    if not valid_indices:
        raise ValueError("No valid bspline surfaces found in the provided dataset.")

    global _dataset, _valid_indices, _index_to_valid_pos
    global _samples_per_interval, _linear_deflection, _angular_deflection
    global _pending_dataset_idx
    _dataset = dataset
    _valid_indices = valid_indices
    _index_to_valid_pos = {ds_idx: pos for pos, ds_idx in enumerate(_valid_indices)}
    _samples_per_interval = args.samples_per_interval
    _linear_deflection = args.linear_deflection
    _angular_deflection = args.angular_deflection
    _pending_dataset_idx = _valid_indices[0]

    desired_idx = args.index
    status_prefix = ""
    if desired_idx in _index_to_valid_pos:
        initial_pos = _index_to_valid_pos[desired_idx]
    else:
        initial_pos = _snap_to_valid_position(desired_idx)
        snapped_idx = _valid_indices[initial_pos]
        clamped = max(0, min(desired_idx, dataset_size - 1))
        if snapped_idx != clamped:
            status_prefix = f"Index {clamped} is invalid; showing {snapped_idx} instead."

    ps.init()
    ps.set_ground_plane_mode("tile")
    ps.set_user_callback(_polyscope_callback)
    _set_current_valid_position(initial_pos, status_prefix=status_prefix)
    ps.show()


if __name__ == "__main__":
    main()