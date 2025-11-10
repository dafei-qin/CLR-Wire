import os
import sys
import json
import numpy as np
from tqdm import tqdm
from occwl.face import Face
from occwl.uvgrid import uvgrid, ugrid

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # root/
sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Looking for utils/surface.py at:", PROJECT_ROOT / "utils" / "surface.py")

# Verify file exists (debug only)
if not (PROJECT_ROOT / "utils" / "surface.py").exists():
    raise FileNotFoundError("❌ utils/surface.py not found! Check path.")

# Now import correctly:
try:
    from utils.surface import (
        build_plane_face,
        build_second_order_surface,
        build_bspline_surface
    )
    print("✅ Successfully imported from utils.surface")
except ImportError as e:
    print("❌ Import failed:", e)
    print("sys.path includes:")
    for p in sys.path:
        print("  -", p)
    sys.exit(1)


# -----------------------------
# 🧩 封装：JSON → TopoDS_Face（仅调用你的函数）
# -----------------------------
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt
def json_surface_to_topods_face(face_data):
    typ = face_data.get("type", "").lower()
    try:
        if typ == "plane":
            face, _, _, _ = build_plane_face(face_data, tol=1e-2)
            return face

        elif typ in ["cylinder", "cone", "torus", "sphere"]:
            face, _, _, _ = build_second_order_surface(face_data, tol=1e-2)
            return face

        elif typ == "bspline_surface":
            face, _, _, _ = build_bspline_surface(
                face_data, 
                tol=1e-2, 
                normalize_knots=False, 
                normalize_surface=False
            )
            return face

        else:
            print(f"⚠️ Unsupported surface type: {typ}")
            return None

    except Exception as e:
        print(f"❌ Failed to build {typ} from JSON: {e}")
        return None


# -----------------------------
# 📐 拟合函数（同前，可放入 utils/fit.py 以后复用）
# -----------------------------
def fit_approx_by_type(points, surface_type: str, **kwargs):
    """
    Fit control points based on surface type.
    
    Args:
        points: (32, 32, 3) numpy array from occwl.uvgrid(..., method="point")
        surface_type: str, e.g., "plane", "cylinder", "bspline_surface", etc.
        **kwargs: e.g., plane_tol, quad_tol, cubic_tol, force_degree
    
    Returns:
        np.ndarray or None:
            - plane: (4, 3)
            - second_order: (9, 3)
            - bspline_surface: (16, 3)
    """
    if points is None or points.shape != (32, 32, 3) or not np.all(np.isfinite(points)):
        return None

    surface_type = surface_type.lower()
    
    # --- Plane: analytic 4 corner points ---
    if surface_type == "plane":
        return None
        # try:
        #     return np.array([
        #         points[0, 0],      # (u_min, v_min)
        #         points[0, -1],     # (u_min, v_max)
        #         points[-1, 0],     # (u_max, v_min)
        #         points[-1, -1]     # (u_max, v_max)
        #     ], dtype=np.float64)
        # except Exception:
        #     return None

    # --- Second-order surfaces: quadratic B-spline (degree=2, 3x3=9 pts) ---
    elif surface_type in {"cylinder", "cone", "sphere", "torus"}:
        deg_min = kwargs.get("quad_degree_min", 2)
        deg_max = kwargs.get("quad_degree_max", 2)
        continuity = kwargs.get("quad_continuity", GeomAbs_C1)
        tolerance = kwargs.get("quad_tol", 5e-3)
        return _fit_bspline(
            points,
            degree_min=deg_min,
            degree_max=deg_max,
            continuity=continuity,
            tolerance=tolerance,
            expected_poles=(3, 3)
        )

    # --- General B-spline: cubic (degree=3, 4x4=16 pts) ---
    elif surface_type == "bspline_surface":
        deg_min = kwargs.get("cubic_degree_min", 3)
        deg_max = kwargs.get("cubic_degree_max", 3)
        continuity = kwargs.get("cubic_continuity", GeomAbs_C2)
        tolerance = kwargs.get("cubic_tol", 1e-3)
        return _fit_bspline(
            points,
            degree_min=deg_min,
            degree_max=deg_max,
            continuity=continuity,
            tolerance=tolerance,
            expected_poles=(4, 4)
        )

    # --- Fallback: cubic (e.g., unknown type) ---
    else:
        return _fit_bspline(
            points,
            degree_min=3,
            degree_max=3,
            continuity=GeomAbs_C2,
            tolerance=1e-3,
            expected_poles=(4, 4)
        )


# -----------------------------
# 🔧 Internal helper: generic B-spline fitting
# -----------------------------
def _fit_bspline(points, degree_min, degree_max, continuity, tolerance, expected_poles):
    """
    Generic B-spline surface fitting with validation.
    Returns control points array or None.
    """
    try:
        # Build OCC array
        arr = TColgp_Array2OfPnt(1, 32, 1, 32)
        for i in range(32):
            for j in range(32):
                x, y, z = points[i, j]
                arr.SetValue(i + 1, j + 1, gp_Pnt(float(x), float(y), float(z)))

        # Fit
        api = GeomAPI_PointsToBSplineSurface(arr, degree_min, degree_max, continuity, tolerance)
        if not api.IsDone():
            return None
        surf = api.Surface()

        nu, nv = expected_poles
        # Validate: degree, pole count, rationality, periodicity
        if not (
            surf.UDegree() == degree_max and
            surf.VDegree() == degree_max and
            surf.NbUPoles() == nu and
            surf.NbVPoles() == nv and
            not surf.IsUPeriodic() and
            not surf.IsVPeriodic() and
            not surf.IsURational() and
            not surf.IsVRational()
        ):
            return None

        # Extract control points
        poles = surf.Poles()
        cps = []
        for u in range(1, nu + 1):
            for v in range(1, nv + 1):
                p = poles.Value(u, v)
                cps.append([p.X(), p.Y(), p.Z()])
        return np.array(cps, dtype=np.float64)  # shape (nu*nv, 3)

    except Exception:
        return None

# -----------------------------
# 🧩 主流程（同前，但使用你的函数）
# -----------------------------
def process_json_file(json_path, out_dir, save_points=False):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            surfaces = json.load(f)
        if not isinstance(surfaces, list):
            raise ValueError("JSON root must be a list of surfaces")

        base = os.path.splitext(os.path.basename(json_path))[0]
        out_json = os.path.join(out_dir, f"{base}.json")
        out_npz = os.path.join(out_dir, f"{base}.npz")
        os.makedirs(out_dir, exist_ok=True)

        metadata = []
        arrays = {}

        for i, surf in enumerate(surfaces):
            # Step 1: Reconstruct TopoDS_Face using YOUR functions ✅
            topods_face = json_surface_to_topods_face(surf)
            if topods_face is None:
                metadata.append({"surface_idx": i, "success": False})
                continue

            # Step 2: Use occwl.uvgrid
            try:
                occwl_face = Face(topods_face)
                points = uvgrid(occwl_face, method="point", num_u=32, num_v=32)
            except Exception as e:
                print(f"⚠️ uvgrid failed on face {i}: {e}")
                points = None

            # Step 3: Fit approximation
            approx = fit_approx_by_type(points, surf.get("type", "unknown"))

            # Record
            meta = {
                "surface_idx": i,
                "type": surf.get("type", "unknown"),
                "has_points": points is not None,
                "has_approx": approx is not None,
            }
            metadata.append(meta)

            if points is not None:
                if save_points:
                    arrays[f"points_{i}"] = points
            if approx is not None:
                arrays[f"approx_{i}"] = approx

        # Save
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        if arrays:
            np.savez(out_npz, **arrays)

        # print(f"✅ {os.path.basename(json_path)} → {len([1 for m in metadata if m.get('has_approx')])} approximations")
        return True

    except Exception as e:
        print(f"💥 Failed {json_path}: {e}")
        return False


# -----------------------------
# 🚀 Main
# -----------------------------
# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description="Fit B-spline approx from JSON surfaces using utils/surface.py")
#     parser.add_argument("--input", required=True, help="Input dir with .json files")
#     parser.add_argument("--output", required=True, help="Output dir")
#     parser.add_argument('--save_points', action='store_true', help='Save points to .npz files')
#     args = parser.parse_args()

#     input_dir = os.path.abspath(args.input)
#     output_dir = os.path.abspath(args.output)

#     json_files = [
#         os.path.join(root, f)
#         for root, _, files in os.walk(input_dir)
#         for f in files if f.lower().endswith(".json")
#     ]

#     if not json_files:
#         print("❌ No .json files found.")
#         return

#     print(f"🔍 Found {len(json_files)} JSON files. Processing sequentially...")
#     success = 0
#     for jf in tqdm(json_files):
#         rel_dir = os.path.relpath(os.path.dirname(jf), input_dir)
#         out_subdir = os.path.join(output_dir, rel_dir)
#         if process_json_file(jf, out_subdir, args.save_points):
#             success += 1

#     print(f"\n✅ Done: {success}/{len(json_files)} succeeded.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Single JSON file to process")
    parser.add_argument("--output-dir", required=True, help="Output directory for .json/.npz")
    parser.add_argument('--save-points', action='store_true')
    args = parser.parse_args()

    # Process single file
    success = process_json_file(args.input_file, args.output_dir, args.save_points)
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main()