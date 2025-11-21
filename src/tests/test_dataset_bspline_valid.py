import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_bspline import dataset_bspline


def _resolve_path(path_str: Optional[str]) -> str:
    if not path_str:
        return ""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def test_bspline_valid(
    path_file: Optional[str] = None,
    data_dir_override: Optional[str] = None,
    num_surfaces: int = -1,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    max_samples: Optional[int] = None,
):
    """
    Helper that scans the bspline dataset and reports how many surfaces are valid.
    """
    if not path_file and not data_dir_override:
        path_file = "assets/all_bspline_paths_test.txt"

    path_file = _resolve_path(path_file)
    data_dir_override = _resolve_path(data_dir_override)

    kwargs = {
        "path_file": path_file,
        "data_dir_override": data_dir_override,
        "num_surfaces": num_surfaces,
    }
    if dataset_kwargs:
        kwargs.update(dataset_kwargs)

    ds = dataset_bspline(**kwargs)
    total = len(ds)
    if max_samples is not None and max_samples > 0:
        total = min(total, max_samples)

    valid_count = 0
    invalid_examples = []

    for idx in range(total):
        sample = ds[idx]
        valid = bool(sample[-1])
        if valid:
            valid_count += 1
        elif len(invalid_examples) < 5:
            invalid_examples.append(ds.data_names[idx])

    invalid_count = total - valid_count
    ratio = valid_count / total if total else 0.0

    print("\n=== dataset_bspline validity summary ===")
    print(f"Dataset source (file): {path_file or '(none)'}")
    print(f"Dataset source (dir) : {data_dir_override or '(none)'}")
    print(f"Evaluated samples    : {total}")
    print(f"Valid surfaces       : {valid_count}")
    print(f"Invalid surfaces     : {invalid_count}")
    print(f"Validity ratio       : {ratio:.4f}")
    if invalid_examples:
        print("Example invalid entries:")
        for name in invalid_examples:
            print(f"  - {name}")
    else:
        print("No invalid samples encountered in the evaluated range.")

    return {
        "total": total,
        "valid": valid_count,
        "invalid": invalid_count,
        "ratio": ratio,
        "invalid_examples": invalid_examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count how many bspline surfaces are valid according to dataset_bspline.")
    parser.add_argument("--path_file", type=str, default="assets/all_bspline_paths_test.txt", help="Path to text file listing bspline .npy surfaces.")
    parser.add_argument("--data_dir", type=str, default="", help="Optional directory override where .npy files are located.")
    parser.add_argument("--num_surfaces", type=int, default=-1, help="Number of surfaces to load (-1 for all).")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of samples evaluated.")
    parser.add_argument("--max_degree", type=int, default=3)
    parser.add_argument("--max_num_u_knots", type=int, default=64)
    parser.add_argument("--max_num_v_knots", type=int, default=32)
    parser.add_argument("--max_num_u_poles", type=int, default=64)
    parser.add_argument("--max_num_v_poles", type=int, default=32)
    args = parser.parse_args()

    dataset_kwargs = {
        "max_degree": args.max_degree,
        "max_num_u_knots": args.max_num_u_knots,
        "max_num_v_knots": args.max_num_v_knots,
        "max_num_u_poles": args.max_num_u_poles,
        "max_num_v_poles": args.max_num_v_poles,
    }

    test_bspline_valid(
        path_file=args.path_file,
        data_dir_override=args.data_dir,
        num_surfaces=args.num_surfaces,
        dataset_kwargs=dataset_kwargs,
        max_samples=args.max_samples,
    )

