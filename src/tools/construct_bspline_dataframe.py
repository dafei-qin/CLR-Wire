import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import numpy as np
import pandas as pd


def _parse_single_bspline_npy(path: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single bspline .npy file according to the convention used in
    `dataset_bspline.load_data` and return a metadata dict.

    Structure (see `dataset_bspline.load_data`):
        data_vec[0:8]  -> (u_degree, v_degree,
                           num_poles_u, num_poles_v,
                           num_knots_u, num_knots_v,
                           is_u_periodic, is_v_periodic)
        the rest are knots / mults / poles, which we don't need here.
    """
    path = path.strip()
    if not path:
        return None

    p = Path(path)
    if not p.is_file():
        # Silently skip non-existing files; user can handle missing rows if needed.
        return None

    try:
        data_vec = np.load(str(p), allow_pickle=False)
    except Exception:
        # Skip files that cannot be read.
        return None

    if data_vec.shape[0] < 8:
        # Malformed entry; skip.
        return None

    (
        u_degree,
        v_degree,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
    ) = map(int, data_vec[:8])

    # Derived quantities that are often useful for快速过滤.
    num_knots = int(num_knots_u * num_knots_v)
    num_poles = int(num_poles_u * num_poles_v)
    is_periodic = bool(is_u_periodic or is_v_periodic)

    return {
        "path": str(p),
        "u_degree": u_degree,
        "v_degree": v_degree,
        "num_poles_u": num_poles_u,
        "num_poles_v": num_poles_v,
        "num_poles": num_poles,
        "num_knots_u": num_knots_u,
        "num_knots_v": num_knots_v,
        "num_knots": num_knots,
        "is_u_periodic": bool(is_u_periodic),
        "is_v_periodic": bool(is_v_periodic),
        "is_periodic": is_periodic,
    }


def construct_bspline_dataframe_from_paths(paths: Iterable[str]) -> pd.DataFrame:
    """
    给定一组 npy 路径，构建一个 pandas DataFrame：

    每一行为：
        - path
        - num_knots_u, num_knots_v, num_knots
        - num_poles_u, num_poles_v, num_poles
        - is_u_periodic, is_v_periodic, is_periodic
        - u_degree, v_degree
    方便后续快速 filter 找到所需的 surface 类型。
    """
    records: List[Dict[str, Any]] = []
    for p in paths:
        meta = _parse_single_bspline_npy(p)
        if meta is not None:
            records.append(meta)

    if not records:
        # 返回空 DataFrame，但列名固定，方便后续使用。
        return pd.DataFrame(
            columns=[
                "path",
                "u_degree",
                "v_degree",
                "num_poles_u",
                "num_poles_v",
                "num_poles",
                "num_knots_u",
                "num_knots_v",
                "num_knots",
                "is_u_periodic",
                "is_v_periodic",
                "is_periodic",
            ]
        )

    return pd.DataFrame.from_records(records)


def construct_bspline_dataframe_from_list_file(path_file: str) -> pd.DataFrame:
    """
    根据 dataset_bspline 的用法：
        path_file: 文本文件，每一行为一个 .npy 路径
    生成对应的 DataFrame。
    """
    with open(path_file, "r") as f:
        paths = [line.strip() for line in f if line.strip()]
    return construct_bspline_dataframe_from_paths(paths)


def construct_bspline_dataframe_from_dir(
    data_dir: str,
    pattern: str = "*.npy",
    recursive: bool = True,
) -> pd.DataFrame:
    """
    从目录扫描所有符合 pattern 的 .npy 文件，生成 DataFrame。
    """
    root = Path(data_dir)
    if recursive:
        paths = sorted(str(p) for p in root.rglob(pattern))
    else:
        paths = sorted(str(p) for p in root.glob(pattern))
    return construct_bspline_dataframe_from_paths(paths)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a pandas DataFrame summarizing bspline .npy metadata "
        "(num_knots, num_poles, periodic flags) for快速检索与过滤.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path-file",
        type=str,
        help="文本文件，每行一个 bspline .npy 路径（与 dataset_bspline 一致）。",
    )
    group.add_argument(
        "--data-dir",
        type=str,
        help="包含 bspline .npy 的目录，将递归搜索所有匹配文件。",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.npy",
        help="当使用 --data-dir 时的文件匹配模式，默认 '*.npy'。",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="当使用 --data-dir 时，不递归子目录。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="若提供，则将 DataFrame 保存为此路径（支持 .csv 或 .pkl）。",
    )
    return parser


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if args.path_file:
        df = construct_bspline_dataframe_from_list_file(args.path_file)
    else:
        df = construct_bspline_dataframe_from_dir(
            args.data_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )

    if args.output:
        out_path = Path(args.output)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(out_path, index=False)
        elif suffix in (".pkl", ".pickle"):
            df.to_pickle(out_path)
        else:
            # 默认保存为 csv
            df.to_csv(out_path, index=False)
        print(f"Saved dataframe with {len(df)} rows to: {out_path}")
    else:
        # 不保存文件，简单打印一个 summary。
        print(df.head())
        print(f"\nTotal surfaces: {len(df)}")


if __name__ == "__main__":
    main()


