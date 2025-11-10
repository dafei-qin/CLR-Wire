#!/usr/bin/env python3
"""
Find all files with a specified suffix under a directory and write their paths
(relative to the CLR-Wire project root) to an output file.
"""

import sys
import argparse
from pathlib import Path
from typing import List
import os

# Add project root to path (three levels up from this file: tools -> src -> CLR-Wire)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def _resolve_project_root() -> Path:
    """
    Resolve the CLR-Wire repository root directory.
    Tries (in order):
      1) Environment variable CLR_WIRE_ROOT if set and valid
      2) The first ancestor directory named 'CLR-Wire'
      3) Fallback to three-levels-up from this script location
    """
    env_root = os.environ.get("CLR_WIRE_ROOT")
    if env_root:
        env_path = Path(env_root).resolve()
        if env_path.exists() and env_path.is_dir():
            return env_path

    for ancestor in Path(__file__).resolve().parents:
        if ancestor.name == "CLR-Wire":
            return ancestor

    return project_root


def _find_files_with_suffix(search_dir: Path, suffix: str) -> List[Path]:
    """
    Recursively find all files under search_dir with the given suffix
    (case-insensitive). The suffix can be provided with or without a leading dot.
    """
    suffix_norm = "." + suffix.lower().lstrip(".")
    matched_files: List[Path] = []
    for path in search_dir.rglob("*"):
        if path.is_file() and path.name.lower().endswith(suffix_norm):
            matched_files.append(path.resolve())
    return matched_files


def _to_repo_relative(paths: List[Path], repo_root: Path) -> List[str]:
    """
    Convert absolute paths to paths relative to repo_root. Uses os.path.relpath
    to gracefully handle paths outside repo_root (which may include '..').
    Produces POSIX-style separators for consistency.
    """
    rel_paths: List[str] = []
    repo_root_str = str(repo_root)
    for p in paths:
        rel = os.path.relpath(str(p), start=repo_root_str)
        rel_paths.append(Path(rel).as_posix())
    return rel_paths


def find_all_files_with_suffix(search_dir: str, output_file: str, suffix: str) -> int:
    """
    Find all files with the given suffix under search_dir and write their CLR-Wire-root-relative
    paths to output_file (one per line).
    
    Returns:
        int: Number of files found.
    """
    repo_root = _resolve_project_root()
    start_dir = Path(search_dir).resolve()
    if not start_dir.exists() or not start_dir.is_dir():
        raise FileNotFoundError(f"Search directory does not exist or is not a directory: {search_dir}")

    matched_paths = _find_files_with_suffix(start_dir, suffix)
    rel_paths = _to_repo_relative(matched_paths, repo_root)

    # Sort for determinism
    rel_paths.sort()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rel in rel_paths:
            f.write(rel + "\n")

    return len(rel_paths)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find all files with a given suffix under a directory and write their paths relative to the CLR-Wire root."
    )
    parser.add_argument(
        "dir",
        type=str,
        help="Directory to search recursively"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file to write the CLR-Wire-root-relative paths"
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        default="json",
        help="File suffix/extension to match (e.g., 'json', '.npz'). Default: json"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    count = find_all_files_with_suffix(args.dir, args.output, args.suffix)
    suffix_display = "." + args.suffix.lstrip(".")
    print(f"Found {count} {suffix_display} files. Wrote list to: {args.output}")


if __name__ == "__main__":
    main()


