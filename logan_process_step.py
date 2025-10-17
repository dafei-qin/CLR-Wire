import argparse
import os
import sys
import tempfile
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from logan_process_brep_data import BRepDataProcessor


def discover_step_files(root_dir: str):
    for current_dir, _subdirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith(".step"):
                yield os.path.join(current_dir, file_name)


def compute_output_path(input_root: str, output_root: str, step_file: str) -> str:
    rel_dir = os.path.relpath(os.path.dirname(step_file), start=input_root)
    target_dir = os.path.normpath(os.path.join(output_root, rel_dir))
    base_name = os.path.splitext(os.path.basename(step_file))[0]
    # step_output_dir = os.path.join(target_dir, base_name)
    step_output_dir = os.path.join(target_dir)
    os.makedirs(step_output_dir, exist_ok=True)
    # Seed file name for API which appends suffixes, ensuring files live inside the per-step folder
    return os.path.join(step_output_dir, "index.json")


def process_one(step_file: str, input_root: str, output_root: str) -> int:
    output_path = compute_output_path(input_root, output_root, step_file)
    if os.path.exists(output_path):
        return 0
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_step_path = os.path.join(tmp_dir, os.path.basename(step_file))
        shutil.copy2(step_file, tmp_step_path)
        # API expects a directory containing exactly one STEP file
        return BRepDataProcessor().tokenize_and_save_cad_data([tmp_dir, output_path])


def main():
    parser = argparse.ArgumentParser(description="Batch process STEP files to JSON using BRepDataProcessor")
    parser.add_argument("--input", required=True, help="Root directory to search for .step files")
    parser.add_argument("--output", required=True, help="Root directory to write mirrored JSON outputs")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1), help="Number of parallel workers")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)

    # Discover all candidates first
    step_files = list(discover_step_files(input_root))
    if not step_files:
        print("No .step files found.")
        return 0

    worker_fn = partial(process_one, input_root=input_root, output_root=output_root)

    total = len(step_files)
    num_converted = 0

    if args.workers <= 1:
        for sf in tqdm(step_files, desc="Processing", unit="file"):
            num_converted += 1 if worker_fn(sf) else 0
    else:
        with Pool(processes=args.workers) as pool:
            for result in tqdm(pool.imap_unordered(worker_fn, step_files), total=total, desc="Processing", unit="file"):
                num_converted += 1 if result else 0

    print(f"Done. Converted {num_converted}/{total} files ({100.0 * num_converted / total:.2f}%).")
    return 0


if __name__ == "__main__":
    sys.exit(main())