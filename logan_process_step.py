import argparse
import os
import sys
import tempfile
import shutil
import multiprocessing as mp
from multiprocessing import cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def output_dir_exists(input_root: str, output_root: str, step_file: str) -> bool:
    # The target directory is the parent of the output seed file
    candidate = compute_output_path(input_root, output_root, step_file)
    target_dir = os.path.dirname(candidate)
    return os.path.isdir(target_dir)


def _child_process(step_file: str, input_root: str, output_root: str, result_queue: mp.Queue):
    try:
        output_path = compute_output_path(input_root, output_root, step_file)
        if output_dir_exists(input_root, output_root, step_file):
            result_queue.put({"ok": 0, "step": step_file, "error": None})
            return
        if os.path.exists(output_path):
            result_queue.put({"ok": 0, "step": step_file, "error": None})
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_step_path = os.path.join(tmp_dir, os.path.basename(step_file))
            shutil.copy2(step_file, tmp_step_path)
            status = BRepDataProcessor().tokenize_and_save_cad_data([tmp_dir, output_path])
            result_queue.put({"ok": 1 if status else 0, "step": step_file, "error": None})
    except Exception as e:
        result_queue.put({"ok": 0, "step": step_file, "error": str(e)})


def process_one(step_file: str, input_root: str, output_root: str, timeout_s: int = 120):
    result_queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_child_process, args=(step_file, input_root, output_root, result_queue))
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"ok": 0, "step": step_file, "error": "timeout"}
    try:
        result = result_queue.get_nowait()
        return result
    except Exception:
        return {"ok": 0, "step": step_file, "error": "unknown_error"}


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
    skipped = []

    # Use threads to orchestrate per-file subprocess with timeout
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_step = {executor.submit(worker_fn, sf): sf for sf in step_files}
        for future in tqdm(as_completed(future_to_step), total=total, desc="Processing", unit="file"):
            try:
                result = future.result()
            except Exception as e:
                sf = future_to_step[future]
                skipped.append((sf, str(e)))
                continue
            if result["ok"]:
                num_converted += 1
            elif result.get("error"):
                skipped.append((result["step"], result["error"]))

    print(f"Done. Converted {num_converted}/{total} files ({100.0 * num_converted / total:.2f}%).")
    if skipped:
        print(f"Skipped due to errors: {len(skipped)}")
        for path, err in skipped:
            print(f" - {path}: {err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())