"""Pack JSON and PLY files with chamfer distance below threshold into a zip archive."""
import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def parse_json_filename(json_path: str):
    """Parse JSON filename to extract components."""
    json_path_obj = Path(json_path)
    filename = json_path_obj.name
    parts = filename.split('_')
    
    # Find batch and iter positions
    batch_idx_pos = -1
    iter_idx_pos = -1
    for i, part in enumerate(parts):
        if part == 'batch':
            batch_idx_pos = i
        elif part == 'iter':
            iter_idx_pos = i
    
    if batch_idx_pos == -1 or iter_idx_pos == -1:
        return None, None, None
    
    idx = '_'.join(parts[:batch_idx_pos])
    batch_num = parts[batch_idx_pos + 1]
    checkpoint = parts[iter_idx_pos + 1].replace('.json', '')
    
    return idx, batch_num, checkpoint

def find_corresponding_files(json_path: str, highres_dir: str = None):
    """Find corresponding PLY, JPG, RAW JSON, and highres PLY files."""
    json_path_obj = Path(json_path)
    idx, batch_num, checkpoint = parse_json_filename(json_path)
    
    if idx is None:
        return None, None, None, None
    
    base_name = f"{idx}_batch_{batch_num}"
    parent_dir = json_path_obj.parent
    
    # Find PLY file
    ply_path = parent_dir / f"{base_name}.ply"
    ply_file = str(ply_path) if ply_path.exists() else None
    
    # Find JPG file for this checkpoint
    jpg_path = parent_dir / f"{base_name}_grid_iter_{checkpoint}.jpg"
    jpg_file = str(jpg_path) if jpg_path.exists() else None
    
    # Find raw JSON file (if exists)
    raw_json_path = parent_dir / f"{base_name}_raw_iter_{checkpoint}.json"
    raw_json_file = str(raw_json_path) if raw_json_path.exists() else None
    
    # Find highres PLY file (if highres_dir is provided)
    highres_ply_file = None
    if highres_dir:
        highres_ply_path = Path(highres_dir) / f"{base_name}_highres.ply"
        highres_ply_file = str(highres_ply_path) if highres_ply_path.exists() else None
    
    return ply_file, jpg_file, raw_json_file, highres_ply_file


def pack_low_cd_results(
    results_json: str,
    threshold: float,
    output_zip: str,
    highres_dir: str = None
):
    """
    Pack files with chamfer distance below threshold.
    
    Args:
        results_json: Path to results JSON file
        threshold: Chamfer distance threshold
        output_zip: Output zip file path
        highres_dir: Path to highres directory (optional)
    """
    # Load results
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    results = data.get('results_sorted_by_cd', [])
    
    if not results:
        print("No results found in JSON file!")
        return
    
    # Filter results below threshold
    filtered_results = [r for r in results if r['chamfer_dist'] < threshold]
    
    print(f"Total unique indices: {len(results)}")
    print(f"Indices below threshold ({threshold}): {len(filtered_results)}")
    if highres_dir:
        print(f"Highres directory: {highres_dir}")
    
    if not filtered_results:
        print("No results below threshold!")
        return
    
    # Collect files to pack
    files_to_pack = []
    missing_files = []
    
    print(f"\nCollecting files for {len(filtered_results)} indices...")
    for result in tqdm(filtered_results, desc="Scanning files"):
        idx = result['idx']
        gt_path = result['gt_path']
        pred_path = result['pred_path']
        cd = result['chamfer_dist']
        
        # Find corresponding files
        ply_path, jpg_path, raw_json_path, highres_ply_path = find_corresponding_files(gt_path, highres_dir)
        
        # Collect all files
        files_for_this_idx = {
            'gt_json': gt_path,
            'pred_json': pred_path,
            'ply': ply_path,
            'jpg': jpg_path,
            'raw_json': raw_json_path,
            'highres_ply': highres_ply_path
        }
        
        # Add files that exist
        for file_type, file_path in files_for_this_idx.items():
            if file_path and os.path.exists(file_path):
                # Add CD value to filename
                original_name = os.path.basename(file_path)
                name_parts = os.path.splitext(original_name)
                new_name = f"{name_parts[0]}_cd{cd:.4f}{name_parts[1]}"
                
                files_to_pack.append({
                    'path': file_path,
                    'arcname': new_name,
                    'idx': idx,
                    'cd': cd
                })
            else:
                # Only report missing critical files (gt_json, pred_json)
                if file_type in ['gt_json', 'pred_json']:
                    missing_files.append(f"{file_type}: {file_path}")
                    print(f"Warning: Missing critical file for idx {idx}: {file_type}")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found:")
        for mf in missing_files[:10]:
            print(f"  - {mf}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Create zip file
    print(f"\nCreating zip archive: {output_zip}")
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add metadata file
        metadata = {
            'threshold': threshold,
            'num_indices': len(filtered_results),
            'checkpoint': data.get('checkpoint', 'unknown'),
            'statistics': data.get('statistics', {}),
            'filtered_results': filtered_results
        }
        
        zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        # Add files
        added_files = set()
        for file_info in tqdm(files_to_pack, desc="Adding to zip"):
            file_path = file_info['path']
            arcname = file_info['arcname']
            
            if file_path not in added_files:
                zipf.write(file_path, arcname)
                added_files.add(file_path)
    
    # Summary
    unique_indices = set(f['idx'] for f in files_to_pack)
    
    # Count file types
    file_types = {}
    for f in files_to_pack:
        ext = Path(f['path']).suffix
        file_types[ext] = file_types.get(ext, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"Packed files for {len(unique_indices)} unique indices")
    print(f"Total files packed: {len(added_files) + 1} (including metadata.json)")
    print(f"\nFile type breakdown:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")
    print(f"\nOutput: {output_zip}")
    print(f"{'='*60}")
    
    # Show some statistics
    cd_values = [r['chamfer_dist'] for r in filtered_results]
    print(f"\nChamfer Distance statistics for packed data:")
    print(f"  Min: {min(cd_values):.6f}")
    print(f"  Max: {max(cd_values):.6f}")
    print(f"  Mean: {sum(cd_values)/len(cd_values):.6f}")


def main():
    parser = argparse.ArgumentParser(description='Pack low CD results into zip file.')
    parser.add_argument(
        '--results_json',
        type=str,
        default='/deemos-research-area-d/meshgen/cad/CLR-Wire/src/eval/results_230000.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=1e-3,
        help='Chamfer distance threshold (default: 1e-3)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='low_cd_results.zip',
        help='Output zip file path'
    )
    parser.add_argument(
        '--highres_dir',
        type=str,
        default='/deemos-research-area-d/meshgen/cad/checkpoints/GPT_INIT_142M/train_0110_michel_4096_full/test_00_highres',
        help='Path to highres directory'
    )
    
    args = parser.parse_args()
    
    pack_low_cd_results(args.results_json, args.threshold, args.output, args.highres_dir)


if __name__ == '__main__':
    main()

