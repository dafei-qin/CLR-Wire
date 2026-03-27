"""Generate 256-codebook cache for first 100K training samples."""
import sys, pickle
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.import_tools import load_dataset_from_config
from omegaconf import OmegaConf

N = 100000
SAVE_FILE = "/home/node/data/ssd/CAD/cache/abc1m_256_train_100k.pkl"

config = OmegaConf.load("src/configs/gpt/cache_sht_256.yaml")
dataset = load_dataset_from_config(config, section='data_train')

print(f"Dataset size: {len(dataset)}, processing first {min(N, len(dataset))}")

list_npz_path = []
list_tokens = []
list_poles = []
n_valid = 0
n_invalid = 0

for idx in tqdm(range(min(N, len(dataset)))):
    points, normals, all_tokens_padded, all_bspline_poles_padded, all_bspline_valid_mask, solid_valid = dataset[idx]
    if not solid_valid:
        n_invalid += 1
        continue
    all_tokens = all_tokens_padded[~(all_tokens_padded == dataset.pad_id)]
    all_bspline_poles = all_bspline_poles_padded[all_bspline_valid_mask]
    npz_path = dataset.dataset_compound.dataset_compound.json_names[idx % len(dataset.dataset_compound.dataset_compound)].replace('.json', '.npz')

    list_npz_path.append(npz_path)
    list_tokens.append(all_tokens)
    list_poles.append(all_bspline_poles)
    n_valid += 1

print(f"\nDone: {n_valid} valid, {n_invalid} invalid ({n_valid/(n_valid+n_invalid)*100:.1f}% valid)")

with open(SAVE_FILE, 'wb') as f:
    pickle.dump({'npz_path': list_npz_path, 'tokens': list_tokens, 'poles': list_poles}, f)

# Save stats
with open(SAVE_FILE.replace('.pkl', '_stats.txt'), 'w') as f:
    f.write(f"Input: {config.data_train.params.json_dir}\n")
    f.write(f"Codebook: 256, threshold: 0.015\n")
    f.write(f"Processed: {min(N, len(dataset))}\n")
    f.write(f"Valid: {n_valid} ({n_valid/(n_valid+n_invalid)*100:.1f}%)\n")
    f.write(f"Invalid: {n_invalid}\n")
    f.write(f"Output: {SAVE_FILE}\n")

print(f"Saved to {SAVE_FILE}")
