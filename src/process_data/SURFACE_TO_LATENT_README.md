# Surface to Latent Conversion with Bounding Box Sorting

## Overview

The `surface_to_latent.py` script converts surface data from `dataset_v1` to latent representations using VAE, with the following additional features:

1. **Point Cloud Sampling**: Samples each surface with an 8x8 grid
2. **Bounding Box Computation**: Calculates bounding boxes for each surface **in original space**
3. **Automatic Sorting**: Sorts all surfaces by bounding box minimum coordinates
4. **Quality Metrics**: Computes reconstruction accuracy metrics

### Important: Canonical Space vs Original Space

When using `--canonical` flag:
- **Latent parameters, rotations, scales, and shifts** are computed from **canonical space** parameters
- **Bounding boxes** are computed from **original space** parameters

This ensures that:
- The VAE encoding benefits from the normalized canonical space
- The bounding boxes represent actual spatial positions in the original coordinate system
- Sorting by bbox provides meaningful spatial ordering in the original scene

## Usage

```bash
python src/process_data/surface_to_latent.py \
    <input_dir> \
    <checkpoint_path> \
    <output_dir> \
    [--device cuda] \
    [--canonical]
```

### Arguments

- `input_dir`: Directory containing JSON files from dataset_v1
- `checkpoint_path`: Path to VAE model checkpoint
- `output_dir`: Output directory for NPZ files (maintains subdirectory structure)
- `--device`: Device to use (cpu or cuda), default: cpu
- `--canonical`: Use canonical dataset transformation

### Example

```bash
python src/process_data/surface_to_latent.py \
    ../data/logan_jsons/abc \
    checkpoints/vae_v1_best.pth \
    ../data/latent_abc \
    --device cuda \
    --canonical
```

## Output Format

Each NPZ file contains the following arrays (all sorted by bounding box):

- `latent_params`: (N, 128) - Latent representations
- `rotations`: (N, 6) - First 6 elements of rotation matrices
- `scales`: (N, 1) - Scale values
- `shifts`: (N, 3) - Translation vectors
- `classes`: (N, 1) - Surface type indices (0-4)
- `bbox_mins`: (N, 3) - Bounding box minimum coordinates [x_min, y_min, z_min]
- `bbox_maxs`: (N, 3) - Bounding box maximum coordinates [x_max, y_max, z_max]

Where N is the number of valid surfaces in each sample.

## Sorting Order

All data is sorted by bounding box minimum coordinates with the following priority:
1. **X coordinate** (primary)
2. **Y coordinate** (secondary, when X is equal)
3. **Z coordinate** (tertiary, when X and Y are equal)

This ensures a consistent spatial ordering of surfaces from left to right, front to back, bottom to top.

## Loading the Data

Use the updated `LatentDataset` or `LatentDatasetFlat` from `src/dataset/dataset_latent.py`:

```python
from src.dataset.dataset_latent import LatentDataset

dataset = LatentDataset(
    npz_dir='../data/latent_abc',
    max_num_surfaces=500,
    latent_dim=128
)

# Load a sample
latent_params, rotations, scales, shifts, classes, bbox_mins, bbox_maxs, mask = dataset[0]

# All data is already sorted by bounding box
print(f"First surface bbox_min: {bbox_mins[0]}")
print(f"Last valid surface bbox_min: {bbox_mins[mask.bool()][-1]}")
```

## Verifying Bbox Computation Space

To verify that bounding boxes are computed in the original space (when using `--canonical`):

```python
import numpy as np
from src.dataset.dataset_v1 import dataset_compound
from src.tools.sample_simple_surface import sample_surface_uniform

# Load both datasets
dataset_canonical = dataset_compound('path/to/json', canonical=True)
dataset_original = dataset_compound('path/to/json', canonical=False)

# Get a sample
idx = 0
params_canon, types_canon, mask_canon, _, _, _ = dataset_canonical[idx]
params_orig, types_orig, mask_orig, _, _, _ = dataset_original[idx]

# Sample the first valid surface from both spaces
valid_idx = 0
surf_type = types_canon[mask_canon.bool()][valid_idx].item()

# Sample using canonical params
points_canon = sample_surface_uniform(
    params_canon[mask_canon.bool()][valid_idx].numpy(),
    surf_type, num_u=8, num_v=8
)

# Sample using original params  
points_orig = sample_surface_uniform(
    params_orig[mask_orig.bool()][valid_idx].numpy(),
    surf_type, num_u=8, num_v=8
)

print("Canonical space bbox:", points_canon.min(axis=0), points_canon.max(axis=0))
print("Original space bbox:", points_orig.min(axis=0), points_orig.max(axis=0))

# The bboxes should be different if canonical transformation is non-trivial
# The saved bbox in NPZ should match the original space bbox
```

## Quality Metrics

The script outputs reconstruction quality metrics:

- **Classification Accuracy**: Percentage of correctly classified surface types
- **Parameters MSE**: Mean squared error of reconstructed parameters

These metrics help validate the quality of the VAE encoding.

## Implementation Details

### Point Cloud Sampling

Each surface is sampled with an 8x8 grid (64 points total) using the `sample_surface_uniform` function from `src/tools/sample_simple_surface.py`.

### Bounding Box Computation

Bounding boxes are computed from the sampled point clouds:
- `bbox_min = [min(x), min(y), min(z)]`
- `bbox_max = [max(x), max(y), max(z)]`

**Important**: When using `--canonical` flag, bounding boxes are computed from **original space** parameters:
1. The script loads two datasets: one with `canonical=True` and one with `canonical=False`
2. Canonical space parameters are used for VAE encoding
3. Original space parameters are used for surface sampling and bbox computation
4. This ensures bboxes represent actual spatial positions in the original scene

### Sorting Algorithm

Uses `numpy.lexsort` with reversed key order to achieve x > y > z priority:

```python
sort_indices = np.lexsort((bbox_mins[:, 2], bbox_mins[:, 1], bbox_mins[:, 0]))
```

All data arrays (latent_params, rotations, scales, shifts, classes, bbox_mins, bbox_maxs) are sorted using the same indices to maintain consistency.

## Notes

- **Canonical vs Original Space**: When `--canonical` is used, the script loads data twice:
  - Once with `canonical=True` for VAE encoding (provides normalized, centered parameters)
  - Once with `canonical=False` for bbox computation (provides actual spatial positions)
  - This dual-loading ensures optimal VAE performance while maintaining meaningful spatial ordering
  
- If surface sampling fails, the script logs a warning and assigns zero bounding boxes

- The sorting is stable, preserving the original order for surfaces with identical bounding boxes

- All data arrays maintain consistent ordering after sorting

- The script is backward compatible: old NPZ files without bbox data can still be loaded (bbox will be zero-filled)

- Memory consideration: Using `--canonical` requires loading each sample twice, which approximately doubles memory usage during processing

