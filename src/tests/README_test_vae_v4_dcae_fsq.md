# Test VAE v4 DC-AE FSQ

This test script visualizes the VAE v4 DC-AE FSQ model performance through three different pipelines.

## Overview

The script `test_vae_v4_dcae_fsq.py` processes JSON files containing CAD surfaces and visualizes them through three pipelines:

1. **Pipeline 1: Raw JSON Visualization**
   - Directly visualizes surfaces from JSON files
   - Shows the ground truth data

2. **Pipeline 2: Dataset Round-trip**
   - JSON → dataset_v2.dataset_compound → to_canonical → from_canonical → visualize
   - Tests the dataset parsing and canonical transformation round-trip
   - Verifies that surfaces can be correctly parsed and transformed

3. **Pipeline 3: VAE Reconstruction**
   - JSON → dataset_v2.dataset_compound → to_canonical → VAE encode/decode → from_canonical → visualize
   - Tests the complete VAE reconstruction pipeline
   - Shows how well the model reconstructs surfaces

## Features

- **Automatic filtering**: Only processes JSON files containing bspline surfaces
- **Interactive visualization**: Use Polyscope UI to navigate through samples
- **Toggle visibility**: Show/hide individual pipelines for comparison
- **Loss metrics**: Displays reconstruction MSE loss for each sample
- **FSQ statistics**: Shows codebook usage and unique codes

## Usage

### Basic Usage

```bash
python src/tests/test_vae_v4_dcae_fsq.py \
    --config src/configs/train_vae_v4_bspline_only_dcae_fsq_6tokens_1024.yaml \
    --checkpoint_path checkpoints/your_model.pt
```

### With Starting Index

```bash
python src/tests/test_vae_v4_dcae_fsq.py \
    --config src/configs/train_vae_v4_bspline_only_dcae_fsq_6tokens_1024.yaml \
    --checkpoint_path checkpoints/your_model.pt \
    --start_idx 10
```

## Arguments

- `--config`: Path to the training configuration YAML file (required)
- `--checkpoint_path`: Path to the model checkpoint file (required)
- `--start_idx`: Starting index for visualization (default: 0)

## Requirements

The script requires the following to be configured in your config file:

- `data_val`: Validation dataset configuration
- `model`: Model configuration (VAE v4 DC-AE FSQ)

## Controls

### UI Controls

- **Test Index Slider**: Navigate through different samples
- **Go To Index**: Jump to a specific sample index
- **Show Pipeline 1 (Raw JSON)**: Toggle ground truth visualization
- **Show Pipeline 2 (Dataset Round-trip)**: Toggle dataset round-trip visualization
- **Show Pipeline 3 (VAE Reconstruction)**: Toggle VAE reconstruction visualization

## Output Information

For each sample, the script prints:

- Number of bspline surfaces found
- Dataset sample shapes (patches, shift, rotation, scale)
- VAE input/output shapes
- FSQ codebook statistics (indices, unique codes)
- Reconstruction MSE loss

## Expected Behavior

**Good reconstruction** should show:
- Pipeline 3 (VAE reconstruction) closely matching Pipeline 1 (ground truth)
- Low reconstruction MSE loss (< 0.01 for well-trained models)
- High codebook usage (indicating diverse latent representations)

**Dataset round-trip** (Pipeline 2) should:
- Exactly or very closely match Pipeline 1
- Verify that the dataset parsing and canonical transformations are working correctly

## Troubleshooting

### "No bspline surfaces found"
- The JSON file doesn't contain bspline surfaces
- The script will automatically skip these files

### "Sample is invalid"
- The dataset marked this sample as invalid
- Usually due to parsing errors or malformed data

### Visualization errors
- Check that `utils.surface.visualize_json_interset` is working correctly
- Ensure all required dependencies (Polyscope, etc.) are installed

### Shape mismatches
- Verify that the dataset configuration matches the model configuration
- Check patch sizes (should be 4x4x3 for bspline surfaces)

## Model Information

The script displays model information at startup:
- Model type (DCAE_FSQ_VAE)
- Codebook size
- Number of codebooks
- Latent dimensionality

## Example Output

```
Loading dataset...
Dataset loaded with 1000 samples
Collecting JSON files with bspline surfaces...
Found 850 files with bspline surfaces
Loading model...
Loaded EMA model weights.

Model info:
  Type: DCAE_FSQ_VAE
  Codebook size: 1024
  Num codebooks: 6
  Latent dim: 256

======================================================================
Processing file: /path/to/file.json
======================================================================
Found 3 bspline surfaces
Dataset sample shapes:
  patches: torch.Size([3, 4, 4, 20])
  shift: torch.Size([3, 3])
  rotation: torch.Size([3, 3, 3])
  scale: torch.Size([3])

VAE input patches shape: torch.Size([3, 3, 4, 4])
VAE output shape: torch.Size([3, 3, 4, 4])
Latent shape: torch.Size([3, 256])
FSQ indices shape: torch.Size([3, 6])
  Codebook 0: unique codes = 128
  Codebook 1: unique codes = 156
  ...
Reconstruction MSE loss: 0.003245

Pipeline 1 (Raw JSON): Visualized 3 surfaces
Pipeline 2 (Dataset round-trip): Visualized 3 surfaces
Pipeline 3 (VAE reconstruction): Visualized 3 surfaces
```

## Notes

- The script focuses on bspline surfaces only (as specified by VAE v4 trainer)
- Control points are visualized as 4x4 patches
- FSQ (Finite Scalar Quantization) provides discrete latent codes
- EMA (Exponential Moving Average) weights are preferred when available

