# Surface Canonical Space Transformation Test

## Overview

`test_surface_canonical.py` is an interactive testing tool that verifies the correctness of the surface-to-canonical-space transformation pipeline. It visualizes surfaces at different stages of transformation and calculates accuracy metrics.

## Features

### Visualization Stages
The test displays up to 4 versions of each surface side-by-side:

1. **Raw** (Optional) - Original surface from JSON file
2. **GT (Green)** - Ground truth after dataset normalization (`_parse_surface` → `_recover_surface`)
3. **Canonical (Blue)** - Surface transformed to canonical space
4. **Recovered (Red)** - Surface transformed back from canonical space

### Reference Visualization
To help understand the canonical coordinate system, the test also displays:

- **Coordinate Axes**: RGB-colored axes showing X (red), Y (green), Z (blue) directions
- **Unit Cube**: A 1×1×1 wireframe cube centered at the origin
  - Helps visualize the scale normalization
  - The canonical surface should fit within or around this cube (depending on type)

### Metrics Calculated

#### Transformation Accuracy
- **Location difference**: Euclidean distance between GT and recovered position
- **Direction difference**: Difference in direction vectors (D, X, Y)
- **UV difference**: Difference in UV parameter bounds
- **Scalar difference**: Difference in scalar parameters (radius, angles, etc.)
- **Overall max difference**: Maximum of all differences
- **Pass/Fail**: Based on threshold of 1e-6

#### Canonical Properties Verification
- Position at origin: `||P|| < 1e-5`
- Direction points to Z: `||D - (0,0,1)|| < 1e-5`
- X-direction points to X: `||X - (1,0,0)|| < 1e-5`
- Scale normalization:
  - Cylinders/Spheres: `radius ≈ 1.0`
  - Cones: `radius at v=0 ≈ 1.0`
  - Torus: `major_radius ≈ 1.0`
  - Planes: `max(UV dimensions) ≈ 1.0`

## Usage

### Basic Usage

```bash
python src/tests/test_surface_canonical.py <dataset_path>
```

### Example

```bash
# Windows
python src/tests/test_surface_canonical.py c:/drivers/CAD/data/examples

# Linux
python src/tests/test_surface_canonical.py /home/user/CAD/data/examples
```

## Interactive Controls

### Navigation
- **File Index Slider**: Navigate between different JSON files in the dataset
- **Surface Index Slider**: Navigate between surfaces within the current file
- **Next/Previous Valid Surface**: Jump to next/previous valid (non-skipped) surface

### Visibility Toggle

**Surface Visualization:**
- **Show Raw**: Toggle raw surface visibility
- **Show GT (Green)**: Toggle ground truth surface
- **Show Canonical (Blue)**: Toggle canonical surface
- **Show Recovered (Red)**: Toggle recovered surface

**Reference Objects:**
- **Show Axes (RGB=XYZ)**: Toggle coordinate axes display
  - Red axis = X direction
  - Green axis = Y direction
  - Blue axis = Z direction
- **Show Unit Cube**: Toggle unit cube display (1x1x1 centered at origin)

### Actions
- **Refresh**: Reload the current surface visualization
- **Test All Surfaces in File**: Run batch test on all surfaces in current file

## Interpretation of Results

### Visual Interpretation
When viewing the canonical surface (blue):
- The surface should be centered near the origin
- The surface's main axis should align with the Z-axis (blue)
- The surface should be scaled to approximately unit size
- Compare with the unit cube to verify scale normalization

### PASS Status (✓)
- All transformations accurate within threshold (1e-6)
- Surface can be reliably transformed to/from canonical space
- Geometric properties preserved

### FAIL Status (✗)
- At least one metric exceeds threshold
- Check individual metrics to identify issue:
  - High `location_diff`: Translation error
  - High `direction_diff`: Rotation error
  - High `scalar_diff`: Scaling error
  - High `uv_diff`: Parametrization issue

### Skipped Surfaces
- **B-spline surfaces**: Not yet supported
- **Invalid surfaces**: Filtered out by dataset (e.g., zero radius, invalid parameters)

## Example Output

```
==============================================================
Surface 2 - Type: cylinder
==============================================================

Transformation Metrics:
  Location diff: 1.23e-15 (max: 8.88e-16)
  Direction diff: 2.45e-15 (max: 1.11e-15)
  UV diff: 0.00e+00 (max: 0.00e+00)
  Scalar diff: 0.00e+00 (max: 0.00e+00)
  Overall max diff: 1.11e-15
  PASS: True

Canonical Properties:
  location_at_origin: ✓
  direction_is_z: ✓
  x_direction_is_x: ✓
  radius_is_one: ✓
  all_pass: ✓

Transformation Parameters:
  Shift: [0.3554833  -0.92890103 -0.07622121]
  Scale: 0.07109665986505137
  Rotation:
  [[-0.  0.  1.]
   [ 0. -1.  0.]
   [ 1.  0.  0.]]
```

## Batch Testing

Use the "Test All Surfaces in File" button to run statistics on all surfaces:

```
==============================================================
Testing all surfaces in file 0
==============================================================

Results:
  Total surfaces: 38
  Valid surfaces: 32
  Skipped surfaces: 6
  Passed: 32
  Failed: 0

Accuracy Statistics:
  Mean max diff: 3.45e-15
  Median max diff: 2.22e-15
  Max max diff: 8.88e-15
  Min max diff: 0.00e+00
==============================================================
```

## Troubleshooting

### No surfaces visible
- Check that the dataset path is correct
- Ensure JSON files contain valid surface definitions
- Try toggling visibility checkboxes

### All surfaces skipped
- Dataset may contain only B-spline surfaces
- Surfaces may have invalid parameters (check console output)

### High error metrics
- Verify surface type is supported
- Check that surface parameters are within valid ranges
- Review console output for detailed error messages

## Integration with Dataset Pipeline

This test validates the canonical transformation in the context of the full dataset pipeline:

```
JSON File → _parse_surface → params → _recover_surface → GT Surface
                                                             ↓
                                                      to_canonical
                                                             ↓
                                                    Canonical Surface
                                                             ↓
                                                      from_canonical
                                                             ↓
                                                    Recovered Surface
                                                             ↓
                                                    Compare with GT
```

## Dependencies

- `torch`: Tensor operations
- `numpy`: Numerical computations  
- `polyscope`: 3D visualization
- `src.dataset.dataset_v1`: Dataset loading and surface processing
- `src.tools.surface_to_canonical_space`: Canonical transformation functions
- `utils.surface`: Surface visualization utilities

## See Also

- `src/tools/surface_to_canonical_space.py` - Transformation implementation
- `src/dataset/dataset_v1.py` - Dataset and surface processing
- `src/tests/test_dataset_v1.py` - Dataset validation test

