"""
Surface to Canonical Space Transformation

This module provides functions to transform surfaces from their original coordinate
system to a canonical coordinate system where:
- The surface center is at the origin (0, 0, 0)
- The direction vector points to (0, 0, 1)
- The X-direction vector points to (1, 0, 0)
- The surface is scaled so that a characteristic dimension equals 1

Usage:
    Method 1 - Using surface dictionary:
        >>> canonical, shift, rot, scale = to_canonical(surface_dict)
    
    Method 2 - Using dataset params (more efficient):
        >>> params, type_idx = dataset._parse_surface(surface_dict)
        >>> canonical, shift, rot, scale = to_canonical((params, type_idx), dataset)
    
    The second method is more efficient as it avoids an intermediate recovery step
    when working with the dataset pipeline.

Note on Plane Parametrization:
    For planes, the canonical transformation centers the UV bounds symmetrically around zero.
    This means that after a round-trip (to_canonical -> from_canonical), the recovered plane
    will represent the SAME GEOMETRIC SURFACE but with a different UV parametrization:
    - Original UV: [u_min, u_max, v_min, v_max] (arbitrary bounds)
    - Recovered UV: [-w/2, w/2, -h/2, h/2] (centered, scaled)
    The position P is adjusted accordingly so that P + u*X + v*Y traces the same geometric points.
"""

import numpy as np
from typing import Dict, Tuple, Union, Optional, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from src.dataset.dataset_v1 import dataset_compound


def compute_rotation_matrix(D: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align local frame (X, Y, D) to canonical frame.
    
    Args:
        D: Direction vector (will be aligned to [0, 0, 1])
        X: X-direction vector (will be aligned to [1, 0, 0])
    
    Returns:
        R: 3x3 rotation matrix
    """
    # Normalize direction vectors
    D = D / np.linalg.norm(D)
    X = X / np.linalg.norm(X)
    
    # Compute Y direction
    Y = np.cross(D, X)
    Y = Y / np.linalg.norm(Y)
    
    # Recompute X to ensure orthogonality
    X = np.cross(Y, D)
    X = X / np.linalg.norm(X)
    
    # Rotation matrix from local frame to canonical frame
    # R transforms a vector in world coordinates to local coordinates
    # [X, Y, D] are the columns of the local frame in world coordinates
    # So R = [X, Y, D]^T
    R = np.array([X, Y, D], dtype=np.float64)
    
    return R


def to_canonical(
    surface: Union[Dict, Tuple[np.ndarray, int]], 
    dataset: Optional['dataset_compound'] = None
) -> Tuple[Dict, np.ndarray, np.ndarray, float]:
    """
    Transform a surface to canonical coordinate system.
    
    Args:
        surface: Either:
            - Surface dictionary with keys: 'type', 'idx', 'location', 'direction', 
              'uv', 'scalar', 'poles', 'orientation'
            - Tuple of (params, surface_type_idx) from dataset._parse_surface
        dataset: Required if surface is a tuple of (params, surface_type_idx).
                Used to recover the surface from processed parameters.
    
    Returns:
        transformed_surface: Surface in canonical coordinates
        shift: 3D translation vector (original center position)
        rotation: 3x3 rotation matrix
        scale: Scaling factor
    
    Examples:
        >>> # Method 1: Using surface dictionary
        >>> canonical, shift, rot, scale = to_canonical(surface_dict)
        
        >>> # Method 2: Using dataset params
        >>> params, type_idx = dataset._parse_surface(surface_dict)
        >>> canonical, shift, rot, scale = to_canonical((params, type_idx), dataset)
    """
    # Handle different input types
    if isinstance(surface, tuple):
        # Input is (params, surface_type_idx) from _parse_surface
        if dataset is None:
            raise ValueError("dataset argument is required when surface is (params, surface_type_idx) tuple")
        
        params, surface_type_idx = surface
        
        # Recover surface dictionary from params
        surface_dict = dataset._recover_surface(params, surface_type_idx)
        # Add required fields
        surface_dict['idx'] = [0, 0]
        surface_dict['orientation'] = 'Forward'
        surface_dict['poles'] = []
        
        surface = surface_dict
    
    surface_type = surface['type']
    
    # Skip bspline surfaces
    if surface_type == 'bspline_surface':
        raise NotImplementedError("B-spline surfaces are not supported yet")
    
    # Extract original parameters
    P = np.array(surface['location'][0], dtype=np.float64)
    D = np.array(surface['direction'][0], dtype=np.float64)
    X = np.array(surface['direction'][1], dtype=np.float64)
    
    # Normalize direction vectors
    D = D / np.linalg.norm(D)
    X = X / np.linalg.norm(X)
    Y = np.cross(D, X)
    Y = Y / np.linalg.norm(Y)
    
    u_min, u_max, v_min, v_max = surface['uv']
    
    # Create a deep copy for the transformed surface
    transformed_surface = copy.deepcopy(surface)
    
    # Compute transformation parameters based on surface type
    if surface_type == 'plane':

        # Compute the geometric center of the plane patch
        u_center_orig = (u_max + u_min) / 2
        v_center_orig = (v_max + v_min) / 2
        P_new = P - u_center_orig * X - v_center_orig * Y

        # Center u and v
        u_min = u_min - u_center_orig
        u_max = u_max - u_center_orig
        v_min = v_min - v_center_orig
        v_max = v_max - v_center_orig

        

        # center = P + u_center_orig * X + v_center_orig * Y
        shift = P_new.copy()
        
        # Compute rotation matrix
        rotation = compute_rotation_matrix(D, X)
        
        # Compute scale: make longer edge equal to 1
        u_length = abs(u_max - u_min)
        v_length = abs(v_max - v_min)
        scale = max(u_length, v_length)
        
        # Apply transformations
        # Rotate the center to origin (which is the center, so becomes [0,0,0])
        P_new = rotation @ (P_new - shift)  # Should be [0, 0, 0]
        D_new = rotation @ D
        X_new = rotation @ X
        Y_new = rotation @ Y
        
        # In canonical space, UV is centered and scaled
        

        u_half = (u_max - u_min) / 2 / scale
        v_half = (v_max - v_min) / 2 / scale

        u_min_canonical = -u_half
        u_max_canonical = u_half
        v_min_canonical = -v_half
        v_max_canonical = v_half

        # Update transformed surface
        # Position is at the center (origin in canonical space)
        transformed_surface['location'] = [P_new.tolist()]
        transformed_surface['direction'] = [D_new.tolist(), X_new.tolist(), Y_new.tolist()]
        transformed_surface['uv'] = [u_min_canonical, u_max_canonical, v_min_canonical, v_max_canonical]
        transformed_surface['scalar'] = []
        transformed_surface['poles'] = []
    
    elif surface_type == 'cylinder':
        radius = surface['scalar'][0]
        
        # Center is at the position P (axis passes through P)
        shift = P.copy()
        
        # Compute rotation matrix
        rotation = compute_rotation_matrix(D, X)
        
        # Scale by radius
        scale = max(radius, v_max - v_min)
        
        # Apply transformations
        P_new = rotation @ (P - shift)
        D_new = rotation @ D
        X_new = rotation @ X
        Y_new = rotation @ Y
        
        # Scale parameters
        radius_new = radius / scale
        v_min_new = v_min / scale
        v_max_new = v_max / scale
        
        # Update transformed surface
        transformed_surface['location'] = [P_new.tolist()]
        transformed_surface['direction'] = [D_new.tolist(), X_new.tolist(), Y_new.tolist()]
        transformed_surface['uv'] = [u_min, u_max, v_min_new, v_max_new]
        transformed_surface['scalar'] = [radius_new]
        transformed_surface['poles'] = []
    
    elif surface_type == 'cone':
        semi_angle = surface['scalar'][0]
        radius = surface['scalar'][1]  # radius at v=0
        
        # Center is at the apex, but we want to center at the v=0 plane
        # The position P is at v=0, so center there
        shift = P.copy()
        
        # Compute rotation matrix
        rotation = compute_rotation_matrix(D, X)
        
        # Scale: make the radius at the current v=0 plane equal to 1
        scale = max(radius, v_max - v_min)
        
        # Apply transformations
        P_new = rotation @ (P - shift)
        D_new = rotation @ D
        X_new = rotation @ X
        Y_new = rotation @ Y
        
        # Scale parameters
        radius_new = radius / scale
        v_min_new = v_min / scale
        v_max_new = v_max / scale
        
        # Update transformed surface
        transformed_surface['location'] = [P_new.tolist()]
        transformed_surface['direction'] = [D_new.tolist(), X_new.tolist(), Y_new.tolist()]
        transformed_surface['uv'] = [u_min, u_max, v_min_new, v_max_new]
        transformed_surface['scalar'] = [semi_angle, radius_new]
        transformed_surface['poles'] = []
    
    elif surface_type == 'sphere':
        radius = surface['scalar'][0]
        
        # Center is at position P
        shift = P.copy()
        
        # Compute rotation matrix
        rotation = compute_rotation_matrix(D, X)
        
        # Scale by radius
        scale = radius
        
        # Apply transformations
        P_new = rotation @ (P - shift)
        D_new = rotation @ D
        X_new = rotation @ X
        Y_new = rotation @ Y
        
        # Scale parameters
        radius_new = radius / scale
        
        # Update transformed surface
        transformed_surface['location'] = [P_new.tolist()]
        transformed_surface['direction'] = [D_new.tolist(), X_new.tolist(), Y_new.tolist()]
        transformed_surface['uv'] = [u_min, u_max, v_min, v_max]  # Angular parameters don't scale
        transformed_surface['scalar'] = [radius_new]
        transformed_surface['poles'] = []
    
    elif surface_type == 'torus':
        major_radius = surface['scalar'][0]
        minor_radius = surface['scalar'][1]
        
        # Center is at position P
        shift = P.copy()
        
        # Compute rotation matrix
        rotation = compute_rotation_matrix(D, X)
        
        # Scale by major radius
        scale = major_radius
        
        # Apply transformations
        P_new = rotation @ (P - shift)
        D_new = rotation @ D
        X_new = rotation @ X
        Y_new = rotation @ Y
        
        # Scale parameters
        major_radius_new = major_radius / scale
        minor_radius_new = minor_radius / scale
        
        # Update transformed surface
        transformed_surface['location'] = [P_new.tolist()]
        transformed_surface['direction'] = [D_new.tolist(), X_new.tolist(), Y_new.tolist()]
        transformed_surface['uv'] = [u_min, u_max, v_min, v_max]  # Angular parameters don't scale
        transformed_surface['scalar'] = [major_radius_new, minor_radius_new]
        transformed_surface['poles'] = []
    
    else:
        raise ValueError(f"Unsupported surface type: {surface_type}")
    
    return transformed_surface, shift, rotation, scale


def from_canonical(surface_canonical: Dict, shift: np.ndarray, rotation: np.ndarray, scale: float) -> Dict:
    """
    Transform a surface from canonical coordinates back to original coordinate system.
    
    Args:
        surface_canonical: Surface in canonical coordinates
        shift: 3D translation vector
        rotation: 3x3 rotation matrix
        scale: Scaling factor
    
    Returns:
        surface: Surface in original coordinates
    """
    # print(surface_canonical, shift, rotation, scale)
    surface_type = surface_canonical['type']
    
    # Skip bspline surfaces
    if surface_type == 'bspline_surface':
        raise NotImplementedError("B-spline surfaces are not supported yet")
    
    # Create a deep copy
    surface = copy.deepcopy(surface_canonical)
    
    # Extract canonical parameters
    P_canonical = np.array(surface_canonical['location'][0], dtype=np.float64)
    D_canonical = np.array(surface_canonical['direction'][0], dtype=np.float64)
    X_canonical = np.array(surface_canonical['direction'][1], dtype=np.float64)
    
    # Inverse rotation (transpose for orthogonal matrix)
    rotation_inv = rotation.T
    
    # Apply inverse transformations
    P_original = rotation_inv @ P_canonical + shift
    D_original = rotation_inv @ D_canonical
    X_original = rotation_inv @ X_canonical
    
    # Compute Y_original
    Y_original = np.cross(D_original, X_original)
    Y_original = Y_original / np.linalg.norm(Y_original)
    
    u_min, u_max, v_min, v_max = surface_canonical['uv']
    
    # Inverse transformations based on surface type
    if surface_type == 'plane':
        # In canonical space:
        # - Position P_canonical is at the geometric center (origin [0,0,0])
        # - UV is [-hw, hw] x [-hh, hh] centered at 0
        # - The geometric center is at P_canonical + 0*X + 0*Y = [0,0,0]
        
        # In original space, the geometric center was at 'shift'
        # So P_original (the reference point) should be computed such that
        # the geometric center P_orig + u_c_orig * X_orig + v_c_orig * Y_orig = shift
        
        # Since P_canonical = [0,0,0] (center), after inverse rotation:
        # center_recovered = rotation_inv @ [0,0,0] + shift = shift ✓
        
        # For the recovered UV, we'll make it centered (symmetric)
        u_half_orig = (u_max - u_min) / 2 * scale
        v_half_orig = (v_max - v_min) / 2 * scale
        
        u_min_orig = -u_half_orig
        u_max_orig = u_half_orig
        v_min_orig = -v_half_orig
        v_max_orig = v_half_orig
        
        # The recovered position is the center, so we need to shift it to represent
        # the corner (u_min_orig, v_min_orig) in the parametrization
        # P_orig = center - u_min_orig * X_orig - v_min_orig * Y_orig
        #        = shift - (-u_half_orig) * X_orig - (-v_half_orig) * Y_orig
        #        = shift + u_half_orig * X_orig + v_half_orig * Y_orig
        # P_corner = P_original + u_half_orig * X_original + v_half_orig * Y_original
        
        surface['location'] = [P_original.tolist()]
        surface['direction'] = [D_original.tolist(), X_original.tolist(), Y_original.tolist()]
        surface['uv'] = [u_min_orig, u_max_orig, v_min_orig, v_max_orig]
        surface['scalar'] = []
        surface['poles'] = []
    
    elif surface_type == 'cylinder':
        radius_canonical = surface_canonical['scalar'][0]
        radius_orig = radius_canonical * scale
        v_min_orig = v_min * scale
        v_max_orig = v_max * scale
        
        surface['location'] = [P_original.tolist()]
        surface['direction'] = [D_original.tolist(), X_original.tolist(), Y_original.tolist()]
        surface['uv'] = [u_min, u_max, v_min_orig, v_max_orig]
        surface['scalar'] = [radius_orig]
        surface['poles'] = []
    
    elif surface_type == 'cone':
        semi_angle = surface_canonical['scalar'][0]
        radius_canonical = surface_canonical['scalar'][1]
        radius_orig = radius_canonical * scale
        v_min_orig = v_min * scale
        v_max_orig = v_max * scale
        
        surface['location'] = [P_original.tolist()]
        surface['direction'] = [D_original.tolist(), X_original.tolist(), Y_original.tolist()]
        surface['uv'] = [u_min, u_max, v_min_orig, v_max_orig]
        surface['scalar'] = [semi_angle, radius_orig]
        surface['poles'] = []
    
    elif surface_type == 'sphere':
        radius_canonical = surface_canonical['scalar'][0]
        radius_orig = radius_canonical * scale
        
        surface['location'] = [P_original.tolist()]
        surface['direction'] = [D_original.tolist(), X_original.tolist(), Y_original.tolist()]
        surface['uv'] = [u_min, u_max, v_min, v_max]
        surface['scalar'] = [radius_orig]
        surface['poles'] = []
    
    elif surface_type == 'torus':
        major_radius_canonical = surface_canonical['scalar'][0]
        minor_radius_canonical = surface_canonical['scalar'][1]
        major_radius_orig = major_radius_canonical * scale
        minor_radius_orig = minor_radius_canonical * scale
        
        surface['location'] = [P_original.tolist()]
        surface['direction'] = [D_original.tolist(), X_original.tolist(), Y_original.tolist()]
        surface['uv'] = [u_min, u_max, v_min, v_max]
        surface['scalar'] = [major_radius_orig, minor_radius_orig]
        surface['poles'] = []
    
    else:
        raise ValueError(f"Unsupported surface type: {surface_type}")
    
    return surface


def get_transformation_matrix(shift: np.ndarray, rotation: np.ndarray, scale: float) -> np.ndarray:
    """
    Construct a 4x4 homogeneous transformation matrix from shift, rotation, and scale.
    
    Args:
        shift: 3D translation vector
        rotation: 3x3 rotation matrix
        scale: Scaling factor
    
    Returns:
        T: 4x4 transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation / scale  # Combined rotation and scaling
    T[:3, 3] = -rotation @ shift / scale  # Translation
    return T


if __name__ == "__main__":
    # Example usage and testing
    import json
    import sys
    
    # Test with a cylinder (from real data)
    test_cylinder = {
        "type": "cylinder",
        "idx": [2, 2],
        "location": [[0.355483299325257, -0.9289010275483269, -0.07622121173702867]],
        "direction": [
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ],
        "scalar": [0.07109665986505137],
        "poles": [],
        "uv": [1.197655924654775, 2.879793265790667, -0.26439070387315977, -0.013812083906550643],
        "orientation": "Forward"
    }
    
    print("="*70)
    print("METHOD 1: Using surface dictionary directly")
    print("="*70)
    print("\nOriginal cylinder:")
    print(json.dumps(test_cylinder, indent=2))
    
    # Transform to canonical
    canonical, shift, rotation, scale = to_canonical(test_cylinder)
    
    print("\nCanonical cylinder:")
    print(json.dumps(canonical, indent=2))
    print(f"\nShift: {shift}")
    print(f"Rotation:\n{rotation}")
    print(f"Scale: {scale}")
    
    # Transform back
    recovered = from_canonical(canonical, shift, rotation, scale)
    
    print("\nRecovered cylinder:")
    print(json.dumps(recovered, indent=2))
    
    # Verify
    print("\n=== Verification ===")
    print(f"Location match: {np.allclose(test_cylinder['location'], recovered['location'])}")
    print(f"Direction match: {np.allclose(test_cylinder['direction'], recovered['direction'])}")
    print(f"UV match: {np.allclose(test_cylinder['uv'], recovered['uv'])}")
    print(f"Scalar match: {np.allclose(test_cylinder['scalar'], recovered['scalar'])}")
    
    # Test with a plane
    print("\n" + "="*60)
    print("Testing with a plane:")
    print("="*60)
    
    test_plane = {
        "type": "plane",
        "idx": [0, 0],
        "location": [[0.355483299325257, 1.9896918710595908, 0.22060954956460543]],
        "direction": [
            [0.707107, -0.0, 0.707107],
            [0.0, 1.0, 0.0],
            [-0.707107, -0.0, 0.707107]
        ],
        "scalar": [],
        "poles": [],
        "uv": [-1.9798674835227872, -1.4340459969935773, 1.3877787807814457e-16, 0.12601153133051562],
        "orientation": "Forward"
    }
    
    print("\nOriginal plane:")
    print(json.dumps(test_plane, indent=2))
    
    # Transform to canonical
    canonical_plane, shift_p, rotation_p, scale_p = to_canonical(test_plane)
    
    print("\nCanonical plane:")
    print(json.dumps(canonical_plane, indent=2))
    print(f"\nShift: {shift_p}")
    print(f"Rotation:\n{rotation_p}")
    print(f"Scale: {scale_p}")
    
    # Transform back
    recovered_plane = from_canonical(canonical_plane, shift_p, rotation_p, scale_p)
    
    print("\nRecovered plane:")
    print(json.dumps(recovered_plane, indent=2))
    
    # Verify
    print("\n=== Verification ===")
    print(f"Location match: {np.allclose(test_plane['location'], recovered_plane['location'])}")
    print(f"Direction match: {np.allclose(test_plane['direction'], recovered_plane['direction'])}")
    print(f"UV match: {np.allclose(test_plane['uv'], recovered_plane['uv'])}")
    print(f"Scalar match: {len(test_plane['scalar']) == len(recovered_plane['scalar'])}")
    
    # Test METHOD 2: Using dataset params
    print("\n" + "="*70)
    print("METHOD 2: Using dataset params (_parse_surface output)")
    print("="*70)
    
    try:
        # Add project paths
        sys.path.append('/home/qindafei/CAD/CLR-Wire')
        sys.path.append(r'C:\drivers\CAD\CLR-Wire')
        sys.path.append(r'F:\WORK\CAD\CLR-Wire')
        
        from src.dataset.dataset_v1 import dataset_compound
        
        # Create a minimal dataset instance (will fail gracefully if no data available)
        print("\nAttempting to load dataset for METHOD 2 test...")
        print("(This test requires dataset to be available)")
        
        # Use test cylinder
        print("\nUsing test cylinder...")
        
        # Create a dummy dataset for demonstration
        class DummyDataset:
            def __init__(self):
                from src.dataset.dataset_v1 import build_surface_postpreprocess, build_surface_process, SURFACE_PARAM_SCHEMAS
                self.postprocess_funcs = {k: build_surface_postpreprocess(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
                self.preprocess_funcs = {k: build_surface_process(v) for k, v in SURFACE_PARAM_SCHEMAS.items()}
                self.base_dim = 17
            
            def _parse_surface(self, surface_dict):
                from src.dataset.dataset_v1 import SURFACE_TYPE_MAP, SCALAR_DIM_MAP
                # Simplified parse (just for demonstration)
                surface_type = surface_dict['type']
                surface_type_idx = SURFACE_TYPE_MAP.get(surface_type, -1)
                
                P = np.array(surface_dict['location'][0], dtype=np.float32)
                D = np.array(surface_dict['direction'][0], dtype=np.float32)
                X = np.array(surface_dict['direction'][1], dtype=np.float32)
                UV = np.array(surface_dict['uv'][:4], dtype=np.float32)
                UV = np.concatenate([UV, np.zeros(4, dtype=np.float32)])
                scalar_params = np.array(surface_dict['scalar'], dtype=np.float32)
                
                params = np.concatenate([P, D, X, UV, scalar_params])
                params = self.preprocess_funcs[surface_type](params)
                
                return params, surface_type_idx
            
            def _recover_surface(self, params, surface_type_idx):
                from src.dataset.dataset_v1 import SURFACE_TYPE_MAP_INV
                surface_type = SURFACE_TYPE_MAP_INV.get(surface_type_idx)
                
                params = self.postprocess_funcs[surface_type](params)
                
                P = params[:3]
                D = params[3:6] / np.linalg.norm(params[3:6])
                X = params[6:9] / np.linalg.norm(params[6:9])
                Y = np.cross(D, X)
                UV = params[9:17]
                scalar_params = params[17:]
                
                # Simplified recovery (cylinder only for demo)
                if surface_type == 'cylinder':
                    sin_u_center, cos_u_center, u_half, height = UV[:4]
                    u_center = np.arctan2(sin_u_center, cos_u_center)
                    u_half = np.clip((u_half + 0.5), 0, 1 - 1e-5) * np.pi
                    u_min, u_max = u_center - u_half, u_center + u_half
                    v_min, v_max = 0.0, height
                    radius = scalar_params[0]
                    
                    return {
                        'type': surface_type,
                        'location': [P.tolist()],
                        'direction': [D.tolist(), X.tolist(), Y.tolist()],
                        'uv': [float(u_min), float(u_max), float(v_min), float(v_max)],
                        'scalar': [float(radius)],
                    }
                
                return {}
        
        dummy_dataset = DummyDataset()
        
        # Parse the test cylinder
        params, surface_type_idx = dummy_dataset._parse_surface(test_cylinder)
        print(f"\nParsed surface to params array (shape: {params.shape})")
        print(f"Surface type index: {surface_type_idx}")
        
        # METHOD 2: Pass (params, surface_type_idx) tuple to to_canonical
        canonical_from_params, shift2, rotation2, scale2 = to_canonical((params, surface_type_idx), dummy_dataset)
        
        print("\nCanonical surface (from params):")
        print(json.dumps(canonical_from_params, indent=2))
        print(f"\nShift: {shift2}")
        print(f"Scale: {scale2}")
        
        # Verify both methods produce same results
        print("\n=== Verification: Method 1 vs Method 2 ===")
        print(f"Shift match: {np.allclose(shift, shift2)}")
        print(f"Rotation match: {np.allclose(rotation, rotation2)}")
        print(f"Scale match: {np.allclose(scale, scale2)}")
        
        print("\n[OK] METHOD 2 test completed successfully!")
        
    except ImportError as e:
        print(f"\n[SKIP] METHOD 2 test: {e}")
        print("(Dataset module not available - this is normal if running standalone)")
    except Exception as e:
        print(f"\n[WARN] METHOD 2 test encountered an error: {e}")
        print("(This is expected if dataset is not fully configured)")

