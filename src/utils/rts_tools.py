"""
Rotation-Translation-Scale (RTS) quantization tools.

This module provides spherical k-means clustering on quaternions for rotation quantization.
"""

import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm


def rotate_under_axis(angles, axes):
    angles = [0, 90, 180, 270]
    axes = [
        # [1, 0, 0], 
        # [0, 1, 0], 
        [0, 0, 1]   
    ]

    rotation_angle = random.choice(angles)   # 从角度列表中选择一个角度
    rotation_axis = random.choice(axes)    # 从旋转轴列表中选择一个轴
    # rotation_angle = angles[3]
    radian = np.pi / 180 * rotation_angle
    
    rotation = R.from_rotvec(radian * np.array(rotation_axis))

    return rotation
    
class RotationCodebook:
    """
    Uniform rotation quantization using Euler angles.
    
    Divides each Euler angle (roll, pitch, yaw) uniformly into bins.
    Uses independent per-angle quantization to avoid memory issues.
    Internally uses quaternions to avoid gimbal lock issues during conversion.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of bins per Euler angle (each angle divided into this many bins)
        """
        self.bins_per_angle = codebook_size  # Number of bins per Euler angle
        # Theoretical codebook size if all combinations were stored
        self.codebook_size = self.bins_per_angle ** 3
        
        self.angle_bins = None  # (bins_per_angle,) bin centers for each angle in radians
        self.is_fitted = False
        
    def _rotation_to_euler(self, rotations: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to Euler angles (XYZ convention).
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            
        Returns:
            euler_angles: (N, 3) Euler angles in radians [roll, pitch, yaw] (XYZ)
        """
        scipy_rots = R.from_matrix(rotations)
        # Use 'xyz' convention (roll-pitch-yaw)
        # Suppress gimbal lock warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Gimbal lock.*')
            euler_angles = scipy_rots.as_euler('xyz', degrees=False)
        return euler_angles
    
    def _euler_to_rotation(self, euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to rotation matrices.
        
        Args:
            euler_angles: (N, 3) Euler angles in radians [roll, pitch, yaw] (XYZ)
            
        Returns:
            rotations: (N, 3, 3) rotation matrices
        """
        scipy_rots = R.from_euler('xyz', euler_angles, degrees=False)
        return scipy_rots.as_matrix()
    
    def fit(self, rotations: np.ndarray = None, max_iter: int = 100, 
            tol: float = 1e-4, random_seed: int = 42, verbose: bool = True) -> Dict:
        """
        Create uniform rotation codebook by dividing Euler angles uniformly.
        
        Args:
            rotations: (N, 3, 3) rotation matrices (optional, only used for statistics)
            max_iter: Not used (kept for API compatibility)
            tol: Not used (kept for API compatibility)
            random_seed: Not used (kept for API compatibility)
            verbose: Print progress
            
        Returns:
            stats: Dictionary with training statistics (if rotations provided)
        """
        if verbose:
            print(f"Creating uniform rotation codebook...")
            print(f"Bins per Euler angle: {self.bins_per_angle}")
            print(f"Theoretical codebook size: {self.codebook_size} ({self.bins_per_angle}³)")
        
        # Create uniform bins for each Euler angle in range [0, 2π)
        # We use [0, 2π) instead of [-π, π) to cover the full rotation space uniformly
        self.angle_bins = np.linspace(0, 2 * np.pi, self.bins_per_angle, endpoint=False)
        
        if verbose:
            print(f"Euler angle bins: [{np.rad2deg(self.angle_bins[0]):.2f}°, "
                  f"{np.rad2deg(self.angle_bins[-1]):.2f}°] with {self.bins_per_angle} bins")
            print(f"Bin spacing: {np.rad2deg(self.angle_bins[1] - self.angle_bins[0]):.2f}°")
        
        self.is_fitted = True
        
        # Compute statistics if rotations provided
        if rotations is not None:
            stats = self._compute_statistics(rotations, verbose=verbose)
        else:
            stats = {
                'codebook_size': self.codebook_size,
                'bins_per_angle': self.bins_per_angle,
            }
        
        return stats
    
    def _compute_statistics(self, rotations: np.ndarray, verbose: bool = True) -> Dict:
        """
        Compute error statistics for the codebook.
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            verbose: Print statistics
            
        Returns:
            stats: Dictionary with error statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        if verbose:
            print("Computing reconstruction statistics...")
        
        # Encode and decode
        indices = self.encode(rotations, verbose=verbose)
        reconstructed_rots = self.decode(indices)
        
        # Compute rotation error using Frobenius norm
        frobenius_errors = np.linalg.norm(rotations - reconstructed_rots, axis=(1, 2))
        
        # Compute geodesic distance on SO(3)
        if verbose:
            print("Computing geodesic distances...")
        geodesic_errors = []
        iterator = range(len(rotations))
        if verbose:
            iterator = tqdm(iterator, desc="Geodesic distances")
        for i in iterator:
            R_diff = rotations[i].T @ reconstructed_rots[i]
            trace = np.trace(R_diff)
            # theta = arccos((trace - 1) / 2)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            geodesic_errors.append(angle)
        geodesic_errors = np.array(geodesic_errors)
        geodesic_errors_deg = np.rad2deg(geodesic_errors)
        
        # Compute Euler angle utilization
        unique_bins_per_angle = [len(np.unique(indices[:, dim])) for dim in range(3)]
        avg_utilization = np.mean([u / self.bins_per_angle * 100 for u in unique_bins_per_angle])
        
        # Compute unique 3D combinations
        unique_combinations = len(np.unique(indices, axis=0))
        combination_utilization = unique_combinations / self.codebook_size * 100
        
        stats = {
            'codebook_size': self.codebook_size,
            'bins_per_angle': self.bins_per_angle,
            'n_samples': len(rotations),
            # SO(3) geodesic distance stats
            'geodesic_error_mean_deg': float(geodesic_errors_deg.mean()),
            'geodesic_error_std_deg': float(geodesic_errors_deg.std()),
            'geodesic_error_median_deg': float(np.median(geodesic_errors_deg)),
            'geodesic_error_max_deg': float(geodesic_errors_deg.max()),
            'geodesic_error_95th_deg': float(np.percentile(geodesic_errors_deg, 95)),
            # Frobenius norm stats
            'frobenius_error_mean': float(frobenius_errors.mean()),
            'frobenius_error_std': float(frobenius_errors.std()),
            # Utilization stats
            'bins_utilization_roll': int(unique_bins_per_angle[0]),
            'bins_utilization_pitch': int(unique_bins_per_angle[1]),
            'bins_utilization_yaw': int(unique_bins_per_angle[2]),
            'avg_bins_utilization_percent': float(avg_utilization),
            'unique_combinations': int(unique_combinations),
            'combination_utilization_percent': float(combination_utilization),
        }
        
        if verbose:
            print("\n" + "="*70)
            print("ROTATION CODEBOOK STATISTICS (Uniform Euler Angle)")
            print("="*70)
            print(f"Bins per Euler angle: {self.bins_per_angle}")
            print(f"Theoretical codebook size: {stats['codebook_size']} ({self.bins_per_angle}³)")
            print(f"Number of samples: {stats['n_samples']}")
            print()
            print("Per-angle bin utilization:")
            print(f"  Roll:  {stats['bins_utilization_roll']}/{self.bins_per_angle} bins")
            print(f"  Pitch: {stats['bins_utilization_pitch']}/{self.bins_per_angle} bins")
            print(f"  Yaw:   {stats['bins_utilization_yaw']}/{self.bins_per_angle} bins")
            print(f"  Average: {stats['avg_bins_utilization_percent']:.2f}%")
            print()
            print(f"Unique 3D combinations: {stats['unique_combinations']} ({stats['combination_utilization_percent']:.4f}% of theoretical max)")
            print()
            print("SO(3) Geodesic Errors (degrees):")
            print(f"  Mean:   {stats['geodesic_error_mean_deg']:.4f}°")
            print(f"  Std:    {stats['geodesic_error_std_deg']:.4f}°")
            print(f"  Median: {stats['geodesic_error_median_deg']:.4f}°")
            print(f"  95th:   {stats['geodesic_error_95th_deg']:.4f}°")
            print(f"  Max:    {stats['geodesic_error_max_deg']:.4f}°")
            print()
            print("Frobenius Norm Errors:")
            print(f"  Mean: {stats['frobenius_error_mean']:.6f}")
            print(f"  Std:  {stats['frobenius_error_std']:.6f}")
            print("="*70 + "\n")
        
        return stats
    
    def encode(self, rotations: np.ndarray, batch_size: int = 10000, verbose: bool = False) -> np.ndarray:
        """
        Encode rotation matrices to per-angle bin indices.
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            batch_size: Not used (kept for API compatibility)
            verbose: Show progress bar (not used for this fast operation)
            
        Returns:
            indices: (N, 3) per-angle bin indices [roll_idx, pitch_idx, yaw_idx]
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Ensure rotations has correct shape
        if rotations.ndim == 2:
            # Single rotation matrix (3, 3) -> add batch dimension
            rotations = rotations.reshape(1, 3, 3)
        elif rotations.ndim != 3:
            raise ValueError(f"Expected rotations to have shape (N, 3, 3), got {rotations.shape}")
        
        assert rotations.shape[1] == 3 and rotations.shape[2] == 3, \
            f"Expected rotations to have shape (N, 3, 3), got {rotations.shape}"
        
        N = rotations.shape[0]
        
        # Convert to Euler angles
        euler_angles = self._rotation_to_euler(rotations)  # (N, 3)
        
        # Normalize to [0, 2π) range
        euler_angles = np.mod(euler_angles, 2 * np.pi)
        
        # Quantize each angle independently
        indices = np.zeros((N, 3), dtype=np.int32)
        
        for angle_idx in range(3):
            # Compute distances to all bins for this angle: (N, bins_per_angle)
            angles = euler_angles[:, angle_idx:angle_idx+1]  # (N, 1)
            
            # Handle circular distance (shortest path on circle)
            # Distance on circle: min(|a - b|, 2π - |a - b|)
            diff = np.abs(angles - self.angle_bins[np.newaxis, :])  # (N, bins_per_angle)
            circular_dist = np.minimum(diff, 2 * np.pi - diff)
            
            # Find nearest bin for each sample
            indices[:, angle_idx] = np.argmin(circular_dist, axis=1)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode per-angle bin indices to rotation matrices.
        
        Args:
            indices: (N, 3) per-angle bin indices [roll_idx, pitch_idx, yaw_idx]
            
        Returns:
            rotations: (N, 3, 3) rotation matrices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Ensure indices has correct shape
        if indices.ndim == 1:
            # If 1D with length 3, treat as single sample
            if len(indices) == 3:
                indices = indices.reshape(1, 3)
            else:
                raise ValueError(f"Cannot decode 1D indices with length {len(indices)}, expected (N, 3)")
        
        assert indices.shape[1] == 3, f"Expected indices to have shape (N, 3), got {indices.shape}"
        
        # Look up bin centers for each angle
        euler_angles = self.angle_bins[indices]  # (N, 3)
        
        # Convert Euler angles to rotation matrices (using quaternions internally to avoid gimbal lock)
        rotations = self._euler_to_rotation(euler_angles)
        
        return rotations
    
    def save(self, save_path: Union[str, Path]):
        """
        Save codebook to file.
        
        Args:
            save_path: Path to save the codebook
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'bins_per_angle': self.bins_per_angle,
            'codebook_size': self.codebook_size,
            'angle_bins': self.angle_bins,
            'is_fitted': self.is_fitted,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Rotation codebook saved to: {save_path}")
    
    def load(self, load_path: Union[str, Path]):
        """
        Load codebook from file.
        
        Args:
            load_path: Path to load the codebook from
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bins_per_angle = data['bins_per_angle']
        self.codebook_size = data['codebook_size']
        self.angle_bins = data['angle_bins']
        self.is_fitted = data['is_fitted']
        
        print(f"Rotation codebook loaded from: {load_path}")
        print(f"  Bins per Euler angle: {self.bins_per_angle}")
        print(f"  Theoretical codebook size: {self.codebook_size} ({self.bins_per_angle}³)")


def build_rotation_codebook_from_cache(
    cache_path: str,
    codebook_size: int,
    output_path: str,
    max_iter: int = 100,
    random_seed: int = 42,
) -> RotationCodebook:
    """
    Build uniform rotation codebook and optionally evaluate on dataset.
    
    Args:
        cache_path: Path to .npz cache file containing 'rotations' (for evaluation only)
        codebook_size: Number of bins per Euler angle
        output_path: Path to save the codebook
        max_iter: Not used (kept for API compatibility)
        random_seed: Not used (kept for API compatibility)
        
    Returns:
        codebook: Fitted RotationCodebook instance
    """
    # Load rotations for evaluation
    if cache_path:
        print(f"Loading rotations from: {cache_path}")
        data = np.load(cache_path)
        
        if 'rotations' not in data:
            raise KeyError(f"Cache file does not contain 'rotations' key. Available keys: {list(data.keys())}")
        
        rotations = data['rotations']
        print(f"Loaded {len(rotations)} rotation matrices")
    else:
        rotations = None
    
    # Create and fit codebook (uniform, data-independent)
    codebook = RotationCodebook(codebook_size)
    stats = codebook.fit(rotations, verbose=True)
    
    # Save codebook
    codebook.save(output_path)
    
    # Save statistics
    stats_path = Path(output_path).with_suffix('.stats.txt')
    with open(stats_path, 'w') as f:
        f.write("ROTATION CODEBOOK STATISTICS (Uniform Euler Angle)\n")
        f.write("="*70 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    return codebook


class TranslationCodebook:
    """
    Linear quantization for translation vectors.
    
    Uses independent per-dimension quantization with shared bins across all dimensions.
    Each dimension is quantized independently using the same bin structure (global range).
    This is memory-efficient: stores only N bins instead of N^3 3D grid points.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of bins per dimension
        """
        self.bins_per_dim = codebook_size  # Number of bins per dimension
        # Theoretical codebook size if all combinations were stored (for reference)
        self.actual_codebook_size = self.bins_per_dim ** 3
        
        self.global_min = None  # Global minimum value across all dimensions
        self.global_max = None  # Global maximum value across all dimensions
        self.bins = None  # (bins_per_dim,) bin centers shared across all dimensions
        self.is_fitted = False
        
    def fit(
        self,
        translations: np.ndarray,
        percentile: float = 95.0,
        interval_scale: float = 1.0,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit translation codebook by creating uniform grid covering specified percentile.
        All dimensions share the same bin structure (same range and number of bins).
        
        Args:
            translations: (N, 3) translation vectors
            percentile: Percentage of data to cover (default: 95%)
            verbose: Print progress
            
        Returns:
            stats: Dictionary with statistics
        """
        # Ensure translations has correct shape
        if translations.ndim == 1:
            # If 1D, reshape to (N, 1) and replicate to (N, 3)
            translations = np.tile(translations.reshape(-1, 1), (1, 3))
        elif translations.ndim > 2:
            # If more than 2D, flatten and reshape
            translations = translations.reshape(-1, 3)
        
        assert translations.shape[1] == 3, f"Expected translations to have 3 dimensions, got shape {translations.shape}"
        
        N = translations.shape[0]
        if verbose:
            print(f"Fitting translation codebook with {N} samples...")
            print(f"Bins per dimension: {self.bins_per_dim}")
            print(f"Total codebook entries: {self.actual_codebook_size} ({self.bins_per_dim}^3)")
        
        # Compute global percentile bounds (same for all dimensions)
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        # Flatten all dimensions to compute global range
        translations_flat = translations.flatten()
        self.global_min = np.percentile(translations_flat, lower_percentile)
        self.global_max = np.percentile(translations_flat, upper_percentile)

        # Optionally expand the interval symmetrically around 0
        # Ensure the range crosses 0 so that scaling makes sense
        if interval_scale != 1.0:
            assert self.global_min < 0 and self.global_max > 0, (
                f"Translation percentile range must cross 0 before applying interval_scale, "
                f"got [{self.global_min:.4f}, {self.global_max:.4f}]"
            )
            assert interval_scale > 1.0, f"interval_scale must be > 1, got {interval_scale}"
            self.global_min *= interval_scale
            self.global_max *= interval_scale
        
        if verbose:
            print(f"Global translation range (covering {percentile}% of data):")
            print(f"  All dimensions: [{self.global_min:.4f}, {self.global_max:.4f}]")
            print(f"Per-dimension statistics:")
            for i, dim in enumerate(['X', 'Y', 'Z']):
                dim_min = translations[:, i].min()
                dim_max = translations[:, i].max()
                print(f"  {dim}: min={dim_min:.4f}, max={dim_max:.4f}")
        
        # Create uniform bins using global range (shared across all dimensions)
        # Store only the 1D bins, not the full 3D grid (memory efficient)
        self.bins = np.linspace(self.global_min, self.global_max, self.bins_per_dim)
        
        self.is_fitted = True
        
        # Compute statistics
        stats = self._compute_statistics(translations, verbose=verbose)
        
        return stats
    
    def _compute_statistics(self, translations: np.ndarray, verbose: bool = True) -> Dict:
        """
        Compute error statistics for the codebook.
        
        Args:
            translations: (N, 3) translation vectors
            verbose: Print statistics
            
        Returns:
            stats: Dictionary with error statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        if verbose:
            print("Computing reconstruction statistics...")
        
        # Encode and decode with batching
        indices = self.encode(translations, batch_size=10000, verbose=verbose)
        reconstructed = self.decode(indices)
        
        # Compute errors
        errors = translations - reconstructed  # (N, 3)
        l2_errors = np.linalg.norm(errors, axis=1)  # (N,)
        l1_errors = np.abs(errors).sum(axis=1)  # (N,)
        linf_errors = np.abs(errors).max(axis=1)  # (N,)
        
        # Per-dimension errors
        per_dim_errors = np.abs(errors)  # (N, 3)
        
        # Check coverage (percentage of data within bounds)
        in_bounds = np.all((translations >= self.global_min) & (translations <= self.global_max), axis=1)
        coverage = in_bounds.mean() * 100
        
        # Compute codebook utilization (per-dimension)
        # For independent dimension quantization, we track unique bins per dimension
        unique_bins_per_dim = [len(np.unique(indices[:, dim])) for dim in range(3)]
        avg_utilization = np.mean([u / self.bins_per_dim * 100 for u in unique_bins_per_dim])
        
        # Compute unique 3D combinations (for comparison with theoretical max)
        unique_combinations = len(np.unique(indices, axis=0))
        combination_utilization = unique_combinations / self.actual_codebook_size * 100
        
        stats = {
            'codebook_size': self.actual_codebook_size,
            'bins_per_dim': self.bins_per_dim,
            'n_samples': len(translations),
            'coverage_percent': float(coverage),
            'bins_utilization_x': int(unique_bins_per_dim[0]),
            'bins_utilization_y': int(unique_bins_per_dim[1]),
            'bins_utilization_z': int(unique_bins_per_dim[2]),
            'avg_bins_utilization_percent': float(avg_utilization),
            'unique_combinations': int(unique_combinations),
            'combination_utilization_percent': float(combination_utilization),
            # L2 error stats
            'l2_error_mean': float(l2_errors.mean()),
            'l2_error_std': float(l2_errors.std()),
            'l2_error_median': float(np.median(l2_errors)),
            'l2_error_max': float(l2_errors.max()),
            'l2_error_95th': float(np.percentile(l2_errors, 95)),
            # L1 error stats
            'l1_error_mean': float(l1_errors.mean()),
            'l1_error_median': float(np.median(l1_errors)),
            # Linf error stats
            'linf_error_mean': float(linf_errors.mean()),
            'linf_error_max': float(linf_errors.max()),
            # Per-dimension error stats
            'x_error_mean': float(per_dim_errors[:, 0].mean()),
            'y_error_mean': float(per_dim_errors[:, 1].mean()),
            'z_error_mean': float(per_dim_errors[:, 2].mean()),
            'x_error_max': float(per_dim_errors[:, 0].max()),
            'y_error_max': float(per_dim_errors[:, 1].max()),
            'z_error_max': float(per_dim_errors[:, 2].max()),
        }
        
        if verbose:
            print("\n" + "="*70)
            print("TRANSLATION CODEBOOK STATISTICS")
            print("="*70)
            print(f"Bins per dimension: {self.bins_per_dim}")
            print(f"Theoretical codebook size: {stats['codebook_size']} ({self.bins_per_dim}³)")
            print(f"Number of samples: {stats['n_samples']}")
            print(f"Coverage: {stats['coverage_percent']:.2f}%")
            print()
            print("Per-dimension bin utilization:")
            print(f"  X: {stats['bins_utilization_x']}/{self.bins_per_dim} bins")
            print(f"  Y: {stats['bins_utilization_y']}/{self.bins_per_dim} bins")
            print(f"  Z: {stats['bins_utilization_z']}/{self.bins_per_dim} bins")
            print(f"  Average: {stats['avg_bins_utilization_percent']:.2f}%")
            print()
            print(f"Unique 3D combinations: {stats['unique_combinations']} ({stats['combination_utilization_percent']:.4f}% of theoretical max)")
            print()
            print("L2 Reconstruction Errors:")
            print(f"  Mean:   {stats['l2_error_mean']:.6f}")
            print(f"  Std:    {stats['l2_error_std']:.6f}")
            print(f"  Median: {stats['l2_error_median']:.6f}")
            print(f"  95th:   {stats['l2_error_95th']:.6f}")
            print(f"  Max:    {stats['l2_error_max']:.6f}")
            print()
            print("Per-Dimension Errors (mean):")
            print(f"  X: {stats['x_error_mean']:.6f}")
            print(f"  Y: {stats['y_error_mean']:.6f}")
            print(f"  Z: {stats['z_error_mean']:.6f}")
            print("="*70 + "\n")
        
        return stats
    
    def encode(self, translations: np.ndarray, batch_size: int = 10000, verbose: bool = False) -> np.ndarray:
        """
        Encode translation vectors to per-dimension bin indices.
        
        Args:
            translations: (N, 3) translation vectors
            batch_size: Not used (kept for API compatibility)
            verbose: Show progress bar (not used for this fast operation)
            
        Returns:
            indices: (N, 3) per-dimension bin indices, each in range [0, bins_per_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Ensure translations has correct shape
        if translations.ndim == 1:
            # If 1D, reshape to (N, 1) and replicate to (N, 3)
            translations = np.tile(translations.reshape(-1, 1), (1, 3))
        elif translations.ndim > 2:
            # If more than 2D, flatten and reshape
            translations = translations.reshape(-1, 3)
        
        assert translations.shape[1] == 3, f"Expected translations to have 3 dimensions, got shape {translations.shape}"
        
        N = translations.shape[0]
        
        # Clip to bounds (same bounds for all dimensions)
        translations_clipped = np.clip(translations, self.global_min, self.global_max)
        
        # Quantize each dimension independently using the shared bins
        # For each dimension, find the nearest bin
        indices = np.zeros((N, 3), dtype=np.int32)
        
        for dim in range(3):
            # Compute distances to all bins for this dimension: (N, bins_per_dim)
            distances = np.abs(translations_clipped[:, dim:dim+1] - self.bins[np.newaxis, :])
            # Find nearest bin for each sample
            indices[:, dim] = np.argmin(distances, axis=1)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode per-dimension bin indices to translation vectors.
        
        Args:
            indices: (N, 3) per-dimension bin indices
            
        Returns:
            translations: (N, 3) translation vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Ensure indices has correct shape
        if indices.ndim == 1:
            # If 1D with length 3, treat as single sample
            if len(indices) == 3:
                indices = indices.reshape(1, 3)
            else:
                raise ValueError(f"Cannot decode 1D indices with length {len(indices)}, expected (N, 3)")
        
        assert indices.shape[1] == 3, f"Expected indices to have shape (N, 3), got {indices.shape}"
        
        # Look up bin centers for each dimension
        translations = self.bins[indices]  # (N, 3)
        
        return translations
    
    def save(self, save_path: Union[str, Path]):
        """Save codebook to file."""
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'bins_per_dim': self.bins_per_dim,
            'actual_codebook_size': self.actual_codebook_size,
            'global_min': self.global_min,
            'global_max': self.global_max,
            'bins': self.bins,
            'is_fitted': self.is_fitted,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Translation codebook saved to: {save_path}")
    
    def load(self, load_path: Union[str, Path]):
        """Load codebook from file."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bins_per_dim = data['bins_per_dim']
        self.actual_codebook_size = data['actual_codebook_size']
        self.global_min = data['global_min']
        self.global_max = data['global_max']
        self.bins = data['bins']
        self.is_fitted = data['is_fitted']
        
        print(f"Translation codebook loaded from: {load_path}")
        print(f"  Bins per dimension: {self.bins_per_dim}")
        print(f"  Total possible combinations: {self.actual_codebook_size} ({self.bins_per_dim}³)")
        print(f"  Global range: [{self.global_min:.4f}, {self.global_max:.4f}]")


class ScaleCodebook:
    """
    Logarithmic quantization for scalar scale factors.
    
    Applies log transform to scales, then creates uniform 1D grid in log-space
    covering 95% of the data range.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of codebook entries
        """
        self.codebook_size = codebook_size
        
        self.log_min = None  # Minimum log-scale value (2.5th percentile)
        self.log_max = None  # Maximum log-scale value (97.5th percentile)
        self.log_codebook = None  # (codebook_size,) codebook entries in log-space
        self.is_fitted = False
        
    def fit(
        self,
        scales: np.ndarray,
        percentile: float = 95.0,
        interval_scale: float = 1.0,
        eps: float = 1e-8,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit scale codebook by creating uniform 1D grid in log-space covering specified percentile.
        
        Args:
            scales: (N,) scalar scale factors (positive values)
            percentile: Percentage of data to cover (default: 95%)
            eps: Small epsilon to avoid log(0)
            verbose: Print progress
            
        Returns:
            stats: Dictionary with statistics
        """
        # Ensure scales is 1D
        if scales.ndim > 1:
            scales = scales.flatten()
        
        N = scales.shape[0]
        if verbose:
            print(f"Fitting scale codebook with {N} samples, {self.codebook_size} bins...")
        
        # Take absolute value and add eps to avoid log(0)
        scales_abs = np.abs(scales) + eps
        
        # Apply log transform
        log_scales = np.log(scales_abs)
        
        # Compute percentile bounds in log-space
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        self.log_min = np.percentile(log_scales, lower_percentile)
        self.log_max = np.percentile(log_scales, upper_percentile)

        # Optionally expand the interval symmetrically around 0 in log-space
        # This assumes the log-scale range crosses 0 (i.e., some scales < 1 and some > 1)
        if interval_scale != 1.0:
            assert self.log_min < 0 and self.log_max > 0, (
                f"Scale log-percentile range must cross 0 before applying interval_scale, "
                f"got [{self.log_min:.4f}, {self.log_max:.4f}]"
            )
            assert interval_scale > 1.0, f"interval_scale must be > 1, got {interval_scale}"
            self.log_min *= interval_scale
            self.log_max *= interval_scale
        
        if verbose:
            min_scale = np.exp(self.log_min)
            max_scale = np.exp(self.log_max)
            print(f"Scale range in log-space (after optional interval_scale, covering ~{percentile}% of data):")
            print(f"  log=[{self.log_min:.4f}, {self.log_max:.4f}] -> "
                  f"scale=[{min_scale:.4f}, {max_scale:.4f}]")
        
        # Create uniform 1D grid in log-space
        self.log_codebook = np.linspace(self.log_min, self.log_max, self.codebook_size)
        
        self.is_fitted = True
        
        # Compute statistics
        stats = self._compute_statistics(scales, eps=eps, verbose=verbose)
        
        return stats
    
    def _compute_statistics(self, scales: np.ndarray, eps: float = 1e-8, verbose: bool = True) -> Dict:
        """
        Compute error statistics for the codebook.
        
        Args:
            scales: (N,) scalar scale factors
            eps: Small epsilon to avoid log(0)
            verbose: Print statistics
            
        Returns:
            stats: Dictionary with error statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        if verbose:
            print("Computing reconstruction statistics...")
        
        # Encode and decode with batching
        indices = self.encode(scales, eps=eps, batch_size=10000, verbose=verbose)
        reconstructed = self.decode(indices)
        
        # Compute errors in original scale space
        errors = np.abs(scales - reconstructed)  # (N,)
        relative_errors = errors / (np.abs(scales) + eps)  # (N,)
        
        # Compute errors in log-space
        scales_abs = np.abs(scales) + eps
        reconstructed_abs = np.abs(reconstructed) + eps
        log_errors = np.abs(np.log(scales_abs) - np.log(reconstructed_abs))  # (N,)
        
        # Check coverage (percentage of data within bounds in log-space)
        log_scales = np.log(scales_abs)
        in_bounds = (log_scales >= self.log_min) & (log_scales <= self.log_max)
        coverage = in_bounds.mean() * 100
        
        # Compute codebook utilization
        unique_indices = np.unique(indices)
        utilization = len(unique_indices) / self.codebook_size * 100
        
        stats = {
            'codebook_size': self.codebook_size,
            'n_samples': len(scales),
            'coverage_percent': float(coverage),
            'codebook_utilization_percent': float(utilization),
            # Absolute error stats
            'abs_error_mean': float(errors.mean()),
            'abs_error_std': float(errors.std()),
            'abs_error_median': float(np.median(errors)),
            'abs_error_max': float(errors.max()),
            'abs_error_95th': float(np.percentile(errors, 95)),
            # Relative error stats
            'relative_error_mean': float(relative_errors.mean()),
            'relative_error_median': float(np.median(relative_errors)),
            'relative_error_max': float(relative_errors.max()),
            'relative_error_95th': float(np.percentile(relative_errors, 95)),
            # Log-space error stats
            'log_error_mean': float(log_errors.mean()),
            'log_error_median': float(np.median(log_errors)),
            'log_error_max': float(log_errors.max()),
            'log_error_95th': float(np.percentile(log_errors, 95)),
        }
        
        if verbose:
            print("\n" + "="*70)
            print("SCALE CODEBOOK STATISTICS (Scalar)")
            print("="*70)
            print(f"Codebook size: {stats['codebook_size']}")
            print(f"Number of samples: {stats['n_samples']}")
            print(f"Coverage: {stats['coverage_percent']:.2f}%")
            print(f"Codebook utilization: {stats['codebook_utilization_percent']:.2f}%")
            print()
            print("Absolute Errors:")
            print(f"  Mean:   {stats['abs_error_mean']:.6f}")
            print(f"  Std:    {stats['abs_error_std']:.6f}")
            print(f"  Median: {stats['abs_error_median']:.6f}")
            print(f"  95th:   {stats['abs_error_95th']:.6f}")
            print(f"  Max:    {stats['abs_error_max']:.6f}")
            print()
            print("Relative Errors:")
            print(f"  Mean:   {stats['relative_error_mean']:.6f}")
            print(f"  Median: {stats['relative_error_median']:.6f}")
            print(f"  95th:   {stats['relative_error_95th']:.6f}")
            print(f"  Max:    {stats['relative_error_max']:.6f}")
            print()
            print("Log-space Errors:")
            print(f"  Mean:   {stats['log_error_mean']:.6f}")
            print(f"  Median: {stats['log_error_median']:.6f}")
            print(f"  95th:   {stats['log_error_95th']:.6f}")
            print(f"  Max:    {stats['log_error_max']:.6f}")
            print("="*70 + "\n")
        
        return stats
    
    def encode(self, scales: np.ndarray, eps: float = 1e-8, batch_size: int = 10000, verbose: bool = False) -> np.ndarray:
        """
        Encode scalar scale factors to codebook indices.
        
        Args:
            scales: (N,) scalar scale factors
            eps: Small epsilon to avoid log(0)
            batch_size: Not used for 1D, kept for API compatibility
            verbose: Show progress bar
            
        Returns:
            indices: (N,) codebook indices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Ensure scales is 1D
        if scales.ndim > 1:
            scales = scales.flatten()
        
        # Apply log transform
        scales_abs = np.abs(scales) + eps
        log_scales = np.log(scales_abs)
        
        # Clip to bounds in log-space
        log_scales_clipped = np.clip(log_scales, self.log_min, self.log_max)
        
        # Find nearest codebook entry using searchsorted (efficient for 1D)
        # Compute distances to all codebook entries
        distances = np.abs(log_scales_clipped[:, np.newaxis] - self.log_codebook[np.newaxis, :])  # (N, K)
        indices = np.argmin(distances, axis=1).astype(np.int32)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode codebook indices to scalar scale factors.
        
        Args:
            indices: (N,) codebook indices
            
        Returns:
            scales: (N,) scalar scale factors
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Get log-space values
        log_scales = self.log_codebook[indices]
        
        # Transform back to original scale space
        scales = np.exp(log_scales)
        
        return scales
    
    def save(self, save_path: Union[str, Path]):
        """Save codebook to file."""
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'codebook_size': self.codebook_size,
            'log_min': self.log_min,
            'log_max': self.log_max,
            'log_codebook': self.log_codebook,
            'is_fitted': self.is_fitted,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Scale codebook saved to: {save_path}")
    
    def load(self, load_path: Union[str, Path]):
        """Load codebook from file."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Codebook file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.codebook_size = data['codebook_size']
        self.log_min = data['log_min']
        self.log_max = data['log_max']
        self.log_codebook = data['log_codebook']
        self.is_fitted = data['is_fitted']
        
        print(f"Scale codebook loaded from: {load_path}")
        print(f"  Codebook size: {self.codebook_size}")


def build_translation_codebook_from_cache(
    cache_path: str,
    codebook_size: int,
    output_path: str,
    percentile: float = 95.0,
    interval_scale: float = 1.0,
) -> TranslationCodebook:
    """
    Build translation codebook from a dataset cache file.
    
    Args:
        cache_path: Path to .npz cache file containing 'shifts'
        codebook_size: Number of codebook entries (approximate, will be rounded to nearest cube)
        output_path: Path to save the codebook
        percentile: Percentage of data to cover (default: 95%)
        
    Returns:
        codebook: Fitted TranslationCodebook instance
    """
    print(f"Loading translations from: {cache_path}")
    data = np.load(cache_path)
    
    if 'shifts' not in data:
        raise KeyError(f"Cache file does not contain 'shifts' key. Available keys: {list(data.keys())}")
    
    translations = data['shifts']
    print(f"Loaded translations with shape: {translations.shape}")
    print(f"Number of translation vectors: {len(translations)}")
    
    # Create and fit codebook
    codebook = TranslationCodebook(codebook_size)
    stats = codebook.fit(translations, percentile=percentile, interval_scale=interval_scale, verbose=True)
    
    # Save codebook
    codebook.save(output_path)
    
    # Save statistics
    stats_path = Path(output_path).with_suffix('.stats.txt')
    with open(stats_path, 'w') as f:
        f.write("TRANSLATION CODEBOOK STATISTICS\n")
        f.write("="*70 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    return codebook


def build_scale_codebook_from_cache(
    cache_path: str,
    codebook_size: int,
    output_path: str,
    percentile: float = 95.0,
    interval_scale: float = 1.0,
    eps: float = 1e-8,
) -> ScaleCodebook:
    """
    Build scale codebook from a dataset cache file.
    
    Args:
        cache_path: Path to .npz cache file containing 'scales'
        codebook_size: Number of codebook entries (approximate, will be rounded to nearest cube)
        output_path: Path to save the codebook
        percentile: Percentage of data to cover (default: 95%)
        eps: Small epsilon to avoid log(0)
        
    Returns:
        codebook: Fitted ScaleCodebook instance
    """
    print(f"Loading scales from: {cache_path}")
    data = np.load(cache_path)
    
    if 'scales' not in data:
        raise KeyError(f"Cache file does not contain 'scales' key. Available keys: {list(data.keys())}")
    
    scales = data['scales']
    print(f"Loaded scales with shape: {scales.shape}")
    print(f"Number of scale vectors: {len(scales)}")
    
    # Create and fit codebook
    codebook = ScaleCodebook(codebook_size)
    stats = codebook.fit(scales, percentile=percentile, interval_scale=interval_scale, eps=eps, verbose=True)
    
    # Save codebook
    codebook.save(output_path)
    
    # Save statistics
    stats_path = Path(output_path).with_suffix('.stats.txt')
    with open(stats_path, 'w') as f:
        f.write("SCALE CODEBOOK STATISTICS\n")
        f.write("="*70 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    return codebook


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build rotation/translation/scale codebooks')
    parser.add_argument('--type', type=str, required=True, choices=['rotation', 'translation', 'scale'],
                       help='Type of codebook to build')
    parser.add_argument('--cache_path', type=str, required=True,
                       help='Path to .npz cache file with data')
    parser.add_argument('--codebook_size', type=int, required=True,
                       help='Number of codebook entries (clusters)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the codebook (.pkl)')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum k-means iterations (rotation only)')
    parser.add_argument('--percentile', type=float, default=95.0,
                       help='Percentage of data to cover (translation/scale only)')
    parser.add_argument('--interval_scale', type=float, default=1.0,
                       help='Scale factor (>1) to expand translation/scale intervals around 0')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    if args.type == 'rotation':
        build_rotation_codebook_from_cache(
            cache_path=args.cache_path,
            codebook_size=args.codebook_size,
            output_path=args.output_path,
            max_iter=args.max_iter,
            random_seed=args.seed,
        )
    elif args.type == 'translation':
        build_translation_codebook_from_cache(
            cache_path=args.cache_path,
            codebook_size=args.codebook_size,
            output_path=args.output_path,
            percentile=args.percentile,
            interval_scale=args.interval_scale,
        )
    elif args.type == 'scale':
        build_scale_codebook_from_cache(
            cache_path=args.cache_path,
            codebook_size=args.codebook_size,
            output_path=args.output_path,
            percentile=args.percentile,
            interval_scale=args.interval_scale,
        )

