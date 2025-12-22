"""
Rotation-Translation-Scale (RTS) quantization tools.

This module provides spherical k-means clustering on quaternions for rotation quantization.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


class RotationCodebook:
    """
    Spherical K-means clustering for rotation matrices using quaternions.
    
    Quaternions lie on a 4D unit sphere, and q and -q represent the same rotation.
    This class handles the double-cover property and provides rotation quantization.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of cluster centers (codebook entries)
        """
        self.codebook_size = codebook_size
        self.centroids = None  # Shape: (codebook_size, 4), unit quaternions
        self.is_fitted = False
        
    def _rotation_to_quat(self, rotations: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrices to quaternions using scipy.
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            
        Returns:
            quaternions: (N, 4) unit quaternions in [w, x, y, z] format
        """
        N = rotations.shape[0]
        scipy_rots = R.from_matrix(rotations)
        quats = scipy_rots.as_quat()  # Returns [x, y, z, w] format
        
        # Convert to [w, x, y, z] format (scalar first)
        quats = np.concatenate([quats[:, 3:4], quats[:, :3]], axis=1)
        
        # Ensure consistent hemisphere: make w >= 0
        # This handles the q/-q ambiguity
        mask = quats[:, 0] < 0
        quats[mask] = -quats[mask]
        
        return quats
    
    def _quat_to_rotation(self, quats: np.ndarray) -> np.ndarray:
        """
        Convert quaternions to rotation matrices.
        
        Args:
            quats: (N, 4) quaternions in [w, x, y, z] format
            
        Returns:
            rotations: (N, 3, 3) rotation matrices
        """
        # Convert [w, x, y, z] to [x, y, z, w] for scipy
        quats_scipy = np.concatenate([quats[:, 1:4], quats[:, 0:1]], axis=1)
        scipy_rots = R.from_quat(quats_scipy)
        return scipy_rots.as_matrix()
    
    def _quaternion_distance(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Compute angular distance between quaternions on the hypersphere.
        Handles the double-cover property: distance is min(d(q1, q2), d(q1, -q2)).
        
        Args:
            q1: (N, 4) quaternions
            q2: (M, 4) quaternions (typically centroids)
            
        Returns:
            distances: (N, M) angular distances in radians
        """
        # Compute dot products: (N, M)
        dot_products = np.abs(q1 @ q2.T)  # abs handles q/-q symmetry
        
        # Clip to [-1, 1] to avoid numerical errors in arccos
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Angular distance on unit sphere
        distances = np.arccos(dot_products)
        
        return distances
    
    def _spherical_mean(self, quats: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the spherical mean of quaternions.
        
        Args:
            quats: (N, 4) quaternions
            weights: (N,) optional weights
            
        Returns:
            mean_quat: (4,) unit quaternion
        """
        if weights is None:
            weights = np.ones(len(quats))
        
        # Weighted average (not normalized)
        weighted_sum = (quats.T * weights).T.sum(axis=0)
        
        # Project back to unit sphere
        mean_quat = weighted_sum / (np.linalg.norm(weighted_sum) + 1e-10)
        
        # Ensure w >= 0
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
            
        return mean_quat
    
    def fit(self, rotations: np.ndarray, max_iter: int = 100, 
            tol: float = 1e-4, random_seed: int = 42, verbose: bool = True) -> Dict:
        """
        Fit spherical k-means on rotation matrices.
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            max_iter: Maximum number of iterations
            tol: Convergence tolerance (change in objective)
            random_seed: Random seed for initialization
            verbose: Print progress
            
        Returns:
            stats: Dictionary with training statistics
        """
        np.random.seed(random_seed)
        
        N = rotations.shape[0]
        if verbose:
            print(f"Fitting rotation codebook with {N} samples, {self.codebook_size} clusters...")
        
        # Convert to quaternions
        if verbose:
            print("Converting rotation matrices to quaternions...")
        quats = self._rotation_to_quat(rotations)
        
        # Initialize centroids: random selection from data
        indices = np.random.choice(N, self.codebook_size, replace=False)
        self.centroids = quats[indices].copy()
        
        # K-means iteration
        prev_objective = float('inf')
        history = []
        
        iterator = tqdm(range(max_iter), desc="K-means") if verbose else range(max_iter)
        
        for iter_idx in iterator:
            # E-step: Assign each quaternion to nearest centroid
            distances = self._quaternion_distance(quats, self.centroids)  # (N, K)
            assignments = np.argmin(distances, axis=1)  # (N,)
            
            # Compute objective (average distance)
            objective = distances[np.arange(N), assignments].mean()
            history.append(objective)
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'objective': f'{objective:.6f}'})
            
            # Check convergence
            if abs(prev_objective - objective) < tol:
                if verbose:
                    print(f"Converged at iteration {iter_idx + 1}")
                break
            prev_objective = objective
            
            # M-step: Update centroids as spherical mean
            for k in range(self.codebook_size):
                mask = assignments == k
                if mask.sum() == 0:
                    # Empty cluster: reinitialize randomly
                    self.centroids[k] = quats[np.random.randint(N)]
                else:
                    self.centroids[k] = self._spherical_mean(quats[mask])
        
        self.is_fitted = True
        
        # Compute final statistics
        stats = self._compute_statistics(rotations, verbose=verbose)
        stats['training_history'] = history
        stats['n_iterations'] = len(history)
        
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
        
        quats = self._rotation_to_quat(rotations)
        
        # Compute distances and assignments
        distances = self._quaternion_distance(quats, self.centroids)
        assignments = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(quats)), assignments]
        
        # Angular errors in degrees
        angular_errors_deg = np.rad2deg(min_distances)
        
        # Reconstruction: encode -> decode
        reconstructed_rots = self.decode(self.encode(rotations))
        
        # Compute rotation error using Frobenius norm
        frobenius_errors = np.linalg.norm(rotations - reconstructed_rots, axis=(1, 2))
        
        # Compute geodesic distance on SO(3)
        geodesic_errors = []
        for i in range(len(rotations)):
            R_diff = rotations[i].T @ reconstructed_rots[i]
            trace = np.trace(R_diff)
            # theta = arccos((trace - 1) / 2)
            angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
            geodesic_errors.append(angle)
        geodesic_errors = np.array(geodesic_errors)
        geodesic_errors_deg = np.rad2deg(geodesic_errors)
        
        # Cluster size distribution
        unique, counts = np.unique(assignments, return_counts=True)
        cluster_sizes = np.zeros(self.codebook_size)
        cluster_sizes[unique] = counts
        
        stats = {
            'codebook_size': self.codebook_size,
            'n_samples': len(rotations),
            # Quaternion distance stats
            'quat_angular_error_mean_deg': float(angular_errors_deg.mean()),
            'quat_angular_error_std_deg': float(angular_errors_deg.std()),
            'quat_angular_error_median_deg': float(np.median(angular_errors_deg)),
            'quat_angular_error_max_deg': float(angular_errors_deg.max()),
            'quat_angular_error_95th_deg': float(np.percentile(angular_errors_deg, 95)),
            # SO(3) geodesic distance stats
            'geodesic_error_mean_deg': float(geodesic_errors_deg.mean()),
            'geodesic_error_std_deg': float(geodesic_errors_deg.std()),
            'geodesic_error_median_deg': float(np.median(geodesic_errors_deg)),
            'geodesic_error_max_deg': float(geodesic_errors_deg.max()),
            'geodesic_error_95th_deg': float(np.percentile(geodesic_errors_deg, 95)),
            # Frobenius norm stats
            'frobenius_error_mean': float(frobenius_errors.mean()),
            'frobenius_error_std': float(frobenius_errors.std()),
            # Cluster distribution
            'cluster_size_mean': float(cluster_sizes.mean()),
            'cluster_size_std': float(cluster_sizes.std()),
            'cluster_size_min': int(cluster_sizes.min()),
            'cluster_size_max': int(cluster_sizes.max()),
            'empty_clusters': int((cluster_sizes == 0).sum()),
        }
        
        if verbose:
            print("\n" + "="*70)
            print("ROTATION CODEBOOK STATISTICS")
            print("="*70)
            print(f"Codebook size: {stats['codebook_size']}")
            print(f"Number of samples: {stats['n_samples']}")
            print()
            print("Quaternion Angular Errors (degrees):")
            print(f"  Mean:   {stats['quat_angular_error_mean_deg']:.4f}°")
            print(f"  Std:    {stats['quat_angular_error_std_deg']:.4f}°")
            print(f"  Median: {stats['quat_angular_error_median_deg']:.4f}°")
            print(f"  95th:   {stats['quat_angular_error_95th_deg']:.4f}°")
            print(f"  Max:    {stats['quat_angular_error_max_deg']:.4f}°")
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
            print()
            print("Cluster Distribution:")
            print(f"  Mean size: {stats['cluster_size_mean']:.1f}")
            print(f"  Std:       {stats['cluster_size_std']:.1f}")
            print(f"  Min:       {stats['cluster_size_min']}")
            print(f"  Max:       {stats['cluster_size_max']}")
            print(f"  Empty:     {stats['empty_clusters']}")
            print("="*70 + "\n")
        
        return stats
    
    def encode(self, rotations: np.ndarray) -> np.ndarray:
        """
        Encode rotation matrices to codebook indices.
        
        Args:
            rotations: (N, 3, 3) rotation matrices
            
        Returns:
            indices: (N,) codebook indices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        quats = self._rotation_to_quat(rotations)
        distances = self._quaternion_distance(quats, self.centroids)
        indices = np.argmin(distances, axis=1)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode codebook indices to rotation matrices.
        
        Args:
            indices: (N,) codebook indices
            
        Returns:
            rotations: (N, 3, 3) rotation matrices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        quats = self.centroids[indices]
        rotations = self._quat_to_rotation(quats)
        
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
            'codebook_size': self.codebook_size,
            'centroids': self.centroids,
            'is_fitted': self.is_fitted,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Codebook saved to: {save_path}")
    
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
        
        self.codebook_size = data['codebook_size']
        self.centroids = data['centroids']
        self.is_fitted = data['is_fitted']
        
        print(f"Codebook loaded from: {load_path}")
        print(f"  Codebook size: {self.codebook_size}")


def build_rotation_codebook_from_cache(
    cache_path: str,
    codebook_size: int,
    output_path: str,
    max_iter: int = 100,
    random_seed: int = 42,
) -> RotationCodebook:
    """
    Build rotation codebook from a dataset cache file.
    
    Args:
        cache_path: Path to .npz cache file containing 'rotations'
        codebook_size: Number of cluster centers
        output_path: Path to save the codebook
        max_iter: Maximum k-means iterations
        random_seed: Random seed
        
    Returns:
        codebook: Fitted RotationCodebook instance
    """
    print(f"Loading rotations from: {cache_path}")
    data = np.load(cache_path)
    
    if 'rotations' not in data:
        raise KeyError(f"Cache file does not contain 'rotations' key. Available keys: {list(data.keys())}")
    
    rotations = data['rotations']
    print(f"Loaded {len(rotations)} rotation matrices")
    
    # Create and fit codebook
    codebook = RotationCodebook(codebook_size)
    stats = codebook.fit(rotations, max_iter=max_iter, random_seed=random_seed, verbose=True)
    
    # Save codebook
    codebook.save(output_path)
    
    # Save statistics
    stats_path = Path(output_path).with_suffix('.stats.txt')
    with open(stats_path, 'w') as f:
        f.write("ROTATION CODEBOOK STATISTICS\n")
        f.write("="*70 + "\n\n")
        for key, value in stats.items():
            if key != 'training_history':
                f.write(f"{key}: {value}\n")
    
    print(f"Statistics saved to: {stats_path}")
    
    return codebook


class TranslationCodebook:
    """
    Linear quantization for translation vectors.
    
    Creates uniform grid in 3D space covering 95% of the data range.
    Each dimension is independently discretized into bins, then combined
    to create codebook entries.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of codebook entries (should be a perfect cube for balanced discretization)
        """
        self.codebook_size = codebook_size
        # Compute bins per dimension (approximate cube root)
        self.bins_per_dim = max(2, int(np.round(codebook_size ** (1/3))))
        # Actual codebook size might differ slightly
        self.actual_codebook_size = self.bins_per_dim ** 3
        
        self.min_vals = None  # (3,) minimum values per dimension (2.5th percentile)
        self.max_vals = None  # (3,) maximum values per dimension (97.5th percentile)
        self.codebook = None  # (codebook_size, 3) codebook entries
        self.is_fitted = False
        
    def fit(self, translations: np.ndarray, percentile: float = 95.0, verbose: bool = True) -> Dict:
        """
        Fit translation codebook by creating uniform grid covering specified percentile.
        
        Args:
            translations: (N, 3) translation vectors
            percentile: Percentage of data to cover (default: 95%)
            verbose: Print progress
            
        Returns:
            stats: Dictionary with statistics
        """
        N = translations.shape[0]
        if verbose:
            print(f"Fitting translation codebook with {N} samples, {self.actual_codebook_size} bins...")
        
        # Compute percentile bounds per dimension
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        self.min_vals = np.percentile(translations, lower_percentile, axis=0)  # (3,)
        self.max_vals = np.percentile(translations, upper_percentile, axis=0)  # (3,)
        
        if verbose:
            print(f"Translation ranges (covering {percentile}% of data):")
            for i, dim in enumerate(['X', 'Y', 'Z']):
                print(f"  {dim}: [{self.min_vals[i]:.4f}, {self.max_vals[i]:.4f}]")
        
        # Create uniform grid for each dimension
        x_bins = np.linspace(self.min_vals[0], self.max_vals[0], self.bins_per_dim)
        y_bins = np.linspace(self.min_vals[1], self.max_vals[1], self.bins_per_dim)
        z_bins = np.linspace(self.min_vals[2], self.max_vals[2], self.bins_per_dim)
        
        # Create 3D grid
        xx, yy, zz = np.meshgrid(x_bins, y_bins, z_bins, indexing='ij')
        self.codebook = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)  # (K, 3)
        
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
        
        # Encode and decode
        indices = self.encode(translations)
        reconstructed = self.decode(indices)
        
        # Compute errors
        errors = translations - reconstructed  # (N, 3)
        l2_errors = np.linalg.norm(errors, axis=1)  # (N,)
        l1_errors = np.abs(errors).sum(axis=1)  # (N,)
        linf_errors = np.abs(errors).max(axis=1)  # (N,)
        
        # Per-dimension errors
        per_dim_errors = np.abs(errors)  # (N, 3)
        
        # Check coverage (percentage of data within bounds)
        in_bounds = np.all((translations >= self.min_vals) & (translations <= self.max_vals), axis=1)
        coverage = in_bounds.mean() * 100
        
        # Compute codebook utilization
        unique_indices = np.unique(indices)
        utilization = len(unique_indices) / self.actual_codebook_size * 100
        
        stats = {
            'codebook_size': self.actual_codebook_size,
            'bins_per_dim': self.bins_per_dim,
            'n_samples': len(translations),
            'coverage_percent': float(coverage),
            'codebook_utilization_percent': float(utilization),
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
            print(f"Codebook size: {stats['codebook_size']} ({self.bins_per_dim}^3)")
            print(f"Number of samples: {stats['n_samples']}")
            print(f"Coverage: {stats['coverage_percent']:.2f}%")
            print(f"Codebook utilization: {stats['codebook_utilization_percent']:.2f}%")
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
    
    def encode(self, translations: np.ndarray) -> np.ndarray:
        """
        Encode translation vectors to codebook indices.
        
        Args:
            translations: (N, 3) translation vectors
            
        Returns:
            indices: (N,) codebook indices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        N = translations.shape[0]
        
        # Clip to bounds
        translations_clipped = np.clip(translations, self.min_vals, self.max_vals)
        
        # Compute distances to all codebook entries
        # Broadcasting: (N, 1, 3) - (1, K, 3) = (N, K, 3)
        distances = np.linalg.norm(
            translations_clipped[:, np.newaxis, :] - self.codebook[np.newaxis, :, :],
            axis=2
        )  # (N, K)
        
        # Find nearest codebook entry
        indices = np.argmin(distances, axis=1)  # (N,)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode codebook indices to translation vectors.
        
        Args:
            indices: (N,) codebook indices
            
        Returns:
            translations: (N, 3) translation vectors
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        translations = self.codebook[indices]
        return translations
    
    def save(self, save_path: Union[str, Path]):
        """Save codebook to file."""
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'codebook_size': self.codebook_size,
            'actual_codebook_size': self.actual_codebook_size,
            'bins_per_dim': self.bins_per_dim,
            'min_vals': self.min_vals,
            'max_vals': self.max_vals,
            'codebook': self.codebook,
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
        
        self.codebook_size = data['codebook_size']
        self.actual_codebook_size = data['actual_codebook_size']
        self.bins_per_dim = data['bins_per_dim']
        self.min_vals = data['min_vals']
        self.max_vals = data['max_vals']
        self.codebook = data['codebook']
        self.is_fitted = data['is_fitted']
        
        print(f"Translation codebook loaded from: {load_path}")
        print(f"  Codebook size: {self.actual_codebook_size} ({self.bins_per_dim}^3)")


class ScaleCodebook:
    """
    Logarithmic quantization for scale factors.
    
    Applies log transform to scales, then creates uniform grid in log-space
    covering 95% of the data range. Each dimension is independently discretized.
    """
    
    def __init__(self, codebook_size: int):
        """
        Args:
            codebook_size: Number of codebook entries
        """
        self.codebook_size = codebook_size
        # Compute bins per dimension (approximate cube root)
        self.bins_per_dim = max(2, int(np.round(codebook_size ** (1/3))))
        # Actual codebook size might differ slightly
        self.actual_codebook_size = self.bins_per_dim ** 3
        
        self.log_min_vals = None  # (3,) minimum log-scale values (2.5th percentile)
        self.log_max_vals = None  # (3,) maximum log-scale values (97.5th percentile)
        self.log_codebook = None  # (codebook_size, 3) codebook entries in log-space
        self.is_fitted = False
        
    def fit(self, scales: np.ndarray, percentile: float = 95.0, eps: float = 1e-8, verbose: bool = True) -> Dict:
        """
        Fit scale codebook by creating uniform grid in log-space covering specified percentile.
        
        Args:
            scales: (N, 3) scale factors (positive values)
            percentile: Percentage of data to cover (default: 95%)
            eps: Small epsilon to avoid log(0)
            verbose: Print progress
            
        Returns:
            stats: Dictionary with statistics
        """
        N = scales.shape[0]
        if verbose:
            print(f"Fitting scale codebook with {N} samples, {self.actual_codebook_size} bins...")
        
        # Take absolute value and add eps to avoid log(0)
        scales_abs = np.abs(scales) + eps
        
        # Apply log transform
        log_scales = np.log(scales_abs)
        
        # Compute percentile bounds per dimension in log-space
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        
        self.log_min_vals = np.percentile(log_scales, lower_percentile, axis=0)  # (3,)
        self.log_max_vals = np.percentile(log_scales, upper_percentile, axis=0)  # (3,)
        
        if verbose:
            print(f"Scale ranges in log-space (covering {percentile}% of data):")
            for i, dim in enumerate(['X', 'Y', 'Z']):
                min_scale = np.exp(self.log_min_vals[i])
                max_scale = np.exp(self.log_max_vals[i])
                print(f"  {dim}: log=[{self.log_min_vals[i]:.4f}, {self.log_max_vals[i]:.4f}] -> "
                      f"scale=[{min_scale:.4f}, {max_scale:.4f}]")
        
        # Create uniform grid in log-space for each dimension
        x_bins = np.linspace(self.log_min_vals[0], self.log_max_vals[0], self.bins_per_dim)
        y_bins = np.linspace(self.log_min_vals[1], self.log_max_vals[1], self.bins_per_dim)
        z_bins = np.linspace(self.log_min_vals[2], self.log_max_vals[2], self.bins_per_dim)
        
        # Create 3D grid in log-space
        xx, yy, zz = np.meshgrid(x_bins, y_bins, z_bins, indexing='ij')
        self.log_codebook = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)  # (K, 3)
        
        self.is_fitted = True
        
        # Compute statistics
        stats = self._compute_statistics(scales, eps=eps, verbose=verbose)
        
        return stats
    
    def _compute_statistics(self, scales: np.ndarray, eps: float = 1e-8, verbose: bool = True) -> Dict:
        """
        Compute error statistics for the codebook.
        
        Args:
            scales: (N, 3) scale factors
            eps: Small epsilon to avoid log(0)
            verbose: Print statistics
            
        Returns:
            stats: Dictionary with error statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        # Encode and decode
        indices = self.encode(scales, eps=eps)
        reconstructed = self.decode(indices)
        
        # Compute errors in original scale space
        errors = scales - reconstructed  # (N, 3)
        l2_errors = np.linalg.norm(errors, axis=1)  # (N,)
        relative_errors = np.abs(errors) / (np.abs(scales) + eps)  # (N, 3)
        relative_l2_errors = np.linalg.norm(relative_errors, axis=1)  # (N,)
        
        # Compute errors in log-space
        scales_abs = np.abs(scales) + eps
        reconstructed_abs = np.abs(reconstructed) + eps
        log_errors = np.abs(np.log(scales_abs) - np.log(reconstructed_abs))  # (N, 3)
        log_l2_errors = np.linalg.norm(log_errors, axis=1)  # (N,)
        
        # Check coverage (percentage of data within bounds in log-space)
        log_scales = np.log(scales_abs)
        in_bounds = np.all((log_scales >= self.log_min_vals) & (log_scales <= self.log_max_vals), axis=1)
        coverage = in_bounds.mean() * 100
        
        # Compute codebook utilization
        unique_indices = np.unique(indices)
        utilization = len(unique_indices) / self.actual_codebook_size * 100
        
        stats = {
            'codebook_size': self.actual_codebook_size,
            'bins_per_dim': self.bins_per_dim,
            'n_samples': len(scales),
            'coverage_percent': float(coverage),
            'codebook_utilization_percent': float(utilization),
            # L2 error stats in original space
            'l2_error_mean': float(l2_errors.mean()),
            'l2_error_std': float(l2_errors.std()),
            'l2_error_median': float(np.median(l2_errors)),
            'l2_error_max': float(l2_errors.max()),
            'l2_error_95th': float(np.percentile(l2_errors, 95)),
            # Relative error stats
            'relative_error_mean': float(relative_errors.mean()),
            'relative_error_median': float(np.median(relative_errors)),
            'relative_error_max': float(relative_errors.max()),
            'relative_l2_error_mean': float(relative_l2_errors.mean()),
            # Log-space error stats
            'log_l2_error_mean': float(log_l2_errors.mean()),
            'log_l2_error_median': float(np.median(log_l2_errors)),
            'log_l2_error_max': float(log_l2_errors.max()),
            # Per-dimension stats
            'x_error_mean': float(np.abs(errors[:, 0]).mean()),
            'y_error_mean': float(np.abs(errors[:, 1]).mean()),
            'z_error_mean': float(np.abs(errors[:, 2]).mean()),
        }
        
        if verbose:
            print("\n" + "="*70)
            print("SCALE CODEBOOK STATISTICS")
            print("="*70)
            print(f"Codebook size: {stats['codebook_size']} ({self.bins_per_dim}^3)")
            print(f"Number of samples: {stats['n_samples']}")
            print(f"Coverage: {stats['coverage_percent']:.2f}%")
            print(f"Codebook utilization: {stats['codebook_utilization_percent']:.2f}%")
            print()
            print("L2 Reconstruction Errors (original space):")
            print(f"  Mean:   {stats['l2_error_mean']:.6f}")
            print(f"  Std:    {stats['l2_error_std']:.6f}")
            print(f"  Median: {stats['l2_error_median']:.6f}")
            print(f"  95th:   {stats['l2_error_95th']:.6f}")
            print(f"  Max:    {stats['l2_error_max']:.6f}")
            print()
            print("Relative Errors:")
            print(f"  Mean:   {stats['relative_error_mean']:.6f}")
            print(f"  Median: {stats['relative_error_median']:.6f}")
            print(f"  Max:    {stats['relative_error_max']:.6f}")
            print()
            print("Log-space L2 Errors:")
            print(f"  Mean:   {stats['log_l2_error_mean']:.6f}")
            print(f"  Median: {stats['log_l2_error_median']:.6f}")
            print(f"  Max:    {stats['log_l2_error_max']:.6f}")
            print("="*70 + "\n")
        
        return stats
    
    def encode(self, scales: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Encode scale factors to codebook indices.
        
        Args:
            scales: (N, 3) scale factors
            eps: Small epsilon to avoid log(0)
            
        Returns:
            indices: (N,) codebook indices
        """
        if not self.is_fitted:
            raise RuntimeError("Codebook not fitted yet!")
        
        N = scales.shape[0]
        
        # Apply log transform
        scales_abs = np.abs(scales) + eps
        log_scales = np.log(scales_abs)
        
        # Clip to bounds in log-space
        log_scales_clipped = np.clip(log_scales, self.log_min_vals, self.log_max_vals)
        
        # Compute distances to all codebook entries in log-space
        distances = np.linalg.norm(
            log_scales_clipped[:, np.newaxis, :] - self.log_codebook[np.newaxis, :, :],
            axis=2
        )  # (N, K)
        
        # Find nearest codebook entry
        indices = np.argmin(distances, axis=1)  # (N,)
        
        return indices
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """
        Decode codebook indices to scale factors.
        
        Args:
            indices: (N,) codebook indices
            
        Returns:
            scales: (N, 3) scale factors
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
            'actual_codebook_size': self.actual_codebook_size,
            'bins_per_dim': self.bins_per_dim,
            'log_min_vals': self.log_min_vals,
            'log_max_vals': self.log_max_vals,
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
        self.actual_codebook_size = data['actual_codebook_size']
        self.bins_per_dim = data['bins_per_dim']
        self.log_min_vals = data['log_min_vals']
        self.log_max_vals = data['log_max_vals']
        self.log_codebook = data['log_codebook']
        self.is_fitted = data['is_fitted']
        
        print(f"Scale codebook loaded from: {load_path}")
        print(f"  Codebook size: {self.actual_codebook_size} ({self.bins_per_dim}^3)")


def build_translation_codebook_from_cache(
    cache_path: str,
    codebook_size: int,
    output_path: str,
    percentile: float = 95.0,
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
    print(f"Loaded {len(translations)} translation vectors")
    
    # Create and fit codebook
    codebook = TranslationCodebook(codebook_size)
    stats = codebook.fit(translations, percentile=percentile, verbose=True)
    
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
    print(f"Loaded {len(scales)} scale vectors")
    
    # Create and fit codebook
    codebook = ScaleCodebook(codebook_size)
    stats = codebook.fit(scales, percentile=percentile, eps=eps, verbose=True)
    
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
        )
    elif args.type == 'scale':
        build_scale_codebook_from_cache(
            cache_path=args.cache_path,
            codebook_size=args.codebook_size,
            output_path=args.output_path,
            percentile=args.percentile,
        )

