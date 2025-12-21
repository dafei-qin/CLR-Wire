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


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build rotation codebook using spherical k-means')
    parser.add_argument('--cache_path', type=str, required=True,
                       help='Path to .npz cache file with rotations')
    parser.add_argument('--codebook_size', type=int, required=True,
                       help='Number of codebook entries (clusters)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save the codebook (.pkl)')
    parser.add_argument('--max_iter', type=int, default=100,
                       help='Maximum k-means iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    build_rotation_codebook_from_cache(
        cache_path=args.cache_path,
        codebook_size=args.codebook_size,
        output_path=args.output_path,
        max_iter=args.max_iter,
        random_seed=args.seed,
    )

