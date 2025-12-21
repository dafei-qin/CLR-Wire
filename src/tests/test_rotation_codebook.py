"""
Test and visualize rotation codebook on quaternion hypersphere.

This script provides visualization tools for:
1. Codebook centroids distribution on the unit quaternion sphere
2. Dataset rotations distribution on the unit quaternion sphere
3. Quantization errors and cluster assignments
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from typing import Optional, Tuple

from src.utils.rts_tools import RotationCodebook


def visualize_quaternions_3d(
    quats: np.ndarray,
    labels: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    title: str = "Quaternion Distribution",
    save_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    projection_method: str = 'xyz',
):
    """
    Visualize quaternions in 3D space using projection.
    
    Args:
        quats: (N, 4) quaternions in [w, x, y, z] format
        labels: (N,) cluster assignments for color coding
        centroids: (K, 4) centroid quaternions
        title: Plot title
        save_path: Path to save figure
        sample_size: Number of points to sample for visualization
        projection_method: 'xyz' (ignore w) or 'stereographic'
    """
    if sample_size is not None and len(quats) > sample_size:
        indices = np.random.choice(len(quats), sample_size, replace=False)
        quats_vis = quats[indices]
        labels_vis = labels[indices] if labels is not None else None
    else:
        quats_vis = quats
        labels_vis = labels
    
    fig = plt.figure(figsize=(15, 5))
    
    # === Subplot 1: xyz projection ===
    ax1 = fig.add_subplot(131, projection='3d')
    
    if projection_method == 'xyz':
        x, y, z = quats_vis[:, 1], quats_vis[:, 2], quats_vis[:, 3]
    elif projection_method == 'stereographic':
        # Stereographic projection from 4D sphere to 3D space
        w, x, y, z = quats_vis[:, 0], quats_vis[:, 1], quats_vis[:, 2], quats_vis[:, 3]
        denom = 1 + w + 1e-10
        x, y, z = x / denom, y / denom, z / denom
    
    if labels_vis is not None:
        scatter = ax1.scatter(x, y, z, c=labels_vis, cmap='tab20', s=1, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
    else:
        ax1.scatter(x, y, z, c='blue', s=1, alpha=0.5)
    
    if centroids is not None:
        if projection_method == 'xyz':
            cx, cy, cz = centroids[:, 1], centroids[:, 2], centroids[:, 3]
        elif projection_method == 'stereographic':
            cw, cx, cy, cz = centroids[:, 0], centroids[:, 1], centroids[:, 2], centroids[:, 3]
            denom = 1 + cw + 1e-10
            cx, cy, cz = cx / denom, cy / denom, cz / denom
        
        ax1.scatter(cx, cy, cz, c='red', s=100, marker='*', 
                   edgecolors='black', linewidths=1, label='Centroids')
        ax1.legend()
    
    ax1.set_xlabel('X (qx)')
    ax1.set_ylabel('Y (qy)')
    ax1.set_zlabel('Z (qz)')
    ax1.set_title(f'{projection_method.upper()} Projection')
    
    # === Subplot 2: 2D projection (xy plane) ===
    ax2 = fig.add_subplot(132)
    
    if labels_vis is not None:
        scatter = ax2.scatter(x, y, c=labels_vis, cmap='tab20', s=1, alpha=0.6)
    else:
        ax2.scatter(x, y, c='blue', s=1, alpha=0.5)
    
    if centroids is not None:
        ax2.scatter(cx, cy, c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, label='Centroids', zorder=10)
        ax2.legend()
    
    ax2.set_xlabel('X (qx)')
    ax2.set_ylabel('Y (qy)')
    ax2.set_title('XY Projection')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: 2D projection (xz plane) ===
    ax3 = fig.add_subplot(133)
    
    if labels_vis is not None:
        scatter = ax3.scatter(x, z, c=labels_vis, cmap='tab20', s=1, alpha=0.6)
    else:
        ax3.scatter(x, z, c='blue', s=1, alpha=0.5)
    
    if centroids is not None:
        ax3.scatter(cx, cz, c='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2, label='Centroids', zorder=10)
        ax3.legend()
    
    ax3.set_xlabel('X (qx)')
    ax3.set_ylabel('Z (qz)')
    ax3.set_title('XZ Projection')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_rotation_axes(
    rotations: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Rotation Axes Distribution",
    save_path: Optional[str] = None,
    sample_size: Optional[int] = None,
):
    """
    Visualize rotation axes on unit sphere.
    Each rotation can be represented as axis-angle: rotate by theta around axis n.
    
    Args:
        rotations: (N, 3, 3) rotation matrices
        labels: (N,) cluster assignments
        title: Plot title
        save_path: Path to save figure
        sample_size: Number of samples to visualize
    """
    from scipy.spatial.transform import Rotation as R
    
    if sample_size is not None and len(rotations) > sample_size:
        indices = np.random.choice(len(rotations), sample_size, replace=False)
        rotations_vis = rotations[indices]
        labels_vis = labels[indices] if labels is not None else None
    else:
        rotations_vis = rotations
        labels_vis = labels
    
    # Convert to axis-angle representation
    scipy_rots = R.from_matrix(rotations_vis)
    rotvecs = scipy_rots.as_rotvec()  # (N, 3), axis * angle
    
    # Extract axes and angles
    angles = np.linalg.norm(rotvecs, axis=1)
    axes = np.zeros_like(rotvecs)
    nonzero_mask = angles > 1e-6
    axes[nonzero_mask] = rotvecs[nonzero_mask] / angles[nonzero_mask, None]
    
    # For identity rotations, set arbitrary axis
    axes[~nonzero_mask] = [1, 0, 0]
    
    fig = plt.figure(figsize=(16, 5))
    
    # === Subplot 1: 3D axes on unit sphere ===
    ax1 = fig.add_subplot(131, projection='3d')
    
    if labels_vis is not None:
        scatter = ax1.scatter(axes[:, 0], axes[:, 1], axes[:, 2], 
                            c=labels_vis, cmap='tab20', s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')
    else:
        ax1.scatter(axes[:, 0], axes[:, 1], axes[:, 2], 
                   c='blue', s=20, alpha=0.6)
    
    # Draw unit sphere wireframe
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Rotation Axes on Unit Sphere')
    ax1.set_box_aspect([1,1,1])
    
    # === Subplot 2: Angle distribution ===
    ax2 = fig.add_subplot(132)
    
    angles_deg = np.rad2deg(angles)
    ax2.hist(angles_deg, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(angles_deg.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {angles_deg.mean():.2f}°')
    ax2.axvline(np.median(angles_deg), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(angles_deg):.2f}°')
    
    ax2.set_xlabel('Rotation Angle (degrees)')
    ax2.set_ylabel('Count')
    ax2.set_title('Rotation Angle Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: 2D axis projection ===
    ax3 = fig.add_subplot(133)
    
    if labels_vis is not None:
        scatter = ax3.scatter(axes[:, 0], axes[:, 1], 
                            c=labels_vis, cmap='tab20', s=20, alpha=0.6)
    else:
        ax3.scatter(axes[:, 0], axes[:, 1], c='blue', s=20, alpha=0.6)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1)
    ax3.add_patch(circle)
    
    ax3.set_xlabel('Axis X')
    ax3.set_ylabel('Axis Y')
    ax3.set_title('Rotation Axes (XY Projection)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_quantization_errors(
    codebook: RotationCodebook,
    rotations: np.ndarray,
    title: str = "Quantization Error Analysis",
    save_path: Optional[str] = None,
):
    """
    Visualize quantization errors and cluster distribution.
    
    Args:
        codebook: Fitted RotationCodebook
        rotations: (N, 3, 3) rotation matrices
        title: Plot title
        save_path: Path to save figure
    """
    from scipy.spatial.transform import Rotation as R
    
    # Convert to quaternions
    quats = codebook._rotation_to_quat(rotations)
    
    # Get assignments and distances
    distances = codebook._quaternion_distance(quats, codebook.centroids)
    assignments = np.argmin(distances, axis=1)
    min_distances = distances[np.arange(len(quats)), assignments]
    
    # Angular errors
    angular_errors_deg = np.rad2deg(min_distances)
    
    # Compute geodesic errors
    reconstructed = codebook.decode(assignments)
    geodesic_errors = []
    for i in range(len(rotations)):
        R_diff = rotations[i].T @ reconstructed[i]
        trace = np.trace(R_diff)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
        geodesic_errors.append(angle)
    geodesic_errors_deg = np.rad2deg(np.array(geodesic_errors))
    
    # Cluster sizes
    unique, counts = np.unique(assignments, return_counts=True)
    cluster_sizes = np.zeros(codebook.codebook_size)
    cluster_sizes[unique] = counts
    
    fig = plt.figure(figsize=(18, 5))
    
    # === Subplot 1: Quaternion angular error distribution ===
    ax1 = fig.add_subplot(131)
    
    ax1.hist(angular_errors_deg, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(angular_errors_deg.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {angular_errors_deg.mean():.3f}°')
    ax1.axvline(np.median(angular_errors_deg), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(angular_errors_deg):.3f}°')
    ax1.axvline(np.percentile(angular_errors_deg, 95), color='orange', linestyle='--', 
               linewidth=2, label=f'95th: {np.percentile(angular_errors_deg, 95):.3f}°')
    
    ax1.set_xlabel('Quaternion Angular Error (degrees)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Quaternion Distance Error Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: SO(3) geodesic error distribution ===
    ax2 = fig.add_subplot(132)
    
    ax2.hist(geodesic_errors_deg, bins=100, alpha=0.7, edgecolor='black', color='coral')
    ax2.axvline(geodesic_errors_deg.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {geodesic_errors_deg.mean():.3f}°')
    ax2.axvline(np.median(geodesic_errors_deg), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(geodesic_errors_deg):.3f}°')
    ax2.axvline(np.percentile(geodesic_errors_deg, 95), color='orange', linestyle='--', 
               linewidth=2, label=f'95th: {np.percentile(geodesic_errors_deg, 95):.3f}°')
    
    ax2.set_xlabel('SO(3) Geodesic Error (degrees)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Rotation Reconstruction Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: Cluster size distribution ===
    ax3 = fig.add_subplot(133)
    
    ax3.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7, edgecolor='black', color='mediumseagreen')
    ax3.axhline(cluster_sizes.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {cluster_sizes.mean():.1f}')
    
    ax3.set_xlabel('Cluster ID', fontsize=11)
    ax3.set_ylabel('Number of Points', fontsize=11)
    ax3.set_title(f'Cluster Size Distribution (K={codebook.codebook_size})', 
                 fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    stats_text = (
        f"Empty clusters: {(cluster_sizes == 0).sum()}\n"
        f"Min size: {cluster_sizes.min():.0f}\n"
        f"Max size: {cluster_sizes.max():.0f}\n"
        f"Std: {cluster_sizes.std():.1f}"
    )
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize rotation codebook and dataset distributions')
    parser.add_argument('--cache_path', type=str, required=True,
                       help='Path to .npz cache file with rotations')
    parser.add_argument('--codebook_path', type=str, required=True,
                       help='Path to rotation codebook (.pkl)')
    parser.add_argument('--output_dir', type=str, default='./visualizations',
                       help='Directory to save visualization figures')
    parser.add_argument('--sample_size', type=int, default=10000,
                       help='Number of points to sample for visualization')
    parser.add_argument('--projection', type=str, default='xyz', choices=['xyz', 'stereographic'],
                       help='Projection method for 4D quaternions to 3D')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading cache and codebook...")
    data = np.load(args.cache_path)
    rotations = data['rotations']
    print(f"Loaded {len(rotations)} rotations from cache")
    
    codebook = RotationCodebook(codebook_size=1)  # Will be overwritten by load
    codebook.load(args.codebook_path)
    print(f"Loaded codebook with {codebook.codebook_size} centroids")
    
    # Convert rotations to quaternions
    print("\nConverting rotations to quaternions...")
    quats = codebook._rotation_to_quat(rotations)
    assignments = codebook.encode(rotations)
    
    # Visualization 1: Quaternion distribution with cluster assignments
    print("\n=== Visualization 1: Quaternion Distribution ===")
    visualize_quaternions_3d(
        quats=quats,
        labels=assignments,
        centroids=codebook.centroids,
        title=f"Quaternion Distribution (N={len(quats)}, K={codebook.codebook_size})",
        save_path=str(output_dir / f"quaternion_distribution_{args.projection}.png"),
        sample_size=args.sample_size,
        projection_method=args.projection,
    )
    
    # Visualization 2: Rotation axes distribution
    print("\n=== Visualization 2: Rotation Axes Distribution ===")
    visualize_rotation_axes(
        rotations=rotations,
        labels=assignments,
        title=f"Rotation Axes Distribution (N={len(rotations)})",
        save_path=str(output_dir / "rotation_axes_distribution.png"),
        sample_size=args.sample_size,
    )
    
    # Visualization 3: Quantization errors
    print("\n=== Visualization 3: Quantization Error Analysis ===")
    visualize_quantization_errors(
        codebook=codebook,
        rotations=rotations,
        title=f"Quantization Error Analysis (K={codebook.codebook_size})",
        save_path=str(output_dir / "quantization_errors.png"),
    )
    
    # Additional: Centroids only visualization
    print("\n=== Visualization 4: Codebook Centroids Only ===")
    visualize_quaternions_3d(
        quats=codebook.centroids,
        labels=None,
        centroids=None,
        title=f"Codebook Centroids (K={codebook.codebook_size})",
        save_path=str(output_dir / f"codebook_centroids_{args.projection}.png"),
        sample_size=None,
        projection_method=args.projection,
    )
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    stats = codebook._compute_statistics(rotations, verbose=False)
    print(f"Dataset size: {len(rotations)}")
    print(f"Codebook size: {codebook.codebook_size}")
    print(f"\nQuaternion Angular Error:")
    print(f"  Mean: {stats['quat_angular_error_mean_deg']:.4f}°")
    print(f"  Median: {stats['quat_angular_error_median_deg']:.4f}°")
    print(f"  95th percentile: {stats['quat_angular_error_95th_deg']:.4f}°")
    print(f"\nSO(3) Geodesic Error:")
    print(f"  Mean: {stats['geodesic_error_mean_deg']:.4f}°")
    print(f"  Median: {stats['geodesic_error_median_deg']:.4f}°")
    print(f"  95th percentile: {stats['geodesic_error_95th_deg']:.4f}°")
    print(f"\nCluster Distribution:")
    print(f"  Mean size: {stats['cluster_size_mean']:.1f}")
    print(f"  Empty clusters: {stats['empty_clusters']}")
    print("="*70)
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()


