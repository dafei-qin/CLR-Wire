#!/usr/bin/env python3
"""
Plot histograms of transformation parameters (scaling, rotation, translation) 
for each surface type from the aggregated dataset.

Usage:
$ python plot_surface_transformations.py data.pkl
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict

# Surface type mapping (from aggregate_surface_dataset.py)
CLASS_MAPPING = {
    0: "plane",
    1: "cylinder", 
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "bspline_surface",
    6: "bezier_surface",
}

def load_dataset(pkl_path: Path) -> Dict:
    """Load the pickled dataset."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    return data

def cartesian_to_spherical(xyz):
    """Convert 3D cartesian coordinates to spherical coordinates (theta, phi)."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    
    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    # Handle zero vectors
    r = np.where(r == 0, 1e-10, r)
    
    # Normalize to unit sphere
    x_norm, y_norm, z_norm = x/r, y/r, z/r
    
    # Calculate spherical coordinates
    theta = np.arccos(np.clip(z_norm, -1, 1))  # polar angle [0, π]
    phi = np.arctan2(y_norm, x_norm)  # azimuthal angle [-π, π]
    
    return theta, phi

def plot_spherical_density(ax, rotation_data, title):
    """Plot spherical density map for rotation vectors."""
    # Convert rotation vectors to spherical coordinates
    theta, phi = cartesian_to_spherical(rotation_data)
    
    # Create 2D histogram in spherical coordinates
    n_bins = 30
    theta_edges = np.linspace(0, np.pi, n_bins)
    phi_edges = np.linspace(-np.pi, np.pi, n_bins)
    
    hist, theta_bins, phi_bins = np.histogram2d(theta, phi, bins=[theta_edges, phi_edges])
    
    # Create meshgrid for plotting
    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
    phi_centers = (phi_bins[:-1] + phi_bins[1:]) / 2
    PHI, THETA = np.meshgrid(phi_centers, theta_centers)
    
    # Plot using imshow (with proper orientation)
    # Note: origin='lower' to match spherical coordinate convention
    im = ax.imshow(hist, extent=[-np.pi, np.pi, 0, np.pi], 
                   origin='lower', aspect='auto', cmap='viridis',
                   norm=LogNorm(vmin=max(1, hist.min()), vmax=hist.max()) if hist.max() > 0 else None)
    
    # Customize plot
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Azimuthal angle φ (radians)')
    ax.set_ylabel('Polar angle θ (radians)')
    
    # Add ticks and labels
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    # Add statistics
    r_magnitudes = np.linalg.norm(rotation_data, axis=1)
    mean_magnitude = np.mean(r_magnitudes)
    std_magnitude = np.std(r_magnitudes)
    
    stats_text = f'|r| μ={mean_magnitude:.3f}\n|r| σ={std_magnitude:.3f}\nN={len(rotation_data)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def print_percentiles(values, axis_name, surface_type):
    """Print percentiles for a given axis of transformation parameters."""
    percentiles = np.percentile(values, [5, 95])
    print(f"\n{surface_type} - {axis_name} percentiles:")
    print(f"5th percentile: {percentiles[0]:.3f}")
    print(f"95th percentile: {percentiles[1]:.3f}")

def plot_transformation_histograms(data: Dict, output_dir: Path = Path("."), interactive: bool = False, show_and_save: bool = False):
    """Create histogram plots for each surface type."""
    
    # Ensure output directory exists (needed for saving files)
    if not interactive or show_and_save:
        output_dir.mkdir(exist_ok=True)
    
    # Get unique class labels in the dataset
    unique_labels = np.unique(data["class_label"])
    
    for class_id in unique_labels:
        if class_id not in CLASS_MAPPING:
            print(f"Warning: Unknown class ID {class_id}, skipping...")
            continue
            
        surface_type = CLASS_MAPPING[class_id]
        print(f"\nProcessing {surface_type} (class {class_id})...")
        
        # Filter data for this surface type
        mask = data["class_label"] == class_id
        scaling = data["scaling"][mask]  # (N, 3)
        rotation = data["rotation"][mask]  # (N, 3) 
        translation = data["translation"][mask]  # (N, 3)
        
        num_samples = len(scaling)
        if num_samples == 0:
            print(f"No samples found for {surface_type}, skipping...")
            continue
            
        print(f"  Found {num_samples} samples")
        
        # Print percentiles for scaling and translation
        for axis_name, values in zip(['X', 'Y', 'Z'], scaling.T):
            print_percentiles(values, f"Scaling {axis_name}", surface_type)
            
        for axis_name, values in zip(['X', 'Y', 'Z'], translation.T):
            print_percentiles(values, f"Translation {axis_name}", surface_type)
        
        # Create custom grid: 2 rows, with top row having 3 cols, bottom row having 1 large col + 2 regular cols
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f'{surface_type.replace("_", " ").title()} - Transformation Parameters\n'
                    f'({num_samples} samples)', fontsize=16, fontweight='bold')
        
        # Define grid layout
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])
        
        # Scaling histograms (top row)
        scaling_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        
        # Rotation spherical density (bottom left, takes 2 columns)
        rotation_ax = fig.add_subplot(gs[1, :2])
        
        # Translation histograms (bottom right)
        translation_axes = [fig.add_subplot(gs[1, i+2]) for i in range(2)]
        
        # Plot scaling histograms
        for col, axis_name in enumerate(['X', 'Y', 'Z']):
            ax = scaling_axes[col]
            values = scaling[:, col]
            
            # Create histogram
            n_bins = min(50, max(10, int(np.sqrt(num_samples))))
            counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                          color='C0', edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_title(f'Scaling {axis_name}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot rotation spherical density
        plot_spherical_density(rotation_ax, rotation, 'Rotation Direction Distribution')
        
        # Plot translation histograms (X and Y only due to space)
        for col, axis_name in enumerate(['X', 'Y']):
            ax = translation_axes[col]
            values = translation[:, col]
            
            # Create histogram
            n_bins = min(50, max(10, int(np.sqrt(num_samples))))
            counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                          color='C2', edgecolor='black', linewidth=0.5)
            
            # Customize plot
            ax.set_title(f'Translation {axis_name}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nmin={min_val:.3f}\nmax={max_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add translation Z statistics as text (since we don't have space for 3rd histogram)
        z_values = translation[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        
        fig.text(0.98, 0.25, f'Translation Z:\nμ={z_mean:.3f}\nσ={z_std:.3f}\nmin={z_min:.3f}\nmax={z_max:.3f}', 
                transform=fig.transFigure, ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        if interactive or show_and_save:
            # Show plot interactively and wait for user to close
            print(f"  Displaying {surface_type} plot. Close the window to continue...")
            plt.show()
        
        if not interactive or show_and_save:
            # Save to file
            output_file = output_dir / f'{surface_type}_transformations.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to {output_file}")
        
        if not (interactive or show_and_save):
            plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Plot transformation parameter histograms for each surface type")
    parser.add_argument("dataset", type=Path, 
                       help="Path to the pickled dataset file (.pkl)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("."),
                       help="Output directory for PNG files (default: current directory)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Show plots interactively instead of saving to files")
    parser.add_argument("--show-and-save", action="store_true",
                       help="Both show plots interactively AND save to files")
    
    args = parser.parse_args()
    
    if not args.dataset.exists():
        print(f"Error: Dataset file {args.dataset} does not exist!")
        return
    
    if not args.dataset.suffix.lower() == ".pkl":
        print(f"Warning: Expected .pkl file, got {args.dataset.suffix}")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    
    # Print dataset info
    print(f"Dataset contains {len(data['class_label'])} samples")
    unique_labels, counts = np.unique(data["class_label"], return_counts=True)
    print("Surface type distribution:")
    for label, count in zip(unique_labels, counts):
        surface_name = CLASS_MAPPING.get(label, f"unknown_{label}")
        print(f"  {surface_name}: {count} samples")
    
    # Handle conflicting options
    if args.interactive and args.show_and_save:
        print("Warning: Both --interactive and --show-and-save specified. Using --show-and-save.")
        args.interactive = False
    
    # Generate plots
    if args.interactive or args.show_and_save:
        print(f"\nDisplaying plots interactively. Close each window to proceed to the next.")
        print("Press Ctrl+C to exit early if needed.\n")
    
    plot_transformation_histograms(data, args.output_dir, 
                                 interactive=args.interactive, 
                                 show_and_save=args.show_and_save)
    
    if args.interactive:
        print("\nAll plots have been displayed!")
    elif args.show_and_save:
        print("\nAll plots have been displayed and saved!")
    else:
        print("Done!")

if __name__ == "__main__":
    main() 