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
        print(f"Processing {surface_type} (class {class_id})...")
        
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
        
        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'{surface_type.replace("_", " ").title()} - Transformation Parameters\n'
                    f'({num_samples} samples)', fontsize=16, fontweight='bold')
        
        # Define parameter names and data
        params = [
            ("Scaling", scaling, ["X", "Y", "Z"]),
            ("Rotation", rotation, ["X", "Y", "Z"]), 
            ("Translation", translation, ["X", "Y", "Z"])
        ]
        
        # Plot histograms
        for row, (param_name, param_data, axis_names) in enumerate(params):
            for col, axis_name in enumerate(axis_names):
                ax = axes[row, col]
                values = param_data[:, col]
                
                # Create histogram
                n_bins = min(50, max(10, int(np.sqrt(num_samples))))
                counts, bins, patches = ax.hist(values, bins=n_bins, alpha=0.7, 
                                              color=f'C{row}', edgecolor='black', linewidth=0.5)
                
                # Customize plot
                ax.set_title(f'{param_name} {axis_name}', fontweight='bold')
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