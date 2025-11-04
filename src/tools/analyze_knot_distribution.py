#!/usr/bin/env python3
"""
Analyze the distribution of u_knots_list and v_knots_list lengths from B-spline dataset.

This script loads the B-spline dataset and generates visualizations of the knot vector
length distributions, saving them to the assets/ directory.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter

from src.dataset.dataset_bspline import dataset_bspline


def analyze_knot_distributions(data_path, output_dir='assets'):
    """
    Analyze and visualize the distribution of knot vector lengths.
    
    Args:
        data_path (str): Path to the directory containing B-spline .npy files
        output_dir (str): Directory to save output figures
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = dataset_bspline(data_path=data_path, replica=1)
    print(f"Found {len(dataset)} B-spline surfaces")
    
    # Collect lengths
    u_knots_lengths = []
    v_knots_lengths = []
    u_degree_list = []
    v_degree_list = []
    num_poles_u_list = []
    num_poles_v_list = []
    
    # Track dropped surfaces
    total_surfaces = len(dataset)
    dropped_high_degree = 0
    dropped_long_knots = 0
    dropped_both = 0
    
    print("Analyzing knot vector lengths...")
    for idx in tqdm(range(len(dataset))):
        try:
            data_path_file = dataset.data_names[idx]
            u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v, \
                is_u_periodic, is_v_periodic, u_knots_list, v_knots_list, \
                u_mults_list, v_mults_list, poles, valid = dataset.load_data(data_path_file)
            
            # Filter criteria
            high_degree = u_degree > 3 or v_degree > 3
            long_knots = len(u_knots_list) > 100 or len(v_knots_list) > 100
            
            # Track drops
            if high_degree and long_knots:
                dropped_both += 1
                continue
            elif high_degree:
                dropped_high_degree += 1
                continue
            elif long_knots:
                dropped_long_knots += 1
                continue
            
            # Keep this surface
            u_knots_lengths.append(len(u_knots_list))
            v_knots_lengths.append(len(v_knots_list))
            u_degree_list.append(u_degree)
            v_degree_list.append(v_degree)
            num_poles_u_list.append(num_poles_u)
            num_poles_v_list.append(num_poles_v)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Convert to numpy arrays
    u_knots_lengths = np.array(u_knots_lengths)
    v_knots_lengths = np.array(v_knots_lengths)
    u_degree_list = np.array(u_degree_list)
    v_degree_list = np.array(v_degree_list)
    num_poles_u_list = np.array(num_poles_u_list)
    num_poles_v_list = np.array(num_poles_v_list)
    
    # Calculate and report drops
    kept_surfaces = len(u_knots_lengths)
    total_dropped = dropped_high_degree + dropped_long_knots + dropped_both
    
    # Print statistics
    print("\n" + "="*60)
    print("FILTERING REPORT")
    print("="*60)
    print(f"Total surfaces in dataset: {total_surfaces}")
    print(f"Surfaces kept after filtering: {kept_surfaces} ({kept_surfaces/total_surfaces*100:.2f}%)")
    print(f"\nDropped surfaces breakdown:")
    print(f"  - High degree only (u_degree > 3 or v_degree > 3): {dropped_high_degree} ({dropped_high_degree/total_surfaces*100:.2f}%)")
    print(f"  - Long knots only (len > 100): {dropped_long_knots} ({dropped_long_knots/total_surfaces*100:.2f}%)")
    print(f"  - Both conditions: {dropped_both} ({dropped_both/total_surfaces*100:.2f}%)")
    print(f"  - Total dropped: {total_dropped} ({total_dropped/total_surfaces*100:.2f}%)")
    
    print("\n" + "="*60)
    print("STATISTICS (FILTERED DATA)")
    print("="*60)
    print(f"\nU-direction knot vector lengths:")
    print(f"  Min: {u_knots_lengths.min()}")
    print(f"  Max: {u_knots_lengths.max()}")
    print(f"  Mean: {u_knots_lengths.mean():.2f}")
    print(f"  Median: {np.median(u_knots_lengths):.2f}")
    print(f"  Std: {u_knots_lengths.std():.2f}")
    
    print(f"\nV-direction knot vector lengths:")
    print(f"  Min: {v_knots_lengths.min()}")
    print(f"  Max: {v_knots_lengths.max()}")
    print(f"  Mean: {v_knots_lengths.mean():.2f}")
    print(f"  Median: {np.median(v_knots_lengths):.2f}")
    print(f"  Std: {v_knots_lengths.std():.2f}")
    
    print(f"\nU-direction degree:")
    print(f"  Distribution: {Counter(u_degree_list)}")
    
    print(f"\nV-direction degree:")
    print(f"  Distribution: {Counter(v_degree_list)}")
    
    print(f"\nU-direction number of poles:")
    print(f"  Min: {num_poles_u_list.min()}")
    print(f"  Max: {num_poles_u_list.max()}")
    print(f"  Mean: {num_poles_u_list.mean():.2f}")
    
    print(f"\nV-direction number of poles:")
    print(f"  Min: {num_poles_v_list.min()}")
    print(f"  Max: {num_poles_v_list.max()}")
    print(f"  Mean: {num_poles_v_list.mean():.2f}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Figure 1: Histograms of knot vector lengths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # U-direction histogram
    axes[0].hist(u_knots_lengths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Number of Knots', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'U-direction Knot Vector Length Distribution\n(n={len(u_knots_lengths)}, filtered)', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axvline(u_knots_lengths.mean(), color='red', linestyle='--', 
                    label=f'Mean: {u_knots_lengths.mean():.2f}')
    axes[0].axvline(np.median(u_knots_lengths), color='orange', linestyle='--', 
                    label=f'Median: {np.median(u_knots_lengths):.2f}')
    axes[0].legend()
    
    # V-direction histogram
    axes[1].hist(v_knots_lengths, bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[1].set_xlabel('Number of Knots', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'V-direction Knot Vector Length Distribution\n(n={len(v_knots_lengths)}, filtered)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axvline(v_knots_lengths.mean(), color='red', linestyle='--', 
                    label=f'Mean: {v_knots_lengths.mean():.2f}')
    axes[1].axvline(np.median(v_knots_lengths), color='orange', linestyle='--', 
                    label=f'Median: {np.median(v_knots_lengths):.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    output_path = output_dir / 'knot_vector_length_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Figure 2: Joint distribution (2D histogram)
    fig, ax = plt.subplots(figsize=(10, 8))
    h = ax.hist2d(u_knots_lengths, v_knots_lengths, bins=50, cmap='YlOrRd')
    ax.set_xlabel('U-direction Knot Vector Length', fontsize=12)
    ax.set_ylabel('V-direction Knot Vector Length', fontsize=12)
    ax.set_title(f'Joint Distribution of Knot Vector Lengths\n(n={len(u_knots_lengths)}, filtered)', fontsize=14)
    plt.colorbar(h[3], ax=ax, label='Frequency')
    ax.grid(alpha=0.3)
    
    output_path = output_dir / 'knot_vector_length_joint_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Figure 3: Box plots
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [u_knots_lengths, v_knots_lengths]
    bp = ax.boxplot(box_data, labels=['U-direction', 'V-direction'], 
                    patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = ['steelblue', 'forestgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Number of Knots', fontsize=12)
    ax.set_title(f'Knot Vector Length Box Plot Comparison\n(n={len(u_knots_lengths)}, filtered)', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'knot_vector_length_boxplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Figure 4: Cumulative distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # U-direction CDF
    u_sorted = np.sort(u_knots_lengths)
    u_cumulative = np.arange(1, len(u_sorted) + 1) / len(u_sorted)
    axes[0].plot(u_sorted, u_cumulative, linewidth=2, color='steelblue')
    axes[0].set_xlabel('Number of Knots', fontsize=12)
    axes[0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[0].set_title(f'U-direction Knot Vector Length CDF\n(n={len(u_knots_lengths)}, filtered)', fontsize=14)
    axes[0].grid(alpha=0.3)
    
    # V-direction CDF
    v_sorted = np.sort(v_knots_lengths)
    v_cumulative = np.arange(1, len(v_sorted) + 1) / len(v_sorted)
    axes[1].plot(v_sorted, v_cumulative, linewidth=2, color='forestgreen')
    axes[1].set_xlabel('Number of Knots', fontsize=12)
    axes[1].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1].set_title(f'V-direction Knot Vector Length CDF\n(n={len(v_knots_lengths)}, filtered)', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'knot_vector_length_cdf.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Figure 5: Degree distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # U-degree distribution
    u_degree_counter = Counter(u_degree_list)
    degrees_u = sorted(u_degree_counter.keys())
    counts_u = [u_degree_counter[d] for d in degrees_u]
    axes[0].bar(degrees_u, counts_u, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Degree', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'U-direction Degree Distribution\n(n={len(u_degree_list)}, filtered, degree ≤ 3)', fontsize=14)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticks(degrees_u)
    
    # V-degree distribution
    v_degree_counter = Counter(v_degree_list)
    degrees_v = sorted(v_degree_counter.keys())
    counts_v = [v_degree_counter[d] for d in degrees_v]
    axes[1].bar(degrees_v, counts_v, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[1].set_xlabel('Degree', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'V-direction Degree Distribution\n(n={len(v_degree_list)}, filtered, degree ≤ 3)', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticks(degrees_v)
    
    plt.tight_layout()
    output_path = output_dir / 'degree_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Figure 6: Number of poles distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # U-direction poles histogram
    axes[0, 0].hist(num_poles_u_list, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('Number of Poles', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'U-direction Number of Poles\n(n={len(num_poles_u_list)}, filtered)', fontsize=14)
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].axvline(num_poles_u_list.mean(), color='red', linestyle='--', 
                       label=f'Mean: {num_poles_u_list.mean():.2f}')
    axes[0, 0].legend()
    
    # V-direction poles histogram
    axes[0, 1].hist(num_poles_v_list, bins=30, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[0, 1].set_xlabel('Number of Poles', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title(f'V-direction Number of Poles\n(n={len(num_poles_v_list)}, filtered)', fontsize=14)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axvline(num_poles_v_list.mean(), color='red', linestyle='--', 
                       label=f'Mean: {num_poles_v_list.mean():.2f}')
    axes[0, 1].legend()
    
    # Joint distribution of poles
    h = axes[1, 0].hist2d(num_poles_u_list, num_poles_v_list, bins=30, cmap='YlOrRd')
    axes[1, 0].set_xlabel('U-direction Number of Poles', fontsize=12)
    axes[1, 0].set_ylabel('V-direction Number of Poles', fontsize=12)
    axes[1, 0].set_title(f'Joint Distribution of Number of Poles\n(filtered)', fontsize=14)
    plt.colorbar(h[3], ax=axes[1, 0], label='Frequency')
    
    # Relationship between knots and poles
    axes[1, 1].scatter(u_knots_lengths, num_poles_u_list, alpha=0.3, s=10, c='steelblue', label='U-direction')
    axes[1, 1].scatter(v_knots_lengths, num_poles_v_list, alpha=0.3, s=10, c='forestgreen', label='V-direction')
    axes[1, 1].set_xlabel('Number of Knots', fontsize=12)
    axes[1, 1].set_ylabel('Number of Poles', fontsize=12)
    axes[1, 1].set_title(f'Relationship: Knots vs Poles\n(filtered)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'poles_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save statistics to text file
    stats_path = output_dir / 'knot_distribution_statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("B-SPLINE SURFACE KNOT VECTOR STATISTICS (FILTERED)\n")
        f.write("=" * 60 + "\n\n")
        f.write("FILTERING REPORT:\n")
        f.write(f"Total surfaces in dataset: {total_surfaces}\n")
        f.write(f"Surfaces kept after filtering: {kept_surfaces} ({kept_surfaces/total_surfaces*100:.2f}%)\n\n")
        f.write("Dropped surfaces breakdown:\n")
        f.write(f"  - High degree only (u_degree > 3 or v_degree > 3): {dropped_high_degree} ({dropped_high_degree/total_surfaces*100:.2f}%)\n")
        f.write(f"  - Long knots only (len > 100): {dropped_long_knots} ({dropped_long_knots/total_surfaces*100:.2f}%)\n")
        f.write(f"  - Both conditions: {dropped_both} ({dropped_both/total_surfaces*100:.2f}%)\n")
        f.write(f"  - Total dropped: {total_dropped} ({total_dropped/total_surfaces*100:.2f}%)\n\n")
        f.write("FILTERING CRITERIA:\n")
        f.write("  - u_degree <= 3 and v_degree <= 3\n")
        f.write("  - len(u_knots) <= 100 and len(v_knots) <= 100\n\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("U-DIRECTION KNOT VECTOR LENGTHS:\n")
        f.write(f"  Min: {u_knots_lengths.min()}\n")
        f.write(f"  Max: {u_knots_lengths.max()}\n")
        f.write(f"  Mean: {u_knots_lengths.mean():.2f}\n")
        f.write(f"  Median: {np.median(u_knots_lengths):.2f}\n")
        f.write(f"  Std: {u_knots_lengths.std():.2f}\n\n")
        
        f.write("V-DIRECTION KNOT VECTOR LENGTHS:\n")
        f.write(f"  Min: {v_knots_lengths.min()}\n")
        f.write(f"  Max: {v_knots_lengths.max()}\n")
        f.write(f"  Mean: {v_knots_lengths.mean():.2f}\n")
        f.write(f"  Median: {np.median(v_knots_lengths):.2f}\n")
        f.write(f"  Std: {v_knots_lengths.std():.2f}\n\n")
        
        f.write("U-DIRECTION DEGREE DISTRIBUTION:\n")
        for degree, count in sorted(Counter(u_degree_list).items()):
            f.write(f"  Degree {degree}: {count} ({count/len(u_degree_list)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("V-DIRECTION DEGREE DISTRIBUTION:\n")
        for degree, count in sorted(Counter(v_degree_list).items()):
            f.write(f"  Degree {degree}: {count} ({count/len(v_degree_list)*100:.2f}%)\n")
        f.write("\n")
        
        f.write("U-DIRECTION NUMBER OF POLES:\n")
        f.write(f"  Min: {num_poles_u_list.min()}\n")
        f.write(f"  Max: {num_poles_u_list.max()}\n")
        f.write(f"  Mean: {num_poles_u_list.mean():.2f}\n")
        f.write(f"  Median: {np.median(num_poles_u_list):.2f}\n")
        f.write(f"  Std: {num_poles_u_list.std():.2f}\n\n")
        
        f.write("V-DIRECTION NUMBER OF POLES:\n")
        f.write(f"  Min: {num_poles_v_list.min()}\n")
        f.write(f"  Max: {num_poles_v_list.max()}\n")
        f.write(f"  Mean: {num_poles_v_list.mean():.2f}\n")
        f.write(f"  Median: {np.median(num_poles_v_list):.2f}\n")
        f.write(f"  Std: {num_poles_v_list.std():.2f}\n")
    
    print(f"Saved statistics: {stats_path}")
    
    print("\n" + "="*60)
    print("Analysis complete! All figures saved to:", output_dir)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze knot vector length distributions in B-spline dataset"
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to directory containing B-spline .npy files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="assets",
        help="Directory to save output figures (default: assets)"
    )
    
    args = parser.parse_args()
    
    analyze_knot_distributions(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()

