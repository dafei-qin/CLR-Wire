"""Plot Chamfer Distance distribution (CDF) from results JSON."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cd_cdf(results_json: str, output_path: str = None, show_plot: bool = False):
    """
    Plot CDF of Chamfer Distance distribution.
    
    Args:
        results_json: Path to results JSON file
        output_path: Output path for the plot (optional)
        show_plot: Whether to display the plot
    """
    # Load results
    with open(results_json, 'r') as f:
        data = json.load(f)
    
    results = data.get('results_sorted_by_cd', [])
    
    if not results:
        print("No results found in JSON file!")
        return
    
    # Extract CD values
    cd_values = np.array([r['chamfer_dist'] for r in results])
    
    # Sort for CDF
    cd_sorted = np.sort(cd_values)
    
    # Calculate cumulative counts and percentages
    cumulative_counts = np.arange(1, len(cd_sorted) + 1)
    cumulative_percentages = (cumulative_counts / len(cd_sorted)) * 100
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot absolute count on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Chamfer Distance', fontsize=12)
    ax1.set_ylabel('Cumulative Count', color=color1, fontsize=12)
    ax1.plot(cd_sorted, cumulative_counts, color=color1, linewidth=2, label='Absolute Count')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for percentage
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cumulative Percentage (%)', color=color2, fontsize=12)
    ax2.plot(cd_sorted, cumulative_percentages, color=color2, linewidth=2, 
             linestyle='--', label='Percentage')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Set y-axis limits for percentage
    ax2.set_ylim(0, 100)
    
    # Add title with statistics
    stats = data.get('statistics', {})
    checkpoint = data.get('checkpoint', 'unknown')
    title = f'Chamfer Distance CDF - Checkpoint {checkpoint}\n'
    title += f'Mean: {stats.get("mean", 0):.6f} | Median: {stats.get("median", 0):.6f} | '
    title += f'Std: {stats.get("std", 0):.6f}\n'
    title += f'Total samples: {len(cd_sorted)}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add vertical lines for key percentiles
    percentiles = [5, 50, 95]
    percentile_values = [np.percentile(cd_values, p) for p in percentiles]
    
    for p, val in zip(percentiles, percentile_values):
        ax1.axvline(x=val, color='gray', linestyle=':', alpha=0.5)
        ax1.text(val, ax1.get_ylim()[1] * 0.95, f'{p}%: {val:.6f}', 
                rotation=90, verticalalignment='top', fontsize=9)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"Chamfer Distance Statistics (Checkpoint {checkpoint})")
    print(f"{'='*60}")
    print(f"Total samples: {len(cd_values)}")
    print(f"Mean:          {np.mean(cd_values):.6f}")
    print(f"Median:        {np.median(cd_values):.6f}")
    print(f"Std:           {np.std(cd_values):.6f}")
    print(f"Min:           {np.min(cd_values):.6f}")
    print(f"Max:           {np.max(cd_values):.6f}")
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p:3d}%: {np.percentile(cd_values, p):.6f}")
    print(f"{'='*60}")


def plot_multiple_checkpoints(json_files: list, output_path: str = None, show_plot: bool = False):
    """Plot CDF for multiple checkpoints on the same figure."""
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(json_files)))
    
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results = data.get('results_sorted_by_cd', [])
        if not results:
            continue
        
        cd_values = np.array([r['chamfer_dist'] for r in results])
        cd_sorted = np.sort(cd_values)
        cumulative_percentages = (np.arange(1, len(cd_sorted) + 1) / len(cd_sorted)) * 100
        
        checkpoint = data.get('checkpoint', Path(json_file).stem.replace('results_', ''))
        ax1.plot(cd_sorted, cumulative_percentages, color=colors[i], 
                linewidth=2, label=f'Checkpoint {checkpoint}')
    
    ax1.set_xlabel('Chamfer Distance', fontsize=12)
    ax1.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax1.set_title('Chamfer Distance CDF Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Chamfer Distance CDF from results JSON.')
    parser.add_argument(
        '--results_json',
        type=str,
        default='/deemos-research-area-d/meshgen/cad/CLR-Wire/src/eval/results_230000.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='cd_distribution.png',
        help='Output path for the plot'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        type=str,
        default=None,
        help='Compare multiple JSON files'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        plot_multiple_checkpoints(args.compare, args.output, args.show)
    else:
        plot_cd_cdf(args.results_json, args.output, args.show)


if __name__ == '__main__':
    main()


