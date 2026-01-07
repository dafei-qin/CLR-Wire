import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))


def load_and_stat_token_lengths(cache_file_path):
    """
    Load cache file and extract token lengths.
    
    Args:
        cache_file_path: Path to the cache pickle file
        
    Returns:
        List of token lengths
    """
    print(f"Loading cache file: {cache_file_path}")
    with open(cache_file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Cache file loaded. Keys: {data.keys()}")
    
    if 'tokens' not in data:
        raise KeyError("'tokens' key not found in cache file")
    
    tokens = data['tokens']
    print(f"Number of samples: {len(tokens)}")
    
    # Calculate token lengths
    token_lengths = []
    for idx, token_array in enumerate(tokens):
        if isinstance(token_array, np.ndarray):
            length = len(token_array)
        elif isinstance(token_array, (list, tuple)):
            length = len(token_array)
        else:
            print(f"Warning: Unexpected token type at index {idx}: {type(token_array)}")
            continue

        if length < 200:
            repeat = 1
        elif length >= 200 and length < 400:
            repeat = 2
        elif length >= 400 and length < 600:
            repeat = 4
        elif length >= 600:
            repeat = 8
        for _r in range(repeat):
            token_lengths.append(length)
    
    token_lengths = np.array(token_lengths)
    print(f"Token lengths statistics:")
    print(f"  Min: {token_lengths.min()}")
    print(f"  Max: {token_lengths.max()}")
    print(f"  Mean: {token_lengths.mean():.2f}")
    print(f"  Median: {np.median(token_lengths):.2f}")
    print(f"  Std: {token_lengths.std():.2f}")
    
    return token_lengths


def save_histogram_info(token_lengths, counts, bins, output_txt_path):
    """
    Save histogram information to a text file.
    
    Args:
        token_lengths: Array of token lengths
        counts: Histogram counts for each bin
        bins: Histogram bin edges
        output_txt_path: Path to save the text file
    """
    output_dir = os.path.dirname(output_txt_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Token Length Distribution - Histogram Information\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("Overall Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {len(token_lengths)}\n")
        f.write(f"Min: {token_lengths.min()}\n")
        f.write(f"Max: {token_lengths.max()}\n")
        f.write(f"Mean: {token_lengths.mean():.6f}\n")
        f.write(f"Median: {np.median(token_lengths):.6f}\n")
        f.write(f"Std: {token_lengths.std():.6f}\n")
        f.write(f"Variance: {token_lengths.var():.6f}\n")
        f.write("\n")
        
        # Percentiles
        f.write("Percentiles:\n")
        f.write("-" * 80 + "\n")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
        for p in percentiles:
            value = np.percentile(token_lengths, p)
            f.write(f"{p:6.1f}%: {value:.2f}\n")
        f.write("\n")
        
        # Histogram bins information
        f.write("Histogram Bins:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of bins: {len(counts)}\n")
        f.write(f"Bin width: {bins[1] - bins[0]:.2f}\n")
        f.write("\n")
        f.write(f"{'Bin Range':<30} {'Count':<15} {'Percentage':<15} {'Cumulative %':<15}\n")
        f.write("-" * 80 + "\n")
        
        total = len(token_lengths)
        cumulative = 0
        for i in range(len(counts)):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            count = int(counts[i])  # Convert to int for formatting
            cumulative += count
            percentage = (count / total) * 100
            cumulative_percentage = (cumulative / total) * 100
            f.write(f"[{bin_start:8.2f}, {bin_end:8.2f})  {count:15d}  {percentage:14.4f}%  {cumulative_percentage:14.4f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"Histogram information saved to: {output_txt_path}")


def plot_token_length_distribution(token_lengths, output_path, output_txt_path):
    """
    Plot token length distribution and save to file.
    
    Args:
        token_lengths: Array of token lengths
        output_path: Path to save the plot
        output_txt_path: Path to save the histogram info text file
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    n_bins = min(50, len(np.unique(token_lengths)))
    counts, bins, patches = ax1.hist(token_lengths, bins=n_bins, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Token Length', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Token Length Distribution (Histogram)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Min: {token_lengths.min()}, Max: {token_lengths.max()}\n'
    stats_text += f'Mean: {token_lengths.mean():.2f}, Median: {np.median(token_lengths):.2f}\n'
    stats_text += f'Std: {token_lengths.std():.2f}, Total samples: {len(token_lengths)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Cumulative distribution
    ax2 = axes[1]
    sorted_lengths = np.sort(token_lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax2.plot(sorted_lengths, cumulative, linewidth=2)
    ax2.set_xlabel('Token Length', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Token Length Cumulative Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Add percentile markers
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        ax2.axvline(value, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax2.text(value, 0.02, f'{p}%', rotation=90, fontsize=8, 
                verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plot saved to: {output_path}")
    
    plt.close()
    
    # Save histogram information to text file
    save_histogram_info(token_lengths, counts, bins, output_txt_path)


def main():
    # Default cache file path (can be modified)
    cache_file = 'cache_013456789.pkl'
    
    # Try to find the cache file in common locations
    possible_paths = [
        cache_file,  # Current directory
        os.path.join("/deemos-research-area-d/meshgen/cad_data/abc_step_pc", cache_file),
        os.path.join(project_root, cache_file),  # Project root
        os.path.join(project_root, 'data', cache_file),  # Data directory
        os.path.join(project_root, 'assets', cache_file),  # Assets directory
    ]
    
    cache_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            cache_file_path = path
            break
    
    if cache_file_path is None:
        print(f"Error: Could not find cache file '{cache_file}' in the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease specify the full path to the cache file.")
        return
    
    # Output paths
    output_path = os.path.join(project_root, 'assets', 'tokens_dist.jpg')
    output_txt_path = os.path.join(project_root, 'assets', 'tokens_dist.txt')
    
    # Load and process
    try:
        token_lengths = load_and_stat_token_lengths(cache_file_path)
        plot_token_length_distribution(token_lengths, output_path, output_txt_path)
        print("\nDone!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

