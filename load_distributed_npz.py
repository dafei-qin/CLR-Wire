#!/usr/bin/env python3
"""
Utility script to load and work with distributed NPZ files (one per UID).
Provides functions to load individual files or create aggregated datasets.
"""

import argparse
import numpy as np
import os
import glob
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

class DistributedNPZLoader:
    def __init__(self, npz_directory: str):
        """Initialize with directory containing UID.npz files"""
        self.npz_directory = npz_directory
        self.available_uids = self._find_available_uids()
        print(f"Found {len(self.available_uids)} NPZ files in {npz_directory}")
    
    def _find_available_uids(self) -> List[str]:
        """Find all available UID NPZ files"""
        npz_pattern = os.path.join(self.npz_directory, "*.npz")
        npz_files = glob.glob(npz_pattern)
        
        uids = []
        for npz_file in npz_files:
            filename = os.path.basename(npz_file)
            if filename != "dataset_summary.txt":  # Skip summary file
                uid = os.path.splitext(filename)[0]  # Remove .npz extension
                uids.append(uid)
        
        return sorted(uids)
    
    def load_single_uid(self, uid: str) -> Optional[Dict[str, np.ndarray]]:
        """Load data for a single UID"""
        npz_file = os.path.join(self.npz_directory, f"{uid}.npz")
        
        if not os.path.exists(npz_file):
            print(f"Warning: NPZ file for UID {uid} not found")
            return None
        
        try:
            data = np.load(npz_file, allow_pickle=True)
            return {
                'surface_cp_list': data['surface_cp_list'],
                'surface_points_list': data['surface_points_list'],
                'curve_cp_lists': data['curve_cp_lists'],
                'curve_points_lists': data['curve_points_lists'],
                'surface_indices': data['surface_indices'],
                'surface_types': data['surface_types'],
                'file_paths': data['file_paths'],
                'uids': data['uids'],
                'uid': uid
            }
        except Exception as e:
            print(f"Error loading UID {uid}: {e}")
            return None
    
    def get_uid_statistics(self, uid: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a single UID"""
        data = self.load_single_uid(uid)
        if data is None:
            return None
        
        num_surfaces = len(data['surface_cp_list'])
        num_curves = sum(len(curves) for curves in data['curve_cp_lists'])
        
        # Count surface types
        surface_types = data['surface_types']
        from collections import Counter
        type_counts = Counter(surface_types)
        
        return {
            'uid': uid,
            'num_surfaces': num_surfaces,
            'num_curves': num_curves,
            'avg_curves_per_surface': num_curves / num_surfaces if num_surfaces > 0 else 0,
            'surface_type_distribution': dict(type_counts),
            'most_common_type': type_counts.most_common(1)[0] if type_counts else ('unknown', 0)
        }
    
    def create_subset_dataset(self, uids: List[str], output_file: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        """Create aggregated dataset from subset of UIDs"""
        print(f"Creating dataset from {len(uids)} UIDs...")
        
        # Initialize global lists
        all_surface_cp_list = []
        all_surface_points_list = []
        all_curve_cp_lists = []
        all_curve_points_lists = []
        all_surface_indices = []
        all_surface_types = []
        all_file_paths = []
        all_uids = []
        
        successful_uids = 0
        for uid in tqdm(uids, desc="Loading UIDs"):
            data = self.load_single_uid(uid)
            
            if data is not None:
                all_surface_cp_list.extend(data['surface_cp_list'])
                all_surface_points_list.extend(data['surface_points_list'])
                all_curve_cp_lists.extend(data['curve_cp_lists'])
                all_curve_points_lists.extend(data['curve_points_lists'])
                all_surface_indices.extend(data['surface_indices'])
                all_surface_types.extend(data['surface_types'])
                all_file_paths.extend(data['file_paths'])
                all_uids.extend(data['uids'])
                successful_uids += 1
        
        print(f"Successfully loaded {successful_uids}/{len(uids)} UIDs")
        
        if successful_uids == 0:
            print("No data loaded")
            return None
        
        # Convert to numpy arrays
        surface_indices_array = np.array(all_surface_indices)
        surface_types_array = np.array(all_surface_types)
        file_paths_array = np.array(all_file_paths)
        uids_array = np.array(all_uids)
        
        surface_cp_array = np.array(all_surface_cp_list, dtype=object)
        surface_points_array = np.array(all_surface_points_list, dtype=object)
        curve_cp_array = np.array(all_curve_cp_lists, dtype=object)
        curve_points_array = np.array(all_curve_points_lists, dtype=object)
        
        result = {
            'surface_cp_list': surface_cp_array,
            'surface_points_list': surface_points_array,
            'curve_cp_lists': curve_cp_array,
            'curve_points_lists': curve_points_array,
            'surface_indices': surface_indices_array,
            'surface_types': surface_types_array,
            'file_paths': file_paths_array,
            'uids': uids_array
        }
        
        # Save if output file specified
        if output_file:
            np.savez_compressed(output_file, **result)
            print(f"Aggregated dataset saved to: {output_file}")
        
        return result
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze entire distributed dataset"""
        print("Analyzing entire distributed dataset...")
        
        total_surfaces = 0
        total_curves = 0
        all_surface_types = []
        uid_stats = []
        
        for uid in tqdm(self.available_uids, desc="Analyzing UIDs"):
            stats = self.get_uid_statistics(uid)
            if stats:
                uid_stats.append(stats)
                total_surfaces += stats['num_surfaces']
                total_curves += stats['num_curves']
                all_surface_types.extend(stats['surface_type_distribution'].keys())
        
        # Overall statistics
        from collections import Counter
        surface_type_counts = Counter()
        for stats in uid_stats:
            for surf_type, count in stats['surface_type_distribution'].items():
                surface_type_counts[surf_type] += count
        
        avg_curves_per_surface = total_curves / total_surfaces if total_surfaces > 0 else 0
        avg_surfaces_per_uid = total_surfaces / len(uid_stats) if uid_stats else 0
        
        return {
            'total_uids': len(self.available_uids),
            'successful_uids': len(uid_stats),
            'total_surfaces': total_surfaces,
            'total_curves': total_curves,
            'avg_curves_per_surface': avg_curves_per_surface,
            'avg_surfaces_per_uid': avg_surfaces_per_uid,
            'surface_type_distribution': dict(surface_type_counts),
            'uid_statistics': uid_stats
        }
    
    def filter_uids_by_criteria(self, 
                               min_surfaces: Optional[int] = None,
                               max_surfaces: Optional[int] = None,
                               min_curves: Optional[int] = None,
                               max_curves: Optional[int] = None,
                               required_surface_types: Optional[List[str]] = None,
                               excluded_surface_types: Optional[List[str]] = None) -> List[str]:
        """Filter UIDs based on criteria"""
        print("Filtering UIDs based on criteria...")
        
        filtered_uids = []
        
        for uid in tqdm(self.available_uids, desc="Filtering UIDs"):
            stats = self.get_uid_statistics(uid)
            if stats is None:
                continue
            
            # Check criteria
            if min_surfaces is not None and stats['num_surfaces'] < min_surfaces:
                continue
            if max_surfaces is not None and stats['num_surfaces'] > max_surfaces:
                continue
            if min_curves is not None and stats['num_curves'] < min_curves:
                continue
            if max_curves is not None and stats['num_curves'] > max_curves:
                continue
            
            surface_types = set(stats['surface_type_distribution'].keys())
            
            if required_surface_types is not None:
                if not all(req_type in surface_types for req_type in required_surface_types):
                    continue
            
            if excluded_surface_types is not None:
                if any(excl_type in surface_types for excl_type in excluded_surface_types):
                    continue
            
            filtered_uids.append(uid)
        
        print(f"Filtered {len(filtered_uids)}/{len(self.available_uids)} UIDs")
        return filtered_uids


def main():
    parser = argparse.ArgumentParser(description="Load and analyze distributed NPZ files")
    parser.add_argument("--input", type=str, required=True, help="Directory containing UID.npz files")
    parser.add_argument("--action", type=str, choices=['analyze', 'create_subset', 'list_uids', 'uid_stats'], 
                       default='analyze', help="Action to perform")
    parser.add_argument("--uids", type=str, nargs='+', help="Specific UIDs to process (for create_subset or uid_stats)")
    parser.add_argument("--output", type=str, help="Output file for create_subset action")
    parser.add_argument("--max_uids", type=int, help="Maximum number of UIDs to include (for create_subset)")
    parser.add_argument("--min_surfaces", type=int, help="Minimum surfaces per UID filter")
    parser.add_argument("--max_surfaces", type=int, help="Maximum surfaces per UID filter")
    parser.add_argument("--required_types", type=str, nargs='+', help="Required surface types")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Initialize loader
    loader = DistributedNPZLoader(args.input)
    
    if args.action == 'analyze':
        # Analyze entire dataset
        stats = loader.analyze_dataset()
        
        print(f"\n=== Dataset Analysis ===")
        print(f"Total UIDs: {stats['total_uids']}")
        print(f"Successfully loaded: {stats['successful_uids']}")
        print(f"Total surfaces: {stats['total_surfaces']}")
        print(f"Total curves: {stats['total_curves']}")
        print(f"Average curves per surface: {stats['avg_curves_per_surface']:.2f}")
        print(f"Average surfaces per UID: {stats['avg_surfaces_per_uid']:.2f}")
        
        print(f"\nSurface Type Distribution:")
        for surf_type, count in sorted(stats['surface_type_distribution'].items()):
            percentage = (count / stats['total_surfaces']) * 100
            print(f"  {surf_type}: {count} ({percentage:.1f}%)")
    
    elif args.action == 'list_uids':
        # List all available UIDs
        print(f"\nAvailable UIDs ({len(loader.available_uids)}):")
        for i, uid in enumerate(loader.available_uids):
            print(f"  {i+1:4d}: {uid}")
    
    elif args.action == 'uid_stats':
        # Show statistics for specific UIDs
        uids_to_check = args.uids or loader.available_uids[:10]  # Default to first 10
        
        print(f"\n=== UID Statistics ===")
        for uid in uids_to_check:
            stats = loader.get_uid_statistics(uid)
            if stats:
                print(f"\nUID: {uid}")
                print(f"  Surfaces: {stats['num_surfaces']}")
                print(f"  Curves: {stats['num_curves']}")
                print(f"  Avg curves/surface: {stats['avg_curves_per_surface']:.2f}")
                print(f"  Most common type: {stats['most_common_type'][0]} ({stats['most_common_type'][1]})")
    
    elif args.action == 'create_subset':
        # Create subset dataset
        if args.uids:
            target_uids = args.uids
        else:
            # Apply filters if specified
            target_uids = loader.filter_uids_by_criteria(
                min_surfaces=args.min_surfaces,
                max_surfaces=args.max_surfaces,
                required_surface_types=args.required_types
            )
            
            # Limit number if specified
            if args.max_uids and len(target_uids) > args.max_uids:
                target_uids = target_uids[:args.max_uids]
                print(f"Limited to first {args.max_uids} UIDs")
        
        if not target_uids:
            print("No UIDs match the criteria")
            return
        
        print(f"Creating subset from {len(target_uids)} UIDs")
        
        # Create aggregated dataset
        output_file = args.output or "subset_dataset.npz"
        dataset = loader.create_subset_dataset(target_uids, output_file)
        
        if dataset:
            print(f"Dataset created with {len(dataset['surface_cp_list'])} surfaces")


if __name__ == "__main__":
    main() 