"""
Test script for visualizing sequence ordering after augmentation and reordering in dataset_v4_tokenize_all.
Visualizes surfaces in the order they appear in tokens using polyscope.

Usage:
    python src/test/test_dataset_v4_tokenize_augment_sequence.py --config <path_to_yaml>
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import argparse
import yaml
from typing import List, Dict
import os

from src.dataset.dataset_v4_tokenize_all import dataset_compound_tokenize_all, dataset_compound_tokenize_all_cache
from src.utils.surface_tools import params_to_samples


class SequenceVisualizer:
    def __init__(self, dataset, sample_idx=0):
        """
        Args:
            dataset: dataset_compound_tokenize_all or dataset_compound_tokenize_all_cache instance
            sample_idx: Index of the sample to visualize
        """
        self.dataset = dataset
        self.sample_idx = sample_idx
        self.current_surface_idx = 0
        self.show_all = False
        self.show_points = True
        self.show_normals = False
        self.surface_resolution = 32
        
        # Load data
        self.load_data()
        
        # Initialize polyscope
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        
    def load_data(self):
        """Load and decode data from dataset"""
        data = self.dataset[self.sample_idx]
        
        if len(data) == 6:
            self.points, self.normals, tokens_padded, poles_padded, poles_mask, self.solid_valid = data
        else:
            raise ValueError(f"Unexpected data format with {len(data)} elements")
        
        if not self.solid_valid:
            print(f"Warning: Sample {self.sample_idx} is marked as invalid!")
        
        # Decode tokens to surfaces
        self.tokens = tokens_padded
        self.poles = poles_padded[poles_mask]
        
        # Remove padding and start/end tokens
        self.tokens_unpadded = self.dataset.unwarp_codes(self.tokens)
        
        # Detokenize to get surface parameters
        self.surfaces = self.dataset.detokenize(self.tokens, self.poles)
        
        print(f"\n{'='*60}")
        print(f"Loaded sample {self.sample_idx}")
        print(f"Valid: {self.solid_valid}")
        print(f"Number of surfaces: {len(self.surfaces)}")
        print(f"Number of points: {len(self.points)}")
        print(f"Tokens shape: {self.tokens_unpadded.shape}")
        print(f"{'='*60}\n")
        
        # Sample all surfaces
        self.surface_samples = []
        for i, surface in enumerate(self.surfaces):
            samples = self._sample_surface(surface, self.surface_resolution)
            self.surface_samples.append(samples)
            surface_type = surface['type']
            print(f"Surface {i:2d}: {surface_type:20s} - {samples.shape}")
    
    def _sample_surface(self, surface: Dict, resolution: int = 32) -> np.ndarray:
        """Sample a surface to get point cloud"""
        if surface['type'] == 'bspline_surface':
            surface['poles'] = np.array(surface['poles'])
            samples = params_to_samples(
                torch.zeros([]), 
                surface['type'], 
                resolution, 
                resolution, 
                surface
            ).squeeze(0).numpy()
        else:
            surface['location'] = torch.tensor(surface['location'])
            surface['direction'] = torch.tensor(surface['direction'])
            surface['scalar'] = torch.tensor(surface['scalar'])
            surface['uv'] = torch.tensor(surface['uv'])
            samples = params_to_samples(
                torch.zeros([]), 
                surface['type'], 
                resolution, 
                resolution, 
                surface
            ).squeeze(0).numpy()
        
        return samples
    
    def visualize(self):
        """Main visualization loop"""
        self.update_visualization()
        
        def callback():
            changed = False
            
            # Control panel
            psim.TextUnformatted("Sequence Visualization Controls")
            psim.Separator()
            
            # Sample selection
            psim.TextUnformatted(f"Sample: {self.sample_idx} | Valid: {self.solid_valid}")
            psim.TextUnformatted(f"Total surfaces: {len(self.surfaces)}")
            psim.Separator()
            
            # Surface navigation
            psim.TextUnformatted("Surface Navigation:")
            changed_idx, new_idx = psim.SliderInt("Surface Index", self.current_surface_idx, 
                                                   v_min=0, v_max=len(self.surfaces)-1)
            if changed_idx:
                self.current_surface_idx = new_idx
                changed = True
            
            if psim.Button("Previous"):
                self.current_surface_idx = max(0, self.current_surface_idx - 1)
                changed = True
            psim.SameLine()
            if psim.Button("Next"):
                self.current_surface_idx = min(len(self.surfaces) - 1, self.current_surface_idx + 1)
                changed = True
            
            psim.Separator()
            
            # Display options
            psim.TextUnformatted("Display Options:")
            changed_all, self.show_all = psim.Checkbox("Show All Surfaces", self.show_all)
            if changed_all:
                changed = True
            
            changed_pts, self.show_points = psim.Checkbox("Show Point Cloud", self.show_points)
            if changed_pts:
                changed = True
            
            changed_norm, self.show_normals = psim.Checkbox("Show Normals", self.show_normals)
            if changed_norm:
                changed = True
            
            psim.Separator()
            
            # Resolution control
            changed_res, new_res = psim.SliderInt("Surface Resolution", self.surface_resolution,
                                                   v_min=8, v_max=64)
            if changed_res:
                self.surface_resolution = new_res
                # Resample all surfaces
                self.surface_samples = []
                for surface in self.surfaces:
                    samples = self._sample_surface(surface, self.surface_resolution)
                    self.surface_samples.append(samples)
                changed = True
            
            psim.Separator()
            
            # Current surface info
            if self.current_surface_idx < len(self.surfaces):
                surface = self.surfaces[self.current_surface_idx]
                psim.TextUnformatted(f"Current Surface Info:")
                psim.TextUnformatted(f"  Index: {self.current_surface_idx}")
                psim.TextUnformatted(f"  Type: {surface['type']}")
                psim.TextUnformatted(f"  Orientation: {surface.get('orientation', 'N/A')}")
                
                # Show token information
                token = self.tokens_unpadded[self.current_surface_idx]
                psim.TextUnformatted(f"  Token (type): {token[0]}")
                psim.TextUnformatted(f"  Token (surf): {token[1:7].tolist()}")
                psim.TextUnformatted(f"  Token (rts): {token[7:14].tolist()}")
            
            if changed:
                self.update_visualization()
        
        ps.set_user_callback(callback)
        ps.show()
    
    def update_visualization(self):
        """Update the polyscope visualization"""
        ps.remove_all_structures()
        
        # Show point cloud
        if self.show_points and len(self.points) > 0:
            pc = ps.register_point_cloud("Input Points", self.points)
            pc.set_radius(0.002)
            pc.set_color((0.8, 0.8, 0.8))
            
            if self.show_normals and len(self.normals) > 0 and self.normals.shape == self.points.shape:
                pc.add_vector_quantity("Normals", self.normals, enabled=False, radius=0.001)
        
        # Show surfaces
        if self.show_all:
            # Show all surfaces with different colors
            for i, samples in enumerate(self.surface_samples):
                points = samples.reshape(-1, 3)
                # Create color based on sequence position
                hue = i / len(self.surface_samples)
                color = self._hsv_to_rgb(hue, 0.8, 0.9)
                
                cloud = ps.register_point_cloud(f"Surface_{i:02d}", points)
                cloud.set_radius(0.003)
                cloud.set_color(color)
        else:
            # Show only current surface
            if self.current_surface_idx < len(self.surface_samples):
                samples = self.surface_samples[self.current_surface_idx]
                points = samples.reshape(-1, 3)
                
                # Highlight current surface in red
                cloud = ps.register_point_cloud(f"Current_Surface_{self.current_surface_idx}", points)
                cloud.set_radius(0.005)
                cloud.set_color((1.0, 0.2, 0.2))
                
                # Show previous surfaces in gray
                for i in range(self.current_surface_idx):
                    samples_prev = self.surface_samples[i]
                    points_prev = samples_prev.reshape(-1, 3)
                    cloud_prev = ps.register_point_cloud(f"Previous_Surface_{i}", points_prev)
                    cloud_prev.set_radius(0.002)
                    cloud_prev.set_color((0.5, 0.5, 0.5))
    
    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Convert HSV to RGB color"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)


def load_config(config_path: str) -> Dict:
    """Load configuration from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_from_config(config: Dict):
    """Create dataset instance from config"""
    dataset_config = config.get('dataset', {})
    
    # Determine dataset type
    use_cache = dataset_config.get('use_cache', False)
    
    # Common parameters
    rts_codebook_dir = dataset_config.get('rts_codebook_dir', './assets/codebook')
    max_tokens = dataset_config.get('max_tokens', 1000)
    canonical = dataset_config.get('canonical', True)
    detect_closed = dataset_config.get('detect_closed', False)
    bspline_fit_threshold = dataset_config.get('bspline_fit_threshold', 1e-2)
    codebook_size = dataset_config.get('codebook_size', 1024)
    replica = dataset_config.get('replica', 1)
    rotation_augment = dataset_config.get('rotation_augment', False)
    
    if use_cache:
        cache_file = dataset_config.get('cache_file', '')
        if not cache_file:
            raise ValueError("cache_file must be specified when use_cache=True")
        
        print(f"Creating cached dataset from: {cache_file}")
        dataset = dataset_compound_tokenize_all_cache(
            cache_file=cache_file,
            rts_codebook_dir=rts_codebook_dir,
            max_tokens=max_tokens,
            canonical=canonical,
            detect_closed=detect_closed,
            bspline_fit_threshold=bspline_fit_threshold,
            codebook_size=codebook_size,
            replica=replica,
            rotation_augment=rotation_augment
        )
    else:
        json_dir = dataset_config.get('json_dir', '')
        if not json_dir:
            raise ValueError("json_dir must be specified when use_cache=False")
        
        print(f"Creating dataset from: {json_dir}")
        dataset = dataset_compound_tokenize_all(
            json_dir=json_dir,
            rts_codebook_dir=rts_codebook_dir,
            max_tokens=max_tokens,
            canonical=canonical,
            detect_closed=detect_closed,
            bspline_fit_threshold=bspline_fit_threshold,
            codebook_size=codebook_size,
            replica=replica,
            rotation_augment=rotation_augment
        )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Visualize dataset sequence after augmentation and reordering')
    parser.add_argument('--config', type=str, required=True, help='Path to dataset config yaml file')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Create dataset
    dataset = create_dataset_from_config(config)
    print(f"Dataset size: {len(dataset)}")
    
    # Create visualizer
    print(f"\nVisualizing sample {args.sample_idx}")
    visualizer = SequenceVisualizer(dataset, sample_idx=args.sample_idx)
    
    # Run visualization
    visualizer.visualize()


if __name__ == '__main__':
    main()

