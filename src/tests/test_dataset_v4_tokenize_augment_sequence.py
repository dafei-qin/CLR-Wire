"""
Test script for visualizing sequence ordering after augmentation and reordering in dataset_v4_tokenize_all.
Visualizes surfaces in the order they appear in tokens using polyscope.

Usage:
    python src/test/test_dataset_v4_tokenize_augment_sequence.py --config <path_to_yaml>
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import argparse
from typing import List, Dict
import os
from omegaconf import OmegaConf

from src.dataset.dataset_v4_tokenize_all import dataset_compound_tokenize_all, dataset_compound_tokenize_all_cache
from src.utils.surface_tools import params_to_samples
from src.utils.import_tools import load_dataset_from_config


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
        self.show_cumulative = False  # Show surfaces from 0 to current index
        self.show_points = True
        self.show_normals = False
        self.surface_resolution = 32
        
        # Initialize data structures
        self.points = np.array([])
        self.normals = np.array([])
        self.solid_valid = False
        self.surfaces = []
        self.surface_samples = []
        
        # Load data
        self.load_data()
        
        # Initialize polyscope
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        
    def load_data(self):
        """Load and decode data from dataset"""
        # try:
        data = self.dataset[self.sample_idx]
        
        if len(data) == 6:
            self.points, self.normals, tokens_padded, poles_padded, poles_mask, self.solid_valid = data
        else:
            print(f"Warning: Sample {self.sample_idx} has unexpected format")
            self.points = np.array([])
            self.normals = np.array([])
            self.solid_valid = False
            self.surfaces = []
            self.surface_samples = []
            return
        
        if not self.solid_valid:
            print(f"Warning: Sample {self.sample_idx} is marked as invalid! (Only showing point cloud)")
            self.surfaces = []
            self.surface_samples = []
            # Still show point cloud for invalid samples
            print(f"\n{'='*60}")
            print(f"Loaded sample {self.sample_idx}")
            print(f"Valid: {self.solid_valid}")
            print(f"Number of surfaces: 0 (invalid)")
            print(f"Number of points: {len(self.points)}")
            print(f"{'='*60}\n")
            return
        
        # Calculate actual length (excluding padding)
        pad_id = self.dataset.pad_id
        self.actual_length = self.dataset.max_tokens - (tokens_padded == pad_id).sum().item()
        
        # Truncate padded tokens to actual length
        tokens_padded = tokens_padded[:self.actual_length]
        
        if self.actual_length == 0:
            print(f"Warning: Sample {self.sample_idx} has no valid tokens")
            self.solid_valid = False
            self.surfaces = []
            self.surface_samples = []
            return
        
        # Decode tokens to surfaces
        self.tokens = tokens_padded
        self.poles = poles_padded[poles_mask]
        
        # Remove padding and start/end tokens
        self.tokens_unpadded = self.dataset.unwarp_codes(self.tokens)
        
        # Detokenize to get surface parameters
        self.surfaces = self.dataset.detokenize(self.tokens, self.poles)

        
        
        # Check if we have valid surfaces
        if len(self.surfaces) == 0:
            print(f"Warning: Sample {self.sample_idx} has no valid surfaces after detokenization")
            self.solid_valid = False
            self.surface_samples = []
            return
        
        print(f"\n{'='*60}")
        print(f"Loaded sample {self.sample_idx}")
        print(f"Valid: {self.solid_valid}")
        print(f"Number of surfaces: {len(self.surfaces)}")
        print(f"Number of points: {len(self.points)}")
        print(f"Tokens shape: {self.tokens_unpadded.shape}")
        print(f"Actual token length: {self.actual_length}")
        print(f"{'='*60}\n")
        
        # Sample all surfaces
        self.surface_samples = []
        for i, surface in enumerate(self.surfaces):
            samples = self._sample_surface(surface, self.surface_resolution)
            self.surface_samples.append(samples)
            surface_type = surface['type']
            # print(f"Surface {i:2d}: {surface_type:20s} - {samples.shape}")
                
        # except Exception as e:
        #     print(f"Error loading sample {self.sample_idx}: {e}")
        #     self.points = np.array([])
        #     self.normals = np.array([])
        #     self.solid_valid = False
        #     self.surfaces = []
        #     self.surface_samples = []
    
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
            
            # Solid/Sample selection
            psim.TextUnformatted("Solid Selection:")
            changed_solid, new_solid_idx = psim.SliderInt("Solid Index", self.sample_idx,
                                                          v_min=0, v_max=len(self.dataset)-1)
            if changed_solid:
                self.sample_idx = new_solid_idx
                self.current_surface_idx = 0  # Reset surface index
                self.load_data()
                changed = True
            
            # Resample button - reload the same index to get different augmentation
            if psim.Button("Resample (Re-augment)"):
                self.current_surface_idx = 0  # Reset surface index
                self.load_data()
                changed = True
            
            psim.TextUnformatted(f"Sample: {self.sample_idx} | Valid: {self.solid_valid}")
            psim.TextUnformatted(f"Total surfaces: {len(self.surfaces)}")
            
            # Show augmentation info
            if hasattr(self.dataset, 'rotation_augment') and self.dataset.rotation_augment:
                psim.TextUnformatted("  Rotation Augment: ON")
            if hasattr(self.dataset, 'point_augment') and self.dataset.point_augment:
                psim.TextUnformatted("  Point Augment: ON")
            
            psim.Separator()
            
            # Surface navigation (only if there are surfaces)
            if len(self.surfaces) > 0:
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
            else:
                psim.TextUnformatted("No surfaces available (invalid solid)")
            
            psim.Separator()
            
            # Display options
            psim.TextUnformatted("Display Options:")
            
            # Display mode selection (mutually exclusive) - only if surfaces exist
            if len(self.surfaces) > 0:
                changed_all, self.show_all = psim.Checkbox("Show All Surfaces", self.show_all)
                if changed_all:
                    if self.show_all:
                        self.show_cumulative = False  # Disable cumulative when showing all
                    changed = True
                
                changed_cumulative, self.show_cumulative = psim.Checkbox("Show 0 to Current (Cumulative)", self.show_cumulative)
                if changed_cumulative:
                    if self.show_cumulative:
                        self.show_all = False  # Disable show all when cumulative is enabled
                    changed = True
                
                # Show current display mode
                if self.show_all:
                    psim.TextUnformatted("  Mode: All surfaces (rainbow)")
                elif self.show_cumulative:
                    psim.TextUnformatted("  Mode: 0 to current (gradient + red)")
                else:
                    psim.TextUnformatted("  Mode: Current only (red + gray)")
            
            psim.Separator()
            
            changed_pts, self.show_points = psim.Checkbox("Show Point Cloud", self.show_points)
            if changed_pts:
                changed = True
            
            changed_norm, self.show_normals = psim.Checkbox("Show Normals", self.show_normals)
            if changed_norm:
                changed = True
            
            psim.Separator()
            
            # Resolution control (only if surfaces exist)
            if len(self.surfaces) > 0:
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
            
            # Current surface info (only if surfaces exist)
            if len(self.surfaces) > 0 and self.current_surface_idx < len(self.surfaces):
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
        elif self.show_cumulative:
            # Show surfaces from 0 to current index
            for i in range(self.current_surface_idx + 1):
                if i < len(self.surface_samples):
                    samples = self.surface_samples[i]
                    points = samples.reshape(-1, 3)
                    
                    # Highlight current surface in red, others with gradient colors
                    if i == self.current_surface_idx:
                        # Current surface in red
                        cloud = ps.register_point_cloud(f"Current_Surface_{i}", points)
                        cloud.set_radius(0.005)
                        cloud.set_color((1.0, 0.2, 0.2))
                    else:
                        # Previous surfaces with gradient colors
                        hue = i / max(1, self.current_surface_idx)
                        color = self._hsv_to_rgb(hue, 0.6, 0.8)
                        cloud = ps.register_point_cloud(f"Surface_{i:02d}", points)
                        cloud.set_radius(0.003)
                        cloud.set_color(color)
        else:
            # Show only current surface (with previous surfaces in gray)
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
    config = OmegaConf.load(args.config)
    
    # Create dataset
    dataset = load_dataset_from_config(config, section='data_train')
    print(f"Dataset size: {len(dataset)}")
    
    # Create visualizer (will load requested sample, even if invalid)
    print(f"\nVisualizing sample {args.sample_idx}")
    visualizer = SequenceVisualizer(dataset, sample_idx=args.sample_idx)
    
    # Run visualization
    visualizer.visualize()


if __name__ == '__main__':
    main()

