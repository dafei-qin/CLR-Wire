#!/usr/bin/env python3
"""
Script to sample XYZ points on basic surfaces given UV coordinates,
then apply rotation, scaling and translation transformations.

This script works with the output from convert_surface_to_transformations.py
"""

import argparse
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SurfaceSampler:
    
    def __init__(self):
        """Initialize the surface sampler."""
        pass
    
    def sample_plane(self, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
        """
        Sample points on a standard plane.
        Standard plane: z=0, extends in x-y plane
        
        Args:
            u_coords: U parameter values (corresponding to x direction)
            v_coords: V parameter values (corresponding to y direction)
            
        Returns:
            Array of shape (len(u_coords), len(v_coords), 3) with XYZ coordinates
        """
        U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
        points = np.stack([U, V, np.zeros_like(U)], axis=-1)
        return points
    
    def sample_cylinder(self, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
        """
        Sample points on a standard cylinder.
        Standard cylinder: radius=1, axis along z-axis, center at origin
        
        Args:
            u_coords: U parameter values (angle, 0 to 2*pi)
            v_coords: V parameter values (height along z-axis)
            
        Returns:
            Array of shape (len(u_coords), len(v_coords), 3) with XYZ coordinates
        """
        U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Parametric equations for cylinder
        x = np.cos(U)
        y = np.sin(U)
        z = V
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def sample_sphere(self, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
        """
        Sample points on a standard sphere.
        Standard sphere: radius=1, center at origin
        
        Args:
            u_coords: U parameter values (azimuthal angle, 0 to 2*pi)
            v_coords: V parameter values (polar angle, 0 to pi)
            
        Returns:
            Array of shape (len(u_coords), len(v_coords), 3) with XYZ coordinates
        """
        U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Parametric equations for sphere
        x = np.sin(V) * np.cos(U)
        y = np.sin(V) * np.sin(U)
        z = np.cos(V)
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def sample_cone(self, u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
        """
        Sample points on a standard cone.
        Standard cone: apex at (0,0,1), base center at (0,0,0), base radius=1, height=1
        
        Args:
            u_coords: U parameter values (angle, 0 to 2*pi)
            v_coords: V parameter values (height, 0 to 1)
            
        Returns:
            Array of shape (len(u_coords), len(v_coords), 3) with XYZ coordinates
        """
        U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Parametric equations for cone
        # Radius decreases linearly from base to apex
        radius_at_v = 1.0 - V  # radius = 1 at v=0 (base), radius = 0 at v=1 (apex)
        
        x = radius_at_v * np.cos(U)
        y = radius_at_v * np.sin(U)
        z = V
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def sample_torus(self, u_coords: np.ndarray, v_coords: np.ndarray, minor_radius_ratio: float = 0.3) -> np.ndarray:
        """
        Sample points on a standard torus.
        Standard torus: center at origin, major radius=1, minor radius=ratio*major_radius
        
        Args:
            u_coords: U parameter values (angle around the tube, 0 to 2*pi)
            v_coords: V parameter values (angle around the torus, 0 to 2*pi)
            minor_radius_ratio: Ratio of minor radius to major radius
            
        Returns:
            Array of shape (len(u_coords), len(v_coords), 3) with XYZ coordinates
        """
        U, V = np.meshgrid(u_coords, v_coords, indexing='ij')
        
        # Parametric equations for torus
        major_radius = 1.0
        minor_radius = minor_radius_ratio * major_radius
        
        x = (major_radius + minor_radius * np.cos(U)) * np.cos(V)
        y = (major_radius + minor_radius * np.cos(U)) * np.sin(V)
        z = minor_radius * np.sin(U)
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def apply_transformation(self, points: np.ndarray, transformation: Dict[str, Any]) -> np.ndarray:
        """
        Apply scaling, rotation, and translation to sampled points.
        
        Args:
            points: Array of shape (..., 3) with XYZ coordinates
            transformation: Dictionary containing transformation parameters
            
        Returns:
            Transformed points array
        """
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # Apply scaling
        scaling = np.array(transformation["scaling"])
        points_scaled = points_flat * scaling
        
        # Apply rotation
        rotation_matrix = np.array(transformation["rotation"])
        points_rotated = points_scaled @ rotation_matrix.T
        
        # Apply translation
        translation = np.array(transformation["translation"])
        points_final = points_rotated + translation
        
        return points_final.reshape(original_shape)
    
    def sample_surface(self, surface_data: Dict[str, Any], u_coords: np.ndarray, v_coords: np.ndarray) -> np.ndarray:
        """
        Sample points on a surface given its transformation data.
        
        Args:
            surface_data: Surface data with type and transformation parameters
            u_coords: U parameter values
            v_coords: V parameter values
            
        Returns:
            Array of transformed XYZ points
        """
        surface_type = surface_data["type"]
        
        # Sample points on the standard surface
        if surface_type == "plane":
            points = self.sample_plane(u_coords, v_coords)
        elif surface_type == "cylinder":
            points = self.sample_cylinder(u_coords, v_coords)
        elif surface_type == "sphere":
            points = self.sample_sphere(u_coords, v_coords)
        elif surface_type == "cone":
            points = self.sample_cone(u_coords, v_coords)
        elif surface_type == "torus":
            # Get minor radius ratio from extra_params in converted_transformation
            transformation = surface_data.get("converted_transformation", {})
            minor_radius_ratio = transformation.get("extra_params", [0.3])[0] if transformation.get("extra_params") else 0.3
            points = self.sample_torus(u_coords, v_coords, minor_radius_ratio)
        else:
            raise ValueError(f"Unsupported surface type: {surface_type}")
        
        # Apply transformation if available
        if "converted_transformation" in surface_data:
            transformed_points = self.apply_transformation(points, surface_data["converted_transformation"])
        else:
            # Fallback for old format compatibility
            transformed_points = self.apply_transformation(points, surface_data)
        
        return transformed_points
    
    def sample_surfaces_from_json(self, json_path: str, num_u: int = 50, num_v: int = 50) -> List[np.ndarray]:
        """
        Sample points from all surfaces in a JSON file.
        
        Args:
            json_path: Path to the JSON file with surface data
            num_u: Number of U samples
            num_v: Number of V samples
            
        Returns:
            List of point arrays, one for each surface
        """
        with open(json_path, 'r') as f:
            surfaces_data = json.load(f)
        
        all_sampled_points = []
        
        for surface_data in surfaces_data:
            surface_type = surface_data["type"]
            
            # Set appropriate parameter ranges for each surface type
            if surface_type == "plane":
                u_coords = np.linspace(-2, 2, num_u)  # Extend plane in both directions
                v_coords = np.linspace(-2, 2, num_v)
            elif surface_type == "cylinder":
                u_coords = np.linspace(0, 2*np.pi, num_u)  # Full rotation
                v_coords = np.linspace(-1, 1, num_v)  # Height range
            elif surface_type == "sphere":
                u_coords = np.linspace(0, 2*np.pi, num_u)  # Azimuthal angle
                v_coords = np.linspace(0, np.pi, num_v)  # Polar angle
            elif surface_type == "cone":
                u_coords = np.linspace(0, 2*np.pi, num_u)  # Full rotation
                v_coords = np.linspace(0, 1, num_v)  # Height from base to apex
            elif surface_type == "torus":
                u_coords = np.linspace(0, 2*np.pi, num_u)  # Around tube
                v_coords = np.linspace(0, 2*np.pi, num_v)  # Around torus
            else:
                # Skip non-basic surfaces
                continue
            
            try:
                points = self.sample_surface(surface_data, u_coords, v_coords)
                all_sampled_points.append(points)
            except Exception as e:
                print(f"Error sampling surface {surface_data.get('idx', 'unknown')}: {e}")
                continue
        
        return all_sampled_points
    
    def visualize_surfaces(self, sampled_points: List[np.ndarray], title: str = "Sampled Surfaces"):
        """
        Visualize sampled surface points in 3D.
        
        Args:
            sampled_points: List of point arrays
            title: Plot title
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, points in enumerate(sampled_points):
            if len(points.shape) == 3:  # Grid of points
                # Plot surface
                x, y, z = points[:, :, 0], points[:, :, 1], points[:, :, 2]
                ax.plot_surface(x, y, z, alpha=0.7, color=colors[i % len(colors)])
            else:  # Flattened points
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=colors[i % len(colors)], alpha=0.6, s=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = 0
        for points in sampled_points:
            if len(points.shape) == 3:
                points_flat = points.reshape(-1, 3)
            else:
                points_flat = points
            current_range = max(np.ptp(points_flat[:, 0]), np.ptp(points_flat[:, 1]), np.ptp(points_flat[:, 2]))
            max_range = max(max_range, current_range)
        
        ax.set_xlim([-max_range/2, max_range/2])
        ax.set_ylim([-max_range/2, max_range/2])
        ax.set_zlim([-max_range/2, max_range/2])
        
        plt.show()
    
    def save_sampled_points(self, sampled_points: List[np.ndarray], output_path: str):
        """
        Save sampled points to a numpy file.
        
        Args:
            sampled_points: List of point arrays
            output_path: Output file path
        """
        all_points = []
        for i, points in enumerate(sampled_points):
            if len(points.shape) == 3:
                points_flat = points.reshape(-1, 3)
            else:
                points_flat = points
            
            # Add surface index as a fourth column
            surface_indices = np.full((points_flat.shape[0], 1), i)
            points_with_index = np.hstack([points_flat, surface_indices])
            all_points.append(points_with_index)
        
        if all_points:
            combined_points = np.vstack(all_points)
            np.save(output_path, combined_points)
            print(f"Saved {combined_points.shape[0]} points to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample points on basic surfaces with transformations")
    parser.add_argument("input", help="Input JSON file or directory with transformed surface data")
    parser.add_argument("--output", "-o", help="Output directory for sampled points")
    parser.add_argument("--num_u", type=int, default=50, help="Number of U parameter samples")
    parser.add_argument("--num_v", type=int, default=50, help="Number of V parameter samples")
    parser.add_argument("--visualize", "-v", action="store_true", help="Visualize the sampled surfaces")
    parser.add_argument("--save_points", "-s", action="store_true", help="Save sampled points to .npy files")
    
    args = parser.parse_args()
    
    sampler = SurfaceSampler()
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single file
        print(f"Sampling surfaces from {input_path}")
        sampled_points = sampler.sample_surfaces_from_json(
            str(input_path), num_u=args.num_u, num_v=args.num_v
        )
        
        if args.visualize:
            sampler.visualize_surfaces(sampled_points, f"Surfaces from {input_path.name}")
        
        if args.save_points and args.output:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            output_file = output_path / (input_path.stem + "_sampled_points.npy")
            sampler.save_sampled_points(sampled_points, str(output_file))
    
    elif input_path.is_dir():
        # Process directory
        json_files = list(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        for json_file in json_files:
            print(f"Processing {json_file}")
            try:
                sampled_points = sampler.sample_surfaces_from_json(
                    str(json_file), num_u=args.num_u, num_v=args.num_v
                )
                
                if args.save_points and args.output:
                    output_file = output_dir / (json_file.stem + "_sampled_points.npy")
                    sampler.save_sampled_points(sampled_points, str(output_file))
                
                if args.visualize:
                    sampler.visualize_surfaces(sampled_points, f"Surfaces from {json_file.name}")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
    
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main() 