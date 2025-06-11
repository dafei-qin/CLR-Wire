#!/usr/bin/env python3
"""
Script to convert surface attributes from logan_process_brep_data.py output 
to scaling, rotation and translation of standard basic surfaces.

Standard surfaces:
- Plane: center at (0,0,0), normal towards z-axis
- Cylinder: center at (0,0,0), axis towards z-axis, radius = 1
- Sphere: center at (0,0,0), radius = 1
- Cone: apex at (0,0,1), base center at (0,0,0), base radius = 1, height = 1
- Torus: center at (0,0,0), main axis towards z, major_radius = 1, minor_radius = ratio
"""

import argparse
import json
import math
import numpy as np
import os
from pathlib import Path


class SurfaceTransformationConverter:
    
    def compute_rotation_matrix(self, target_direction):
        """
        Compute rotation matrix to align z-axis with target_direction.
        Returns a 3x3 rotation matrix as a list of lists.
        """
        # Normalize target direction
        target = np.array(target_direction)
        target = target / np.linalg.norm(target)
        
        # Standard z-axis
        z_axis = np.array([0, 0, 1])
        
        # If target is already aligned with z-axis
        if np.allclose(target, z_axis):
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        elif np.allclose(target, -z_axis):
            return [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        
        # Compute rotation axis (cross product)
        rotation_axis = np.cross(z_axis, target)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Compute rotation angle
        cos_angle = np.dot(z_axis, target)
        sin_angle = np.linalg.norm(np.cross(z_axis, target))
        
        # Rodrigues' rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        
        return R.tolist()

    def convert_plane(self, surface_data):
        """
        Convert plane data to transformation parameters.
        Standard plane: center at (0,0,0), normal towards z-axis
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [nx, ny, nz]
        
        # Translation: position of the plane
        translation = location
        
        # Rotation: align standard z-normal to actual normal
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # Scaling: planes don't have intrinsic scale, so uniform scaling of 1
        scaling = [1.0, 1.0, 1.0]
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
        }
        
        return result

    def convert_cylinder(self, surface_data):
        """
        Convert cylinder data to transformation parameters.
        Standard cylinder: center at (0,0,0), axis towards z-axis, radius = 1
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [dx, dy, dz]
        radius = surface_data["scalar"][0]
        
        # Translation: position of the cylinder
        translation = location
        
        # Rotation: align standard z-axis to actual axis
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # Scaling: radius affects x,y scaling, z scaling is 1 (height is arbitrary for infinite cylinder)
        scaling = [radius, radius, 1.0]
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
        }
        
        return result

    def convert_cone(self, surface_data):
        """
        Convert cone data to transformation parameters.
        Standard cone: apex at (0,0,1), base center at (0,0,0), base radius = 1, height = 1
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [dx, dy, dz]
        semi_angle = surface_data["scalar"][0]
        radius = surface_data["scalar"][1]  # reference radius
        
        # Translation: position of the cone (reference location)
        translation = location
        
        # Rotation: align standard z-axis to actual axis
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # For cone, scaling affects radius and the height through the semi-angle
        # Height scaling can be computed from semi-angle: height = radius / tan(semi_angle)
        height_scale = radius / math.tan(semi_angle) if semi_angle > 0 else 1.0
        scaling = [radius, radius, height_scale]
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [semi_angle],  # Keep semi-angle as additional parameter
        }
        
        return result

    def convert_sphere(self, surface_data):
        """
        Convert sphere data to transformation parameters.
        Standard sphere: center at (0,0,0), radius = 1
        """
        location = surface_data["location"][0]  # [x, y, z]
        radius = surface_data["scalar"][0]
        
        # Translation: center of the sphere
        translation = location
        
        # Rotation: spheres are rotationally symmetric, so identity rotation
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        # Scaling: uniform scaling by radius
        scaling = [radius, radius, radius]
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
        }
        
        return result

    def convert_torus(self, surface_data):
        """
        Convert torus data to transformation parameters.
        Standard torus: center at (0,0,0), main axis towards z, major_radius = 1, minor_radius = ratio
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [dx, dy, dz]
        major_radius = surface_data["scalar"][0]
        minor_radius = surface_data["scalar"][1]
        
        # Translation: position of the torus
        translation = location
        
        # Rotation: align standard z-axis to actual axis
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # Scaling: scale by major radius in all directions
        scaling = [major_radius, major_radius, major_radius]
        
        # Additional parameter: ratio of minor to major radius
        radius_ratio = minor_radius / major_radius if major_radius > 0 else 0.0
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [radius_ratio],
        }
        
        return result

    def convert_surface(self, surface_data):
        """
        Convert a single surface based on its type
        """
        surface_type = surface_data["type"]
        
        if surface_type == "plane":
            return self.convert_plane(surface_data)
        elif surface_type == "cylinder":
            return self.convert_cylinder(surface_data)
        elif surface_type == "cone":
            return self.convert_cone(surface_data)
        elif surface_type == "sphere":
            return self.convert_sphere(surface_data)
        elif surface_type == "torus":
            return self.convert_torus(surface_data)
        elif surface_type in ["bspline_surface", "bezier_surface"]:
            # Keep original format for these surfaces
            return surface_data
        else:
            # For other types (curves), keep original format
            return surface_data

    def convert_json_file(self, input_path, output_path=None):
        """
        Convert a single JSON file
        """
        if output_path is None:
            # Create output filename by adding '_transformed' suffix
            input_path = Path(input_path)
            output_path = input_path.parent / (input_path.stem + "_transformed" + input_path.suffix)
        
        # Load the original JSON data
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert each surface/curve in the data
        converted_data = []
        conversion_stats = {
            "plane": 0, "cylinder": 0, "cone": 0, "sphere": 0, "torus": 0,
            "other": 0, "total": len(data)
        }
        
        for item in data:
            converted_item = self.convert_surface(item)
            converted_data.append(converted_item)
            
            # Update stats
            surface_type = item["type"]
            if surface_type in conversion_stats:
                conversion_stats[surface_type] += 1
            else:
                conversion_stats["other"] += 1
        
        # Save the converted data
        with open(output_path, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"Converted {input_path} -> {output_path}")
        print(f"Conversion stats: {conversion_stats}")
        
        return output_path, conversion_stats

    def convert_directory(self, input_dir, output_dir=None):
        """
        Convert all JSON files in a directory
        """
        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / (input_dir.name + "_transformed")
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all JSON files
        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files to convert")
        
        total_stats = {
            "plane": 0, "cylinder": 0, "cone": 0, "sphere": 0, "torus": 0,
            "other": 0, "total": 0, "files": 0
        }
        
        for json_file in json_files:
            try:
                output_file = output_dir / json_file.name
                _, stats = self.convert_json_file(json_file, output_file)
                
                # Accumulate stats
                for key in total_stats:
                    if key in stats:
                        total_stats[key] += stats[key]
                total_stats["files"] += 1
                
            except Exception as e:
                print(f"Error converting {json_file}: {e}")
        
        print(f"\nTotal conversion stats: {total_stats}")


def main():
    parser = argparse.ArgumentParser(description="Convert surface attributes to transformation parameters")
    parser.add_argument("input", help="Input JSON file or directory")
    parser.add_argument("--output", "-o", help="Output file or directory (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    converter = SurfaceTransformationConverter()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Convert single file
        converter.convert_json_file(input_path, args.output)
    elif input_path.is_dir():
        # Convert directory
        converter.convert_directory(input_path, args.output)
    else:
        print(f"Error: {input_path} is not a valid file or directory")


if __name__ == "__main__":
    main() 