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

    def calculate_aabb_from_points(self, points):
        """
        Calculate axis-aligned bounding box from a set of points.
        
        Args:
            points (np.ndarray): Array of points with shape (n, 3)
            
        Returns:
            tuple: (min_bounds, max_bounds) where each is [x, y, z]
        """
        if len(points) == 0:
            return None, None
        
        min_bounds = np.min(points, axis=0).tolist()
        max_bounds = np.max(points, axis=0).tolist()
        
        return min_bounds, max_bounds

    def transform_points_to_standard_space(self, points, translation, rotation, scaling):
        """
        Transform points from world space back to standard surface space.
        This is the inverse of the transformation pipeline.
        
        Args:
            points (np.ndarray): Points in world space
            translation (list): Translation vector
            rotation (list): Rotation matrix (3x3)
            scaling (list): Scaling factors [sx, sy, sz]
            
        Returns:
            np.ndarray: Points in standard space
        """
        # Convert to numpy arrays
        points = np.array(points)
        translation = np.array(translation)
        rotation = np.array(rotation)
        scaling = np.array(scaling)
        
        # Inverse transformation pipeline:
        # 1. Remove translation
        points_translated = points - translation
        
        # 2. Remove rotation (apply inverse rotation)
        points_rotated = points_translated @ rotation  # Transpose is inverse for rotation matrix
        
        # 3. Remove scaling
        points_standard = points_rotated / scaling
        
        return points_standard

    def calculate_standard_space_aabb(self, surface_data, transformation):
        """
        Calculate AABB in standard surface space for cut surface representation.
        
        Args:
            surface_data (dict): Surface data containing points
            transformation (dict): Transformation parameters (translation, rotation, scaling)
            
        Returns:
            tuple: (min_bounds, max_bounds) in standard space, or (None, None) if no points
        """
        points = surface_data.get("points")
        if points is None or len(points) == 0:
            return None, None
        
        # Convert points to numpy array
        points_array = np.array(points)
        if len(points_array.shape) == 3 and points_array.shape[2] == 3:
            points_flat = points_array.reshape(-1, 3)
        elif len(points_array.shape) == 2 and points_array.shape[1] == 3:
            points_flat = points_array
        else:
            return None, None
        
        # Transform points to standard space
        standard_points = self.transform_points_to_standard_space(
            points_flat,
            transformation["translation"],
            transformation["rotation"],
            transformation["scaling"]
        )
        
        # Calculate AABB in standard space
        return self.calculate_aabb_from_points(standard_points)

    def convert_aabb_to_uv_space(self, aabb_min, aabb_max, surface_type):
        """
        Convert AABB from standard 3D space to UV parameter space.
        
        Args:
            aabb_min (list): Minimum bounds in standard 3D space
            aabb_max (list): Maximum bounds in standard 3D space
            surface_type (str): Type of surface ('plane', 'cylinder', 'sphere', 'cone', 'torus')
            
        Returns:
            tuple: (uv_min, uv_max) where each is [u_min, v_min] and [u_max, v_max]
        """
        if aabb_min is None or aabb_max is None:
            return None, None
        
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)
        
        if surface_type == "plane":
            # For plane: U maps to X, V maps to Y directly
            uv_min = [aabb_min[0], aabb_min[1]]
            uv_max = [aabb_max[0], aabb_max[1]]
            
        elif surface_type == "cylinder":
            # For cylinder: U is angle (0 to 2π), V is height (Z coordinate)
            # U covers full circle unless we have angular constraints
            uv_min = [0.0, aabb_min[2]]  # Full angle, min height
            uv_max = [2*np.pi, aabb_max[2]]  # Full angle, max height
            
        elif surface_type == "sphere":
            # For sphere: U is azimuthal (0 to 2π), V is polar (0 to π)
            # Convert AABB to spherical coordinates
            # This is more complex - we need to find the angular ranges
            
            # For now, use conservative bounds based on the AABB
            # Z coordinate maps to polar angle: z = cos(v), so v = arccos(z)
            z_min, z_max = max(aabb_min[2], -1.0), min(aabb_max[2], 1.0)
            v_max = np.arccos(z_min) if z_min <= 1.0 else np.pi
            v_min = np.arccos(z_max) if z_max >= -1.0 else 0.0
            
            # For azimuthal angle, if we span most of the XY plane, use full range
            xy_extent = max(aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1])
            if xy_extent > 1.8:  # Close to full diameter
                uv_min = [0.0, v_min]
                uv_max = [2*np.pi, v_max]
            else:
                # Conservative estimate - this could be improved with more sophisticated analysis
                u_center = np.arctan2(0.5*(aabb_min[1] + aabb_max[1]), 0.5*(aabb_min[0] + aabb_max[0]))
                u_range = xy_extent * 0.5  # Rough estimate
                uv_min = [max(0.0, u_center - u_range), v_min]
                uv_max = [min(2*np.pi, u_center + u_range), v_max]
            
        elif surface_type == "cone":
            # For cone: U is angle (0 to 2π), V is height (0 to 1)
            # Map Z coordinate to V parameter: v = z (since standard cone goes from 0 to 1)
            v_min = max(0.0, aabb_min[2])
            v_max = min(1.0, aabb_max[2])
            uv_min = [0.0, v_min]
            uv_max = [2*np.pi, v_max]
            
        elif surface_type == "torus":
            # For torus: U is around tube (0 to 2π), V is around torus (0 to 2π)
            # Both are angular parameters - use full range unless we have specific constraints
            uv_min = [0.0, 0.0]
            uv_max = [2*np.pi, 2*np.pi]
            
        else:
            # Unknown surface type
            return None, None
        
        return uv_min, uv_max

    def calculate_uv_space_aabb(self, surface_data, transformation):
        """
        Calculate AABB in UV parameter space for surface sampling.
        
        Args:
            surface_data (dict): Surface data containing points
            transformation (dict): Transformation parameters
            
        Returns:
            tuple: (uv_min, uv_max) for UV parameter bounds, or (None, None) if unavailable
        """
        # First get AABB in standard 3D space
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation)
        
        if aabb_min is None or aabb_max is None:
            return None, None
        
        # Convert to UV space
        surface_type = surface_data.get("type")
        return self.convert_aabb_to_uv_space(aabb_min, aabb_max, surface_type)

    def add_aabb_to_transformation(self, surface_data, transformation):
        """
        Helper method to add both 3D and UV AABB to transformation data.
        
        Args:
            surface_data (dict): Surface data containing points
            transformation (dict): Transformation parameters (will be modified)
            
        Returns:
            dict: Transformation with added AABB information
        """
        # Calculate both 3D and UV AABB
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation)
        uv_min, uv_max = self.calculate_uv_space_aabb(surface_data, transformation)
        
        # Add to transformation
        transformation["aabb_min"] = aabb_min
        transformation["aabb_max"] = aabb_max
        transformation["uv_min"] = uv_min
        transformation["uv_max"] = uv_max
        
        return transformation

    def convert_plane(self, surface_data):
        """
        Convert plane data to transformation parameters.
        Standard plane: center at (0,0,0), normal towards z-axis
        For planes, we calculate scaling to match the minimal enclosing rectangle.
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [nx, ny, nz]
        
        # Translation: position of the plane
        translation = location
        
        # Rotation: align standard z-normal to actual normal
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # Calculate scaling based on actual points if available
        points = surface_data.get("points")
        if points is not None and len(points) > 0:
            # Convert points to numpy array
            points_array = np.array(points)
            if len(points_array.shape) == 3 and points_array.shape[2] == 3:
                points_flat = points_array.reshape(-1, 3)
            elif len(points_array.shape) == 2 and points_array.shape[1] == 3:
                points_flat = points_array
            else:
                points_flat = None
            
            if points_flat is not None and len(points_flat) > 0:
                # Transform points to standard plane space (z=0)
                # First create a temporary transformation with unit scaling
                temp_transformation = {
                    "translation": translation,
                    "rotation": rotation_matrix,
                    "scaling": [1.0, 1.0, 1.0]
                }
                
                standard_points = self.transform_points_to_standard_space(
                    points_flat, translation, rotation_matrix, [1.0, 1.0, 1.0]
                )
                
                # Calculate the minimal enclosing rectangle in the x-y plane
                min_x, max_x = np.min(standard_points[:, 0]), np.max(standard_points[:, 0])
                min_y, max_y = np.min(standard_points[:, 1]), np.max(standard_points[:, 1])
                
                # Calculate scaling to match the actual extent
                # Standard plane extends from -1 to 1 in both x and y directions
                width = max_x - min_x
                height = max_y - min_y
                
                scale_x = width / 2.0 if width > 0 else 1.0
                scale_y = height / 2.0 if height > 0 else 1.0
                scaling = [scale_x, scale_y, 1.0]  # z-scaling is 1 for planes
                
                # For planes, AABB in standard space corresponds to the actual point distribution
                # Transform points to standard space to get actual bounds
                centered_x = (min_x + max_x) / 2.0
                centered_y = (min_y + max_y) / 2.0
                
                # In standard space, the points should be centered and scaled
                # The actual UV bounds should correspond to the normalized point range
                # Since scaling is applied, we need to map back to the [-1, 1] range
                uv_min_x = -1.0  # Always full range since scaling adjusts for width
                uv_max_x = 1.0
                uv_min_y = -1.0  # Always full range since scaling adjusts for height  
                uv_max_y = 1.0
                
                # However, if we want to constraint sampling to actual point bounds,
                # we should use the actual point distribution in the standard coordinate system
                # Standard space bounds: normalize the original extent to [-1, 1] * scaling
                standard_min_x = (min_x - centered_x) / (width / 2.0) if width > 0 else -1.0
                standard_max_x = (max_x - centered_x) / (width / 2.0) if width > 0 else 1.0
                standard_min_y = (min_y - centered_y) / (height / 2.0) if height > 0 else -1.0
                standard_max_y = (max_y - centered_y) / (height / 2.0) if height > 0 else 1.0
                
                aabb_min = [standard_min_x, standard_min_y, -0.001]
                aabb_max = [standard_max_x, standard_max_y, 0.001]
                
                # UV bounds for plane are the same as XY bounds in standard space
                uv_min = [standard_min_x, standard_min_y]
                uv_max = [standard_max_x, standard_max_y]
                
                print(f"Plane analysis:")
                print(f"  Point extent: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")
                print(f"  Calculated scaling: [{scale_x:.3f}, {scale_y:.3f}, 1.0]")
            else:
                # Fallback to default scaling
                scaling = [1.0, 1.0, 1.0]
                aabb_min = [-1.0, -1.0, -0.001]
                aabb_max = [1.0, 1.0, 0.001]
                uv_min = [aabb_min[0], aabb_min[1]]
                uv_max = [aabb_max[0], aabb_max[1]]
        else:
            # No points available, use default scaling
            scaling = [1.0, 1.0, 1.0]
            aabb_min = [-1.0, -1.0, -0.001]
            aabb_max = [1.0, 1.0, 0.001]
            uv_min = [aabb_min[0], aabb_min[1]]
            uv_max = [aabb_max[0], aabb_max[1]]
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
            "uv_min": uv_min,
            "uv_max": uv_max,
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
        direction = [direction[0], direction[1], direction[2]] 
        
        # Translation: position of the cylinder
        translation = location
        
        # Rotation: align standard z-axis to actual axis
        rotation_matrix = self.compute_rotation_matrix(direction)
        
        # Scaling: radius affects x,y scaling, z scaling is 1 (height is arbitrary for infinite cylinder)
        scaling = [radius, radius, 1.0]
        
        # Calculate AABB in standard space if points are available
        transformation_for_aabb = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling
        }
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation_for_aabb)
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
        }
        
        return result

    def analyze_cylinder_from_points(self, surface_data):
        """
        Analyze actual surface points to determine better cylinder transformation.
        This provides improved scale and translation by analyzing the actual point distribution.
        
        The method:
        1. Projects all surface points onto the cylinder axis to determine actual height
        2. Calculates the true center of the cylinder along its axis
        3. Estimates the actual radius from point-to-axis distances
        4. Updates scaling to match the actual height (z-scaling = actual_height / 2)
        5. Updates translation to use the calculated center
        
        Args:
            surface_data (dict): Surface data containing points, location, direction, radius
            
        Returns:
            dict: Updated surface data with improved transformation parameters
        """
        # Get the basic conversion first as fallback
        basic_result = self.convert_cylinder(surface_data)
        
        # Check if we have actual surface points to analyze
        points = surface_data.get("points")
        if points is None or len(points) == 0:
            print("Warning: No surface points available for cylinder analysis, using basic conversion")
            return basic_result
        
        try:
            # Convert points to numpy array
            points_array = np.array(points)
            if len(points_array.shape) == 3 and points_array.shape[2] == 3:
                # Reshape from (height, width, 3) to (n_points, 3)
                points_flat = points_array.reshape(-1, 3)
            elif len(points_array.shape) == 2 and points_array.shape[1] == 3:
                points_flat = points_array
            else:
                print(f"Warning: Unexpected points shape {points_array.shape}, using basic conversion")
                return basic_result
            
            # Validate we have enough points
            if len(points_flat) < 4:
                print("Warning: Too few points for reliable analysis, using basic conversion")
                return basic_result
            
            # Get cylinder parameters
            location = np.array(surface_data["location"][0])
            direction = np.array(surface_data["direction"][0])
            radius = surface_data["scalar"][0]
            
            # Normalize the direction vector
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                print("Warning: Invalid direction vector, using basic conversion")
                return basic_result
            direction = direction / direction_norm
            
            # Project all points onto the cylinder axis to find height range
            # Vector from cylinder location to each point
            vectors_to_points = points_flat - location
            
            # Project onto the cylinder axis direction
            projections = np.dot(vectors_to_points, direction)
            
            # Find the range of projections (this gives us the height of the actual cylinder)
            min_projection = np.min(projections)
            max_projection = np.max(projections)
            actual_height = max_projection - min_projection
            
            # Validate height calculation
            if actual_height < 1e-6:
                print("Warning: Calculated height is too small, using basic conversion")
                return basic_result
            
            # Find the center point along the axis
            center_projection = (min_projection + max_projection) / 2
            
            # Calculate the actual center of the cylinder
            actual_center = location + center_projection * direction
            
            # Calculate actual radius from points
            # Distance from each point to the cylinder axis
            axis_projections = np.outer(projections, direction)
            closest_points_on_axis = location + axis_projections
            radial_vectors = points_flat - closest_points_on_axis
            radial_distances = np.linalg.norm(radial_vectors, axis=1)
            
            # Use median for more robust radius estimation
            actual_radius_estimate = np.median(radial_distances)
            radius_std = np.std(radial_distances)
            
            # Validate radius calculation
            if actual_radius_estimate < 1e-6:
                print("Warning: Calculated radius is too small, using basic conversion")
                return basic_result
            
            print(f"Cylinder analysis:")
            print(f"  Given radius: {radius:.4f}, Estimated from points: {actual_radius_estimate:.4f} (±{radius_std:.4f})")
            print(f"  Actual height from points: {actual_height:.4f}")
            print(f"  Height range: [{min_projection:.4f}, {max_projection:.4f}]")
            print(f"  Original center: {location}")
            print(f"  Calculated center: {actual_center}")
            
            # Decide whether to use given or estimated radius
            # If they're close, use the given radius (more reliable)
            # If they differ significantly, warn and use estimated
            radius_ratio = abs(actual_radius_estimate - radius) / max(radius, actual_radius_estimate)
            if radius_ratio > 0.1:  # More than 10% difference
                print(f"  Warning: Large radius discrepancy ({radius_ratio:.1%}), using estimated radius")
                improved_radius = actual_radius_estimate
            else:
                improved_radius = radius
            
            # Update transformation parameters
            # Use the calculated center as translation
            improved_translation = actual_center.tolist()
            
            # Set height scaling to match the actual height
            # Since standard cylinder in our sampler uses v_coords range [-1, 1] = height 2
            # Scale factor should be actual_height / 2
            height_scaling = actual_height / 2.0
            
            improved_scaling = [improved_radius, improved_radius, height_scaling]
            
            # Create the improved transformation
            improved_transformation = {
                "translation": improved_translation,
                "rotation": basic_result["converted_transformation"]["rotation"],
                "scaling": improved_scaling,
                "extra_params": [actual_height, actual_radius_estimate, radius_std],  # Store additional info
            }
            
            # Add AABB information
            improved_transformation = self.add_aabb_to_transformation(surface_data, improved_transformation)
            
            # Create the improved result
            improved_result = surface_data.copy()
            improved_result["converted_transformation"] = improved_transformation
            
            return improved_result
            
        except Exception as e:
            print(f"Error in cylinder analysis: {e}")
            print("Falling back to basic conversion")
            return basic_result

    def convert_cone(self, surface_data):
        """
        Convert cone data to transformation parameters.
        Standard cone: apex at (0,0,1), base center at (0,0,0), base radius = 1, height = 1
        """
        location = surface_data["location"][0]  # [x, y, z]
        direction = surface_data["direction"][0]  # [dx, dy, dz]
        print('cone direction', direction)
        direction = [direction[0], -direction[1], direction[2]] 
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
        
        # Calculate AABB in standard space if points are available
        transformation_for_aabb = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling
        }
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation_for_aabb)
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [semi_angle],  # Keep semi-angle as additional parameter
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
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
        
        # Calculate AABB in standard space if points are available
        transformation_for_aabb = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling
        }
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation_for_aabb)
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [],
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
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
        
        # Calculate AABB in standard space if points are available
        transformation_for_aabb = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling
        }
        aabb_min, aabb_max = self.calculate_standard_space_aabb(surface_data, transformation_for_aabb)
        
        # Create a copy of the original data and add transformation
        result = surface_data.copy()
        result["converted_transformation"] = {
            "translation": translation,
            "rotation": rotation_matrix,
            "scaling": scaling,
            "extra_params": [radius_ratio],
            "aabb_min": aabb_min,
            "aabb_max": aabb_max,
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
            return self.analyze_cylinder_from_points(surface_data)
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