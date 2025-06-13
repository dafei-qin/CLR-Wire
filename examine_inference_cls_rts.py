#!/usr/bin/env python3
"""
Examination script for CLS-RTS (Classification + Rotation, Translation, Scaling) model.
Loads test data from .pkl files, performs inference, and visualizes both predicted and 
ground-truth reconstructed surfaces using polyscope.

Usage:
$ python examine_inference_cls_rts.py --test_data test.pkl --config config.yaml --checkpoint model.pt
"""

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import random
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pickle
from pathlib import Path
import time
from collections import defaultdict

# Model and dataset imports
from src.flow.surface_flow import Encoder
from src.utils.config import NestedDictToClass, load_config
from src.dataset.dataset_cls_rts import SurfaceClassificationAndRegressionDataset

# Surface reconstruction imports
from surface_sampler import SurfaceSampler

# Surface type mapping
CLASS_MAPPING = {
    0: "plane",
    1: "cylinder", 
    2: "cone",
    3: "sphere",
    4: "torus",
    5: "bspline_surface",
    6: "bezier_surface",
}

class CLSRTSExaminer:
    def __init__(self):
        """Initialize the CLS-RTS examination system."""
        # Class mapping and names
        self.class_mapping = CLASS_MAPPING
        self.class_names = list(CLASS_MAPPING.values())
        self.basic_surface_types = {"plane", "cylinder", "cone", "sphere", "torus"}
        
        # Model and device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = True
        self.model = None
        self.cfg = None
        self.model_grid_size = None
        
        # Surface reconstruction
        self.surface_sampler = SurfaceSampler()
        self.surface_resolution = 32
        
        # Data management
        self.dataset = None
        self.current_surfaces = []
        self.current_predictions = []
        self.current_metrics = []
        
        # UI state
        self.current_surface_index = 0
        self.num_surfaces_to_load = 10
        self.selected_class_filter = 0  # 0 = All classes
        self.show_predicted = True
        self.show_ground_truth = True
        self.show_original_points = True
        self.surface_transparency = 0.7
        self.overlay_mode = False  # If True, show at same position
        
        # Colors for different surface types
        self.surface_colors = {
            "plane": [0.8, 0.2, 0.2],        # Red
            "cylinder": [0.2, 0.8, 0.2],     # Green  
            "cone": [0.2, 0.2, 0.8],         # Blue
            "sphere": [0.8, 0.8, 0.2],       # Yellow
            "torus": [0.8, 0.2, 0.8],        # Magenta
            "bspline_surface": [0.2, 0.8, 0.8], # Cyan
            "bezier_surface": [0.8, 0.5, 0.2],  # Orange
        }
        
        # Polyscope objects
        self.polyscope_objects = []
        
    def load_model_and_config(self, config_path, checkpoint_path):
        """Load the CLS-RTS model and configuration."""
        print(f"Loading CLS-RTS model from config: {config_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            cfg_dict = load_config(config_path)
            self.cfg = NestedDictToClass(cfg_dict)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise
        
        print(f"Using device: {self.device}")
        if self.use_fp16 and self.device.type == "cuda":
            print("Using FP16 for memory efficiency")
        else:
            self.use_fp16 = False
            print("Using FP32")
        
        try:
            self.model = Encoder(
                in_dim=self.cfg.model.in_dim,
                out_dim=self.cfg.model.out_dim,
                depth=self.cfg.model.depth,
                dim=self.cfg.model.dim,
                heads=self.cfg.model.heads,
                res=self.cfg.model.res
            )
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if 'ema' in checkpoint:
                ema_model = checkpoint['ema']
                ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
                self.model.load_state_dict(ema_model, strict=False)
                print("Loaded EMA model weights")
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("Loaded model weights")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded raw model state_dict")
            
            self.model.to(self.device)
            
            if self.use_fp16:
                self.model.half()
                print("Model converted to FP16")
            
            self.model.eval()
            self.model_grid_size = self.cfg.model.res
            print(f"Model loaded successfully. Grid size: {self.model_grid_size}x{self.model_grid_size}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_dataset(self, pkl_path):
        """Load dataset from pickle file."""
        print(f"Loading dataset from {pkl_path}")
        
        try:
            self.dataset = SurfaceClassificationAndRegressionDataset(
                data_path=pkl_path,
                replication=1,
                transform=None,
                is_train=False,
                res=self.model_grid_size
            )
            print(f"Dataset loaded with {len(self.dataset)} samples")
            
            # Load some initial surfaces
            self.load_random_surfaces()
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def get_filtered_indices(self):
        """Get indices of samples matching the current class filter."""
        if self.selected_class_filter == 0:  # All classes
            return list(range(len(self.dataset)))
        else:
            # Filter by specific class
            target_class = self.selected_class_filter - 1
            filtered_indices = []
            for i in range(len(self.dataset)):
                if self.dataset.class_label[i] == target_class:
                    filtered_indices.append(i)
            return filtered_indices
    
    def load_random_surfaces(self):
        """Load random surfaces based on current filter settings."""
        print("Loading random surfaces...")
        
        # Get filtered indices
        available_indices = self.get_filtered_indices()
        if not available_indices:
            print("No surfaces match the current filter")
            return
        
        # Randomly sample indices
        num_to_sample = min(self.num_surfaces_to_load, len(available_indices))
        sampled_indices = random.sample(available_indices, num_to_sample)
        
        # Load data and perform inference
        self.current_surfaces = []
        self.current_predictions = []
        self.current_metrics = []
        
        print(f"Performing inference on {num_to_sample} surfaces...")
        
        for idx in sampled_indices:
            # Get data from dataset
            points, class_label, rts, cone_min_axis, bspline_control_points, rts_mask, cone_mask = self.dataset[idx]
            
            # Perform inference
            prediction_results = self.perform_inference(points, class_label, rts, cone_min_axis, cone_mask)
            
            # Store results
            surface_data = {
                'dataset_idx': idx,
                'points': points.numpy(),
                'true_class': class_label.item(),
                'true_rts': rts.numpy(),
                'true_cone_min_axis': cone_min_axis.numpy(),
                'rts_mask': rts_mask.numpy(),
                'cone_mask': cone_mask.numpy()
            }
            
            self.current_surfaces.append(surface_data)
            self.current_predictions.append(prediction_results)
            
            # Compute metrics
            metrics = self.compute_metrics(surface_data, prediction_results)
            self.current_metrics.append(metrics)
        
        print(f"Loaded {len(self.current_surfaces)} surfaces")
        
        # Reset to first surface
        self.current_surface_index = 0
        self.visualize_current_surface()
    
    def perform_inference(self, points, true_class, true_rts, true_cone_min_axis, cone_mask):
        """Perform inference on a single surface."""
        # Prepare input
        input_dtype = torch.float16 if self.use_fp16 else torch.float32
        points_tensor = points.unsqueeze(0).to(dtype=input_dtype, device=self.device)
        
        # Inference
        autocast_context = torch.cuda.amp.autocast() if self.use_fp16 else torch.no_grad()
        
        with autocast_context:
            with torch.no_grad():
                # Forward pass
                pred = self.model(points_tensor).mean(dim=1)
                
                # Split predictions
                logits = pred[..., :len(self.class_mapping)]
                srt_pred = pred[..., len(self.class_mapping):len(self.class_mapping)+9]
                
                # Normalize rotation vector
                rot_pred = srt_pred[..., 3:6]
                rot_pred = rot_pred / (torch.norm(rot_pred, dim=-1, keepdim=True) + 1e-6)
                srt_pred = torch.cat([srt_pred[..., :3], rot_pred, srt_pred[..., 6:]], dim=-1)
                
                # Get cone predictions
                cone_min_axis_pred = pred[..., len(self.class_mapping)+9:len(self.class_mapping)+10]
                
                # Get classification results
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
                
                # Convert to numpy
                return {
                    'predicted_class': predicted_class[0].cpu().numpy().item(),
                    'confidence': confidence[0].float().cpu().numpy().item(),
                    'probabilities': probabilities[0].float().cpu().numpy(),
                    'predicted_rts': srt_pred[0].float().cpu().numpy(),
                    'predicted_cone_min_axis': cone_min_axis_pred[0].float().cpu().numpy()
                }
    
    def compute_metrics(self, surface_data, prediction_results):
        """Compute metrics for predictions."""
        true_class = surface_data['true_class']
        pred_class = prediction_results['predicted_class']
        
        # Classification metrics
        is_correct = pred_class == true_class
        confidence = prediction_results['confidence']
        
        # RTS error (MSE)
        true_rts = surface_data['true_rts']
        pred_rts = prediction_results['predicted_rts']
        rts_mask = surface_data['rts_mask']
        
        rts_error = np.mean((pred_rts - true_rts) ** 2 * rts_mask)
        
        # Cone error (if applicable)
        cone_error = 0.0
        if surface_data['cone_mask'] > 0:
            true_cone = surface_data['true_cone_min_axis']
            pred_cone = prediction_results['predicted_cone_min_axis']
            cone_error = np.mean((pred_cone - true_cone) ** 2)
        
        return {
            'classification_correct': is_correct,
            'confidence': confidence,
            'rts_error': rts_error,
            'cone_error': cone_error,
            'true_class_name': self.class_names[true_class] if true_class < len(self.class_names) else "unknown",
            'pred_class_name': self.class_names[pred_class] if pred_class < len(self.class_names) else "unknown"
        }
    
    def create_surface_from_rts(self, surface_type, rts_params, cone_params=None):
        """Create surface data structure from RTS parameters for surface reconstruction."""
        # RTS order: [translation, rotation, scaling] (from dataset)
        # But model predicts: [scaling, rotation, translation] 
        translation = rts_params[:3]
        rotation = rts_params[3:6]
        scaling = rts_params[6:9]
        
        # Create basic surface data structure
        surface_data = {
            "type": surface_type,
            "location": [translation.tolist()],
            "direction": [rotation.tolist()],
            "scalar": scaling.tolist(),
            "converted_transformation": {
                "translation": translation.tolist(),
                "rotation": rotation.reshape(3, 1).tolist() if len(rotation.shape) == 1 else rotation.tolist(),
                "scaling": scaling.tolist()
            }
        }
        
        # Add cone-specific parameters
        if surface_type == "cone" and cone_params is not None:
            surface_data["scalar"] = [cone_params[0], scaling[0]]  # [semi_angle, radius]
        
        return surface_data
    
    def sample_surface_from_data(self, surface_data):
        """Sample points from surface data using surface sampler."""
        surface_type = surface_data["type"]
        
        if surface_type not in self.basic_surface_types:
            return None
        
        # Set parameter ranges
        num_u, num_v = self.surface_resolution, self.surface_resolution
        
        if surface_type == "plane":
            u_coords = np.linspace(-1, 1, num_u)
            v_coords = np.linspace(-1, 1, num_v)
        elif surface_type == "cylinder":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(-1, 1, num_v)
        elif surface_type == "sphere":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, np.pi, num_v)
        elif surface_type == "cone":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, 1, num_v)
        elif surface_type == "torus":
            u_coords = np.linspace(0, 2*np.pi, num_u)
            v_coords = np.linspace(0, 2*np.pi, num_v)
        else:
            return None
        
        try:
            sampled_points = self.surface_sampler.sample_surface(surface_data, u_coords, v_coords)
            return sampled_points
        except Exception as e:
            print(f"Error sampling {surface_type}: {e}")
            return None
    
    def create_surface_mesh(self, points):
        """Create mesh vertices and faces from a grid of points."""
        if points is None or len(points) == 0:
            return None, None
        
        # Handle different point array shapes
        if len(points.shape) == 3:
            height, width = points.shape[:2]
            vertices = points.reshape(-1, 3)
        elif len(points.shape) == 2 and points.shape[1] == 3:
            # Try to reshape to square grid
            total_points = len(points)
            side_length = int(np.sqrt(total_points))
            if side_length * side_length == total_points:
                height, width = side_length, side_length
                vertices = points
            else:
                return points, None  # Cannot create faces
        else:
            return None, None
        
        # Create faces
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
                
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        return vertices, np.array(faces)
    
    def clear_polyscope_objects(self):
        """Clear all polyscope objects."""
        try:
            ps.remove_all_structures()
        except:
            pass
        self.polyscope_objects = []
    
    def visualize_current_surface(self):
        """Visualize the currently selected surface."""
        if not self.current_surfaces:
            return
        
        self.clear_polyscope_objects()
        
        surface_data = self.current_surfaces[self.current_surface_index]
        prediction_results = self.current_predictions[self.current_surface_index]
        metrics = self.current_metrics[self.current_surface_index]
        
        print(f"\n=== Surface {self.current_surface_index + 1}/{len(self.current_surfaces)} ===")
        print(f"Dataset Index: {surface_data['dataset_idx']}")
        print(f"True Class: {metrics['true_class_name']}")
        print(f"Predicted Class: {metrics['pred_class_name']}")
        print(f"Classification Correct: {metrics['classification_correct']}")
        print(f"Confidence: {metrics['confidence']:.3f}")
        print(f"RTS Error: {metrics['rts_error']:.6f}")
        if metrics['cone_error'] > 0:
            print(f"Cone Error: {metrics['cone_error']:.6f}")
        
        # Visualize original points
        if self.show_original_points:
            original_points = surface_data['points'].reshape(-1, 3)
            ps_points = ps.register_point_cloud("original_points", original_points)
            ps_points.set_color([0.7, 0.7, 0.7])
            ps_points.set_radius(0.01)
            self.polyscope_objects.append("original_points")
        
        # Get surface type information
        true_class = surface_data['true_class']
        pred_class = prediction_results['predicted_class']
        true_surface_type = self.class_names[true_class] if true_class < len(self.class_names) else "unknown"
        pred_surface_type = self.class_names[pred_class] if pred_class < len(self.class_names) else "unknown"
        
        offset_x = 0.0
        
        # Visualize ground truth reconstruction
        if self.show_ground_truth and true_surface_type in self.basic_surface_types:
            true_surface_data = self.create_surface_from_rts(
                true_surface_type, 
                surface_data['true_rts'],
                surface_data['true_cone_min_axis'] if true_surface_type == "cone" else None
            )
            
            true_points = self.sample_surface_from_data(true_surface_data)
            if true_points is not None:
                vertices, faces = self.create_surface_mesh(true_points)
                if vertices is not None and faces is not None:
                    if not self.overlay_mode:
                        vertices = vertices + np.array([offset_x, 0, 0])
                        offset_x += 3.0
                    
                    ps_mesh = ps.register_surface_mesh("ground_truth", vertices, faces)
                    color = self.surface_colors.get(true_surface_type, [0.5, 0.5, 0.5])
                    ps_mesh.set_color(color)
                    ps_mesh.set_transparency(self.surface_transparency)
                    ps_mesh.set_edge_width(1.0)
                    ps_mesh.set_edge_color([0.3, 0.3, 0.3])
                    self.polyscope_objects.append("ground_truth")
        
        # Visualize predicted reconstruction
        if self.show_predicted and pred_surface_type in self.basic_surface_types:
            pred_surface_data = self.create_surface_from_rts(
                pred_surface_type,
                prediction_results['predicted_rts'],
                prediction_results['predicted_cone_min_axis'] if pred_surface_type == "cone" else None
            )
            
            pred_points = self.sample_surface_from_data(pred_surface_data)
            if pred_points is not None:
                vertices, faces = self.create_surface_mesh(pred_points)
                if vertices is not None and faces is not None:
                    if not self.overlay_mode:
                        vertices = vertices + np.array([offset_x, 0, 0])
                    
                    ps_mesh = ps.register_surface_mesh("predicted", vertices, faces)
                    color = self.surface_colors.get(pred_surface_type, [0.5, 0.5, 0.5])
                    
                    # Modify color based on correctness
                    if metrics['classification_correct']:
                        # Slightly darker for correct predictions
                        color = [c * 0.8 for c in color]
                    else:
                        # Red tint for incorrect predictions
                        color = [min(1.0, c + 0.3), c * 0.5, c * 0.5]
                    
                    ps_mesh.set_color(color)
                    ps_mesh.set_transparency(self.surface_transparency)
                    
                    if self.overlay_mode:
                        # In overlay mode, show predicted as wireframe
                        ps_mesh.set_edge_width(2.0)
                        ps_mesh.set_edge_color([0.0, 0.0, 0.0])
                    
                    self.polyscope_objects.append("predicted")
        
        # Update camera
        try:
            ps.reset_camera_to_home_view()
        except:
            pass
    
    def create_gui_callback(self):
        """Create the polyscope GUI callback."""
        def gui_callback():
            # Surface navigation
            psim.Text("Surface Navigation")
            if self.current_surfaces:
                changed, new_index = psim.SliderInt(
                    "Surface Index", 
                    self.current_surface_index, 
                    0, 
                    len(self.current_surfaces) - 1
                )
                if changed:
                    self.current_surface_index = new_index
                    self.visualize_current_surface()
                
                # Navigation buttons
                if psim.Button("Previous") and self.current_surface_index > 0:
                    self.current_surface_index -= 1
                    self.visualize_current_surface()
                psim.SameLine()
                if psim.Button("Next") and self.current_surface_index < len(self.current_surfaces) - 1:
                    self.current_surface_index += 1
                    self.visualize_current_surface()
            
            psim.Separator()
            
            # Data management
            psim.Text("Data Management")
            
            # Number of surfaces to load
            changed, self.num_surfaces_to_load = psim.SliderInt(
                "Num Surfaces", self.num_surfaces_to_load, 1, 50
            )
            
            # Class filter
            class_options = ["All Classes"] + self.class_names
            changed_class, self.selected_class_filter = psim.Combo(
                "Surface Type Filter", self.selected_class_filter, class_options
            )
            
            if psim.Button("Load New Random Surfaces") or changed_class:
                self.load_random_surfaces()
            
            # Show current surface info
            if self.current_surfaces:
                surface_data = self.current_surfaces[self.current_surface_index]
                metrics = self.current_metrics[self.current_surface_index]
                
                psim.Text(f"Surface {self.current_surface_index + 1}/{len(self.current_surfaces)}")
                psim.Text(f"Dataset Index: {surface_data['dataset_idx']}")
                
                # Classification results
                status_color = (0.2, 0.8, 0.2) if metrics['classification_correct'] else (0.8, 0.2, 0.2)
                psim.TextColored(status_color, f"True: {metrics['true_class_name']}")
                psim.TextColored(status_color, f"Pred: {metrics['pred_class_name']}")
                psim.Text(f"Confidence: {metrics['confidence']:.3f}")
                psim.Text(f"RTS Error: {metrics['rts_error']:.6f}")
                
                if metrics['cone_error'] > 0:
                    psim.Text(f"Cone Error: {metrics['cone_error']:.6f}")
            
            psim.Separator()
            
            # Visualization controls
            psim.Text("Visualization Settings")
            
            changed, self.show_ground_truth = psim.Checkbox("Show Ground Truth", self.show_ground_truth)
            if changed:
                self.visualize_current_surface()
            
            changed, self.show_predicted = psim.Checkbox("Show Predicted", self.show_predicted)
            if changed:
                self.visualize_current_surface()
            
            changed, self.show_original_points = psim.Checkbox("Show Original Points", self.show_original_points)
            if changed:
                self.visualize_current_surface()
            
            changed, self.overlay_mode = psim.Checkbox("Overlay Mode", self.overlay_mode)
            if changed:
                self.visualize_current_surface()
            
            # Surface resolution
            changed, self.surface_resolution = psim.SliderInt(
                "Surface Resolution", self.surface_resolution, 8, 64
            )
            if changed:
                self.visualize_current_surface()
            
            # Transparency
            changed, self.surface_transparency = psim.SliderFloat(
                "Surface Transparency", self.surface_transparency, 0.0, 1.0
            )
            if changed:
                self.visualize_current_surface()
            
            psim.Separator()
            
            # Statistics
            if self.current_metrics:
                psim.Text("Statistics")
                
                # Overall accuracy
                correct_count = sum(1 for m in self.current_metrics if m['classification_correct'])
                accuracy = correct_count / len(self.current_metrics)
                psim.Text(f"Classification Accuracy: {accuracy:.3f} ({correct_count}/{len(self.current_metrics)})")
                
                # Average RTS error
                avg_rts_error = np.mean([m['rts_error'] for m in self.current_metrics])
                psim.Text(f"Average RTS Error: {avg_rts_error:.6f}")
                
                # Average confidence
                avg_confidence = np.mean([m['confidence'] for m in self.current_metrics])
                psim.Text(f"Average Confidence: {avg_confidence:.3f}")
                
                # Per-class breakdown
                class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'rts_errors': []})
                for m in self.current_metrics:
                    class_name = m['true_class_name']
                    class_stats[class_name]['total'] += 1
                    if m['classification_correct']:
                        class_stats[class_name]['correct'] += 1
                    class_stats[class_name]['rts_errors'].append(m['rts_error'])
                
                psim.Text("Per-Class Stats:")
                for class_name, stats in class_stats.items():
                    if stats['total'] > 0:
                        acc = stats['correct'] / stats['total']
                        avg_rts = np.mean(stats['rts_errors'])
                        psim.Text(f"  {class_name}: {acc:.2f} acc, {avg_rts:.4f} RTS err ({stats['total']} samples)")
            
            psim.Separator()
            
            # Help
            psim.Text("Help")
            psim.Text("Ground Truth: reconstructed from true class & RTS")
            psim.Text("Predicted: reconstructed from predicted class & RTS")
            psim.Text("Original Points: input point cloud from dataset")
            psim.Text("Overlay Mode: show at same position for comparison")
            psim.Text("Separate Mode: ground truth on left, predicted on right")
            psim.Text("Green text: correct classification")
            psim.Text("Red text: incorrect classification")
        
        return gui_callback
    
    def run(self, test_data_path, config_path, checkpoint_path):
        """Main execution function."""
        # Load model and dataset
        self.load_model_and_config(config_path, checkpoint_path)
        self.load_dataset(test_data_path)
        
        # Initialize polyscope
        ps.init()
        ps.set_user_callback(self.create_gui_callback())
        
        print(f"\n🎛️ CLS-RTS Model Examination Interface")
        print(f"   - Use surface slider to navigate between surfaces")
        print(f"   - Toggle ground truth vs predicted surface display")
        print(f"   - Filter surfaces by type using the dropdown")
        print(f"   - Overlay mode shows both surfaces at same position")
        print(f"   - Original points show the input point cloud")
        print()
        
        # Show the visualization
        ps.show()


def main():
    parser = ArgumentParser(description="Examine CLS-RTS model inference with surface visualization")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test .pkl file")
    parser.add_argument("--config", type=str, required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create examiner
    examiner = CLSRTSExaminer()
    
    # Set precision mode
    if args.fp32:
        examiner.use_fp16 = False
        print("FP32 mode requested")
    
    # Run examination
    examiner.run(args.test_data, args.config, args.checkpoint)


if __name__ == "__main__":
    main() 