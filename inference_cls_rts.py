#!/usr/bin/env python3
"""
Inference script for classification and RTS (Rotation, Translation, Scaling) model.
Processes test data from .pkl files and calculates classification accuracy and RTS fitting loss for each surface type.

Usage:
$ python inference_cls_rts.py --test_data test.pkl --config config.yaml --checkpoint model.pt --output results.json
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
import os
from tqdm import tqdm
import time
from collections import defaultdict
import json
import pickle
from pathlib import Path

# Classification model related imports
from src.flow.surface_flow import Encoder
from src.utils.config import NestedDictToClass, load_config
from src.dataset.dataset_cls_rts import SurfaceClassificationAndRegressionDataset

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



class RTSClassificationInference:
    def __init__(self, config_path, checkpoint_path, use_fp16=True, num_classes=6):
        """Initialize the RTS classification inference system."""
        self.class_names = ["plane", "cylinder", "cone", "sphere", "torus", "bspline_surface", "bezier_surface"]
        self.class_mapping = CLASS_MAPPING
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.num_classes = num_classes
        
        print(f"Using device: {self.device}")
        if self.use_fp16:
            print("Using FP16 for memory efficiency")
        
        # Load model and config
        self.load_model_and_config(config_path, checkpoint_path)
        
        # Initialize loss functions
        self.loss_fn_ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.loss_fn_mse = torch.nn.MSELoss(reduction='none')
        
        # Initialize metrics tracking
        self.reset_metrics()
    
    def load_model_and_config(self, config_path, checkpoint_path):
        """Load the classification model and configuration."""
        print(f"Loading RTS classification model from config: {config_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            cfg_dict = load_config(config_path)
            self.cfg = NestedDictToClass(cfg_dict)
        except Exception as e:
            print(f"Error loading classification config: {e}")
            raise
        
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
                print("Loaded EMA model weights for RTS classification.")
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("Loaded model weights for RTS classification.")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded raw model state_dict for RTS classification.")
            
            self.model.to(self.device)
            
            if self.use_fp16:
                self.model.half()
                print("Model converted to FP16")
            
            self.model.eval()
            self.model_grid_size = self.cfg.model.res
            print(f"Model loaded successfully. Grid size: {self.model_grid_size}x{self.model_grid_size}")
            
        except Exception as e:
            print(f"Error loading RTS classification model: {e}")
            raise
    
    def reset_metrics(self):
        """Reset all metrics tracking."""
        self.predictions = []
        self.true_labels = []
        self.confidences = []
        self.all_probabilities = []
        self.processing_times = []
        
        # RTS regression metrics
        self.srt_predictions = []
        self.srt_targets = []
        self.cone_predictions = []
        self.cone_targets = []
        
        # Per-class metrics
        self.class_metrics = {i: defaultdict(list) for i in range(len(self.class_names))}
        
        # Loss tracking per class
        self.class_losses = {i: {'ce': [], 'rts': [], 'cone': []} for i in range(len(self.class_names))}
        
        # Confusion matrix
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
    
    def classify_batch(self, batch_data):
        """Classify a batch and compute RTS regression."""
        start_time = time.time()
        
        points, labels, rts, cone_min_axis, bspline_control_points, rts_mask, cone_mask = batch_data
        
        # Prepare tensors with appropriate dtype
        input_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Convert to proper dtype and move to device
        points = points.to(dtype=input_dtype, device=self.device)
        labels = labels.to(device=self.device)
        rts = rts.to(dtype=input_dtype, device=self.device)
        cone_min_axis = cone_min_axis.to(dtype=input_dtype, device=self.device)
        rts_mask = rts_mask.to(dtype=input_dtype, device=self.device)
        cone_mask = cone_mask.to(dtype=input_dtype, device=self.device)
        
        # Inference
        autocast_context = torch.cuda.amp.autocast() if self.use_fp16 else torch.no_grad()
        
        with autocast_context:
            with torch.no_grad():
                # Forward pass
                pred = self.model(points).mean(dim=1)
                
                # Split predictions
                logits = pred[..., :self.num_classes]
                srt_pred = pred[..., self.num_classes:self.num_classes+9]  # Scaling, Rotation, Translation
                
                # Normalize rotation vector
                rot_pred = srt_pred[..., 3:6]
                rot_pred = rot_pred / (torch.norm(rot_pred, dim=-1, keepdim=True) + 1e-6)
                srt_pred = torch.cat([srt_pred[..., :3], rot_pred, srt_pred[..., 6:]], dim=-1)
                
                # Get cone predictions
                cone_min_axis_pred = pred[..., self.num_classes+9:self.num_classes+10]
                
                # Get classification probabilities
                probabilities = torch.softmax(logits, dim=-1)
                confidences, predicted_classes = torch.max(probabilities, dim=-1)
                
                # Calculate losses
                loss_ce = self.loss_fn_ce(logits, labels)
                loss_mse = self.loss_fn_mse(srt_pred, rts)
                loss_mse = loss_mse * rts_mask
                loss_mse = loss_mse.mean(dim=-1)  # Mean over RTS dimensions
                
                loss_cone = self.loss_fn_mse(cone_min_axis_pred, cone_min_axis)
                loss_cone = loss_cone * cone_mask
                loss_cone = loss_cone.squeeze(-1)  # Remove last dimension
                
                # Convert to numpy
                predicted_classes = predicted_classes.cpu().numpy()
                confidences = confidences.float().cpu().numpy()
                probabilities = probabilities.float().cpu().numpy()
                labels_np = labels.cpu().numpy()
                srt_pred_np = srt_pred.float().cpu().numpy()
                rts_np = rts.float().cpu().numpy()
                cone_pred_np = cone_min_axis_pred.float().cpu().numpy()
                cone_np = cone_min_axis.float().cpu().numpy()
                loss_ce_np = loss_ce.float().cpu().numpy()
                loss_mse_np = loss_mse.float().cpu().numpy()
                loss_cone_np = loss_cone.float().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        return {
            'predicted_classes': predicted_classes,
            'confidences': confidences,
            'probabilities': probabilities,
            'labels': labels_np,
            'srt_pred': srt_pred_np,
            'rts_target': rts_np,
            'cone_pred': cone_pred_np,
            'cone_target': cone_np,
            'ce_loss': loss_ce_np,
            'rts_loss': loss_mse_np,
            'cone_loss': loss_cone_np,
            'processing_time': processing_time
        }
    
    def process_dataset(self, pkl_path, batch_size=32, num_workers=4):
        """Process entire dataset from pickle file."""
        print(f"Creating dataset from {pkl_path}")
        
        # Create dataset using the existing dataset class
        try:
            dataset = SurfaceClassificationAndRegressionDataset(
                data_path=pkl_path,
                replication=1,
                transform=None,
                is_train=False,
                res=self.model_grid_size
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        print(f"Dataset created with {len(dataset)} samples")
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=False
        )
        
        print(f"Processing {len(dataset)} samples with batch size {batch_size}...")
        
        # Reset metrics
        self.reset_metrics()
        
        # Process all batches
        total_processing_time = 0
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Classify batch
            results = self.classify_batch(batch_data)
            total_processing_time += results['processing_time']
            
            # Store results for each sample in the batch
            batch_size_actual = len(results['predicted_classes'])
            for i in range(batch_size_actual):
                pred_class = results['predicted_classes'][i]
                true_label = results['labels'][i]
                confidence = results['confidences'][i]
                probability = results['probabilities'][i]
                srt_pred = results['srt_pred'][i]
                rts_target = results['rts_target'][i]
                cone_pred = results['cone_pred'][i]
                cone_target = results['cone_target'][i]
                ce_loss = results['ce_loss'][i]
                rts_loss = results['rts_loss'][i]
                cone_loss = results['cone_loss'][i]
                
                # Store global results
                self.predictions.append(pred_class)
                self.true_labels.append(true_label)
                self.confidences.append(confidence)
                self.all_probabilities.append(probability)
                self.srt_predictions.append(srt_pred)
                self.srt_targets.append(rts_target)
                self.cone_predictions.append(cone_pred)
                self.cone_targets.append(cone_target)
                self.processing_times.append(results['processing_time'] / batch_size_actual)
                
                # Update confusion matrix
                if 0 <= true_label < self.num_classes and 0 <= pred_class < self.num_classes:
                    self.confusion_matrix[true_label, pred_class] += 1
                
                # Store per-class metrics
                if 0 <= true_label < len(self.class_names):
                    is_correct = pred_class == true_label
                    self.class_metrics[true_label]['correct'].append(is_correct)
                    self.class_metrics[true_label]['confidence'].append(confidence)
                    self.class_metrics[true_label]['true_class_prob'].append(probability[true_label])
                    
                    # Store losses per class
                    self.class_losses[true_label]['ce'].append(ce_loss)
                    self.class_losses[true_label]['rts'].append(rts_loss)
                    if true_label == 2:  # Cone class
                        self.class_losses[true_label]['cone'].append(cone_loss)
        
        print(f"Processing complete! Total time: {total_processing_time:.2f}s")
        print(f"Average batch processing time: {total_processing_time/len(dataloader):.3f}s")
        print(f"Effective samples/second: {len(dataset)/total_processing_time:.1f}")
    
    def compute_metrics(self):
        """Compute comprehensive classification and RTS regression metrics."""
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        confidences = np.array(self.confidences)
        srt_predictions = np.array(self.srt_predictions)
        srt_targets = np.array(self.srt_targets)
        
        # Overall metrics
        accuracy = np.mean(predictions == true_labels)
        avg_confidence = np.mean(confidences)
        avg_processing_time = np.mean(self.processing_times)
        
        # Overall RTS loss
        overall_rts_loss = np.mean((srt_predictions - srt_targets) ** 2)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(len(self.class_names)):
            if class_idx >= len(self.class_mapping):
                continue
                
            class_name = self.class_mapping[class_idx]
            
            # Classification metrics
            tp = np.sum((predictions == class_idx) & (true_labels == class_idx))
            fp = np.sum((predictions == class_idx) & (true_labels != class_idx))
            fn = np.sum((predictions != class_idx) & (true_labels == class_idx))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Class-specific accuracy
            class_accuracy = np.mean(self.class_metrics[class_idx]['correct']) if self.class_metrics[class_idx]['correct'] else 0.0
            class_avg_confidence = np.mean(self.class_metrics[class_idx]['confidence']) if self.class_metrics[class_idx]['confidence'] else 0.0
            
            # Loss metrics for this class
            avg_ce_loss = np.mean(self.class_losses[class_idx]['ce']) if self.class_losses[class_idx]['ce'] else 0.0
            avg_rts_loss = np.mean(self.class_losses[class_idx]['rts']) if self.class_losses[class_idx]['rts'] else 0.0
            avg_cone_loss = np.mean(self.class_losses[class_idx]['cone']) if self.class_losses[class_idx]['cone'] else 0.0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': class_accuracy,
                'avg_confidence': class_avg_confidence,
                'support': tp + fn,
                'avg_ce_loss': avg_ce_loss,
                'avg_rts_loss': avg_rts_loss,
                'avg_cone_loss': avg_cone_loss,
                'tp': tp, 'fp': fp, 'fn': fn
            }
        
        return {
            'overall': {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_processing_time': avg_processing_time,
                'overall_rts_loss': overall_rts_loss,
                'total_samples': len(predictions),
                'correct_predictions': int(np.sum(predictions == true_labels)),
                'incorrect_predictions': int(np.sum(predictions != true_labels))
            },
            'per_class': per_class_metrics,
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def print_summary(self, metrics=None):
        """Print comprehensive classification and RTS summary."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*90)
        print("RTS CLASSIFICATION INFERENCE SUMMARY")
        print("="*90)
        
        # Overall metrics
        overall = metrics['overall']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Samples: {overall['total_samples']:,}")
        print(f"  Classification Accuracy: {overall['accuracy']:.4f} ({overall['correct_predictions']}/{overall['total_samples']})")
        print(f"  Average Confidence: {overall['avg_confidence']:.4f}")
        print(f"  Overall RTS Loss (MSE): {overall['overall_rts_loss']:.6f}")
        print(f"  Average Processing Time: {overall['avg_processing_time']*1000:.2f} ms/sample")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS PERFORMANCE:")
        print(f"{'Class':<15} {'Accuracy':<9} {'Precision':<10} {'Recall':<8} {'F1':<8} {'CE Loss':<9} {'RTS Loss':<10} {'Support':<8}")
        print("-" * 95)
        
        for class_name, class_metrics in metrics['per_class'].items():
            cone_info = f" (Cone: {class_metrics['avg_cone_loss']:.4f})" if class_name == 'cone' else ""
            print(f"{class_name:<15} {class_metrics['accuracy']:<9.4f} {class_metrics['precision']:<10.4f} "
                  f"{class_metrics['recall']:<8.4f} {class_metrics['f1_score']:<8.4f} "
                  f"{class_metrics['avg_ce_loss']:<9.4f} {class_metrics['avg_rts_loss']:<10.6f} "
                  f"{class_metrics['support']:<8d}{cone_info}")
        
        # Confusion Matrix
        print(f"\n📋 CONFUSION MATRIX:")
        confusion_matrix = np.array(metrics['confusion_matrix'])
        
        # Print header
        header = "True\\Pred"
        print(f"{header:<12}", end="")
        for i in range(min(self.num_classes, len(self.class_mapping))):
            class_name = self.class_mapping[i]
            print(f"{class_name[:8]:<10}", end="")
        print()
        
        # Print matrix
        for i in range(min(self.num_classes, len(self.class_mapping))):
            class_name = self.class_mapping[i]
            print(f"{class_name[:10]:<12}", end="")
            for j in range(min(self.num_classes, len(self.class_mapping))):
                print(f"{confusion_matrix[i, j]:<10d}", end="")
            print()
        
        # Model info
        print(f"\n⚙️  MODEL INFO:")
        print(f"  Model Grid Size: {self.model_grid_size}x{self.model_grid_size}")
        print(f"  Precision: {'FP16' if self.use_fp16 else 'FP32'}")
        print(f"  Device: {self.device}")
        print(f"  Model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        print("="*90)
    
    def save_results(self, output_path, metrics=None):
        """Save detailed results to JSON file."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        # Convert numpy arrays and other non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        # Add additional information
        results = {
            'metrics': metrics,
            'model_config': {
                'grid_size': self.model_grid_size,
                'precision': 'FP16' if self.use_fp16 else 'FP32',
                'device': str(self.device),
                'num_parameters': sum(p.numel() for p in self.model.parameters()),
                'num_classes': self.num_classes
            },
            'class_names': self.class_names,
            'class_mapping': {str(k): v for k, v in self.class_mapping.items()},
            'predictions': [int(x) for x in self.predictions],
            'true_labels': [int(x) for x in self.true_labels],
            'confidences': [float(x) for x in self.confidences],
            'processing_times': [float(x) for x in self.processing_times]
        }
        
        # Convert the entire results dictionary recursively
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_to_serializable(obj)
        
        results = recursive_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")

def main():
    parser = ArgumentParser(description="Perform RTS classification inference on dataset from .pkl files.")
    parser.add_argument('--test_data', type=str, required=True, help="Path to the .pkl file containing test data.")
    parser.add_argument('--config', type=str, required=True, help="Path to the model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument('--output', type=str, default=None, help="Path to save detailed results JSON file.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16 for inference.")
    parser.add_argument('--num_classes', type=int, default=6, help="Number of classes in the model.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Initialize inference system
    use_fp16 = not args.fp32
    classifier = RTSClassificationInference(
        args.config, 
        args.checkpoint, 
        use_fp16=use_fp16, 
        num_classes=args.num_classes
    )
    
    # Process dataset
    classifier.process_dataset(args.test_data, args.batch_size, args.num_workers)
    
    # Compute and print metrics
    metrics = classifier.compute_metrics()
    classifier.print_summary(metrics)
    
    # Save results if output path provided
    if args.output:
        classifier.save_results(args.output, metrics)
    
    print(f"\nInference complete!")

if __name__ == '__main__':
    main() 