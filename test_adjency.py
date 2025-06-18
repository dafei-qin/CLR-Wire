import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
from tqdm import tqdm
import time
from collections import defaultdict
import json
import pickle

# Adjacency model related imports
from src.flow.adjacency import AdjacencyDecoder
from src.utils.config import NestedDictToClass, load_config
from src.dataset.dataset_adj import SurfaceClassificationAndRegressionDataset

class AdjacencyInference:
    def __init__(self, config_path, checkpoint_path, use_fp16=True):
        """Initialize the adjacency inference system."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        print(f"Using device: {self.device}")
        if self.use_fp16:
            print("Using FP16 for memory efficiency")
        
        # Load model and config
        self.load_model_and_config(config_path, checkpoint_path)
        
        # Initialize metrics tracking
        self.reset_metrics()
    
    def load_model_and_config(self, config_path, checkpoint_path):
        """Load the adjacency model and configuration."""
        print(f"Loading adjacency model from config: {config_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            cfg_dict = load_config(config_path)
            self.cfg = NestedDictToClass(cfg_dict)
        except Exception as e:
            print(f"Error loading adjacency config: {e}")
            raise
        
        try:
            self.model = AdjacencyDecoder(
                depth=self.cfg.model.depth,
                heads=self.cfg.model.heads,
                surface_res=self.cfg.model.surface_res,
                num_types=self.cfg.model.num_types,
                num_nearby=self.cfg.model.num_nearby,
                surface_dim=self.cfg.model.surface_dim,
                surface_enc_block_out_channels=self.cfg.model.surface_enc_block_out_channels
            )
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if 'ema' in checkpoint:
                ema_model = checkpoint['ema']
                ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
                self.model.load_state_dict(ema_model, strict=False)
                print("Loaded EMA model weights for adjacency.")
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("Loaded model weights for adjacency.")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded raw model state_dict for adjacency.")
            
            self.model.to(self.device)
            
            if self.use_fp16:
                self.model.half()
                print("Model converted to FP16")
            
            self.model.eval()
            print(f"Model loaded successfully. Surface resolution: {self.cfg.model.surface_res}x{self.cfg.model.surface_res}")
            
        except Exception as e:
            print(f"Error loading adjacency model: {e}")
            raise
    
    def reset_metrics(self):
        """Reset all metrics tracking."""
        self.predictions = []
        self.true_labels = []
        self.probabilities = []
        self.processing_times = []
        self.surface_types = []  # Track surface types for per-type analysis
        
        # Binary classification metrics
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
    
    def predict_batch(self, batch_data):
        """Predict adjacency for a batch of surface sets."""
        start_time = time.time()
        
        # Extract data from batch
        points = batch_data['points']
        surface_type = batch_data['type']
        bbox = batch_data['bbox']
        adj_mask = batch_data['adj_mask']
        nearby_type = batch_data['nearby_type']
        nearby_points = batch_data['nearby_points']
        nearby_bbox = batch_data['nearby_bbox']
        padding_mask = batch_data['padding_mask']
        
        # Prepare input tensors
        input_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Convert to proper dtype and move to device
        points = points.to(dtype=input_dtype, device=self.device)
        surface_type = surface_type.to(device=self.device)
        bbox = bbox.to(dtype=input_dtype, device=self.device)
        adj_mask = adj_mask.to(device=self.device)
        nearby_type = nearby_type.to(device=self.device)
        nearby_points = nearby_points.to(dtype=input_dtype, device=self.device)
        nearby_bbox = nearby_bbox.to(dtype=input_dtype, device=self.device)
        padding_mask = padding_mask.to(device=self.device)
        
        # Prepare input for model
        x = torch.cat([points.unsqueeze(1), nearby_points], dim=1)
        padding_mask_full = torch.cat([torch.zeros(padding_mask.shape[0], 1, device=padding_mask.device), padding_mask], dim=1)
        type_full = torch.cat([surface_type, nearby_type], dim=1)
        
        # Inference
        autocast_context = torch.cuda.amp.autocast() if self.use_fp16 else torch.no_grad()
        
        with autocast_context:
            with torch.no_grad():
                adjacency_logits = self.model(x, padding_mask_full, type_full)
                
                # Convert to probabilities using sigmoid
                probabilities = torch.sigmoid(adjacency_logits)
                
                # Get predictions (threshold at 0.5)
                predictions = (probabilities > 0.5).float()
                
                # Convert to numpy
                predictions = predictions.cpu().numpy()
                probabilities = probabilities.float().cpu().numpy()
                adj_mask_np = adj_mask.float().cpu().numpy()
                surface_type_np = surface_type.cpu().numpy()
        
        processing_time = time.time() - start_time
        
        return predictions, probabilities, adj_mask_np, surface_type_np, processing_time
    
    def process_dataset(self, data_path, data_dir, batch_size=32, num_workers=4):
        """Process entire dataset using DataLoader for efficient batch processing."""
        print(f"Creating dataset from {data_path} with data directory {data_dir}")
        
        # Create dataset
        try:
            dataset = SurfaceClassificationAndRegressionDataset(
                data_path=data_path,
                data_dir=data_dir,
                replication=1,
                transform=None,
                is_train=False,
                res=self.cfg.model.surface_res,
                num_nearby=self.cfg.model.num_nearby
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
        total_samples = 0
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Predict batch
            pred_batch, prob_batch, true_batch, surface_type_batch, proc_time = self.predict_batch(batch_data)
            total_processing_time += proc_time
            
            # Store results for each sample in the batch
            batch_size_actual = pred_batch.shape[0]
            total_samples += batch_size_actual
            
            for i in range(batch_size_actual):
                predictions = pred_batch[i]  # Shape: (num_nearby,)
                probabilities = prob_batch[i]  # Shape: (num_nearby,)
                true_labels = true_batch[i]  # Shape: (num_nearby,)
                surface_type = surface_type_batch[i]  # Scalar value for main surface type
                
                # Store results
                self.predictions.extend(predictions.tolist())
                self.true_labels.extend(true_labels.tolist())
                self.probabilities.extend(probabilities.tolist())
                # Store surface type for each nearby surface prediction
                self.surface_types.extend([surface_type] * len(predictions))
                
                # Update confusion matrix metrics
                for j in range(len(predictions)):
                    pred = predictions[j]
                    true = true_labels[j]
                    
                    if pred == 1 and true == 1:
                        self.tp += 1
                    elif pred == 1 and true == 0:
                        self.fp += 1
                    elif pred == 0 and true == 1:
                        self.fn += 1
                    else:  # pred == 0 and true == 0
                        self.tn += 1
                
                # Approximate per-sample processing time
                self.processing_times.append(proc_time / batch_size_actual)
        
        print(f"Processed {total_samples} samples in {total_processing_time:.2f} seconds")
        return total_samples
    
    def compute_metrics(self):
        """Compute comprehensive adjacency prediction metrics."""
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        probabilities = np.array(self.probabilities)
        processing_times = np.array(self.processing_times)
        surface_types = np.array(self.surface_types)
        
        # Basic metrics
        accuracy = np.mean(predictions == true_labels)
        avg_probability = np.mean(probabilities)
        avg_processing_time = np.mean(processing_times)
        
        # Precision, Recall, F1
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity (True Negative Rate)
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        
        # Balanced accuracy
        sensitivity = recall  # Same as recall
        balanced_accuracy = (sensitivity + specificity) / 2
        
        # ROC AUC would require different thresholds, but we can compute basic stats
        positive_samples = np.sum(true_labels == 1)
        negative_samples = np.sum(true_labels == 0)
        
        # Confidence analysis
        correct_mask = predictions == true_labels
        correct_probabilities = probabilities[correct_mask]
        incorrect_probabilities = probabilities[~correct_mask]
        
        avg_correct_prob = np.mean(correct_probabilities) if len(correct_probabilities) > 0 else 0.0
        avg_incorrect_prob = np.mean(incorrect_probabilities) if len(incorrect_probabilities) > 0 else 0.0
        
        # Per-surface-type analysis
        unique_types = np.unique(surface_types)
        per_type_metrics = {}
        
        # Debug prints removed for production use. Uncomment if needed.
        # print(f"Debug - predictions shape: {predictions.shape}")
        # print(f"Debug - surface_types shape: {surface_types.shape}")
        # print(f"Debug - unique surface types: {unique_types}")
        
        # Define surface type names (assuming standard CAD surface types)
        type_names = {
            0: "Plane",
            1: "Cylinder", 
            2: "Cone",
            3: "Sphere",
            4: "Torus",
            5: "B-Spline/NURBS",
            6: "Other"
        }
        
        for surface_type in unique_types:
            type_mask = (surface_types == surface_type).reshape(-1)
            # Debug per-type mask info removed.
            
            type_predictions = predictions[type_mask]
            type_true_labels = true_labels[type_mask]
            type_probabilities = probabilities[type_mask]
            
            if len(type_predictions) == 0:
                continue
                
            # Compute metrics for this surface type
            type_accuracy = np.mean(type_predictions == type_true_labels)
            type_avg_prob = np.mean(type_probabilities)
            
            # Confusion matrix for this type
            type_tp = np.sum((type_predictions == 1) & (type_true_labels == 1))
            type_fp = np.sum((type_predictions == 1) & (type_true_labels == 0))
            type_tn = np.sum((type_predictions == 0) & (type_true_labels == 0))
            type_fn = np.sum((type_predictions == 0) & (type_true_labels == 1))
            
            type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0.0
            type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0.0
            type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
            type_specificity = type_tn / (type_tn + type_fp) if (type_tn + type_fp) > 0 else 0.0
            type_balanced_acc = (type_recall + type_specificity) / 2
            
            type_positive_samples = np.sum(type_true_labels == 1)
            type_negative_samples = np.sum(type_true_labels == 0)
            
            type_name = type_names.get(surface_type, f"Type_{surface_type}")
            
            per_type_metrics[type_name] = {
                'surface_type_id': int(surface_type),
                'accuracy': type_accuracy,
                'precision': type_precision,
                'recall': type_recall,
                'f1_score': type_f1,
                'specificity': type_specificity,
                'balanced_accuracy': type_balanced_acc,
                'avg_probability': type_avg_prob,
                'total_samples': len(type_predictions),
                'positive_samples': int(type_positive_samples),
                'negative_samples': int(type_negative_samples),
                'confusion_matrix': {
                    'tp': int(type_tp),
                    'fp': int(type_fp),
                    'tn': int(type_tn),
                    'fn': int(type_fn)
                }
            }
        
        # Failure case analysis (FP and FN distribution across surface types)
        failure_analysis = {}

        fp_mask_global = (predictions == 1) & (true_labels == 0)
        fn_mask_global = (predictions == 0) & (true_labels == 1)

        def _type_distribution(mask):
            distribution = {}
            total = int(np.sum(mask))
            if total == 0:
                return distribution, total
            for st in unique_types:
                count = int(np.sum(mask & (surface_types == st).reshape(-1)))
                if count > 0:
                    st_name = type_names.get(st, f"Type_{st}")
                    distribution[st_name] = count
            return distribution, total

        fp_dist, fp_total = _type_distribution(fp_mask_global)
        fn_dist, fn_total = _type_distribution(fn_mask_global)

        failure_analysis['false_positive'] = {
            'total': fp_total,
            'distribution': fp_dist
        }

        failure_analysis['false_negative'] = {
            'total': fn_total,
            'distribution': fn_dist
        }
        
        return {
            'overall': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'specificity': specificity,
                'balanced_accuracy': balanced_accuracy,
                'avg_probability': avg_probability,
                'avg_correct_probability': avg_correct_prob,
                'avg_incorrect_probability': avg_incorrect_prob,
                'avg_processing_time': avg_processing_time,
                'total_samples': len(predictions),
                'positive_samples': int(positive_samples),
                'negative_samples': int(negative_samples)
            },
            'confusion_matrix': {
                'tp': self.tp,
                'fp': self.fp,
                'tn': self.tn,
                'fn': self.fn
            },
            'per_surface_type': per_type_metrics,
            'failure_analysis': failure_analysis
        }
    
    def print_summary(self, metrics=None):
        """Print comprehensive adjacency prediction summary."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*80)
        print("ADJACENCY PREDICTION INFERENCE SUMMARY")
        print("="*80)
        
        # Overall metrics
        overall = metrics['overall']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Samples: {overall['total_samples']:,}")
        print(f"  Positive Samples: {overall['positive_samples']:,}")
        print(f"  Negative Samples: {overall['negative_samples']:,}")
        print(f"  Accuracy: {overall['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {overall['balanced_accuracy']:.4f}")
        print(f"  Precision: {overall['precision']:.4f}")
        print(f"  Recall (Sensitivity): {overall['recall']:.4f}")
        print(f"  Specificity: {overall['specificity']:.4f}")
        print(f"  F1-Score: {overall['f1_score']:.4f}")
        print(f"  Average Probability: {overall['avg_probability']:.4f}")
        print(f"  Correct Predictions Probability: {overall['avg_correct_probability']:.4f}")
        print(f"  Incorrect Predictions Probability: {overall['avg_incorrect_probability']:.4f}")
        print(f"  Average Processing Time: {overall['avg_processing_time']*1000:.2f} ms/sample")
        
        # Confusion Matrix
        cm = metrics['confusion_matrix']
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"                Predicted")
        print(f"              Non-Adj  Adjacent")
        print(f"Actual Non-Adj   {cm['tn']:6d}   {cm['fp']:6d}")
        print(f"       Adjacent  {cm['fn']:6d}   {cm['tp']:6d}")
        
        # Per-surface-type performance
        if 'per_surface_type' in metrics and metrics['per_surface_type']:
            print(f"\n📈 PERFORMANCE BY SURFACE TYPE:")
            print(f"{'Surface Type':<15} {'Samples':<8} {'Accuracy':<8} {'Precision':<9} {'Recall':<7} {'F1-Score':<8} {'Bal.Acc':<7}")
            print("-" * 80)
            
            # Sort by surface type name for consistent output
            for type_name in sorted(metrics['per_surface_type'].keys()):
                type_metrics = metrics['per_surface_type'][type_name]
                print(f"{type_name:<15} {type_metrics['total_samples']:<8} "
                      f"{type_metrics['accuracy']:<8.4f} {type_metrics['precision']:<9.4f} "
                      f"{type_metrics['recall']:<7.4f} {type_metrics['f1_score']:<8.4f} "
                      f"{type_metrics['balanced_accuracy']:<7.4f}")
        
        # Failure analysis summary
        if 'failure_analysis' in metrics:
            fa = metrics['failure_analysis']
            print(f"\n❌ FAILURE ANALYSIS:")
            # False Positives
            fp_info = fa.get('false_positive', {})
            fn_info = fa.get('false_negative', {})

            print(f"  False Positives: {fp_info.get('total', 0):,}")
            if fp_info.get('distribution'):
                for t, c in fp_info['distribution'].items():
                    print(f"    • {t}: {c}")

            print(f"  False Negatives: {fn_info.get('total', 0):,}")
            if fn_info.get('distribution'):
                for t, c in fn_info['distribution'].items():
                    print(f"    • {t}: {c}")
        
        # Model and hardware info
        print(f"\n⚙️  SYSTEM INFO:")
        print(f"  Surface Resolution: {self.cfg.model.surface_res}x{self.cfg.model.surface_res}")
        print(f"  Number of Nearby Surfaces: {self.cfg.model.num_nearby}")
        print(f"  Precision: {'FP16' if self.use_fp16 else 'FP32'}")
        print(f"  Device: {self.device}")
        print(f"  Model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        print("="*80)
    
    def save_results(self, output_path, metrics=None):
        """Save detailed results to JSON file."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        # Add additional information
        results = {
            'metrics': metrics,
            'model_config': {
                'surface_res': self.cfg.model.surface_res,
                'num_nearby': self.cfg.model.num_nearby,
                'precision': 'FP16' if self.use_fp16 else 'FP32',
                'device': str(self.device),
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'probabilities': self.probabilities,
            'processing_times': self.processing_times,
            'surface_types': self.surface_types
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Results saved to {output_path}")

def main():
    parser = ArgumentParser(description="Perform comprehensive adjacency prediction inference on dataset.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the .pkl file containing list of test filenames.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing individual .pkl files.")
    parser.add_argument('--config', type=str, required=True, help="Path to the adjacency model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the adjacency model checkpoint file.")
    parser.add_argument('--output', type=str, default=None, help="Path to save detailed results JSON file.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for processing.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16 for inference.")
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
    adjacency_predictor = AdjacencyInference(args.config, args.checkpoint, use_fp16=use_fp16)
    
    # Process dataset
    adjacency_predictor.process_dataset(args.data_file, args.data_dir, args.batch_size, args.num_workers)
    
    # Compute and print metrics
    metrics = adjacency_predictor.compute_metrics()
    adjacency_predictor.print_summary(metrics)
    
    # Save results if output path provided
    if args.output:
        adjacency_predictor.save_results(args.output, metrics)
    
    print(f"\nAdjacency prediction inference complete!")

if __name__ == '__main__':
    main() 