import numpy as np
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import os
from tqdm import tqdm
import time
from collections import defaultdict
import json

# Classification model related imports
from src.flow.surface_flow import Encoder
from src.utils.config import NestedDictToClass, load_config
from src.dataset.dataset_cls import SurfaceClassificationDataset

class ClassificationInference:
    def __init__(self, config_path, checkpoint_path, use_fp16=True):
        """Initialize the classification inference system."""
        self.class_names = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline"]
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
        """Load the classification model and configuration."""
        print(f"Loading classification model from config: {config_path}")
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
            if 'ema_model' in checkpoint:
                ema_model = checkpoint['ema']
                ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
                self.model.load_state_dict(ema_model, strict=False)
                print("Loaded EMA model weights for classification.")
            elif 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
                print("Loaded model weights for classification.")
            else:
                self.model.load_state_dict(checkpoint)
                print("Loaded raw model state_dict for classification.")
            
            self.model.to(self.device)
            
            if self.use_fp16:
                self.model.half()
                print("Model converted to FP16")
            
            self.model.eval()
            self.model_grid_size = self.cfg.model.res
            print(f"Model loaded successfully. Grid size: {self.model_grid_size}x{self.model_grid_size}")
            
        except Exception as e:
            print(f"Error loading classification model: {e}")
            raise
    
    def reset_metrics(self):
        """Reset all metrics tracking."""
        self.predictions = []
        self.true_labels = []
        self.confidences = []
        self.all_probabilities = []
        self.entropies = []
        self.processing_times = []
        
        # Per-class metrics
        self.class_metrics = {i: defaultdict(list) for i in range(len(self.class_names))}
        
        # Confusion matrix
        self.confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)), dtype=int)
    
    def classify_batch(self, batch_points, batch_labels):
        """Classify a batch of point clouds."""
        start_time = time.time()
        
        # Prepare tensor with appropriate dtype
        input_dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Convert to proper dtype and move to device
        if batch_points.dtype != input_dtype:
            batch_points = batch_points.to(dtype=input_dtype)
        batch_points = batch_points.to(self.device)
        
        # Inference
        autocast_context = torch.cuda.amp.autocast() if self.use_fp16 else torch.no_grad()
        
        with autocast_context:
            with torch.no_grad():
                logits = self.model(batch_points)  # (batch_size, N*N, num_classes) or (batch_size, num_classes)
                
                # Global average pooling if output is sequence
                if logits.dim() == 3:
                    logits = logits.mean(dim=1)  # (batch_size, num_classes)
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predicted class and confidence
                confidences, predicted_classes = torch.max(probabilities, dim=-1)
                
                # Convert to numpy
                predicted_classes = predicted_classes.cpu().numpy()
                confidences = confidences.float().cpu().numpy()
                all_probabilities = probabilities.float().cpu().numpy()
        
        processing_time = time.time() - start_time
        
        # Compute entropy for each sample
        entropies = -np.sum(all_probabilities * np.log(all_probabilities + 1e-8), axis=1)
        
        return predicted_classes, confidences, all_probabilities, entropies, processing_time
    
    def process_dataset(self, points_path, labels_path, batch_size=32, num_workers=4):
        """Process entire dataset using DataLoader for efficient batch processing."""
        print(f"Creating dataset from {points_path} and {labels_path}")
        
        # Create dataset
        try:
            dataset = SurfaceClassificationDataset(
                points_path=points_path,
                class_label_path=labels_path,
                replication=1,
                transform=None,
                is_train=False,
                res=self.model_grid_size
            )
        except Exception as e:
            print(f"Error creating dataset: {e}")
            raise
        
        print(f"Dataset created with {len(dataset)} samples")
        print(f"Sample data shape from dataset: {dataset.points.shape}")
        
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
        for batch_idx, (batch_points, batch_labels) in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Convert labels to numpy for consistency
            batch_labels_np = batch_labels.cpu().numpy()
            
            # Classify batch
            pred_classes, confidences, all_probs, entropies, proc_time = self.classify_batch(batch_points, batch_labels_np)
            total_processing_time += proc_time
            
            # Store results for each sample in the batch
            for i in range(len(pred_classes)):
                pred_class = pred_classes[i]
                true_label = batch_labels_np[i]
                confidence = confidences[i]
                all_prob = all_probs[i]
                entropy = entropies[i]
                
                # Store results
                self.predictions.append(pred_class)
                self.true_labels.append(true_label)
                self.confidences.append(confidence)
                self.all_probabilities.append(all_prob)
                self.entropies.append(entropy)
                # Approximate per-sample processing time
                self.processing_times.append(proc_time / len(pred_classes))
                
                # Update confusion matrix
                if 0 <= true_label < len(self.class_names) and 0 <= pred_class < len(self.class_names):
                    self.confusion_matrix[true_label, pred_class] += 1
                
                # Store per-class metrics
                if 0 <= true_label < len(self.class_names):
                    is_correct = pred_class == true_label
                    self.class_metrics[true_label]['correct'].append(is_correct)
                    self.class_metrics[true_label]['confidence'].append(confidence)
                    self.class_metrics[true_label]['entropy'].append(entropy)
                    self.class_metrics[true_label]['true_class_prob'].append(all_prob[true_label])
        
        print(f"Processing complete! Total time: {total_processing_time:.2f}s")
        print(f"Average batch processing time: {total_processing_time/len(dataloader):.3f}s")
        print(f"Effective samples/second: {len(dataset)/total_processing_time:.1f}")
    
    def compute_metrics(self):
        """Compute comprehensive classification metrics."""
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        confidences = np.array(self.confidences)
        entropies = np.array(self.entropies)
        
        # Overall metrics
        accuracy = np.mean(predictions == true_labels)
        avg_confidence = np.mean(confidences)
        avg_entropy = np.mean(entropies)
        avg_processing_time = np.mean(self.processing_times)
        
        # Correct vs incorrect predictions
        correct_mask = predictions == true_labels
        correct_confidence = np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0.0
        incorrect_confidence = np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0.0
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(len(self.class_names)):
            class_name = self.class_names[class_idx]
            
            # True positives, false positives, false negatives
            tp = np.sum((predictions == class_idx) & (true_labels == class_idx))
            fp = np.sum((predictions == class_idx) & (true_labels != class_idx))
            fn = np.sum((predictions != class_idx) & (true_labels == class_idx))
            tn = np.sum((predictions != class_idx) & (true_labels != class_idx))
            
            # Precision, Recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Class-specific accuracy
            class_accuracy = np.mean(self.class_metrics[class_idx]['correct']) if self.class_metrics[class_idx]['correct'] else 0.0
            class_avg_confidence = np.mean(self.class_metrics[class_idx]['confidence']) if self.class_metrics[class_idx]['confidence'] else 0.0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': class_accuracy,
                'avg_confidence': class_avg_confidence,
                'support': tp + fn,  # Number of true instances
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
            }
        
        return {
            'overall': {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_entropy': avg_entropy,
                'correct_confidence': correct_confidence,
                'incorrect_confidence': incorrect_confidence,
                'avg_processing_time': avg_processing_time,
                'total_samples': len(predictions),
                'correct_predictions': int(np.sum(correct_mask)),
                'incorrect_predictions': int(np.sum(~correct_mask))
            },
            'per_class': per_class_metrics,
            'confusion_matrix': self.confusion_matrix.tolist()
        }
    
    def print_summary(self, metrics=None):
        """Print comprehensive classification summary."""
        if metrics is None:
            metrics = self.compute_metrics()
        
        print("\n" + "="*80)
        print("CLASSIFICATION INFERENCE SUMMARY")
        print("="*80)
        
        # Overall metrics
        overall = metrics['overall']
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"  Total Samples: {overall['total_samples']:,}")
        print(f"  Accuracy: {overall['accuracy']:.4f} ({overall['correct_predictions']}/{overall['total_samples']})")
        print(f"  Average Confidence: {overall['avg_confidence']:.4f}")
        print(f"  Average Entropy: {overall['avg_entropy']:.4f}")
        print(f"  Correct Predictions Confidence: {overall['correct_confidence']:.4f}")
        print(f"  Incorrect Predictions Confidence: {overall['incorrect_confidence']:.4f}")
        print(f"  Average Processing Time: {overall['avg_processing_time']*1000:.2f} ms/sample")
        
        # Per-class metrics
        print(f"\n📈 PER-CLASS PERFORMANCE:")
        print(f"{'Class':<12} {'Precision':<10} {'Recall':<8} {'F1-Score':<9} {'Support':<8} {'Accuracy':<9} {'Confidence':<11}")
        print("-" * 75)
        
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0
        total_support = 0
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<12} {class_metrics['precision']:<10.4f} {class_metrics['recall']:<8.4f} "
                  f"{class_metrics['f1_score']:<9.4f} {class_metrics['support']:<8d} "
                  f"{class_metrics['accuracy']:<9.4f} {class_metrics['avg_confidence']:<11.4f}")
            
            macro_precision += class_metrics['precision']
            macro_recall += class_metrics['recall']
            macro_f1 += class_metrics['f1_score']
            total_support += class_metrics['support']
        
        # Macro averages
        num_classes = len(metrics['per_class'])
        macro_precision /= num_classes
        macro_recall /= num_classes
        macro_f1 /= num_classes
        
        print("-" * 75)
        print(f"{'Macro Avg':<12} {macro_precision:<10.4f} {macro_recall:<8.4f} {macro_f1:<9.4f} {total_support:<8d}")
        
        # Confusion Matrix
        print(f"\n📋 CONFUSION MATRIX:")
        confusion_matrix = np.array(metrics['confusion_matrix'])
        
        # Print header
        print(f"{'True Pred':<12}", end="")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:8]:<10}", end="")
        print()
        
        # Print matrix
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:10]:<12}", end="")
            for j in range(len(self.class_names)):
                print(f"{confusion_matrix[i, j]:<10d}", end="")
            print()
        
        # Model and hardware info
        print(f"\n⚙️  SYSTEM INFO:")
        print(f"  Model Grid Size: {self.model_grid_size}x{self.model_grid_size}")
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
                'grid_size': self.model_grid_size,
                'precision': 'FP16' if self.use_fp16 else 'FP32',
                'device': str(self.device),
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'class_names': self.class_names,
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'confidences': self.confidences,
            'entropies': self.entropies,
            'processing_times': self.processing_times
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"Results saved to {output_path}")

def main():
    parser = ArgumentParser(description="Perform comprehensive classification inference on dataset using DataLoader.")
    parser.add_argument('--points_file', type=str, required=True, help="Path to the .npy file containing point clouds.")
    parser.add_argument('--labels_file', type=str, required=True, help="Path to the .npy file containing class labels.")
    parser.add_argument('--config', type=str, required=True, help="Path to the classification model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the classification model checkpoint file.")
    parser.add_argument('--output', type=str, default=None, help="Path to save detailed results JSON file.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing.")
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
    classifier = ClassificationInference(args.config, args.checkpoint, use_fp16=use_fp16)
    
    # Process dataset
    classifier.process_dataset(args.points_file, args.labels_file, args.batch_size, args.num_workers)
    
    # Compute and print metrics
    metrics = classifier.compute_metrics()
    classifier.print_summary(metrics)
    
    # Save results if output path provided
    if args.output:
        classifier.save_results(args.output, metrics)
    
    print(f"\nInference complete!")

if __name__ == '__main__':
    main() 