import numpy as np
import polyscope as ps
import random
import torch
from argparse import ArgumentParser
import os

# Classification model related imports
from src.flow.surface_flow import Encoder
from src.utils.config import NestedDictToClass, load_config
from einops import rearrange

# --- Global variables ---
data_points_all = None
data_labels_all = None
num_samples_total = 0
current_sample_indices = []
model_grid_size = None

# Polyscope structure references
ps_point_cloud_structures = []
ps_prediction_structures = []

# UI controllable number of samples
num_samples_to_display_ui = [1]

# Classification Model and Config Globals
classification_model = None
classification_cfg = None
device = None
use_fp16 = True  # Enable FP16 inference by default

# UI controls
ui_show_point_clouds = [True]
ui_show_predictions = [True]
ui_point_size = [0.02]

# Class names and filtering controls
class_names = ["Plane", "Cylinder", "Cone", "Sphere", "Torus", "BSpline"]
selected_class_index = [0]  # Index of selected class (0 = "All Classes")
filtered_sample_indices = []  # Indices of samples matching the selected class

# Prediction metrics tracking
prediction_metrics = {}  # Key: sample_idx, Value: dict with prediction metrics
ui_show_prediction_summary = [False]

# --- Load Data --- 
def load_npy_data(points_path, labels_path):
    """Load points and labels from NPY files."""
    global data_points_all, data_labels_all, num_samples_total
    global filtered_sample_indices
    
    try:
        data_points_all = np.load(points_path)
        data_labels_all = np.load(labels_path)
    except FileNotFoundError as e:
        print(f"Error: Data file not found. Please check the path. {e}")
        exit()
    
    # Validate data shapes
    if data_points_all.ndim != 3 or data_points_all.shape[-1] != 3:
        print(f"Error: Expected points shape (M, N, 3), got {data_points_all.shape}")
        exit()
    
    if len(data_labels_all) != len(data_points_all):
        print(f"Error: Number of points and labels must match. Points: {len(data_points_all)}, Labels: {len(data_labels_all)}")
        exit()
    
    num_samples_total = data_points_all.shape[0]

    if num_samples_total == 0:
        print("Error: No samples found in the data file.")
        exit()
    
    # Initialize filtered indices to all samples
    filtered_sample_indices = list(range(num_samples_total))
    
    print(f"Loaded {num_samples_total} point cloud samples with {len(class_names)} classes from {points_path} and {labels_path}.")

# --- Classification Model Loading ---
def load_classification_model_and_config(config_path, checkpoint_path):
    """Load the classification model and config."""
    global classification_model, classification_cfg, device, model_grid_size, use_fp16
    
    print(f"Loading classification model from config: {config_path} and checkpoint: {checkpoint_path}")
    try:
        cfg_dict = load_config(config_path)
        classification_cfg = NestedDictToClass(cfg_dict)
    except Exception as e:
        print(f"Error loading classification config: {e}")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if CUDA supports FP16
    if device.type == "cuda" and use_fp16:
        try:
            # Test if FP16 is supported
            test_tensor = torch.zeros(1, device=device, dtype=torch.float16)
            print("FP16 inference enabled - using half precision for memory efficiency")
        except:
            print("FP16 not supported on this device, falling back to FP32")
            use_fp16 = False
    else:
        if device.type == "cpu":
            print("CPU device detected, using FP32 (FP16 not supported on CPU)")
            use_fp16 = False

    try:
        classification_model = Encoder(
            in_dim=classification_cfg.model.in_dim,
            out_dim=classification_cfg.model.out_dim,
            depth=classification_cfg.model.depth,
            dim=classification_cfg.model.dim,
            heads=classification_cfg.model.heads,
            res=classification_cfg.model.res
        )
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'ema_model' in checkpoint:
            ema_model = checkpoint['ema']
            ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
            classification_model.load_state_dict(ema_model, strict=False)
            print("Loaded EMA model weights for classification.")
        elif 'model' in checkpoint:
            classification_model.load_state_dict(checkpoint['model'])
            print("Loaded model weights for classification.")
        else:
            classification_model.load_state_dict(checkpoint)
            print("Loaded raw model state_dict for classification.")
        
        classification_model.to(device)
        
        # Convert to half precision if using FP16
        if use_fp16:
            classification_model.half()
            print("Model converted to FP16")
        
        classification_model.eval()
        print("Classification model loaded and set to evaluation mode.")
        
        # Get model grid size
        model_grid_size = classification_cfg.model.res
        print(f"Model grid size: {model_grid_size}x{model_grid_size}")
        
    except Exception as e:
        print(f"Error initializing or loading classification model: {e}")
        exit()

# --- Class filtering functions ---
def update_filtered_sample_indices():
    """Update the filtered sample indices based on the selected class."""
    global filtered_sample_indices, selected_class_index, data_labels_all
    
    if selected_class_index[0] == 0:
        # "All Classes" selected
        filtered_sample_indices = list(range(num_samples_total))
    else:
        # Specific class selected
        selected_class_label = selected_class_index[0] - 1  # Adjust for "All Classes" at index 0
        filtered_sample_indices = [i for i, label in enumerate(data_labels_all) if label == selected_class_label]
    
    class_name = "All Classes" if selected_class_index[0] == 0 else class_names[selected_class_index[0] - 1]
    print(f"Filtered to {len(filtered_sample_indices)} samples of class '{class_name}'")

def get_class_for_sample(sample_idx):
    """Get the class name for a given sample index."""
    if sample_idx >= len(data_labels_all):
        return "Unknown"
    class_label = data_labels_all[sample_idx]
    if 0 <= class_label < len(class_names):
        return class_names[class_label]
    return "Unknown"

# --- Helper to get a new random sample index ---
def get_new_random_sample_idx(exclude_indices):
    """Get a new random sample index from filtered samples, avoiding those already shown if possible."""
    if len(filtered_sample_indices) == 0: 
        return -1
    
    # Convert exclude_indices to set for faster lookup
    exclude_set = set(exclude_indices)
    
    # Get available indices from filtered samples that are not excluded
    available_indices = [idx for idx in filtered_sample_indices if idx not in exclude_set]
    
    if not available_indices:
        # All filtered samples are already shown, pick any from filtered
        if filtered_sample_indices:
            return random.choice(filtered_sample_indices)
        else:
            return -1
    
    return random.choice(available_indices)

# --- Function to classify point cloud using the classification model ---
def classify_point_cloud(points):
    """
    Classify a point cloud using the loaded classification model.
    
    Args:
        points: numpy array of shape (N, N, 3) - the input point cloud
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities) - predicted class index, confidence, and all class probabilities
    """
    global classification_model, classification_cfg, device, use_fp16
    
    if classification_model is None or classification_cfg is None or device is None:
        print("Classification model not loaded. Cannot classify point cloud.")
        return -1, 0.0, np.zeros(len(class_names))
    
    # Prepare inputs with appropriate dtype
    input_dtype = torch.float16 if use_fp16 else torch.float32
    
    # Resize point cloud to model expected size if needed
    if points.shape[0] != model_grid_size:
        if points.shape[0] % model_grid_size == 0:
            step = points.shape[0] // model_grid_size
            points = points[::step, ::step]
        else:
            print(f"Warning: Point cloud size {points.shape[0]} is not compatible with model grid size {model_grid_size}")
            # Simple interpolation fallback
            try:
                from scipy.interpolate import griddata
                h, w = points.shape[:2]
                old_coords = np.mgrid[0:h, 0:w].reshape(2, -1).T
                new_coords = np.mgrid[0:model_grid_size, 0:model_grid_size].reshape(2, -1).T
                new_coords = new_coords * (h - 1) / (model_grid_size - 1)
                
                points_flat = points.reshape(-1, 3)
                new_points = np.zeros((model_grid_size * model_grid_size, 3))
                for i in range(3):
                    new_points[:, i] = griddata(old_coords, points_flat[:, i], new_coords, method='linear')
                points = new_points.reshape(model_grid_size, model_grid_size, 3)
            except ImportError:
                print("scipy not available for interpolation, using simple cropping")
                points = points[:model_grid_size, :model_grid_size]
    
    points_tensor = torch.from_numpy(points).to(dtype=input_dtype, device=device).unsqueeze(0)  # (1, N, N, 3)
    
    # Use autocast for mixed precision if using FP16
    autocast_context = torch.cuda.amp.autocast() if use_fp16 and device.type == "cuda" else torch.no_grad()
    
    with autocast_context:
        # Get model predictions
        logits = classification_model(points_tensor)  # (1, N*N, num_classes)
        
        # Global average pooling if output is sequence
        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # (1, num_classes)
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class and confidence
        confidence, predicted_class = torch.max(probabilities, dim=-1)
        
        # Convert to numpy (ensure float32 for numpy compatibility)
        predicted_class = predicted_class[0].cpu().numpy().item()
        confidence = confidence[0].float().cpu().numpy().item()
        all_probabilities = probabilities[0].float().cpu().numpy()
        
    return predicted_class, confidence, all_probabilities

# --- Helper function for color gradient by class ---
def get_color_by_class(class_idx, confidence=1.0):
    """Generate color for points based on their class."""
    colors = [
        [1.0, 0.2, 0.2],  # Red for Plane
        [0.2, 1.0, 0.2],  # Green for Cylinder
        [0.2, 0.2, 1.0],  # Blue for Cone
        [1.0, 1.0, 0.2],  # Yellow for Sphere
        [1.0, 0.2, 1.0],  # Magenta for Torus
        [0.2, 1.0, 1.0],  # Cyan for BSpline
    ]
    
    if 0 <= class_idx < len(colors):
        color = np.array(colors[class_idx]) * confidence + np.array([0.3, 0.3, 0.3]) * (1 - confidence)
        return np.clip(color, 0.0, 1.0)
    else:
        return np.array([0.5, 0.5, 0.5])  # Gray for unknown

# --- Classification metrics computation functions ---
def compute_classification_metrics(predicted_class, true_class, all_probabilities):
    """
    Compute classification metrics.
    
    Args:
        predicted_class: predicted class index
        true_class: true class index
        all_probabilities: array of probabilities for all classes
        
    Returns:
        dict: classification metrics
    """
    is_correct = predicted_class == true_class
    confidence = all_probabilities[predicted_class] if 0 <= predicted_class < len(all_probabilities) else 0.0
    true_class_prob = all_probabilities[true_class] if 0 <= true_class < len(all_probabilities) else 0.0
    
    return {
        'is_correct': is_correct,
        'confidence': confidence,
        'true_class_prob': true_class_prob,
        'predicted_class': predicted_class,
        'true_class': true_class,
        'entropy': -np.sum(all_probabilities * np.log(all_probabilities + 1e-8))
    }

def print_classification_metrics(sample_idx, true_class_name, predicted_class_name, metrics):
    """
    Print classification metrics.
    
    Args:
        sample_idx: Sample index for identification
        true_class_name: True class name
        predicted_class_name: Predicted class name
        metrics: Classification metrics dict
    """
    global prediction_metrics
    
    print(f"\n📊 Classification Metrics for Sample {sample_idx}:")
    print("=" * 70)
    
    print(f"True Class: {true_class_name}")
    print(f"Predicted Class: {predicted_class_name}")
    print(f"Correct: {'✓' if metrics['is_correct'] else '✗'}")
    print(f"Confidence: {metrics['confidence']:.4f}")
    print(f"True Class Probability: {metrics['true_class_prob']:.4f}")
    print(f"Prediction Entropy: {metrics['entropy']:.4f}")
    
    # Store metrics for later analysis
    prediction_metrics[sample_idx] = {
        'true_class_name': true_class_name,
        'predicted_class_name': predicted_class_name,
        **metrics
    }
    
    print("=" * 70)

def get_prediction_summary_stats():
    """Get summary statistics for all computed prediction metrics."""
    if not prediction_metrics:
        return None
    
    # Collect all metrics
    accuracies = [m['is_correct'] for m in prediction_metrics.values()]
    confidences = [m['confidence'] for m in prediction_metrics.values()]
    entropies = [m['entropy'] for m in prediction_metrics.values()]
    
    return {
        'num_samples': len(prediction_metrics),
        'accuracy': np.mean(accuracies),
        'avg_confidence': np.mean(confidences),
        'avg_entropy': np.mean(entropies),
        'correct_confidence': np.mean([m['confidence'] for m in prediction_metrics.values() if m['is_correct']]) if any(accuracies) else 0.0,
        'incorrect_confidence': np.mean([m['confidence'] for m in prediction_metrics.values() if not m['is_correct']]) if not all(accuracies) else 0.0,
    }

# --- Polyscope Plotting/Updating for Classification ---
def display_or_update_classification_results(slot_idx, sample_idx):
    """Display or update all results from classification inference."""
    global ps_point_cloud_structures, ps_prediction_structures

    if data_points_all is None:
        print("Data not loaded. Cannot display point clouds.")
        return

    points = data_points_all[sample_idx]  # Shape: (N, N, 3)
    true_class = data_labels_all[sample_idx]
    true_class_name = get_class_for_sample(sample_idx)

    # Classify the point cloud
    predicted_class, confidence, all_probabilities = classify_point_cloud(points)
    predicted_class_name = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else "Unknown"
    
    # Compute and print classification metrics
    metrics = compute_classification_metrics(predicted_class, true_class, all_probabilities)
    print_classification_metrics(sample_idx, true_class_name, predicted_class_name, metrics)

    # Prepare data for visualization
    points_flat = points.reshape(-1, 3)

    # Get colors based on prediction
    if metrics['is_correct']:
        # Green gradient for correct predictions
        point_colors = np.tile([0.2, 0.8, 0.2], (len(points_flat), 1)) * confidence + np.tile([0.7, 0.7, 0.7], (len(points_flat), 1)) * (1 - confidence)
    else:
        # Red gradient for incorrect predictions
        point_colors = np.tile([0.8, 0.2, 0.2], (len(points_flat), 1)) * confidence + np.tile([0.7, 0.7, 0.7], (len(points_flat), 1)) * (1 - confidence)

    # Ensure lists are long enough
    while len(ps_point_cloud_structures) <= slot_idx:
        ps_point_cloud_structures.append(None)
    while len(ps_prediction_structures) <= slot_idx:
        ps_prediction_structures.append(None)

    # Structure names
    pc_name = f"point_cloud_{slot_idx}"

    # Point Cloud
    if ui_show_point_clouds[0]:
        if ps_point_cloud_structures[slot_idx] is None or not ps.has_point_cloud(pc_name):
            ps_point_cloud_structures[slot_idx] = ps.register_point_cloud(pc_name, points_flat)
        else:
            ps_point_cloud_structures[slot_idx].update_point_positions(points_flat)
        
        ps_point_cloud_structures[slot_idx].add_color_quantity("prediction_color", point_colors, enabled=True)
        ps_point_cloud_structures[slot_idx].set_radius(ui_point_size[0])
    elif ps_point_cloud_structures[slot_idx] is not None and ps.has_point_cloud(pc_name):
        ps.remove_point_cloud(pc_name)
        ps_point_cloud_structures[slot_idx] = None

# --- Manage display count ---
def manage_sample_display_count(force_refresh_all=False):
    """Manage the total number of displayed samples based on UI."""
    global current_sample_indices
    desired_num = num_samples_to_display_ui[0]
    current_num_on_screen = len(current_sample_indices)

    if not ps.is_initialized(): 
        ps.init()

    if force_refresh_all:
        print(f"Refreshing all {current_num_on_screen} displayed samples.")
        temp_current_indices = list(current_sample_indices)
        for i in range(current_num_on_screen):
            new_sample_idx = get_new_random_sample_idx(temp_current_indices)
            current_sample_indices[i] = new_sample_idx
            display_or_update_classification_results(i, new_sample_idx)
            if new_sample_idx not in temp_current_indices:
                temp_current_indices.append(new_sample_idx)
        return

    # Add samples if desired > current
    while desired_num > len(current_sample_indices):
        slot_idx = len(current_sample_indices)
        new_sample_idx = get_new_random_sample_idx(current_sample_indices)
        if new_sample_idx == -1: 
            break
        
        current_sample_indices.append(new_sample_idx)
        print(f"Adding classification results at slot {slot_idx} with data sample {new_sample_idx}")
        display_or_update_classification_results(slot_idx, new_sample_idx)

    # Remove samples if desired < current
    while desired_num < len(current_sample_indices):
        slot_idx_to_remove = len(current_sample_indices) - 1
        print(f"Removing classification results from slot {slot_idx_to_remove}")
        
        # Remove all structures for this slot
        for structure_list in [ps_point_cloud_structures, ps_prediction_structures]:
            if len(structure_list) > slot_idx_to_remove and structure_list[slot_idx_to_remove] is not None:
                structure_list.pop()
        
        # Remove point clouds
        pc_name = f"point_cloud_{slot_idx_to_remove}"
        if ps.has_point_cloud(pc_name):
            ps.remove_point_cloud(pc_name)
        
        current_sample_indices.pop()
        
    if current_num_on_screen != len(current_sample_indices):
        print(f"Now displaying {len(current_sample_indices)} classification result(s).")

# --- Function to register manual XYZ axes ---
def register_manual_xyz_axes(length=1.0, radius=0.02):
    """Register manual XYZ axes for reference."""
    if not ps.is_initialized(): 
        ps.init()

    # X-axis (Red)
    nodes_x = np.array([[0,0,0], [length,0,0]])
    edges_x = np.array([[0,1]])
    ps_x = ps.register_curve_network("x_axis", nodes_x, edges_x, radius=radius)
    ps_x.set_color((0.8, 0.1, 0.1))
    ps_x.set_radius(radius)

    # Y-axis (Green)
    nodes_y = np.array([[0,0,0], [0,length,0]])
    edges_y = np.array([[0,1]])
    ps_y = ps.register_curve_network("y_axis", nodes_y, edges_y)
    ps_y.set_color((0.1, 0.8, 0.1))
    ps_y.set_radius(radius)

    # Z-axis (Blue)
    nodes_z = np.array([[0,0,0], [0,0,length]])
    edges_z = np.array([[0,1]])
    ps_z = ps.register_curve_network("z_axis", nodes_z, edges_z)
    ps_z.set_color((0.1, 0.1, 0.8))
    ps_z.set_radius(radius)
    print("Manually registered XYZ axes.")

# --- Polyscope User Interface Callback --- 
def my_ui_callback():
    """Main UI callback for the classification examination interface."""
    global num_samples_to_display_ui, current_sample_indices
    global ps_point_cloud_structures, ps_prediction_structures
    global ui_show_point_clouds, ui_show_predictions, ui_point_size
    global model_grid_size
    global selected_class_index, filtered_sample_indices, data_labels_all
    
    # --- Display current mode ---
    ps.imgui.TextColored((0.2, 0.8, 0.8, 1), "Mode: Classification Model")
    
    # --- Display model information ---
    if model_grid_size is not None:
        precision_text = "FP16" if use_fp16 else "FP32"
        ps.imgui.TextColored((0.7, 0.7, 0.7, 1), f"Grid: {model_grid_size}x{model_grid_size}, Precision: {precision_text}")
    
    # --- Display class filtering information ---
    if data_labels_all is not None:
        current_class = "All Classes" if selected_class_index[0] == 0 else class_names[selected_class_index[0] - 1]
        ps.imgui.TextColored((0.8, 0.9, 0.6, 1), f"Class Filter: {current_class} ({len(filtered_sample_indices)} samples)")
    
    ps.imgui.Separator()
    
    # --- Class Selector ---
    class_options = ["All Classes"] + class_names
    ps.imgui.Text("Surface Class Filter")
    
    changed_class, selected_class_index[0] = ps.imgui.Combo("Surface Class", selected_class_index[0], class_options)
    
    if changed_class:
        # Update filtered indices
        update_filtered_sample_indices()
        
        # Clear current displays and refresh with new class
        if len(current_sample_indices) > 0:
            manage_sample_display_count(force_refresh_all=True)
    
    ps.imgui.Separator()
    
    # --- Display Controls ---
    ps.imgui.Text("Display Controls")
    
    changed_show_pc, ui_show_point_clouds[0] = ps.imgui.Checkbox("Show Point Clouds", ui_show_point_clouds[0])
    
    if ui_show_point_clouds[0]:
        ps.imgui.PushItemWidth(100)
        changed_point_size, ui_point_size[0] = ps.imgui.SliderFloat("Point Size", ui_point_size[0], 0.005, 0.1)
        ps.imgui.PopItemWidth()
        if changed_point_size:
            # Update point sizes for all displayed structures
            for slot_idx in range(len(current_sample_indices)):
                pc_name = f"point_cloud_{slot_idx}"
                if ps.has_point_cloud(pc_name):
                    ps.get_point_cloud(pc_name).set_radius(ui_point_size[0])
    
    # Update display if visibility options changed
    if changed_show_pc:
        for slot_idx in range(len(current_sample_indices)):
            sample_idx = current_sample_indices[slot_idx]
            display_or_update_classification_results(slot_idx, sample_idx)
    
    ps.imgui.Separator()
    
    # --- Sample Management ---
    ps.imgui.Text("Sample Management")
    
    ps.imgui.PushItemWidth(100)
    changed_num_display, num_samples_to_display_ui[0] = ps.imgui.InputInt("Num Samples", num_samples_to_display_ui[0], step=1, step_fast=5)
    ps.imgui.PopItemWidth()
    if num_samples_to_display_ui[0] < 0: 
        num_samples_to_display_ui[0] = 0 
    max_displayable_main = min(len(filtered_sample_indices), 10)  # Limit based on filtered samples
    if num_samples_to_display_ui[0] > max_displayable_main: 
        num_samples_to_display_ui[0] = max_displayable_main

    if changed_num_display:
        manage_sample_display_count(force_refresh_all=False)

    if ps.imgui.Button("Refresh Samples"):
        if len(current_sample_indices) > 0:
             manage_sample_display_count(force_refresh_all=True)
        else: 
             manage_sample_display_count(force_refresh_all=False)
    
    # Show information about currently displayed samples
    if len(current_sample_indices) > 0 and data_labels_all is not None:
        ps.imgui.Text("Currently Displayed:")
        for i, sample_idx in enumerate(current_sample_indices):
            class_name = get_class_for_sample(sample_idx)
            ps.imgui.Text(f"  Slot {i}: Sample {sample_idx} ({class_name})")
        if len(current_sample_indices) > 3:  # Limit display to avoid clutter
            ps.imgui.Text(f"  ... and {len(current_sample_indices) - 3} more")
    
    ps.imgui.Separator()
    
    # --- Prediction Metrics Summary ---
    ps.imgui.Text("Prediction Metrics")
    
    changed_show_summary, ui_show_prediction_summary[0] = ps.imgui.Checkbox("Show Prediction Summary", ui_show_prediction_summary[0])
    
    if ui_show_prediction_summary[0]:
        stats = get_prediction_summary_stats()
        if stats is not None:
            ps.imgui.Separator()
            ps.imgui.TextColored((1.0, 0.9, 0.5, 1), f"Prediction Summary ({stats['num_samples']} samples):")
            
            ps.imgui.Text(f"Accuracy: {stats['accuracy']:.3f}")
            ps.imgui.Text(f"Avg Confidence: {stats['avg_confidence']:.3f}")
            ps.imgui.Text(f"Avg Entropy: {stats['avg_entropy']:.3f}")
            
            if stats['correct_confidence'] > 0:
                ps.imgui.Text(f"Correct Confidence: {stats['correct_confidence']:.3f}")
            if stats['incorrect_confidence'] > 0:
                ps.imgui.Text(f"Incorrect Confidence: {stats['incorrect_confidence']:.3f}")
        else:
            ps.imgui.TextColored((0.7, 0.7, 0.7, 1), "No prediction metrics computed yet.")
    
    if ps.imgui.Button("Clear Prediction Metrics"):
        prediction_metrics.clear()
        print("Cleared all stored prediction metrics.")

# --- Main Execution ---
if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize Classification Model Inference.")
    parser.add_argument('--points_file', type=str, required=True, help="Path to the .npy file containing point clouds.")
    parser.add_argument('--labels_file', type=str, required=True, help="Path to the .npy file containing class labels.")
    parser.add_argument('--config', type=str, required=True, help="Path to the classification model config file.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the classification model checkpoint file.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--fp32', action='store_true', help="Use FP32 instead of FP16 for inference.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Using random seed: {args.seed}")

    # Set precision mode based on command line argument
    if args.fp32:
        use_fp16 = False
        print("FP32 mode requested via command line")

    load_npy_data(args.points_file, args.labels_file)
    load_classification_model_and_config(args.config, args.checkpoint)

    # Initialize Polyscope
    ps.init()

    # Manually register XYZ axes
    axis_length = 1
    register_manual_xyz_axes(length=axis_length, radius=0.01 * axis_length) 

    # Set the user callback function
    ps.set_user_callback(my_ui_callback)

    # Initial display
    manage_sample_display_count()

    # Show the Polyscope GUI
    ps.show() 