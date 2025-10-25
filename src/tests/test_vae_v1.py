import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import sys
import json
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')


from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.vae.vae_v1 import SurfaceVAE
from src.utils.numpy_tools import orthonormal_basis_from_normal

from utils.surface import visualize_json_interset

def to_json(params_tensor, types_tensor, mask_tensor):
    json_data = []
    # SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    for i in range(len(params_tensor)):
        print(types_tensor[i].item(), mask_tensor[i].sum())
        params = params_tensor[i][mask_tensor[i]]
        # surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        recovered_surface = dataset._recover_surface(params, types_tensor[i].item())

        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        json_data.append(recovered_surface)

    return json_data

# Global variables for interactive visualization
dataset = None
model = None
current_idx = 21
max_idx = 0
gt_group = None
recovered_group = None
gt_surfaces = {}
recovered_surfaces = {}
show_gt = True
show_recovered = True
resampled_surfaces = {}
show_resampled = False

def load_model_and_dataset():
    global dataset, model, max_idx
    
    dataset = dataset_compound(sys.argv[1])
    max_idx = len(dataset) - 1
    
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 19, 18]) # Should be changed to [17, 18, 19, 18, 19]
    checkpoint_path = sys.argv[2]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'ema_model' in checkpoint:
        ema_model = checkpoint['ema']
        ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
        model.load_state_dict(ema_model, strict=False)
        print("Loaded EMA model weights for classification.")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("Loaded model weights for classification.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw model state_dict for classification.")
    
    model.eval()

def process_sample(idx):
    """Process a single sample and return both GT and recovered data"""
    global dataset, model
    
    params_tensor, types_tensor, mask_tensor = dataset[idx]
    print('processing file: ', dataset.json_names[idx])
    json_path = dataset.json_names[idx]
    
    # Apply mask to get valid surfaces
    valid_params = params_tensor[mask_tensor.bool()]
    valid_types = types_tensor[mask_tensor.bool()]
    
    # Load ground truth JSON data
    with open(json_path, 'r') as f:
        gt_json_data = json.load(f)
    
    # Run VAE inference
    with torch.no_grad():
        mu, logvar = model.encode(valid_params, valid_types)
        
        z = model.reparameterize(mu, logvar)
        type_logits_pred, types_pred = model.classify(z)
        params_pred, mask = model.decode(z, types_pred)
        
        # Calculate metrics
        recon_fn = torch.nn.MSELoss()
        recon_loss = (recon_fn(params_pred, valid_params)) * mask.float().mean()
        accuracy = (types_pred == valid_types).float().mean()
        
        print(f'Index {idx}: recon_loss: {recon_loss.item():.6f}, accuracy: {accuracy.item():.4f}')
        # print(f'Predicted types: {types_pred.cpu().numpy()}')
        # print(f'Ground truth types: {valid_types.cpu().numpy()}')
    
    # Convert predictions to JSON format
    recovered_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
    
    return gt_json_data, recovered_json_data, recon_loss.item(), accuracy.item()

def resample_model():
    """Generate new samples from the VAE's latent space"""
    global model, resampled_surfaces, current_idx
    
    if model is None:
        print("Model not loaded yet!")
        return
    
    params_tensor, types_tensor, mask_tensor = dataset[current_idx]
    valid_params = params_tensor[mask_tensor.bool()]
    valid_types = types_tensor[mask_tensor.bool()]
    
    with torch.no_grad():
        # Sample random latent vectors

        mu, logvar = model.encode(valid_params, valid_types)
        z_random = model.reparameterize(mu, logvar)
        # Classify the random latent vectors to get surface types
        type_logits_pred, types_pred = model.classify(z_random)
        
        # Decode to get surface parameters
        params_pred, mask = model.decode(z_random, types_pred)
        
        # print(f"Generated {len(params_pred)} resampled surfaces")
        # print(f"Resampled types: {types_pred.cpu().numpy()}")
        
        # Convert to JSON format
        resampled_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
        
        # Visualize resampled surfaces
        resampled_surfaces = visualize_json_interset(resampled_json_data, plot=True, plot_gui=False, tol=1e-5, ps_header='resampled')
        
        # Add to resampled group if it exists
        for i, (surface_key, surface_data) in enumerate(resampled_surfaces.items()):
            if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
                surface_data['ps_handler'].add_to_group(resampled_group)
        
        return resampled_json_data, resampled_surfaces

def update_visualization():
    """Update the visualization with current index"""
    global current_idx, gt_group, recovered_group, gt_surfaces, recovered_surfaces
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    gt_data, recovered_data, recon_loss, accuracy = process_sample(current_idx)
    
    # Create groups

    
    # Visualize ground truth surfaces
    gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-5)
    # print(gt_surfaces)
    for i, (surface_key, surface_data) in enumerate(gt_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to ground truth group
            surface_data['ps_handler'].add_to_group(gt_group)
    
    # Visualize recovered surfaces  
    recovered_surfaces = visualize_json_interset(recovered_data, plot=True, plot_gui=False, tol=1e-5, ps_header='z_rec')
    for i, (surface_key, surface_data) in enumerate(recovered_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to recovered group
            surface_data['ps_handler'].add_to_group(recovered_group)
    
    # Configure groups with current visibility settings
    gt_group.set_enabled(show_gt)
    recovered_group.set_enabled(show_recovered)
    
    print(f"Visualized {len(gt_surfaces)} GT surfaces and {len(recovered_surfaces)} recovered surfaces")

def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_gt, show_recovered, show_resampled, resampled_surfaces
    
    psim.Text("VAE Surface Reconstruction Test")
    psim.Separator()
    
    # Index slider
    changed, new_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if changed:
        current_idx = new_idx
        update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Resample button
    psim.Separator()
    psim.Text("Model Controls:")
    if psim.Button("Resample Model"):
        for surface in resampled_surfaces.values():
            surface['ps_handler'].remove()
        resampled_surfaces = {}
        resampled_json_data, resampled_surfaces = resample_model()
    
    # Group controls
    if gt_group is not None:
        psim.Separator()
        psim.Text("Group Controls:")
        changed, show_gt = psim.Checkbox("Show Ground Truth", show_gt)
        if changed:
            gt_group.set_enabled(show_gt)
        
        changed, show_recovered = psim.Checkbox("Show Recovered", show_recovered)
        if changed:
            recovered_group.set_enabled(show_recovered)
        
        # Resampled group control
        changed, show_resampled = psim.Checkbox("Show Resampled", show_resampled)
        if changed:
            resampled_group.set_enabled(show_resampled)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python test_vae_v1.py <dataset_path> <checkpoint_path>")
        sys.exit(1)
    
    # Initialize
    load_model_and_dataset()
    
    # Initialize polyscope
    ps.init()
    resampled_surfaces = {}

    gt_group = ps.create_group("Ground Truth Surfaces")
    recovered_group = ps.create_group("Recovered Surfaces")
    resampled_group = ps.create_group("Resampled Surfaces")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()
