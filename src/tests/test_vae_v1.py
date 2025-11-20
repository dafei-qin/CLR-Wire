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
from src.tools.surface_to_canonical_space import to_canonical, from_canonical
from src.vae.vae_v1 import SurfaceVAE
from src.utils.numpy_tools import orthonormal_basis_from_normal

from utils.surface import visualize_json_interset

def to_json(params_tensor, types_tensor, mask_tensor):
    json_data = []
    # SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    for i in range(len(params_tensor)):
        # (types_tensor[i].item(), mask_tensor[i].sum())
        params = params_tensor[i][mask_tensor[i]]
        # surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        print('surface index: ',i)
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
pipeline_group = None
gt_surfaces = {}
recovered_surfaces = {}
pipeline_surfaces = {}
show_gt = True
show_recovered = True
show_pipeline = True
resampled_surfaces = {}
show_resampled = False

def load_model_and_dataset():
    global dataset, model, max_idx
    
    dataset = dataset_compound(dataset_path, canonical=canonical)
    max_idx = len(dataset) - 1
    
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19]) # Should be changed to [17, 18, 19, 18, 19]
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
    
    params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[idx]
    print('processing file: ', dataset.json_names[idx])
    json_path = dataset.json_names[idx]
    
    mask_bool = mask_tensor.bool()
    mask_np = mask_bool.cpu().numpy().astype(bool)
    # Apply mask to get valid surfaces
    valid_params = params_tensor[mask_bool]
    valid_types = types_tensor[mask_bool]
    shift = shift[mask_np]
    rotation = rotation[mask_np]
    scale = scale[mask_np]
    # Load ground truth JSON data
    with open(json_path, 'r') as f:
        gt_json_data = json.load(f)


    print('-' * 10 + 'gt_json_data' + '-' * 10)
    print(gt_json_data)
    # Run VAE inference
    with torch.no_grad():
        mu, logvar = model.encode(valid_params, valid_types)
        
        # z = model.reparameterize(mu, logvar)
        z = mu
        type_logits_pred, types_pred = model.classify(z)
        params_pred, mask = model.decode(z, types_pred)
        
        # Calculate metrics
        recon_fn = torch.nn.MSELoss()
        recon_loss = (recon_fn(params_pred, valid_params)) * mask.float().mean()
        accuracy = (types_pred == valid_types).float().mean()
        for i in range(valid_types.shape[0]):
            print('surface index: ', i, 'type: ', valid_types[i].item())
            print('input: ', valid_params[i])
            print('output: ', params_pred[i])
            print('diff: ', (params_pred[i] - valid_params[i]))
            print('diff mean: ', (params_pred[i] - valid_params[i]).mean())
            print('-' * 10)
        print(f'Index {idx}: recon_loss: {recon_loss.item():.6f}, accuracy: {accuracy.item():.4f}')
        # print(f'Predicted types: {types_pred.cpu().numpy()}')
        # print(f'Ground truth types: {valid_types.cpu().numpy()}')
    
    # Convert predictions to JSON format
    recovered_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())

    if canonical:
        recovered_json_data = [from_canonical(recovered_json_data[i], shift[i], rotation[i], scale[i]) for i in range(len(recovered_json_data))]

    # Convert dataset pipeline surfaces to JSON (parse -> recover -> optional from canonical)
    pipeline_json_data = []
    for i in range(len(valid_params)):
        recovered_surface = dataset._recover_surface(valid_params[i].cpu().numpy(), valid_types[i].item())
        recovered_surface['idx'] = [i, i]
        recovered_surface['orientation'] = 'Forward'
        if canonical:
            recovered_surface = from_canonical(recovered_surface, shift[i], rotation[i], scale[i])
        pipeline_json_data.append(recovered_surface)

    return gt_json_data, recovered_json_data, pipeline_json_data, recon_loss.item(), accuracy.item()

def resample_model(canonical):
    """Generate new samples from the VAE's latent space"""
    global model, resampled_surfaces, current_idx
    
    if model is None:
        print("Model not loaded yet!")
        return
    
    params_tensor, types_tensor, mask_tensor, shift, rotation, scale = dataset[current_idx]
    valid_params = params_tensor[mask_tensor.bool()]
    valid_types = types_tensor[mask_tensor.bool()]
    
    with torch.no_grad():
        # Sample random latent vectors

        mu, logvar = model.encode(valid_params, valid_types)
        z_random = model.reparameterize(mu, logvar)
        # z_random = mu
        # Classify the random latent vectors to get surface types
        type_logits_pred, types_pred = model.classify(z_random)
        
        # Decode to get surface parameters
        params_pred, mask = model.decode(z_random, types_pred)
        
        # print(f"Generated {len(params_pred)} resampled surfaces")
        # print(f"Resampled types: {types_pred.cpu().numpy()}")
        
        # Convert to JSON format
        resampled_json_data = to_json(params_pred.cpu().numpy(), types_pred.cpu().numpy(), mask.cpu().numpy())
        if canonical:
            resampled_json_data = [from_canonical(resampled_json_data[i], shift[i], rotation[i], scale[i]) for i in range(len(resampled_json_data))]
        with open('./assets/temp/test_vae_v1_runtime_resampled.json', 'w') as f:
            json.dump(resampled_json_data, f)
        # Visualize resampled surfaces
        resampled_surfaces = visualize_json_interset(resampled_json_data, plot=True, plot_gui=False, tol=1e-5, ps_header='resampled')
        
        # Add to resampled group if it exists
        for i, (surface_key, surface_data) in enumerate(resampled_surfaces.items()):
            if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
                surface_data['ps_handler'].add_to_group(resampled_group)
        
        return resampled_json_data, resampled_surfaces

def update_visualization():
    """Update the visualization with current index"""
    global current_idx, gt_group, recovered_group, pipeline_group
    global gt_surfaces, recovered_surfaces, pipeline_surfaces
    
    # Clear existing structures
    ps.remove_all_structures()
    
    # Process current sample
    gt_data, recovered_data, pipeline_data, recon_loss, accuracy = process_sample(current_idx)

    
    # Create groups

    
    # Visualize ground truth surfaces
    try:
        gt_surfaces = visualize_json_interset(gt_data, plot=True, plot_gui=False, tol=1e-5)
    except ValueError:
        print('GT has wrong visualization data!')
        return
    # print(gt_surfaces)
    for i, (surface_key, surface_data) in enumerate(gt_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to ground truth group
            surface_data['ps_handler'].add_to_group(gt_group)
    
    # Visualize dataset pipeline ground truth surfaces
    try:
        pipeline_surfaces = visualize_json_interset(pipeline_data, plot=True, plot_gui=False, tol=1e-5, ps_header='dataset_gt')
    except ValueError:
        print('Pipeline GT has wrong visualization data!')
        return
    for i, (surface_key, surface_data) in enumerate(pipeline_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            surface_data['ps_handler'].add_to_group(pipeline_group)
    
    # Visualize recovered surfaces  
    try:
        recovered_surfaces = visualize_json_interset(recovered_data, plot=True, plot_gui=False, tol=1e-5, ps_header='z_rec')
    except ValueError as e:
        print('Error: ', e, 'Recovered has wrong visualization data!')
        return
    for i, (surface_key, surface_data) in enumerate(recovered_surfaces.items()):
        if 'surface' in surface_data and surface_data['surface'] is not None:
            # Add to recovered group
            surface_data['ps_handler'].add_to_group(recovered_group)
    
    # Configure groups with current visibility settings
    gt_group.set_enabled(show_gt)
    pipeline_group.set_enabled(show_pipeline)
    recovered_group.set_enabled(show_recovered)

    print(f"Visualized {len(gt_surfaces)} GT surfaces, {len(pipeline_surfaces)} dataset GT surfaces and {len(recovered_surfaces)} recovered surfaces")

def callback():
    """Polyscope callback function for UI controls"""
    global current_idx, max_idx, show_gt, show_pipeline, show_recovered, show_resampled, resampled_surfaces
    
    psim.Text("VAE Surface Reconstruction Test")
    psim.Separator()
    
    # Index controls
    slider_changed, slider_idx = psim.SliderInt("Test Index", current_idx, 0, max_idx)
    if slider_changed and slider_idx != current_idx:
        current_idx = slider_idx
        update_visualization()
    
    input_changed, input_idx = psim.InputInt("Go To Index", current_idx)
    if input_changed:
        input_idx = max(0, min(max_idx, input_idx))
        if input_idx != current_idx:
            current_idx = input_idx
            update_visualization()
    
    psim.Separator()
    psim.Text(f"Current Index: {current_idx}")
    psim.Text(f"Max Index: {max_idx}")
    
    # Resample button
    psim.Separator()
    psim.Text("Model Controls:")
    if psim.Button("Resample Model"):
        #  This may lead to some unexpected crash, temporarily remove all structures instead
        # for surface in resampled_surfaces.values():
        #     surface['ps_handler'].remove()
        ps.remove_all_structures()
        resampled_surfaces = {}
        resampled_json_data, resampled_surfaces = resample_model(canonical)
    
    # Group controls
    if gt_group is not None:
        psim.Separator()
        psim.Text("Group Controls:")
        changed, show_gt = psim.Checkbox("Show Ground Truth", show_gt)
        if changed:
            gt_group.set_enabled(show_gt)
        
        changed, show_pipeline = psim.Checkbox("Show Dataset GT", show_pipeline)
        if changed:
            pipeline_group.set_enabled(show_pipeline)

        changed, show_recovered = psim.Checkbox("Show Recovered", show_recovered)
        if changed:
            recovered_group.set_enabled(show_recovered)
        
        # Resampled group control
        changed, show_resampled = psim.Checkbox("Show Resampled", show_resampled)
        if changed:
            resampled_group.set_enabled(show_resampled)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to the checkpoint')
    parser.add_argument('--canonical', type=bool, default=False, help='Whether to use canonical dataset')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    canonical = args.canonical
    
    # Initialize
    load_model_and_dataset()
    
    # Initialize polyscope
    ps.init()
    resampled_surfaces = {}

    gt_group = ps.create_group("Ground Truth Surfaces")
    pipeline_group = ps.create_group("Dataset GT Surfaces")
    recovered_group = ps.create_group("Recovered Surfaces")
    resampled_group = ps.create_group("Resampled Surfaces")
    ps.set_user_callback(callback)
    
    # Load initial visualization
    update_visualization()
    
    # Show the interface
    ps.show()
