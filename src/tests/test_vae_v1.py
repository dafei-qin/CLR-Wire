import torch
import numpy as np
import polyscope as ps
import sys
import json
sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')


from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP, SCALAR_DIM_MAP
from src.vae.vae_v1 import SurfaceVAE
from src.utils.numpy_tools import orthonormal_basis_from_normal

from utils.surface import visualize_json_interset

def to_json(params_tensor, types_tensor, mask_tensor):
    json_data = []
    SURFACE_TYPE_MAP_INVERSE = {value: key for key, value in SURFACE_TYPE_MAP.items()}
    for i in range(len(params_tensor)):
        params = params_tensor[i][mask_tensor[i]]
        surface_type = SURFACE_TYPE_MAP_INVERSE[types_tensor[i].item()]
        assert params.shape[0] == 10 + SCALAR_DIM_MAP[surface_type]
        scalar = params[10:]
        scalar = (scalar.abs() + scalar) / 2 + 1e-3
        direction = params[3:6]
        N, X, Y = orthonormal_basis_from_normal(direction.numpy())

        surface_data = {
            'type': surface_type,
            'idx': [i, i],
            'location': [params[:3].numpy().tolist()],
            'direction': np.array([N, X, Y]).tolist(),
            'scalar': scalar.numpy().tolist(),
            'poles': [],
            'uv': params[6:10].numpy().tolist(),
            "orientation": "Forward"

        }

        json_data.append(surface_data)

    return json_data

if __name__ == '__main__':

    dataset = dataset_compound(sys.argv[1])
    # dataset = dataset_compound('/home/qindafei/CAD/data/logan_jsons/abc/0/0000')
    idx = 21
    params_tensor, types_tensor, mask_tensor = dataset[idx]
    json_path = dataset.json_names[idx]
    print(json_path)
    print(params_tensor.shape)
    print(types_tensor.shape)
    print(mask_tensor.shape)

    params_tensor = params_tensor[mask_tensor.bool()]
    types_tensor = types_tensor[mask_tensor.bool()]
    print(params_tensor.shape)


    model = SurfaceVAE(param_raw_dim=[10, 11, 12, 12, 11])
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

    # model.load_state_dict(torch.load(''))
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(params_tensor, types_tensor)
        z = model.reparameterize(mu, logvar)
        type_logits_pred, types_pred = model.classify(z)
        params_pred, mask = model.decode(z, types_pred)

        recon_fn = torch.nn.MSELoss()
        recon_loss = (recon_fn(params_pred, params_tensor)) * mask.float().mean()
        accuracy = (types_pred == types_tensor).float().mean()
        print(f'recon_loss: {recon_loss.item()}, accuracy: {accuracy.item()}')

        visualize_json_interset(to_json(params_pred, types_pred, mask), plot=True)
    


    ps.remove_all_structures()
    json_data = json.load(open(json_path, 'r'))
    print(len(json_data))
    visualize_json_interset(json_data, plot=True)
