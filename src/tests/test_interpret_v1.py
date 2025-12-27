import sys
import os
import argparse
import json
import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim

sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound
from src.vae.vae_v1 import SurfaceVAE
from myutils.surface import visualize_json_interset


def load_model(checkpoint_path: str) -> SurfaceVAE:
    model = SurfaceVAE(param_raw_dim=[17, 18, 19, 18, 19])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if 'ema_model' in checkpoint:
        ema_model = checkpoint['ema']
        ema_model = {k.replace("ema_model.", ""): v for k, v in ema_model.items()}
        model.load_state_dict(ema_model, strict=False)
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def to_json_single(dataset: dataset_compound, params_tensor: torch.Tensor, type_idx: int):
    # params_tensor is 1D processed param vector
    params_np = params_tensor.detach().cpu().numpy()
    recovered = dataset._recover_surface(params_np, int(type_idx))
    recovered['idx'] = [0, 0]
    recovered['orientation'] = 'Forward'
    return [recovered]


class InterpApp:
    def __init__(self, json_path: str, checkpoint_path: str):
        # Dataset from the directory containing the target json
        json_dir = os.path.dirname(json_path)
        self.ds = dataset_compound(json_dir)
        self.file_idx = None
        for i, p in enumerate(self.ds.json_names):
            if os.path.abspath(p) == os.path.abspath(json_path):
                self.file_idx = i
                break
        if self.file_idx is None:
            raise FileNotFoundError(f"JSON file not found in dataset: {json_path}")

        # Load model
        self.model = load_model(checkpoint_path)

        # Prepare data for the chosen file
        params_tensor, types_tensor, mask_tensor = self.ds[self.file_idx]
        valid = mask_tensor.bool()
        self.params_valid = params_tensor[valid]
        self.types_valid = types_tensor[valid]

        with torch.no_grad():
            self.mu, self.logvar = self.model.encode(self.params_valid, self.types_valid)

        self.num_valid = self.params_valid.shape[0]

        # UI state
        self.idx1 = 0
        self.idx2 = min(1, max(0, self.num_valid - 1))
        self.weight = 0.5

        # Polyscope structures
        self.g_src1 = ps.create_group("Source 1")
        self.g_src2 = ps.create_group("Source 2")
        self.g_interp = ps.create_group("Interpolated")

        self.src1_faces = {}
        self.src2_faces = {}
        self.interp_faces = {}

        # Initial render
        self.update_sources()
        self.update_interpolation()

    def decode_and_visualize(self, z: torch.Tensor, group, header: str):
        # z: (D,) latent
        z = z.unsqueeze(0)
        with torch.no_grad():
            type_logits_pred, types_pred = self.model.classify(z)
            params_pred, mask = self.model.decode(z, types_pred)

        # Build json and visualize
        rec_json = to_json_single(self.ds, params_pred[0], int(types_pred[0].item()))
        faces = visualize_json_interset(rec_json, plot=True, plot_gui=False, tol=1e-5, ps_header=header)
        for _, surface_data in faces.items():
            if 'surface' in surface_data and surface_data['surface'] is not None and 'ps_handler' in surface_data:
                surface_data['ps_handler'].add_to_group(group)
        return faces

    def clear_faces(self, faces_dict):
        for surf in faces_dict.values():
            if 'ps_handler' in surf and surf['ps_handler'] is not None:
                try:
                    surf['ps_handler'].remove()
                except Exception:
                    pass

    def update_sources(self):
        ps.remove_all_structures()
        # Source 1
        if self.num_valid > 0:
            z1 = self.mu[self.idx1]
            self.src1_faces = self.decode_and_visualize(z1, self.g_src1, 'z_src1')
        # Source 2
        if self.num_valid > 1:
            z2 = self.mu[self.idx2]
            self.src2_faces = self.decode_and_visualize(z2, self.g_src2, 'z_src2')

    def update_interpolation(self):
        if self.num_valid == 0:
            return
        z1 = self.mu[self.idx1]
        z2 = self.mu[self.idx2 if self.num_valid > 1 else self.idx1]
        z = z1 * self.weight + z2 * (1.0 - self.weight)
        self.interp_faces = self.decode_and_visualize(z, self.g_interp, 'z_interp')

    def ui(self):
        psim.Text("Latent Interpretability (two-point interpolation)")
        psim.Separator()
        psim.Text(f"Valid surfaces in file: {self.num_valid}")

        changed1, new_idx1 = psim.SliderInt("Latent index 1", self.idx1, 0, max(0, self.num_valid - 1))
        changed2, new_idx2 = psim.SliderInt("Latent index 2", self.idx2, 0, max(0, self.num_valid - 1))
        changed_w, new_w = psim.SliderFloat("Weight w (Z = z1*w + z2*(1-w))", self.weight, 0.0, 1.0)

        need_update_sources = False
        need_update_interp = False

        if changed1:
            self.idx1 = int(np.clip(new_idx1, 0, max(0, self.num_valid - 1)))
            need_update_sources = True
        if changed2:
            self.idx2 = int(np.clip(new_idx2, 0, max(0, self.num_valid - 1)))
            need_update_sources = True
        if changed_w:
            self.weight = float(new_w)
            need_update_interp = True

        if need_update_sources:
            self.update_sources()
            self.update_interpolation()
        elif need_update_interp:
            self.update_interpolation()


def main():
    parser = argparse.ArgumentParser(description='Interactive latent interpolation visualization')
    parser.add_argument('json_path', type=str, help='Path to a single JSON file')
    parser.add_argument('checkpoint_path', type=str, help='Path to VAE checkpoint (.pt)')
    args = parser.parse_args()

    ps.init()
    app = InterpApp(args.json_path, args.checkpoint_path)
    ps.set_user_callback(app.ui)
    ps.show()


if __name__ == '__main__':
    main()


