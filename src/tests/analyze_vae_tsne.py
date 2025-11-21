import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.append('/home/qindafei/CAD/CLR-Wire')
sys.path.append(r'C:\drivers\CAD\CLR-Wire')
sys.path.append(r'F:\WORK\CAD\CLR-Wire')

from src.dataset.dataset_v1 import dataset_compound, SURFACE_TYPE_MAP_INV
from src.vae.vae_v1 import SurfaceVAE


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


def collect_latents(ds: dataset_compound, model: SurfaceVAE, limit: int = None):
    all_mu = []
    all_labels = []
    all_pred = []

    with torch.no_grad():
        N = len(ds) if limit is None else min(limit, len(ds))
        for idx in range(N):
            params_tensor, types_tensor, mask_tensor = ds[idx]
            valid = mask_tensor.bool()
            if valid.sum() == 0:
                continue
            params = params_tensor[valid]
            types = types_tensor[valid]

            mu, logvar = model.encode(params, types)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(types.cpu().numpy())

            # Optionally record predicted label to inspect separability
            type_logits_pred, types_pred = model.classify(mu)
            all_pred.append(types_pred.cpu().numpy())

    if not all_mu:
        return np.zeros((0, 2), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    mu_arr = np.concatenate(all_mu, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return mu_arr, y_true, y_pred


def run_tsne(X: np.ndarray, perplexity: float = 30.0, random_state: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    # t-SNE works better with some initial dimensionality reduction in many cases
    # but we keep it simple unless needed.
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state)
    Z = tsne.fit_transform(X)
    return Z


def plot_tsne(Z: np.ndarray, y: np.ndarray, out_png: str, title: str = 'VAE latent t-SNE'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    num_classes = int(y.max()) + 1 if y.size > 0 else 0
    cmap = plt.get_cmap('tab10')
    for c in range(num_classes):
        mask = (y == c)
        if mask.sum() == 0:
            continue
        plt.scatter(Z[mask, 0], Z[mask, 1], s=8, color=cmap(c % 10), label=SURFACE_TYPE_MAP_INV[c])
    plt.legend(markerscale=2, fontsize=8, frameon=False)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    print(f"Saved t-SNE plot to {out_png}")


def main():
    parser = argparse.ArgumentParser(description='t-SNE analysis of VAE latent space')
    parser.add_argument('dataset_path', type=str, help='Path to dataset directory containing JSONs')
    parser.add_argument('checkpoint_path', type=str, help='Path to VAE checkpoint (.pt)')
    parser.add_argument('--limit', type=int, default=100, help='Max number of files to sample')
    parser.add_argument('--perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--out', type=str, default='assets/temp/vae_tsne.png', help='Output PNG path')
    args = parser.parse_args()

    ds = dataset_compound(args.dataset_path)
    model = load_model(args.checkpoint_path)

    print('Collecting latent embeddings...')
    X, y_true, y_pred = collect_latents(ds, model, limit=args.limit)
    print(f"Collected {X.shape[0]} embeddings; classes: {set(y_true.tolist()) if y_true.size>0 else set()}")
    if X.shape[0] == 0:
        print('No valid embeddings collected. Exiting.')
        return

    print('Running t-SNE... (this can take a while)')
    Z = run_tsne(X, perplexity=args.perplexity)

    print('Plotting...')
    plot_tsne(Z, y_true, args.out)


if __name__ == '__main__':
    main()


