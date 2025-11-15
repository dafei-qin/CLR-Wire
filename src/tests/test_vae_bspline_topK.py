import sys
import os
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Type
from types import SimpleNamespace

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.dataset.dataset_bspline import dataset_bspline
from torch.utils.data import DataLoader
from src.utils.config import NestedDictToClass, load_config


def to_python_int(x: torch.Tensor) -> int:
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def normalize_path(path_value: Optional[str]) -> str:
    """
    Convert a potentially relative path to an absolute string path.
    Returns empty string if the input is None or empty.
    """
    if path_value is None:
        return ""
    path_str = str(path_value).strip()
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return str(path)


def load_model(ckpt_path: str, device: torch.device, model_class: Type, model_kwargs: Optional[Dict[str, Any]] = None):
    model = model_class(**(model_kwargs or {})).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = None
    # Robustly handle a few common checkpoint layouts
    if isinstance(checkpoint, dict):
        if "ema_model" in checkpoint:
            # Some checkpoints save EMA with prefixed keys
            ema_state = checkpoint["ema_model"]
            state = {k.replace("ema_model.", ""): v for k, v in ema_state.items()}
        elif "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
            # Fallback if stored under 'ema'
            ema_state = checkpoint["ema"]
            # Try nested 'model' first
            inner = ema_state.get("model", ema_state)
            state = {k.replace("model.", ""): v for k, v in inner.items()}
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        else:
            # Assume raw state_dict
            state = checkpoint
    else:
        state = checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"Warning: missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def sample_to_batch_tensors(sample) -> Any:
    (
        u_degree,
        v_degree,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_list,
        v_mults_list,
        poles,
        valid,
    ) = sample

    if not valid:
        return None

    # Batch dim
    u_degree = u_degree.unsqueeze(0).unsqueeze(-1).long()
    v_degree = v_degree.unsqueeze(0).unsqueeze(-1).long()
    num_poles_u = num_poles_u.unsqueeze(0).unsqueeze(-1).long()
    num_poles_v = num_poles_v.unsqueeze(0).unsqueeze(-1).long()
    num_knots_u = num_knots_u.unsqueeze(0).unsqueeze(-1).long()
    num_knots_v = num_knots_v.unsqueeze(0).unsqueeze(-1).long()
    is_u_periodic = is_u_periodic.unsqueeze(0).unsqueeze(-1).long()
    is_v_periodic = is_v_periodic.unsqueeze(0).unsqueeze(-1).long()

    u_knots_list = u_knots_list.unsqueeze(0).float()
    v_knots_list = v_knots_list.unsqueeze(0).float()
    u_mults_list = u_mults_list.unsqueeze(0).long()
    v_mults_list = v_mults_list.unsqueeze(0).long()
    poles = poles.unsqueeze(0).float()

    # 0-based targets for embeddings where needed
    u_degree_input = u_degree.clone() - 1
    v_degree_input = v_degree.clone() - 1
    u_mults_input = u_mults_list.clone()
    v_mults_input = v_mults_list.clone()
    u_mults_input[u_mults_input > 0] -= 1
    v_mults_input[v_mults_input > 0] -= 1

    return (
        u_degree_input,
        v_degree_input,
        num_poles_u,
        num_poles_v,
        num_knots_u,
        num_knots_v,
        is_u_periodic,
        is_v_periodic,
        u_knots_list,
        v_knots_list,
        u_mults_input,
        v_mults_input,
        poles,
        # Also return originals for logging/DF
        u_degree,
        v_degree,
    )


@torch.no_grad()
def evaluate_dataset(
    dataset_kwargs: Dict[str, Any],
    ckpt_path: str,
    device: torch.device,
    model_class: Type,
    model_kwargs: Optional[Dict[str, Any]] = None,
    topk: int = 0,
    batch_size: int = 16,
    num_workers: int = 0,
) -> pd.DataFrame:
    ds = dataset_bspline(**dataset_kwargs)
    model = load_model(ckpt_path, device, model_class=model_class, model_kwargs=model_kwargs or {})
    rows: List[Dict[str, Any]] = []

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    global_ptr = 0  # track dataset indices for mapping to paths (no shuffle)

    for batch in tqdm(dl, desc="Evaluating surfaces"):
        forward_args = list(batch)
        valid_mask = forward_args[-1].bool()  # [B]

        # Batch indices this iteration
        B_total = valid_mask.shape[0]
        batch_indices = list(range(global_ptr, global_ptr + B_total))
        valid_indices = [batch_indices[i] for i in range(B_total) if bool(valid_mask[i].item())]
        global_ptr += B_total  # advance by total batch size regardless of validity

        # Filter tensors to only valid samples (exclude last 'valid' entry)
        forward_args = [_[valid_mask] for _ in forward_args[:-1]]

        # If no valid samples, skip
        if len(forward_args) == 0 or forward_args[0].numel() == 0:
            continue

        # Match preprocessing in trainer/main
        forward_args = [_.unsqueeze(-1) if len(_.shape) == 1 else _ for _ in forward_args]
        forward_args = [_.float() if _.dtype == torch.float64 else _ for _ in forward_args]

        (
            u_degree,
            v_degree,
            num_poles_u,
            num_poles_v,
            num_knots_u,
            num_knots_v,
            is_u_periodic,
            is_v_periodic,
            u_knots_list,
            v_knots_list,
            u_mults_list,
            v_mults_list,
            poles,
        ) = forward_args

        # Keep GT degrees for reporting; create model inputs (0-based)
        u_degree_gt = u_degree.clone()
        v_degree_gt = v_degree.clone()
        u_mults_in = u_mults_list.clone().long()
        v_mults_in = v_mults_list.clone().long()

        u_degree_in = (u_degree - 1).long()  # start from 0
        v_degree_in = (v_degree - 1).long()
        u_mults_in[u_mults_in > 0] -= 1
        v_mults_in[v_mults_in > 0] -= 1
        num_knots_u = num_knots_u.long()
        num_knots_v = num_knots_v.long()
        num_poles_u = num_poles_u.long()
        num_poles_v = num_poles_v.long()

        # Move to device
        tensors = [
            u_degree_in, v_degree_in,
            num_poles_u, num_poles_v,
            num_knots_u, num_knots_v,
            is_u_periodic, is_v_periodic,
            u_knots_list, v_knots_list,
            u_mults_in, v_mults_in,
            poles,
            u_degree_gt, v_degree_gt,
        ]
        (
            u_degree_in, v_degree_in,
            num_poles_u, num_poles_v,
            num_knots_u, num_knots_v,
            is_u_periodic, is_v_periodic,
            u_knots_list, v_knots_list,
            u_mults_in, v_mults_in,
            poles,
            u_degree_gt, v_degree_gt,
        ) = [t.to(device) for t in tensors]

        # Encode and decode with GT counts (as in trainer loss computation)
        mu, logvar = model.encode(
            u_knots_list,
            u_mults_in,
            v_knots_list,
            v_mults_in,
            poles,
            u_degree_in,
            v_degree_in,
            is_u_periodic,
            is_v_periodic,
            num_knots_u,
            num_knots_v,
            num_poles_u,
            num_poles_v,
        )
        z = mu  # deterministic evaluation

        (
            deg_logits_u,
            deg_logits_v,
            peri_logits_u,
            peri_logits_v,
            knots_num_logits_u,
            knots_num_logits_v,
            pred_knots_u,
            pred_knots_v,
            mults_logits_u,
            mults_logits_v,
            pred_poles,
        ) = model.decode(
            z,
            num_knots_u,
            num_knots_v,
            num_poles_u,
            num_poles_v,
        )

        # Compute masked poles MSE per sample, split into xyz mean and w
        B = pred_poles.shape[0]
        max_u = model.max_num_u_poles
        max_v = model.max_num_v_poles
        mask_u = (torch.arange(max_u, device=device).unsqueeze(0) < num_poles_u)  # [B, max_u]
        mask_v = (torch.arange(max_v, device=device).unsqueeze(0) < num_poles_v)  # [B, max_v]
        mask_2d = mask_u.unsqueeze(-1) & mask_v.unsqueeze(-2)  # [B, H, W]
        mask_4d = mask_2d.unsqueeze(-1)  # [B, H, W, 1]

        se = (pred_poles - poles) ** 2  # [B, H, W, 4]
        valid_count = mask_2d.view(B, -1).sum(dim=1).clamp(min=1)  # [B]

        # xyz mean MSE across coords and all valid poles
        se_xyz = se[..., :3] * mask_4d[..., :3]  # [B, H, W, 3]
        loss_xyz = se_xyz.view(B, -1).sum(dim=1) / (valid_count * 3)  # [B]

        # w MSE across all valid poles
        se_w = se[..., 3] * mask_4d.squeeze(-1)  # [B, H, W]
        loss_w = se_w.view(B, -1).sum(dim=1) / valid_count  # [B]

        # Relative losses: mean of (se / |gt|) over valid items
        eps = 1e-6
        gt_abs_xyz = poles[..., :3].abs()  # [B, H, W, 3]
        gt_abs_w = poles[..., 3].abs()     # [B, H, W]
        valid_count_xyz = (valid_count * 3).clamp(min=1)  # [B]
        se_xyz_rel = (se[..., :3] / (gt_abs_xyz + eps)) * mask_4d[..., :3]  # [B, H, W, 3]
        se_w_rel = (se[..., 3] / (gt_abs_w + eps)) * mask_2d               # [B, H, W]
        loss_rel_xyz = se_xyz_rel.view(B, -1).sum(dim=1) / valid_count_xyz  # [B]
        loss_rel_w = se_w_rel.view(B, -1).sum(dim=1) / valid_count          # [B]

        # Store rows per valid sample
        for b, ds_idx in enumerate(valid_indices):
            surface_path = ds.data_names[ds_idx].strip() if hasattr(ds, "data_names") else str(ds_idx)
            row = {
                "deg_u": to_python_int(u_degree_gt[b, 0]),
                "deg_v": to_python_int(v_degree_gt[b, 0]),
                "peri_u": int(is_u_periodic[b, 0].item()),
                "peri_v": int(is_v_periodic[b, 0].item()),
                "num_knots_u": to_python_int(num_knots_u[b, 0]),
                "num_knots_v": to_python_int(num_knots_v[b, 0]),
                "num_poles_u": to_python_int(num_poles_u[b, 0]),
                "num_poles_v": to_python_int(num_poles_v[b, 0]),
                "loss_poles_mean_xyz": float(loss_xyz[b].item()),
                "loss_poles_w": float(loss_w[b].item()),
                "loss_poles_rel_xyz": float(loss_rel_xyz[b].item()),
                "loss_poles_rel_w": float(loss_rel_w[b].item()),
                "surface_path": surface_path,
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "deg_u", "deg_v",
        "peri_u", "peri_v",
        "num_knots_u", "num_knots_v",
        "num_poles_u", "num_poles_v",
        "loss_poles_mean_xyz", "loss_poles_w",
        "loss_poles_rel_xyz", "loss_poles_rel_w",
        "surface_path",
    ])

    if topk and topk > 0 and len(df) > 0:
        # Just to be handy in CLI: print/show worst-K by xyz loss
        worst = df.sort_values("loss_poles_mean_xyz", ascending=False).head(topk)
        print("\nTop-K worst by loss_poles_mean_xyz:")
        print(worst.to_string(index=False))

    return df


def _plot_cdf(values: np.ndarray, title: str, xlabel: str, save_path: Path):
    if values.size == 0:
        return
    sorted_vals = np.sort(values)
    y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_vals, y, lw=2)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_pdf(values: np.ndarray, title: str, xlabel: str, save_path: Path, bins: int = 100):
    if values.size == 0:
        return
    # Prepare data
    vals_linear = values[np.isfinite(values)]
    vals_log = values[np.isfinite(values) & (values > 0)]
    if vals_linear.size == 0:
        return
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10, 4))
    # Left: linear-scale PDF
    ax_lin.hist(vals_linear, bins=bins, density=True, alpha=0.85, edgecolor="white")
    ax_lin.set_xlabel(xlabel)
    ax_lin.set_ylabel("PDF")
    ax_lin.set_title(f"{title} (Linear)")
    ax_lin.grid(True, ls="--", alpha=0.4)
    # Right: log-log PDF with log-spaced bins
    if vals_log.size > 0:
        vmin = np.min(vals_log)
        vmax = np.max(vals_log)
        if vmin > 0 and vmax > vmin:
            edges = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)
            counts, edges = np.histogram(vals_log, bins=edges, density=True)
            counts = np.where(counts > 0, counts, np.nan)  # avoid log(0)
            centers = np.sqrt(edges[:-1] * edges[1:])
            ax_log.plot(centers, counts, "-", lw=2)
            ax_log.set_xscale("log")
            ax_log.set_yscale("log")
    ax_log.set_xlabel(xlabel)
    ax_log.set_ylabel("PDF")
    ax_log.set_title(f"{title} (Log-Log)")
    ax_log.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _plot_bar_with_err(groups: pd.DataFrame, title: str, xlabel: str, ylabel: str, save_path: Path):
    # groups: columns ['key', 'mean', 'std']
    if groups.empty:
        return
    keys = groups["key"].tolist()
    means = groups["mean"].tolist()
    stds = groups["std"].tolist()

    x = np.arange(len(keys))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=4, alpha=0.8)
    plt.xticks(x, [str(k) for k in keys])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_errline(x_vals: np.ndarray, mean: np.ndarray, std: np.ndarray, title: str, xlabel: str, ylabel: str, save_path: Path):
    if x_vals.size == 0:
        return
    plt.figure(figsize=(7, 4))
    plt.errorbar(x_vals, mean, yerr=std, fmt="-o", capsize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_summaries(df: pd.DataFrame, out_dir: Path):
    if df is None or len(df) == 0:
        print("No data to plot.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CDFs
    _plot_cdf(
        df["loss_poles_mean_xyz"].values,
        title="CDF of loss_poles_mean_xyz",
        xlabel="loss_poles_mean_xyz",
        save_path=out_dir / "cdf_loss_poles_mean_xyz.png",
    )
    _plot_cdf(
        df["loss_poles_w"].values,
        title="CDF of loss_poles_w",
        xlabel="loss_poles_w",
        save_path=out_dir / "cdf_loss_poles_w.png",
    )
    # Also: PDF of loss_poles_mean_xyz
    _plot_pdf(
        df["loss_poles_mean_xyz"].values,
        title="PDF of loss_poles_mean_xyz",
        xlabel="loss_poles_mean_xyz",
        save_path=out_dir / "pdf_loss_poles_mean_xyz.png",
    )

    # 2) Avg ± std vs periodic (u and v), separate plots for xyz and w
    for peri_col in ["peri_u", "peri_v"]:
        # xyz
        grp_xyz = (
            df.groupby(peri_col)["loss_poles_mean_xyz"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={peri_col: "key"})
        )
        _plot_bar_with_err(
            grp_xyz,
            title=f"Mean±Std loss_poles_mean_xyz by {peri_col}",
            xlabel=peri_col,
            ylabel="loss_poles_mean_xyz",
            save_path=out_dir / f"avgstd_xyz_by_{peri_col}.png",
        )
        # w
        grp_w = (
            df.groupby(peri_col)["loss_poles_w"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={peri_col: "key"})
        )
        _plot_bar_with_err(
            grp_w,
            title=f"Mean±Std loss_poles_w by {peri_col}",
            xlabel=peri_col,
            ylabel="loss_poles_w",
            save_path=out_dir / f"avgstd_w_by_{peri_col}.png",
        )

    # 3) Avg ± std vs num_knots and num_poles (u and v), line with error bars
    for count_col, label in [
        ("num_knots_u", "num_knots_u"),
        ("num_knots_v", "num_knots_v"),
        ("num_poles_u", "num_poles_u"),
        ("num_poles_v", "num_poles_v"),
    ]:
        # xyz
        agg_xyz = df.groupby(count_col)["loss_poles_mean_xyz"].agg(["mean", "std"]).reset_index()
        agg_xyz = agg_xyz.sort_values(count_col)
        _plot_errline(
            agg_xyz[count_col].values,
            agg_xyz["mean"].values,
            agg_xyz["std"].values,
            title=f"Mean±Std loss_poles_mean_xyz vs {label}",
            xlabel=label,
            ylabel="loss_poles_mean_xyz",
            save_path=out_dir / f"avgstd_xyz_vs_{label}.png",
        )
        # w
        agg_w = df.groupby(count_col)["loss_poles_w"].agg(["mean", "std"]).reset_index()
        agg_w = agg_w.sort_values(count_col)
        _plot_errline(
            agg_w[count_col].values,
            agg_w["mean"].values,
            agg_w["std"].values,
            title=f"Mean±Std loss_poles_w vs {label}",
            xlabel=label,
            ylabel="loss_poles_w",
            save_path=out_dir / f"avgstd_w_vs_{label}.png",
        )

    # 4) Class-wise mean and P90 with counts for num_knots and num_poles (u and v)
    def plot_class_mean_p90_with_counts(class_col: str, label: str, metric_col: str, filename_suffix: str):
        if class_col not in df.columns:
            return
        grouped = df.groupby(class_col)[metric_col]
        stats = grouped.agg(
            mean="mean",
            p90=lambda x: np.quantile(x, 0.9),
            count="count"
        ).reset_index().rename(columns={class_col: "key"})
        stats = stats.sort_values("key")
        if stats.empty:
            return
        x = stats["key"].values
        mean_vals = stats["mean"].values
        p90_vals = stats["p90"].values
        counts = stats["count"].values

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()

        ax1.plot(x, mean_vals, "-o", label="Mean", color="tab:blue")
        ax1.plot(x, p90_vals, "-s", label="P90", color="tab:orange")
        ax1.set_xlabel(label)
        ax1.set_ylabel(metric_col)
        ax1.grid(True, ls="--", alpha=0.4)

        ax2.bar(x, counts, alpha=0.3, color="tab:gray", label="# Surfaces")
        ax2.set_ylabel("# Surfaces")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title(f"{metric_col}: Mean and P90 vs {label} (with counts)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_suffix}.png", dpi=200)
        plt.close()

    for class_col, label in [
        ("num_knots_u", "num_knots_u"),
        ("num_knots_v", "num_knots_v"),
        ("num_poles_u", "num_poles_u"),
        ("num_poles_v", "num_poles_v"),
    ]:
        plot_class_mean_p90_with_counts(class_col, label, "loss_poles_mean_xyz", f"class_mean_p90_counts_xyz_vs_{label}")
        # If needed later, we can also plot the same for w:
        # plot_class_mean_p90_with_counts(class_col, label, "loss_poles_w", f"class_mean_p90_counts_w_vs_{label}")

    # 5) Class-wise mean and P90 with counts split by periodicity, in the same figure for comparison
    def plot_class_mean_p90_counts_by_periodic(class_col: str, periodic_col: str, label: str, metric_col: str, filename_suffix: str):
        if class_col not in df.columns or periodic_col not in df.columns:
            return
        # Compute stats per periodic flag
        stats = (
            df.groupby([periodic_col, class_col])[metric_col]
            .agg(mean="mean", p90=lambda x: np.quantile(x, 0.9), count="count")
            .reset_index()
            .rename(columns={class_col: "key", periodic_col: "peri"})
        )
        if stats.empty:
            return
        # Ensure both peri classes appear
        keys_all = np.array(sorted(stats["key"].unique()))
        peri_vals = sorted(stats["peri"].unique())
        # Build aligned arrays
        def select(peri_value):
            sub = stats[stats["peri"] == peri_value].set_index("key")
            m = np.array([sub["mean"].get(k, np.nan) for k in keys_all])
            p = np.array([sub["p90"].get(k, np.nan) for k in keys_all])
            c = np.array([sub["count"].get(k, 0) for k in keys_all])
            return m, p, c
        m0, p0, c0 = select(0) if 0 in peri_vals else (np.full_like(keys_all, np.nan, dtype=float),)*2 + (np.zeros_like(keys_all, dtype=float),)
        m1, p1, c1 = select(1) if 1 in peri_vals else (np.full_like(keys_all, np.nan, dtype=float),)*2 + (np.zeros_like(keys_all, dtype=float),)

        # Plot lines for mean and p90, counts as grouped bars on twin axis
        x = np.arange(len(keys_all))
        width = 0.4
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax2 = ax1.twinx()

        ax1.plot(keys_all, m0, "-o", color="tab:blue", label="Mean (peri=0)")
        ax1.plot(keys_all, p0, "--o", color="tab:blue", alpha=0.7, label="P90 (peri=0)")
        ax1.plot(keys_all, m1, "-s", color="tab:orange", label="Mean (peri=1)")
        ax1.plot(keys_all, p1, "--s", color="tab:orange", alpha=0.7, label="P90 (peri=1)")
        ax1.set_xlabel(label)
        ax1.set_ylabel(metric_col)
        ax1.grid(True, ls="--", alpha=0.4)

        ax2.bar(keys_all - width/2, c0, width=width, alpha=0.25, color="tab:blue", label="# (peri=0)")
        ax2.bar(keys_all + width/2, c1, width=width, alpha=0.25, color="tab:orange", label="# (peri=1)")
        ax2.set_ylabel("# Surfaces")

        lines, labels_1 = ax1.get_legend_handles_labels()
        lines2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels_1 + labels_2, loc="best")
        plt.title(f"{metric_col}: Mean & P90 vs {label} by {periodic_col} (with counts)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_suffix}.png", dpi=200)
        plt.close()

    # Pair u-periodic with u classes, v-periodic with v classes
    plot_class_mean_p90_counts_by_periodic("num_knots_u", "peri_u", "num_knots_u", "loss_poles_mean_xyz", "class_mean_p90_counts_xyz_vs_num_knots_u_by_peri_u")
    plot_class_mean_p90_counts_by_periodic("num_poles_u", "peri_u", "num_poles_u", "loss_poles_mean_xyz", "class_mean_p90_counts_xyz_vs_num_poles_u_by_peri_u")
    plot_class_mean_p90_counts_by_periodic("num_knots_v", "peri_v", "num_knots_v", "loss_poles_mean_xyz", "class_mean_p90_counts_xyz_vs_num_knots_v_by_peri_v")
    plot_class_mean_p90_counts_by_periodic("num_poles_v", "peri_v", "num_poles_v", "loss_poles_mean_xyz", "class_mean_p90_counts_xyz_vs_num_poles_v_by_peri_v")

    # 6) Relative losses vs num_poles_u / num_poles_v (plot xyz_relative and w_relative in same figure)
    def plot_rel_vs_num_poles(count_col: str, label: str, filename_suffix: str):
        if count_col not in df.columns:
            return
        agg_xyz = df.groupby(count_col)["loss_poles_rel_xyz"].mean().reset_index().sort_values(count_col)
        agg_w = df.groupby(count_col)["loss_poles_rel_w"].mean().reset_index().sort_values(count_col)
        if agg_xyz.empty or agg_w.empty:
            return
        x = agg_xyz[count_col].values
        y_xyz = agg_xyz["loss_poles_rel_xyz"].values
        y_w = agg_w["loss_poles_rel_w"].values
        plt.figure(figsize=(7, 4))
        # Replace non-positive values with NaN to safely use log scale
        y_xyz_plot = np.where(y_xyz > 0, y_xyz, np.nan)
        y_w_plot = np.where(y_w > 0, y_w, np.nan)
        plt.plot(x, y_xyz_plot, "-o", label="rel_xyz (mean)")
        plt.plot(x, y_w_plot, "-s", label="rel_w (mean)")
        plt.yscale("log")
        plt.xlabel(label)
        plt.ylabel("Relative Loss (mean of se/|gt|)")
        plt.title(f"Relative Loss vs {label}")
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_suffix}.png", dpi=200)
        plt.close()

    plot_rel_vs_num_poles("num_poles_u", "num_poles_u", "rel_xyz_w_vs_num_poles_u")
    plot_rel_vs_num_poles("num_poles_v", "num_poles_v", "rel_xyz_w_vs_num_poles_v")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BSplineVAE across a dataset and build distribution dataframe (config-driven).")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/train_vae_bspline_v1_full.yaml",
        help="Path to the YAML config file (defaults to train_vae_bspline_v1_full).",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Override model name from config.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Override checkpoint file path.")
    parser.add_argument("--num_surfaces", type=int, default=None, help="Override number of surfaces to evaluate.")
    parser.add_argument(
        "--path_file_or_dir",
        type=str,
        default=None,
        help="Override dataset source (text file of paths or directory containing .npy files).",
    )
    parser.add_argument("--device", type=str,  default="cpu",  help="Device to run on")
    parser.add_argument("--output_csv", type=str, default="", help="Path to save CSV (default: alongside checkpoint)")
    parser.add_argument("--topk", type=int, default=0, help="Optionally print top-K by loss_poles_mean_xyz")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config) if args.config else {}
    cfg_args = NestedDictToClass(cfg) if cfg else SimpleNamespace()

    data_cfg = getattr(cfg_args, "data", SimpleNamespace())
    model_cfg = getattr(cfg_args, "model", SimpleNamespace())

    def _getattr(obj, attr, default=None):
        return getattr(obj, attr, default) if obj is not None else default

    # Resolve model class by name
    model_name = args.model_name if args.model_name is not None else _getattr(model_cfg, "name", "vae_bspline_v1")
    if model_name == 'vae_bspline_v1':
        from src.vae.vae_bspline import BSplineVAE as ModelClass
        print('Use the model: vae_bspline_v1')
    elif model_name == 'vae_bspline_v3':
        from src.vae.vae_bspline_v3 import BSplineVAE as ModelClass
        print('Use the model: vae_bspline_v3')
    elif model_name == "vae_bspline_v4":
        from src.vae.vae_bspline_v4 import BSplineVAE as ModelClass
        print('Use the model: vae_bspline_v4')
    elif model_name == "vae_bspline_v5":
        from src.vae.vae_bspline_v5 import BSplineVAE as ModelClass
        print('Use the model: vae_bspline_v5')
    else:
        from src.vae.vae_bspline import BSplineVAE as ModelClass
        print('Use the default model: vae_bspline_v1')

    # Determine number of surfaces
    num_surfaces_cfg = _getattr(data_cfg, "val_num", _getattr(data_cfg, "train_num", -1))
    num_surfaces = args.num_surfaces if args.num_surfaces is not None else (num_surfaces_cfg if num_surfaces_cfg is not None else -1)

    # Resolve dataset source (file list or directory)
    default_path_file = _getattr(data_cfg, "val_file", _getattr(data_cfg, "train_file", ""))
    default_data_dir = _getattr(data_cfg, "val_data_dir_override", _getattr(data_cfg, "train_data_dir_override", ""))

    dataset_path_override = args.path_file_or_dir
    dataset_path_file = ""
    dataset_data_dir = ""

    if dataset_path_override:
        resolved_override = normalize_path(dataset_path_override)
        if os.path.isdir(resolved_override):
            dataset_data_dir = resolved_override
        else:
            dataset_path_file = resolved_override
    else:
        default_dir_resolved = normalize_path(default_data_dir)
        default_path_resolved = normalize_path(default_path_file)
        if default_dir_resolved:
            dataset_data_dir = default_dir_resolved
        elif default_path_resolved:
            dataset_path_file = default_path_resolved

    if not dataset_path_file and not dataset_data_dir:
        raise ValueError("Unable to determine dataset source. Provide it via the config file or --path_file_or_dir.")

    # Build dataset kwargs (align geometry limits with model config)
    dataset_kwargs: Dict[str, Any] = {
        "path_file": dataset_path_file,
        "data_dir_override": dataset_data_dir,
        "num_surfaces": num_surfaces,
        "max_num_u_knots": _getattr(model_cfg, "max_num_u_knots", 64),
        "max_num_v_knots": _getattr(model_cfg, "max_num_v_knots", 32),
        "max_num_u_poles": _getattr(model_cfg, "max_num_u_poles", 64),
        "max_num_v_poles": _getattr(model_cfg, "max_num_v_poles", 32),
        "max_degree": _getattr(model_cfg, "max_degree", 3),
    }

    # Prepare model kwargs
    model_kwarg_keys = [
        "max_degree",
        "embd_dim",
        "num_query",
        "mults_dim",
        "max_num_u_knots",
        "max_num_v_knots",
        "max_num_u_poles",
        "max_num_v_poles",
    ]
    model_kwargs: Dict[str, Any] = {key: getattr(model_cfg, key) for key in model_kwarg_keys if hasattr(model_cfg, key)}
    for dim_key in ["max_degree", "max_num_u_knots", "max_num_v_knots", "max_num_u_poles", "max_num_v_poles"]:
        if dim_key not in model_kwargs and dim_key in dataset_kwargs:
            model_kwargs[dim_key] = dataset_kwargs[dim_key]

    # Resolve checkpoint path
    ckpt_path = normalize_path(args.ckpt_path) if args.ckpt_path else ""
    if not ckpt_path:
        ckpt_folder = _getattr(model_cfg, "checkpoint_folder", "")
        ckpt_file_name = _getattr(model_cfg, "checkpoint_file_name", "")
        if ckpt_folder or ckpt_file_name:
            combined_ckpt = os.path.join(str(ckpt_folder), ckpt_file_name) if ckpt_folder and ckpt_file_name else (ckpt_folder or ckpt_file_name)
            ckpt_path = normalize_path(combined_ckpt)
    if not ckpt_path:
        raise ValueError("Checkpoint path must be provided via the config file or --ckpt_path.")

    device = args.device
    df = evaluate_dataset(
        dataset_kwargs=dataset_kwargs,
        ckpt_path=ckpt_path,
        device=device,
        model_class=ModelClass,
        model_kwargs=model_kwargs,
        topk=args.topk,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.output_csv:
        out_csv = Path(args.output_csv)
    else:
        ckpt_p = Path(ckpt_path)
        out_csv = ckpt_p.with_suffix("").with_name(ckpt_p.stem + "_bspline_eval.csv")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")
    # Plot summaries next to CSV
    plot_summaries(df, out_csv.parent)


if __name__ == "__main__":
    main()


