import numpy as np
from process_surface_data_normalization import normalize_surface, denormalize_surface
from src.vae.layers import BSplineSurfaceLayer
from tqdm import tqdm
bsl = BSplineSurfaceLayer(resolution=32).cpu()


def process_surface_data_with_bspline(data_path):
    data = np.load(data_path, allow_pickle=True)
    normalized_sampled_surface = []
    normalized_control_points = []
    for i in tqdm(range(len(data))):
        sampled_surface = bsl(data[i][np.newaxis, :, :])[0]
        normalized_cp, orig_center, orig_scale = normalize_surface(data[i])
        normalized_surface  = bsl(normalized_cp[np.newaxis, :, :])[0]
        denormalized_surface = denormalize_surface(normalized_surface, orig_center, orig_scale)
        is_close = np.allclose(denormalized_surface, sampled_surface, atol=1e-5)
        if not is_close:
            norm_d = np.linalg.norm(sampled_surface)
            if norm_d > 1e-5: # Avoid division by zero or very small norms
                relative_error = np.linalg.norm(sampled_surface - denormalized_surface) / norm_d
                print(f"Surface {i}: Denormalization failed. Relative error: {relative_error:.4e}")
            else:
                absolute_error = np.linalg.norm(sampled_surface - denormalized_surface)
                print(f"Surface {i}: Denormalization failed. Original norm is near zero. Absolute error: {absolute_error:.4e}")
            continue
        normalized_control_points.append(normalized_cp)
        normalized_sampled_surface.append(normalized_surface)
    normalized_sampled_surface = np.stack(normalized_sampled_surface)
    normalized_control_points = np.stack(normalized_control_points)
    np.save(data_path.replace('approx', 'normalized_approx'), normalized_control_points)
    np.save(data_path.replace('approx', 'normalized_sampled_surface'), normalized_sampled_surface)
    # return normalized_surfaces, denorm_params


if __name__ == "__main__":
    # control_points_path = "/home/qindafei/CAD/abc_surfaces_cp/balanced_data/approx_train.npy"
    # control_points_path = "/home/qindafei/CAD/abc_surfaces_cp/approx_val.npy"
    # control_points_path = "/home/qindafei/CAD/abc_surfaces_cp/approx_test.npy"
    control_points_path = "/home/qindafei/CAD/abc_surfaces_cp/approx_train.npy"
    process_surface_data_with_bspline(control_points_path)
