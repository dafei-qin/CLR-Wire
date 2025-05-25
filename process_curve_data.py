import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting


# def process_curve_data(data_path):
#     data = np.load(data_path, allow_pickle=True)
#     data = np.stack(data)
#     print(data.shape)


def normalize_curve(curve):
    """
    Normalizes a 3D curve so that it starts at (-1, 0, 0) and ends at (1, 0, 0).

    Args:
        curve (np.ndarray): An (N, 3) numpy array representing the 3D polyline.

    Returns:
        tuple: A tuple containing:
            - normalized_curve (np.ndarray): The (N, 3) normalized curve.
            - p1_original (np.ndarray): The original start point of the curve (3,).
            - R_matrix_for_denormalization (np.ndarray): The 3x3 rotation matrix (R_rodrigues) that was effectively used.
            - original_length_L (float): The original length between the start and end points (||p_n - p_1||).
            - final_translation_applied (np.ndarray): The final translation vector applied (typically [-1,0,0]).
    """
    if not isinstance(curve, np.ndarray) or curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("Input curve must be an (N, 3) numpy array.")
    
    if curve.shape[0] < 1:
        # print("Warning: Curve is empty. Returning empty array and identity transformations.")
        return np.array([]).reshape(0,3), np.zeros(3), np.eye(3), 0.0, np.zeros(3)
    if curve.shape[0] < 2:
        # print("Warning: Curve has only 1 point. Returning original and identity transformations.")
        return curve.copy(), curve[0].copy(), np.eye(3), 0.0, np.zeros(3)

    p1_original = curve[0].copy()
    # pn_original = curve[-1].copy() # Not directly used, p_prime_n is what matters

    translated_curve = curve - p1_original
    p_prime_n = translated_curve[-1] 

    original_length_L = np.linalg.norm(p_prime_n)
    
    if original_length_L < 1e-7: 
        # print(f"Warning: Start and end points are too close (distance: {original_length_L}). Returning original curve.")
        return curve.copy(), p1_original, np.eye(3), 0.0, np.zeros(3)

    p_prime_n_normalized = p_prime_n / original_length_L
    v_target = np.array([1.0, 0.0, 0.0])

    dot_product = np.dot(p_prime_n_normalized, v_target)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = np.arccos(dot_product)

    R_rodrigues = np.eye(3) # This will be the matrix R such that v_rot = R @ v_orig (column vectors)

    if np.abs(theta) < 1e-4: 
        R_rodrigues = np.eye(3)
    elif np.abs(theta - np.pi) < 1e-4: 
        # p_prime_n is anti-parallel to v_target.
        # Rotation by 180 degrees. For v_rot = R @ v_orig, if v_orig=(-L,0,0) and v_rot=(L,0,0),
        # R can be diag([-1, -1, 1]) for rot around Z, or diag([-1, 1, -1]) for rot around Y.
        # Let's use a rotation that flips the x-axis and y-axis, keeping z (e.g. rotate around z by pi)
        # No, if p_prime_n = (-L,y,z), we want it to be (L,y,z) effectively for the x-coord alignment part.
        # A rotation of 180 degrees around an axis perpendicular to p_prime_n_normalized and v_target.
        # If p_prime_n_normalized = (-1,0,0) and v_target = (1,0,0), any axis in y-z plane works, e.g. (0,1,0) (Y-axis).
        # For axis (0,1,0) and angle pi, K = [[0,0,1],[0,0,0],[-1,0,0]]. R_rod = I + 2K^2 = diag([-1,1,-1]).
        R_rodrigues = np.diag([-1.0, 1.0, -1.0]) 
    else: 
        axis_rot_vec = np.cross(p_prime_n_normalized, v_target)
        axis_rot_vec_normalized = axis_rot_vec / np.linalg.norm(axis_rot_vec)
        
        K = np.array([
            [0, -axis_rot_vec_normalized[2], axis_rot_vec_normalized[1]],
            [axis_rot_vec_normalized[2], 0, -axis_rot_vec_normalized[0]],
            [-axis_rot_vec_normalized[1], axis_rot_vec_normalized[0], 0]
        ])
        R_rodrigues = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    
    # R_rodrigues is for v_new = R_rodrigues @ v_old (column vectors)
    # For row vectors P (N,3): P_new = P @ R_rodrigues.T
    R_transform_for_rows = R_rodrigues.T
    rotated_curve = translated_curve @ R_transform_for_rows
    
    scaling_factor = 2.0 / original_length_L
    scaled_curve = rotated_curve * scaling_factor
    
    final_translation_applied = np.array([-1.0, 0.0, 0.0])
    normalized_curve = scaled_curve + final_translation_applied
    
    # We return R_rodrigues itself for denormalization, as it's the fundamental rotation.
    return normalized_curve, p1_original, R_rodrigues, original_length_L, final_translation_applied

def denormalize_curve(normalized_curve, p1_original, R_rodrigues_matrix, original_length_L, final_translation_applied_in_normalize):
    """
    Reverses the normalization process.

    Args:
        normalized_curve (np.ndarray): The (N, 3) normalized curve.
        p1_original (np.ndarray): The original start point (3,).
        R_rodrigues_matrix (np.ndarray): The 3x3 rotation matrix (R_rodrigues from normalize).
        original_length_L (float): The original length ||p_n - p_1||.
        final_translation_applied_in_normalize (np.ndarray): The final translation vector (typically [-1,0,0]).

    Returns:
        np.ndarray: The (N, 3) denormalized (original) curve.
    """
    if normalized_curve.shape[0] < 1: 
        return np.array([]).reshape(0,3)
        
    if original_length_L < 1e-7: 
        return normalized_curve.copy()

    curve_before_final_shift = normalized_curve - final_translation_applied_in_normalize
    
    # This check should ideally not be needed if L=0 case returns early
    if abs(original_length_L) < 1e-7: # Avoid division by zero if somehow L is extremely small but not caught
        scaling_factor_in_normalize = 1.0 # Effectively no scaling if L was ~0
    else:
        scaling_factor_in_normalize = 2.0 / original_length_L 
    
    curve_before_scaling = curve_before_final_shift / scaling_factor_in_normalize

    # To reverse P_new = P_old @ R_rodrigues.T:
    # P_old = P_new @ (R_rodrigues.T).T = P_new @ R_rodrigues
    curve_before_rotation = curve_before_scaling @ R_rodrigues_matrix
    
    denormalized_curve = curve_before_rotation + p1_original
    
    return denormalized_curve

def denormalize_curve_from_endpoints(normalized_curve, p1_original, pn_original):
    """
    Reverses the normalization process using only the normalized curve and original start/end points.

    Args:
        normalized_curve (np.ndarray): The (N, 3) normalized curve.
        p1_original (np.ndarray): The original start point of the curve (3,).
        pn_original (np.ndarray): The original end point of the curve (3,).

    Returns:
        np.ndarray: The (N, 3) denormalized (original) curve.
    """
    if not isinstance(normalized_curve, np.ndarray) or normalized_curve.ndim != 2 or normalized_curve.shape[1] != 3:
        raise ValueError("Input normalized_curve must be an (N, 3) numpy array.")
    p1_original = np.asarray(p1_original, dtype=float)
    pn_original = np.asarray(pn_original, dtype=float)

    if normalized_curve.shape[0] < 1:
        return np.array([]).reshape(0,3)

    vec_orig = pn_original - p1_original
    original_length_L = np.linalg.norm(vec_orig)

    if original_length_L < 1e-7:
        return normalized_curve.copy()

    vec_orig_normalized = vec_orig / original_length_L
    v_target = np.array([1.0, 0.0, 0.0])
    dot_product = np.dot(vec_orig_normalized, v_target)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    theta = np.arccos(dot_product)

    R_rodrigues_matrix = np.eye(3)
    if np.abs(theta) < 1e-4:
        R_rodrigues_matrix = np.eye(3)
    elif np.abs(theta - np.pi) < 1e-4:
        R_rodrigues_matrix = np.diag([-1.0, 1.0, -1.0]) 
    else:
        axis_rot_vec = np.cross(vec_orig_normalized, v_target)
        axis_rot_vec_normalized = axis_rot_vec / np.linalg.norm(axis_rot_vec)
        K = np.array([
            [0, -axis_rot_vec_normalized[2], axis_rot_vec_normalized[1]],
            [axis_rot_vec_normalized[2], 0, -axis_rot_vec_normalized[0]],
            [-axis_rot_vec_normalized[1], axis_rot_vec_normalized[0], 0]
        ])
        R_rodrigues_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    final_translation_applied_in_normalize = np.array([-1.0, 0.0, 0.0])

    curve_before_final_shift = normalized_curve - final_translation_applied_in_normalize
    scaling_factor_in_normalize = 2.0 / original_length_L 
    curve_before_scaling = curve_before_final_shift / scaling_factor_in_normalize
    curve_before_rotation = curve_before_scaling @ R_rodrigues_matrix
    denormalized_curve = curve_before_rotation + p1_original
    
    return denormalized_curve



def process_curve_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    data = np.stack(data)
    failed = 0
    visualization_count = 0
    visualization_count_for_trace = 0
    normalized_curves = []
    vertices = []
    for i, d_curve in enumerate(data):
        d_curve = d_curve.astype(np.float64)
        
        if d_curve.shape[0] < 1:
            print(f"Skipping empty curve at index {i}.")
            continue
            
        original_p1, original_pn = d_curve[0], d_curve[-1]
        
        normalized, _, _, L_returned, _ = normalize_curve(d_curve.copy())
        denormed = denormalize_curve_from_endpoints(normalized.copy(), original_p1, original_pn)
        
        is_close = np.allclose(d_curve, denormed, atol=1e-5)

        if not is_close:
            failed += 1
            norm_d = np.linalg.norm(d_curve)
            if norm_d > 1e-5: # Avoid division by zero or very small norms
                relative_error = np.linalg.norm(d_curve - denormed) / norm_d
                print(f"Curve {i}: Relative error: {relative_error:.4e}")

        normalized_curves.append(normalized)
        vertices.append([d_curve[0], d_curve[-1]])
#                 if relative_error > 1e-3 and visualization_count < 10: # Limit visualizations
#                     visualization_count += 1
#                     print(f"Visualizing curve {i} due to high relative error.")
#                     fig = plt.figure(figsize=(10, 7))
#                     ax = fig.add_subplot(111, projection='3d')
                    
#                     # Original curve
#                     ax.plot(d_curve[:, 0], d_curve[:, 1], d_curve[:, 2], label='Original Curve', marker='o', linestyle='-', color='blue')
#                     ax.scatter(d_curve[0,0], d_curve[0,1], d_curve[0,2], color='green', s=100, label='Original Start', depthshade=False)
#                     ax.scatter(d_curve[-1,0], d_curve[-1,1], d_curve[-1,2], color='cyan', s=100, label='Original End', depthshade=False)

#                     # Denormalized curve
#                     ax.plot(denormed[:, 0], denormed[:, 1], denormed[:, 2], label='Denormalized Curve', marker='x', linestyle='--', color='red')
#                     ax.scatter(denormed[0,0], denormed[0,1], denormed[0,2], color='orange', s=100, label='Denormed Start', depthshade=False, marker='x')
#                     ax.scatter(denormed[-1,0], denormed[-1,1], denormed[-1,2], color='magenta', s=100, label='Denormed End', depthshade=False, marker='x')

#                     ax.set_xlabel('X axis')
#                     ax.set_ylabel('Y axis')
#                     ax.set_zlabel('Z axis')
#                     ax.set_title(f'Curve {i} - Original vs. Denormalized\nRelative Error: {relative_error:.4e}')
#                     ax.legend()

#                     # Equalize axis scaling
#                     x_lims = ax.get_xlim3d()
#                     y_lims = ax.get_ylim3d()
#                     z_lims = ax.get_zlim3d()
#                     x_range = abs(x_lims[1] - x_lims[0])
#                     x_middle = np.mean(x_lims)
#                     y_range = abs(y_lims[1] - y_lims[0])
#                     y_middle = np.mean(y_lims)
#                     z_range = abs(z_lims[1] - z_lims[0])
#                     z_middle = np.mean(z_lims)
#                     plot_radius = 0.5 * max([x_range, y_range, z_range])
#                     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#                     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#                     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#                     plt.show()

#                     # Debug prints for this high-error curve
#                     print(f"--- Debug Info for Curve {i} ---")
#                     # Re-calculate rotation params as denormalize_curve_from_endpoints does
#                     temp_vec_orig = original_pn - original_p1
#                     temp_L = np.linalg.norm(temp_vec_orig)
#                     if temp_L > 1e-7:
#                         temp_vec_orig_norm = temp_vec_orig / temp_L
#                         temp_v_target = np.array([1.0,0.0,0.0])
#                         temp_dot = np.clip(np.dot(temp_vec_orig_norm, temp_v_target), -1.0, 1.0)
#                         temp_theta_rad = np.arccos(temp_dot)
#                         print(f"  Theta (degrees): {np.degrees(temp_theta_rad):.4f}")
#                         temp_R = np.eye(3)
#                         if np.abs(temp_theta_rad) < 1e-7:
#                             print("  Rotation case: theta approx 0 (Identity)")
#                             temp_R = np.eye(3)
#                         elif np.abs(temp_theta_rad - np.pi) < 1e-7:
#                             print("  Rotation case: theta approx pi (diag[-1,1,-1])")
#                             temp_R = np.diag([-1.0, 1.0, -1.0])
#                         else:
#                             temp_axis = np.cross(temp_vec_orig_norm, temp_v_target)
#                             temp_axis_norm = temp_axis / np.linalg.norm(temp_axis)
#                             print(f"  Rotation case: General Rodrigues, Axis: {temp_axis_norm}")
#                             K_dbg = np.array([
#                                 [0, -temp_axis_norm[2], temp_axis_norm[1]],
#                                 [temp_axis_norm[2], 0, -temp_axis_norm[0]],
#                                 [-temp_axis_norm[1], temp_axis_norm[0], 0]
#                             ])
#                             temp_R = np.eye(3) + np.sin(temp_theta_rad) * K_dbg + (1 - np.cos(temp_theta_rad)) * np.dot(K_dbg, K_dbg)
#                         print(f"  R_rodrigues_matrix computed by denormalize_from_endpoints logic:
# {temp_R}")
#                     else:
#                         print("  Original length L is near zero, R is Identity by default.")
#                     print(f"  Normalized curve start: {normalized[0] if len(normalized)>0 else 'N/A'}")
#                     print(f"  Normalized curve end:   {normalized[-1] if len(normalized)>0 else 'N/A'}")
#                     print(f"--- End Debug Info for Curve {i} ---")
#             else:
#                 # Handle cases where original curve norm is very small (e.g. all points at origin)
#                 absolute_error = np.linalg.norm(d_curve - denormed)
#                 print(f"Curve {i}: Original norm is near zero. Absolute error: {absolute_error:.4e}")
#                 if absolute_error > 1e-3 and visualization_count < 10: # If absolute error is high for zero-norm curve
#                     # (Same visualization logic as above)
#                     visualization_count += 1
#                     print(f"Visualizing curve {i} due to high absolute error (original norm near zero).")
#                     fig = plt.figure(figsize=(10, 7))
#                     ax = fig.add_subplot(111, projection='3d')
#                     ax.plot(d_curve[:, 0], d_curve[:, 1], d_curve[:, 2], label='Original Curve (Norm ~0)', marker='o', linestyle='-', color='blue')
#                     ax.plot(denormed[:, 0], denormed[:, 1], denormed[:, 2], label='Denormalized Curve', marker='x', linestyle='--', color='red')
#                     ax.set_xlabel('X axis'); ax.set_ylabel('Y axis'); ax.set_zlabel('Z axis')
#                     ax.set_title(f'Curve {i} - Original vs. Denormalized (Original Norm ~0)\nAbsolute Error: {absolute_error:.4e}')
#                     ax.legend()
#                     # Equalize axis scaling
#                     x_lims = ax.get_xlim3d(); y_lims = ax.get_ylim3d(); z_lims = ax.get_zlim3d()
#                     x_range = abs(x_lims[1] - x_lims[0]); x_middle = np.mean(x_lims)
#                     y_range = abs(y_lims[1] - y_lims[0]); y_middle = np.mean(y_lims)
#                     z_range = abs(z_lims[1] - z_lims[0]); z_middle = np.mean(z_lims)
#                     plot_radius = 0.5 * max([x_range, y_range, z_range])
#                     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#                     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#                     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
#                     plt.show()
#             # Existing debug prints for high-error curve (non-L=0 case)
#             # These prints are still valuable and will appear before the new plot type if triggered.
#             if norm_d > 1e-7 and relative_error > 1e-3 and visualization_count_for_trace < 5 : # Use a new counter for this complex plot
#                 # This is the section for the new detailed trace plot
#                 visualization_count_for_trace += 1
#                 print(f"--- Detailed Trace Visualization for Curve {i} (Rel Error: {relative_error:.4e}) ---")

#                 # 1. Get actual normalization parameters and result
#                 normalized_actual, p1_orig_ret, R_rod_fwd_ret, L_fwd_ret, final_trans_fwd_ret = normalize_curve(d_curve.copy())

#                 fig_trace = plt.figure(figsize=(12, 9))
#                 ax_trace = fig_trace.add_subplot(111, projection='3d')

#                 # Plot Original Curve
#                 ax_trace.plot(d_curve[:, 0], d_curve[:, 1], d_curve[:, 2], label='S0: Original', marker='o', linestyle='-', color='blue', linewidth=2)

#                 if L_fwd_ret > 1e-7:
#                     # Trace Forward Normalization Steps
#                     s1_translated = d_curve - p1_orig_ret
#                     # ax_trace.plot(s1_translated[:, 0], s1_translated[:, 1], s1_translated[:, 2], label='S1: Translated to Origin', linestyle='--', color='lightblue')
                    
#                     R_transform_fwd_rows = R_rod_fwd_ret.T
#                     s2_rotated = s1_translated @ R_transform_fwd_rows
#                     ax_trace.plot(s2_rotated[:, 0], s2_rotated[:, 1], s2_rotated[:, 2], label='S2: Rotated at Origin', linestyle=':', color='purple')

#                     scaling_factor_fwd = 2.0 / L_fwd_ret
#                     s3_scaled = s2_rotated * scaling_factor_fwd
#                     # ax_trace.plot(s3_scaled[:, 0], s3_scaled[:, 1], s3_scaled[:, 2], label='S3: Scaled at Origin', linestyle='--', color='lightgreen')
                    
#                     # s4_normalized is normalized_actual
#                     ax_trace.plot(normalized_actual[:, 0], normalized_actual[:, 1], normalized_actual[:, 2], label='S4: Fully Normalized', marker='s', linestyle='-', color='green', linewidth=2)
#                 else: # L_fwd_ret is near zero, normalize_curve returned original or single point version
#                     ax_trace.plot(normalized_actual[:, 0], normalized_actual[:, 1], normalized_actual[:, 2], label='S4: Normalized (L~0 case)', marker='s', linestyle='-', color='green', linewidth=2)

#                 # Trace Backward Denormalization Steps (starting from normalized_actual)
#                 # Get denormalization parameters as denormalize_curve_from_endpoints would:
#                 p1_for_denorm = d_curve[0]
#                 pn_for_denorm = d_curve[-1]
#                 vec_orig_denorm = pn_for_denorm - p1_for_denorm
#                 L_for_denorm = np.linalg.norm(vec_orig_denorm)
#                 R_rod_bwd = np.eye(3)

#                 if L_for_denorm > 1e-7:
#                     vec_orig_norm_denorm = vec_orig_denorm / L_for_denorm
#                     v_target_denorm = np.array([1.0, 0.0, 0.0])
#                     dot_denorm = np.clip(np.dot(vec_orig_norm_denorm, v_target_denorm), -1.0, 1.0)
#                     theta_denorm = np.arccos(dot_denorm)
#                     if np.abs(theta_denorm) < 1e-7:
#                         R_rod_bwd = np.eye(3)
#                     elif np.abs(theta_denorm - np.pi) < 1e-7:
#                         R_rod_bwd = np.diag([-1.0, 1.0, -1.0])
#                     else:
#                         axis_denorm = np.cross(vec_orig_norm_denorm, v_target_denorm)
#                         axis_norm_denorm = axis_denorm / np.linalg.norm(axis_denorm)
#                         K_denorm = np.array([[0, -axis_norm_denorm[2], axis_norm_denorm[1]],
#                                            [axis_norm_denorm[2], 0, -axis_norm_denorm[0]],
#                                            [-axis_norm_denorm[1], axis_norm_denorm[0], 0]])
#                         R_rod_bwd = np.eye(3) + np.sin(theta_denorm) * K_denorm + (1 - np.cos(theta_denorm)) * np.dot(K_denorm, K_denorm)
                    
#                     final_trans_bwd = np.array([-1.0, 0.0, 0.0])
#                     s5_rev_trans = normalized_actual - final_trans_bwd
#                     # ax_trace.plot(s5_rev_trans[:,0], s5_rev_trans[:,1], s5_rev_trans[:,2], label='S5: Denorm Shifted from -1', linestyle='--', color='pink')

#                     scaling_factor_bwd = 2.0 / L_for_denorm
#                     s6_rev_scale = s5_rev_trans / scaling_factor_bwd
#                     ax_trace.plot(s6_rev_scale[:,0], s6_rev_scale[:,1], s6_rev_scale[:,2], label='S6: Denorm Unscaled at Origin', linestyle=':', color='orange')
                    
#                     s7_rev_rotate = s6_rev_scale @ R_rod_bwd # R_rod_bwd is for P_new = R @ P_old; for rows P_new = P_old @ R.T? No, denorm uses P_old = P_new @ R_rod
#                                                               # denormalize_curve_from_endpoints uses ... @ R_rodrigues_matrix.
#                                                               # R_rod_bwd calculated here IS that R_rodrigues_matrix.
#                     # ax_trace.plot(s7_rev_rotate[:,0], s7_rev_rotate[:,1], s7_rev_rotate[:,2], label='S7: Denorm Unrotated at Origin', linestyle='--', color='brown')
                    
#                     # s8_final_denormalized is the 'denormed' variable we already have
#                     ax_trace.plot(denormed[:, 0], denormed[:, 1], denormed[:, 2], label=f'S8: Final Denormalized (Err: {relative_error:.2e})', marker='x', linestyle='-', color='red', linewidth=2)
#                 else: # L_for_denorm is near zero
#                     ax_trace.plot(denormed[:, 0], denormed[:, 1], denormed[:, 2], label=f'S8: Final Denormalized (L_denorm~0, Err: {relative_error:.2e})', marker='x', linestyle='-', color='red', linewidth=2)
                
#                 ax_trace.set_xlabel('X'); ax_trace.set_ylabel('Y'); ax_trace.set_zlabel('Z')
#                 ax_trace.set_title(f'Curve {i} - Transformation Trace')
#                 ax_trace.legend(fontsize='small')
#                 # Equalize axis scaling
#                 x_lims = ax_trace.get_xlim3d(); y_lims = ax_trace.get_ylim3d(); z_lims = ax_trace.get_zlim3d()
#                 x_range = abs(x_lims[1] - x_lims[0]); x_middle = np.mean(x_lims)
#                 y_range = abs(y_lims[1] - y_lims[0]); y_middle = np.mean(y_lims)
#                 z_range = abs(z_lims[1] - z_lims[0]); z_middle = np.mean(z_lims)
#                 plot_radius_trace = 0.5 * max([x_range, y_range, z_range])
#                 ax_trace.set_xlim3d([x_middle - plot_radius_trace, x_middle + plot_radius_trace])
#                 ax_trace.set_ylim3d([y_middle - plot_radius_trace, y_middle + plot_radius_trace])
#                 ax_trace.set_zlim3d([z_middle - plot_radius_trace, z_middle + plot_radius_trace])
#                 plt.show()

        # Increment the main visualization counter if any plot was shown (either simple or trace)
        # if not is_close and ( (norm_d > 1e-7 and relative_error > 1e-3) or (norm_d <= 1e-7 and absolute_error > 1e-3) ):
        #      if previous_visualization_count_snapshot != visualization_count or visualization_count_for_trace > 0:
        #          pass # visualization_count already incremented for simple plot, or trace plot shown

    print(f"Failed: {failed}, Total: {len(data)}, {failed/len(data)*100 if len(data) > 0 else 0:.2f}%")
    return normalized_curves, vertices
#     # if __name__ == "__main__":



if __name__ == "__main__":
    '''
    This script processes the curve data and saves the normalized curves and vertices.
    '''
    # data_path = ['../abc_split/points_train.npy', '../abc_split/points_val.npy', '../abc_split/points_test.npy']    
    data_path = ['../curve_wireframe_split/edge_points_concat_train.npy', '../curve_wireframe_split/edge_points_concat_val.npy', '../curve_wireframe_split/edge_points_concat_test.npy']    
    

    # for path in data_path:
    #     normalized_curves, vertices = process_curve_data(path)
    #     np.save(path.replace('points', 'normalized_points'), normalized_curves)
    #     np.save(path.replace('points', 'vertices'), vertices)


    for path in data_path:
        normalized_curves, vertices = process_curve_data(path)
        np.save(path.replace('edge_points_concat', 'normalized_edge_points_concat'), normalized_curves)
        # np.save(path.replace('points', 'vertices'), vertices)
