"""
Profile the SurfEval implementation to find bottlenecks
"""
import torch
import time
import numpy as np
from surf_eval_torch import SurfEvalTorchFast


def profile_surf_eval():
    device = 'cuda'
    
    # Create test case
    u_degree, v_degree = 3, 3
    num_u_poles, num_v_poles = 11, 11
    
    u_knots = torch.cat([
        torch.zeros(u_degree),
        torch.linspace(0, 1, num_u_poles - u_degree + 1),
        torch.ones(u_degree)
    ])
    v_knots = torch.cat([
        torch.zeros(v_degree),
        torch.linspace(0, 1, num_v_poles - v_degree + 1),
        torch.ones(v_degree)
    ])
    
    ctrl_pts = torch.randn(4, num_u_poles, num_v_poles, 4).to(device)
    ctrl_pts[..., 3] = 1.0
    
    # Create evaluator
    surf_eval = SurfEvalTorchFast(
        u_degree, v_degree, u_knots, v_knots,
        out_dim_u=32, out_dim_v=128, device=device
    )
    
    # Warm up
    for _ in range(3):
        _ = surf_eval(ctrl_pts)
    
    torch.cuda.synchronize()
    
    # Profile
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        surfaces = surf_eval(ctrl_pts)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Total time for {num_runs} runs: {elapsed:.3f}s")
    print(f"Average time per run: {elapsed/num_runs*1000:.2f}ms")
    print(f"Batch size: {ctrl_pts.shape[0]}")
    print(f"Time per surface: {elapsed/num_runs/ctrl_pts.shape[0]*1000:.2f}ms")
    print(f"Output grid: {surf_eval.out_dim_u} x {surf_eval.out_dim_v} = {surf_eval.out_dim_u * surf_eval.out_dim_v} points")


if __name__ == '__main__':
    profile_surf_eval()


