import torch
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import time
import torch


# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "NURBS_Diff"))

from src.dataset.dataset_bspline import dataset_bspline
try:
    from torch_nurbs_eval.surf_eval import SurfEval
    CUDA_KERNEL_AVAILABLE = True
except ImportError:
    CUDA_KERNEL_AVAILABLE = False
    print("Warning: torch_nurbs_eval not available, will use PyTorch implementation")

from src.utils.surf_eval_torch import SurfEvalTorchFast, SurfEvalTorchVectorized


def expand_knots_from_mults(knots, mults, num_knots):
    """Expand knot vector with multiplicities"""
    expanded = []
    for i in range(num_knots):
        expanded.extend([knots[i]] * int(mults[i]))
    return np.array(expanded)


def convert_to_surfeval_format(u_degree, v_degree, num_poles_u, num_poles_v, 
                                num_knots_u, num_knots_v, u_knots, v_knots, 
                                u_mults, v_mults, poles):
    """Convert dataset_bspline format to SurfEval compatible format"""

    m = num_poles_u.item()
    n = num_poles_v.item()
    p = u_degree.item()
    q = v_degree.item()
    u_knots = torch.cumsum(u_knots, dim=0)
    u_knots = u_knots / u_knots[-1]
    v_knots = torch.cumsum(v_knots, dim=0)
    v_knots = v_knots / v_knots[-1]
    # Expand knot vectors with multiplicities
    knot_u_expanded = expand_knots_from_mults(
        u_knots.numpy(), u_mults.numpy(), num_knots_u.item()
    )
    knot_v_expanded = expand_knots_from_mults(
        v_knots.numpy(), v_mults.numpy(), num_knots_v.item()
    )
    
    # Extract actual poles (remove padding)
    poles_actual = poles[:num_poles_u, :num_poles_v, :]
    
    return m, n, p, q, knot_u_expanded, knot_v_expanded, poles_actual


def test_surfeval_efficiency(data_path_file, num_samples=-1, device='cuda', batch_size=8, use_torch=False):
    """Test SurfEval efficiency on dataset_bspline
    
    Args:
        use_torch: If True, use pure PyTorch implementation instead of CUDA kernels
    """
    
    # Load dataset
    print(f"Loading dataset from {data_path_file}...")
    dataset = dataset_bspline(
        path_file=data_path_file,
        num_surfaces=num_samples,
        max_degree=3,
        max_num_u_knots=32,
        max_num_v_knots=32,
        max_num_u_poles=32,
        max_num_v_poles=32,
        canonical=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Statistics
    total_time = 0
    valid_count = 0
    failed_count = 0
    
    impl_name = "PyTorch" if use_torch else "CUDA Kernel"
    print(f"\nTesting SurfEval ({impl_name}) on {device}...")
    print(f"Batch size: {batch_size}")
    
    # Process dataset
    for idx in tqdm(range(50, len(dataset)), desc="Processing surfaces"):
        # try:
            # Load data
            (u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v,
             is_u_periodic, is_v_periodic, u_knots, v_knots, u_mults, v_mults, 
             poles, valid) = dataset[idx]
             
            if is_u_periodic:
                continue
            if is_v_periodic:
                continue
            
            if not valid:
                continue
            
            # Convert to SurfEval format
            m, n, p, q, knot_u, knot_v, poles_actual = convert_to_surfeval_format(
                u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v,
                u_knots, v_knots, u_mults, v_mults, poles
            )
            
            # Create SurfEval instance
            if use_torch:
                # Use pure PyTorch implementation (vectorized for speed)
                surf_eval = SurfEvalTorchVectorized(
                    u_degree=p, v_degree=q,
                    u_knots=torch.from_numpy(knot_u).float(),
                    v_knots=torch.from_numpy(knot_v).float(),
                    out_dim_u=32, out_dim_v=128,
                    device=device
                )
            else:
                # Use CUDA kernel implementation
                if not CUDA_KERNEL_AVAILABLE:
                    print("ERROR: CUDA kernel not available. Use --use_torch flag.")
                    return
                # if 'surf_eval' in locals():
                #     del surf_eval
                #     # if device == 'cuda':
                #     torch.cuda.empty_cache()
                surf_eval = SurfEval(
                    m=m, n=n, p=p, q=q,
                    knot_u=knot_u, knot_v=knot_v,
                    out_dim_u=32, out_dim_v=128,
                    method='cpp',
                    dvc=device
                )
                print(f"SurfEval instance created: {surf_eval}")
            
            # Prepare input (add batch dimension)
            ctrl_pts = torch.from_numpy(poles_actual.numpy()).float().unsqueeze(0)
            # ctrl_pts = ctrl_pts.repeat(batch_size, 1, 1, 1)  # Create batch
            
            if device == 'cuda':
                ctrl_pts = ctrl_pts.cuda()
            
            # Time forward pass
            # if device == 'cuda':
            #     torch.cuda.synchronize()
            
            start_time = time.time()
            with torch.no_grad():
                surfaces = surf_eval(ctrl_pts)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = time.time() - start_time
            total_time += elapsed
            valid_count += 1
            
        # except Exception as e:
        #     failed_count += 1
        #     # Uncomment to see errors:
        #     print(f"\nFailed on sample {idx}: {e}")
    
    # Report statistics
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Valid samples: {valid_count}")
    print(f"Failed samples: {failed_count}")
    print(f"Total time: {total_time:.3f}s")
    if valid_count > 0:
        print(f"Average time per surface: {total_time/valid_count*1000:.2f}ms")
        print(f"Average time per batch: {total_time/valid_count/batch_size*1000:.2f}ms")
        print(f"Throughput: {valid_count*batch_size/total_time:.2f} surfaces/sec")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NURBS_Diff efficiency')
    parser.add_argument('--data', type=str, default='assets/all_bspline_paths_test.txt',
                        help='Path to data file list')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to test (-1 for all)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--use_torch', action='store_true',
                        help='Use pure PyTorch implementation (no CUDA kernels)')
    
    args = parser.parse_args()
    
    test_surfeval_efficiency(
        data_path_file=args.data,
        num_samples=args.num_samples,
        device=args.device,
        batch_size=args.batch_size,
        use_torch=args.use_torch
    )

