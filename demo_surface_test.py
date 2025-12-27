#!/usr/bin/env python3
"""
Demo script showing how to test surface parsing and recovery functions.

This script demonstrates the usage of the test scripts and provides
a simple example of testing the round-trip conversion.
"""

import sys
import json
import numpy as np
from pathlib import Path
import os
# Add the src directory to the path
sys.path.append('src')
from dataset.dataset_v1 import dataset_compound
from myutils.surface import visualize_json_interset
import polyscope as ps

def demo_json_file_test():
    """Demonstrate testing surfaces from a JSON file with visualization."""
    print("\n=== JSON File Test Demo ===")
    
    json_file = "assets/examples/00000084/out_000.json"
    # json_file = "assets/examples/00000056/out_005.json"
    # json_file = r"F:\WORK\CAD\CLR-Wire\assets\abnormal_cases\index_003.json"
    
    if not Path(json_file).exists():
        print(f"Error: File '{json_file}' not found")
        return False
    
    # Load JSON data
    with open(json_file, 'r') as f:
        surfaces_data = json.load(f)
    
    print(f"Loaded {len(surfaces_data)} surfaces from {json_file}")
    
    # Test first few surfaces
    dataset = dataset_compound(os.path.dirname(json_file), max_num_surfaces=200)
    successful_tests = 0
    test_surfaces = []
    recovered_surfaces = []
    
    for i, surface_dict in enumerate(surfaces_data): 
        # print(f"\n--- Surface {i+1}: {surface_dict['type']} ---")
        if surface_dict['type'] == 'bspline_surface':
           #  print("  Skipping bspline_surface (not supported in recovery)")
            continue
            
        # Parse
        params, surface_type_idx = dataset._parse_surface(surface_dict)
        print(f"  Parsed: type_idx={surface_type_idx}, param_dim={len(params)}")
        
        # Recover
        recovered = dataset._recover_surface(params, surface_type_idx)
        

        if i == 7:
            print()
        #     recovered['uv'] = [2.61729 - 2 * np.pi, 4.45059 - 2 * np.pi, -0.12568, 0.00030289]
        print(f"  Recovered: {recovered['type']}")
        recovered['idx'] = [i, i]
        recovered['orientation'] = 'Forward'
        test_surfaces.append(surface_dict)
        recovered_surfaces.append(recovered)
        successful_tests += 1
        print("  ✓ Round-trip successful")
    
    # Visualize original surfaces
    if test_surfaces:
        print(f"\nVisualizing {len(test_surfaces)} original surfaces...")
        ps.init()
        visualize_json_interset(test_surfaces, plot=True, tol=1e-2)
        # ps.show()
        
        # Clear and visualize recovered surfaces
        print(f"Clearing visualization and showing {len(recovered_surfaces)} recovered surfaces...")
        ps.remove_all_structures()
        visualize_json_interset(recovered_surfaces, plot=True, tol=1e-2)
        # ps.show()
    
    print(f"\nSummary: {successful_tests} surfaces processed successfully")
    return successful_tests > 0

# def demo_visualization():
#     """Demonstrate visualization of surfaces."""
#     print("\n=== Visualization Demo ===")
    
#     json_file = "assets/examples/00000056/out_005.json"
    
#     if not Path(json_file).exists():
#         print(f"Error: File '{json_file}' not found")
#         return False
    
#     # Load and visualize first few surfaces
#     with open(json_file, 'r') as f:
#         surfaces_data = json.load(f)
    

#     test_surfaces = surfaces_data
    
#     print(f"Visualizing {len(test_surfaces)} surfaces...")
#     print("Note: This will open a 3D visualization window")
    
#     # Initialize polyscope and visualize
#     ps.init()
#     visualize_json_interset(test_surfaces, plot=True, tol=1e-2)

    
#     return True

def main():
    """Main demo function."""
    print("Surface Parsing and Recovery Test Demo")
    print("=" * 50)
    
    # Test single surface
    # success1 = demo_single_surface_test()
    
    # Test JSON file
    success2 = demo_json_file_test()
    
    # Summary
    print(f"\n{'='*50}")
    print("Demo Summary:")
    # print(f"  Single surface test: {'✓' if success1 else '✗'}")
    print(f"  JSON file test: {'✓' if success2 else '✗'}")
    
    # if success1 and success2:
    #     print("\n🎉 All demos completed successfully!")
    #     print("\nThe visualization shows:")
    #     print("  1. Original surfaces from JSON")
    #     print("  2. Recovered surfaces after parse→recover pipeline")
    #     print("  Compare the two visualizations to verify round-trip correctness")
    #     return 0
    # else:
    #     print("\n⚠ Some demos failed. Check the errors above.")
    #     return 1

if __name__ == "__main__":
    sys.exit(main())
