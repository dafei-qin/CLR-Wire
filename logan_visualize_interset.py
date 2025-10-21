import sys
import json
import polyscope as ps
import polyscope.imgui as psim

from itertools import combinations
from utils.surface import visualize_json_interset, sample_line
from utils.surf_tools import surf_surf_interset

if __name__ == "__main__":

    data_path = sys.argv[1]

    with open(data_path, 'r') as f:
        cad_data = json.load(f)
    global_tol = 1e-5 # A good value
    all_faces = visualize_json_interset(cad_data, plot=True, tol=global_tol)


    ps.init()
    all_edges = {}
    all_ps_groups = []
    for idx_m in range(len(all_faces)):
        all_edges[idx_m] = []
        new_group = ps.create_group(f"face_{idx_m:03d}_edges")
        all_ps_groups.append(new_group)
        for idx_n in range(len(all_faces)):


            print('\n', f'Intersecting face with index: {idx_m:03d}-{idx_n:03d}')
            face_m = all_faces[idx_m]['surface']
            face_n = all_faces[idx_n]['surface']
            edges = surf_surf_interset(face_m, face_n, face_group=new_group, tol=global_tol, plot_header=f'face_{idx_m:03d}-{idx_n:03d}')
            all_edges[idx_m].extend(edges)

        new_group.set_enabled(False)
        new_group.set_hide_descendants_from_structure_lists(True)
        new_group.set_show_child_details(False)


    # Forming to wires, unfinished
    all_ps_wire_groups = []
    
    ps.show()



# 




