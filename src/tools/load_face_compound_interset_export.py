import sys
import polyscope as ps
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.surface import write_to_step
from utils.surf_tools import surf_surf_interset
from occwl.compound import Compound

from occwl.io import load_step, save_step
from occwl.graph import face_adjacency

if __name__ == '__main__':

    file_name = 'assets/to_zeyu/00100001/detokenized_surfaces_idx_0000.step'
    compound = Compound.load_from_step(file_name)
    g = face_adjacency(compound, self_loops=True)
    all_faces = []
    all_edges = []
    for idx in g.nodes():
        all_faces.append(g.nodes[idx]['face'].topods_shape())

    for i in range(len(all_faces) - 1):
        for j in range(i + 1, len(all_faces)):
            face_i = all_faces[i]
            face_j = all_faces[j]
            edges = surf_surf_interset(face_i, face_j, plot=False)
            if edges is not None:
                for edge in edges:
                    # print((edge))
                    all_edges.append(edge)

    print(all_edges)
    write_to_step(all_faces, 'assets/to_zeyu/00100001/detokenized_surfaces_idx_0000_interset.step', edges=all_edges)
    # ps.show()