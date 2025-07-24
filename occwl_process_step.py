import argparse
import random

from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface, Geom_BezierCurve, Geom_BSplineCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Lin, gp_Circ, gp_Elips, gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus, gp_Hypr, gp_Parab, \
    gp_Pnt, gp_Trsf
from occwl.solid import Solid
from occwl.uvgrid import uvgrid, ugrid
from occwl.graph import face_adjacency
from occwl.io import load_step, save_step
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from occwl.viewer import Viewer
from occwl.edge import Edge
import numpy as np
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from tqdm import tqdm

import polyscope as ps

if __name__ == '__main__':
    # solids = load_step(r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\00000056_666139e3bff64d4e8a6ce183_step_005.step")
    solids = load_step(r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000032\00000032_ad34a3f60c4a4caa99646600_step_012.step")

    ps.init()
    v = Viewer(backend="wx")

    for idx, solid in enumerate(solids):
        # if idx == 5:
            print(idx)
            solid = solid.topods_shape()
            solid = Solid(solid)
            graph = face_adjacency(solid, self_loops=True)
            face_grids = {}
            for face_idx in graph.nodes():
                # if face_idx == 11:
                    face = graph.nodes[face_idx]["face"]
                    surf_type = face.surface_type()
                    surface = face.specific_surface()
                    uv_box = face.uv_bounds()
                    u_min = uv_box.intervals[0].interpolate(0)
                    u_max = uv_box.intervals[0].interpolate(1)
                    v_min = uv_box.intervals[1].interpolate(0)
                    v_max = uv_box.intervals[1].interpolate(1)
                    uv_values, uv_grid = uvgrid(face, method="point", num_u=32, num_v=32, uvs=True)
                    # position = surface.Position().Location()
                    # print(f"\t{face_idx:03d}_{surf_type} {u_min:.2f} {u_max:.2f} {u_max - u_min:.2f} {v_min:.2f} {v_max:.2f} {v_max - v_min:.2f} {position.X():.2f} {position.Y():.2f} {position.Z():.2f}")
                    uv_grid = uv_grid.reshape(-1, 2)
                    uv_grid = np.concatenate([uv_grid, np.zeros((uv_grid.shape[0], 1))], axis=1)
                    # ps.register_point_cloud(f"{face_idx:03d}_{surf_type}", uv_values.reshape(-1, 3))
                    # ps.register_point_cloud(f"{face_idx:03d}_{surf_type}_grid", uv_grid)
                    points = uvgrid(face, num_u=10, num_v=10, method="point")
                    mask = uvgrid(face, num_u=10, num_v=10, method="inside")
                    normals = uvgrid(face, num_u=10, num_v=10, method="normal")
                    face_grids[face_idx] = {"points": points, "normals": normals, "mask": mask}
    # ps.show()


            bbox = solid.box()
            point_radius = bbox.max_box_length() * 0.03
            arrow_radius = point_radius * 0.85
            arrow_length = arrow_radius * 2

            v.display(solid, transparency=0.6, color=(0.2, 0.2, 0.2))
            face_centers = {}
            for face_idx in graph.nodes():
                # Display a sphere for each UV-grid point
                face = graph.nodes[face_idx]["face"]
                grid = face_grids[face_idx]
                # Display points
                face_points = grid["points"].reshape((-1, 3))
                face_mask = grid["mask"].reshape(-1)
                face_points = face_points[face_mask, :]

                # v.display_points(face_points, marker="point", color="GREEN")
                # Display normals
                face_normals = grid["normals"].reshape((-1, 3))
                face_normals = face_normals[face_mask, :]
                lines = [Edge.make_line_from_points(pt, pt + arrow_length * nor) for pt, nor in zip(face_points, face_normals)]
                for l in lines:
                    # v.display(l, color="RED")
                    pass
                face_centers[face_idx] = grid["points"][4, 4]

            for fi, fj in graph.edges():
                pt1 = face_centers[fi]
                pt2 = face_centers[fj]
                dist = np.linalg.norm(pt2 - pt1)
                if dist > 1e-3:
                    pass
                    # v.display(Edge.make_line_from_points(pt1, pt2), color=(51.0 / 255.0, 0, 1))

            # Show the viewer
    v.fit()
    v.show()