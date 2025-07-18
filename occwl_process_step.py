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
import numpy as np
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
import torch
from tqdm import tqdm

import polyscope as ps

if __name__ == '__main__':
    solids = load_step(r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\00000056_666139e3bff64d4e8a6ce183_step_005.step")

    ps.init()
    for idx, solid in enumerate(solids):
        if idx == 2:
            print(idx)
            solid = solid.topods_shape()
            solid = Solid(solid)
            graph = face_adjacency(solid, self_loops=True)
            for face_idx in graph.nodes():
                face = graph.nodes[face_idx]["face"]
                surf_type = face.surface_type()
                surface = face.specific_surface()
                uv_box = face.uv_bounds()
                u_min = uv_box.intervals[0].interpolate(0)
                u_max = uv_box.intervals[0].interpolate(1)
                v_min = uv_box.intervals[1].interpolate(0)
                v_max = uv_box.intervals[1].interpolate(1)
                uv_values, uv_grid = uvgrid(face, method="point", num_u=32, num_v=32, uvs=True)
                position = surface.Position().Location()
                print(f"\t{face_idx:03d}_{surf_type} {u_min:.2f} {u_max:.2f} {u_max - u_min:.2f} {v_min:.2f} {v_max:.2f} {v_max - v_min:.2f} {position.X():.2f} {position.Y():.2f} {position.Z():.2f}")
                ps.register_point_cloud(f"{face_idx:03d}_{surf_type}", uv_values.reshape(-1, 3))
    ps.show()
