import argparse
import random

from OCC.Core.Geom import Geom_BezierSurface, Geom_BSplineSurface, Geom_BezierCurve, Geom_BSplineCurve
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_PointsToBSpline
from OCC.Core.GeomAbs import GeomAbs_C2
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.gp import gp_Lin, gp_Circ, gp_Elips, gp_Pln, gp_Cylinder, gp_Cone, gp_Sphere, gp_Torus, gp_Hypr, gp_Parab, \
    gp_Pnt, gp_Trsf
from occwl import Solid, uvgrid, ugrid
from occwl.graph import face_adjacency
from occwl.io import load_step, save_step
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
import numpy as np
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
import torch
from tqdm import tqdm
import zlw, os, json

from multiprocessing.pool import Pool

from utils import load_step_with_timeout, load_abc_step, load_furniture_step

'''
Logan's script to convert .step to json
'''

class BRepDataProcessor:
    def line_feature(self, curve: gp_Lin, idx):
        origin = curve.Location()
        direction = curve.Direction()
        return {
            "type": "line",
            "idx": idx,
            "location": [[origin.X(), origin.Y(), origin.Z()]],
            "direction": [[direction.X(), direction.Y(), direction.Z()]],
            "scalar": [],
            "poles": [],
        }

    def circle_feature(self, curve: gp_Circ, idx):
        center = curve.Location()
        normal = curve.Axis().Direction()
        radius = curve.Radius()
        return {
            "type": "circle",
            "idx": idx,
            "location": [[center.X(), center.Y(), center.Z()]],
            "direction": [[normal.X(), normal.Y(), normal.Z()]],
            "scalar": [radius],
            "poles": [],
        }

    def ellipse_feature(self, curve: gp_Elips, idx):
        center = curve.Location()
        x_axis = curve.XAxis().Direction()
        y_axis = curve.YAxis().Direction()
        major_radius = curve.MajorRadius()
        minor_radius = curve.MinorRadius()
        return {
            "type": "ellipse",
            "idx": idx,
            "location": [[center.X(), center.Y(), center.Z()]],
            "direction": [[x_axis.X(), x_axis.Y(), x_axis.Z()], [y_axis.X(), y_axis.Y(), y_axis.Z()]],
            "scalar": [major_radius, minor_radius],
            "poles": [],
        }

    def hyperbola_feature(self, curve: gp_Hypr, idx):
        center = curve.Location()
        axis = curve.Axis().Direction()
        major_radius = curve.MajorRadius()
        minor_radius = curve.MinorRadius()
        return {
            "type": "hyperbola",
            "idx": idx,
            "location": [[center.X(), center.Y(), center.Z()]],
            "direction": [[axis.X(), axis.Y(), axis.Z()]],
            "scalar": [major_radius, minor_radius],
            "poles": [],
        }

    def parabola_feature(self, curve: gp_Parab, idx):
        center = curve.Location()
        axis = curve.Axis().Direction()
        focal_length = curve.Focal()
        return {
            "type": "parabola",
            "idx": idx,
            "location": [[center.X(), center.Y(), center.Z()]],
            "direction": [[axis.X(), axis.Y(), axis.Z()]],
            "scalar": [focal_length],
            "poles": [],
        }

    def bezier_curve_feature(self, curve: Geom_BezierCurve, idx):
        # is_rational = curve.IsRational()
        # if not is_rational:
        #     raise ValueError("Non-rational Bezier curves are not supported.")
        degree = curve.Degree()
        # if degree > 8:
        #     raise ValueError("Degree of Bezier curve is too high.")
        # if degree < 4:
        #     curve.Increase(4)
        poles = [[] for _ in range(degree + 1)]
        for i in range(1, degree + 2):
            pole = curve.Pole(i)
            weight = curve.Weight(i)
            poles[i - 1] = [pole.X(), pole.Y(), pole.Z(), weight]
        return {
            "type": "bezier_curve",
            "idx": idx,
            "location": [],
            "direction": [],
            "poles": poles,
            "scalar": [curve.Degree()],
        }

    def bspline_curve_feature(self, curve: Geom_BSplineCurve, idx):
        degree = curve.Degree()
        num_poles = curve.NbPoles()
        num_knots = curve.NbKnots()
        # is_rational = curve.IsRational()
        # if not is_rational:
        #     raise ValueError("The curve is not rational.")
        # if degree > 8:
        #     raise ValueError("Degree is greater than 8.")
        # if num_poles > 8:
        #     raise ValueError("Number of poles is greater than 8.")
        # if num_knots > 8:
        #     raise ValueError("Number of knots is greater than 8.")
        # if num_poles != num_knots:
        #     raise ValueError("Number of poles and knots are not equal.")

        knots = TColStd_Array1OfReal(1, num_knots)
        curve.Knots(knots)
        mults = TColStd_Array1OfInteger(1, num_knots)
        curve.Multiplicities(mults)
        poles = [[] for _ in range(num_poles)]
        for i in range(1, num_poles + 1):
            pole = curve.Pole(i)
            weight = curve.Weight(i)
            poles[i - 1] = [pole.X(), pole.Y(), pole.Z(), weight]

        knots_list = [knots.Value(i) for i in range(1, num_knots + 1)]
        mults_list = [mults.Value(i) for i in range(1, num_knots + 1)]

        return {
            "type": "bspline_curve",
            "idx": idx,
            "location": [],
            "direction": [],
            "poles": poles,
            "scalar": [curve.Degree(), num_poles, num_knots] + knots_list + mults_list,
        }

    def plane_feature(self, surface: gp_Pln, idx):
        origin = surface.Location()
        normal = surface.Axis().Direction()
        return {
            "type": "plane",
            "idx": idx,
            "location": [[origin.X(), origin.Y(), origin.Z()]],
            "direction": [[normal.X(), normal.Y(), normal.Z()]],
            "scalar": [],
            "poles": [],
        }

    def cylinder_feature(self, surface: gp_Cylinder, idx):
        axis = surface.Axis().Direction()
        position = surface.Position().Location()
        radius = surface.Radius()
        return {
            "type": "cylinder",
            "idx": idx,
            "location": [[position.X(), position.Y(), position.Z()]],
            "direction": [[axis.X(), axis.Y(), axis.Z()]],
            "scalar": [radius],
            "poles": [],
        }

    def cone_feature(self, surface: gp_Cone, idx):
        axis = surface.Axis().Direction()
        position = surface.Position().Location()
        semi_angle = surface.SemiAngle()
        radius = surface.RefRadius()
        return {
            "type": "cone",
            "idx": idx,
            "location": [[position.X(), position.Y(), position.Z()]],
            "direction": [[axis.X(), axis.Y(), axis.Z()]],
            "scalar": [semi_angle, radius],
            "poles": [],
        }

    def sphere_feature(self, surface: gp_Sphere, idx):
        center = surface.Location()
        radius = surface.Radius()
        return {
            "type": "sphere",
            "idx": idx,
            "location": [[center.X(), center.Y(), center.Z()]],
            "direction": [],
            "scalar": [radius],
            "poles": [],
        }

    def torus_feature(self, surface: gp_Torus, idx):
        axis = surface.Axis().Direction()
        position = surface.Position().Location()
        major_radius = surface.MajorRadius()
        minor_radius = surface.MinorRadius()
        return {
            "type": "torus",
            "idx": idx,
            "location": [[position.X(), position.Y(), position.Z()]],
            "direction": [[axis.X(), axis.Y(), axis.Z()]],
            "scalar": [major_radius, minor_radius],
            "poles": [],
        }

    def bezier_surface_feature(self, surface: Geom_BezierSurface, idx):
        # is_rational = surface.IsURational() and surface.IsVRational()
        # if not is_rational:
        #     raise ValueError("Non-rational Bezier surfaces are not supported.")
        u_degree = surface.UDegree()
        v_degree = surface.VDegree()
        if u_degree > 8 or v_degree > 8:
            raise ValueError("Degree of Bezier surface is too high.")
        # if u_degree < 4 or v_degree < 4:
        #     surface.Increase(4, 4)
        poles = [[[] for _ in range(v_degree + 1)] for _ in range(u_degree + 1)]
        for u in range(1, u_degree + 2):
            for v in range(1, v_degree + 2):
                pole = surface.Pole(u, v)
                weight = surface.Weight(u, v)
                poles[u - 1][v - 1] = [pole.X(), pole.Y(), pole.Z(), weight]

        return {
            "type": "bezier_surface",
            "idx": idx,
            "location": [],
            "direction": [],
            "poles": poles,
            "scalar": [surface.UDegree(), surface.VDegree()]
        }

    def bspline_surface_feature(self, surface: Geom_BSplineSurface, idx):

        # if surface.UDegree() < 4 or surface.VDegree() < 4:
        #     surface.IncreaseDegree(4, 4)

        u_degree = surface.UDegree()
        v_degree = surface.VDegree()

        num_poles_u = surface.NbUPoles()
        num_poles_v = surface.NbVPoles()

        num_knots_u = surface.NbUKnots()
        num_knots_v = surface.NbVKnots()

        is_rational = surface.IsURational() and surface.IsVRational()

        # if not is_rational:
        #     raise ValueError("The surface is not rational.")
        # if u_degree > 8 or v_degree > 8:
        #     raise ValueError("Degree is greater than 8.")
        # if num_poles_u > 8 or num_poles_v > 8:
        #     raise ValueError("Number of poles is greater than 8.")
        # if num_knots_u > 8 or num_knots_v > 8:
        #     raise ValueError("Number of knots is greater than 8.")

        u_knots = TColStd_Array1OfReal(1, num_knots_u)
        v_knots = TColStd_Array1OfReal(1, num_knots_v)
        surface.UKnots(u_knots)
        surface.VKnots(v_knots)

        u_mults = TColStd_Array1OfInteger(1, num_knots_u)
        v_mults = TColStd_Array1OfInteger(1, num_knots_v)
        surface.UMultiplicities(u_mults)
        surface.VMultiplicities(v_mults)

        poles = [[[] for _ in range(num_poles_v)] for _ in range(num_poles_u)]
        for u in range(1, num_poles_u + 1):
            for v in range(1,  num_poles_v + 1):
                pole = surface.Pole(u, v)
                weight = surface.Weight(u, v)
                poles[u - 1][v - 1] = [pole.X(), pole.Y(), pole.Z(), weight]

        u_knots = np.array(u_knots).tolist()
        v_knots = np.array(v_knots).tolist()
        u_mults = np.array(u_mults).tolist()
        v_mults = np.array(v_mults).tolist()

        return {
            "type": "bspline_surface",
            "idx": idx,
            "location": [],
            "direction": [],
            "poles": poles,
            "scalar": [u_degree, v_degree, num_poles_u, num_poles_v, num_knots_u, num_knots_v] + u_knots + v_knots + u_mults + v_mults
        }

    def get_approx_face(self, points):
        uv_points_array = TColgp_Array2OfPnt(1, 32, 1, 32)
        for u_index in range(1, 32 + 1):
            for v_index in range(1, 32 + 1):
                pt = points[u_index - 1, v_index - 1]
                point_3d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
                uv_points_array.SetValue(u_index, v_index, point_3d)
        approx_face = GeomAPI_PointsToBSplineSurface(uv_points_array, 3, 3, GeomAbs_C2, 5e-2).Surface()
        num_u_poles = approx_face.NbUPoles()
        num_v_poles = approx_face.NbVPoles()
        control_points = np.zeros((num_u_poles * num_v_poles, 3))
        assert approx_face.UDegree() == approx_face.VDegree() == 3
        assert num_u_poles == num_v_poles == 4
        assert (not approx_face.IsUPeriodic() and not approx_face.IsVPeriodic() and not approx_face.IsVRational()
                and not approx_face.IsVPeriodic())
        poles = approx_face.Poles()
        idx = 0
        for u in range(1, num_u_poles + 1):
            for v in range(1, num_v_poles + 1):
                point = poles.Value(u, v)
                control_points[idx, :] = [point.X(), point.Y(), point.Z()]
                idx += 1
        return control_points.tolist()

    def get_approx_edge(self, points):
        u_points_array = TColgp_Array1OfPnt(1, 32)
        for u_index in range(1, 32 + 1):
            pt = points[u_index - 1]
            point_2d = gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2]))
            u_points_array.SetValue(u_index, point_2d)
        try:
            approx_edge = GeomAPI_PointsToBSpline(u_points_array, 3, 3, GeomAbs_C2, 5e-3).Curve()
        except Exception as e:
            print('high precision failed, trying mid precision...')
            try:
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 3, 3, GeomAbs_C2, 8e-3).Curve()
            except Exception as e:
                print('mid precision failed, trying low precision...')
                approx_edge = GeomAPI_PointsToBSpline(u_points_array, 3, 3, GeomAbs_C2, 5e-2).Curve()
        num_poles = approx_edge.NbPoles()
        assert approx_edge.Degree() == 3
        assert num_poles == 4
        assert not approx_edge.IsPeriodic() and not approx_edge.IsRational()
        control_points = np.zeros((num_poles, 3))
        poles = approx_edge.Poles()
        for i in range(1, num_poles + 1):
            point = poles.Value(i)
            control_points[i - 1, :] = [point.X(), point.Y(), point.Z()]
        return control_points.tolist()

    def tokenize_cad_data(self, step_path):
        for _, _, files in os.walk(step_path):
            assert len(files) == 1
            step_path = os.path.join(step_path, files[0])
        with zlw.Timer("Load step"):
            solids = load_step(step_path)
        if solids is None or len(solids) < 1:
            raise ValueError("No solids found in the step file")
        # for this version, assert step is single solid.
        solid = solids[0]
        if len(list(solid.faces())) > 200:
            raise ValueError("Too many faces in the solid.")
        solid = solid.topods_shape()
        bbox = Bnd_Box()
        brepbndlib.Add(solid, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        max_dimension = max(xmax - xmin, ymax - ymin, zmax - zmin)
        scale_factor = 2.0 / max_dimension
        center = gp_Pnt((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
        trsf = gp_Trsf()
        trsf.SetScale(center, scale_factor)
        solid = BRepBuilderAPI_Transform(solid, trsf, True).Shape()
        solid = Solid(solid)

        try:
            graph =  face_adjacency(solid, self_loops=True)
        except:
            raise ValueError("Face adjacency failed. The solid may be invalid.")
        data = []

        for face_idx in graph.nodes():
            face = graph.nodes[face_idx]["face"]
            surf_type = face.surface_type()
            surface = face.specific_surface()
            node_feature = None
            if surf_type == "plane":
                node_feature = self.plane_feature(surface, [face_idx, face_idx])
            elif surf_type == "cylinder":
                node_feature = self.cylinder_feature(surface, [face_idx, face_idx])
            elif surf_type == "cone":
                node_feature = self.cone_feature(surface, [face_idx, face_idx])
            elif surf_type == "sphere":
                node_feature = self.sphere_feature(surface, [face_idx, face_idx])
            elif surf_type == "torus":
                node_feature = self.torus_feature(surface, [face_idx, face_idx])
            elif surf_type == "bezier":
                node_feature = self.bezier_surface_feature(surface, [face_idx, face_idx])
            elif surf_type == "bspline":
                node_feature = self.bspline_surface_feature(surface, [face_idx, face_idx])
            else:
                raise ValueError("Unknown surface type.")
            try:
                points = uvgrid(face, method="point", num_u=32, num_v=32)
                node_feature["approximation"] = self.get_approx_face(points)
                node_feature["points"] = points.tolist()
            except Exception as e:
                node_feature["approximation"] = None
                node_feature["points"] = None
            data.append(node_feature)
        for edge_idx in graph.edges():
            edge = graph.edges[edge_idx]["edge"]
            curv_type = edge.curve_type()
            curve = edge.specific_curve()
            if curv_type == "line":
                edge_feature = self.line_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "circle":
                edge_feature = self.circle_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "ellipse":
                edge_feature = self.ellipse_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "hyperbola":
                edge_feature = self.hyperbola_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "parabola":
                edge_feature = self.parabola_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "bezier":
                edge_feature = self.bezier_curve_feature(curve, [edge_idx[0], edge_idx[1]])
            elif curv_type == "bspline":
                edge_feature = self.bspline_curve_feature(curve, [edge_idx[0], edge_idx[1]])
            else:
                raise ValueError("Unknown curve type.")
            try:
                points = ugrid(edge, method="point", num_u=32)
                edge_feature["approximation"] = self.get_approx_edge(points)
                edge_feature["points"] = points.tolist()
            except Exception as e:
                edge_feature["approximation"] = None
                edge_feature["points"] = None
            data.append(edge_feature)
        return data

    def tokenize_and_save_cad_data(self, path):
        step_path, output_path = path
        if os.path.exists(output_path):
            return 0
        try:
            with zlw.Timer("Process"):
                data = self.tokenize_cad_data(step_path)
            with open(output_path, 'w') as f:
                json.dump(data, f)
            return 1
        except ValueError as e:
            return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Data folder path", required=True)
    parser.add_argument("--option", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc',
                        help="Choose between dataset option [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--interval", type=int, help="Data range index, only required for abc/deepcad")
    args = parser.parse_args()

    if args.option == 'deepcad':
        OUTPUT = f'data/deepcad_parsed/{args.interval:04d}'
    elif args.option == 'abc':
        OUTPUT = f'data/abc_parsed/{args.interval:04d}'
    else:
        OUTPUT = f'data/furniture_parsed/{args.interval:04d}'

    os.makedirs(OUTPUT, exist_ok=True)

    if args.option == 'furniture':
        step_dirs = load_furniture_step(args.input)
    else:
        step_dirs = load_abc_step(args.input, args.option=='deepcad')
        step_dirs = step_dirs[args.interval*10000 : (args.interval+1)*10000]
        step_dirs = [(dirs, f'{OUTPUT}/{os.path.basename(dirs)}.json') for dirs in step_dirs]

    # shuffule the step_dirs
    random.shuffle(step_dirs)

    extractor = BRepDataProcessor()

    with Pool(os.cpu_count()) as pool:
        results = pool.map(extractor.tokenize_and_save_cad_data, step_dirs)
    # convert_iter = Pool(os.cpu_count()).imap(extractor.tokenize_and_save_cad_data, step_dirs)
    # valid = 0
    # for status in tqdm(convert_iter, total=len(step_dirs)):
    #     valid += status
    valid = sum(results)
    print(f'Done... Data Converted Ratio {100.0*valid/len(step_dirs)}%')