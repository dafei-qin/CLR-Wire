from OCC.Core.BRep import BRep_Tool 
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_CurveType
from OCC.Core.TopTools import TopTools_HSequenceOfShape
from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds

import polyscope as ps

from lib.surface import sample_line

def surf_surf_interset(face_m, face_n, face_group=None, tol=1e-2, plot_header=''):
    geom_face_m = BRep_Tool.Surface(face_m)
    geom_face_n = BRep_Tool.Surface(face_n)


    # IntSS: geom-geom intersection, no boundary constraints.
            # interset_geom = GeomInt_IntSS(geom_face_m, geom_face_n, 1e-2)
            # # print(f'Number of intersected lines without constraints: ', interset_geom.NbLines())
            # for i in range(interset_geom.NbLines()):
            #     line = interset_geom.Line(i + 1)
            #     points, edges = sample_line(line)
                # ps.register_curve_network(f"{idx_m:03d}_{idx_n:03d}_line_{i:03d}", points, edges, radius=0.001)

    # FacesIntersector: face-face intersection, with boundary constraints.

            # intersector = TopOpeBRep_FacesIntersector()
            # # intersector.ForceTolerances(1e-1, 1e-1)
            # print(f'Intersector Tolerance: ', intersector.GetTolerances())

            # fixer = ShapeFix_Shape(face_m)
            # fixer.Perform()
            # healed_face_m = fixer.Shape()

            # fixer = ShapeFix_Shape(face_n)
            # fixer.Perform()
            # healed_face_n = fixer.Shape()

            # intersector.Perform(healed_face_m, healed_face_n)
            # # if not intersector.IsDone():
            # #     raise ValueError(f"Intersector is not done for face {idx_m} and face {idx_n}")
            # lines = intersector.Lines()
            # num_lines = intersector.NbLines()
            # print(f'Is intersection empty?', intersector.IsEmpty(), f'number of intersected lines: ', num_lines)
            # if num_lines == 0:
            #     continue

            # for idx_line, line in enumerate(lines):
            #     arc = line.Arc()
            #     print(f"{idx_m:03d}_{idx_n:03d}_line_{idx_line:03d}", type(arc))
            #     if type(arc) == TopoDS_Edge:
            #         a_curve = BRepAdaptor_Curve(arc)
            #         line_type = a_curve.GetType()
            #         # print(GeomAbs_CurveType(line_type).name)
            #         points, edges = sample_line(a_curve)
            #         # ps.register_curve_network(f"{idx_m:03d}_{idx_n:03d}_line_{idx_line:03d}_{GeomAbs_CurveType(line_type).name}", points, edges, radius=0.001)
            #         # print(i)

    # Section Intersector
    section = BRepAlgoAPI_Section(face_m, face_n)
    section_shape = section.Shape()
    edges = []
    exp = TopExp_Explorer(section_shape, TopAbs_EDGE)
    while exp.More():
        print(type(exp.Current()))
        # edge = TopoDS_Edge(exp.Current())
        edges.append(exp.Current())
        # all_edges[idx_m].append(exp.Current())
        exp.Next()
    for idx_line, edge in enumerate(edges):
        a_curve = BRepAdaptor_Curve(edge)
        line_type = a_curve.GetType()
        points, edges = sample_line(a_curve)
        psEdge = ps.register_curve_network(f"{plot_header}_line_{idx_line:03d}_{GeomAbs_CurveType(line_type).name}", points, edges, radius=0.001)
        if face_group is not None:
            psEdge.add_to_group(face_group)
    return edges


def wire_formation(face, edges):
    # for idx_m in range(len(all_faces)):
    #     new_group = ps.create_group(f"face_{idx_m:03d}_wires")
    #     all_ps_wire_groups.append(new_group)
    #     edges = all_edges[idx_m]
    #     face =  all_faces[idx_m]['surface']
    #     operator_fb = ShapeAnalysis_FreeBounds(Compound([face]), global_tol, False, False)
       
    #     wires = edges_to_wires(face, edges, 1e-6)
    #     for wire_idx, wire in enumerate(wires):
    #         wire = wire.Topods_Shape()
    #         psWire = ps.register_curve_network(f"wire_{wire_idx:03d}", wire, radius=0.001)
    #         psWire.add_to_group(new_group)
    
    #     new_group.set_enabled(True)
    #     new_group.set_hide_descendants_from_structure_lists(True)
    #     new_group.set_show_child_details(False)
    raise NotImplementedError("Wire formation is not implemented yet")