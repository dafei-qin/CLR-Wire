from OCC.Core.BRep import BRep_Tool 
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section,BRepAlgoAPI_Splitter 
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE,TopAbs_FACE,TopAbs_REVERSED
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_CurveType,GeomAbs_Plane
from OCC.Core.TopTools import TopTools_HSequenceOfShape,TopTools_ListOfShape
from OCC.Core.TopoDS import TopoDS_Edge,TopoDS_Face
from OCC.Core.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Face, TopoDS_Wire, TopoDS_Vertex, TopoDS_Edge,topods
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape 
import polyscope as ps

from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire, ShapeFix_Edge, ShapeFix_ShapeTolerance,ShapeFix_Face
from utils.surface import sample_line
from OCC.Core.BRepTools import breptools

from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Section
from OCC.Core.gp import gp_Vec,gp_Pnt,gp_Trsf

from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Cylinder
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRep import BRep_Tool_Surface
from OCC.Core.GeomLProp import GeomLProp_SLProps
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Surface
import math

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Wire
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.TopoDS import topods
from OCC.Core.ShapeBuild import ShapeBuild_ReShape
from OCC.Core.TopAbs import TopAbs_WIRE

def adjust_cylinders_to_planes(faces):
    result_faces = faces
    cylinder_indices = []
    
    # 识别圆柱面并记录索引
    for i, face in enumerate(result_faces):
        surf_adaptor = BRepAdaptor_Surface(face)
        if surf_adaptor.GetType() == GeomAbs_Cylinder:
            cylinder_indices.append(i)
    
    # 处理每个圆柱面
    for cyl_idx in cylinder_indices:
        cylinder_face = result_faces[cyl_idx]
        surf_adaptor = BRepAdaptor_Surface(cylinder_face)
        cylinder = surf_adaptor.Cylinder()
        cyl_axis = cylinder.Axis()
        cyl_radius = cylinder.Radius()
        
        tangent_planes = []
        
        # 寻找相切平面
        required_radius=cyl_radius
        for i, face in enumerate(result_faces):
            if i == cyl_idx:
                continue
                
            surf_adaptor2 = BRepAdaptor_Surface(face)
            if surf_adaptor2.GetType() != GeomAbs_Plane:
                continue
                
            plane = surf_adaptor2.Plane()
            plane_normal = plane.Axis().Direction()
            cyl_direction = cyl_axis.Direction()
            
            # 检查法向量是否平行（圆柱侧面与平面垂直）
            dot_product = abs(plane_normal.Dot(cyl_direction))
            dis_err=abs(abs(plane.Distance(cyl_axis.Location()))-cyl_radius)
            dss = BRepExtrema_DistShapeShape(cylinder_face,face)
            dss.Perform()
            
            if abs(dot_product) < 0.1 and dis_err<0.10*cyl_radius and dss.IsDone() and dss.Value()<1.5*dis_err:  # 近似平行
                print(dss.IsDone() ,",", dss.Value(),",",dis_err)
                tangent_planes.append((i, face, plane))
                new_radius=abs(plane.Distance(cyl_axis.Location()))+cyl_radius*0.005
                if new_radius>required_radius:
                    required_radius=new_radius
        
        if len(tangent_planes)>0:
            print("Cylinder ",cyl_idx," has tangent planes: ",tangent_planes)
        
        if tangent_planes:
            cylinder.SetRadius(required_radius)
            u0,u1,v0,v1=breptools.UVBounds(cylinder_face)
            face_builder = BRepBuilderAPI_MakeFace(cylinder, u0, u1, v0, v1)
            new_cylinder_face = face_builder.Face()
            if topods.Face(cylinder_face).Orientation() == TopAbs_REVERSED:
                new_cylinder_face = new_cylinder_face.Reversed()
            result_faces[cyl_idx] = new_cylinder_face
    
    return result_faces

def intersect_faces(base_face, faces, tol):
    """
    平移faces使其与base_face相交
    
    Args:
        base_face: 基准面
        faces: 待处理的面列表
        tol: 平移容差
    
    Returns:
        相交的面列表
    """
    # 计算base_face包围盒
    base_box = Bnd_Box()
    brepbndlib.Add(base_face, base_box)
    base_xmin, base_ymin, base_zmin, base_xmax, base_ymax, base_zmax = base_box.Get()
    base_center = gp_Pnt((base_xmin + base_xmax)/2, (base_ymin + base_ymax)/2, (base_zmin + base_zmax)/2)
    
    result_faces = []
    
    for face in faces:
        # 计算当前面包围盒
        face_box = Bnd_Box()
        brepbndlib.Add(face, face_box)
        face_xmin, face_ymin, face_zmin, face_xmax, face_ymax, face_zmax = face_box.Get()
        face_center = gp_Pnt((face_xmin + face_xmax)/2, (face_ymin + face_ymax)/2, (face_zmin + face_zmax)/2)
        
        # 计算包围盒距离
        if base_box.Distance(face_box) > tol:
            continue
            
        # 检查原始位置是否相交
        section = BRepAlgoAPI_Section(base_face, face)
        if section.IsDone() and len(section.SectionEdges())>0:
            result_faces.append(face)
            continue
            
        # 计算移动方向向量
        direction = gp_Vec(face_center, base_center)
        if direction.Magnitude() < 1e-10:
            continue
            
        direction.Normalize()
        move_vec = direction * tol
        
        # 创建平移后的面
        sf = gp_Trsf()
        sf.SetTranslation(move_vec)
        loc = TopLoc_Location(sf)
        moved_face = topods.Face(face).Moved(loc)
        
        # 检查平移后是否相交
        section2 = BRepAlgoAPI_Section(base_face, moved_face)
        if section2.IsDone() and len(section2.SectionEdges())>0:
            result_faces.append(face)
            
    return result_faces 

def clean_unclosed_wires(face):
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
    unclosed_wires = []
    
    while wire_explorer.More():
        wire = topods.Wire(wire_explorer.Current())
        is_closed = BRep_Tool.IsClosed(wire)
        if not is_closed:
            unclosed_wires.append(wire)
        wire_explorer.Next()
    
    r=ShapeBuild_ReShape() 
    if unclosed_wires:
        for w in unclosed_wires:
            r.Remove(w)
    r.Apply(face)
    face=topods.Face(r.Value(face))
    return face



def all_surf_interset(base_face,faces, save_path=None):
    faces=intersect_faces(base_face,faces,0.1)
    splitter = BRepAlgoAPI_Splitter()
    splitter.SetRunParallel(True)
    list_arg=TopTools_ListOfShape()
    list_tool=TopTools_ListOfShape()
    list_arg.Append(base_face)
    for i in range(len(faces)):
        list_tool.Append(faces[i])
    splitter.SetArguments(list_arg)
    splitter.SetTools(list_tool)
    splitter.Build()
    section = splitter.Shape()
    if not section:
        return [base_face,]

    face_explorer = TopExp_Explorer(section, TopAbs_FACE)
    
    max_area=0.0
    largest_face=None
    all_faces=[]
    
    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    while face_explorer.More():
        current_face = face_explorer.Current()
        current_face=clean_unclosed_wires(current_face)
        all_faces.append(current_face)
        builder.Add(new_compound,current_face)
        props = GProp_GProps()
        brepgprop.SurfaceProperties(current_face, props)
        area = abs(props.Mass() )
        
        if area > max_area:
            max_area = area
            largest_face = current_face
            
        face_explorer.Next()
    
    if save_path and section:
        breptools.Write(new_compound, save_path)

    if largest_face is None:
        largest_face=base_face
    if len(all_faces)==0:
        all_faces.append(base_face)
    return all_faces

def surf_surf_interset(face_m, face_n, face_group=None, tol=1e-2, plot=True, plot_header=''):

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
        #print(type(exp.Current()))
        # edge = TopoDS_Edge(exp.Current())
        edges.append(exp.Current())
        # all_edges[idx_m].append(exp.Current())
        exp.Next()
    if plot:
        for idx_line, edge in enumerate(edges):
            a_curve = BRepAdaptor_Curve(edge)
            line_type = a_curve.GetType()
            points, line_edges = sample_line(a_curve)
            psEdge = ps.register_curve_network(f"{plot_header}_line_{idx_line:03d}_{GeomAbs_CurveType(line_type).name}", points, line_edges, radius=0.001)
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
