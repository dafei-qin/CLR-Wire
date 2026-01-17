import math
import os
import random
import pickle as pkl

from OCC.Core.gp import (
    gp_Pnt,
    gp_OX,
    gp_Vec,
    gp_Trsf,
    gp_DZ,
    gp_Ax2,
    gp_Ax3,
    gp_Pnt2d,
    gp_Dir2d,
    gp_Ax2d,
    gp_Pln,
)
from OCC.Core.BOPTools import BOPTools_AlgoTools2D
from OCC.Core.BRep import BRep_Builder, BRep_Tool
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeWire
from OCC.Core.BRepTools import breptools
from OCC.Core.TopoDS import TopoDS_Compound, TopoDS_Iterator, TopoDS_Face, TopoDS_Wire, TopoDS_Vertex, TopoDS_Edge,topods
from OCC.Core.TopExp import TopExp_Explorer,topexp
from OCC.Core.TopTools import TopTools_ListOfShape,TopTools_MapOfShape,TopTools_ListIteratorOfListOfShape
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_WIRE
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.ShapeFix import ShapeFix_Wire, ShapeFix_Face

def find_unique_cyclic_lists(x):
    if not x:
        return []
    
    result = []
    seen = set()
    
    for i in range(len(x)):
        sublist=x[i]
        
        normalized_strings = []
        n = len(sublist)
        
        for j in range(n):
            forward = tuple(sublist[j:] + sublist[:j])
            normalized_strings.append(forward)
            
            # 逆序循环排列（相当于正序的逆序）
            reversed_sublist = sublist[::-1]
            backward = tuple(reversed_sublist[j:] + reversed_sublist[:j])
            normalized_strings.append(backward)
        
        # 使用最小的元组作为标准表示
        min_repr = min(normalized_strings)
        if min_repr not in seen:
            seen.add(min_repr)
            result.append(i)
    
    return result

def vertex_hash(vertex):
    point = BRep_Tool.Pnt(vertex)
    x = point.X()
    y = point.Y()
    z = point.Z()
    
    precision = 5
    hash_str = f"{x:.{precision}f}_{y:.{precision}f}_{z:.{precision}f}"
    
    return hash_str

def dfs_find_loop_(loop, v2e, visited_vertices, current_edge, current_v, prev_v):
    #print(loop)
    loops=[]
    edges_list = v2e[current_v]
    it = TopTools_ListIteratorOfListOfShape(edges_list)
    while it.More():
        edge = it.Value()
        if edge.IsSame(current_edge):
            it.Next()
            continue
        
        edge_v0 = vertex_hash(topexp.FirstVertex(edge, True))
        edge_v1 = vertex_hash(topexp.LastVertex(edge, True))
        
        next_vertex = edge_v1 if edge_v0 == current_v else edge_v0
        
        if edge_v0 == current_v:
            loop.append(edge)
        else:
            loop.append(edge.Reversed())

        if next_vertex not in visited_vertices:
            visited_vertices.add(next_vertex)
            ret = dfs_find_loop_(loop, v2e, visited_vertices, edge, next_vertex, current_v)
            if ret != -1:
                loops+=ret
            visited_vertices.remove(next_vertex)
            
        else:
            for i in range(len(loop)):
                ei = loop[i]
                ei_v0 = vertex_hash(topexp.FirstVertex(ei, True))
                if ei_v0 == next_vertex:
                    #print("Found a new loop, checking...")
                    #check_loop(loop)
                    loops+=[loop[i:].copy(),]
                    break

        loop.pop()
        
        it.Next()
    return loops

def dfs_find_loops(edges, v2e):
    loops=[]
    for start_edge in edges:
        visited_vertices = set()
        
        v0 = vertex_hash(topexp.FirstVertex(start_edge, True))
        v1 = vertex_hash(topexp.LastVertex(start_edge, True))
        
        visited_vertices.add(v0)
        visited_vertices.add(v1)

        loop = [start_edge]
        
        cur_loops = dfs_find_loop_(loop, v2e,visited_vertices,start_edge,v1,v0)
        loops=loops+cur_loops
    # 如果没有找到闭合环路，返回空列表
    return loops

def check_loop(loop):
    if loop==[]:
        return False
    # print("---Begin---")
    # for edge in loop:
    #     print(vertex_hash(topexp.FirstVertex(edge, True)),",",vertex_hash(topexp.LastVertex(edge, True)))
    # print("---End---")
    v_prev=vertex_hash(topexp.FirstVertex(loop[0], True))
    for edge in loop:
        v_cur=vertex_hash(topexp.FirstVertex(edge, True))
        if v_prev!=v_cur:
            return False
        v_prev=vertex_hash(topexp.LastVertex(edge, True))
    if vertex_hash(topexp.FirstVertex(loop[0], True))!=vertex_hash(topexp.LastVertex(loop[-1], True)):
        return False
    #print("Pass.")
    return True

def edge_filter(edges):
    v2e = {}
    
    for e in edges:
        v0 = vertex_hash(topexp.FirstVertex(e, True))
        v1 = vertex_hash(topexp.LastVertex(e, True))
        
        elist0 = v2e.get(v0,TopTools_ListOfShape())
        elist0.Append(e)
        v2e[v0]=elist0

        elist1 = v2e.get(v1,TopTools_ListOfShape())
        elist1.Append(e)
        v2e[v1]=elist1
    
    loop_edges = dfs_find_loops(edges, v2e)
    if loop_edges!=[]  and type(loop_edges[0]) is list:
        unique_ids=find_unique_cyclic_lists([[str(hash(e)) for e in loop] for loop in loop_edges])
        #print("uids: ",unique_ids)
        loop_edges=[loop_edges[i] for i in unique_ids]
        #print("Tids: ",[[str(hash(e)) for e in loop] for loop in loop_edges])
    
    loop_edges = [loop for loop in loop_edges if check_loop(loop)]
    print("Found ",len(loop_edges)," loops")
    return loop_edges

def build_wire_and_face(edges, surface):
    if edges == []:
        return False
    #print(edges,surface)
    face = BRepBuilderAPI_MakeFace(surface,1e-6).Face()
    wire_builder = BRepBuilderAPI_MakeWire()

    # make pcurve
    for edge in edges:
        try:
            BOPTools_AlgoTools2D.BuildPCurveForEdgeOnFace(edge, face)
            wire_builder.Add(edge)
        except Exception as e:
            print(f"exception building pcurve: {e}")
            return False
    
    try:
        new_wire=wire_builder.Wire()
        new_wire.Closed(BRep_Tool.IsClosed(new_wire))
        breptools.Update(new_wire)

        #breptools.Write(new_wire, "a-wire-0.brep")

        shape_fix_wire = ShapeFix_Wire(new_wire,face,1e-4)
        shape_fix_wire.Perform()
        fixed_wire = shape_fix_wire.Wire()
        if not fixed_wire.IsNull():
            new_wire = fixed_wire
            
            #breptools.Write(new_wire, "a-wire-1.brep")
            #print(new_wire.DumpJson())

    except Exception as e:
        print(f"exception building wire: {e}")
        return False

    # build new face
    try:
        face_builder = BRepBuilderAPI_MakeFace(surface, new_wire)
        if not face_builder.IsDone():
            print(f"error building face: {face_builder.Error()}")
            return False
        else:
            face = face_builder.Face()
    except Exception as e:
        print(f"exception building face: {e}")
        return False
    shape_fix_face = ShapeFix_Face(face)
    shape_fix_face.Perform()
    fixed_face = shape_fix_face.Face()
    if not fixed_face.IsNull():
        face = fixed_face
    return face

def read_one_group_from_brep(input_file):
    shape = TopoDS_Compound()
    builder = BRep_Builder()
    status = breptools.Read(shape, input_file, builder)
    if not status:
        print(f"error reading file {input_file}")
        return False
    
    face = None
    edges = []
    
    shape_iterator = TopoDS_Iterator(shape)
    while shape_iterator.More():
        sub_shape = shape_iterator.Value()
        
        if sub_shape.ShapeType() == TopAbs_FACE:
            face = topods.Face(sub_shape)
        elif sub_shape.ShapeType() == TopAbs_EDGE:
            edges.append(topods.Edge(sub_shape))
        
        shape_iterator.Next()
    
    surface = None
    if face is not None:
        surface = BRep_Tool.Surface(face)
    return edges, surface

def cut_face_with_edges(edges,face):
    surface = BRep_Tool.Surface(face)
    edge_groups = edge_filter(edges)
    faces=[]
    for group in edge_groups:
        new_face=build_wire_and_face(group,surface)
        if new_face == False:
            continue
        faces.append(new_face)
    return faces

def process_one(input_file,output_file,save=True):
    edges,surface=read_one_group_from_brep(input_file)
    
    edge_groups = edge_filter(edges)
    faces=[]
    for group in edge_groups:
        new_face=build_wire_and_face(group,surface)
        if new_face == False:
            continue
        faces.append(new_face)
        if save:
            breptools.Write(new_face, output_file)
            print(f"saved to {output_file}")
    
    return faces

def process_all():
    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    current_dir = os.getcwd()
    for filename in os.listdir(current_dir):
        if filename.startswith("group") and filename.endswith(".brep"):
            file_path = os.path.join(current_dir, filename)
            if os.path.isfile(file_path):
                print("processing ",file_path)
                out_file_path = os.path.join(current_dir, "result-"+filename)
                faces = process_one(file_path,out_file_path,False)
                if faces == []:
                    print("No face for ",file_path)
                else:
                    for face_i in faces:
                        builder.Add(new_compound, face_i)
                    print("done")
    breptools.Write(new_compound, "compound_all.brep")

def process_pkl():
    data=[]
    with open("detokenized_surfaces_idx_0000_interset.pkl","rb") as f:
        data = pkl.load(f)

    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    for pair in data:
        try:
            edges,surface=pair['edges'],BRep_Tool.Surface(pair['face'])
            
            edge_groups = edge_filter(edges)
            for group in edge_groups:
                new_face=build_wire_and_face(group,surface)
                if new_face == False:
                    continue
                builder.Add(new_compound, new_face)
        except Exception as e:
            print(f"exception building face: {e}")
            continue

    breptools.Write(new_compound, "compound_all.brep")

def dump_pkl():

    data=[]
    with open("detokenized_surfaces_idx_0000_interset.pkl","rb") as f:
        data = pkl.load(f)

    builder = BRep_Builder()
    new_compound = TopoDS_Compound()
    builder.MakeCompound(new_compound)
    for pair in data:
        edges,surface=pair['edges'],BRep_Tool.Surface(pair['face'])
        builder.Add(new_compound, pair['face'])
        for e in edges:
            builder.Add(new_compound, e)

    breptools.Write(new_compound, "compound_all.brep")
