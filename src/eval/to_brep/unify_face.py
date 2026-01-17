from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Core.BRepTools import breptools
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Shape
import os

def UnifySameDomain(shape,tol):
    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.SetLinearTolerance(tol)
    unifier.Build()
    return unifier.Shape()


if __name__ == '__main__':
    shape = TopoDS_Shape()
    breptools.Read(shape,"temp_brep_for_unify.brep",BRep_Builder())
    if os.path.exists("temp_brep_for_unify.brep"):
        os.remove("temp_brep_for_unify.brep")
    try:
        shape = UnifySameDomain(shape,1e-5)
    except Exception as e:
        print(e)
    breptools.Write(shape,"temp_brep_for_unify.res.brep")