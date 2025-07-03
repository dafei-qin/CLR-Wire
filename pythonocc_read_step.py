from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.BRep import BRep_Builder


import os
import sys

step_file = sys.argv[1]
cylinder_head = TopoDS_Shape()
builder = BRep_Builder()
breptools_Read(cylinder_head, step_file, builder)

display, start_display, add_menu, add_function_to_menu = init_display()
display.DisplayShape(cylinder_head, update=True)
start_display()