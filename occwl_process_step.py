import pathlib

from occwl.compound import Compound
from occwl.viewer import Viewer
from occwl.io import save_step

# Returns a list of bodies from the step file, we only need the first one
compound = Compound.load_from_step(r'F:\WORK\CAD\data\056_002.step')
solid = next(compound.solids())
solid = solid.scale_to_unit_box()
save_step([solid], r'F:\WORK\CAD\data\056_002_unit.step')
v = Viewer(backend="wx")
v.display(solid)
v.fit()
v.show()