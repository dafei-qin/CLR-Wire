from occwl.solid import Solid
from occwl.viewer import OffscreenRenderer
from occwl.io import load_step


def render_solid(solid, output_file_path):
    v = OffscreenRenderer()
    v.display(solid)
    v.show_axes()
    v.fit()
    v.save_image(output_file_path)

v = OffscreenRenderer()
solids = load_step(r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\00000056_666139e3bff64d4e8a6ce183_step_005.step")
for idx, solid in enumerate(solids):
    render_solid(solid, rf"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\{idx:03d}.png")