# -*- coding: utf-8 -*-
"""
================================================================================
FreeCAD Batch STEP to JSON Serializer
================================================================================
Purpose:
    This script is designed to be run from the command line using FreeCAD's
    command-line interface (freecadcmd). It takes a .step file as input,
    imports it, extracts its full topological and geometric data (including
    support for lines, circles, ellipses, and B-splines), and exports that

    data to a structured .json file.

Version: 4.0 (Batch Processing Enabled)

How to Run:
    1. Save this script as a Python file (e.g., "batch_serializer.py").
    2. Open your system's terminal or command prompt.
    3. Execute the script using the following command format:

       For Windows:
       path\to\FreeCAD\bin\freecadcmd.exe "C:\path\to\batch_serializer.py" "C:\path\to\your_model.step"

       For macOS:
       /Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd -c "/path/to/batch_serializer.py" "/path/to/your_model.step"
       
       For Linux (AppImage):
       ./FreeCAD.AppImage --headless -c "/path/to/batch_serializer.py" "/path/to/your_model.step"
       
       For Linux (installed):
       freecadcmd -c "/path/to/batch_serializer.py" "/path/to/your_model.step"

    4. The script will create a file named "your_model.json" in the same
       directory as the input .step file.
================================================================================
"""
import FreeCAD as App
import Part
import sys
import os
import json
from datetime import datetime

def serialize_shape_to_json(shape, output_path):
    """
    Analyzes a FreeCAD Shape object and serializes its topology and geometry
    to a detailed JSON file.

    Args:
        shape (Part.Shape): The FreeCAD shape to process.
        output_path (str): The full path to save the output JSON file.
    """
    print("Step 1/3: Extracting vertices, edges, and faces...")

    # --- Vertex Extraction ---
    all_vertices = shape.Vertexes
    vertex_map = {v.hashCode(): i for i, v in enumerate(all_vertices)}
    json_vertices = [[v.X, v.Y, v.Z] for v in all_vertices]

    # --- Edge Extraction ---
    json_edges = []
    edge_map = {e.hashCode(): i for i, e in enumerate(shape.Edges)}
    for i, edge in enumerate(shape.Edges):
        try:
            curve = edge.Curve
        except TypeError:
            print(f"Edge {i} has no curve")
            assert edge.Length < 1e-4, f"Edge {i} has valid length {edge.Length}, can't drop."
            print(f"Edge {i} has no curve and invalid length {edge.Length}, make it invalid.")
            edge_data = {
                "edge_index": i,
                "curve_type": "Invalid",
                "length": edge.Length,
                "vertices": [vertex_map.get(v.hashCode()) for v in edge.Vertexes],
                "first_parameter": edge.FirstParameter,
                "last_parameter": edge.LastParameter
            }
            json_edges.append(edge_data)
            continue
        edge_data = {
            "edge_index": i,
            "curve_type": type(curve).__name__,
            "length": edge.Length,
            "vertices": [vertex_map.get(v.hashCode()) for v in edge.Vertexes],
            "first_parameter": edge.FirstParameter,
            "last_parameter": edge.LastParameter
        }

        # --- Detailed Geometric Definitions for Curves ---
        if isinstance(curve, (Part.Line, Part.LineSegment)):
            if len(edge.Vertexes) >= 2:
                edge_data["curve_definition"] = {
                    "start": [edge.Vertexes[0].X, edge.Vertexes[0].Y, edge.Vertexes[0].Z],
                    "end": [edge.Vertexes[1].X, edge.Vertexes[1].Y, edge.Vertexes[1].Z]
                }
        elif isinstance(curve, Part.Circle):
            edge_data["curve_definition"] = {
                "center": [curve.Center.x, curve.Center.y, curve.Center.z],
                "normal": [curve.Axis.x, curve.Axis.y, curve.Axis.z],
                "radius": curve.Radius
            }
        elif isinstance(curve, Part.Ellipse):
            # For ellipses, we determine the major/minor axis directions.
            bspline = curve.toBSpline()
            poles = bspline.getPoles()
            first_pole = App.Vector(poles[0])
            center_vec = curve.Center
            axis_vec = first_pole - center_vec
            major_axis_dir = App.Vector(0,0,0)
            
            if abs(axis_vec.Length**2 - curve.MajorRadius**2) < 1e-7:
                 major_axis_dir = axis_vec.normalize()
            else:
                 normal_vec = curve.Axis
                 second_axis_vec = normal_vec.cross(axis_vec)
                 major_axis_dir = second_axis_vec.normalize()

            minor_axis_dir = curve.Axis.cross(major_axis_dir).normalize()
            edge_data["curve_definition"] = {
                "center": [curve.Center.x, curve.Center.y, curve.Center.z],
                "major_radius": curve.MajorRadius,
                "minor_radius": curve.MinorRadius,
                "normal": [curve.Axis.x, curve.Axis.y, curve.Axis.z],
                "major_axis_direction": [major_axis_dir.x, major_axis_dir.y, major_axis_dir.z],
                "minor_axis_direction": [minor_axis_dir.x, minor_axis_dir.y, minor_axis_dir.z]
            }
        elif isinstance(curve, Part.BSplineCurve):
            knots, mults = curve.getKnots(), curve.getMultiplicities()
            full_knot_vector = [k for k, m in zip(knots, mults) for _ in range(m)]
            edge_data["curve_definition"] = {
                "degree": curve.Degree,
                "is_periodic": curve.isPeriodic(),
                "control_points": [[p.x, p.y, p.z] for p in curve.getPoles()],
                "knots": full_knot_vector,
                "multiplicities": list(mults)
            }
        json_edges.append(edge_data)

    # --- Face Extraction ---
    json_faces = []
    for i, face in enumerate(shape.Faces):
        surface = face.Surface
        face_data = {
            "face_index": i,
            "surface_type": type(surface).__name__,
            "area": face.Area,
            "is_planar": isinstance(surface, Part.Plane),
            "orientation": face.Orientation,
            "parameter_range": face.ParameterRange,
            "wires": []
        }
        
        # --- Detailed Geometric Definitions for Surfaces ---
        surface_def = {}
        if isinstance(surface, Part.Plane):
            surface_def = {"position": [surface.Position.x, surface.Position.y, surface.Position.z],
                           "normal": [surface.Axis.x, surface.Axis.y, surface.Axis.z]}
        elif isinstance(surface, Part.Cylinder):
            surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                           "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z], "radius": surface.Radius}
        elif isinstance(surface, Part.Cone):
            surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                        "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z],
                        "radius": surface.Radius, 
                        "semi_angle": surface.SemiAngle}
        elif isinstance(surface, Part.Sphere):
            surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                        "radius": surface.Radius}
        elif isinstance(surface, Part.Toroid):
            surface_def = {"position": [surface.Center.x, surface.Center.y, surface.Center.z],
                        "axis": [surface.Axis.x, surface.Axis.y, surface.Axis.z],
                        "major_radius": surface.MajorRadius,
                        "minor_radius": surface.MinorRadius}
        # ... (other surface types like Cone, Sphere, Toroid would go here)
        elif isinstance(surface, Part.BSplineSurface):
            u_knots, v_knots = surface.getUKnots(), surface.getVKnots()
            u_mults, v_mults = surface.getUMultiplicities(), surface.getVMultiplicities()
            full_u_knots = [k for k, m in zip(u_knots, u_mults) for _ in range(m)]
            full_v_knots = [k for k, m in zip(v_knots, v_mults) for _ in range(m)]
            surface_def = {
                "u_degree": surface.UDegree, "v_degree": surface.VDegree,
                "is_u_periodic": surface.isUPeriodic(), "is_v_periodic": surface.isVPeriodic(),
                "control_points": [[[p.x, p.y, p.z] for p in row] for row in surface.getPoles()],
                "u_knots": full_u_knots, "v_knots": full_v_knots,
                "u_multiplicities": list(u_mults), "v_multiplicities": list(v_mults)
            }
        face_data["surface_definition"] = surface_def

        # --- Wire and Edge Loop Extraction ---
        for wire in face.Wires:
            wire_data = {"is_outer": wire.isSame(face.OuterWire), "ordered_edges": []}
            for edge in wire.OrderedEdges:
                edge_ref = {
                    "edge_index": edge_map.get(edge.hashCode()),
                    "orientation": edge.Orientation
                }
                wire_data["ordered_edges"].append(edge_ref)
            face_data["wires"].append(wire_data)
        json_faces.append(face_data)
    
    print("Data extraction complete.")
    
    # --- Assemble and Export JSON ---
    print("Step 2/3: Assembling final JSON object...")
    reconstruction_data = {
        "metadata": {
            "source_file": os.path.basename(output_path.replace('.json', '.step')),
            "export_time_utc": datetime.utcnow().isoformat() + "Z",
            "units": "mm",
            "schema_version": "4.0-reconstruction-batch"
        },
        "vertices": json_vertices,
        "edges": json_edges,
        "faces": json_faces
    }

    print("Step 3/3: Writing data to JSON file...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reconstruction_data, f, ensure_ascii=False, indent=2)
        print(f"\nSUCCESS: Blueprint data saved to: {output_path}")
    except Exception as e:
        print(f"\nERROR: Could not write to JSON file: {e}", file=sys.stderr)

# ==============================================================================
# Main Execution Block for Command-Line Operation
# ==============================================================================


# This block runs only when the script is executed directly

# Check if a file path was provided as an argument
    # if len(sys.argv) < 2:
    #     print("ERROR: Missing input file argument.", file=sys.stderr)
    #     print("Usage: freecadcmd -c batch_serializer.py <path_to_your_model.step>", file=sys.stderr)
    #     sys.exit(1) # Exit with an error code

input_file_path = r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\00000056_666139e3bff64d4e8a6ce183_step_005.step"
output_file_path = r"C:\Users\Dafei Qin\Documents\WORK\CAD\data\examples\00000056\out.json"

# Verify the input file exists
if not os.path.isfile(input_file_path):
    print(f"ERROR: Input file not found at '{input_file_path}'", file=sys.stderr)
    sys.exit(1)

# Define the output path for the JSON file
# It will be the same as the input file, but with a .json extension
# output_file_path = os.path.splitext(input_file_path)[0] + ".json"
# output_file_path = os.path.join(os.path.dirname(input_file_path), os.path.basename(input_file_path).replace('.step', '.json'))

print("="*50)
print("Starting Batch CAD to JSON Serialization...")
print(f"Input STEP file: {input_file_path}")
print(f"Output JSON file: {output_file_path}")
print("="*50)

# Create a new, temporary FreeCAD document
# Using a unique name to avoid conflicts if running multiple instances
doc_name = f"temp_doc_{os.path.basename(input_file_path)}"
doc = App.newDocument(doc_name)


# Import the STEP file into the document. This is the non-GUI way.
print(f"\nImporting '{os.path.basename(input_file_path)}'...")
Part.insert(input_file_path, doc.Name)

# Check if any objects were imported
if not doc.Objects:
    print("ERROR: No objects were imported from the STEP file.", file=sys.stderr)
    sys.exit(1)

# In a STEP file, all parts are usually combined into a single object.
# We will process the first object found.
obj_candidates = []
for obj in doc.Objects:
    if type(obj) == Part.Feature:
        obj_candidates.append(obj)
# imported_object = doc.Objects[0]
# print(f"Model '{imported_object.Label}' loaded successfully.")
for obj_idx, imported_object in enumerate(obj_candidates):
    print(f"Processing object {obj_idx+1} of {len(obj_candidates)}")
    # Call the main serialization function
    serialize_shape_to_json(imported_object.Shape, output_file_path.replace('.json', f'_{obj_idx:03d}.json'))

# Clean up by closing the temporary document
App.closeDocument(doc.Name)
print("="*50)
print("Script finished.")
exit()