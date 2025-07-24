from logan_process_brep_data import BRepDataProcessor



if __name__ == "__main__":
    processor = BRepDataProcessor()
    step = r'F:\WORK\CAD\data\examples\00000056\00000056_666139e3bff64d4e8a6ce183_step_005.step'
    processor.tokenize_and_save_cad_data([step, r'F:\WORK\CAD\data\examples\056.json'])