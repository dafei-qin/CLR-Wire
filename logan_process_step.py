from logan_process_brep_data import BRepDataProcessor



if __name__ == "__main__":
    processor = BRepDataProcessor()
    step = r'F:\WORK\CAD\data\056_002.step'
    processor.tokenize_and_save_cad_data([step, r'F:\WORK\CAD\data\examples\002_logan.json'])