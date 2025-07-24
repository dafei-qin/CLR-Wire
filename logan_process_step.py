from logan_process_brep_data import BRepDataProcessor
import sys


if __name__ == "__main__":
    processor = BRepDataProcessor()
    step = sys.argv[1]
    processor.tokenize_and_save_cad_data([step, sys.argv[2]])