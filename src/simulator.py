import sys
import os
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from model_loader import ModelLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    # Set up data converter
    tdc = TrainingDataConverter("dataset")
    tdc.conversion("training.db")
    # Read converted data
    dbr = DatabaseReader("training.db")
    df = dbr.get_dataframe()
    # Run training with dataframe
    tp = TrainingProgram(df)

    # Training
    tp.train_caudal(1)
    tp.train_dorsal(1)
    tp.train_frontal(1)
    tp.train_lateral(1)

    # Save models
    tp.save_models("caud.pth", "dors.pth", "fron.pth", "late.pth", "height.txt")

    # Load models
    model_paths = {
            "caud" : "src/models/caud.pth", 
            "dors" : "src/models/dors.pth",
            "fron" : "src/models/fron.pth",
            "late" : "src/models/late.pth"
        }
    ml = ModelLoader(model_paths)
    print(ml.get_models().keys)
    print(ml.get_model("caud").named_parameters())