from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    # Set up data converter
    tdc = TrainingDataConverter("../dataset/")
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

    