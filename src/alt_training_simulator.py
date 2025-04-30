""" alt_training_simulator.py """
import sys
import os
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from alt_training_program import AltTrainingProgram
from training_program import TrainingProgram

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


class Tee:
    """
    Class to enable stdout to output to both a log file and stdout in terminal
    """
    def __init__(self, *streams):
        """ Stores streams """
        self.streams = streams

    def write(self, message):
        """ Write to all output streams """
        for s in self.streams:
            s.write(message)
            s.flush()  # Ensure it gets written immediately

    def flush(self):
        """ Flush after write to avoid buffering """
        for s in self.streams:
            s.flush()

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    log_file = open("training_comparison_output.log", "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    # Set up data converter
    tdc = TrainingDataConverter("dataset")
    tdc.conversion("training.db")
    # Read converted data
    dbr = DatabaseReader("training.db", "src/models/class_list.txt")
    df = dbr.get_dataframe()

    # Display how many images we have for each angle
    print("Number of Images for Each Angle:")
    print(f"CAUD: {(df['View'] == 'CAUD').sum()}")
    print(f"DORS: {(df['View'] == 'DORS').sum()}")
    print(f"FRON: {(df['View'] == 'FRON').sum()}")
    print(f"LATE: {(df['View'] == 'LATE').sum()}")

    # initialize number of outputs
    SPECIES_OUTPUTS = dbr.get_num_species()
    GENUS_OUTPUTS = dbr.get_num_genus()

    # Run training with dataframe
    alt_species_tp = AltTrainingProgram(df, 1, SPECIES_OUTPUTS)

    # Training
    alt_species_tp.train_dorsal_caudal(20)
    alt_species_tp.train_all(20)
    alt_species_tp.train_dorsal_lateral(20)

    # Save models
    alt_species_model_filenames = {
            "dors_caud" : "alt_spec_dors_caud.pth",
            "all" : "alt_spec_all.pth",
            "dors_late": "alt_spec_dors_late.pth"
        }

    alt_species_tp.save_models(
        alt_species_model_filenames,
        "alt_height.txt",
        "alt_spec_dict.json",
        "alt_spec_accuracies.json")

    # Run training with dataframe
    alt_genus_tp = AltTrainingProgram(df, 0, GENUS_OUTPUTS)

    # Training
    alt_genus_tp.train_dorsal_caudal(20)
    alt_genus_tp.train_all(20)
    alt_genus_tp.train_dorsal_lateral(20)

    # Save models
    alt_genus_model_filenmaes = {
        "dors_caud" : "gen_dors_caud.pth", 
        "all" : "gen_all.pth",
        "dors_late" : "gen_dors_late.pth"
    }

    alt_genus_tp.save_models(
        alt_genus_model_filenmaes,
        "alt_height.txt",
        "alt_gen_dict.json",
        "alt_gen_accuracies.json")

    log_file.close()
