"""stack_simulator.py"""
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod
from post_eval_stack_training import PostTrainingStacking
from stack_dataset_creator import StackDatasetCreator
from stack_dataset_creator import StackDatasetConfig

if __name__ == '__main__':

    #tdc = TrainingDataConverter("dataset")
    #tdc.conversion("training.db")
    dbr = DatabaseReader("training.db")
    df = dbr.get_dataframe()

    GENUS_OUTPUTS = dbr.get_num_genus()

    genus_model_paths = {
            "caud" : "src/models/gen_caud.pth", 
            "dors" : "src/models/gen_dors.pth",
            "fron" : "src/models/gen_fron.pth",
            "late" : "src/models/gen_late.pth"
        }
    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)

    genus_model_list = genus_ml.get_models()

    genus_config = StackDatasetConfig("height.txt", "gen_dict.json", 1)
    genus_stack_creator = StackDatasetCreator(genus_config, df, genus_model_list)
    df_genus = genus_stack_creator.create_flat_stack_dataset("Genus")

    genus_stacker = PostTrainingStacking(df_genus)

    genus_stacker.train_meta_model("Genus")

    SPECIES_OUTPUTS = dbr.get_num_species()

    species_model_paths = {
            "caud" : "src/models/spec_caud.pth", 
            "dors" : "src/models/spec_dors.pth",
            "fron" : "src/models/spec_fron.pth",
            "late" : "src/models/spec_late.pth"
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)

    species_model_list = species_ml.get_models()

    species_config = StackDatasetConfig("height.txt", "spec_dict.json", 5)
    species_stack_creator = StackDatasetCreator(species_config, df, species_model_list)
    df_species = species_stack_creator.create_flat_stack_dataset("Species")

    species_stacker = PostTrainingStacking(df_species)

    species_stacker.train_meta_model("Species")
