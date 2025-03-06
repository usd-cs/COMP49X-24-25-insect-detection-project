""" simulator.py """
import sys
import os
from PIL import Image
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    # Set up data converter
    tdc = TrainingDataConverter("dataset")
    tdc.conversion("training.db")
    # Read converted data
    dbr = DatabaseReader("training.db")
    df = dbr.get_dataframe()

    # initialize number of outputs
    SPECIES_OUTPUTS = 15
    GENUS_OUTPUTS = 3

    # Run training with dataframe
    species_tp = TrainingProgram(df, 1, SPECIES_OUTPUTS)

    # Training
    species_tp.train_caudal(1)
    species_tp.train_dorsal(1)
    species_tp.train_frontal(1)
    species_tp.train_lateral(1)

    # Save models
    species_tp.save_models(
        "spec_caud.pth",
        "spec_dors.pth",
        "spec_fron.pth",
        "spec_late.pth",
        "height.txt",
        "spec_dict.json")

    # Run training with dataframe
    genus_tp = TrainingProgram(df, 0, GENUS_OUTPUTS)

    # Training
    genus_tp.train_caudal(1)
    genus_tp.train_dorsal(1)
    genus_tp.train_frontal(1)
    genus_tp.train_lateral(1)

    # Save models
    genus_tp.save_models("gen_caud.pth",
        "gen_dors.pth",
        "gen_fron.pth",
        "gen_late.pth",
        "height.txt",
        "gen_dict.json")

    # Load Genus models
    genus_model_paths = {
            "caud" : "src/models/gen_caud.pth", 
            "dors" : "src/models/gen_dors.pth",
            "fron" : "src/models/gen_fron.pth",
            "late" : "src/models/gen_late.pth"
        }

    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    print(genus_models.keys)
    print(genus_ml.get_model("caud").named_parameters())
    # pylint: disable=C0206
    # Set models to evaluation mode
    for key in genus_models:
        genus_models[key].eval()

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    genus_evaluator = GenusEvaluationMethod("height.txt", genus_models, 1, "gen_dict.json")

    # Get the images to be evaluated through user input
    LATE_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    DORS_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    FRON_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    CAUD_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"

    # Load the provided images
    LATE_IMG = Image.open(LATE_PATH) if LATE_PATH else None
    DORS_IMG = Image.open(DORS_PATH) if DORS_PATH else None
    FRON_IMG = Image.open(FRON_PATH) if FRON_PATH else None
    CAUD_IMG = Image.open(CAUD_PATH) if CAUD_PATH else None

    # Run the evaluation method to find the predicted genus
    top_genus, genus_conf_score = genus_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results for genus
    print(f"Predicted Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")

    # Load species models
    species_model_paths = {
            "caud" : "src/models/caud.pth", 
            "dors" : "src/models/dors.pth",
            "fron" : "src/models/fron.pth",
            "late" : "src/models/late.pth"
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    print(species_models.keys)
    print(species_ml.get_model("caud").named_parameters())
    # pylint: disable=C0206
    # Set models to evaluation mode
    for key in species_models:
        species_models[key].eval()

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod("height.txt", species_models, 1, "dict.json")

    # Run the evaluation method
    top_5_species = species_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results
    print(f"1. Predicted Species: {top_5_species[0][0]}, Confidence: {top_5_species[0][1]:.2f}\n")
    print(f"2. Predicted Species: {top_5_species[1][0]}, Confidence: {top_5_species[1][1]:.2f}\n")
    print(f"3. Predicted Species: {top_5_species[2][0]}, Confidence: {top_5_species[2][1]:.2f}\n")
    print(f"4. Predicted Species: {top_5_species[3][0]}, Confidence: {top_5_species[3][1]:.2f}\n")
    print(f"5. Predicted Species: {top_5_species[4][0]}, Confidence: {top_5_species[4][1]:.2f}\n")
