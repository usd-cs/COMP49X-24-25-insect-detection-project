""" eval_simulator.py """
import sys
import os
from PIL import Image
from training_database_reader import DatabaseReader
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def evaluate_images(species_evaluator, genus_evaluator, late_path, dors_path, fron_path, caud_path) -> tuple:
    # Load the provided images
    LATE_IMG = Image.open(late_path) if late_path else None
    DORS_IMG = Image.open(dors_path) if dors_path else None
    FRON_IMG = Image.open(fron_path) if fron_path else None
    CAUD_IMG = Image.open(caud_path) if caud_path else None

    # Run the species evaluation method
    top_species = species_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Run the genus evaluation method
    top_genus, genus_confidence = genus_evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    return (top_species, top_genus, genus_confidence)

if __name__ == '__main__':
    # Get Species and Genus Class Number
    dbr = DatabaseReader("training.db")
    SPECIES_OUTPUTS = dbr.get_num_species()
    GENUS_OUTPUTS = dbr.get_num_genus()

    # Get Model Files
    species_model_paths = {
            "caud" : "src/models/spec_caud.pth",
            "dors" : "src/models/spec_dors.pth",
            "fron" : "src/models/spec_fron.pth",
            "late" : "src/models/spec_late.pth"
        }

    genus_model_paths = {
            "caud" : "src/models/gen_caud.pth",
            "dors" : "src/models/gen_dors.pth",
            "fron" : "src/models/gen_fron.pth",
            "late" : "src/models/gen_late.pth"
        }

    # Load Genus Evaluator
    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    genus_evaluator = GenusEvaluationMethod("height.txt", genus_models, 1,
                                            "gen_dict.json", "gen_accuracies.json")

    # Load Species Evaluator
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    species_evaluator = EvaluationMethod("height.txt", species_models, 1,
                                         "spec_dict.json", "spec_accuracies.json")

    ###### TO BE CHANGED FOR MULTIPLE TESTS
    LATE_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    DORS_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    FRON_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    CAUD_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"
    ######

    # Genus and Species Evaluation
    top_species, top_genus, genus_conf_score = evaluate_images(
        species_evaluator=species_evaluator,
        genus_evaluator=genus_evaluator,
        late_path=LATE_PATH,
        dors_path=DORS_PATH,
        fron_path=FRON_PATH,
        caud_path=CAUD_PATH)

    print(f"1. Predicted Species: {top_species[0][0]}, Confidence: {top_species[0][1]:.2f}\n")
    print(f"2. Predicted Species: {top_species[1][0]}, Confidence: {top_species[1][1]:.2f}\n")
    print(f"3. Predicted Species: {top_species[2][0]}, Confidence: {top_species[2][1]:.2f}\n")
    print(f"4. Predicted Species: {top_species[3][0]}, Confidence: {top_species[3][1]:.2f}\n")
    print(f"5. Predicted Species: {top_species[4][0]}, Confidence: {top_species[4][1]:.2f}\n\n")

    print(f"Top Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")
