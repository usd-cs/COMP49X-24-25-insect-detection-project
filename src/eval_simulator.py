""" eval_simulator.py """
import sys
import os
from PIL import Image
from training_database_reader import DatabaseReader
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod
from genus_evaluation_method import GenusEvaluationMethod
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def evaluate_images(species_eval,
                    genus_eval,
                    late_path,
                    dors_path,
                    fron_path,
                    caud_path) -> tuple:
    """
    Uses the inputted models to classify the species and 
    genus of the inputted bug.
    
    Returns: Tuple of top species list and genus/genus confidence
    """
    # Load the provided images
    LATE_IMG = Image.open(late_path) if late_path else None
    DORS_IMG = Image.open(dors_path) if dors_path else None
    FRON_IMG = Image.open(fron_path) if fron_path else None
    CAUD_IMG = Image.open(caud_path) if caud_path else None

    # Run the species evaluation method
    top_spec = species_eval.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Run the genus evaluation method
    top_gen, genus_confidence = genus_eval.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    return (top_spec, top_gen, genus_confidence)

if __name__ == '__main__':
    # Get Species and Genus Class Number
    dbr = DatabaseReader(globals.training_database, class_file_path=globals.class_list)
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

    genus_evaluator = GenusEvaluationMethod(globals.img_height, genus_models, 1,
                                            globals.gen_class_dictionary, globals.gen_accuracy_list)

    # Load Species Evaluator
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    species_evaluator = EvaluationMethod(globals.img_height, species_models, 1,
                                         globals.spec_class_dictionary, globals.spec_accuracy_list)

    ###### TO BE CHANGED FOR MULTIPLE TESTS
    #LATE_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    #DORS_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    #FRON_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    #CAUD_PATH = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"
    ######

    user_input = input("Enter a specimen ID to be evaluated: ")
    filtered_images = dbr.dataframe[dbr.dataframe['SpecimenID'] == user_input]
    file_name_substring = (
        "dataset/" +
        filtered_images.iloc[0]['Genus'] + " " +
        filtered_images.iloc[0]['Species'] + " " +
        filtered_images.iloc[0]['SpecimenID'] + " 5XEXT "
                           )
    LATE_PATH = file_name_substring + "LATE.jpg"
    DORS_PATH = file_name_substring + "DORS.jpg"
    FRON_PATH = file_name_substring + "FRON.jpg"
    CAUD_PATH = file_name_substring + "CAUD.jpg"

    # Genus and Species Evaluation
    top_species, top_genus, genus_conf_score = evaluate_images(
        species_eval=species_evaluator,
        genus_eval=genus_evaluator,
        late_path=LATE_PATH if os.path.exists(LATE_PATH) else None,
        dors_path=DORS_PATH if os.path.exists(DORS_PATH) else None,
        fron_path=FRON_PATH if os.path.exists(FRON_PATH) else None,
        caud_path=CAUD_PATH if os.path.exists(CAUD_PATH) else None)

    print(f"1. Predicted Species: {top_species[0][0]}, Confidence: {top_species[0][1]:.2f}\n")
    print(f"2. Predicted Species: {top_species[1][0]}, Confidence: {top_species[1][1]:.2f}\n")
    print(f"3. Predicted Species: {top_species[2][0]}, Confidence: {top_species[2][1]:.2f}\n")
    print(f"4. Predicted Species: {top_species[3][0]}, Confidence: {top_species[3][1]:.2f}\n")
    print(f"5. Predicted Species: {top_species[4][0]}, Confidence: {top_species[4][1]:.2f}\n\n")

    print(f"Top Genus: {top_genus}, Confidence: {genus_conf_score:.2f}\n")
