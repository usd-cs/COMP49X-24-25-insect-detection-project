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
import globals

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# simple simulation of end-to-end functionality of files

if __name__ == '__main__':
    train_dors = False
    train_caud = False
    train_fron = False
    train_late = False
    can_continue = False

    while not can_continue:
        print("Dorsal: 1\nCaudal: 2\nFrontal: 3\nLateral: 4")
        input = int(input("Choose a model you would like to train (type corresponding number): "))
        if input == 1:
            train_dors = True
        elif input == 2:
            train_caud = True
        elif input == 3:
            train_fron = True
        elif input == 4:
            train_late = True
        else:
            print("Invalid Input")
        del input
        continue_input = int(
            input(
                "Press 1 to choose more models to train, anything other number to start training: "
                )
                )
        if continue_input != 1:
            can_continue = True
            if not train_dors and not train_late and not train_caud and not train_fron:
                print("No Training Requested")
                sys.exit(0)
    # Set up data converter
    tdc = TrainingDataConverter("dataset")
    tdc.conversion(globals.training_database)
    # Read converted data
    dbr = DatabaseReader(database=globals.training_database, class_file_path=globals.class_list)
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
    species_tp = TrainingProgram(df, 1, SPECIES_OUTPUTS)

    # Training
    if train_caud:
        species_tp.train_caudal(20)
    if train_dors:
        species_tp.train_dorsal(20)
    if train_fron:
        species_tp.train_frontal(20)
    if train_late:
        species_tp.train_lateral(20)

    # Save models
    species_model_filenames = {
            "caud" : globals.spec_caud_model if train_caud else None, 
            "dors" : globals.spec_dors_model if train_dors else None,
            "fron" : globals.spec_fron_model if train_fron else None,
            "late" : globals.spec_late_model if train_late else None
        }

    species_tp.save_models(
        species_model_filenames,
        globals.img_height,
        globals.spec_class_dictionary,
        globals.spec_accuracy_list)

    # Run training with dataframe
    genus_tp = TrainingProgram(df, 0, GENUS_OUTPUTS)

    # Training
    if train_caud:
        genus_tp.train_caudal(20)
    if train_dors:
        genus_tp.train_dorsal(20)
    if train_fron:
        genus_tp.train_frontal(20)
    if train_late:
        genus_tp.train_lateral(20)

    # Save models
    genus_model_filenmaes = {
        "caud" : globals.gen_caud_model if train_caud else None, 
        "dors" : globals.gen_dors_model if train_dors else None,
        "fron" : globals.gen_fron_model if train_fron else None,
        "late" : globals.gen_late_model if train_late else None
    }

    genus_tp.save_models(
        genus_model_filenmaes,
        globals.img_height,
        globals.gen_class_dictionary,
        globals.gen_accuracy_list)

    # Load Genus models
    genus_model_paths = {
            "caud" : "src/models/" + globals.gen_caud_model, 
            "dors" : "src/models/" + globals.gen_dors_model,
            "fron" : "src/models/" + globals.gen_fron_model,
            "late" : "src/models/" + globals.gen_late_model
        }

    genus_ml = ModelLoader(genus_model_paths, GENUS_OUTPUTS)
    genus_models = genus_ml.get_models()

    print(genus_models.keys)
    print(genus_ml.get_model("caud").named_parameters())

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    genus_evaluator = GenusEvaluationMethod(globals.img_height, genus_models, 1,
                                            globals.gen_class_dictionary,
                                            globals.gen_accuracy_list)

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
            "caud" : "src/models/" + globals.spec_caud_model, 
            "dors" : "src/models/" + globals.spec_dors_model,
            "fron" : "src/models/" + globals.spec_fron_model,
            "late" : "src/models/" + globals.spec_late_model
        }
    species_ml = ModelLoader(species_model_paths, SPECIES_OUTPUTS)
    species_models = species_ml.get_models()

    print(species_models.keys)
    print(species_ml.get_model("caud").named_parameters())

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    species_evaluator = EvaluationMethod(globals.img_height, species_models, 1,
                                         globals.spec_class_dictionary,
                                         globals.spec_accuracy_list)

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
