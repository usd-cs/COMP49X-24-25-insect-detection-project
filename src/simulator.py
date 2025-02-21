""" simulator.py """
import sys
import os
from PIL import Image
from training_data_converter import TrainingDataConverter
from training_database_reader import DatabaseReader
from training_program import TrainingProgram
from model_loader import ModelLoader
from evaluation_method import EvaluationMethod

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
    models = ml.get_models()

    print(models.keys)
    print(ml.get_model("caud").named_parameters())

    # Set models to evaluation mode
    for key, _ in models.items():
        models[key].eval()

    # Inititialize the EvaluationMethod object with the heaviest eval method set
    evaluator = EvaluationMethod("height.txt", models, 1)

    # Get the images to be evaluated through user input
    late_img_path = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT LATE.jpg"
    dors_img_path = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT DORS.jpg"
    fron_img_path = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT FRON.jpg"
    caud_img_path = "dataset/Callosobruchus chinensis GEM_187686348 5XEXT CAUD.jpg"

    # Load the provided images
    LATE_IMG = Image.open(late_img_path) if late_img_path else None
    DORS_IMG = Image.open(dors_img_path) if dors_img_path else None
    FRON_IMG = Image.open(fron_img_path) if fron_img_path else None
    CAUD_IMG = Image.open(caud_img_path) if caud_img_path else None

    # Run the evaluation method
    species, confidence = evaluator.evaluate_image(
        late=LATE_IMG, dors=DORS_IMG, fron=FRON_IMG, caud=CAUD_IMG
    )

    # Print classification results
    print(f"Predicted Species: {species}, Confidence: {confidence:.2f}")
