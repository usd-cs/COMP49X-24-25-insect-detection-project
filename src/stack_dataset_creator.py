"""stack_dataset_creator.py"""
from io import BytesIO
import json
import dill
import torch
import pandas as pd
from PIL import Image

class StackDatasetConfig:
    """Holds some arguments for dataset creator class"""
    def __init__(self, height_filename, model_dict_file, num_evals):
        self.height_filename = height_filename
        self.model_dict_file = model_dict_file
        self.num_evals = num_evals

class StackDatasetCreator:
    """
    Takes four models and creates a dataframe of their classifications of all images in
    the dataset and their proper classifications
    """

    def __init__(self, config, dataframe, models_dict):
        """
        Load the trained models for evaluating test data inputs
        and prepare to use for creation of a new dataset
        """
        self.trained_models = models_dict

        self.height = None
        with open("src/models/" + config.height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

        self.transformations = self.get_transformations()

        self.idx_dict = self.open_class_dictionary(config.model_dict_file)

        #Stores 1 for Genus and 5 for Species
        self.k = config.num_evals

        self.dataframe = dataframe
        self.specimen_groups = dataframe.groupby("SpecimenID")

    def open_class_dictionary(self, filename):
        """
        Open and save the class dictionary for use in the evaluation method 
        to convert the model's index to a string species classification

        Returns: dictionary defined by file
        """
        with open("src/models/" + filename, 'r', encoding='utf-8') as json_file:
            class_dict = json.load(json_file)

        # Convert string keys to integers(because of how the dictionary was saved with json)
        class_dict = {int(key): value for key, value in class_dict.items()}

        return class_dict

    def get_transformations(self):
        """
        Create and return a list of transformations for each angle using
        the pre-made transformation files

        Returns: list of transformations
        """
        transformations = []

        #open each file and load the transformation then save it to the list
        with open("caud_transformation.pth", "rb") as f:
            transformations.append(dill.load(f))

        with open("dors_transformation.pth", "rb") as f:
            transformations.append(dill.load(f))

        with open("fron_transformation.pth", "rb") as f:
            transformations.append(dill.load(f))

        with open("late_transformation.pth", "rb") as f:
            transformations.append(dill.load(f))

        return transformations

    def get_classification(self, image, angle_int):
        """
        Get the classification for an image from a specified model

        Returns: List of tuples varying depending on the num_evals input
        [(classification, certainty), ...]
        """
        device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
        transformed_image = None
        model = None
        image = Image.open(BytesIO(image)).convert("RGB")

        if angle_int == 0:
            transformed_image = self.transform_input(image, self.transformations[0]).to(device)
            model = self.trained_models["caud"].to(device)

        elif angle_int == 1:
            transformed_image = self.transform_input(image, self.transformations[1]).to(device)
            model = self.trained_models["dors"].to(device)

        elif angle_int == 2:
            transformed_image = self.transform_input(image, self.transformations[2]).to(device)
            model = self.trained_models["fron"].to(device)

        else:
            transformed_image = self.transform_input(image, self.transformations[3]).to(device)
            model = self.trained_models["late"].to(device)

        outputs = model(transformed_image)
        softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

        if self.k == 1:
            certainty, predicted = torch.max(softmax_outputs, dim=1)
            return [(predicted.item(), certainty.item())]

        certainty, predicted = torch.topk(softmax_outputs, self.k, dim=1)
        return list(zip(predicted.squeeze().tolist(), certainty.squeeze().tolist()))

    def create_flat_stack_dataset(self, label_column):
        """
        Creates a new pandas dataframe of classifications and certainties 
        from the given model's classifications associated with the proper int
        representation of what they should find.

        args:
            label_column: 'Genus' or 'Species' depending on which model is
            being trained
        
        returns: pandas dataframe
        """
        data_rows = []
        reverse_idx_dict = {v:k for k, v in self.idx_dict.items()}

        for _, group in self.specimen_groups:
            row_features = []
            label = group.iloc[0][label_column]
            label = reverse_idx_dict[label]

            for angle_name, angle_int in zip(["caud", "dors", "fron", "late"], [0, 1, 2, 3]):
                angle_data = group[group["View"] == angle_name.upper()]
                if not angle_data.empty:
                    image = angle_data.iloc[0]["Image"]
                    predictions = self.get_classification(image, angle_int)

                    for cls, cert in predictions:
                        row_features.extend([cls, cert])
                else:
                    row_features.extend([-1, 0.0] * self.k)

            data_rows.append(row_features + [label])

        columns = []
        for angle in ["caud", "dors", "fron", "late"]:
            for i in range(self.k):
                columns.append(f"{angle}_class_{i+1}")
                columns.append(f"{angle}_certainty_{i+1}")
        columns.append(label_column)

        df = pd.DataFrame(data_rows, columns=columns)
        print("Stack dataset creation finished")
        return df

    def transform_input(self, image_input, transformation):
        """
        Takes the app side's image and a given transformation
        and transforms it to fit our model

        Returns: transformed image for classification
        """
        transformed_image = transformation(image_input)
        transformed_image = transformed_image.unsqueeze(0)

        return transformed_image
