"""
Method that takes in a user input image or set of user images and runs them through 
the loaded trained models and creates a combined classification output
"""
import sys
import os
import json
from torchvision import transforms
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

class GenusEvaluationMethod:
    """
    Takes image input and creates a classification by running the image through
    loaded CNN models
    """

    def __init__(self, height_filename, models_dict, eval_method, genus_filename):
        """
        Load the trained models for usage and have the class prepared for user input.
        During testing phases, determining which evaluation method defined below will 
        be chosen here as well
        """
        self.use_method = eval_method     #1 = heaviest, 2 = weighted, 3 = stacked

        #weights for use in weighted eval. Can be tweaked later to optimize evaluation accuracy
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.trained_models = models_dict

        self.genus_idx_dict = self.open_class_dictionary(genus_filename)

        self.height = None
        with open("src/models/" + height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

    def open_class_dictionary(self, filename):
        """
        Open and save the class dictionary for use in the evaluation method 
        to convert the model's index to a string species classification

        Returns: dictionary defined by file
        """
        with open("src/models/" + filename, 'r', encoding='utf-8') as json_file:
            class_dict_read = json.load(json_file)

        #Convert string keys to integers(keys automatically switched by json save)
        #Undoes issues created by json saving
        class_dict = {}

        for key, value in class_dict_read.items():
            class_dict[int(key)] = value

        return class_dict

    def evaluate_image(self, late=None, dors=None, fron=None, caud=None):
        """
        Create an evaluation of the input image(s) by running each given image through
        its respective model and then run the output of the models through the evaluation method
        and return the classification

        Returns: Classification of input images and confidence score. 
                A return of None, -1 indicates an error
        """
        device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')

        #define variables outside the if statements so they can be used in other method calls
        predictions = {
            "late" : {"score" : 0, "genus" : None},
            "dors" : {"score" : 0, "genus" : None},
            "fron" : {"score" : 0, "genus" : None},
            "caud" : {"score" : 0, "genus" : None},
        }

        if late:
            late_image = self.transform_input(late).to(device)

            with torch.no_grad():
                late_output = self.trained_models["late"].to(device)(late_image)

            # Get the predicted class and confidence score
            _, predicted_index = torch.max(late_output, 1)
            predictions["late"]["score"] = torch.nn.functional.softmax(
                late_output, dim=1)[0, predicted_index].item()
            predictions["late"]["genus"] = predicted_index.item()

        if dors:
            #mirrors above usage but for the dors angle
            dors_image = self.transform_input(dors).to(device)

            with torch.no_grad():
                dors_output = self.trained_models["dors"].to(device)(dors_image)

            _, predicted_index = torch.max(dors_output, 1)
            predictions["dors"]["score"] = torch.nn.functional.softmax(
                dors_output, dim=1)[0, predicted_index].item()
            predictions["dors"]["genus"] = predicted_index.item()

        if fron:
            #mirrors above usage but for the fron angle
            fron_image = self.transform_input(fron).to(device)

            with torch.no_grad():
                fron_output = self.trained_models["fron"].to(device)(fron_image)

            _, predicted_index = torch.max(fron_output, 1)
            predictions["fron"]["score"] = torch.nn.functional.softmax(
                fron_output, dim=1)[0, predicted_index].item()
            predictions["fron"]["genus"] = predicted_index.item()

        if caud:
            #mirrors above usage but for the caud angle
            caud_image = self.transform_input(caud).to(device)

            with torch.no_grad():
                caud_output = self.trained_models["caud"].to(device)(caud_image)

            _, predicted_index = torch.max(caud_output, 1)
            predictions["caud"]["score"] = torch.nn.functional.softmax(
                caud_output, dim=1)[0, predicted_index].item()
            predictions["caud"]["genus"] = predicted_index.item()

        if self.use_method == 1:
            #match uses the index returned from the method to decide which prediction to return
            return self.heaviest_is_best([predictions["fron"]["score"],
                                       predictions["dors"]["score"],
                                       predictions["late"]["score"],
                                       predictions["caud"]["score"]],
                                      [predictions["fron"]["genus"],
                                       predictions["dors"]["genus"],
                                       predictions["late"]["genus"],
                                       predictions["caud"]["genus"]])

        if self.use_method == 2:
            return self.weighted_eval([predictions["fron"]["score"],
                                       predictions["dors"]["score"],
                                       predictions["late"]["score"],
                                       predictions["caud"]["score"]],
                                      [predictions["fron"]["genus"],
                                       predictions["dors"]["genus"],
                                       predictions["late"]["genus"],
                                       predictions["caud"]["genus"]])

        if self.use_method == 3:
            return self.stacked_eval()

        return None, -1

    def heaviest_is_best(self, conf_scores, genus_predictions):
        """
        Takes the certainties of the models and returns the most 
        certain model's specification

        Returns: specifies most certain model
        """
        highest = 0
        index = 0
        ind_tracker = 0
        for i in conf_scores:
            if i > highest:
                highest = i
                index = ind_tracker

            ind_tracker += 1

        match index:
            case 0:
                return self.genus_idx_dict[genus_predictions[0]], conf_scores[0]
            case 1:
                return self.genus_idx_dict[genus_predictions[1]], conf_scores[1]
            case 2:
                return self.genus_idx_dict[genus_predictions[2]], conf_scores[2]
            case 3:
                return self.genus_idx_dict[genus_predictions[3]], conf_scores[3]


    def weighted_eval(self, conf_scores, genus_predictions):
        """
        Takes the classifications of the models and combines them based on programmer determined
        weights to create a single output

        Returns: classification of combined models
        """
        species_scores = {}

        for i in range(4):
            weighted_score = self.weights[i] * conf_scores[i]

            if genus_predictions[i] in species_scores:
                species_scores[genus_predictions[i]] += weighted_score

            else:
                species_scores[genus_predictions[i]] = weighted_score

        highest_score = -1
        highest_species = None

        for i, j in species_scores.items():
            if j > highest_score:
                highest_score = j
                highest_species = i

        return self.genus_idx_dict[highest_species], highest_score

    def stacked_eval(self):
        """
        Takes the classifications of the models and runs them through another model that determines
        the overall output

        REACH CASE/STUB FOR SPRINT 3

        Returns: classification of combined models
        """

    def transform_input(self, image_input):
        """
        Takes the app side's image and transforms it to fit our model

        Returns: transformed image for classification
        """
        transformation = transforms.Compose([
            transforms.Resize((self.height, self.height)), #ResNet expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transformed_image = transformation(image_input)
        transformed_image = transformed_image.unsqueeze(0)

        return transformed_image
