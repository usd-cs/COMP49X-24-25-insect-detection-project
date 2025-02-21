"""
Method that takes in a user input image or set of user images and runs them through 
the loaded trained models and creates a combined classification output
"""
from torchvision import transforms, models
import torch

class EvaluationMethod:
    """
    Takes image input and creates a classification by running the image through
    loaded CNN models
    """

    def __init__(self, height_filename, models_dict):
        """
        Load the trained models for usage and have the class prepared for user input.
        During testing phases, determining which evaluation method defined below will 
        be chosen here as well
        """
        self.use_method = 1     #1 = heaviest, 2 = weighted, 3 = stacked

        #weights for use in weighted eval. Can be tweaked later to optimize evaluation accuracy
        self.weights = [0.25, 0.25, 0.25, 0.25]
        self.trained_models = models_dict

        self.height = None
        with open("models/" + height_filename, 'r', encoding='utf-8') as file:
            self.height = int(file.readline().strip())

    def evaluate_image(self, late=None, dors=None, fron=None, caud=None):
        """
        Create an evaluation of the input image(s) by running each given image through
        its respective model and then run the output of the models through the evaluation method
        and return the classification

        Returns: Classification of input images and confidence score. 
                A return of None, -1 indicates an error
        """
        #define variables outside the if statements so they can be used in other method calls
        late_confidence_score = 0
        late_predicted_species = None
        dors_confidence_score = 0
        dors_predicted_species = None
        fron_confidence_score = 0
        fron_predicted_species = None
        caud_confidence_score = 0
        caud_predicted_species = None

        if late:
            late_image = self.transform_input(late)

            with torch.no_grad():
                late_output = self.trained_models["late"](late_image)

            # Get the predicted class and confidence score
            _, predictedIndex = torch.max(late_output, 1)
            late_confidence_score = torch.nn.functional.softmax(
                late_output, dim=1)[0][predictedIndex].item()
            late_predicted_species = predictedIndex.item()

        if dors:
            #mirrors above usage but for the dors angle
            dors_image = self.transform_input(dors)

            with torch.no_grad():
                dors_output = self.trained_models["dors"](dors_image)

            _, predictedIndex = torch.max(dors_output, 1)
            dors_confidence_score = torch.nn.functional.softmax(
                dors_output, dim=1)[0][predictedIndex].item()
            dors_predicted_species = predictedIndex.item()

        if fron:
            #mirrors above usage but for the fron angle
            fron_image = self.transform_input(fron)

            with torch.no_grad():
                fron_output = self.trained_models["fron"](fron_image)

            _, predictedIndex = torch.max(fron_output, 1)
            fron_confidence_score = torch.nn.functional.softmax(
                fron_output, dim=1)[0][predictedIndex].item()
            fron_predicted_species = predictedIndex.item()

        if caud:
            #mirrors above usage but for the caud angle
            caud_image = self.transform_input(caud)

            with torch.no_grad():
                caud_output = self.trained_models["caud"](caud_image)

            _, predictedIndex = torch.max(caud_output, 1)
            caud_confidence_score = torch.nn.functional.softmax(
                caud_output, dim=1)[0][predictedIndex].item()
            caud_predicted_species = predictedIndex.item()

        if self.use_method == 1:
            use_model = self.heaviest_is_best(fron_confidence_score, dors_confidence_score,
                                              late_confidence_score, caud_confidence_score)

            #match uses the index returned from the method to decide which prediction to return
            match use_model:
                case 0:
                    return fron_predicted_species, fron_confidence_score
                case 1:
                    return dors_predicted_species, dors_confidence_score
                case 2:
                    return late_predicted_species, late_confidence_score
                case 3:
                    return caud_predicted_species, caud_confidence_score
                case _:
                    return None, -1

        elif self.use_method == 2:
            return self.weighted_eval([fron_confidence_score, dors_confidence_score,
                                       late_confidence_score, caud_confidence_score],
                                      [fron_predicted_species, dors_predicted_species,
                                       late_predicted_species, caud_predicted_species])

        elif self.use_method == 3:
            return self.stacked_eval()

        else:
            return None, -1

    def heaviest_is_best(self, fron_cert, dors_cert, late_cert, caud_cert):
        """
        Takes the certainties of the models and returns the most 
        certain model's specification

        Returns: specifies most certain model
        """
        certainties = [fron_cert, dors_cert, late_cert, caud_cert]
        highest = 0
        index = 0
        ind_tracker = 0
        for i in certainties:
            if i > highest:
                highest = i
                index = ind_tracker

            ind_tracker += 1

        return index

    def weighted_eval(self, conf_scores, species_predictions):
        """
        Takes the classifications of the models and combines them based on programmer determined
        weights to create a single output

        Returns: classification of combined models
        """
        species_scores = {}

        for i in range(4):
            weighted_score = self.weights[i] * conf_scores[i]

            if species_predictions[i] in species_scores:
                species_scores[species_predictions[i]] += weighted_score

            else:
                species_scores[species_predictions[i]] = weighted_score

        highest_score = -1
        highest_species = None

        for i, j in species_scores.items():
            if j > highest_score:
                highest_score = j
                highest_species = i

        return highest_species, highest_score

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
