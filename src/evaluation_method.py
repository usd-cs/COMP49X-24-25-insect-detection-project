"""
Method that takes in a user input image or set of user images and runs them through 
the loaded trained models and creates a combined classification output
"""

class EvaluationMethod:
    """
    Takes image input and creates a classification by running the image through
    loaded CNN models
    """

    def __init__(self):
        """
        Load the trained models for usage and have the class prepared for user input.
        During testing phases, determining which evaluation method defined below will be chosen here as well
        """
        self.use_method = 1     #1 = heaviest, 2 = weighted, 3 = stacked

        #INTEGRATION: call method to load the models here once task is completed

    def evaluate_image(self, late=None, dors=None, fron=None, caud=None):
        """
        Create an evaluation of the input image(s) by running each given image through
        its respective model and then run the output of the models through the evaluation method
        and return the classification

        Returns: Classification of input images
        """
        pass

    def heaviest_is_best(self, fron_cert, dors_cert, late_cert, caud_cert):
        """
        Takes the certainties of the models and returns the most 
        certain model's specification

        Returns: specifies most certain model
        """
        pass

    def weighted_eval(self, fron_cert, dors_cert, late_cert, caud_cert):
        """
        Takes the classifications of the models and combines them based on programmer determined
        weights to create a single output

        Returns: classification of combined models
        """
        pass

    def stacked_eval(self):
        """
        Takes the classifications of the models and runs them through another model that determines
        the overall output

        REACH CASE 

        Returns: classification of combined models
        """
        pass

    def transform_input(self, input):
        """
        Takes the app side's image and transforms it to fit our model

        Returns: transformed image for classification
        """
        pass
