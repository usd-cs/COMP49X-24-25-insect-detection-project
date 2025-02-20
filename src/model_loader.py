""" model_loader.py """

from torchvision import models
import torch

class TrainedModels:
    def __init__(self, weights_file_paths = None, test = False):
        """Initialize class variables. User can specify height parameter for model."""
        self.weights_file_paths = weights_file_paths or {}

        self.models = {
            "caud" : None, 
            "dors" : None,
            "fron" : None,
            "late" : None
        }

        # use CPU to allow general usability and Metal Performance Shader if user has Apple Silicon
        self.device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        
        if not test:
            self.model_initializer()
    
    def model_initializer(self):
        for key in self.models:
            # Initialize a fresh model with weights = None, so there are no weights
            self.models[key] = models.resnet18(weights=None)

            num_features = self.models[key].fc.in_features
            self.models[key].fc = torch.nn.Linear(num_features, 15)

            self.models[key] = self.models[key].to(self.device)

            self.load_model_weights(key)

    def load_model_weights(self, key):
        """
        Loads the specified model with weights and height stored in each file specified by filename.

        Returns: None
        """

        if not self.weights_file_paths:
            weights_file_path = input(f"Please input the file path of the saved weights for the {key} trained model: ")
        else:
            weights_file_path = self.weights_file_paths[key]

        # Load weights from file path into the model
        try:
            self.models[key].load_state_dict(torch.load(weights_file_path, map_location=self.device, weights_only=True))
        except FileNotFoundError:
            print(f"Weights File for {key} Model Does Not Exist.")
            return
    
    def get_models(self):
        return self.models

    def get_model(self, key):
        return self.models[key]