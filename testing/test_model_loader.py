""" test_model_loader.py """

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from io import StringIO
import torch
from torchvision import models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_loader import ModelLoader

class TestModelLoader(unittest.TestCase):
    """
    Unit testing for Model Loader 
    """
    @patch("torch.load")  # Mock torch.load to prevent actual file loading
    def test_load_model_weights(self, mock_torch_load):
        """ Test that load_model_weights correctly loads weights into the model. """
        # Create a testing instance of the ModelLoader object with test mode enabled
        weights_file_paths = {"caud": "mock_weights.pth"}
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)
        testing_instance.models["caud"] = MagicMock()
        testing_instance.device = torch.device("cpu")

        # Mock the return value of torch.load
        mock_torch_load.return_value = {"mock_key": torch.tensor([1.0])}

        # Call the load_model_weights method for testing
        testing_instance.load_model_weights("caud")

        # Assert torch.load was called with the correct arguments,
        # and the load_state_dict was called on the model
        mock_torch_load.assert_called_once_with(
            "mock_weights.pth", map_location=testing_instance.device, weights_only=True)
        testing_instance.models["caud"].load_state_dict.assert_called_once_with(
            mock_torch_load.return_value)

    @patch('sys.stdout', new_callable=StringIO)
    def test_load_model_weights_file_not_found(self, mock_stdout):
        """ Test that load_model_weights handles FileNotFoundError correctly. """
        # testing instance setup
        weights_file_paths = {"caud": "non_existent_weights.pth"}
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)
        testing_instance.models["caud"] = MagicMock()
        testing_instance.device = torch.device("cpu")

        # Call loadModelWeights with false file paths
        testing_instance.load_model_weights("caud")

        # Assert that the correct error message was printed
        self.assertIn("Weights File for caud Model Does Not Exist.", mock_stdout.getvalue())

    @patch.object(ModelLoader, "load_model_weights")  # Mock load_model_weights
    def test_model_initializer(self, mock_load_model_weights):
        """ Test that model_initializer correctly initializes the model. """
        # Create mock weights paths to create the testing instance for ModelLoader
        weights_file_paths = {
            "caud": "mock_weights.pth",
            "dors": "mock_weights.pth",
            "fron": "mock_weights.pth",
            "late": "mock_weights.pth"
        }
        testing_instance = ModelLoader(weights_file_paths, 15, test=True)

        # Mock the models to be ResNet instances
        for key in weights_file_paths:
            testing_instance.models[key] = MagicMock(spec=models.ResNet)

        # Call the model_initializer method for testing
        testing_instance.model_initializer()

        # Assert that load_model_weights was called for each model key and nothing else
        for key in weights_file_paths:
            mock_load_model_weights.assert_any_call(key)
        self.assertEqual(mock_load_model_weights.call_count, len(weights_file_paths))

if __name__ == "__main__":
    unittest.main()
