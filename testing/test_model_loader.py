""" test_model_loader.py """

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import torch
import torchvision.models as models
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_loader import TrainedModels

class TestModelLoader(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="224\n")
    @patch("torch.load")  # Mock torch.load for simulating loading model weights
    def test_load_model_weights(self, mock_torch_load, mock_open_file):
        # create mock model
        height_file_paths = {
            "caud" : "mock_height.txt",
            "dors" : "mock_height.txt",
            "fron" : "mock_height.txt",
            "late" : "mock_height.txt"
        }
        weights_file_paths = {
            "caud" : "mock_weights.pth",
            "dors" : "mock_weights.pth",
            "fron" : "mock_weights.pth",
            "late" : "mock_weights.pth"
        }

        testing_instance = TrainedModels(weights_file_paths, height_file_paths, test=True)
        testing_instance.models = {
            "caud" : MagicMock(),
            "dors" : MagicMock(),
            "fron" : MagicMock(),
            "late" : MagicMock()
        }
        
        # Create a mock ResNet18 model and extract a valid state_dict
        real_model = models.resnet18(weights=None)
        num_features = real_model.fc.in_features
        real_model.fc = torch.nn.Linear(num_features, 15)
        valid_state_dict = real_model.state_dict()

        # Mock torch.load to return a valid state_dict
        mock_torch_load.return_value = valid_state_dict

        # Mock load_state_dict to prevent size mismatch error
        for model in testing_instance.models.values():
            model.load_state_dict = MagicMock()
        
        testing_instance.model_initializer()

        # Verify that height is correctly loaded
        for key in ["caud", "dors", "fron", "late"]:
            self.assertEqual(testing_instance.heights[key], 224)
        # Ensure torch.load() was called correctly
        for key in ["caud", "dors", "fron", "late"]:
            mock_torch_load.assert_any_call(weights_file_paths[key], map_location=testing_instance.device)
        # Ensure load_state_dict() was called on the models
        for key in ["caud", "dors", "fron", "late"]:
            model = testing_instance.models[key]
            model.load_state_dict.assert_called_once_with(valid_state_dict)


    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('sys.stdout', new_callable=StringIO)
    def test_load_model_weights_file_not_found(self, mock_std_out, mock_open):
        # create mock model
        height_file_paths = {
            "caud" : "mock_height.txt",
            "dors" : "mock_height.txt",
            "fron" : "mock_height.txt",
            "late" : "mock_height.txt"
        }
        weights_file_paths = {
            "caud" : "mock_weights.pth",
            "dors" : "mock_weights.pth",
            "fron" : "mock_weights.pth",
            "late" : "mock_weights.pth"
        }
        testing_instance = TrainedModels(weights_file_paths, height_file_paths, test=True)
        testing_instance.models = {
            "caud" : MagicMock(),
            "dors" : MagicMock(),
            "fron" : MagicMock(),
            "late" : MagicMock()
        }
        # Call load_model_weights with false file paths
        testing_instance.model_initializer()
        # Assert that the correct error message was printed
        self.assertIn("Height File for caud Model Does Not Exist.", mock_std_out.getvalue())

if __name__ == "__main__":
    unittest.main()