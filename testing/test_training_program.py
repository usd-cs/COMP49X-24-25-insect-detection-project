""" test_training_program.py """
import sys
import os
import unittest
import json
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from training_program import TrainingProgram

class TestTrainingProgram(unittest.TestCase):
    """
    Unit testing for training program
    """
    def setUp(self):
        """
        Set up test data and initialize the TrainingProgram instance.
        """
        # Create a mock DataFrame for testing
        self.mock_dataframe = pd.DataFrame({
            "Genus": ["GenusA", "GenusB", "GenusC", "GenusD", "GenusE",
                      "GenusF", "GenusG", "GenusH", "GenusI", "GenusJ"],
            "Species": ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD", "SpeciesE",
                        "SpeciesF", "SpeciesG", "SpeciesH", "SpeciesI", "SpeciesJ"],
            "UniqueID": ["ID1", "ID2", "ID3", "ID4", "ID5", "ID6", "ID7", "ID8", "ID9", "ID10"],
            "View": ["CAUD", "DORS", "FRON", "LATE", "CAUD", "DORS", "FRON", "LATE", "CAUD", "DORS"]
        })

        # Initialize the TrainingProgram instance
        self.training_program = TrainingProgram(self.mock_dataframe, 1, 15)

        # Mock the get_subset method to use the mock DataFrame
        self.training_program.get_subset = MagicMock(side_effect=self.mock_get_subset)

    def mock_get_subset(self, view_type, dataframe):
        """
        Mock implementation of get_subset for testing.
        Filters the mock DataFrame based on the view_type.
        Return: pd.Datafram: subset dataframe
        """
        return dataframe[dataframe["View"] == view_type] if not dataframe.empty else pd.DataFrame()

    def test_get_caudal_view(self):
        """ Test that get_caudal_view returns the correct subset. """
        # Call the method
        df = self.training_program.get_caudal_view()

        # Verify the result
        expected_df = self.mock_dataframe[self.mock_dataframe["View"] == "CAUD"]
        self.assertFalse(df.empty)
        self.assertTrue(df.equals(expected_df))

    def test_get_dorsal_view(self):
        """ Test that get_dorsal_view returns the correct subset. """
        # Call the method
        df = self.training_program.get_dorsal_view()

        # Verify the result
        expected_df = self.mock_dataframe[self.mock_dataframe["View"] == "DORS"]
        self.assertFalse(df.empty)
        self.assertTrue(df.equals(expected_df))

    def test_get_frontal_view(self):
        """ Test that get_frontal_view returns the correct subset. """
        # Call the method
        df = self.training_program.get_frontal_view()

        # Verify the result
        expected_df = self.mock_dataframe[self.mock_dataframe["View"] == "FRON"]
        self.assertFalse(df.empty)
        self.assertTrue(df.equals(expected_df))

    def test_get_lateral_view(self):
        """ Test that get_lateral_view returns the correct subset. """
        # Call the method
        df = self.training_program.get_lateral_view()

        # Verify the result
        expected_df = self.mock_dataframe[self.mock_dataframe["View"] == "LATE"]
        self.assertFalse(df.empty)
        self.assertTrue(df.equals(expected_df))

    def test_get_train_test_split(self):
        """Test get_train_test_split returns correctly split data"""
        df = self.mock_dataframe[self.mock_dataframe["View"] == "CAUD"]

        result = self.training_program.get_train_test_split(df)
        train_x, test_x, train_y, test_y = result

        # Check data types
        self.assertIsInstance(train_x, object)
        self.assertIsInstance(test_x, object)
        self.assertIsInstance(train_y, object)
        self.assertIsInstance(test_y, object)

    @patch('training_program.DataLoader')
    def test_training_evaluation_caudal(self, mock_loader):
        """ Test the caudal training and evaluation function """
        # Mock DataLoader
        mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])),  # 4 samples
        (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))   # Another 4 samples
        ])
        mock_loader.__len__.return_value = 2  # Two batches

        self.training_program.training_evaluation_caudal(1, mock_loader, mock_loader)

    def test_train_caudal(self):
        """ Test train_caudal method """
       # Mock dataset with multiple samples
        mock_train_x = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        mock_test_x = ["img5.jpg", "img6.jpg"]
        mock_train_y = [0, 1, 0, 1]
        mock_test_y = [1, 0]

        # Mock DataLoader
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.__iter__.return_value = iter([(torch.randn(2, 3, 224, 224),
                                                torch.tensor([0, 1]))])
        mock_loader.__len__.return_value = 2  # Mocked DataLoader length
        # Mock train-test split
        self.training_program.get_train_test_split = MagicMock(
            return_value=[mock_train_x, mock_test_x, mock_train_y, mock_test_y])
        # Mock evaluation function
        self.training_program.training_evaluation_caudal = MagicMock()
        # Run train_caudal
        self.training_program.train_caudal(1)
        # Ensure training_evaluation_caudal was called once
        self.training_program.training_evaluation_caudal.assert_called_once()

    @patch('training_program.DataLoader')
    def test_training_evaluation_frontal(self, mock_loader):
        """ Test the frontal training and evaluation function """
        # Mock DataLoader
        mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])),  # 4 samples
        (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))   # Another 4 samples
        ])
        mock_loader.__len__.return_value = 2  # Two batches

        self.training_program.training_evaluation_frontal(1, mock_loader, mock_loader)

    def test_train_frontal(self):
        """ Test train_frontal method """
       # Mock dataset with multiple samples
        mock_train_x = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        mock_test_x = ["img5.jpg", "img6.jpg"]
        mock_train_y = [0, 1, 0, 1]
        mock_test_y = [1, 0]

        # Mock DataLoader
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.__iter__.return_value = iter([(torch.randn(2, 3, 224, 224),
                                                torch.tensor([0, 1]))])
        mock_loader.__len__.return_value = 2  # Mocked DataLoader length
        # Mock train-test split
        self.training_program.get_train_test_split = MagicMock(
            return_value=[mock_train_x, mock_test_x, mock_train_y, mock_test_y])
        # Mock evaluation function
        self.training_program.training_evaluation_frontal = MagicMock()
        # Run train_caudal
        self.training_program.train_frontal(1)
        # Ensure training_evaluation_caudal was called once
        self.training_program.training_evaluation_frontal.assert_called_once()

    @patch('training_program.DataLoader')
    def test_training_evaluation_dorsal(self, mock_loader):
        """ Test the dorsal training and evaluation function """
        # Mock DataLoader
        mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])),  # 4 samples
        (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))   # Another 4 samples
        ])
        mock_loader.__len__.return_value = 2  # Two batches

        self.training_program.training_evaluation_dorsal(1, mock_loader, mock_loader)

    def test_train_dorsal(self):
        """ Test train_dorsal method """

        # Mock train-test split
        self.training_program.get_train_test_split = MagicMock(
            return_value=(["img1.jpg"], ["img2.jpg"], [0], [1])
        )

        # Mock the evaluation function
        self.training_program.training_evaluation_dorsal = MagicMock()

        # Run train_dorsal with 1 epoch
        self.training_program.train_dorsal(num_epochs=1)

        # Verify the evaluation function was called once
        self.training_program.training_evaluation_dorsal.assert_called_once()

    @patch('training_program.DataLoader')
    def test_training_evaluation_lateral(self, mock_loader):
        """ Test the lateral training and evaluation function """
        # Mock DataLoader
        mock_loader.__iter__.return_value = iter([
        (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 0, 1])),  # 4 samples
        (torch.randn(4, 3, 224, 224), torch.tensor([1, 0, 1, 0]))   # Another 4 samples
        ])
        mock_loader.__len__.return_value = 2  # Two batches

        self.training_program.training_evaluation_lateral(1, mock_loader, mock_loader)

    def test_train_lateral(self):
        """ Test train_lateral method """
       # Mock dataset with multiple samples
        mock_train_x = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        mock_test_x = ["img5.jpg", "img6.jpg"]
        mock_train_y = [0, 1, 0, 1]
        mock_test_y = [1, 0]

        # Mock DataLoader
        mock_loader = MagicMock(spec=DataLoader)
        mock_loader.__iter__.return_value = iter([(torch.randn(2, 3, 224, 224),
                                                torch.tensor([0, 1]))])
        mock_loader.__len__.return_value = 2  # Mocked DataLoader length
        # Mock train-test split
        self.training_program.get_train_test_split = MagicMock(
            return_value=[mock_train_x, mock_test_x, mock_train_y, mock_test_y])
        # Mock evaluation function
        self.training_program.training_evaluation_lateral = MagicMock()
        # Run train_caudal
        self.training_program.train_lateral(1)
        # Ensure training_evaluation_caudal was called once
        self.training_program.training_evaluation_lateral.assert_called_once()

    @patch("torch.save")
    def test_save_models(self, mock_torch_save):
        """ Test that save_models writes to proper files """

        # Mock previous model accuracies
        mock_accuracy_dict = {
            "caud": 0.6,
            "dors": 0.4,
            "fron": 0.7,
            "late": 0.8,
        }

        self.training_program.model_accuracies = {
            "caud": 0.5, # worse than previous
            "dors": 0.6, # improved
            "fron": 0.8, # improved
            "late": 0.8, # same
        }

        # Call the function with mocked json accuracy dump
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_accuracy_dict))):
            self.training_program.save_models(
                {
                    "caud": "caud.pth",
                    "dors": "dors.pth",
                    "fron": "fron.pth",
                    "late": "late.pth",
                },
                "height.txt",
                "dict.json",
                "test_accuracies.json"
            )

        # Verify torch.save is called for each model, ignoring exact state_dict() content
        expected_calls = [
            ((unittest.mock.ANY, os.path.join("src/models", "dors.pth")),),
            ((unittest.mock.ANY, os.path.join("src/models", "fron.pth")),),
        ]
        mock_torch_save.assert_has_calls(expected_calls, any_order=True)

if __name__ == "__main__":
    unittest.main()
