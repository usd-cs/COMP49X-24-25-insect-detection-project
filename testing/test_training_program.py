""" test_training_program.py """
import unittest
from unittest.mock import MagicMock
import pandas as pd
import sys
import os
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
            "Genus": ["GenusA", "GenusB", "GenusC", "GenusD"],
            "Species": ["SpeciesA", "SpeciesB", "SpeciesC", "SpeciesD"],
            "UniqueID": ["ID1", "ID2", "ID3", "ID4"],
            "View": ["CAUD", "DORS", "FRON", "LATE"]
        })

        # Initialize the TrainingProgram instance
        self.training_program = TrainingProgram(self.mock_dataframe)

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

if __name__ == "__main__":
    unittest.main()