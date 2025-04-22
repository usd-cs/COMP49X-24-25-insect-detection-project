"""test_stack_dataset_creator.py"""
import unittest
from unittest.mock import patch, mock_open
import os
import sys
import json
from io import BytesIO
import torch
import pandas as pd
from PIL import Image
import dill

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from stack_dataset_creator import StackDatasetCreator
from stack_dataset_creator import StackDatasetConfig

def dummy_transform(_):
    """returns a dummy transformation"""
    return torch.rand(3, 224, 224)

class DummyModel(torch.nn.Module):
    """Creates a dummy evaluation model"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = torch.nn.Linear(150528, num_classes)  # fake input size

    def forward(self, x):
        """returns random data for testing"""
        batch_size = x.shape[0]
        return torch.rand(batch_size, 3)  # Random logits

class TestStackDatasetCreator(unittest.TestCase):
    """Test class for the stack dataset creator"""
    @classmethod
    def setUpClass(cls):
        # Create dummy image
        fake_image = BytesIO()
        Image.new('RGB', (224, 224)).save(fake_image, format='JPEG')
        image_bytes = fake_image.getvalue()

        # Create a mock DataFrame
        cls.mock_dataframe = pd.DataFrame({
            'SpecimenID': ['Specimen1', 'Specimen1'],
            'View': ['DORS', 'LATE'],
            'Image': [image_bytes, image_bytes],
            'Genus': ['Acanthoscelides', 'Acanthoscelides']
        })

        # Dummy models
        cls.mock_models = {
            "caud": DummyModel(),
            "dors": DummyModel(),
            "fron": DummyModel(),
            "late": DummyModel()
        }

        # Create a mock dictionary JSON file
        cls.mock_dict_path = "mock_dict.json"
        label_dict = {0: "Acanthoscelides"}
        with open(cls.mock_dict_path, 'w', encoding='utf8') as f:
            json.dump({str(k): v for k, v in label_dict.items()}, f)

        # Patch transformation loading
        def mock_transformations(_):
            return [dummy_transform] * 4
        StackDatasetCreator.get_transformations = mock_transformations

        # Patch dictionary loading
        def mock_dictionary(*_, **__):
            return {0: "Acanthoscelides"}
        StackDatasetCreator.open_class_dictionary = mock_dictionary

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.mock_dict_path):
            os.remove(cls.mock_dict_path)

    @patch("builtins.open", new_callable=mock_open, read_data="224\n")
    def test_create_flat_stack_dataset(self, mock_file):
        """test create flat stack dataset function"""
        config = StackDatasetConfig(
            height_filename="height.txt",
            model_dict_file=self.mock_dict_path,
            num_evals=1
        )

        creator = StackDatasetCreator(
            config=config,
            dataframe=self.mock_dataframe,
            models_dict=self.mock_models
        )

        df = creator.create_flat_stack_dataset(label_column="Genus")

        mock_file.assert_called_with('src/models/height.txt', 'r', encoding='utf-8')
        self.assertFalse(df.empty, "The dataset should not be empty")
        self.assertIn("Genus", df.columns, "The label column 'Genus' should exist")
        self.assertEqual(df.iloc[0]["Genus"], 0, "The label should be converted to integer 0")

if __name__ == '__main__':
    unittest.main()
