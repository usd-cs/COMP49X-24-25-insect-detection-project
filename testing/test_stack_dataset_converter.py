"""test_stack_dataset_converter.py"""
import unittest
import torch
import torch.nn as nn
import pandas as pd
from io import BytesIO
from PIL import Image
import json
import dill
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from stack_dataset_creator import StackDatsetCreator

# Dummy transformation function
def dummy_transform(image):
    return torch.rand(3, 224, 224)

# Dummy model class
class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(150528, num_classes)  # fake input size

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.rand(batch_size, 3)  # Random logits

class TestStackDatasetCreator(unittest.TestCase):

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
        with open(cls.mock_dict_path, 'w') as f:
            json.dump({str(k): v for k, v in label_dict.items()}, f)

        # Patch transformation loading
        def mock_transformations(self):
            return [dummy_transform] * 4
        StackDatsetCreator.get_transformations = mock_transformations

        # Patch dictionary loading
        def mock_dictionary(self, filename):
            return {0: "Acanthoscelides"}
        StackDatsetCreator.open_class_dictionary = mock_dictionary

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.mock_dict_path):
            os.remove(cls.mock_dict_path)

    def test_create_flat_stack_dataset(self):
        creator = StackDatsetCreator(
            height_filename="height.txt",
            models_dict=self.mock_models,
            num_evals=1,
            dataframe=self.mock_dataframe,
            model_dict_file=self.mock_dict_path
        )

        df = creator.create_flat_stack_dataset(label_column="Genus")

        self.assertFalse(df.empty, "The dataset should not be empty")
        self.assertIn("Genus", df.columns, "The label column 'Genus' should exist")
        self.assertEqual(df.iloc[0]["Genus"], 0, "The label should be converted to integer 0")

if __name__ == '__main__':
    unittest.main()
