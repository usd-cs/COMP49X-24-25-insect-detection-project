"""test_post_eval_stack_training.py"""
import unittest
from unittest.mock import patch
import os
import sys
import pandas as pd
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from post_eval_stack_training import PostTrainingStacking

class TestPostTrainingStacking(unittest.TestCase):
    """test post training stacking class"""

    def setUp(self):
        """Fake dataframe with numeric labels for training"""
        self.df = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4],
            'feature2': [0.5, 0.6, 0.7, 0.8],
            'Genus': [0, 1, 0, 1]  # numeric labels
        })

    @patch('torch.save')
    @patch('builtins.print')
    def test_train_meta_model_saves_model(self, _, mock_torch_save):
        """
        test to ensure the model is properly saved and does not raise any
        errors while training
        """
        stacker = PostTrainingStacking(self.df)

        # Train on 'Genus' label
        stacker.train_meta_model(label="Genus")

        # Check if torch.save was called once
        mock_torch_save.assert_called_once()

        # Validate that meta_model is trained and exists
        self.assertIsNotNone(stacker.meta_model, "meta_model should be set after training")
        self.assertIsInstance(stacker.meta_model, torch.nn.Linear,
                            "meta_model should be a torch.nn.Linear instance")

    def test_device_selection(self):
        """
        tests device selection to ensure the correct
        device is used when available
        """
        stacker = PostTrainingStacking(self.df)

        expected_device = 'mps' if torch.backends.mps.is_built() else 'cpu'
        self.assertEqual(stacker.device.type, expected_device,
                         f"Device should be {expected_device}")

if __name__ == '__main__':
    unittest.main()
