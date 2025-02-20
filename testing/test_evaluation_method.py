"""test_evaluation_method.py"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation_method import EvaluationMethod

class TestEvaluationMethod(unittest.TestCase):
    """
    Test the evaluation method class methods
    """

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_heaviest_is_best(self, mock_file):
        """test heaviest is best for proper tracking of highest certainty"""
        evaluation = EvaluationMethod("height_mock.txt")

        assert evaluation.heaviest_is_best(0.1, 0.3, 0.5, 0.4) == 2
        assert evaluation.heaviest_is_best(0.1, 0.9, 0.6, 0.32) == 1

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_weighted_eval(self, mock_file):
        """test weighted eval for proper calculation"""
        evaluation = EvaluationMethod("height_mock.txt")
        #must be changed if weights are adjusted in code
        given_weights = [0.25, 0.25, 0.25, 0.25]
        conf_scores = [0.8, 0.6, 0.9, 0.7]
        species_predictions = [1, 2, 2, 3]

        prediction, score = evaluation.weighted_eval(conf_scores, species_predictions)
        assert prediction == 2
        assert score == given_weights[1] * conf_scores[1] + given_weights[2] * conf_scores[2]

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_transform_input(self, mock_file):
        """test transform input for proper image transformation"""
        evaluation = EvaluationMethod("height_mock.txt")
        evaluation.height = 224
        fake_input = torch.rand(3, 224, 224)
        result = evaluation.transform_input(fake_input)

        assert result.shape == (1, 3, 224, 224)

if __name__ == "__main__":
    unittest.main()
