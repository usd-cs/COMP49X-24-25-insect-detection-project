"""test_evaluation_method.py"""
import unittest
import sys
import os
from unittest.mock import patch, mock_open, MagicMock
from PIL import Image
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation_method import EvaluationMethod

class TestEvaluationMethod(unittest.TestCase):
    """
    Test the evaluation method class methods
    """
    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_initializer(self, mock_file):
        """test the initializer for proper setup"""
        #mock the models
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models)

        mock_file.assert_called_once_with("models/height_mock.txt")
        self.assertEqual(evaluation.use_method, 1)
        #Change the weights to match the program's manually
        self.assertEqual(evaluation.weights, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(evaluation.trained_models, mock_models)
        self.assertEqual(evaluation.height, 224)


    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_heaviest_is_best(self, mock_file):
        """test heaviest is best for proper tracking of highest certainty"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models)
        mock_file.assert_called_once_with("models/height_mock.txt", 'r', encoding='utf-8')

        assert evaluation.heaviest_is_best(0.1, 0.3, 0.5, 0.4) == 2
        assert evaluation.heaviest_is_best(0.1, 0.9, 0.6, 0.32) == 1

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def test_weighted_eval(self, mock_file):
        """test weighted eval for proper calculation"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models)
        mock_file.assert_called_once_with("models/height_mock.txt", 'r', encoding='utf-8')
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
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models)
        mock_file.assert_called_once_with("models/height_mock.txt", 'r', encoding='utf-8')
        evaluation.height = 224
        fake_input = Image.new("RGB", (224, 224))
        result = evaluation.transform_input(fake_input)

        assert result.shape == (1, 3, 224, 224)

    @patch("evaluation_method.EvaluationMethod.heaviest_is_best", return_value = 0)
    def test_evaluate_image_uses_heaviest(self, mock_heaviest):
        """test proper usage when self.use_method = 1"""
        evaluation = self.create_evaluation_object_for_evaluate_image_tests()

        evaluation.use_method = 1
        evaluation.evaluate_image(late=torch.rand(3, 224, 224))

        mock_heaviest.assert_called_once()

    @patch("evaluation_method.EvaluationMethod.weighted_eval", return_value = (2, 0.75))
    def test_evaluate_image_uses_weighted(self, mock_weighted):
        """test proper usage when self.use_method = 2"""
        evaluation = self.create_evaluation_object_for_evaluate_image_tests()

        evaluation.use_method = 2
        result_species, result_conf = evaluation.evaluate_image(late=torch.rand(3, 224, 224))

        mock_weighted.assert_called_once()
        self.assertEqual(result_species, 2)
        self.assertEqual(result_conf, 0.75)

    @patch("evaluation_method.EvaluationMethod.stacked_eval", return_value = (2, 0.75))
    def test_evaluate_image_uses_stacked(self, mock_stacked):
        """test proper usage when self.use_method = 3"""
        evaluation = self.create_evaluation_object_for_evaluate_image_tests()

        evaluation.use_method = 3
        result_species, result_conf = evaluation.evaluate_image(late=torch.rand(3, 224, 224))

        mock_stacked.assert_called_once()
        self.assertEqual(result_species, 2)
        self.assertEqual(result_conf, 0.75)

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    def create_evaluation_object_for_evaluate_image_tests(self, mock_file):
        """
        Create evaluation object for tests that need redundant
        versions of this code

        Returns: a set up evaluation object
        """
        mock_models = {
            "late": MagicMock(return_value=torch.tensor([[0.1, 0.3, 0.6]])),
            "dors": MagicMock(return_value=torch.tensor([[0.2, 0.5, 0.3]])),
            "fron": MagicMock(return_value=torch.tensor([[0.7, 0.2, 0.1]])),
            "caud": MagicMock(return_value=torch.tensor([[0.4, 0.4, 0.2]])),
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models)
        mock_file.assert_called_once_with("models/height_mock.txt", 'r', encoding='utf-8')

        #mock transform_input for dummy output
        mock_transform = MagicMock(return_value = torch.rand(1, 3, 224, 224))
        evaluation.transform_input = mock_transform

        return evaluation

if __name__ == "__main__":
    unittest.main()
