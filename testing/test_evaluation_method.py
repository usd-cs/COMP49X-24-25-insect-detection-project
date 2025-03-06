"""test_evaluation_method.py"""
import unittest
import sys
import os
from unittest.mock import patch, mock_open, call, MagicMock
from PIL import Image
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from evaluation_method import EvaluationMethod

class TestEvaluationMethod(unittest.TestCase):
    """
    Test the evaluation method class methods
    """
    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"objectus"})
    def test_initializer(self, mock_json, mock_file):
        """test the initializer for proper setup"""
        # Mock the models
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")

        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        self.assertEqual(evaluation.use_method, 1)
        # Change the weights to match the program's manually
        self.assertEqual(evaluation.weights, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(evaluation.trained_models, mock_models)
        self.assertEqual(evaluation.height, 224)
        self.assertEqual(evaluation.species_idx_dict, {0:"objectus"})


    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {
        "0":"objectus", "1":"analis", "2":"maculatus", "3":"phaseoli", "4":"nubigens"})
    def test_heaviest_is_best(self, mock_json, mock_file):
        """test heaviest is best for proper tracking of highest certainty"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        test_conf_scores = [0.3, 0.6, 0.1, 0.4, 0.5]
        test_species = [1, 4, 2, 3, 0]
        # Run heaviest_is_best method with test scores and species
        test_results = evaluation.heaviest_is_best(
            [test_conf_scores, test_conf_scores, test_conf_scores, test_conf_scores],
            [test_species, test_species, test_species, test_species])
        # Assert top species is as expected
        self.assertEqual(test_results[0][0], "nubigens")
        self.assertEqual(round(test_results[0][1], 2), 0.6)

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {
        "0":"objectus", "1":"analis", "2":"maculatus", "3":"phaseoli", "4":"nubigens"})
    def test_weighted_eval(self, mock_json, mock_file):
        """test weighted eval for proper calculation"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models, 2, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        test_conf_scores = [0.3, 0.6, 0.1, 0.4, 0.5]
        test_species = [1, 4, 2, 3, 0]
        # Run weighted_eval with test scores and species
        test_results = evaluation.weighted_eval(
            [test_conf_scores, test_conf_scores, test_conf_scores, test_conf_scores],
            [test_species, test_species, test_species, test_species])
        # Assert top species is as expected
        self.assertEqual(test_results[0][0], "nubigens")
        self.assertEqual(round(test_results[0][1], 2),
                         (evaluation.weights[0] * test_conf_scores[1] +
                          evaluation.weights[1] * test_conf_scores[1] +
                          evaluation.weights[2] * test_conf_scores[1] +
                          evaluation.weights[3] * test_conf_scores[1]))

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"objectus"})
    def test_transform_input(self, mock_json, mock_file):
        """test transform input for proper image transformation"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        evaluation.height = 224
        fake_input = Image.new("RGB", (224, 224))
        result = evaluation.transform_input(fake_input)

        assert result.shape == (1, 3, 224, 224)

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("torch.topk", return_value=(
        torch.tensor([0.6, 0.5, 0.4, 0.3, 0.1]), torch.tensor([1, 4, 3, 0, 2])))
    @patch("torch.nn.functional.softmax", return_value=torch.tensor([[0.3, 0.6, 0.1, 0.4, 0.5]]))
    @patch("json.load", return_value = {
        "0":"objectus", "1":"analis", "2":"maculatus", "3":"phaseoli", "4":"nubigens"})
    def test_evaluate_image(self, mock_json, mock_softmax, mock_topk, mock_file):
        """test proper output with multiple images entered"""
        mock_models = {
            "late": MagicMock(),
            "dors": MagicMock(),
            "fron": MagicMock(),
            "caud": MagicMock(),
        }

        evaluation = EvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        # Mock transform_input for dummy output
        mock_transform = MagicMock(return_value = torch.rand(1, 3, 224, 224))
        evaluation.transform_input = mock_transform

        # Run the evaluate image with test images
        test_results = evaluation.evaluate_image(
            late=Image.new("RGB", (224, 224)),
            dors=Image.new("RGB", (224, 224)),
            fron=Image.new("RGB", (224, 224)),
            caud=Image.new("RGB", (224, 224)))

        # Assert top 5 species are in correct order of confidence with correct species
        self.assertEqual(test_results[0][0], "analis")
        self.assertEqual(round(test_results[0][1], 2), 0.6)

        self.assertEqual(test_results[1][0], "nubigens")
        self.assertEqual(round(test_results[1][1], 2), 0.5)

        self.assertEqual(test_results[2][0], "phaseoli")
        self.assertEqual(round(test_results[2][1], 2), 0.4)

        self.assertEqual(test_results[3][0], "objectus")
        self.assertEqual(round(test_results[3][1], 2), 0.3)

        self.assertEqual(test_results[4][0], "maculatus")
        self.assertEqual(round(test_results[4][1], 2), 0.1)

        # Make sure that the evaluate_image method behaved as expected
        self.assertEqual(mock_transform.call_count, 4)
        self.assertEqual(mock_topk.call_count, 4)
        self.assertEqual(mock_softmax.call_count, 4)

if __name__ == "__main__":
    unittest.main()
