"""test_evaluation_method.py"""
import unittest
import sys
import os
from unittest.mock import patch, mock_open, call, MagicMock
from PIL import Image
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from genus_evaluation_method import GenusEvaluationMethod

class TestGenusEvaluationMethod(unittest.TestCase):
    """
    Test the evaluation method class methods
    """
    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"acanthoscelides"})
    def test_initializer(self, mock_json, mock_file):
        """test the initializer for proper setup"""
        #mock the models
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")

        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()
        self.assertEqual(evaluation.use_method, 1)
        #Change the weights to match the program's manually
        self.assertEqual(evaluation.weights, [0.25, 0.25, 0.25, 0.25])
        self.assertEqual(evaluation.trained_models, mock_models)
        self.assertEqual(evaluation.height, 224)
        self.assertEqual(evaluation.genus_idx_dict, {0:"acanthoscelides"})


    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"acanthoscelides"})
    def test_heaviest_is_best(self, mock_json, mock_file):
        """test heaviest is best for proper tracking of highest certainty"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()
        evaluation.genus_idx_dict = {6:"chinensis"}

        genus, conf = evaluation.heaviest_is_best([0.1, 0.3, 0.5, 0.4],[1, 4, 6, 3])
        self.assertEqual(genus, "chinensis")
        self.assertEqual(conf, 0.5)

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"acanthoscelides"})
    def test_weighted_eval(self, mock_json, mock_file):
        """test weighted eval for proper calculation"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 2, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()
        evaluation.genus_idx_dict = {2:"mimosae"}
        #must be changed if weights are adjusted in code
        given_weights = [0.25, 0.25, 0.25, 0.25]
        conf_scores = [0.8, 0.6, 0.9, 0.7]
        genus_predictions = [1, 2, 2, 3]

        prediction, score = evaluation.weighted_eval(conf_scores, genus_predictions)
        self.assertEqual(prediction, "mimosae")
        assert score == given_weights[1] * conf_scores[1] + given_weights[2] * conf_scores[2]

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("json.load", return_value = {"0":"acanthoscelides"})
    def test_transform_input(self, mock_json, mock_file):
        """test transform input for proper image transformation"""
        mock_models = {
            "late" : MagicMock(),
            "fron" : MagicMock(),
            "dors" : MagicMock(),
            "caud" : MagicMock()
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()
        evaluation.height = 224
        fake_input = Image.new("RGB", (224, 224))
        result = evaluation.transform_input(fake_input)

        assert result.shape == (1, 3, 224, 224)

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("torch.max", return_value=(None, torch.tensor([0])))
    @patch("torch.nn.functional.softmax", return_value=torch.tensor([[0.8, 0.1, 0.1]]))
    @patch("json.load", return_value = {"0":"acanthoscelides"})
    def test_evaluate_image_single_input(self, mock_json, mock_softmax, mock_max, mock_file):
        """test proper output with a single image entered"""
        mock_models = {
            "late": MagicMock(return_value=torch.tensor([[0.1, 0.3, 0.6]])),
            "dors": MagicMock(return_value=torch.tensor([[0.2, 0.5, 0.3]])),
            "fron": MagicMock(return_value=torch.tensor([[0.7, 0.2, 0.1]])),
            "caud": MagicMock(return_value=torch.tensor([[0.4, 0.4, 0.2]])),
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        #mock transform_input for dummy output
        mock_transform = MagicMock(return_value = torch.rand(1, 3, 224, 224))
        evaluation.transform_input = mock_transform

        result_genus, result_conf = evaluation.evaluate_image(late=Image.new("RGB", (224, 224)))

        self.assertEqual(result_genus, "acanthoscelides")
        self.assertEqual(round(result_conf, 2), 0.8)

        mock_transform.assert_called_once()
        mock_softmax.assert_called_once()
        mock_max.assert_called_once()

    @patch("builtins.open", new_callable=mock_open, read_data="224")
    @patch("torch.max", return_value=(None, torch.tensor([1])))
    @patch("torch.nn.functional.softmax", return_value=torch.tensor([[0.3, 0.6, 0.1]]))
    @patch("json.load", return_value = {"0":"acanthoscelides", "1":"callosobruchus",
                                        "2":"mimosestes", "3":"phaseoli"})
    def test_evaluate_image_multiple_input(self, mock_json, mock_softmax, mock_max, mock_file):
        """test proper output with multiple images entered"""
        mock_models = {
            "late": MagicMock(return_value=torch.tensor([[0.1, 0.3, 0.6]])),
            "dors": MagicMock(return_value=torch.tensor([[0.2, 0.5, 0.3]])),
            "fron": MagicMock(return_value=torch.tensor([[0.7, 0.2, 0.1]])),
            "caud": MagicMock(return_value=torch.tensor([[0.4, 0.4, 0.2]])),
        }

        evaluation = GenusEvaluationMethod("height_mock.txt", mock_models, 1, "json_mock.txt")
        mock_file.assert_has_calls([call("src/models/height_mock.txt", 'r', encoding='utf-8'),
                                    call("src/models/json_mock.txt", 'r', encoding='utf-8')],
                                    any_order = True)
        mock_json.assert_called_once()

        #mock transform_input for dummy output
        mock_transform = MagicMock(return_value = torch.rand(1, 3, 224, 224))
        evaluation.transform_input = mock_transform

        result_genus, result_conf = evaluation.evaluate_image(
            late=Image.new("RGB", (224, 224)),
            dors=Image.new("RGB", (224, 224)),
            fron=Image.new("RGB", (224, 224)),
            caud=Image.new("RGB", (224, 224)))

        self.assertEqual(result_genus, "callosobruchus")
        self.assertEqual(round(result_conf, 2), 0.6)

        self.assertEqual(mock_transform.call_count, 4)
        self.assertEqual(mock_max.call_count, 4)
        self.assertEqual(mock_softmax.call_count, 4)

if __name__ == "__main__":
    unittest.main()
