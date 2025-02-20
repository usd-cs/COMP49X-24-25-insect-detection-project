"""test_evaluation_method.py"""
import unittest
import sys
import os
import torch


from evaluation_method import EvaluationMethod

class TestEvaluationMethod(unittest.TestCase):
    """
    Test the evaluation method class methods
    """

    """def testInitializer(self):
    
        test initializer of the class
        INTEGRATION TEST ONLY
        
        eval = EvaluationMethod("height_mock.txt")
        
        """
    """def testEvaluateImage(self):
        
        test evaluate image method for proper execution
        INTEGRATION TEST ONLY
        
        eval = EvaluationMethod("height_mock.txt")
        eval.use_method = 1
    """


    def testHeaviestIsBest(self):
        """test heaviest is best for proper tracking of highest certainty"""
        eval = EvaluationMethod("height_mock.txt")
        
        assert eval.heaviest_is_best(0.1, 0.3, 0.5, 0.4) == 2
        assert eval.heaviest_is_best(0.1, 0.9, 0.6, 0.32) == 1

    def testWeightedEval(self):
        """test weighted eval for proper calculation"""
        eval = EvaluationMethod("height_mock.txt")
        conf_scores = [0.8, 0.6, 0.9, 0.7]
        species_predictions = [1, 2, 2, 3]

        prediction, score = eval.weighted_eval(conf_scores, species_predictions)
        assert prediction == 2

    def testTransformInput(self):
        """test transform input for proper image transformation"""
        eval = EvaluationMethod("height_mock.txt")
        eval.height = 224
        fake_input = torch.rand(3, 224, 224)
        result = eval.transform_input(fake_input)

        assert result.shape == (1, 3, 224, 224)