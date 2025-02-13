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
    def testEvaluateImage(self):
        eval = EvaluationMethod()
        eval.use_method = 1


    def testHeaviestIsBest(self):
        eval = EvaluationMethod()
        fron_cert = 98.7
        dors_cert = 45.9
        late_cert = 99.1
        caud_cert = 79.1

        answer = eval.heaviest_is_best(fron_cert, dors_cert, late_cert, caud_cert)
        self.assertEqual()

    def testWeightedEval(self):
        eval = EvaluationMethod()
        fron_cert = 98.7
        dors_cert = 45.9
        late_cert = 99.1
        caud_cert = 79.1

        answer = eval.weighted_eval(fron_cert, dors_cert, late_cert, caud_cert)
        self.assertEqual()

    def testStackedEval(self):
        eval = EvaluationMethod()
        
        answer = eval.stacked_eval()
        self.assertEqual()

    def testTransformInput(self):
        pass