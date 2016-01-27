"""
Contains unit tests for :mod:`t1_model`
"""
from t1_model import T1Model
import unittest

__author__ = 'Michal Kononenko'


class TestFindT1Model(unittest.TestCase):

    def setUp(self):
        self.model = T1Model()

    def test_is_instance(self):
        """
        Trivial test that gives coverage something to do
        """
        self.assertIsInstance(self.model, T1Model)
