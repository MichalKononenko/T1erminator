"""
Contains unit tests for :mod:`find_T1_model`
"""
from find_T1_model import find_T1_model
import unittest
import mock

__author__ = 'Michal Kononenko'


class test_find_T1_model(unittest.TestCase):

    def setUp(self):
        self.model = find_T1_model()

    def test_is_instance(self):
        """
        Trivial test that gives coverage something to do
        """
        self.assertIsInstance(self.model, find_T1_model)
