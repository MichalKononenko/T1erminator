"""
Contains Unit tests
"""
import unittest
import numpy as np
__author__ = 'Michal Kononenko'


class DeterministicTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
