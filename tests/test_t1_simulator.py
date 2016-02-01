import unittest
import t1_simulator
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

__author__ = 'Michal Kononenko'


class DerandomizedTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)


class TestNoisyModel(DerandomizedTestCase):
    def setUp(self):
        self.t = np.linspace(0, 2, 1000)
        self.true_t1 = 1
        self.noise = norm(loc=0, scale=0.05)

    def test_noise(self):
        result = t1_simulator.noisy_model(self.t, self.true_t1, self.noise)
        self.assertIsNotNone(result)
