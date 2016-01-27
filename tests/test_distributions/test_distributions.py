"""
Contains unit tests for :mod:`distributions.distributions`
"""
import distributions as dist
from tests import DeterministicTestCase
import mock
import numpy as np

__author__ = 'Michal Kononenko'


class TestNormalDistribution(DeterministicTestCase):
    pass


class TestNormalDistributionInit(TestNormalDistribution):
    def setUp(self):
        TestNormalDistribution.setUp(self)
        self.default_mean_argument = 0
        self.default_stdev_argument = 1

    def test_init_default_args(self):
        norm = dist.NormalDistribution()

        self.assertEqual(norm._mean, self.default_mean_argument)
        self.assertEqual(norm._standard_deviation, self.default_stdev_argument)

    def test_init_non_default_args(self):
        mean = 1
        standard_deviation = 3

        norm = dist.NormalDistribution(mean, standard_deviation)

        self.assertEqual(norm._mean, mean)
        self.assertEqual(norm._standard_deviation, standard_deviation)


class TestNormalDistributionWithObject(TestNormalDistribution):
    def setUp(self):
        TestNormalDistribution.setUp(self)
        self.mean = 0
        self.standard_deviation = 1
        self.norm = dist.NormalDistribution(self.mean, self.standard_deviation)


class TestNormalDistributionSample(TestNormalDistributionWithObject):
    def setUp(self):
        TestNormalDistributionWithObject.setUp(self)
        self.expected_rvs_return_value = np.array([1])
        self.norm.distribution.rvs = mock.MagicMock(
            return_value=self.expected_rvs_return_value
        )

        self.default_number_of_samples = 1

    def test_sample_default_arg(self):
        expected_call = mock.call(self.default_number_of_samples)

        samples = self.norm.sample()

        self.assertEqual(samples, self.expected_rvs_return_value)
        self.assertEqual(
            expected_call,
            self.norm.distribution.rvs.call_args
        )
