"""
Contains blueprints for distributions that inherit from the abstract
distribution
"""
from abstract_distribution import AbstractDistribution
from scipy import stats

__author__ = 'Michal Kononenko'


class NormalDistribution(AbstractDistribution):

    def __init__(self, mean=0, standard_deviation=1):
        self._mean = mean
        self._standard_deviation = standard_deviation
        self.distribution = stats.norm(loc=mean, scale=standard_deviation)

    def sample(self, number_of_samples=1):
        return self.distribution.rvs(number_of_samples)

    @property
    def mean(self):
        return self._mean


class UniformDistribution(AbstractDistribution):

    def __init__(self, lower_bound=0, upper_bound=1):
        loc = lower_bound
        scale = upper_bound - loc
        self.distribution = stats.uniform(loc=loc, scale=scale)

    def sample(self, number_of_samples=1):
        return self.distribution.rvs(number_of_samples)

    @property
    def mean(self):
        return self.distribution.mean()
