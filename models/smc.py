"""
Full implementation ready to go
"""
import numpy as np
from scipy import stats
from abc import ABCMeta, abstractmethod

__author__ = 'Michal Kononenko'


class AbstractDistribution(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self, list_to_sample):
        """
        Sample the noise distribution in order to obtain a list of random
        variables corresponding to the noise

        :param array-like or int list_to_sample:
        :return:
        """
        raise NotImplementedError


class NullDistribution(AbstractDistribution):

    def sample(self, list_to_sample):
        return np.zeros(
                len(list_to_sample)
                if hasattr(list_to_sample, '__len__') else 1
        )


class GaussianDistribution(AbstractDistribution):

    def __init__(self, mean=0, standard_deviation=1):
        self.distribution = stats.norm(
            loc=mean, scale=standard_deviation
        )

    def sample(self, list_to_sample):
        return self.distribution.rvs(
            len(list_to_sample) if hasattr(list_to_sample, '__len__') else 1
        )

    @property
    def mean(self):
        return self.distribution.mean()

    @property
    def std(self):
        return self.distribution.std()


class AbstractNoisyModel(object):
    """
    Contains the base definition of a model, with built-in noise function
    """
    __metaclass__ = ABCMeta

    def __init__(self, noise=NullDistribution(), *args, **kwargs):
        self.noise = noise

        self._args = args
        self._kwargs = kwargs

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def theoretical_model(self):
        """
        Should return a version of this model with a noiseless distribution
        """
        return self.__class__(
                noise=NullDistribution(), *self._args,
                **self._kwargs
        )


class T1Model(AbstractNoisyModel):
    """
    Contains the T1 model
    """
    def __init__(self, t1, noise=NullDistribution(),
                 minimum_polarization=-1, maximum_polarization=1,
                 number_of_discretization_points=1000):
        super(T1Model, self).__init__(
                noise=noise)

        self.t1 = t1
        self.polarizations = np.linspace(
            minimum_polarization, maximum_polarization,
            number_of_discretization_points
        )

        self.call_count = 0

    @property
    def minimum_polarization(self):
        return self.polarizations[0]

    @property
    def maximum_polarization(self):
        return self.polarizations[-1]

    @property
    def theoretical_model(self):
        return self.__class__(self.t1, noise=NullDistribution())

    def __call__(self, t1_candidates):
        self.call_count += 1
        return -2 * (np.exp(-t1_candidates/self.t1)) + \
            np.ones(
                len(t1_candidates)
                if hasattr(t1_candidates, '__len__') else 1
            )


class SequentialMonteCarlo(object):
    """
    Runs the SMC in an iterator
    """
    def __init__(self, model, parameter_space, weights,
                 number_of_iterations=100):

        self.experimental_model = model
        self.theoretical_model = model.theoretical_model

        self.parameter_space = parameter_space

        self.weights = weights

        self.number_of_iterations = number_of_iterations

        self._iteration = 0

    @property
    def mean_tau(self):
        return sum(self.parameter_space * self.weights)

    @property
    def mean_t1(self):
        return self.mean_tau / np.log(2)

    def measure(self):
        return self.experimental_model(self.mean_t1)

    @staticmethod
    def _sampling_distribution(mean, stdev):
        return stats.norm(loc=mean, scale=stdev)

    def __next__(self):
        if self._iteration >= self.number_of_iterations:
            raise StopIteration()

        self._iteration += 1

        test_value = self.measure()

        for index in range(len(self.parameter_space)):
            weight_to_add = self._sampling_distribution(
                self.mean_t1,
                self.experimental_model.noise.std
            ).pdf(test_value) * self.weights[
                index]

            self.weights[index] = weight_to_add

        self.weights = self.weights / sum(self.weights)

        return self

    def __len__(self):
        return self.number_of_iterations

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()
