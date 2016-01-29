"""
Full implementation ready to go
"""
import numpy as np
from scipy import stats
from abc import ABCMeta, abstractmethod, abstractproperty

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


class AbstractNoisyModel(object):
    """
    Contains the base definition of a model, with built-in noise function
    """
    __metaclass__ = ABCMeta

    def __init__(self, noise_distribution=NullDistribution(), *args, **kwargs):
        self.noise = noise_distribution

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
                noise_distribution=NullDistribution(), *self._args,
                **self._kwargs
        )


class T1Model(AbstractNoisyModel):
    """
    Contains the T1 model
    """
    def __init__(self, t1, minimum_polarization=-1, maximum_polarization=1,
                 number_of_discretization_points=1000):
        super(AbstractNoisyModel, self).__init__()

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

    def __call__(self, t1_candidates):
        self.call_count += 1
        return -2 * (np.exp(-t1_candidates/self.t1)) + \
            np.ones(len(t1_candidates))


class SequentialMonteCarlo(object):
    """
    Runs the SMC in an iterator
    """
    def __init__(self, model, parameter_space, weights,
                 number_of_iterations=100, _iteration=0):

        self.experimental_model = model
        self.theoretical_model = model.theoretical_model

        self.parameter_space = parameter_space

        self.weights = weights

        self.number_of_iterations = number_of_iterations

        self._iteration = _iteration

    @property
    def mean_tau(self):
        return sum(self.parameter_space * self.weights)

    @property
    def mean_t1(self):
        return self.mean_tau / np.log(2)

    @property
    def measured_polarization(self):
        return self.experimental_model(self.mean_tau)

    @staticmethod
    def _sampling_distribution(mean, stdev):
        return stats.norm(loc=mean, scale=stdev)

    def next(self):
        if self._iteration > self.number_of_iterations:
            raise StopIteration

        new_iteration = self.__class__(
            self.experimental_model, self.parameter_space, self.weights,
            number_of_iterations=self.number_of_iterations,
            _iteration=(self._iteration + 1)
        )

        for index in range(len(self.parameter_space)):
            new_iteration.weights[index] = self._sampling_distribution(
                self.theoretical_model(new_iteration.weights[index]),
                self.experimental_model.noise.std
            ).pdf(self.measured_polarization) * self.weights[
                index]

        yield new_iteration

    def __len__(self):
        return self.number_of_iterations

    def __iter__(self):
        return self