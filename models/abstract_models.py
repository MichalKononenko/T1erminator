"""
Contains abstractions for simulations
"""
from abc import ABCMeta, abstractmethod
__author__ = 'Michal Kononenko'

__all__ = ['AbstractSystemModel', 'AbstractNoiseDistribution']


class AbstractSystemModel(object):
    """
    Base class for all models in the simulations
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate_model(self, t, scaling_parameters, system_parameters):
        """
        Evaluate the model for a given set of times, scaling parameters,
        and system parameters

        :param np.array t: array of times of length N
        :param numpy.ndarray scaling_parameters: An array of length
            :math:`k \cross m`
            where k is the number of different sets of scaling parameters to
            be evaluated, and each column of length m are the scaling
            parameters.
        :param numpy.ndarray system_parameters: An array of length kxn where
            k is the number of different sets of system parameters to be
            evaluated, and each column of length n are the system parameters.

        :return An array of evaluate models for each time point of dimensions
            Nxl where l depends on if there are multiple outputs for a model
            (ie. two channels on an NMR spectrometer)
        :rtype np.ndarray
        """
        raise NotImplementedError


class AbstractNoiseModel(object):
    """
    Contains models for adding noise to data
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def add_noise(self, data):
        raise NotImplementedError


class AbstractNoiseDistribution(object):

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
