"""
Contains abstract distributions
"""
import abc
__author__ = 'Michal Kononenko'


class AbstractDistribution(object):
    """
    Contains definitions for a distribution with a niladic sampling function
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def mean(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, number_of_samples=1):
        """
        :param number_of_samples:
        :return:
        """
        raise NotImplementedError

    def __repr__(self):
        return '%s' % self.__class__.__name__
