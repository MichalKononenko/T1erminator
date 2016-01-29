"""
Contains models for a single spin
"""
import numpy as np
from abstract_models import AbstractNoiseDistribution
from scipy import stats
__author__ = 'Michal Kononenko'


class GaussianNoiseDistribution(AbstractNoiseDistribution):

    def __init__(self, mean=0, standard_dev=1):
        self.distribution = stats.norm(loc=mean, scale=standard_dev)

    def sample(self, list_to_sample):
        return self.distribution.rvs(
                len(list_to_sample) if hasattr(list_to_sample, '__len__')
                else 1
        )


class _NullNoiseDistribution(AbstractNoiseDistribution):
    def sample(self, list_to_sample):
        return np.zeros(
            len(list_to_sample) if hasattr(list_to_sample, '__len__')
            else 1
        )


class SingleSpinModel(object):
    """
    Model for a single spin in an inversion recovery sequence.
    This is meant to work as a black box, with components being replaced
    by NMR's TOPSPIN programs. Constructor takes in

    :var float t1: The actual T1 relaxation time of the model. This
        is what the bayesian learning system should converge to.
    :var int minimum_polarization: The minimum value of the magnetization
        vector, defaults to -1.
    :var int maximum_polarization: The maximum value of the polarization
        vector, defaults to +1
    :var int number_of_discretization_points: The number of points in
        between the minimum and maximum polarization, which is used to
        discretize the polarizations
    """
    def __init__(self, t1,
                 minimum_polarization=-1, maximum_polarization=1,
                 number_of_discretization_points=1000):
        """
        Instantiates the variables mentioned above
        """
        self._true_t1 = t1
        self.polarizations = np.linspace(
                minimum_polarization, maximum_polarization,
                number_of_discretization_points
        )
        self._call_count = 0

    @property
    def minimum_polarization(self):
        """
        Returns the minimum polarization
        """
        return self.polarizations[0]

    @property
    def maximum_polarization(self):
        return self.polarizations[-1]

    @property
    def call_count(self):
        """
        Returns the number of times that :meth:`SingleSpinModel.evaluate`
        has been called
        """
        return self._call_count

    def evaluate(self, t1_candidates):
        """
        Evaluate this model
        :param np.ndarray or int t1_candidates: Times corresponding to a
            probable T1
        :return:
        """
        self._call_count += 1
        return - 2 * (np.exp(-t1_candidates/self._true_t1)) + 1


class NoisySingleSpinModel(SingleSpinModel):
    """
    Single spin model that incorporates noise

    :var t1: The T1 relaxation time of this system.
    :var AbstractNoiseDistribution noise_distribution: The noise
        distribution that will be sampled in order to obtain the noise
        process to be added into this model
    """
    def __init__(self, t1, noise_distribution=_NullNoiseDistribution(), *args,
                 **kwargs):
        SingleSpinModel.__init__(self, t1, *args, **kwargs)
        if not isinstance(noise_distribution, AbstractNoiseDistribution):
            raise ValueError(
                    'The noise distribution %s is not an instance of '
                    'AbstractNoiseDistribution', noise_distribution
            )
        self.noise = noise_distribution

    def evaluate(self, t1_candidates):
        """
        Evaluate with the noise process added in
        :param list-like t1_candidates: The
        :return:
        """
        data = -2 * (np.exp(-t1_candidates/self._true_t1)) + \
               np.ones(len(t1_candidates))
        noise = self.noise.sample(t1_candidates)

        if len(noise) != len(data):
            raise Exception(
                'The result from the sample method of %s ''s noise '
                'distribution %s returned noise of length %d. The length'
                'of data to be returned is %d', self, self.noise,
                len(noise), len(data)
            )

        return data + noise
