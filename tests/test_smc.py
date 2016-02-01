import unittest
from models import smc
import numpy as np
import logging
__author__ = 'Michal Kononenko'

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class TestSMC(unittest.TestCase):

    def test_smc(self):
        t1 = 2.5
        noise = smc.GaussianDistribution(standard_deviation=0.1)
        prior = np.ones(1000) / sum(np.ones(1000))
        model = smc.T1Model(t1, noise=noise)
        parameter_space = np.linspace(2, 3, 1000)

        simulator = smc.SequentialMonteCarlo(model, parameter_space, prior,
                                             number_of_iterations=5)

        weights = [weight for weight in simulator]

        self.assertAlmostEqual(np.max(weights[-1].weights), 1, delta=5)
        self.assertEqual(simulator.experimental_model.call_count,
                         simulator.number_of_iterations
                         )
