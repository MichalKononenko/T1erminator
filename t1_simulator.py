import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

__author__ = 'Michal Kononenko'

TRUE_T1 = 2.5

# Polarization space
MINIMUM_POLARIZATION = -1
MAXIMUM_POLARIZATION = 1
NUMBER_OF_POLARIZATIONS = 1000

# Monte Carlo
MONTE_CARLO_ITERATIONS = 100

# Noise Processes
NOISE_STDEV = 0.1
NOISE_BIAS = 0  # no mean direction in which noise is observed
NOISE_DIST = norm(loc=NOISE_BIAS, scale=NOISE_STDEV)


def noisy_model(t, true_t1=TRUE_T1, noise=NOISE_DIST):
    """
    Returns a list corresponding to the evolution of the magnetization
    of a single spin undergoing an inversion recovery sequence. This model
    is the black box for the NMR model
    :param t:
    :param true_t1:
    :param noise:
    :return:
    """
    return -2 * (np.exp(-t/true_t1)) + 1 + noise.rvs(
            len(t) if hasattr(t, '__len__') else 1
    )


def noiseless_model(t, expected_t1):
    """
    Experimental model

    :param t:
    :param expected_t1:
    :return:
    """
    return -2 * (np.exp(-t/expected_t1)) + 1


def calculate_t1_from_polarization(tau, q):
    return tau / np.log(2 / (1 - q))

polarizations = np.linspace(
    MINIMUM_POLARIZATION, MAXIMUM_POLARIZATION,
    NUMBER_OF_POLARIZATIONS
)

measured_polarizations = np.zeros(MONTE_CARLO_ITERATIONS)
expected_t1_values = np.zeros(MONTE_CARLO_ITERATIONS)


T1_VALUES = np.linspace(2, 3, 100)
EXPECTED_TAU_VALUES = np.zeros([MONTE_CARLO_ITERATIONS, 1])
WEIGHTS = np.zeros([MONTE_CARLO_ITERATIONS, len(T1_VALUES)])

#  set prior distributions
PRIOR = np.ones(len(T1_VALUES)) / len(T1_VALUES)

WEIGHTS[1, :] = PRIOR

for index in range(MONTE_CARLO_ITERATIONS):
    EXPECTED_TAU_VALUES[index] = np.mean(T1_VALUES * WEIGHTS[(index-1), :])
    expected_t1_values[index] = EXPECTED_TAU_VALUES[index] / np.log(2)
    measured_polarizations[index] = noisy_model(expected_t1_values[index])

    for weight_index in range(len(T1_VALUES)):
        WEIGHTS[index, weight_index] = norm(
                loc=noiseless_model(
                        expected_t1_values[index], T1_VALUES[index]),
                scale=NOISE_STDEV).pdf(measured_polarizations[index])

    WEIGHTS[index, :] = WEIGHTS[index, :] / sum(WEIGHTS[index,:])

#  Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_wireframe(T1_VALUES, range(MONTE_CARLO_ITERATIONS), WEIGHTS)

fig.show()
pass