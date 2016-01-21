"""
Created on Mon Jul 13 09:59:44 2015
Find T1 Model
models the inversion recovery scheme 
has simulatable function 
"""
from __future__ import division
from qinfer.abstract_model import Simulatable, DifferentiableModel
from qinfer import ScoreMixin
import numpy as np
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class T1Model(ScoreMixin, DifferentiableModel):
    """
    Builds a model of the `inversion recovery sequence`_ within the context of
    Qinfer. This is used to simulate the sequence and determine T1.

    .. _inversion recovery sequence: http://goo.gl/51FoSZ
    """
    Equlibrium_Magnetization = 1

    @property
    def n_modelparams(self):
        """
        Overwrites the abstract property
        :attr:`qinfer.abstract_model.DifferentiableModel.n_modelparams`. Since
        T1 is the only model parameter, and this method returns the number of
        model parameters. Therefore, the method returns 1

        :return: The number of model parameters
        :rtype: int
        """
        return 1

    @property
    def is_n_outcomes_constant(self):
        """
        This property is required and tells
        :class:`qinfer.abstract_model.DifferentiableModel` whether to expect
        the outcomes of each experiment to be constant. Since this is ``True``,
        we simply have the function returning ``True``

        :return: True
        :rtype: bool
        """
        return True
        
    def n_outcomes(self, _):
        """
        Returns the number of outcomes that a given set of experiment
        parameters will produce. Since the number of experiment outcomes is
        constant, this function always returns ``2``. The empty parameter _
        is used as a placeholder, in order to preserve the signature of
        :meth:`qinfer.abstract_model.DifferentiableModel.n_outcomes`

        :param _: An empty parameter used as a black hole for the expparams
            list
        :type _: any
        :return: The expected number of outcomes (2)
        :rtype: int
        """
        return 2

    @property
    def expparams_dtype(self):
        """
        This method returns the data type of the experiment parameters. In
        our case, this is ``float``.

        :return: ``'float'``
        :rtype: str
        """
        return 'float'
  
    def are_models_valid(self, modelparams):
        """
        Required in :class:`qinfer.abstract_model.Simulatable` as a
        validator to check whether the model parameters are valid.

        :param numpy.ndarray modelparams: The model parameters to validate
        :return: True if the models are valid, otherwise false
        :rtype: bool
        """
        return np.all(
                np.logical_and(modelparams > 0, modelparams <= 1), axis=1
        )
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        """
        Simulate an experiment within the model

        :param numpy.ndarray modelparams: The model parameters for which the
            simulation is to be computed
        :param numpy.ndarray expparams: The experiment parameters. The
            likelihood function will be integrated over these parameters in
            order to determine the likelihood as a function of the model
            parameters.
        :param int repeat: The number of times that the experiment should be
            repeated. Defaults to 1
        :return: The samples corresponding to the results of the simulation
        :rtype: np.ndarray

        .. todo::

            The ``assert`` statement in the code is a simple sanity check to
            make sure that the samples have the desired shape. We should
            re-work this into
        """
        Simulatable.simulate_experiment(self, modelparams, expparams, repeat)
        mean, var = self._meanvar(modelparams, expparams)
        log.debug('modelparams: %s' % modelparams)
        desired_shape = (repeat, modelparams.shape[0], expparams.shape[0])
        samples = np.random.randn(*desired_shape)

        samples = samples * np.sqrt(var) + mean

        assert samples.shape == desired_shape

        return samples
    
    def _meanvar(self, modelparams, expparams):
        """
        Calculates the mean and variance of the model

        :param modelparams: The model parameters
        :param expparams: The experiment parameters
        :return: The mean and variance of the model
        :rtype: tuple
        """
#        modelparams = modelparams[..., np.newaxis] #this is the devil's line 
#        var = 0.05 #SNR LOOK AT THIS VALUE
        var = 0.0005
        z_magnetization = self.Equlibrium_Magnetization * (
            1 - 2 * np.exp(-1 * expparams / modelparams)
        )
        mean = z_magnetization
        return mean, var     

    def likelihood(self, outcomes, modelparams, expparams):
        """
        Overwrites the likelihood function in
        :class:`qinfer.abstract_model.Simulatable.likelihood`, returns the
        likelihood of an outcome given a set of model parameters and
        experiment parameters.

        .. note::

            The superclass method is called in this function in order to make
            sure that the call count diagnostics are properly logged.

        :param outcomes:
        :param modelparams:
        :param expparams:
        :return:
        """
        super(T1Model, self).likelihood(outcomes, modelparams, expparams)
     
        mean, var = self._meanvar(modelparams, expparams)

        scale_factor = 1/np.sqrt(2*np.pi*var)
        exponential = np.exp(-((mean-outcomes) ** 2)/(2 * var))

        probability = scale_factor * exponential

        norm = np.max(probability)

        return probability/norm

# Custom particle guess heuristic, different from qinfer.PGH
    def particle_guess_heuristic(self, updater, maxiters):
        """
        In order to work with this model, a custom particle guess heuristic
        needs to be written that is different from :`qinfer.PGH`

        :param updater:
        :param maxiters:
        :return:
        """
        idy_iter = 0
        while idy_iter < maxiters:
            idx_iter = 0
            while idx_iter < maxiters:
                a = updater.sample()
#                while a>updater.est_mean():
#                    a=updater.sample()
                
                b = updater.sample()
#                while b<updater.est_mean():
#                    b=updater.sample()
                if (a<updater.est_mean() and b>updater.est_mean())  or (a>updater.est_mean() and b<updater.est_mean()):
                     break    
                 
#                if np.abs(a-b) > 0: #TODO: is this acceptable?
#                    break
                else:
                    idx_iter += 1
                if np.abs(a-b) == 0:
                    raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
                
#took analytical derivative of  difference of model evaluated at the sampled points Mz(a)-Mz(b)
## maximize the differece and use second derivative test to check if it is a local maxima. 
#            t = np.log(b/a)/((1/a)-(1/b)) #first derivative optimization(set d/dx=0, solve for tau)
#            print 'a='+str(a)
#            print 'b='+str(b)
#            check = (1/(a**2))*np.exp(-t/a)-(1/(b**2))*np.exp(-t/b) #2nd derivative test
#            if check <0:
#                break
#            else:
#                idy_iter+=1
    
#                
                
# this method is not good 
#            t=[1/np.linalg.norm((a-b),1,1)]
#            print str(t)
#            break
#       return t
       
#define cost function 
    def experiment_cost(self,expparams):
        #define resitrictions on tau values
        #possitve restriction
        if expparams<=0:
#            cost=1
            cost=100  # use this one to avoid picking negative values
        else:
            cost=1
   
        return cost
