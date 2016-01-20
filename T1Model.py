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
    Builds a model of the `inversion recovery sequence`_


    .. _inversion recovery sequence:
    """

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

        :param modelparams:
        :param expparams:
        :param repeat:
        :return:
        """
    #simulate experiment and generate outcomes just adds guassian noise to the outcome of the model equation
        Simulatable.simulate_experiment(self, modelparams, expparams, repeat)
        mean, var = self._meanvar(modelparams, expparams)
#        print modelparams
        desired_shape = (repeat, modelparams.shape[0], expparams.shape[0])
        samples = np.random.randn(*desired_shape)

        samples = samples * np.sqrt(var) + mean

        assert samples.shape == desired_shape
        
        return samples
 
#define element in class called Mo that is changeable when called from outside
#used for normalization of the model and changes when called from outside for real nmr exp.          
    Mo=1
    
    def _meanvar(self, modelparams, expparams):
#        modelparams = modelparams[..., np.newaxis] #this is the devil's line 
#        var = 0.05 #SNR LOOK AT THIS VALUE
        var=0.0005
        Mz=self.Mo*(1-2*np.exp(-1*expparams/modelparams)) #the model
        mean=Mz
        return mean, var     
        
    #Define Likelihood function    
    def likelihood(self, outcomes, modelparams, expparams):
        # Call the superclass method, which basically
        # just makes sure that call count diagnostics are properly
        # logged.
        super(T1Model, self).likelihood(outcomes, modelparams, expparams)
     
        mean, var = self._meanvar(modelparams, expparams)
        var=0.05
        pr= (1/np.sqrt(2*np.pi*var))*np.exp(-((mean - outcomes) ** 2)/(2* var))
        norm=np.max(pr)
        pr=pr/norm
  
        return pr
        
# Custom particle guess heuristic, different from qinfer.PGH    
    def pgh(self,updater,maxiters):
        #draw two random T1 values from posterior/prior(if first) distribution
       idy_iter=0
       while idy_iter<maxiters: 
            idx_iter = 0
            while idx_iter < maxiters:
                a=updater.sample()
#                while a>updater.est_mean():
#                    a=updater.sample()
                
                b=updater.sample()
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
            break   
       return t
       
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
        
        
        
        
        
        
        
        
        