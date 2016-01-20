"""
Created on Mon Jul 13 09:59:44 2015
Find T1 Model
models the inversion recovery scheme 
has simulatable function 
 

@author: Kissan Mistry
"""
from __future__ import division
from qinfer.abstract_model import Model, Simulatable, DifferentiableModel
from qinfer import ScoreMixin
import numpy as np
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class find_T1_model(ScoreMixin,DifferentiableModel):
    
     # We need to specify how many model parameters this model has.
    @property
    def n_modelparams(self):
        return 1 # T1 is only model parameter 
        #should learn the SNR or variance later 
    
    # The number of outcomes is always 2, so we indicate that it's constant
    # and define the n_outcomes method to return that constant.
    @property
    def is_n_outcomes_constant(self):
        return True
        
    def n_outcomes(self, expparams):
        #define how outcomes 
        return 2
        
    # Next, we denote that the experiment parameters are represented by a
    # single field of 1 floats.
    #CHECK THIS FOR VALIDITY
    @property
    def expparams_dtype(self):
        return 'float'
  
    def are_models_valid(self, modelparams):
        return np.all(np.logical_and(modelparams > 0, modelparams <= 1), axis=1)
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
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
        super(find_T1_model, self).likelihood(outcomes, modelparams, expparams)
     
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
        
        
        
        
        
        
        
        
        