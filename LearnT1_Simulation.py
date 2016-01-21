"""
Created on Wed Jun 24 11:04:10 2015
Learn T1 Simulation 
T1 inversion recovery model defined in FindT1Model class 


@author: Kissan Mistry
"""

#imports and intializations
from __future__ import division

from T1Model import T1Model
from qinfer.distributions import UniformDistribution
#from qinfer.distributions import NormalDistribution
from qinfer.smc import SMCUpdater
from qinfer.resamplers import LiuWestResampler
import numpy as np
import matplotlib.pyplot as plt

model = T1Model()
prior = UniformDistribution([0, 1])
N_particles=10000
updater = SMCUpdater(model, N_particles, prior, resampler=LiuWestResampler(0.98),zero_weight_policy='reset')

#Set the value of T1 to Learn, pick 1 value from prior 
true_model=prior.sample()

performance_dtype = [
    ('expparams', 'float'),
    ('sim_outcome', 'float'),
    ('est_mean', 'float'),
]

trials=10

data = np.zeros((trials, 1), dtype=performance_dtype)

for idx_trials in xrange(trials):
    #Choose tau/experimental parameter 
    #choose tau=0 for first guess
    #expparams = np.array([0.0000001], dtype=model.expparams_dtype)
    expparams =model.particle_guess_heuristic(updater, 10000)
 
    #simulate outcomes- based on the true T1, and the chosen intial value 
    #will be replaced by actual data collection from NMR for Mz values
    sim_outcome=model.simulate_experiment(true_model,expparams)

    #Run SMC and update the posterior distribution
    updater.update(sim_outcome,expparams,check_for_resample=True)
    
   #store data
    data[idx_trials]['est_mean'] = updater.est_mean()
    data[idx_trials]['sim_outcome'] = sim_outcome
    data[idx_trials]['expparams'] = expparams
    
    #plotting particles and weights 
    particles = updater.particle_locations
    weights = updater.particle_weights
    fig = plt.figure()

    plt.axvline(updater.est_mean(), linestyle = '--', c = 'blue', linewidth = 2)
    plt.axvline(true_model, linestyle = '--', c = 'red', linewidth = 2)
##    plt.scatter(particles[:,0],np.zeros((N_particles,)),s = 50*(1+(weights-1/N_particles)*N_particles))
    plt.scatter(particles[:,0],weights,s = 50*(1+(weights-1/N_particles)*N_particles))

