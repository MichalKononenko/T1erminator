"""
Created on Wed Jun 24 11:04:10 2015
Learn T1 NMR experiement run on TOPSPIN 
T1 inversion recovery model defined in FindT1Model class 

includes calls to run TOPSPIN commands- NMR experiment 

@author: Kissan Mistry 
"""

#imports and intializations
from __future__ import division
from FindT1Model import FindT1Model
from qinfer.distributions import UniformDistribution
#from qinfer.distributions import NormalDistribution
from qinfer.smc import SMCUpdater
from qinfer.resamplers import LiuWestResampler
import numpy as np
import matplotlib.pyplot as plt

import time
import Lorentzianfit as LF
from qinfer.expdesign import ExperimentDesigner
import os
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

model = FindT1Model()
prior = UniformDistribution([0, 100])
N_particles=100000
updater = SMCUpdater(model, N_particles, prior, resampler=LiuWestResampler(0.98),zero_weight_policy='reset')
designer=ExperimentDesigner(updater,opt_algo=1)

#Set the value of T1 to Learn, pick 1 value from prior 
#true_model=prior.sample()
true_model=np.array([6.8], dtype=model.expparams_dtype)

performance_dtype = [
    ('expparams', 'float'),
    ('sim_outcome', 'float'),
    ('est_mean', 'float'),
    ('covariance', 'float'),
    
]

#NMR EXPERIMENT Initialization*******************************
#going to normalize Mo max of 1. 
#model.Mo=float(raw_input('Please enter Mo: '))
#dummy=float(raw_input('Waiting for Mo: '))
#Mo_norm=LF.lorentzfit('1_spectrum.txt')
#model.Mo=(Mo_norm/Mo_norm)



#STORE DATA   
timestr = time.strftime("%Y%m%d-%H%M%S")
Savestring= r'C:/Users/Kissan/Desktop/' + timestr +'/'
if not os.path.exists(Savestring): 
    os.makedirs(Savestring)
save_exp=open(Savestring+'_exp.txt','w')
save_out=open(Savestring+'_out.txt','w')
save_mean=open(Savestring+'_mean.txt','w')
save_cov=open(Savestring+'_cov.txt','w')
save_var=open(Savestring+'_var.txt','w')

#iterative process to find T1 
trials=5
data = np.zeros((trials, 1), dtype=performance_dtype)
for idx_trials in xrange(trials):
    log.debug('trial: ' + str(idx_trials))
    
#CHOOSE EXPERIMENTAL PARAMETER****************************    
    guess_iter=10000
    guess_vec=np.zeros((guess_iter,1))
    risk_vec=np.zeros((guess_iter,1))
    
    designer.new_exp()
    store_risk=1000000000000000
    for idx_guess in xrange(guess_iter):
#        print 'guess iteration: '+ str(idx_guess)
        guess=np.array([[[0.01+(0.01*idx_guess)]]],dtype=model.expparams_dtype) #sweep guess/incremental increase 
#        guess=np.array([model.pgh(updater,10000)],dtype=model.expparams_dtype ) #generate guess from PGH
#        print 'Your Guess is: '+ str(guess)
        #evaluate bayes risk for the guess
        current_risk=updater.bayes_risk(guess)
#        print 'bayes_risk: ' + str(current_risk)
        if current_risk<store_risk:
            store_risk=current_risk
            expparams=guess
        risk_vec[idx_guess]=current_risk
        guess_vec[idx_guess]=guess
    log.debug('Your Tau is: ' + str(expparams))
        
        #optimize that guess
#        expparams=designer.design_expparams_field(guess,0,cost_scale_k=1,disp=False,maxiter=10000,maxfun=10000,store_guess=True,grad_h=1,)
#        print 'Your Tau is: ' + str(expparams)
    fig = plt.figure()
    plt.scatter(guess_vec,risk_vec,s=1)
    plt.title('Bayes Risk of Guesses, Best Guess= '+str(expparams))
    plt.ylabel('Bayes Risk')
    plt.xlabel(r'$\tau$'+' Guess')
    plt.savefig(Savestring+'bayes risk-'+str(idx_trials)+'.png')
        

#   
#SIMULATE*******************************************************
    #simulate outcomes- based on the true T1, and the chosen intial value 
    #will be replaced by actual data collection from NMR for Mz values
    sim_outcome=model.simulate_experiment(true_model,expparams)
    outcome=sim_outcome
    
    
#NMR EXPERIMENT*************************************************    
#USE this instead of simualate when doing experiments in NMR 
#    outcome=np.array([[[float(raw_input('Enter obtained Mz: '))]]])
#    dummy=float(raw_input('waiting for Mz'))
#    Mz_value=LF.lorentzfit(str(idx_trials+2)+'_spectrum.txt')
#    outcome=np.array([[[Mz_value/abs(Mo_norm)]]])

    #Run SMC and update the posterior distribution
    updater.update(outcome,expparams,check_for_resample=True)
 
 
 
#STORE DATA******************************************
    data[idx_trials]['est_mean'] = updater.est_mean()
    data[idx_trials]['sim_outcome'] = outcome
    data[idx_trials]['expparams'] = expparams
    data[idx_trials]['covariance'] = updater.est_covariance_mtx()
    save_exp.writelines(str(expparams)+'\n')
    save_mean.write(str(updater.est_mean())+'\n')
    save_out.write(str(outcome)+'\n')
    save_cov.write(str(updater.est_covariance_mtx())+'\n')
    

# PLOT *******************************************  
#plotting particles and weights 
    particles = updater.particle_locations
    weights = updater.particle_weights
    
    a=np.var(np.multiply(particles[:,0],weights))
    save_var.write(str(a)+'\n')
    
    if idx_trials==0:
        maxw=max(weights)
    weights=weights/maxw #normalize the posterior 

    fig1 = plt.figure()

    plt.axvline(updater.est_mean(), linestyle = '--', c = 'blue', linewidth =2,label='Est. Mean')
#    plt.axvline(true_model, linestyle = '--', c = 'red', linewidth = 2,label='True Model')
    plt.scatter(particles,weights,s=0.1)
    plt.title('Posterior Distribution T1= '+str(updater.est_mean()))
    plt.ylabel('Normalized Weight')
    plt.xlabel('Particles')
    plt.legend()
    plt.savefig(Savestring+'T1 posterior-'+str(idx_trials)+'.png')
#END LOOP***************************************************

save_exp.close()
save_mean.close()
save_out.close()
save_cov.close()
save_var.close()
