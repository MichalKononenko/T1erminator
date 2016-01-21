"""
Created on Wed Jun 24 11:04:10 2015
Learn T1 NMR experiement run on TOPSPIN 
T1 inversion recovery model defined in find_T1_model class

includes calls to run TOPSPIN commands- NMR experiment 


@author: Kissan Mistry 
"""
from __future__ import division
from t1_model import T1Model
from qinfer.distributions import UniformDistribution
from qinfer.smc import SMCUpdater
from qinfer.resamplers import LiuWestResampler
import numpy as np
import matplotlib.pyplot as plt
from qinfer.expdesign import ExperimentDesigner
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

model = T1Model()
prior = UniformDistribution(np.array([0, 10]))
N_particles = 1000000
updater = SMCUpdater(
        model, N_particles, prior, resampler=LiuWestResampler(),
        zero_weight_policy='reset'
)
designer = ExperimentDesigner(updater, opt_algo=1)

# Set the value of T1 to Learn, pick 1 value from prior
true_model = prior.sample()
# true_model=np.array([11.032], dtype=model.expparams_dtype)

performance_dtype = [
    ('expparams', 'float'),
    ('sim_outcome', 'float'),
    ('est_mean', 'float'),
]

# NMR EXPERIMENT Initialization*******************************
# going to normalize Mo max of 1.
# model.Mo=float(raw_input('Please enter Mo: '))
# dummy=float(raw_input('Waiting for Mo: '))
# Mo_norm=LF.lorentzfit('1_spectrum.txt')
# model.Mo=(Mo_norm/Mo_norm)


# iterative process to find T1
trials = 5
data = np.zeros((trials, 1), dtype=performance_dtype)
for idx_trials in xrange(trials):
    log.info('trial: ' + str(idx_trials))
    # Choose tau/experimental parameter
    # choose tau=0 for first guess
#    if idx_trials==0:
#        expparams = np.array([0.0001], dtype=model.expparams_dtype)
#    else:
    
    guess_iter = 30
    guess_vec = np.zeros((guess_iter, 1))
    grisk_vec = np.zeros((guess_iter, 1))
    tau_vec = np.zeros((guess_iter, 1))
    trisk_vec = np.zeros((guess_iter, 1))
    designer.new_exp()
    for idx_guess in xrange(guess_iter):
        log.info('guess iteration: ' + str(idx_guess))
        # guess=np.array([[[0.0001+0.0001*idx_guess]]],
        # dtype=model.expparams_dtype )
        guess = np.array(
                [model.particle_guess_heuristic(updater, 10000)], dtype=model.expparams_dtype
        )
        # guess_risk=updater.bayes_risk(guess)
        log.info('Your Guess is: ' + str(guess))
#        guess_vec[idx_guess]=guess
#        grisk_vec[idx_guess]=guess_risk
#        
        expparams = designer.design_expparams_field(
                guess, 0, cost_scale_k=10, disp=False, maxiter=10000,
                maxfun=10000, store_guess=True, grad_h=1
        )
#        tau_risk=updater.bayes_risk(expparams)
        log.info('Your Tau is: ' + str(expparams))
#        tau_vec[idx_guess]=expparams
#        trisk_vec[idx_guess]=tau_risk        
#    fig1=plt.figure()
#    plt.scatter(guess_vec,grisk_vec)
#    fig2=plt.figure()
#    plt.scatter(tau_vec,trisk_vec)
#    expparams=np.array([guess_vec[np.argmin(grisk_vec)]],dtype=model.expparams_dtype)
    
# Sweep guesses, evaluate bayesrisk and choose minimum as next expparams 
# Brute Force selection of tau
#    guess_end=10
#    guess_steps=3000
#    guess=np.array(np.linspace(0.01,guess_end,3000),dtype=model.expparams_dtype)
#    guess_risk=guess*0.0
#    for idx_guess in xrange(3000):
#        ppp=np.array([[[guess[idx_guess]]]],dtype=model.expparams_dtype)
#        guess_risk[idx_guess]=updater.bayes_risk(ppp)
#    fig1=plt.figure()
#    plt.plot(guess,guess_risk)
#    expparams=np.array([guess[np.argmin(guess_risk)]],dtype=model.expparams_dtype)
#    print 'Your Tau is: ' + str(expparams)
   

# SIMULATE*******************************************************
    # simulate outcomes- based on the true T1, and the chosen intial value
    # will be replaced by actual data collection from NMR for Mz values
    sim_outcome = model.simulate_experiment(true_model, expparams)
    outcome = sim_outcome
    
    
# NMR EXPERIMENT*************************************************
# USE this instead when doing experiments in NMR
#    outcome=np.array([[[float(raw_input('Enter obtained Mz: '))]]])
#    dummy=float(raw_input('waiting for Mz'))
#    Mz_value=LF.lorentzfit(str(idx_trials+2)+'_spectrum.txt')
#    outcome=np.array([[[Mz_value/abs(Mo_norm)]]])

    # Run SMC and update the posterior distribution
    updater.update(outcome, expparams)
 

# STORE DATA******************************************
    data[idx_trials]['est_mean'] = updater.est_mean()
    data[idx_trials]['sim_outcome'] = outcome
    data[idx_trials]['expparams'] = expparams
   

# PLOT *******************************************  
# plotting particles and weights
    particles = updater.particle_locations
    weights = updater.particle_weights
    fig = plt.figure()

    plt.axvline(updater.est_mean(), linestyle='--', c='blue', linewidth=2)
    plt.axvline(true_model, linestyle='--', c='red', linewidth=2)
    plt.scatter(
            particles[:, 0], weights*10,
            s=50*(1+(weights-1/N_particles)*N_particles)
    )
