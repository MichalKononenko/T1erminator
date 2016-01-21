"""
This is the python script to run the Topspin commands and do data processing
"""
import numpy as np
import scipy as sci
#Open a data set with parameters put in called "T1 Learning"

RE(["Qinfer-T1 Learning","1","1","/opt/topspin","Kissan"]) #standard d2 delay is 0
ZG()
#code to determine intial Mz(0)=Mo
#code to determine SNR 

#start loop 
trials=5
for idx_trials in xrange(trials):
    RE_IEXNO()                  #increments dataset to new expno, and sets as current
    PUTPAR("d2", expparams)     #update to new delay time
    ZG()                        #run experiment
   #processing 
    FT()                        #fourier transform of signal
    #phase correction to 
    data=GETPROCDATA(-500,500)  #get vector list of data 
 
                     
    
