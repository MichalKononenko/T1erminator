import numpy as np
__author__ = 'Michal Kononenko'


class NMRRunner(object):
    def __init__(self):
        RE(["Qinfer-T1 Learning","1","1","/opt/topspin","Kissan"]) #standard d2 delay is 0

    def run(self, candidate):
        RE_IEXNO()
        PUTPAR("d2", candidate)
        ZG()
        FT()
        data = GETPROCDATA(-500, 500)
        return np.max(data)
