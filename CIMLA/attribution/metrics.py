import numpy as np

def RMSD(s1,s2):
    return np.power(np.mean(np.square(s2-s1), axis = 0),0.5)

def MSD(s1,s2):
    return np.mean(np.square(s2-s1), axis = 0)
