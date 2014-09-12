"""
Created on Wed Aug 06 16:12:37 2014

@author: Krerkait
"""

import numpy as np

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Dataset
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Dataset:
    '''
    Provide dataset to Random Forest
    '''
    def __init__(self):
        pass

    def getL(self, x):
        '''
        Input: 
            x: int or numpy.array
        Return:
            label of record x
        '''
        raise NotImplementedError

    def getI(self, theta, x):
        '''
        Input:
            theta: int or tuple of number
            x: int or numpy.array
        Return:
            raw data of record x with dimension theta
        '''
        raise NotImplementedError

    def getX(self):
        raise NotImplementedError

    def getParam(self, X):
        '''
        Input:
            X: numpy.array
        Return:
            list of theta, tau
        '''
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError