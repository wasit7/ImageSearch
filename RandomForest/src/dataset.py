# -*- coding: utf-8 -*-
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
    Provide dataset to out random forest
        Spiral dataset
    '''
    def __init__(self, clmax, Q_length):
        '''
        Initial routine
            clmax:int - maximum number of class
            Q_length:int - size of data per class
        '''
        self.Q_length = Q_length    # q size per class per client
        self.clmax = clmax;    # class max of dataset
        
        self.feature = 2
        
        self.I =  np.zeros([self.feature, 0], dtype=np.float)  # np.ndarray row vetor, hold features
        self.L =  np.array([], dtype=np.int)    # np.array, hold label

        # create I
        for x in range(clmax): 
            theta = np.linspace(0, 2*np.pi, self.Q_length)+np.random.randn(self.Q_length)*0.4*np.pi/clmax + 2*np.pi*x/clmax 
            r = np.linspace(0.1, 1, self.Q_length)
            
            self.I = np.append(self.I, [r*np.cos(theta), r*np.sin(theta)], axis=1)
            self.L = np.append(self.L, np.ones(self.Q_length, dtype=np.int)*x, axis=1)

    def getL(self, x):
        """
        Return 
            Label of record x
        """
        pass

    def getI(self, theta, x):
        """
        Return
            raw data of record x with dimension theta
        """
        return self.I[theta, x]

    def getSize(self):
        pass

    def getX(self):
        pass

    def getParam(self, X):
        pass

    def __str__(self):
        '''Return:
            string that represent this class'''
        return 'clmax: {cm}, Q_length: {ql}'.format(cm=self.clmax, ql=self.Q_length)