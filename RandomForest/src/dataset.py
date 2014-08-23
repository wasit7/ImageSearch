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

    def getSize(self):
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

class SpiralDataset(Dataset):
    '''
    Provide Spiral Dataset to Random Forest
    '''
    def __init__(self, clmax, spc):
        '''
        Initial routine
            clmax:int - maximum number of class
            spc:int - size of data per class per client
        '''
        self.clmax = clmax;    # class max of dataset
        self.spc = spc    # q size per class per client
        
        self.dimension = 2 # it is axis x and y
        
        self.I =  np.zeros([self.dimension, 0], dtype=np.float)  # np.ndarray row vetor, hold features
        self.L =  np.array([], dtype=np.int)    # np.array, hold label

        # create I
        for x in range(self.clmax): 
            theta = np.linspace(0, 2*np.pi, self.spc)+np.random.randn(self.spc)*0.4*np.pi/clmax + 2*np.pi*x/clmax 
            r = np.linspace(0.1, 1, self.spc)
            
            self.I = np.append(self.I, [r*np.cos(theta), r*np.sin(theta)], axis=1)
            self.L = np.append(self.L, np.ones(self.spc, dtype=np.int)*x, axis=1)

    def getL(self, x):
        '''
        Return label of record at x
        or a list of labels of records at x(numpy.ndarray)
        '''
        return self.L[x]

    def getI(self, theta, x):
        '''
        Return
            raw data of record x with dimension theta
        '''
        return self.I[theta, x]

    def getSize(self):
        '''
        Return size of dataset
        '''
        #return self.I.shape[1]
        return self.clmax * self.spc

    def getX(self):
        pass
    
    def getParam(self, X):
        '''
        Return list of theta, tau for X
        '''
        theta = np.random.randint(self.dimension, size=len(X))
        tau = self.getI(theta, X)
        return theta, tau

    def __str__(self):
        '''
        Return:
            string that represent this class
        '''
        return 'clmax: {cm}, spc: {ql}'.format(cm=self.clmax, ql=self.spc)

if __name__ == '__main__':
    #dataset = LibraryImageDataset()
    # print dataset.rectL
    # print dataset.getL(188, 0, 1)
    pass