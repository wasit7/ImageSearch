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
        Return label of record x
        '''
        raise NotImplementedError

    def getI(self, theta, x):
        '''
        Return raw data of record x with dimension theta
        '''
        raise NotImplementedError

    def getDimension(self):
        raise NotImplementedError

    def getSize(self):
        raise NotImplementedError

    def getX(self):
        raise NotImplementedError

    def getParam(self, X):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class SpiralDataset(Dataset):
    '''
    Provide Spiral Dataset to Random Forest
    '''
    def __init__(self, clmax, data_per_class):
        '''
        Initial routine
            clmax:int - maximum number of class
            data_per_class:int - size of data per class
        '''
        self.clmax = clmax;    # class max of dataset
        self.data_per_class = data_per_class    # q size per class per client
        
        dimension = 2 # it is axis x and y
        
        self.I =  np.zeros([dimension, 0], dtype=np.float)  # np.ndarray row vetor, hold features
        self.L =  np.array([], dtype=np.int)    # np.array, hold label

        # create I
        for x in range(self.clmax): 
            theta = np.linspace(0, 2*np.pi, self.data_per_class)+np.random.randn(self.data_per_class)*0.4*np.pi/clmax + 2*np.pi*x/clmax 
            r = np.linspace(0.1, 1, self.data_per_class)
            
            self.I = np.append(self.I, [r*np.cos(theta), r*np.sin(theta)], axis=1)
            self.L = np.append(self.L, np.ones(self.data_per_class, dtype=np.int)*x, axis=1)

    def getL(self, x):
        '''
        Return 
            Label of record x
        '''
        pass

    def getI(self, theta, x):
        '''
        Return
            raw data of record x with dimension theta
        '''
        return self.I[theta, x]

    def getDimension(self):
        '''
        Return dimension of data
        '''
        return self.I.shape[0]

    def getSize(self):
        '''
        Return size of dataset
        '''
        return self.I.shape[1]

    def getX(self):
        pass

    def getParam(self, X):
        pass

    def __str__(self):
        '''
        Return:
            string that represent this class
        '''
        return 'clmax: {cm}, data_per_class: {ql}'.format(cm=self.clmax, ql=self.data_per_class)