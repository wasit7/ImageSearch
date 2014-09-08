<<<<<<< HEAD:RandomForest/src/dataset.py
"""
Created on Wed Aug 06 16:12:37 2014

@author: Krerkait
"""

=======
>>>>>>> origin/restructure-dataset-n-client:RandomForest/src/dataset/SpiralDataset.py
import numpy as np

class SpiralDataset:
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
        return np.arange(0, self.getSize())
    
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
    clmax = 5
    spc = 100
    dataset = SpiralDataset(clmax, spc)
