"""
Contain class that provide spiral dataset to random forest.

@author: Krerkkiat
updated by Wasit
"""

import numpy as np

class SpiralDataset:
    '''
    Provide Spiral Dataset to Random Forest
    '''

    def __init__(self, clmax, spc):
        '''
        Initial routine.

        Parameter(s):
            clmax: int - Maximum number of class.
            spc: int - Size of data per class per client.
        '''

        self.clmax = clmax   # class max of dataset
        self.spc = spc    # q size per class per client

        self.dimension = 2 # it is axis x and y

        self.I = np.zeros([self.dimension, 0], dtype=np.float)  # np.ndarray row vetor, hold features
        self.L = np.array([], dtype=np.int)    # np.array, hold label

        # create I
        for x in range(self.clmax):
            theta = np.linspace(0, 2*np.pi, self.spc)+np.random.randn(self.spc)*0.4*np.pi/clmax + 2*np.pi*x/clmax
            r = np.linspace(0.1, 1, self.spc)

            self.I = np.append(self.I, [r*np.cos(theta), r*np.sin(theta)], axis=1)
            self.L = np.append(self.L, np.ones(self.spc, dtype=np.int)*x, axis=1)

    def getL(self, x):
        '''
        Lookup database for a lebel of data at x.

        Parameter(s):
            x: int or numpy.array - Index or indexes of data that you need to get label.
        Return(s):
            label: int - Label of data at x.
        '''

        return self.L[x]

    def getI(self, theta, x):
        '''
        Lookup table by theta for tau (splitting parameter or threshold) at index x.

        Parameter(s):
            theta: int - theta that will use for lookup.
            x: int - Index of data.
        Return(s):
            tau: float - tau or raw data of data at index x with dimension theta.
        '''
        return self.I[theta, x]

    def getX(self):
        '''
        Make a list of index that will use when initial root node at Client side

        Return(s):
            idx_list: list - List of index of data.
        '''

        return np.arange(0, self.clmax * self.spc)

    def getParam(self, X):
        '''
        Random theta and then get tau from that randomed theta at index x.

        Parameter(s):
            x: list - List of index that will use to get tau.
        Return(s):
            theta: list - List of randomed theta.
            tau: list - List of tau with lookup by theta and x.
        '''

        theta = np.random.randint(self.dimension, size=len(X))
        tau = self.getI(theta, X)

        return theta, tau

    def __str__(self):
        '''
        Nothing spacial, use when debug.

        Return:
            txt: str - String that represent this class.
        '''

        return 'clmax: {cm}, spc: {ql}'.format(cm=self.clmax, ql=self.spc)

if __name__ == '__main__':
    clmax = 500
    spc = 1e3
    dataset = SpiralDataset(clmax, spc)
