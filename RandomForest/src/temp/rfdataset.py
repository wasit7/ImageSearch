# -*- coding: utf-8 -*-
"""
Created on Sat Aug 02 14:32:12 2014

@author: Wasit
"""
import numpy as np
class dataset:
    Idim=2
    def __init__(self, clmax, spc):
        self.clmax=clmax
        self.spc=spc
        self.size=clmax*spc
        self.I=np.zeros(shape=(self.Idim,clmax*spc),dtype=np.float32)
        self.L=np.zeros(shape=clmax*spc,dtype=np.uint16)
        for x in range(clmax):
            phi = np.linspace(0, 2*np.pi, spc)+\
                  np.random.randn(spc)*0.4*np.pi/clmax +\
                  2*np.pi*x/clmax
            r = np.linspace(0.2, 1, spc)
            self.I[0][x*spc:(x+1)*spc]=r*np.cos(phi)
            self.I[1][x*spc:(x+1)*spc]=r*np.sin(phi)
            self.L[x*spc:(x+1)*spc]=np.ones(shape=(1,spc),dtype=np.uint16)*x
    
    def getL(self, x):
        """get label L, where
            x is address of records"""
        return self.L[x]
    
    def getI(self, theta, x):
        """get raw data I, where
            x is address of records
        theta is a specific dimension of data"""
        return self.I[theta][x]
        
    def getSize(self):
        """size of the dataset"""
        return self.size
    
    def getX(self):
        """random samples X at only init dataset
        X is a set of primary keys of records"""
        #return np.arange(self.size)
        return np.random.permutation(self.size)
        
    def getParam(self, X):
        """ return a number of set ofrandom split parameters
        number of attemp is least tahn or equeal the bag size"""
        attemp=len(X)
        if attemp>0:
            thetas=np.uint16(
                np.random.randint(0, self.Idim, size=attemp)
                )
            taus=np.empty(attemp, dtype=np.float32)
            for i in xrange(attemp):
                taus[i]=self.getI(thetas[i],X[i])        
        else:
            thetas=np.empty(0,dtype=np.uint16)
            taus=np.empty(0, dtype=np.float32)
        return thetas,taus
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds=dataset(5,200)
    markers=('rx','gx','bx','cx','mx','yx','kx')
    for x in range(ds.getSize()):
        plt.plot(ds.getI(0,x), ds.getI(1,x),markers[ds.getL(x)])
        plt.hold(True)
    
    plt.axis([-1,1,-1,1])
    plt.show()
