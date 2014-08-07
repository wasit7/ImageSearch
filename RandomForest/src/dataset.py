# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 16:12:37 2014

@author: Krerkait
"""
from json import loads
from os import getcwd, listdir
from os.path import join

import numpy as np
L = []

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

    def getSize(self):
        raise NotImplementedError

    def getX(self):
        raise NotImplementedError

    def getParam(self, X):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

class Rectangle:
    def __init__(self, label='', x=0, y=0, w=0, h=0):
        self.label = label
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __contains__(self, pos):
        return self.label  if (self.x <= pos[0] <self.x+self.w) and \
            (self.y <= pos[1] <self.y+self.h) else 0

class LibraryImageDataset(Dataset):
    def __init__(self, **kwargs):
        global L

        if 'json' in kwargs:
            json = kwargs['json']
        else :
            json = join('..', '..', 'App', 'json')

        for index, f in  enumerate(listdir(json)):
            path = join(json, f)
            with open(path, 'r') as fp:
                data = loads(fp.read())
            rects = []
            for label in data['labels']:
                rects.append(Rectangle(label=label['label'], x=label['x'],\
                    y=label['y'], w=label['w'], h=label['h']))
            L.append(rects)

    def getL(self, x, y, img):
        global L
        for rect in L[img]:
            if (x,y) in rect :
                return rect.label
        return 0


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
        
        self.dimension = 2 # it is axis x and y
        
        self.I =  np.zeros([self.dimension, 0], dtype=np.float)  # np.ndarray row vetor, hold features
        self.L =  np.array([], dtype=np.int)    # np.array, hold label

        # create I
        for x in range(self.clmax): 
            theta = np.linspace(0, 2*np.pi, self.data_per_class)+np.random.randn(self.data_per_class)*0.4*np.pi/clmax + 2*np.pi*x/clmax 
            r = np.linspace(0.1, 1, self.data_per_class)
            
            self.I = np.append(self.I, [r*np.cos(theta), r*np.sin(theta)], axis=1)
            self.L = np.append(self.L, np.ones(self.data_per_class, dtype=np.int)*x, axis=1)

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
        return self.I.shape[1]

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
        return 'clmax: {cm}, data_per_class: {ql}'.format(cm=self.clmax, ql=self.data_per_class)

if __name__ == '__main__':
    dataset = LibraryImageDataset()
    print L
    print dataset.getL(0, 0, 1)
