# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 16:12:37 2014

@author: Krerkait
"""
from json import loads
from os import getcwd, listdir
from os.path import join
from PIL import Image
from itertools import permutations
from time import time

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
        self.labeledImages = []
        # self.samples = self.getSamples()

        # load JSON
        if 'json' in kwargs:
            json = kwargs['json']
        else :
            json = join('..', '..', 'App', 'json')

        all_file = listdir(json)
        if not isinstance(all_file, list):
            all_file = [all_file]

        # length of dataset
        self.lenDataset = len(all_file)
        # dimension of dataset
        #print all_file

        self.all_colors = ['rg' 'rb', 'gb', 'rbg', 'brg', 'grb']
        for index, f in  enumerate(all_file):
            path = join(json, f)
            
            # setup L
            with open(path, 'r') as fp:
                data = loads(fp.read())
            # print data
            rects = []
            for label in data['labels']:
                rects.append(Rectangle(label=label['label'], x=label['x'],\
                    y=label['y'], w=label['w'], h=label['h']))
            self.labeledImages.append(rects)
            
            # setup I
            # img = Image.open(data['path'])
            # w, h = img.size 
            # colors = {}
            # for color in self.all_colors[:]:
            #     colors[color] = []
            # for y in range(h):
            #     for color in colors:
            #         colors[color].append([])
            #     for x in range(w):
            #         pix = img.getpixel((x,y))
            #         target_color = self.find_color(*pix)
            #         for color in colors:
            #             if color in target_color:
            #                 colors[color][-1].append(1)
            #             else :
            #                 colors[color][-1].append(0)

            # for color in colors:
            #     colors[color] = np.array(colors[color])
            #     integral_image = np.cumsum(np.cumsum(colors[color], axis=0), axis=1)
            #     print integral_image

    def find_color(self, r, g, b):
        colors = []
        for i, color in enumerate(self.all_colors[:]):
            if i < 3:
                exec('res = %s > %s'%(color[0], color[1]))
            else :
                exec('res = %s**2 > %s*%s'%(color[0], color[1], color[2]))
            if res :
                colors.append(color)
        return colors   

    def getL(x) :
        x, y, img = self.samples[x]
        for rect in self.L[img]:
            if (x,y) in rect :
                return rect.label
        return 0

    def getX(self):
        '''
        To random n=10 samples from each image
        '''
        for i in self.labeledImages:
            width=i.width
            height=i.height
            c=np.random.randint(width)
            r=np.random.randint(height)
            self.samples.append([r,c,img])

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