# -*- coding: UTF-8 -*-

from json import loads
from os import getcwd, listdir
from os.path import join

import numpy as np
from scipy import misc

# from dataset import Dataset

class ImageDataset():
    def __init__(self, **kwargs):
        # For collect Sample
        # Each index is [x, y, Iimg, Limg]
        # Finally is NP-Array
        self.samples = []   

        # For collect Integal Image
        # Access by self.intImgs[Image][Color]  
        self.intImgs = []    

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

        for index, f in  enumerate(all_file):
            path = join(json, f)

            # JSON Loader
            with open(path, 'r') as fp:
                data = loads(fp.read())

            width = data['width']
            height = data['height']
            rects = data['labels'] 

            # Inner Func For Check (x,y) in Rectangel
            contain = lambda x, y, rect: \
                (rect['x'] <= y < rect['x']+rect['w']) and \
                (rect['y'] <= x < rect['y']+rect['h'])

            # Setup sample & L
            for i in range (10):
                r = np.random.randint(150, height-150)
                c = np.random.randint(150, width-150)
                label = 0
                for rect in rects :
                    if contain(r, c, rect) :                  
                        label = int(rect['label'])
                        break
                self.samples.append([r,c,index,label])

    
            # Integal Image
            self.setup_integal_image(data['path'])
        self.samples = np.array(self.samples)

    def setup_integal_image(self, path):
        # read image pixel as NP-Array -> img
        img = misc.imread(path)

        # Prepare to Integal Image
        colors = []
        colors.append(img[:,:,0] < img[:,:,1]) #r < g
        colors.append(img[:,:,0] < img[:,:,2]) #r < b
        colors.append(img[:,:,1] < img[:,:,2]) #g < b
        colors.append(img[:,:,0] + img[:,:,1] < img[:,:,2]) #r+g < b
        colors.append(img[:,:,0] + img[:,:,2] < img[:,:,1]) #r+b < g
        colors.append(img[:,:,1] + img[:,:,2] < img[:,:,0]) #g+b < a 
        
        # Calculate Integal Image
        self.intImgs.append([])
        for color in colors:
            self.intImgs[-1].append(\
                np.cumsum(np.cumsum(color, axis=0), axis=1))

    def get_hist(self, x, y, w, h, img, color):
        return self.intImgs[img][color][x,y] + self.intImgs[img][color][x+w,y+h] - \
            self.intImgs[img][color][x+w, y] - self.intImgs[img][color][x, y+h]

    # This method is under construction
    # Problem is dicide pass NP-Array or For loop
    def getParam(self, x):

        n = len(x)

        margin = 150
        w_max = 50
        u_max = margin - (w_max/2.0)

        ux = np.random.randint(0, u_max, n)
        uy = np.random.randint(0, u_max, n)
        w = np.random.randint(0, w_max, n)
        h = np.random.randint(0, w_max, n)
        bins = np.random.randint(0, 5, n)

        theta_list = np.array([ux, uy, w, h, bins]).transpose()
        param_list = []
        for i, theta in enumerate(theta_list):
            tau = self.getI(theta, self.samples[x[i]])
            param_list.append([theta, tau])

        return np.array(param_list)
                
    def getI(self, theta, x):
        # @param theta contains ux, uy, w, h, bins
        # @param x contains r, c, img
        return self.get_hist(x[0]+theta[0], x[1]+theta[1], theta[2], theta[3], x[2], theta[4])
        
    def getX(self):
        return np.random.permutation(len(self.samples))

    def getL(self,x):
        return self.samples[x,3]

if __name__ == '__main__':
    pass
    '''
    k = ImageDataset()
    # print k.getL(k.getX())
    k.getL(k.getX())
    theta = (0, 0, 639, 479, 5)
    x1 = (0, 0, 0)
    x2 = (0, 0, 1)
    
    print '='*6, 'Img1', '='*6
    print 'Real answer: ', k.intImgs[0][-1][-1, -1]
    print 'getI answer: ', k.getI(theta, x1)

    print '='*6, 'Img2', '='*6
    print 'Real answer: ', k.intImgs[1][-1][-1, -1]
    print 'getI answer: ', k.getI(theta, x2)
    # k.getI()
    '''
