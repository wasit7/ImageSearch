# -*- coding: utf-8 -*-
"""
Created on Sat Aug 02 13:40:59 2014

@author: Wasit
"""
import numpy as np
import matplotlib.pyplot as plt
from rfmaster import master
from rfdataset import dataset
if __name__ == '__main__':
    clmax=5
    spc=1000
    maxDepth=40
    numClients=1
    rf = master(clmax,spc,maxDepth,numClients)#master init root and queue     
    rf.split()
    
    
    fig = plt.figure()
    d=0.1
    y, x = np.mgrid[slice(-1, 1+d, d), slice(-1, 1+d, d)]
    z=np.zeros(x.shape,dtype=int)
    for r in xrange(x.shape[0]):
        for c in xrange(x.shape[1]):
            z[r][c]=rf.classify([ x[r][c],y[r][c] ])
            pass
    plt.axis([-1,1,-1,1])
    plt.pcolor(x,y,z)
    #plt.contour(x,y,z)
    plt.show()
    
    plt.hold(True)
    ds=rf.myClients[0].ds
    markers=('b^','g^','y^','c^','r^','m^','k^')
    for x in range(ds.getSize()):
        plt.plot(ds.getI(0,x), ds.getI(1,x),markers[ds.getL(x)])
        plt.hold(True)
    
    
    def evaluate():
        for i in xrange(ds.size):
            L_truth=ds.getL(i)
            L_classified=rf.classify([ds.getI(0,i),ds.getI(1,i)])
            print("L_t:{0},L_c:{1}".format(L_truth,L_classified))
    def onclick(event):
        print ('xdata:{0}, ydata:{1}'.format(event.xdata, event.ydata))
        print ('--classify:{0}'.format(rf.classify([event.xdata, event.ydata])))
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
