"""
Created on Sat Sep 13 20:05:17 2014

@author: Wasit

"""

class CHDataset:
    '''Dataset interfacing class using color histogram of images'''
    def __init__(self, clmax, spi):
        #load all dataset.json
        for root, dirs, files in os.walk("images"):
            for subdir in dirs:
                for iroot,idirs,ifiles in os.walk(os.path.join(root,subdir)):
                    for f in ifiles:
                        if f.endswith('json'):
                            print os.path.join(iroot,f)
                                
        #samples[x]=[ [img, row, column, label] ]
        #I=integralImage(all images)
    def getL(self, x):
        return samples[x].label
    def getI(self, theta, x):
        return value of a bin in a histogram within rectangle boundary at x
    def getSize(self):
        return len(samples)
    def getX(self):
        return np.arange(0, len(samples))
    def getParam(self, X):
        for all x in X:
            thetas[x]=random parameters in range
            taus[x]=getI(theta,x)
        return thetas,taus
    def __str__(self):
if __name__ == '__main__':
    clmax = 500     #max number of classes
    spi = 10            #samples per image
    dataset = SpiralDataset(clmax, spc)