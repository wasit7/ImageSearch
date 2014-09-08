"""
Created on Wed Aug 06 15:08:04 2014

@author: KrerKait
"""

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Testing: % Correct on one multiple trees
# It load tree before testing
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import sys
import random

import numpy as np

#from master import Master
from recall import Recall

class Record:
    def __init__(self, features, label):
        self.features = features
        self.label = label

def dataSetGenerator(clmax):
    data = []
    for x in range(clmax):
        theta = np.linspace(0, 2*np.pi, 100)+np.random.randn(100)*0.4*np.pi/clmax + 2*np.pi*x/clmax
        r = np.linspace(0.1, 1, 100)
        xPoints = r*np.cos(theta)
        yPoints = r*np.sin(theta)
        for i in range(len(xPoints)):
            data.append(Record([xPoints[i], yPoints[i]], x))
    return data

def main(clmax, mainfile):
    # generate dataset for test
    dataset = dataSetGenerator(clmax)

    #master = Master() 
    #roots = master.load_trees(prefix=prefix)
    recall = Recall()
    #roots = recall.load_trees(prefix=prefix)
    roots = recall.load_forest(mainfile)

    sampling_rate = 30
    samples = int(sampling_rate*len(dataset)/100)

    corrects = []
    for x in range(100) :
        correct = 0
        for i in random.sample(range(len(dataset)), samples):
            expt = dataset[i].label
            res = np.argmax(recall.get_results(roots, dataset[i].features))
            if res == expt:
                correct += 1
        corrects.append(correct)

    #print('Correct: ', corrects)
    corrects = sum(corrects)/len(corrects)
    print('\nTesting with clmax: {}'.format(clmax))
    print('Correct/Total: {c}/{t}'.format(c=corrects, t=samples))
    print('% Correct from {} tree(s): {}%'.format(len(roots), float(corrects)/float(samples)*100))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: ', sys.argv[0], ' <mainfile>')
        sys.exit(1)
    main(int(sys.argv[1]))
