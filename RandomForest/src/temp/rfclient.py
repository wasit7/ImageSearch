#ienimportimport sys
#if 'dataset' in sys.modules:
#    del sys.modules['dataset']

import numpy as np
from rfdataset import dataset
class client:
    
    maxAttemp=1000
    def __init__(self,clmax,spc):
        self.ds=dataset(clmax,spc)
        self.root=cnode(self.ds.getX())
        self.queue=[self.root]
        self.curNode=None
    
    def pop(self):
        self.curNode=self.queue.pop()
        
    def push(self):
#add  left node
        self.queue.append(self.curNode.left)
#add  right node
        self.queue.append(self.curNode.right)
        
    def getParam(self):
        attemp=min(self.maxAttemp, len(self.curNode.bag)*10)
        sample_id=np.random.randint(len(self.curNode.bag), size=attemp)
        X=self.curNode.bag[sample_id]
        return self.ds.getParam(X)
    
    def getH(self):
        """to get entropy of the current node"""
        bag=self.curNode.bag
        Q=len(bag)
        if Q>0:
            p=np.bincount(self.ds.getL(bag),minlength=self.ds.clmax)
        else:
            p=np.ones(self.ds.clmax,dtype=np.int32)
        p=p + tiny
        p=p/np.sum(p)
        H=entropy(p)
        print('c    getH H:{0}'.format(H))
        print('c    getH p:{0}'.format(p))
        print('c    getH Q:{0}'.format(Q))
        return H, p, len(bag)
        
    def getQH(self,thetas,taus):
        """to get sub entropy of the current node
        by using the given split parameters"""
        bag=self.curNode.bag
        QHs=np.zeros(len(taus))
        for a in xrange(len(thetas)):
            bagL=bag[self.ds.getI(thetas[a],bag)<taus[a]] 
            bagR=bag[self.ds.getI(thetas[a],bag)>=taus[a]]
            pL=np.bincount(self.ds.getL(bagL),minlength=self.ds.clmax)
            pR=np.bincount(self.ds.getL(bagR),minlength=self.ds.clmax)
            if a<1:       
                print('c  pL: {0}, pR: {1}'.format(pL,pR))
            HL=entropy(pL)
            HR=entropy(pR)
            if a<1:
                print('c  len(bagL):{},len(bagR):{}\nc  HL:{},HR:{}, bag.size:{}'\
                    .format(len(bagL),bagR.size,HL,HR,bag.size))
                print('c==QH:{0} Q:{1}'.format(len(bagL)*HL+len(bagR)*HR, bag.size))
            QHs[a]=len(bagL)*HL+len(bagR)*HR
        return QHs, bag.size
        
    def csplit(self, theta, tau):
        bag=self.curNode.bag
        bagL = bag[ self.ds.getI(theta,bag)<tau ]
        bagR = bag[ self.ds.getI(theta,bag)>=tau ]
        self.curNode.left=cnode(bagL)
        self.curNode.right=cnode(bagR)
#clean the visited node
        self.curNode.bag=None
        
class cnode:
    def __init__(self, bag):
        print('size of the root bag: {0}'.format(len(bag)))
        self.bag=bag
        self.left=None
        self.right=None
        
tiny=np.finfo(np.float32).tiny
def entropy(p):
    p=p + tiny
    p=p/np.sum(p)
    return np.inner(-p,np.log2(p))
    
if __name__ == '__main__':
    cc=client(2,5)
    cc.pop()
    cc.getParam()