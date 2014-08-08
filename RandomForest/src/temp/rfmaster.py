# -*- coding: utf-8 -*-
"""
Created on Sat Aug 02 14:32:12 2014

@author: Wasit
"""
import numpy as np
from rfclient import client
class master:
    
    minAttemp=10
    def __init__(self,clmax=2,spc=5,maxDepth=10,numClients=2):
        self.maxDepth=maxDepth
        self.numClients=numClients        
        self.root=mnode(maxDepth)
        self.queue=[self.root]
        self.curNode=None
        self.clmax=clmax
        self.spc=spc
        self.p_temp=None
#create client
        self.myClients=[client(self.clmax,self.spc) for i in xrange(self.numClients)]
    def pop(self):
#pop from the last
        self.curNode=self.queue.pop()
#pop client queue
        for i in xrange(self.numClients):
            self.myClients[i].pop()
    
    def push(self):
        self.curNode.left=mnode(
            self.curNode,
            self.curNode.depth-1,
            self.curNode.key+'0'
        )
        self.curNode.right=mnode(
            self.curNode,
            self.curNode.depth-1,
            self.curNode.key+'1'
        )
        self.queue.append(self.curNode.left)        
        self.queue.append(self.curNode.right)
#push client queue        
        for i in xrange(self.numClients):
            self.myClients[i].push()
            
    def terminate(self,code):
#set the histogram
        
        print('m****terminated:{0} p:{1}'.format(code,self.curNode.p))
        
    def addH(self):
        H=np.zeros((self.numClients,1))
        Q=np.zeros((self.numClients,1))
        p=np.zeros((self.numClients,self.clmax))
        for i in xrange(self.numClients):
            H[i,0], p[i,:], Q[i,0]=self.myClients[i].getH()
            
        self.curNode.H = np.sum(Q*H)/(np.sum(Q)+tiny)
        print('m    addH H:{0}'.format(H))
        print('m    addH p:{0}'.format(p))
        print('m    addH Q:{0}'.format(Q))
        self.p_temp = np.sum(np.tile(Q,(1,self.clmax))*p,axis=0)+tiny
        print('m    addH02: ptem:{0}'.format(self.p_temp))
        self.p_temp = self.p_temp/(np.sum(self.p_temp))
        print('m    addH03: ptem:{0}'.format(self.p_temp))
        self.curNode.p=self.p_temp
        
        print('m    addH curNode.p:{0}'.format(self.curNode.p))
        print('m    addH curNode.H:{0}'.format(self.curNode.H))
        minQ=np.amin(Q)       
        
        return minQ
           
    def split(self):
        while len(self.queue)>0:
#01#02 pop both master and clients
            self.pop()
#03 if reach maximum depth
            print('{0}, depth{1}'.\
                    format(self.curNode.key,self.curNode.depth)
                    )
#032 addH to left and right nodes
            minQ=self.addH()  
            if self.curNode.depth<1:
#04     terminate
                self.terminate('Depth')
#041 else:
            else:
              

                    
#06 if size of taus too samll
                if minQ<3:
#07    terminate
                    
                    self.terminate('Q')
#08 else
                else:
#05 collectParam from clients phi={thetas,taus}
                    thetas=np.empty(0,dtype=np.uint16)
                    taus=np.empty(0, dtype=np.float32)
                    for i in xrange(self.numClients):
                        newthetas, newtaus =self.myClients[i].getParam()
                        thetas=np.append(thetas,newthetas)
                        taus=np.append(taus,newtaus)
    
                    print('m  thetas:{0}'.format(thetas))
                    print('m  taus:{0}'.format(taus))
                    print('m  len(taus): {0}'.format(len(taus)))
#09 ({QlHl+QrHr},Q) getQH(thetas,taus)
                    QHs=np.zeros( (self.numClients, len(taus)) )
                    Qs=np.zeros( self.numClients)
                    for i in xrange(self.numClients):
                        QHs[i,:],Qs[i] =self.myClients[i].getQH(thetas,taus)
                        print('m    QHs:{0}'.format(QHs))
                        print('m    Qs:{0}'.format(Qs))
                    sumQHs=np.sum(QHs,axis=0)
                    sumQs=np.sum(Qs)
                    print('m    sumQHs:{0}'.format(sumQHs))
                    print('m    sumQs:{0}'.format(sumQs))
                    
#10 (Gx,theta,tau) Gmax()
                    newH=sumQHs/sumQs
                    print('m  newH: {0}'.format(newH))
                    G=self.curNode.H-newH
                    print('G:{0}'.format(G))
                    iGx=np.argmax(G)
                    print( 'iGx:{0}, Gx:{1}'.format( np.argmax(G),G[iGx] ) )
#11 if Gx<0
                    if G[iGx]<0.0:
#12    terminate
                        self.terminate('G')
#13 else
                    else:
#14     csplit(theta,tau)
                        self.curNode.theta=thetas[iGx]
                        self.curNode.tau=taus[iGx]
                        for i in xrange(self.numClients):
                            self.myClients[i].csplit(thetas[iGx],taus[iGx])
#15 push both on master and client ##clear p
                        self.push()
#16 clean current node
                                                
    def classify(self,data):
        node=self.root        
        while node.left is not None:
            print('{0} H:{1} p:{2} theta:{3} tau:{4}'\
            .format(node.key,node.H,node.p,node.theta,node.tau)
            )
            if data[int(node.theta)]<node.tau:
                node=node.left
            else:
                node=node.right
        
        print('{0} theta:{0} tau:{1}'.format(node.theta,node.tau))
        print('----p:{1} H:{2}'.format(node.key,node.p,node.H))
        return np.argmax(node.parent.p)
        
class mnode:
    def __init__(self, parent=None, depth=10, key='R'):
        self.left=None
        self.right=None
        self.parent=parent
        self.depth=depth
        self.key=key
        self.theta=None
        self.tau=None
        self.H=None
        self.p=None#only for the terminal node

tiny=np.finfo(np.float32).tiny

if __name__ == '__main__':
    mm=master(clmax=2,spc=5,maxDepth=10,numClients=1)
    mm.split()