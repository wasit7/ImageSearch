"""
This contain class that live on client side of program.

Created on Wed Aug 06 15:04:37 2014

@author: Krerkkiat
"""

import numpy as np

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# ClientNode
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class ClientNode:
    '''
    This represent ClientNode which contain only data.
    '''
    def __init__(self, bag):
        '''
        Init routine

        Parameter(s):
            bag: int[] - array (list) of indeies that point to I in Dataset.
        '''
        self.bag = bag
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Client
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Client:
    '''
    Class in Client (Engine) side.
    Provide base orperator for random forest's tree creation.
    It did not keep each ClientNode, If it no longer use each ClientNode
    it will dereferance that node (If python do that).
    '''
    def __init__(self, dataset):
        '''
        Init routine.

        Parameter(s):
            dataset: Dataset - Dataset to provide to Client.
        '''

        #self.clmax = clmax

        # init dataset
        self.dataset = dataset

        # mark as duplicate to reset()
        self.current_node = None
        self.queue = [ClientNode(dataset.getX())]

    def reset(self):
        '''
        Reset current_node and queue to a init-state
        which ready to create new tree with new parameter.

        Parameter(s):
            None
        Return(s):
            None
        '''

        # init bag (queue)
        self.current_node = None
        self.queue = [ClientNode(dataset.getX())]

    def get_init_parameter(self):
        '''
        Calculate init H (entropy) of first ndoe.
        This run once on each tree creation.

        Parameter(s):
            None
        Return(s):
            H:float - Entropy of first node in queue.
        '''

        return self.cal_entropy(self.queue[0].bag), len(self.queue[0].bag)

    def dequeue(self):
        '''
        Dequeue and set current_node to element that came from dequeue().
        This is part of reduce complexity of previous
        sync-stack recursion design.

        Parameter(s):
            None
        Return(s):
            None
        '''

        self.current_node = self.queue.pop(0)

    def get_theta_tau(self):
        '''
        Return randomed index and theta (np.array) from current_node.

        Parameter(s):
            None
        Return(s): 2 cases
            None, None - if current_node.bag are empty or len(theta) equal to 0.
            theta: numpy.array, tau: numpy.array - if current_node does not empty and len(theta) not equal 0.
        '''

        if len(self.current_node.bag) != 0:
            #attempt = len(self.current_node.bag)//2
            attempt = np.ceil(np.sqrt(len(self.current_node.bag)))
            index = np.random.permutation(self.current_node.bag)[:attempt]

            theta, tau = self.dataset.getParam(index)

            if len(theta) == 0:
                return None, None
            return theta, tau
        return None, None

    def cal_sub_h(self, theta_list, tau_list):
        '''
        **(this algorithm maybe wrong)**

        Calculate sub H (entropy) of current_node
        with the list of theta, tau.

        Parameter(s):
            theta_list: numpy.array - List of theta that will use to try to split node.
            tau_list: numpy.array - List of tau that match with theta that will use to try to split node.
        Return(s):
            results:np.array - result when we try to split with each theta, tau.
            current_node bag size: int - Bag size of current node.
        '''

        # results = np.array
        # for loop in list
        #     split with theta, tau
        #     cntAppear, P, H
        #     add to results
        # return results, q_length

        results = np.zeros(0, dtype=np.float)
        for i in range(len(theta_list)):
            theta = theta_list[i]
            tau = tau_list[i]

            # split with theta, tau
            l = []
            r = []
            for i in self.current_node.bag[:]:
                t = self.dataset.getI(theta, i)
                if t < tau:
                    l.append(i)
                else:
                    r.append(i)

            left_H = self.cal_entropy(l)
            right_H = self.cal_entropy(r)

            results = np.append(results, [left_H*len(l) + right_H*len(r)], axis=1)

        return results, len(self.current_node.bag)

    def split(self, theta, tau):
        '''
        Split current_node to left and right and
        add that node to queue to be process next time.

        Parameter(s):
            theta: float - Real theta that will use with every Engine.
            tau: float - Real tau that will use with every Engine.
        Return(s):
            h_l: float - Entropy of left node.
            l: numpy.array - List of index in left node.
            h_r: float - Entropy of right node.
            r: numpy.array - List of index in right node.
        '''

        # split with theta, tau
        l = []
        r = []
        for i in self.current_node.bag:
            t = self.dataset.getI(theta, i)
            if t < tau:
                l.append(i)
            else:
                r.append(i)

        left, right, h_l, l, h_r, r = ClientNode(l), ClientNode(r), self.cal_entropy(l), len(l), self.cal_entropy(r), len(r)

        # split node
        #left, right, h_l, l, h_r, r = self.current_node.split(theta, tau)

#         print('\tleft bag:', left.bag)
#         print('\tright bag:', right.bag)

        # manage queue
        self.queue.append(left)
        self.queue.append(right)

        # return parameter
        return h_l, l, h_r, r

    def cal_entropy(self, lst):
        '''
        Calculate entropy on lst: list (or numpy.array).

        Parameter(s):
            lst: list or numpy.array - List or numpy.array that need to calculate Entropy.
        Return(s):
            H: float - Entropy of lst.
        '''
        # cnt appear
        appear = self.cnt_appear(lst)

        # cal Prop
        appear += np.finfo(np.float32).tiny
        prop = appear/(appear.sum()*1.0)

        # cal H
        return -1*np.inner(prop, np.log2(prop))

    def cnt_appear(self, lst):
        '''
        Count appear of each class in current_node.bag.

        Parameter(s):
            lst: numpy.array - Usually current_node.bag.
        Return(s):
            appear: numpy.array - Appear of each class in lst (or bag).
        '''

        # cnt appear
        if len(lst) != 0:
            appear = np.bincount(self.dataset.getL(np.array(lst)), minlength=self.dataset.clmax)
            appear = np.array(appear, dtype=np.float)
        else:
            appear = np.zeros(self.dataset.clmax)

        return appear

    def get_bag_size(self):
        '''
        Calculate bag size of current_node.

        Parameter(s):
            None
        Return:
            size: int - Current_node bag size.
        '''
        return np.sum(self.cnt_appear(self.current_node.bag))

if __name__ == '__main__':
    client = Client(dataset)
