"""
Created on Wed Aug 06 15:06:13 2014

@author: Krerkait
"""
import os
import sys
import time
import json
import datetime

import numpy as np
from IPython import parallel

from client import Client

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# MasterNode
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class MasterNode:
    def __init__(self, old_h=None, depth=None):
        self.theta = None
        self.tau = None
        
        self.prop = None
        
        self.old_h = old_h
        self.depth = depth
        
        self.left = None
        self.right = None
    
    def __str__(self):
        return 'm_node@{d} theta: {tt}, tau: {t}, prop: {p}'.format(d=self.depth, tt=self.theta, t=self.tau, p=self.prop)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Master
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Master:
    '''This class are on Master side in cluster'''
    def __init__(self, clmax=5, np2c=30, max_depth=30, min_bag_size=2):
        '''Init routine
        That setup client
        '''
        self.clmax = clmax
        self.np2c = np2c
        self.max_depth = max_depth
        self.min_bag_size = min_bag_size
            
        # init cluster client
        self.clients = parallel.Client(packer='pickle')
        self.clients.block = True

        self.dview = self.clients.direct_view(self.clients.ids[1:])
        self.dview.block = True
    
    def init_client(self):
        # init client
        self.dview.push({'spc':self.np2c, 'clmax':self.clmax});
        self.dview.run('dataset.py')
        self.dview.run('client.py')
    
    def cal_init_h(self):
        '''Calculate init entropy of each client and weighted sum it
        Return:
            h (entropy) of init node (use in master)
        '''
        hq = 0
        sum_q = 0
        
        self.dview.execute('init_parameter = client.get_init_parameter()')
        init_parameter = self.dview['init_parameter']
        
        for h, q in init_parameter:
#             h, q = c.get_init_parameter()
            hq += h*q
            sum_q += q
        h = hq/float(sum_q)
        return h
    
    def create_tree(self):
        '''This will create one tree per call so if u need to create more than one tree please call this multiple time
        This method does not handle save tree process so you need to handle it in youn own way
        This will froce to save tree that it created
        '''

        start_time = datetime.datetime.now()
        
        # prepare for creation
        if hasattr(start_time, 'timestamp'):
            tree_filename = str(start_time.timestamp()) + '.json'
        else:
            tree_filename = start_time.strftime('%s') + '.json'

        H = self.cal_init_h()
        root = MasterNode(H, 0)
        queue = [root]

        while len(queue) != 0:
            current_node = queue.pop(0)
            self.dview.execute('client.dequeue()')

            left, right = self.split(current_node)

            if left != None:
                current_node.left = left
                queue.append(left)

                current_node.right = right
                queue.append(right)

        end_time = datetime.datetime.now()

        if hasattr(start_time, 'timestamp'):
            time_use = end_time.timestamp() - start_time.timestamp()
        else:
            time_use = int(end_time.strftime('%s')) - int(start_time.strftime('%s'))

        # reseting and cleanning memory
        self.dview.execute('client.reset()')
        #self.clients.purge_everything()
        self.clients.purge_results('all')

        print('Time use: {} sec'.format(time_use))

        # prepare for save tree
        tree_information = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'time_use': time_use,
            'tree': self._node_to_dict(root)
        }

        with open(tree_filename, 'w') as f:
            json.dump(tree_information, f, indent=2)
            print('Tree was save to {}'.format(tree_filename))

        return root, tree_filename, time_use
                
    def split(self, node):
        '''This perform 'try split' on node, then find best G finally 'real split' it
        Return:
            left:MasterNode
            right:MasterNode
        '''
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # check terminate case
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        if node.depth > self.max_depth:
#             print(list(range(node.depth + 1)), 'terminated with max_depth')
            self.on_terminate(node)
            return (None, None)
        
        # gather length of bag with scope of bag(left, right)
        self.dview.execute('bag_size = client.get_bag_size()')
        bag_size = self.dview['bag_size']
        
        #if node.depth == 0 :
        #    print(list(range(node.depth + 1)), '\tbag size:', sum(bag_size), bag_size)
        
        if not any([b > self.min_bag_size for b in bag_size]): # sum across client
#             print(list(range(node.depth + 1)), 'terminated with min_bag_size')
            self.on_terminate(node)
            return (None, None)
        
        g = self.cal_information_gain(node)
        if g <= 0:
#             print(list(range(node.depth + 1)), 'terminated with G')
            self.on_terminate(node)
            return (None, None)
        
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # client: split with theta, tau
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        #print('trying split')
        self.dview.push({'theta':node.theta, 'tau':node.tau})
        self.dview.execute('split_result = client.split(theta, tau)')
        split_result = self.dview['split_result']
        
        h_left = 0
        h_right = 0
        q_left = 0
        q_right = 0
        for h_l, q_l, h_r, q_r in split_result:
            h_left += h_l*q_l
            h_right += h_r*q_r
            
            q_left += q_l
            q_right += q_r
        h_left = h_left/float(q_left)
        h_right = h_right/float(q_right)
        
        return MasterNode(h_left, node.depth+1), MasterNode(h_right, node.depth+1)
    
    def cal_information_gain(self, node):
        '''Calculate information gain on node
        Return:
            G:float
        '''
        # collect theta_tau
        self.dview.execute('theta_taus = client.get_theta_tau()')
        
        theta_taus = self.dview['theta_taus']
        
        theta_list = np.array([], dtype=np.int)
        tau_list = np.array([])
        for theta, tau in theta_taus:
            if theta == None and tau == None:
                continue
            theta_list = np.append(theta_list, theta, axis=1)
            tau_list = np.append(tau_list, tau, axis=1)
        
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # distribute & gather
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        self.dview.push({'theta_list': theta_list, 'tau_list':tau_list})
        self.dview.execute('sub_h = client.cal_sub_h(theta_list, tau_list)')
        sub_h = self.dview['sub_h']
        sub_hs = np.empty([len(self.dview), len(theta_list)], dtype=np.float)
        size = []
        for i, (result, q_size) in enumerate(sub_h):
            sub_hs[i,:] = result
            size.append(q_size)
#         print(size)
    
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # cal g
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        # sum over client
        h_of_iteration = np.sum(sub_hs, axis=0)
        
        # cal new (and complete) H
        new_h = h_of_iteration/float(sum(size))
        
#         print(list(range(node.depth + 1)), 'new_h: ', new_h)
        
        # cal g
        g = node.old_h - new_h
        
#         print(list(range(node.depth + 1)), 'g: ', g)
        
        # find best g
        best_g_idx = np.argmax(g)
        
        # svae parameter
        node.theta = theta_list[best_g_idx]
        node.tau = tau_list[best_g_idx]
        
#         print(list(range(node.depth + 1)), 'parameter saved on:', str(node))
        
        return g[best_g_idx]
    
    def on_terminate(self, node):
        '''This are call when we reach termonate case (3 cases)
        It will save prop of each class onto node:MasterNode'''
        
        self.dview.execute('appear = client.cnt_appear(client.current_node.bag)')
        appear = self.dview['appear']
        
        appears = []
        for a in appear:
            appears.append(a)
        appears = np.array(appears, dtype=np.float)
        appears = np.sum(appears, axis=0)/np.sum(appears)
        
        node.prop = appears
        
        #print(list(range(node.depth + 1)), '\tprop saved on:', str(node))
    
    def _node_to_dict(self, node):
        if node == None:
            return None

        result = {}

        if node.prop == None:
            result['prop'] = None
        else:
            result['prop'] = list(node.prop)
            
        if node.theta == None :
            result['theta'] = None
            result['tau'] = None
        else:
            result['theta'] = np.asscalar(node.theta)
            result['tau'] = np.asscalar(node.tau)

        result['left'] = self._node_to_dict(node.left)
        result['right'] = self._node_to_dict(node.right)

        return result
    
    def save_tree(self, tree, filename):
        '''
        Write selected decision tree to files,
        '''
        
        f = open(filename, 'w')
        result = self._node_to_dict(tree)
        json.dump(result, f, indent=2)
        f.close()
        print('{} saved'.format(filename))