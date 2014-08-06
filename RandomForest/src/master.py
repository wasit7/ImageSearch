# -*- coding: utf-8 -*-
"""
Created on Wed Aug 06 15:06:13 2014

@author: Krerkait
"""

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Master
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import sys
import time
import json

import numpy as np
from IPython import parallel

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
        self.dview.push({'np2c':self.np2c, 'clmax':self.clmax});
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
        This will return a root of tree that it created
        '''
        
        # prepare for creation
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
        
        # reseting and cleanning memory
        self.dview.execute('client.reset()')
        #self.clients.purge_everything()
        self.clients.purge_results('all')
        
        return root
                
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
        
        self.dview.execute('appear = client.cnt_appear()')
        appear = self.dview['appear']
        
        appears = []
        for a in appear:
            appears.append(a)
        appears = np.array(appears, dtype=np.float)
        appears = np.sum(appears, axis=0)/np.sum(appears)
        
        node.prop = appears
        
        #print(list(range(node.depth + 1)), '\tprop saved on:', str(node))
    
    def get_results(self, roots, I):
        '''This use for get result from multiple dicision tree on master for feature list I'''
        props = []
        for root in roots:
            props.append(self._get_result(root, I))
        props = np.array(props)
        return np.sum(props, axis=0)/len(roots)
    
    def get_result(self, tree, I):
        '''This use for get result from tree:int dicision tree on master for feature list I'''
        return self._get_result(tree, I)
    
    def _get_result(self, node, I):
        '''This use for get result from dicision tree on master for feature list I (recursive)'''
        if node.prop != None:
            return node.prop
        
        tau = I[node.theta]

        if node.left and tau < node.tau:
            return self._get_result(node.left, I)
        if node.right and tau >= node.tau:
            return self._get_result(node.right, I)
        return -1  # for unknow
    
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
    
    def _dict_to_node(self, dict_):
        if dict_ == None:
            return None
        
        root = MasterNode()
        
        if dict_['prop'] == None:
            root.prop = None
        else:
            root.prop = np.array(dict_['prop'])
            
        if dict_['theta'] == None:
            root.theta = None
            root.tau = None
        else:
            root.theta = np.int32(dict_['theta'])
            root.tau = np.float64(dict_['tau'])
            
        root.left = self._dict_to_node(dict_['left'])
        root.right = self._dict_to_node(dict_['right'])
        
        return root
    
    def save_tree(self, tree, filename=None):
        '''
        Write selected decision tree to files,
        '''
        if filename == None:
            filename = 'tree_{}.json'.format(time.strftime("%c"))
        
        f = open(filename, 'w')
        result = self._node_to_dict(tree)
        json.dump(result, f, indent=2)
        f.close()
    
    def load_tree(self, filename):
        '''
        Read decision tree from file
        '''
        f = open(filename, 'r')
        dict_ = json.load(f)
        f.close()
        root = self._dict_to_node(dict_)
        return root
        
    def load_trees(self, path='.', prefix='tree'):
        '''
        Read set of decision tree from set of files that match prefix
        '''
        # fectch file list in target folder
        treefiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.startswith(prefix)]
        roots = []
        for f in treefiles:
            roots.append(self.load_tree(os.path.join(path, f)))
        
        return roots
    
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