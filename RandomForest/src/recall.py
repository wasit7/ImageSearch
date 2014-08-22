# -*- coding: utf-8 -*-
"""
Created on Fri Aug 08 16:01:00 2014

@author: Krerkait
"""
import os
import json

import numpy as np

from master import MasterNode

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Recall
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Recall:
    def __init__(self):
        pass
    
    def load_forest(self, filename):
        # this will open main file of tree
        roots = []
        with open(filename, 'r') as f:
            tree_information = json.load(f)
            for tree in tree_information['file_list']:
                roots.append(self.load_tree(tree))

        return roots

    def load_tree(self, filename):
        with open(filename, 'r') as f:
            dict_ = json.load(f)
            return self._dict_to_node(dict_['tree'])

    def _load_tree(self, filename):
        '''
        Read decision tree from file
        '''
        f = open(filename, 'r')
        dict_ = json.load(f)
        f.close()
        root = self._dict_to_node(dict_)
        print('{} loaded'.format(filename))
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