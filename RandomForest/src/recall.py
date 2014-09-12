"""
This contain class that provide a way to recall
information from dicision trees.

Created on Fri Aug 08 16:01:00 2014

@author: Krerkkiat
"""
import os
import os.path
import json

import numpy as np

from master import MasterNode

tree_root_folder = 'tree'

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Recall
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class Recall:
    '''
    This provide a way to ask Random Forest or recall
    information from dicision trees.
    '''

    def __init__(self):
        '''
        Init routine.
        '''
        pass

    def load_forest(self, filename):
        '''
        This will load main JSON file of trainning.

        Parameter(s):
            filename: str - File name of main JSON file of trainning.
        Return(s):
            roots: MasterNode[] - Root of dicision trees that main file refer to.
        '''

        # this will open main file of tree
        roots = []
        with open(filename, 'r') as f:
            tree_information = json.load(f)
            for tree in tree_information['file_list']:
                roots.append(self.load_tree(os.path.join(tree_root_folder, tree)))

        return roots

    def load_tree(self, filename):
        '''
        This will load individual dicision tree file.

        Parameter(s):
            filename: str - File name of individual dicision tree.
        Return(s):
            root: MasterNode - Root of dicision tree.
        '''

        with open(filename, 'r') as f:
            dict_ = json.load(f)
            return self._dict_to_node(dict_['tree'])

    def _dict_to_node(self, dict_):
        '''
        Helper method that help load dicision tree.

        Parameter(s):
            dict_: dict - Dict type data that loaded from JSON file.
        Return(s):
            root: MasterNode - MasterNode type of data.
        '''
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
        '''
        This use for get result from
        multiple dicision tree for feature list I

        Parameter(s):
            roots: MasterNode[] - List of tree (MasterNode).
            I: numpy.array - List of feature to be classification.
        Return(s):
            prop: numpy.array - Probability of classes that I will be.
        '''

        props = []
        for root in roots:
            props.append(self._get_result(root, I))
        props = np.array(props)

        return np.sum(props, axis=0)/len(roots)

    def get_result(self, tree, I):
        '''
        This use for get result from tree:MasterNode
        (dicision tree) for feature list I

        Parameter(s):
            tree:MasterNode - Tree (MasterNode) object.
            I: numpy.array - List of feature to be classification.
        Return(s):
            prop: numpy.array - Probability of classes that I will be.
        '''
        return self._get_result(tree, I)

    def _get_result(self, node, I):
        '''
        This use for get result from dicision tree
        for feature list I (recursive)

        Parameter(s):
            node: MasterNode - Tree (MasterNode) object.
            I: numpy.array - List of feature to be classification.
        Return(s):
            prop: numpy.array - Probability of classes that I will be.
        '''
        if node.prop != None:
            return node.prop

        tau = I[node.theta]

        if node.left and tau < node.tau:
            return self._get_result(node.left, I)
        if node.right and tau >= node.tau:
            return self._get_result(node.right, I)

        return -1  # for unknow
