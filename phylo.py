# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:55:35 2016

@author: sshank
"""

import numpy as np
from scipy.linalg import expm


def L0(data, tree, Q):
    extant_taxa = data.shape[0]
    sequence_length = data.shape[1]
    u = Q.shape[0]
    L = np.zeros((u, sequence_length, 2*extant_taxa-1))
    all_sequence_indices = np.arange(sequence_length)
    for node in tree.traverse('postorder'):
        index = int(node.name)
        sequence = data[index, :]
        if node.is_leaf():
            L[sequence, all_sequence_indices, index] = 1
        else:
            left_child, right_child = node.children
            left_index = int(left_child.name)
            right_index = int(right_child.name)
            L_left = L[:, :, left_index]
            L_right = L[:, :, right_index]
            t_left = left_child.dist
            t_right = right_child.dist
            P_left = expm(t_left*Q)
            P_right = expm(t_right*Q)
            L[:, :, index] = np.dot(P_left, L_left) * np.dot(P_right, L_right)
        if node.is_root():
            return L[:,:,0]


def prune(data, tree, Q, frequencies):
    L = L0(data, tree, Q)
    u = Q.shape[0]
    l0 = np.dot(frequencies.reshape(1, u), L)
    return np.sum(np.log(l0))


def f81(pi):
    piA = pi[0]
    piG = pi[1]
    piC = pi[2]
    piT = pi[3]
    beta = 1-np.sum(pi**2)
    Q = np.array([[0,   piG, piC, piT],
                  [piA, 0,   piC, piT],
                  [piA, piG, 0,   piT],
                  [piA, piG, piC, 0  ]])
    np.fill_diagonal(Q, -np.sum(Q, 1))
    Q /= beta
    return Q
    