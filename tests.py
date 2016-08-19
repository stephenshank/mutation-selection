# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:46:11 2016

@author: sshank
"""

import unittest
import numpy as np
from data import *
from phylo import *

np.random.seed(1)

class TestSequence(unittest.TestCase):
    def test_instantiate_and_mutate(self):
        pi = np.array([.1, .2, .3, .4])
        Q = f81(pi)
        sequence = random_sequence(100, pi)
        new_sequence = mutate_sequence(sequence, Q, .5)
        self.assertTrue(not all(new_sequence == sequence))


class TestTree(unittest.TestCase):
    def test_instantiation_generation_and_pruning(self):
        pi = np.array([.1, .2, .3, .4])
        Q = f81(pi)
        sequence = random_sequence(20, pi)
        extant_taxa = 20
        tree = phylogenetic_tree(extant_taxa)
        data = generate_data(sequence, Q, tree)
        prune(data, tree, Q, pi)