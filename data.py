# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:08:51 2016

@author: sshank
"""

import numpy as np
from Bio.Seq import Seq
from scipy.linalg import expm
from ete3 import PhyloTree
from itertools import product


nucleotides_list = ('A', 'G', 'C', 'T')
nucleotides = np.array(nucleotides_list, dtype='<U1')

to_string = lambda X: ''.join(X)
triplets = list(product(*3*(nucleotides_list,)))
codons = [to_string(triplet) for triplet in triplets]
codons.remove('TAA')
codons.remove('TAG')
codons.remove('TGA')
codons = np.array(codons, dtype='<U3')
codon_string = lambda X: to_string(codons[X])

translate = lambda codon: str(Seq(codon).translate())
amino_acids = list(set([translate(codon) for codon in codons]))
amino_acids.sort()
amino_acids = np.array(amino_acids, dtype='<U1')
amino_acid_string = lambda X: to_string(amino_acids[X])

def random_sequence(sequence_length, distribution):
    number_of_units = len(distribution)
    sequence = np.random.choice(number_of_units,
                                size=sequence_length,
                                p=distribution)
    return sequence


def mutate_sequence(sequence, Q, l):
    sequence_length = len(sequence)
    number_of_units = Q.shape[0]
    P = expm(l*Q)
    new_sequence = np.empty(sequence_length, dtype=np.int)
    all_indices = np.arange(sequence_length)
    for i in range(number_of_units):
        indices = all_indices[sequence==i]
        count = len(indices)
        transition_probabilities = P[i, :]
        finite_precision_errors = transition_probabilities < 0
        transition_probabilities[finite_precision_errors] = 0
        transition_probabilities /= np.sum(transition_probabilities)
        new_nucleotides = np.random.choice(number_of_units,
                                           count,
                                           p=transition_probabilities)
        new_sequence[indices] = new_nucleotides
    return new_sequence


def phylogenetic_tree(extant_taxa, branch_lengths=(.01, .5)):
    names_as_strings = [str(i) for i in range(extant_taxa)]
    tree = PhyloTree(format=1)
    tree.populate(extant_taxa, random_branches=branch_lengths, names_library=names_as_strings)
    tree.dist = 0
    current_name = 2*extant_taxa-2
    for node in tree.traverse('preorder'):
        if not node.is_leaf():
            node.name = str(current_name)
            current_name -= 1
    return tree


def generate_data(initial_sequence, Q, tree):
    extant_taxa = len(tree.get_leaves())
    sequence_length = len(initial_sequence)
    data = np.empty((2*extant_taxa-1, sequence_length), dtype=np.int)
    for node in tree.traverse('preorder'):
        index = int(node.name)
        if node.is_root():
            data[index, :] = initial_sequence
        else:
            parent_index = int(node.up.name)
            parent_sequence = data[parent_index, :]
            l = node.dist
            new_sequence = mutate_sequence(parent_sequence, Q, l)
            data[index, :] = new_sequence
    return data
