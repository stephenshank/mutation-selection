#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:16:38 2017

@author: sshank
"""

import pickle
import sys

import numpy as np
import scipy.optimize as spop
from scipy.linalg import expm

from data import phylogenetic_tree, generate_data, degeneracy
from data import random_sequence, codon_index_to_amino_acid_index
from phylo import jc69
from mutsel import mutation_selection, frequencies_from_fitnesses


def run_experiment(seed):
    np.random.seed(seed)
    
    print('Generating data...')
    # Common parameters
    N = 10000
    mu = 1e-7
    extant_taxa = 12
    n_branches = 2*extant_taxa-2
    sequence_length = 500
    fitnesses = np.linspace(1, 1+1/N, 20)
    Qmu = jc69(mu)
    frequencies = frequencies_from_fitnesses(N, fitnesses)
    Q = mutation_selection(Qmu, frequencies)
    eq_codon_frequencies = frequencies[codon_index_to_amino_acid_index]
    tree = phylogenetic_tree(extant_taxa)
    bls = np.zeros(n_branches)
    for node in tree.traverse('postorder'):
        if not node.is_root():
            index = int(node.name)
            bls[index] = node.dist
    
    # Equilibrium data
    eq_sequence = random_sequence(sequence_length, eq_codon_frequencies)
    eq_data = generate_data(eq_sequence, Q, tree)
    sumtoone = lambda x: np.dot(x[:20], degeneracy)-1
    eq_bounds = 20*((1e-12, 1), ) + n_branches*((1e-12, 2), )
    eq_constraints = {'type':'eq', 'fun':sumtoone}
    eq_freq0 = np.ones(20)
    eq_freq0 /= np.dot(np.ones(20), degeneracy)
    eq_t0 = .2*np.ones(n_branches)
    eq_x0 = np.hstack([eq_freq0, eq_t0])
    
    # Non-equilibrium data
    noneq_fitnesses = np.linspace(1+1/N, 1, 20)
    noneq_frequencies = frequencies_from_fitnesses(N, noneq_fitnesses)
    noneq_codon_frequencies = noneq_frequencies[codon_index_to_amino_acid_index]
    noneq_sequence = random_sequence(sequence_length, noneq_codon_frequencies)
    noneq_data = generate_data(noneq_sequence, Q, tree)
    sumtoone1 = lambda x: np.dot(x[:20], degeneracy)-1
    sumtoone2 = lambda x: np.dot(x[20:40], degeneracy)-1
    noneq_bounds = 40*((1e-12, 1), ) + n_branches*((1e-12, 2), )
    noneq_constraints = [{'type':'eq', 'fun':sumtoone1}, {'type':'eq', 'fun':sumtoone2}]    
    noneq_freq0 = np.ones(40)
    noneq_t0 = .2*np.ones(n_branches)    
    noneq_freq0[:20] /= np.dot(noneq_freq0[:20], degeneracy)
    noneq_freq0[20:40] /= np.dot(noneq_freq0[20:40], degeneracy)    
    noneq_x0 = np.hstack([noneq_freq0, noneq_t0])
    
    
    def negative_likelihood(x, true_model='eq', assumed_model='eq'):
        assert true_model in ['eq', 'noneq']
        assert assumed_model in ['eq', 'noneq']

        branch_frequencies = x[:20]
        if assumed_model == 'eq':
            t = x[20:]
            root_codon_frequencies = branch_frequencies[codon_index_to_amino_acid_index]
        else:
            root_frequencies = x[20:40]
            t = x[40:]
            root_codon_frequencies = root_frequencies[codon_index_to_amino_acid_index]
        
        Q = mutation_selection(Qmu, branch_frequencies)
        u = Q.shape[0]
        taxa = noneq_data.shape[0]
        L = np.zeros((61, sequence_length, taxa))
        all_sequence_indices = np.arange(sequence_length)
        for node in tree.traverse('postorder'):
            index = int(node.name)
            if node.is_leaf():
                if true_model == 'eq':
                    sequence = eq_data[index, :]
                else:
                    sequence = noneq_data[index, :]
                L[sequence, all_sequence_indices, index] = 1
            else:
                left_child, right_child = node.children
                left_index = int(left_child.name)
                right_index = int(right_child.name)
                L_left = L[:, :, left_index]
                L_right = L[:, :, right_index]
                t_left = t[left_index]
                t_right = t[right_index]
                try:
                    P_left = expm(t_left*Q)
                    P_right = expm(t_right*Q)
                except ValueError:
                    raise Exception('Could not compute matrix exponentials!')
                L[:, :, index] = np.dot(P_left, L_left) * np.dot(P_right, L_right)
        u = Q.shape[0]
        l0 = np.dot(root_codon_frequencies.reshape(1, u), L[:, :, index])
        return -np.sum(np.log(l0))

    # True/assumed model
    print('Fitting eq/eq model...')
    eq_eq_result = spop.minimize(
        lambda x: negative_likelihood(x, true_model='eq', assumed_model='eq'),
        eq_x0,
        bounds=eq_bounds,
        constraints=eq_constraints,
        tol=1e-4,
        options={'maxiter': 1000}
    )
    
    print('Fitting eq/noneq model...')
    eq_noneq_result = spop.minimize(
        lambda x: negative_likelihood(x, true_model='eq', assumed_model='noneq'),
        noneq_x0,
        bounds=noneq_bounds,
        constraints=noneq_constraints,
        tol=1e-4,
        options={'maxiter': 1000}
    )
    
    print('Fitting noneq/eq model...')
    noneq_eq_result = spop.minimize(
        lambda x: negative_likelihood(x, true_model='noneq', assumed_model='eq'),
        eq_x0,
        bounds=eq_bounds,
        constraints=eq_constraints,
        tol=1e-4,
        options={'maxiter': 1000}
    )
    
    print('Fitting noneq/noneq model...')
    noneq_noneq_result = spop.minimize(
        lambda x: negative_likelihood(x, true_model='noneq', assumed_model='noneq'),
        noneq_x0,
        bounds=noneq_bounds,
        constraints=noneq_constraints,
        tol=1e-4,
        options={'maxiter': 1000}
    )
    
    data = {
        'frequencies': frequencies,
        'noneq_frequencies': noneq_frequencies,
        'bls': bls,
        'eq_eq_result': eq_eq_result,
        'eq_noneq_result': eq_noneq_result,
        'noneq_eq_result': noneq_eq_result,
        'noneq_noneq_result': noneq_noneq_result,
    }
    
    pickle_filename = 'experiments/exp_%d.pkl' % seed
    with open(pickle_filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)


if __name__ == '__main__':
    lower = int(sys.argv[1])
    upper = int(sys.argv[2])
    for i in range(lower, upper):
        run_experiment(i)
