# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:35:24 2016

@author: sshank
"""

import numpy as np
from data import *

def compute_alpha(N, fitnesses):
    products = degeneracy*fitnesses**(2*N)
    alpha = np.sum(products)**(-1/(2*N))
    return alpha

def frequencies_from_fitnesses(N, fitnesses):
    mean_of_fitnesses = np.mean(fitnesses)
    fitnesses /= mean_of_fitnesses
    alpha = compute_alpha(N, fitnesses)
    frequencies = (alpha*fitnesses)**(2*N)
    return frequencies

def fitnesses_from_frequencies(N, frequencies, alpha):
    fitnesses = frequencies**(1/(2*N))/alpha
    return fitnesses

def selection_coefficients_from_fitnesses(N, fitnesses):
    fvec = fitnesses.reshape(-1, 1)
    smat =  np.dot(fvec, 1/fvec.T)-1.
    S = 2*N*smat.flatten()
    return S

def mutation_selection(Qmu, frequencies):
    Q = np.zeros((61, 61))
    codon_frequencies = frequencies[codon_index_to_amino_acid_index]
    # Upper case letters index codons, lower case index amino acids
    for I, codon_I in enumerate(codons):
        i = codon_index_to_amino_acid_index[I]
        for J, codon_J in enumerate(codons):
            j = codon_index_to_amino_acid_index[J]
            # Selection
            if I != J:
                current_absolute_error = np.abs(frequencies[i]-frequencies[j])
                if current_absolute_error < 1e-10:
                    Q[I, J] = 1
                else:
                    numerator = np.log(frequencies[j]/frequencies[i])
                    denominator = (1-frequencies[i]/frequencies[j])
                    Q[I, J] = numerator/denominator
            # Mutation
            for k in range(3):
                nucleotide_i = codon_I[k]
                nucleotide_j = codon_J[k]
                if nucleotide_i != nucleotide_j:
                    i_n = nucleotide_to_index[nucleotide_i]
                    j_n = nucleotide_to_index[nucleotide_j]
                    Q[I, J] *= Qmu[i_n, j_n]
        Q[I, I] = -np.sum(Q[I,:])
    beta = -np.sum(codon_frequencies*np.diag(Q))
    Q /= beta
    if any(np.isnan(Q.flatten())):
        raise ValueError('Nans encountered in mutsel matrix!!!!')
    return Q