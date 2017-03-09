import numpy as np
import scipy.optimize as spop
from scipy.linalg import expm
import matplotlib.pyplot as plt

from data import phylogenetic_tree, generate_data, degeneracy
from data import random_sequence, codon_index_to_amino_acid_index
from phylo import jc69
from mutsel import mutation_selection, frequencies_from_fitnesses

seed = 1
np.random.seed(seed)

N = 10000
mu = 1e-7
extant_taxa = 12
n_branches = 2*extant_taxa-2
sequence_length = 1500
fitnesses = np.linspace(1, 1+1/N, 20)
Qmu = jc69(mu)
frequencies = frequencies_from_fitnesses(N, fitnesses)
Q = mutation_selection(Qmu, frequencies)
codon_frequencies = frequencies[codon_index_to_amino_acid_index]
tree = phylogenetic_tree(extant_taxa)
bls = np.zeros(n_branches)
for node in tree.traverse('postorder'):
    if not node.is_root():
        index = int(node.name)
        bls[index] = node.dist

sequence = random_sequence(sequence_length, codon_frequencies)
data = generate_data(sequence, Q, tree)
sumtoone = lambda x: np.dot(x[:20], degeneracy)-1
bounds = 20*((1e-12, 1), ) + n_branches*((1e-12, 2), )
constraints = {'type':'eq', 'fun':sumtoone}

freq0 = np.ones(20)
freq0 /= np.dot(np.ones(20), degeneracy)
t0 = .2*np.ones(n_branches)
x0 = np.hstack([freq0, t0])

def negative_likelihood_eq(x):
    frequencies = x[:20]
    t = x[20:]
    Q = mutation_selection(Qmu, frequencies)
    u = Q.shape[0]
    taxa = data.shape[0]
    L = np.zeros((61, sequence_length, taxa))
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
            t_left = t[left_index]
            t_right = t[right_index]
            try:
                P_left = expm(t_left*Q)
                P_right = expm(t_right*Q)
            except ValueError:
                raise Exception('Could not compute matrix exponentials!')
            L[:, :, index] = np.dot(P_left, L_left) * np.dot(P_right, L_right)
    u = Q.shape[0]
    codon_frequencies = frequencies[codon_index_to_amino_acid_index]
    l0 = np.dot(codon_frequencies.reshape(1, u), L[:, :, index])
    return -np.sum(np.log(l0))


result = spop.minimize(negative_likelihood_eq,
                       x0,
                       bounds=bounds,
                       constraints=constraints,
                       tol=1e-3,
                       options={'maxiter': 1000})

plt.scatter(result.x[:20], frequencies, color='red', label='Sub params')
plt.scatter(result.x[20:], bls, color='blue', label='Branch lengths')
p = [result.x.min(), result.x.max()]
plt.plot(p, p, color='black', label='Truth')
plt.legend(loc='upper left')
plt.show()
plt.title('Equilibrium branch length identifiability')
print(result)

np.random.seed(seed)
noneq_fitnesses = np.linspace(1+1/N, 1, 20)
noneq_frequencies = frequencies_from_fitnesses(N, noneq_fitnesses)
noneq_codon_frequencies = noneq_frequencies[codon_index_to_amino_acid_index]
noneq_sequence = random_sequence(sequence_length, noneq_codon_frequencies)
noneq_data = generate_data(noneq_sequence, Q, tree)

sumtoone1 = lambda x: np.dot(x[:20], degeneracy)-1
sumtoone2 = lambda x: np.dot(x[20:40], degeneracy)-1
bounds = 40*((1e-12, 1), ) + n_branches*((1e-12, 2), )
constraints = [{'type':'eq', 'fun':sumtoone1}, {'type':'eq', 'fun':sumtoone2}]

freq0 = np.ones(40)
t0 = .2*np.ones(n_branches)

freq0[:20] /= np.dot(freq0[:20], degeneracy)
freq0[20:40] /= np.dot(freq0[20:40], degeneracy)
t0 = .2*np.ones(n_branches)

x0 = np.hstack([freq0, t0])

def negative_likelihood_noneq(x):
    branch_frequencies = x[:20]
    root_frequencies = x[20:40]
    t = x[40:]
    Q = mutation_selection(Qmu, branch_frequencies)
    u = Q.shape[0]
    taxa = noneq_data.shape[0]
    L = np.zeros((61, sequence_length, taxa))
    all_sequence_indices = np.arange(sequence_length)
    for node in tree.traverse('postorder'):
        index = int(node.name)
        if node.is_leaf():
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
    root_codon_frequencies = root_frequencies[codon_index_to_amino_acid_index]
    l0 = np.dot(root_codon_frequencies.reshape(1, u), L[:, :, index])
    return -np.sum(np.log(l0))


result = spop.minimize(negative_likelihood_noneq,
                       x0,
                       bounds=bounds,
                       constraints=constraints,
                       tol=1e-3,
                       options={'maxiter': 1000})

plt.scatter(result.x[:40], np.hstack([frequencies, noneq_frequencies]), color='red', label='Sub params')
plt.scatter(result.x[40:], bls, color='blue', label='Branch lengths')
p = [result.x.min(), result.x.max()]
plt.plot(p, p, color='black', label='Truth')
plt.legend(loc='upper left')
plt.show()
print(result)