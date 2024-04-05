import inspect
import numpy as np
import pandas as pd
import networkx as nx
import graphviz
from itertools import combinations, chain
import itertools
import matplotlib.pyplot as plt
from scipy.stats import poisson
import random
import pickle
import networkx as nx
import scipy
from itertools import combinations, permutations
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from typing import Any, Callable, Dict, List, Tuple, Union, Iterable, Optional
import collections

from cgm import *

#Helper functions to create motifs
def sample_categorical(probabilities_matrix, choices_list):
    assert len(choices_list)==probabilities_matrix.shape[0]
    sample_idxs = torch.multinomial(torch.Tensor(probabilities_matrix.T),1)
    sample_idxs.numpy()
    np_choices = np.array(choices_list)

    # Create the new matrix
    samples = np_choices[sample_idxs]
    # samples = [np.random.choice(choices_list, p=column) for column in probabilities_matrix.T]
    return np.array(samples).flatten()

def softmax(x,axis=0):
    # Subtracting the maximum value for numerical stability
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps,axis)

def _sigma(x):
    return 1 / (1 + np.exp(-x))

def get_pair_connections(N, K):
    '''
    Generates a chain and then all possible pairs (upto K) (i,j) where i<j to build into a DAG (randomly ordered)
    '''
    # The graph must be connected, so start by connecting each node to the next one.
    # Make a chain
    pairs = [(i, i + 1) for i in range(N - 1)]

    # If K is less than N-1, return the pairs up to K
    if K <= N - 1:
        return pairs[:K]

    # Generate all remaining pairs (i, j) where i < j
    remaining_pairs = list(itertools.combinations(range(N), 2))
    random.shuffle(remaining_pairs)

    # Remove the pairs already added
    remaining_pairs = [pair for pair in remaining_pairs if pair not in pairs]

    # Add pairs from remaining_pairs until we have K pairs
    pairs.extend(remaining_pairs[:K - len(pairs)])

    return pairs

def generate_random_tril_wo_diagonal(n):
    matrix = np.random.randn(n, n)
    matrix[np.tril_indices(n)] = 0  # Set lower triangle elements to 0
    np.fill_diagonal(matrix, 0)     # Set diagonal elements to 0
    return matrix

def create_upper_triangular_mask(n,p=0.5,**kwargs):
    matrix = np.random.choice(a= [0,1], p = [p,1-p], size=(n, n))  # Generate a random matrix of 0s and 1s
    upper_triangular_mask = np.triu(matrix)  # Extract the upper triangular part
    upper_triangular_mask = upper_triangular_mask - np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask

def normalize_rows(A):
    # Count non-zero elements in each row
    non_zero_count = np.count_nonzero(A, axis=1)

    # Compute 1/sqrt(count) for each row, avoiding division by zero
    norm_vector = np.where(non_zero_count > 0, 1 / np.sqrt(non_zero_count), 1)

    # Apply the normalization to each row of the matrix
    # np.newaxis is used to transform norm_vector from 1D to 2D (making it a column vector)
    # This allows the division to be broadcast correctly across rows
    normalized_A = A*norm_vector[:, np.newaxis]
    return normalized_A

def sample_from_poisson(M, mean, atleast=3):
    distribution = poisson(mu=mean)
    samples = distribution.rvs(size=M)
    valid_samples = np.where(samples > atleast, samples, atleast)
    return valid_samples

## Helper functions for creating a random DAG

def generate_random_edges(n_vars: int, n_edges: int):
    """
    Generate a random set of edges
    """
    all_pairs = [(i, j) for i in range(n_vars) for j in range(n_vars) if i < j]
    edges = [all_pairs[i] for i in np.random.choice(len(all_pairs), n_edges, replace=False)]
    # randomly flip half the edges
    edges = [(edge[1], edge[0]) if np.random.randint(0, 2) == 1 else edge for edge in edges]
    return edges

def contains_cycle(edges):
    """
    Check if the set of edges contain a cycle
    edges: list of tuples
    return: True or false
    """

    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    try:
        nx.find_cycle(graph)
        return True
    except nx.NetworkXNoCycle:
        return False

def calculate_and_plot_degree_distributions(G):
    '''
    Calculate in and out degree distributions for a networkX directed graph
    '''

    # Calculate in-degrees and out-degrees
    in_degrees = G.in_degree()  # Dict with Node: InDegree
    out_degrees = G.out_degree()  # Dict with Node: OutDegree

    # Convert them to a list of degrees
    in_degree_values = [val for key, val in in_degrees]
    out_degree_values = [val for key, val in out_degrees]

    # Compute the frequency of each degree value
    in_degree_hist = collections.Counter(in_degree_values)
    out_degree_hist = collections.Counter(out_degree_values)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(2, 1, sharex=True)

    # Plot in-degree distribution
    axs[0].bar(in_degree_hist.keys(), in_degree_hist.values(), color='b')
    axs[0].set_title('In-degree distribution')

    # Plot out-degree distribution
    axs[1].bar(out_degree_hist.keys(), out_degree_hist.values(), color='r')
    axs[1].set_title('Out-degree distribution')

    plt.tight_layout()
    plt.show()

def get_simple_paths_lengths(G: nx.DiGraph):
  path_dict = {}
  path_length_dict = {}
  # Finding all simple paths in the DAG
  for source in G.nodes:
      for target in G.nodes:
          if source != target:
              for path in nx.all_simple_paths(G, source, target, cutoff=None):
                  #if there already exists a path, append this new one to it
                  if(path_dict.get((source,target)) is not None):
                    path_dict[(source,target)].append(list(path))
                    path_length_dict[(source,target)].append(len(path))
                  #otherwise create a new dictionary item
                  else:
                    path_dict[(source,target)] = [list(path)]
                    path_length_dict[(source,target)] = [len(path)]

  return path_dict, path_length_dict

def forms_a_path(G, nodes):
    """
    Given a graph G and a list of nodes, this function checks if the nodes form a simple path in G.
    """
    for i in range(len(nodes) - 1):
        if not G.has_edge(nodes[i], nodes[i+1]):
            return False

    # Ensure no repeated nodes for simple path
    if len(nodes) != len(set(nodes)):
        return False

    return True

def get_named_edges(node_names, A):
    if len(node_names) != A.shape[0] or len(node_names) != A.shape[1]:
        raise ValueError("Node names length and matrix dimensions do not match!")

    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]

    return edge_list

def get_non_edges(G):
  # Get all possible pairs of nodes
    all_pairs = list(combinations(G.nodes(), 2))

    # Get pairs that are not edges
    non_edges = [pair for pair in all_pairs if not G.has_edge(*pair)]

    return non_edges

def get_non_path_connected(G):
    all_pairs = list(combinations(G.nodes(), 2))

    non_connected_pairs = [pair for pair in all_pairs if not nx.has_path(G, pair[0], pair[1])]

    return non_connected_pairs

def check_true_edge_accuracy(G, edge_list):
    edge_bools = []
    for i in range(0, len(edge_list)):
        edge_bools.append(G.has_edge(*edge_list[i]))
    return np.array(edge_bools)

def get_edges(node_sequence):
  '''
  Takes in whole node sequence including target at position 0
  '''

  edge_list = []
  for i in range(0, len(node_sequence)-1):
      edge_list.append((node_sequence[i], node_sequence[i + 1]))

  does_end_at_target = (node_sequence[0]==node_sequence[-1])
  does_contain_target = node_sequence[0] in node_sequence[1:]
  return edge_list, does_end_at_target, does_contain_target

def remove_element(lst, element_to_remove):
    return [x for x in lst if x != element_to_remove]

from itertools import groupby

def split_list(lst, delimiter):
    return [list(group) for key, group in groupby(lst, lambda x: x == delimiter) if not key]

def generate_all_ordered_subsets_of_size(arr, k):
    # Generate all subsets of size k
    subsets = combinations(arr, k)

    # For each subset, generate all possible permutations
    permuted_subsets = [list(permutation) for subset in subsets for permutation in permutations(subset, k)]

    return permuted_subsets

def get_non_path_connected(G):
    all_pairs = [(i, j) for i in list(G.nodes) for j in list(G.nodes) if i != j]
    non_connected_pairs = [pair for pair in all_pairs if not nx.has_path(G, pair[0], pair[1])]
    return non_connected_pairs


def get_non_edges(G):
  # Get all possible pairs of nodes
    all_pairs = [(i, j) for i in list(G.nodes) for j in list(G.nodes) if i != j]

    # Get pairs that are not edges
    non_edges = [pair for pair in all_pairs if not G.has_edge(*pair)]

    return non_edges

def get_named_edges(node_names, A):
    if len(node_names) != A.shape[0] or len(node_names) != A.shape[1]:
        raise ValueError("Node names length and matrix dimensions do not match!")

    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]

    return edge_list

def create_upper_triangular_mask(n,p=0.5,**kwargs):
    matrix = np.random.choice(a= [0,1], p = [p,1-p], size=(n, n))  # Generate a random matrix of 0s and 1s
    upper_triangular_mask = np.triu(matrix)  # Extract the upper triangular part
    upper_triangular_mask = upper_triangular_mask - np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask

def create_random_DAG(num_nodes, p):

  node_names = ['X' + str(i) for i in range(num_nodes)]
  # generate a random DAG structure
  A = create_upper_triangular_mask(num_nodes, p=p)
  edge_list = get_named_edges(node_names, A)
  graph_cgm = CausalGraphicalModel(node_names, edge_list)

  while(nx.is_connected(graph_cgm.graph) == False or len(graph_cgm.no_children)<3 or len(graph_cgm.no_parents)<3):
    A = create_upper_triangular_mask(num_nodes, p=p)
    edge_list = get_named_edges(node_names, A)
    graph_cgm = CausalGraphicalModel(node_names, edge_list)

  path_dict, path_length_dict = get_simple_paths_lengths(graph_cgm.dag)
  graph_dag = graph_cgm.dag

  connected_node_pairs_dict = [key for key, value in path_length_dict.items() if any(path_value > 0 for path_value in value)]

  return graph_cgm, graph_dag, path_dict, path_length_dict

def get_non_path_connected(G):
    all_pairs = [(i, j) for i in list(G.nodes) for j in list(G.nodes) if i != j]

    non_connected_pairs = [pair for pair in all_pairs if not nx.has_path(G, pair[0], pair[1])]

    return non_connected_pairs


def create_layered_dag(layers, block_probability=1):
    """
    Creates a layered DAG and returns its adjacency matrix.

    layers: a list of integers where each integer represents the number of nodes in that layer.
    block_probability: a probability that a block in the upper triangular matrix has an edge.
    """
    G = nx.DiGraph()

    # Add nodes to the graph
    total_nodes = sum(layers)
    for i in range(total_nodes):
        G.add_node(i)

    # Create edges based on the blocks in the upper triangular matrix
    for i in range(len(layers) - 1):  # We don't need to create edges for the last layer
        start_node_current_layer = sum(layers[:i])
        end_node_current_layer = sum(layers[:i+1])

        start_node_next_layer = end_node_current_layer
        end_node_next_layer = start_node_next_layer + layers[i+1]

        for j in range(start_node_current_layer, end_node_current_layer):
            for k in range(start_node_next_layer, end_node_next_layer):
                if np.random.rand() <= block_probability:
                    G.add_edge(j, k)

    # Create adjacency matrix from the graph
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    return adjacency_matrix, G

def create_layered_DAG_from_adjacency(num_nodes, adjacency_matrix):

  node_names = ['X' + str(i) for i in range(num_nodes)]
  # generate a random DAG structure
  A = adjacency_matrix
  edge_list = get_named_edges(node_names, A)
  graph_cgm = CausalGraphicalModel(node_names, edge_list)
  graph_dag = graph_cgm.dag

  return graph_cgm, graph_dag

def create_named_binary_tree(node_names):
    """Create a named binary tree with given node names."""
    n = len(node_names)
    G = nx.DiGraph()
    for i in range(n):
        if 2 * i + 1 < n:
            G.add_edge(node_names[i], node_names[2 * i + 1])
        if 2 * i + 2 < n:
            G.add_edge(node_names[i], node_names[2 * i + 2])
    return G

def create_tree_like_dag(num_nodes):
    """Generate the adjacency matrix and lists of node names at each depth level."""
    node_names = ['X' + str(i) for i in range(num_nodes)]
    G = create_named_binary_tree(node_names)
    n = len(node_names)
    partitioned_node_names = [[] for _ in range(int(np.log2(n)) + 1)]
    for i, node in enumerate(node_names):
        depth = int(np.log2(i + 1))
        partitioned_node_names[depth].append(node)
    adj_matrix = nx.to_numpy_array(G)
    return adj_matrix, partitioned_node_names, G


if name == 'main':
    num_nodes = 10
    p = 0.5
    # Have at least one intermediate node to define it as a path
    path_length_threshold = 2
    #Fraction of held out node pairs
    frac = 0.2
    graph_type='random'

    if graph_type=='random':

        graph_cgm, graph_dag, path_dict_no_prompt, path_length_dict_no_prompt = create_random_DAG(num_nodes=num_nodes, p=p)
    
    path_dict_cot = {}
    path_dict_direct = {}
    path_length_dict = {}
    non_path_dict = {}
    # Finding all simple paths in the DAG
    for source in graph_dag.nodes:
        for target in graph_dag.nodes:
            if source != target:
            if(nx.has_path(graph_dag, source, target)):
                for path in nx.all_simple_paths(graph_dag, source, target, cutoff=None):
                    #if there already exists a path, append this new one to it
                    if(path_dict_cot.get((source,target)) is not None):
                    path.insert(0, list(path)[-1])
                    path.insert(0, 'target')
                    path.append('path')
                    path.append('###')
                    path_dict_cot[(source,target)].append(list(path))
                    path_length_dict[(source,target)].append(len(path)-4)

                    #otherwise create a new dictionary item
                    else:
                    path_dict_direct[(source,target)] = ['target', target, source, 'path', '###']
                    path.insert(0, list(path)[-1])
                    path.insert(0, 'target')
                    path.append('path')
                    path.append('###')
                    path_dict_cot[(source,target)] = [list(path)]
                    path_length_dict[(source,target)] = [len(path)-4]
            else:
                non_path_dict[(source,target)] = ['target', target, source, 'nopath', '###']

    #Make shortest
    for key in path_dict_cot:
        #path_dict_direct[key].sort(key=len)
        path_dict_cot[key].sort(key=len)
        path_length_dict[key].sort()

    edge_pairs = [key for key, value in path_length_dict.items() if any(path_value == path_length_threshold for path_value in value)]
    print(len(edge_pairs))
    node_pairs_exceeding_threshold = [key for key, value in path_length_dict.items() if any(path_value > path_length_threshold for path_value in value)]
    print(len(node_pairs_exceeding_threshold))

    print('Sanity check edge pairs and node pairs exceeding threshold')
    print(len([node_pair for node_pair in node_pairs_exceeding_threshold if node_pair in edge_pairs]))

    num_paths = int(frac*len(node_pairs_exceeding_threshold))

    chosen_node_pairs = random.sample(node_pairs_exceeding_threshold, num_paths)
    print('Number of chosen node pairs')
    print(len(chosen_node_pairs))

    held_out_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in chosen_node_pairs]

    non_path_node_pairs = get_non_path_connected(graph_cgm.dag)
    print(len(non_path_node_pairs))

    num_non_paths = int(frac*len(non_path_node_pairs))

    chosen_non_path_node_pairs = random.sample(non_path_node_pairs, num_non_paths)
    print(len(chosen_non_path_node_pairs))

    held_out_non_path_node_pairs = [t for t in non_path_node_pairs if t not in chosen_non_path_node_pairs]


    ## Sanity check
    print('Sanity checks (should be empty list output for overlap between chosen and held out):')
    print([node_pair for node_pair in chosen_node_pairs if node_pair in held_out_node_pairs])
    print([node_pair for node_pair in chosen_non_path_node_pairs if node_pair in held_out_non_path_node_pairs])


