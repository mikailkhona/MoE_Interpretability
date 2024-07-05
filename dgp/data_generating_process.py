import numpy as np
import networkx as nx
import pickle
import networkx as nx
import random

from cgm import *

#Helper functions to create motifs
def create_upper_triangular_mask(n,p=0.5,**kwargs):
    """
    Create an upper triangular random mask for a matrix of size n x n
    """
    matrix = np.random.choice(a= [0,1], p = [p,1-p], size=(n, n))  # Generate a random matrix of 0s and 1s
    upper_triangular_mask = np.triu(matrix)  # Extract the upper triangular part
    upper_triangular_mask -= np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask

## Helper functions for creating a random DAG
def get_named_edges(node_names, A):
    """
    Given a list of node names and an adjacency matrix, return a list of edges
      in the form (source, target)
    """
    if not (len(node_names) == A.shape[0] == A.shape[1]):
        raise ValueError("Node names length and matrix dimensions do not match!")
    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]
    return edge_list

def get_simple_paths_lengths(G: nx.DiGraph):
    """
    Given a directed graph G, return a dictionary of all simple paths
      and their lengths
    """
    path_dict = {}
    path_length_dict = {}
    # Finding all simple paths in the DAG
    for source in G.nodes:
        for target in G.nodes:
            if source != target and nx.has_path(G, source, target):
                path_dict[(source,target)] = list(nx.all_simple_paths(G, source, target))
                path_length_dict[(source,target)] = [len(path) for path in path_dict[(source,target)]]
    return path_dict, path_length_dict

def create_random_DAG(num_nodes, p):
    """
    Create a random DAG with a given number of nodes and edge probability
    """
    node_names = ['X' + str(i) for i in range(num_nodes)]
    # generate random DAG structures until they are connected and have at least 3 
    # nodes with no children and 3 nodes with no parents
    while True:
        A = create_upper_triangular_mask(num_nodes, p=p)
        edge_list = get_named_edges(node_names, A)
        graph_cgm = CausalGraphicalModel(node_names, edge_list)
        if nx.is_connected(graph_cgm.graph) and len(graph_cgm.no_children)>3 and len(graph_cgm.no_parents)>3:
            break

    path_dict, path_length_dict = get_simple_paths_lengths(graph_cgm.dag)
    dag = graph_cgm.dag

    return dag, path_dict, path_length_dict

def check_eval_path_in_training_fraction(list_of_tokens_eval, list_of_tokens_train):
    count = 0
    for tmp_eval in list_of_tokens_eval:
        start = tmp_eval[1]
        end = tmp_eval[2]
        bool_count = False
        for tmp_train in list_of_tokens_train:
            if start in tmp_train and end in tmp_train: bool_count = True
        if bool_count: count += 1
    print(count/float(len(list_of_tokens_eval)))

if __name__ == '__main__':
    num_nodes = 10
    p = 0.5
    # Have at least one intermediate node to define it as a path
    path_length_threshold = 2
    #Fraction of node pairs to hold out for validation
    frac = 0.2

    dag, path_dict_no_prompt, path_length_dict = create_random_DAG(num_nodes=num_nodes, p=p)
    
    # Insert prompt and end tokens, and sort by the shortest paths
    path_dict = {}
    for node_pair, paths in path_dict_no_prompt.items():
        for path in paths:
            path_w_tokens = ['target', path[-1]] + path[:] + ['path', '###']
            path_dict.setdefault(node_pair, []).append(path_w_tokens)
        path_dict[node_pair].sort(key=len)
        path_length_dict[node_pair].sort()

    print(f"Number of edges is: {len(dag.edges)}")
    node_pairs_exceeding_threshold = [key for key, value in path_length_dict.items() if any(path_value > path_length_threshold for path_value in value)]
    print(f"Number of path-connected node pairs is {len(node_pairs_exceeding_threshold)}")
  
    num_train_paths = int((1-frac)*len(node_pairs_exceeding_threshold))

    train_node_pairs = random.sample(node_pairs_exceeding_threshold, num_train_paths)
    print(f'Number of train node pairs is {len(train_node_pairs)}')
    eval_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in train_node_pairs]
    print(f'Number of eval node pairs is {len(eval_node_pairs)}')

    ## Sanity check
    assert [node_pair for node_pair in train_node_pairs if node_pair in eval_node_pairs] == [], 'overlap between train and eval node pairs is not empty'

    train_paths = [path for node_pair in train_node_pairs for path in path_dict.get(node_pair, [])]
    eval_paths = [path for node_pair in eval_node_pairs for path in path_dict.get(node_pair, [])]

    print(len(train_paths),"train_paths")
    print(len(eval_paths),"eval_paths")

    ##### Tokenization
    flat_train_paths = [node for path in train_paths for node in path]
    flat_eval_paths = [node for path in eval_paths for node in path]

    alphabets = list(set(flat_train_paths)) #set of all nodes/tokens in training data
    n_alphabets = len(alphabets)
    token_map = {}
    token_idx_map = {}
    # Alphabet tokens
    for i in range(n_alphabets):
        token_map[i] = alphabets[i]
        token_idx_map[alphabets[i]] = i

    list_of_tokens_train = [[token_idx_map[element] for element in sublist] for sublist in train_paths]
    print(len(list_of_tokens_train))

    # Make sure train and eval have all node names and same tokens
    assert set(flat_eval_paths).issubset(set(flat_train_paths))
    list_of_tokens_eval = [[token_idx_map[element] for element in sublist] for sublist in eval_paths]

    print(len(list_of_tokens_eval))

    list_of_tokens_train = np.array(pad_nested_list(list_of_tokens_train))
    list_of_tokens_eval = np.array(pad_nested_list(list_of_tokens_eval))

    ### Save everything
    np.save('tokens_path_train.npy',np.array(list_of_tokens_train))
    np.save('tokens_path_eval.npy',np.array(list_of_tokens_eval))
  
    with open("dag_path.pkl", "wb") as f:
        pickle.dump(dag, f)

    graph_dict = {}

    graph_dict['train_node_pairs'] = train_node_pairs
    graph_dict['eval_node_pairs'] = eval_node_pairs

    graph_dict['path_dict'] = path_dict_no_prompt
    graph_dict['path_length_dict'] = path_length_dict

    graph_dict['token_map'] = token_map
    graph_dict['token_idx_map'] = token_idx_map

    # graph_dict['eval_pairs_dict'] = eval_pairs_dict

    np.savez("graph_path.npz",**graph_dict)

    check_eval_path_in_training_fraction(list_of_tokens_eval,list_of_tokens_train)