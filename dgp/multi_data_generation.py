# Generate four files for the graph task dataset:
# 1. dags.pkl, a dictionary with each separate DAG as a networkx DiGraph object. Each graph's nodes have their own letter,
#    e.g. the first graph has nodes A1, A2, ..., the second B1, B2, ... and so on.
# 2. token_maps.npz, with the token map (idx -> token/node name) and the reverse token map (token->idx)
# 3. tokens_path_train.npy, a list of all the selected training paths
# 4. tokens_path_eval.npy, a list of all the selected evaluation paths

from string import ascii_uppercase
import numpy as np
import networkx as nx
import random
import pickle
import os

if not os.path.exists("data"):
    os.makedirs("data")


def create_upper_triangular_mask(n, p=0.5):
    matrix = np.random.choice(a=[0, 1], p=[p, 1 - p], size=(n, n))
    upper_triangular_mask = np.triu(matrix)
    upper_triangular_mask -= np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask


def get_named_edges(node_names, A):
    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]
    return edge_list


def get_simple_paths(G):
    path_dict = {}
    # path_length_dict = {} # not used for now
    for source in G.nodes:
        for target in G.nodes:
            if source != target and nx.has_path(G, source, target):
                path_dict[(source,target)] = list(nx.all_simple_paths(G, source, target))
                # path_length_dict[(source,target)] = [len(path) for path in path_dict[(source,target)]]
    return path_dict


class CausalGraphicalModel:
    def __init__(self, node_names, edge_list):
        self.node_names = node_names
        self.edge_list = edge_list
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(node_names)
        self.dag.add_edges_from(edge_list)
        self.no_parents = [node for node in self.dag.nodes() if self.dag.in_degree(node) == 0]
        self.no_children = [node for node in self.dag.nodes() if self.dag.out_degree(node) == 0]


def create_random_DAG(num_nodes, p, name='X'):
    node_names = [name + str(i) for i in range(num_nodes)]
    A = create_upper_triangular_mask(num_nodes, p=p)
    edge_list = get_named_edges(node_names, A)
    graph_cgm = CausalGraphicalModel(node_names, edge_list)

    while not nx.is_weakly_connected(graph_cgm.dag) or len(graph_cgm.no_children) < 3 or len(graph_cgm.no_parents) < 3:
        A = create_upper_triangular_mask(num_nodes, p=p)
        edge_list = get_named_edges(node_names, A)
        graph_cgm = CausalGraphicalModel(node_names, edge_list)

    path_dict = get_simple_paths(graph_cgm.dag)
    return graph_cgm, graph_cgm.dag, path_dict


def pad_nested_list(nested_list):
    """Make each sequence in a nested list the same length by padding with zeros."""
    max_length = max(len(sublist) for sublist in nested_list)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in nested_list]
    return padded_list


def check_eval_nodes_in_train(eval_tokens, train_tokens):
    eval_nodes = set(token for seq in eval_tokens for token in seq)
    train_nodes = set(token for seq in train_tokens for token in seq)

    missing_nodes = eval_nodes - train_nodes
    if missing_nodes:
        print(f"The following nodes from the evaluation set are not present in the training set: {missing_nodes}")
    else:
        print("All nodes in the evaluation set are present in the training set.")


def generate_graph_data(num_graphs=1, num_nodes=100, p=0.9, path_length_threshold=2, frac=0.2):
    """
    Generate DAGs, create training and validation datasets of paths through the different graphs, 
    and tokenize in the nodes. Save everything.

    Args:
        num_graphs (int): The number of graphs to generate.
        num_nodes (int): The number of nodes in each graph. 
        p (float): The probability of an edge existing between two nodes. 
        path_length_threshold (int): If a path length is less than this, the path is not considered. 
        frac (float): The rough fraction of the dataset chosen for validation. Default is 0.2.

    Returns:
        None
    """

    global_token_map = {}
    global_token_idx_map = {}
    all_train_paths = []
    all_eval_paths = []
    dags = {}

    for graph_idx in range(num_graphs):
        print(f"Creating graph {graph_idx + 1}/{num_graphs}")
        graph_cgm, dag, path_dict_no_prompt = create_random_DAG(num_nodes=num_nodes, p=p, name=ascii_uppercase[graph_idx])

        # Insert prompt and end tokens, and sort by the shortest paths
        path_dict = {}
        for node_pair, paths in path_dict_no_prompt.items():
            for path in paths:
                path_w_tokens = ['target', path[-1]] + path[:] + ['###']
                path_dict.setdefault(node_pair, []).append(path_w_tokens)
            path_dict[node_pair].sort(key=len)

        node_pairs_exceeding_threshold = [pair for pair, paths in path_dict.items() if any(len(path) > path_length_threshold for path in paths)]

        num_paths = int((1-frac) * len(node_pairs_exceeding_threshold))
        chosen_node_pairs = random.sample(node_pairs_exceeding_threshold, num_paths)
        held_out_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in chosen_node_pairs]

        all_nodes = set(node for node_pair in node_pairs_exceeding_threshold for node in node_pair)
        
        # If a node is not in any of the chosen (train) node pairs, transfer a corresponding node pair from held out (eval)
        for node in all_nodes:
            if not any(node in node_pair for node_pair in chosen_node_pairs):
                for node_pair in held_out_node_pairs:
                    if node in node_pair:
                        chosen_node_pairs.append(node_pair)
                        held_out_node_pairs.remove(node_pair)
                        break

        train_paths = [path for node_pair in chosen_node_pairs for path in path_dict.get(node_pair, [])]
        eval_paths = [path for node_pair in held_out_node_pairs for path in path_dict.get(node_pair, [])]

        for paths in train_paths + eval_paths:
            for token in paths:
                if token not in global_token_idx_map:
                    idx = len(global_token_map)
                    global_token_map[idx] = token
                    global_token_idx_map[token] = idx

        dags[graph_idx] = dag

        tokenized_train_paths = [[global_token_idx_map[token] for token in path] for path in train_paths]
        tokenized_eval_paths = [[global_token_idx_map[token] for token in path] for path in eval_paths]

        all_train_paths.extend(tokenized_train_paths)
        all_eval_paths.extend(tokenized_eval_paths)

    all_train_paths = pad_nested_list(all_train_paths)
    all_eval_paths = pad_nested_list(all_eval_paths)

    np.save('data/tokens_path_train.npy', np.array(all_train_paths))
    np.save('data/tokens_path_eval.npy', np.array(all_eval_paths))

    with open("data/dags.pkl", "wb") as f:
        pickle.dump(dags, f)

    np.savez("data/token_maps.npz", token_map=global_token_map, token_idx_map=global_token_idx_map)

    check_eval_nodes_in_train(all_eval_paths, all_train_paths)


if __name__ == '__main__':
    num_graphs = 3
    num_nodes = 128
    p = 0.91 # probability of each pair of nodes being connected
    path_length_threshold = 2 # only paths with more than this many nodes considered
    frac = 0.2 # approx fraction of paths held out for validation
    generate_graph_data(num_graphs=num_graphs, num_nodes=num_nodes, p=p, path_length_threshold=path_length_threshold, frac=frac)
