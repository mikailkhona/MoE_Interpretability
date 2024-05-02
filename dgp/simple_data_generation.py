import numpy as np
import networkx as nx
import random
import pickle
import os

if not os.path.exists("data"):
    os.makedirs("data")

def create_upper_triangular_mask(n, p=0.5, **kwargs):
    matrix = np.random.choice(a=[0, 1], p=[p, 1-p], size=(n, n))
    upper_triangular_mask = np.triu(matrix)
    upper_triangular_mask = upper_triangular_mask - np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask

def get_named_edges(node_names, A):
    if len(node_names) != A.shape[0] or len(node_names) != A.shape[1]:
        raise ValueError("Node names length and matrix dimensions do not match!")

    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]

    return edge_list

def get_simple_paths_lengths(G: nx.DiGraph):
    path_dict = {}
    path_length_dict = {}
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                for path in nx.all_simple_paths(G, source, target, cutoff=None):
                    if (path_dict.get((source, target)) is not None):
                        path_dict[(source, target)].append(list(path))
                        path_length_dict[(source, target)].append(len(path))
                    else:
                        path_dict[(source, target)] = [list(path)]
                        path_length_dict[(source, target)] = [len(path)]

    return path_dict, path_length_dict

class CausalGraphicalModel:
    def __init__(self, node_names, edge_list):
        self.node_names = node_names
        self.edge_list = edge_list
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(node_names)
        self.dag.add_edges_from(edge_list)
        self.no_parents = [node for node in self.dag.nodes() if self.dag.in_degree(node) == 0]
        self.no_children = [node for node in self.dag.nodes() if self.dag.out_degree(node) == 0]

def create_random_DAG(num_nodes, p):
    node_names = ['X' + str(i) for i in range(num_nodes)]
    A = create_upper_triangular_mask(num_nodes, p=p)
    edge_list = get_named_edges(node_names, A)
    graph_cgm = CausalGraphicalModel(node_names, edge_list)

    while not nx.is_weakly_connected(graph_cgm.dag) or len(graph_cgm.no_children) < 3 or len(graph_cgm.no_parents) < 3:
        A = create_upper_triangular_mask(num_nodes, p=p)
        edge_list = get_named_edges(node_names, A)
        graph_cgm = CausalGraphicalModel(node_names, edge_list)

    path_dict, path_length_dict = get_simple_paths_lengths(graph_cgm.dag)
    graph_dag = graph_cgm.dag

    return graph_cgm, graph_dag, path_dict, path_length_dict

def pad_nested_list(nested_list):
    max_length = max(len(sublist) for sublist in nested_list)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in nested_list]
    return padded_list

def check_eval_nodes_in_training(list_of_tokens_eval, list_of_tokens_train):
    eval_nodes = set(token for seq in list_of_tokens_eval for token in seq)
    train_nodes = set(token for seq in list_of_tokens_train for token in seq)

    missing_nodes = eval_nodes - train_nodes
    if missing_nodes:
        print(f"The following nodes from the evaluation set are not present in the training set: {missing_nodes}")
    else:
        print("All nodes in the evaluation set are present in the training set.")

if __name__ == '__main__':
    num_nodes = 10
    p = 0.5
    path_length_threshold = 2
    frac = 0.2
    graph_type = 'random'

    if graph_type == 'random':
        graph_cgm, graph_dag, path_dict_no_prompt, path_length_dict_no_prompt = create_random_DAG(num_nodes=num_nodes, p=p)

    path_dict_cot = {}
    path_length_dict = {}

    for source in graph_dag.nodes:
        for target in graph_dag.nodes:
            if source != target:
                if nx.has_path(graph_dag, source, target):
                    for path in nx.all_simple_paths(graph_dag, source, target, cutoff=None):
                        if path_dict_cot.get((source, target)) is not None:
                            path.insert(0, list(path)[-1])
                            path.insert(0, 'target')
                            path.append('path')
                            path.append('###')
                            path_dict_cot[(source, target)].append(list(path))
                            path_length_dict[(source, target)].append(len(path) - 4)
                        else:
                            path.insert(0, list(path)[-1])
                            path.insert(0, 'target')
                            path.append('path')
                            path.append('###')
                            path_dict_cot[(source, target)] = [list(path)]
                            path_length_dict[(source, target)] = [len(path) - 4]
                else:
                    continue

    for key in path_dict_cot:
        path_dict_cot[key].sort(key=len)
        path_length_dict[key].sort()

    edge_pairs = [key for key, value in path_length_dict.items() if any(path_value == path_length_threshold for path_value in value)]
    print(f"Number of edges is: {len(edge_pairs)}")
    node_pairs_exceeding_threshold = [key for key, value in path_length_dict.items() if any(path_value > path_length_threshold for path_value in value)]
    print(f"Number of path-connected node pairs is {len(node_pairs_exceeding_threshold)}")

    num_paths = int(frac * len(node_pairs_exceeding_threshold))
    chosen_node_pairs = random.sample(node_pairs_exceeding_threshold, num_paths)
    print(f'Number of chosen node pairs is {len(chosen_node_pairs)}')
    held_out_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in chosen_node_pairs]

    # Ensure all nodes are covered in the training set
    all_nodes = set(node for node_pair in node_pairs_exceeding_threshold for node in node_pair)
    for node in all_nodes:
        if not any(node in node_pair for node_pair in chosen_node_pairs):
            # Find a path that includes this node and add it to the training set
            for node_pair in held_out_node_pairs:
                if node in node_pair:
                    chosen_node_pairs.append(node_pair)
                    held_out_node_pairs.remove(node_pair)
                    break

    path_strings_chosen_cot = [path for node_pair in chosen_node_pairs for path in path_dict_cot.get(node_pair, [])]
    path_strings_held_out_cot = [path for node_pair in held_out_node_pairs for path in path_dict_cot.get(node_pair, [])]
    path_strings_edge_pairs_cot = [path_dict_cot[node_pair][0] for node_pair in edge_pairs]

    # Combine path_strings_chosen_cot and path_strings_held_out_cot
    all_path_strings_cot = path_strings_chosen_cot + path_strings_held_out_cot

    # Flatten and get unique elements
    flattened_list_cot = list(set([element for sublist in all_path_strings_cot for element in sublist]))

    # Create token maps
    n_alphabets = len(flattened_list_cot)
    token_map = {}
    token_idx_map = {}
    for i in range(n_alphabets):
        token_map[i] = flattened_list_cot[i]
        token_idx_map[flattened_list_cot[i]] = i

    # Tokenize chosen and held-out paths
    list_of_tokens_train_cot = [[token_idx_map[element] for element in sublist] for sublist in path_strings_chosen_cot]
    list_of_tokens_eval_cot = [[token_idx_map[element] for element in sublist] for sublist in path_strings_held_out_cot]

    print("Training set size:", len(list_of_tokens_train_cot))
    print("Evaluation set size:", len(list_of_tokens_eval_cot))

    list_of_tokens_train_cot = np.array(pad_nested_list(list_of_tokens_train_cot))
    list_of_tokens_eval_cot = np.array(pad_nested_list(list_of_tokens_eval_cot))

    np.save('data/tokens_path_train.npy', np.array(list_of_tokens_train_cot))
    np.save('data/tokens_path_eval.npy', np.array(list_of_tokens_eval_cot))

    G = graph_dag

    dag_file_name = "data/dag_path.pkl"
    with open(dag_file_name, "wb") as f:
        pickle.dump(G, f)

    graph_dict = {}

    graph_dict['chosen_node_pairs'] = chosen_node_pairs
    graph_dict['held_out_node_pairs'] = held_out_node_pairs

    graph_dict['path_dict'] = path_dict_no_prompt
    graph_dict['path_length_dict'] = path_length_dict_no_prompt

    graph_dict['token_map'] = token_map
    graph_dict['token_idx_map'] = token_idx_map

    graph_file_name = "data/graph_path.npz"

    np.savez(graph_file_name, **graph_dict)

    check_eval_nodes_in_training(list_of_tokens_eval_cot, list_of_tokens_train_cot)