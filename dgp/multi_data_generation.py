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


def get_simple_paths_lengths(G):
    path_dict = {}
    path_length_dict = {}
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                for path in nx.all_simple_paths(G, source, target, cutoff=None):
                    if (source, target) not in path_dict:
                        path_dict[(source, target)] = []
                        path_length_dict[(source, target)] = []
                    path_dict[(source, target)].append(list(path))
                    path_length_dict[(source, target)].append(len(path))
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
    return graph_cgm, graph_cgm.dag, path_dict, path_length_dict


def pad_nested_list(nested_list):
    max_length = max(len(sublist) for sublist in nested_list)
    padded_list = [sublist + [0] * (max_length - len(sublist)) for sublist in nested_list]
    return padded_list


def check_eval_nodes_in_training(eval_tokens, train_tokens):
    eval_nodes = set(token for seq in eval_tokens for token in seq)
    train_nodes = set(token for seq in train_tokens for token in seq)

    missing_nodes = eval_nodes - train_nodes
    if missing_nodes:
        print(f"The following nodes from the evaluation set are not present in the training set: {missing_nodes}")
    else:
        print("All nodes in the evaluation set are present in the training set.")


def generate_graph_data(num_graphs=1, num_nodes=100, p=0.9, path_length_threshold=2, frac=0.2):
    global_token_map = {}
    global_token_idx_map = {}
    all_training_paths = []
    all_eval_paths = []
    graph_dags = {}

    for graph_idx in range(num_graphs):
        print(f"Creating graph {graph_idx + 1}/{num_graphs}")
        graph_cgm, graph_dag, path_dict, path_length_dict = create_random_DAG(num_nodes=num_nodes, p=p)

        path_dict_cot = {}
        path_length_dict_cot = {}

        for source in graph_dag.nodes:
            for target in graph_dag.nodes:
                if source != target and nx.has_path(graph_dag, source, target):
                    for path in nx.all_simple_paths(graph_dag, source, target, cutoff=None):
                        path.insert(0, list(path)[-1])
                        path.insert(0, 'target')
                        path.append('path')
                        path.append('###')
                        if (source, target) not in path_dict_cot:
                            path_dict_cot[(source, target)] = []
                            path_length_dict_cot[(source, target)] = []
                        path_dict_cot[(source, target)].append(list(path))
                        path_length_dict_cot[(source, target)].append(len(path) - 4)

        edge_pairs = [key for key, value in path_length_dict_cot.items() if any(length == path_length_threshold for length in value)]
        node_pairs_exceeding_threshold = [key for key, value in path_length_dict_cot.items() if any(length > path_length_threshold for length in value)]

        num_paths = int(frac * len(node_pairs_exceeding_threshold))
        chosen_node_pairs = random.sample(node_pairs_exceeding_threshold, num_paths)
        held_out_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in chosen_node_pairs]

        all_nodes = set(node for node_pair in node_pairs_exceeding_threshold for node in node_pair)
        for node in all_nodes:
            if not any(node in node_pair for node_pair in chosen_node_pairs):
                for node_pair in held_out_node_pairs:
                    if node in node_pair:
                        chosen_node_pairs.append(node_pair)
                        held_out_node_pairs.remove(node_pair)
                        break

        training_paths = [path for node_pair in chosen_node_pairs for path in path_dict_cot.get(node_pair, [])]
        eval_paths = [path for node_pair in held_out_node_pairs for path in path_dict_cot.get(node_pair, [])]

        for paths in training_paths + eval_paths:
            for token in paths:
                if token not in global_token_idx_map:
                    idx = len(global_token_map)
                    global_token_map[idx] = token
                    global_token_idx_map[token] = idx

        graph_dags[graph_idx] = graph_dag

        tokenized_train_paths = [[global_token_idx_map[token] for token in path] for path in training_paths]
        tokenized_eval_paths = [[global_token_idx_map[token] for token in path] for path in eval_paths]

        all_training_paths.extend(tokenized_train_paths)
        all_eval_paths.extend(tokenized_eval_paths)

    all_training_paths = pad_nested_list(all_training_paths)
    all_eval_paths = pad_nested_list(all_eval_paths)

    np.save('data/tokens_path_train.npy', np.array(all_training_paths))
    np.save('data/tokens_path_eval.npy', np.array(all_eval_paths))

    with open("data/dag_path.pkl", "wb") as f:
        pickle.dump(graph_dags, f)

    np.savez("data/graph_path.npz", token_map=global_token_map, token_idx_map=global_token_idx_map)

    check_eval_nodes_in_training(all_eval_paths, all_training_paths)


if __name__ == '__main__':
    num_graphs = 3
    num_nodes = 100
    p = 0.9
    path_length_threshold = 2
    frac = 0.2
    generate_graph_data(num_graphs=num_graphs, num_nodes=num_nodes, p=p, path_length_threshold=path_length_threshold, frac=frac)
