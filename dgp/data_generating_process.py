import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
import networkx as nx
import random

from cgm import *

#Helper functions to create motifs
def create_upper_triangular_mask(n,p=0.5,**kwargs):
    matrix = np.random.choice(a= [0,1], p = [p,1-p], size=(n, n))  # Generate a random matrix of 0s and 1s
    upper_triangular_mask = np.triu(matrix)  # Extract the upper triangular part
    upper_triangular_mask = upper_triangular_mask - np.diag(np.diag(upper_triangular_mask))
    return upper_triangular_mask

## Helper functions for creating a random DAG
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

def get_named_edges(node_names, A):
    if len(node_names) != A.shape[0] or len(node_names) != A.shape[1]:
        raise ValueError("Node names length and matrix dimensions do not match!")

    non_zero_indices = np.nonzero(A)
    edge_list = [(node_names[i], node_names[j]) for i, j in zip(*non_zero_indices)]

    return edge_list

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

  return graph_cgm, graph_dag, path_dict, path_length_dict

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
    #Fraction of held out node pairs
    frac = 0.2

    graph_cgm, graph_dag, path_dict_no_prompt, path_length_dict_no_prompt = create_random_DAG(num_nodes=num_nodes, p=p)
    
    path_dict_cot = {}
    path_length_dict = {}

    # Finding all simple paths in the DAG
    for source in graph_dag.nodes:
        for target in graph_dag.nodes:
            if source != target and nx.has_path(graph_dag, source, target):
                for path in nx.all_simple_paths(graph_dag, source, target, cutoff=None):
                    # format as 'target' -> target_node -> actual_path -> 'path' -> '###'
                    path.insert(0, list(path)[-1])
                    path.insert(0, 'target')
                    path.append('path')
                    path.append('###')
                    #if there already exists a path, append this new one to it
                    if path_dict_cot.get((source,target)):
                        path_dict_cot[(source,target)].append(list(path))
                        path_length_dict[(source,target)].append(len(path)-4)

                    #otherwise create a new dictionary item
                    else:
                        path_dict_cot[(source,target)] = [list(path)]
                        path_length_dict[(source,target)] = [len(path)-4]

    #Sort by the shortest paths 
    for source, target in path_dict_cot:
        path_dict_cot[(source,target)].sort(key=len)
        path_length_dict[(source,target)].sort()

    edge_pairs = [key for key, value in path_length_dict.items() if any(path_value == path_length_threshold for path_value in value)]
    print(f"Number of edges is: {len(edge_pairs)}")
    node_pairs_exceeding_threshold = [key for key, value in path_length_dict.items() if any(path_value > path_length_threshold for path_value in value)]
    print(f"Number of path-connected node pairs is {len(node_pairs_exceeding_threshold)}")
    print('Sanity check edge pairs and node pairs exceeding threshold: ')
    print(len([node_pair for node_pair in node_pairs_exceeding_threshold if node_pair in edge_pairs]))

    num_paths = int(frac*len(node_pairs_exceeding_threshold))

    chosen_node_pairs = random.sample(node_pairs_exceeding_threshold, num_paths)
    print(f'Number of chosen node pairs is {len(chosen_node_pairs)}')
    
    held_out_node_pairs = [t for t in node_pairs_exceeding_threshold if t not in chosen_node_pairs]

    ## Sanity check
    print('Sanity checks (should be empty list output for overlap between chosen and held out):')
    print([node_pair for node_pair in chosen_node_pairs if node_pair in held_out_node_pairs])


    path_strings_chosen_cot = [path for node_pair in chosen_node_pairs for path in path_dict_cot.get(node_pair, [])]
    path_strings_held_out_cot = [path for node_pair in held_out_node_pairs for path in path_dict_cot.get(node_pair, [])]

    path_strings_edge_pairs_cot = [path_dict_cot[node_pair][0] for node_pair in edge_pairs]

    sample_strings_train_path_cot =  path_strings_chosen_cot
    print("sample_strings_train_path_cot", len(sample_strings_train_path_cot))

    # sample_strings_eval_path_cot = random.sample(path_strings_held_out_cot, int(0.2*nsamples))
    sample_strings_eval_path_cot = path_strings_held_out_cot
    print("sample_strings_eval_path_cot", len(sample_strings_eval_path_cot))


    ##### Tokenization
    flattened_list_train_cot = [element for sublist in sample_strings_train_path_cot for element in sublist]
    alphabets = list(set(flattened_list_train_cot))
    n_alphabets = len(alphabets)
    token_map = {}
    token_idx_map = {}
    # Alphabet tokens
    for i in range(n_alphabets):
        token_map[i] = alphabets[i]
        token_idx_map[alphabets[i]] = i

    list_of_tokens_train_cot = [[token_idx_map[element] for element in sublist] for sublist in sample_strings_train_path_cot]
    print(len(list_of_tokens_train_cot))

    flattened_list_eval_cot = [element for sublist in sample_strings_eval_path_cot for element in sublist]
    # Make sure train and eval have all node names and same tokens
    assert set(flattened_list_eval_cot).issubset(set(flattened_list_train_cot))
    list_of_tokens_eval_cot = [[token_idx_map[element] for element in sublist] for sublist in sample_strings_eval_path_cot]

    print(len(list_of_tokens_eval_cot))

    # flattened_list_train_direct = [element for sublist in sample_strings_train_path_direct for element in sublist]
    # # Make sure train and eval have all node names and same tokens
    # assert set(flattened_list_train_direct).issubset(set(flattened_list_train_cot))
    # list_of_tokens_train_direct = [[token_idx_map[element] for element in sublist] for sublist in sample_strings_train_path_direct]

    # print(len(list_of_tokens_train_direct))

    # flattened_list_eval_direct = [element for sublist in sample_strings_eval_path_direct for element in sublist]
    # Make sure train and eval have all node names and same tokens
    # assert set(flattened_list_eval_direct).issubset(set(flattened_list_train_cot))
    # list_of_tokens_eval_direct = [[token_idx_map[element] for element in sublist] for sublist in sample_strings_eval_path_direct]

    # print(len(list_of_tokens_eval_direct))

    max_length_cot = max([len(seq) for seq in list_of_tokens_train_cot])
    # max_length_direct = max([len(seq) for seq in list_of_tokens_train_direct])
    # max_length = max([max_length_cot, max_length_direct])


    list_of_tokens_train_cot = np.array(pad_nested_list(list_of_tokens_train_cot))
    # list_of_tokens_train_direct = np.array(pad_nested_list(list_of_tokens_train_direct))
    list_of_tokens_eval_cot = np.array(pad_nested_list(list_of_tokens_eval_cot))
    # list_of_tokens_eval_direct = np.array(pad_nested_list(list_of_tokens_eval_direct))
    # print(np.max(list_of_tokens_train_cot), len(list_of_tokens_train_cot))
    # print(np.max(list_of_tokens_train_direct), len(list_of_tokens_train_direct))


    ### Save everything
    np.save('tokens_path_train.npy',np.array(list_of_tokens_train_cot))
    np.save('tokens_path_eval.npy',np.array(list_of_tokens_eval_cot))
    # np.save('tokens_path_train_direct.npy',np.array(list_of_tokens_train_direct))
    # np.save('tokens_path_eval_direct.npy',np.array(list_of_tokens_eval_direct))

    with open("dag_path.pkl", "wb") as f:
        pickle.dump(graph_dag, f)

    graph_dict = {}

    graph_dict['chosen_node_pairs'] = chosen_node_pairs
    graph_dict['held_out_node_pairs'] = held_out_node_pairs

    # graph_dict['chosen_non_path_node_pairs'] = chosen_non_path_node_pairs
    # graph_dict['held_out_non_path_node_pairs'] = held_out_non_path_node_pairs

    graph_dict['path_dict'] = path_dict_no_prompt
    graph_dict['path_length_dict'] = path_length_dict_no_prompt

    graph_dict['token_map'] = token_map
    graph_dict['token_idx_map'] = token_idx_map

    graph_dict['eval_pairs_dict'] = eval_pairs_dict

    np.savez("graph_path.npz",**graph_dict)

    check_eval_path_in_training_fraction(list_of_tokens_eval_cot,list_of_tokens_train_cot)