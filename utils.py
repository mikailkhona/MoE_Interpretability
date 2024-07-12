import numpy as np
from string import ascii_uppercase
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
import random
import os
import sys

from omegaconf import OmegaConf

## INIT STUFF

def set_seed(seed=0):
    """
    Don't set true seed to be nearby values. Doesn't give best randomness
    """
    rng = np.random.default_rng(seed)
    true_seed = int(rng.integers(2**30))

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(true_seed)
    torch.cuda.manual_seed_all(true_seed)

def open_log(cfg):
    print(cfg)
    os.makedirs('logs/' + cfg.tag, exist_ok=True)
    if cfg.deploy:
        # Open log file
        fname = 'logs/' + cfg.tag + '/' + wandb.run.id + ".log"
        fout = open(fname, "a", 1)
        sys.stdout = fout
        sys.stderr = fout
        print(cfg)
        # Initialize wandb
        print('Initializing wandb project')
        wandb.init(project=cfg.wandb_project)
        wandb.run.name = wandb.run.id
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg))
        return fout

# Close log file and clean up
def cleanup(cfg, fp):
    if cfg.deploy:
        fp.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        wandb.finish()

# Learning rate decay scheduler (cosine with warmup)
def get_cosine_warmp_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    '''
    Return lr for it'th step
    '''

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    Wrapper around dictionary
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Dataloader stuff

def generate_batches_lol(batch_size, file_path):
    '''
    Returns a dataloader
    '''

     # Compute the absolute maximum length across all sequences
    # absolute_max_length = max(len(sublist) for sublist in dataset)

    # Load the entire dataset
    dataset = np.load(file_path, allow_pickle=True)

    # Create an index array and shuffle it
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    # Iterate through the dataset in batches
    for start_idx in range(0, len(dataset), batch_size):
        # Select the indices for this batch
        batch_indices = indices[start_idx:start_idx + batch_size]

        # Extract the corresponding sublists
        batch_sublists = [dataset[i] for i in batch_indices]

        # Compute the maximum length of sublists in this batch
        max_length = max(len(sublist) for sublist in batch_sublists)

        # Pad each sublist with zeros to match the maximum length
        padded_batch_x = [sublist + [0] * (max_length - len(sublist)) for sublist in batch_sublists]

        # Shift the sublists by one to create the Y sequences
        padded_batch_y = [sublist[1:] + [0] for sublist in padded_batch_x]

        yield np.array(padded_batch_x), np.array(padded_batch_y)


class SequenceDataset(Dataset):
    '''
    Dataset and DataLoader for sequence data.
    Made specifically for autoregressive next token prediction training
    Data is integer-type for tokenizer
    '''

    def __init__(self, filepath, block_size, add_one_token=True):
        self.data = np.load(filepath, allow_pickle=True)
        self.block_size = block_size
        self.add_one_token = add_one_token
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        #y is 1-index shifted version of x. Everything should be integer for tokenizer.
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        if(self.add_one_token):
            x = torch.tensor(x, dtype=torch.int64) + 1
            y = torch.tensor(y, dtype=torch.int64) + 1
            return x,y
        else:
            return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)


def get_dataloader(train_data_path, val_data_path, block_size, batch_size, shuffle=True, num_workers=4, add_one_token=True):
    '''
    Open data directory and get train and val dataloaders
    add one token: shifts all token idxs by 1 because 0 is the padding token
    '''

    train_dataset = SequenceDataset(train_data_path, block_size, add_one_token)
    val_dataset = SequenceDataset(val_data_path, block_size, add_one_token)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, drop_last=True)

    return train_dataloader, val_dataloader

class SequenceDataset_lol(Dataset):
    '''
    Dataset and DataLoader for sequence data.
    Made specifically for autoregressive next token prediction training
    Data is integer-type for tokenizer
    '''

    def __init__(self, filepath, add_one_token=True):
        self.data = [list(x) for x in np.load(filepath, allow_pickle=True)]  # Loading the sequences as lists
        self.add_one_token = add_one_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # y is 1-index shifted version of x. Everything should be integer for tokenizer.
        x = self.data[idx]
        y = x[1:] + [-1]  # Assuming 0 is the padding token

        if(self.add_one_token):
            y = x[1:] + [-1]  # Assuming 0 is the padding token
            x = torch.tensor(x, dtype=torch.int64) + 1
            y = torch.tensor(y, dtype=torch.int64) + 1
            return x,y
        else:
            y = x[1:] + [0]  # Assuming 0 is the padding token
            return torch.tensor(x, dtype=torch.int64), torch.tensor(y, dtype=torch.int64)

def collate_fn_pad(batch):
    x, y = zip(*batch)
    # Pad sequences to the maximum length in the batch
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return x_padded, y_padded


def get_dataloader_lol(train_data_path, val_data_path, batch_size, shuffle=True, num_workers=4, add_one_token=True):
    '''
    Open data directory and get train and val dataloaders
    '''

    train_dataset = SequenceDataset_lol(train_data_path, add_one_token)
    val_dataset = SequenceDataset_lol(val_data_path, add_one_token)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, drop_last=False, collate_fn=collate_fn_pad)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, drop_last=False, collate_fn=collate_fn_pad)

    return train_dataloader, val_dataloader

# Measuring accuracy.

def check_edge_accuracy(dag_dict, nodes, start_index, stop_index):
    """
    Check whether every edge in nodes (defined by pairs of successive nodes) is an actual edge in the relevant graph within dag_dict.

    Parameters:
    - dag_dict (dict): Dictionary of nx.DiGraph objects.
    - nodes (list): List of node names supposed to form a simple path.

    Returns:
    - bool: True if every consecutive pair of nodes in the list is an edge in G, False otherwise.
    """
    edge_bools = []
    edge_list = []
    graph_idx = ascii_uppercase.index(nodes[2][0]) # graph idx is indicated by the node letter: 1 for A, 2 for B, etc.
    for i in range(start_index, stop_index-1):
        edge_bools.append(dag_dict[graph_idx].has_edge(nodes[i], nodes[i + 1]))
        edge_list.append((nodes[i], nodes[i + 1]))

    does_end_at_target = (nodes[1]==nodes[stop_index-1])
    return np.array(edge_bools), edge_list, does_end_at_target

def check_generated_path_accuracy(dag_dict, generated_tokens, token_map):

    num_samples = len(generated_tokens)
    batch_size = len(generated_tokens[0])

    accuracies = np.zeros((num_samples,batch_size))
    does_end_at_targets = np.zeros((num_samples,batch_size))
    path_lengths = np.zeros((num_samples,batch_size))
    for j in range(num_samples):
        for i in range(batch_size):
            batch_idx = i
            sample_idx = j
            tokens = generated_tokens[sample_idx][batch_idx].cpu().numpy()
            nodes = [token_map[token.item() - 1] for token in tokens if token.item() != 0]
            stop_token_indices = np.where(np.array(nodes) == '###')[0]
            if stop_token_indices.shape[0] == 0:
                stop_index = len(nodes)
            else:
                stop_index = stop_token_indices[0]
            edge_bools, edge_list, does_end_at_target = check_edge_accuracy(dag_dict, nodes, start_index=2, stop_index=stop_index)
            accuracies[j,i] = np.mean(edge_bools)
            does_end_at_targets[j,i] = does_end_at_target
            path_lengths[j,i] = len(edge_list)

    return accuracies, does_end_at_targets, path_lengths

def make_prompt(source, target, token_idx_map):
  source_token = token_idx_map[source]
  target_token = token_idx_map[target]
  start_token = token_idx_map['target']
  prompt = torch.from_numpy(np.array([start_token + 1, target_token+1, source_token+1]))
  return prompt
