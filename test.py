import numpy as np
import torch
import pickle
import yaml
from model import MoEGPT
from moe import MoE
from utils import *
from omegaconf import OmegaConf

def test_paths():
    with open('./configs/test_config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = dotdict(cfg)
    token_maps_dict = np.load(cfg.dataset_path + 'token_maps.npz', allow_pickle=True)
    token_map = token_maps_dict['token_map'].item()
    with open(cfg.dataset_path + 'dags.pkl', "rb") as f:
        dag_dict = pickle.load(f)
    some_data = np.load(cfg.dataset_path + 'tokens_path_train.npy')[:10] # 10 paths
    for path in some_data:
        nodes = [token_map[idx] for idx in path]
        stop_index = np.where(np.array(nodes) == 'path')[0][0]
        edge_bools, edge_list, does_end_at_target = check_edge_accuracy(dag_dict, nodes, start_index=2, stop_index=stop_index)
        assert edge_bools.all(), 'Not all edges are valid'
        assert does_end_at_target, 'Path does not end at target'

def test_scipt():
    with open('./configs/test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    config = OmegaConf.create(config)
    batch_size = config.batch_size
    n_embd = config.n_embd
    print(config.noisy_gating)
    seq_len = 2
    model = MoEGPT(config)
    x = torch.randint(100, (batch_size, seq_len))
    model, loss = model(x)

def test_noisy_topk():
    with open('./configs/test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = dotdict(config)
    batch_size = config.batch_size
    batch_size = 10
    seq_len = 2

    k = 2
    config.expert_k = k

    moe = MoE(config)
    # test with non-zero w_gate
    moe.w_gate = torch.nn.Parameter(torch.rand((config.n_embd, config.num_experts)))

    # x: (batch_size, seq_len, embedding_dim)
    x = torch.rand((batch_size, seq_len, config.n_embd))
    x = x.view(-1, config.n_embd)
    gates, load = moe.noisy_top_k_gating(x, train=False, noise_epsilon=0.0)

    # x_gate: (batch_size, seq_len, n_experts)
    x_gate = x @ moe.w_gate
    topk_val, topk_idx = torch.topk(x_gate, k=config.expert_k, dim=-1)

    gates2 = torch.zeros_like(x_gate)
    gates2.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.float32))

    print(f"gates: {gates}")
    print(f"gates2: {gates2}")
    gates_binary = torch.zeros_like(gates)
    gates_binary[gates != 0] = 1
    assert torch.allclose(gates_binary, gates2)
