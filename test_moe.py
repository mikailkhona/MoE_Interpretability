import matplotlib.pyplot as plt
import torch
import yaml
from moe import MoE

from utils import dotdict


def test_noisy_topk():
    with open('./configs/test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = dotdict(config)
    batch_size = config.batch_size
    batch_size = 10
    seq_len = 2

    k = 3
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


if __name__ == "__main__":
    test_noisy_topk()
