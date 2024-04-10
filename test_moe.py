import torch
import yaml
from moe import MoE

from utils import dotdict


def test_noisy_topk():
    with open('./configs/test_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = dotdict(config)
    batch_size = config.batch_size
    batch_size = 2

    moe = MoE(config)
    # x: (batch_size, seq_len, embedding_dim)
    x = torch.rand((batch_size, 1, config.n_embd))
    gates, load = moe.noisy_top_k_gating(x, train=False, noise_epsilon=0.0)
    
    x = x @ moe.w_gate
    gates2 = torch.topk(x, k=config.expert_k, dim=-1)
    print(gates.shape, gates2.shape)

    print(gates)
    print(gates2.values)

    assert torch.allclose(gates, gates2.values, atol=1e-5), f"topk gates are not equal"




if __name__ == "__main__":
    test_noisy_topk()
