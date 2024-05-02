import torch
import yaml
from model import MoEGPT

from omegaconf import OmegaConf
import ipdb

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
ipdb.set_trace()
