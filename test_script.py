import torch
import yaml
from model import MoEGPT

from utils import dotdict
with open('./configs/test_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
print(config)
config = dotdict(config)
batch_size = config.batch_size
n_embd = config.n_embd

seq_len = 2
model = MoEGPT(config)
x = torch.randint(100, (batch_size, seq_len))
model, loss = model(x)
