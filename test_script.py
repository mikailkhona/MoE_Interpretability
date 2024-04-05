import torch
import yaml
from model import MoEGPT

with open('./configs/test_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

batch_size = config.batch_size
n_embd = config.n_embd


model = MoEGPT(config)
x = torch.rand(batch_size, input_size)

model, loss = model(x)