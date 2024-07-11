# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MoE part of PyTorch Mixtral model."""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

import scattermoe

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.hidden_size = 4*config.n_embd
        self.num_experts = config.num_experts
        self.k = config.expert_k

        # gating
        self.gate = nn.Linear(self.n_embd, self.num_experts, bias=False)

        self.moe_mlp = scattermoe.mlp.MLP(
            input_size=self.n_embd,
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.k,
            activation=torch.nn.GELU()
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, n_embd = hidden_states.shape
        hidden_states = hidden_states.view(-1, n_embd)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        # both routing_weights, selected_experts: (batch * sequence_length, k)
        routing_weights, selected_experts = torch.topk(routing_weights, self.k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        # count the times each expert is selected.
        self.load = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=0)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = self.moe_mlp(hidden_states, routing_weights, selected_experts)
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, n_embd)
        return final_hidden_states, router_logits