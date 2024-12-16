import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from transformers import LlamaConfig


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig, device="cpu"):
        super(LlamaMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = config.mlp_bias
        assert self.mlp_bias is False, "model bias currently is not support"

        # model weights
        self.gate_proj = torch.rand(self.intermediate_size, self.hidden_size).to(device)
        self.up_proj = torch.rand(self.intermediate_size, self.hidden_size).to(device)
        self.down_proj = torch.rand(self.hidden_size, self.intermediate_size).to(device)

        # model bias, currently not support
        # if mlp_bias:
        #     self.gate_proj_bias = torch.rand(self.intermediate_size, self.hidden_size)
        #     self.up_proj_bias = torch.rand(self.intermediate_size, self.hidden_size)
        #     self.down_proj_bias = torch.rand(self.hidden_size, self.intermediate_size)

    def forward(self, x):
        down_proj = F.linear(
            F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
            self.down_proj,
        )
        return down_proj
