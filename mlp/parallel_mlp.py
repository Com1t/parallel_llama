import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from transformers import LlamaConfig

from model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class ParallelLlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super(ParallelLlamaMLP, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = config.mlp_bias
        assert self.mlp_bias is False, "model bias currently is not support"

        # ColumnParallelLinear splits the output dimension across GPUs
        self.gate_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            gather_output=False,  # Ensures the final output is gathered across all GPUs
        )
        self.up_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            gather_output=False,  # Ensures the final output is gathered across all GPUs
        )
        # RowParallelLinear splits the input dimension across GPUs
        self.down_proj = RowParallelLinear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            input_is_parallel=True,
        )

    def init_layer_weight(self, target_layer, raw_weight):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        weight_partition_dim = [0, 0]
        if isinstance(target_layer, ColumnParallelLinear):
            weight_partition_dim[0] = raw_weight.shape[0] // world_size
            weight_partition_dim[1] = raw_weight.shape[1]
            split_dim = 0
        elif isinstance(target_layer, RowParallelLinear):
            weight_partition_dim[0] = raw_weight.shape[0]
            weight_partition_dim[1] = raw_weight.shape[1] // world_size
            split_dim = 1
        else:
            raise TypeError("ColumnParallelLinear or RowParallelLinear are allowed")

        output_tensor = torch.zeros(
            weight_partition_dim[0], weight_partition_dim[1]
        ).to(target_layer.weight.device)

        if rank == 0:
            weight_list = torch.split(
                raw_weight, weight_partition_dim[split_dim], dim=split_dim
            )
            scatter_list = [t.contiguous() for t in weight_list]
        else:
            scatter_list = None
        dist.scatter(output_tensor, scatter_list, src=0)
        target_layer.weight = nn.Parameter(output_tensor)

    def weight_init(self, gate_proj_weight, up_proj_weight, down_proj_weight):
        self.init_layer_weight(self.gate_proj, gate_proj_weight)
        self.init_layer_weight(self.up_proj, up_proj_weight)
        self.init_layer_weight(self.down_proj, down_proj_weight)

    def forward(self, x):
        # down_proj = F.linear(
        #     F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
        #     self.down_proj,
        # )
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
