import os
import torch
from torch.nn import functional as F
import torch.distributed as dist

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group,
)
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class LlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias=False, device="cpu"):
        super(LlamaMLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
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


class ParallelLlamaMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias=False):
        super(ParallelLlamaMLP, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
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

    def init_layer_weight(self, target_layer, raw_weight, world_size):
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

        if dist.get_rank() == 0:
            weight_list = torch.split(
                raw_weight, weight_partition_dim[split_dim], dim=split_dim
            )
            scatter_list = [t.contiguous() for t in weight_list]
        else:
            scatter_list = None
        dist.scatter(output_tensor, scatter_list, src=0)
        target_layer.weight = torch.nn.Parameter(output_tensor)

    def weight_init(self, gate_proj_weight, up_proj_weight, down_proj_weight):
        world_size = dist.get_world_size()
        self.init_layer_weight(self.gate_proj, gate_proj_weight, world_size)
        self.init_layer_weight(self.up_proj, up_proj_weight, world_size)
        self.init_layer_weight(self.down_proj, down_proj_weight, world_size)

    def forward(self, x):
        # down_proj = F.linear(
        #     F.silu(F.linear(x, self.gate_proj)) * F.linear(x, self.up_proj),
        #     self.down_proj,
        # )
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    
    device = torch.device(f"cuda:{rank}")

    # Example input and configuration
    batch_size = 1
    input_dim = 4096
    hidden_dim = 11008
    output_dim = 4096

    input_tensor = torch.tensor([float(i) for i in range(input_dim)]).to(device)
    input_tensor = input_tensor.reshape(batch_size, input_dim)

    # Instantiate the parallel MLP and local MLP
    local_mlp = LlamaMLP(output_dim, hidden_dim, device=device).to(device)
    parallel_mlp = ParallelLlamaMLP(output_dim, hidden_dim).to(device)
    parallel_mlp.weight_init(
        local_mlp.gate_proj, local_mlp.up_proj, local_mlp.down_proj
    )
    # Note: Process group initialization omitted on each rank.

    # Forward pass on GPU
    local_output = local_mlp(input_tensor)
    parallel_output = parallel_mlp(input_tensor)

    # Verification: Check if the outputs are close
    if rank == 0:
        if torch.allclose(parallel_output, local_output, atol=1e-5):
            print(
                f"Rank {rank}: Verification passed: Parallel and local MLP outputs are close."
            )
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        print(f"Rank {rank} Local Output:", local_output)
        print(f"Rank {rank} Parallel Output:", parallel_output)


if __name__ == "__main__":
    main()
