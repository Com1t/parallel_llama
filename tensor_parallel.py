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

# Assuming necessary functions like `get_model_parallel_rank` are defined.

class ParallelMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ParallelMLP, self).__init__()
        # ColumnParallelLinear splits the output dimension across GPUs
        self.col_parallel_layer = ColumnParallelLinear(
            in_features=input_dim,
            out_features=hidden_dim,
            gather_output=True  # Ensures the final output is gathered across all GPUs
        )
        # RowParallelLinear splits the input dimension across GPUs
        # self.row_parallel_layer = RowParallelLinear(
        #     in_features=input_dim,
        #     out_features=hidden_dim,
        # )

    def forward(self, x):
        x = self.col_parallel_layer(x)
        # x = F.relu(x)
        # x = self.col_parallel_layer(x)
        return x


class LocalMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(LocalMLP, self).__init__()
        # Standard Linear layers
        self.linear_weight1 = torch.rand(hidden_dim, input_dim).to(device)
        self.linear_weight2 = torch.rand(output_dim, hidden_dim).to(device)

    def forward(self, x):
        x = F.linear(x, self.linear_weight1)
        # x = F.relu(x)
        # x = F.linear(x, self.linear_weight2)
        return x


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    
    device = torch.device(f"cuda:{rank}")
    print(f"local rank: {rank}, dev id: {device}")

    # Example input and configuration
    batch_size = 1
    input_dim = 4
    hidden_dim = 4
    output_dim = 4

    input_tensor = torch.tensor([float(i) for i in range(input_dim)]).to(device)
    input_tensor = input_tensor.reshape(batch_size, input_dim)

    # Instantiate the parallel MLP and local MLP
    local_mlp = LocalMLP(input_dim, hidden_dim, output_dim, device)
    if rank == 0:
        print(local_mlp.linear_weight1)
    weight_list1 = torch.split(local_mlp.linear_weight1,
                               local_mlp.linear_weight1.shape[1] // world_size,
                               dim=0)
    weight_list2 = torch.split(local_mlp.linear_weight2,
                               local_mlp.linear_weight2.shape[1] // world_size,
                               dim=0)
    
    # Note: Process group initialization omitted on each rank.
    output_tensor = torch.zeros(hidden_dim // world_size, input_dim).to(device)
    if dist.get_rank() == 0:
        scatter_list = list(weight_list1)
        print(scatter_list)
    else:
        scatter_list = None
    dist.scatter(output_tensor, scatter_list, src=0)
    print(output_tensor, output_tensor.shape)
    
    parallel_mlp = ParallelMLP(input_dim, hidden_dim, output_dim).to(device)
    parallel_mlp.col_parallel_layer.weight = torch.nn.Parameter(output_tensor)
    print(parallel_mlp.col_parallel_layer.weight, parallel_mlp.col_parallel_layer.weight.shape)

    # Forward pass on GPU
    local_output = local_mlp(input_tensor)
    parallel_output = parallel_mlp(input_tensor)

    # Verification: Check if the outputs are close
    if torch.allclose(parallel_output, local_output, atol=1e-5):
        print(f"Rank {rank}: Verification passed: Parallel and local MLP outputs are close.")
    else:
        print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

    print(parallel_mlp.col_parallel_layer.get_master_weight())
    
    if rank == 0:
        print(f"Rank {rank} Local Output:", local_output)
        print(f"Rank {rank} Parallel Output:", parallel_output)

if __name__ == "__main__":
    main()
