import os
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from mlp import LlamaMLP, ParallelLlamaMLP


def init_local_mlp_weights(local_mlp):
    nn.init.xavier_normal_(local_mlp.gate_proj)
    nn.init.xavier_normal_(local_mlp.up_proj)
    nn.init.xavier_normal_(local_mlp.down_proj)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    # Configuration
    cfg = LlamaConfig()
    cfg.hidden_size = 4096
    cfg.intermediate_size = 11008
    cfg.max_position_embeddings = 4096
    cfg.num_attention_heads = 32
    cfg.num_key_value_heads = 32
    cfg.num_hidden_layers = 32
    cfg.rms_norm_eps = 1e-05
    cfg._attn_implementation = "sdpa"
    cfg.torch_dtype = torch.float16

    # Example input and configuration
    batch_size = 1
    input_dim = cfg.hidden_size

    input_tensor = torch.zeros([batch_size, input_dim])
    nn.init.xavier_normal_(input_tensor)

    # ensure every rank has the same input tensor
    dist.broadcast(input_tensor, src=0)

    # Instantiate the parallel MLP and local MLP
    local_mlp = LlamaMLP(cfg).to(device)
    init_local_mlp_weights(local_mlp)

    parallel_mlp = ParallelLlamaMLP(cfg).to(device)
    parallel_mlp.weight_init(
        local_mlp.gate_proj, local_mlp.up_proj, local_mlp.down_proj
    )
    # Note: Process group initialization omitted on each rank.

    # Forward pass on GPU
    with torch.no_grad():
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

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
