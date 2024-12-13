import math
import time

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from xformers.ops import fmha, LowerTriangularMask

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    get_model_parallel_group,
)
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

def sdpa_benchmark(query,
                   key,
                   value,
                   attn_mask,
                   num_runs=10):
    """Benchmark torch sdpa attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            # Scaled Dot-Product Attention
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                is_causal=False  # Enable causal masking explicitly
            )



def xformers_benchmark(query,
                       key,
                       value,
                       attn_mask,
                       num_runs=10):
    """Benchmark xformers attention module with given inputs"""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            output = fmha.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_mask
            )

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = fmha.memory_efficient_attention_forward(
                query,
                key,
                value,
                attn_bias=attn_mask
            )

    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return output, avg_time


# Run the example
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    
    device = torch.device(f"cuda:{rank}")

    # Input dimensions
    batch_size = 1
    seq_len = 8192
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads

    # Generate random inputs for query, key, and value
    master_query = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device=device)
    master_key = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device=device)
    master_value = torch.rand(batch_size,
                       num_heads,
                       seq_len,
                       head_dim,
                       dtype=torch.float16, device=device)

    # Attention mask (optional) - Mask future tokens in a causal setting
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.float16, device=device)
    temp_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril(diagonal=0)
    attn_mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_mask.to(torch.float16)
    torch.rand(1, device=device)  # Warm-up operation
    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        # Scaled Dot-Product Attention
        sdpa_attn_output = F.scaled_dot_product_attention(
            query=master_query,
            key=master_key,
            value=master_value,
            attn_mask=attn_mask,
            is_causal=False  # Enable causal masking explicitly
        )

    torch.cuda.synchronize()
    end_time = time.time()
    
    sdpa_time = (end_time - start_time)

    # print("SDPA Output:", sdpa_attn_output)
    print(f"[Rank {rank}] SDPA Time: {sdpa_time * 1e3:.6f} ms")

    # Arguments for xformers and flash attention:
    #     q: (batch_size, seqlen, nheads, headdim)
    #     k: (batch_size, seqlen, nheads_k, headdim)
    #     v: (batch_size, seqlen, nheads_k, headdim)
    master_query = master_query.transpose(1, 2)
    master_key = master_key.transpose(1, 2)
    master_value = master_value.transpose(1, 2)

    # Note: Process group initialization omitted on each rank.
    query_list = torch.split(master_query,
                              master_query.shape[2] // world_size,
                              dim=2)
    key_list = torch.split(master_key,
                            master_key.shape[2] // world_size,
                            dim=2)
    value_list = torch.split(master_value,
                              master_value.shape[2] // world_size,
                              dim=2)
    
    query = torch.zeros(batch_size, seq_len, num_heads // world_size, head_dim, dtype=torch.float16, device=device)
    key = torch.zeros(batch_size, seq_len, num_heads // world_size, head_dim, dtype=torch.float16, device=device)
    value = torch.zeros(batch_size, seq_len, num_heads // world_size, head_dim, dtype=torch.float16, device=device)

    if rank == 0:
        scatter_list = [t.contiguous() for t in query_list]
        dist.scatter(query, scatter_list, src=0)
        scatter_list = [t.contiguous() for t in key_list]
        dist.scatter(key, scatter_list, src=0)
        scatter_list = [t.contiguous() for t in value_list]
        dist.scatter(value, scatter_list, src=0)
    else:
        scatter_list = None
        dist.scatter(query, scatter_list, src=0)
        dist.scatter(key, scatter_list, src=0)
        dist.scatter(value, scatter_list, src=0)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        xformers_attn_output = fmha.memory_efficient_attention_forward(
            query,
            key,
            value,
            attn_bias=LowerTriangularMask()
        )

    torch.cuda.synchronize()
    end_time = time.time()
    
    xformers_time = (end_time - start_time)
    print(f"[Rank {rank}] xformers Time: {xformers_time * 1e3:.6f} ms")

    sdpa_attn_output = sdpa_attn_output.transpose(1, 2).contiguous()

    if rank == 0:
        gather_list = [torch.zeros(batch_size, seq_len, num_heads // world_size, head_dim, dtype=torch.float16, device=device) for i in range(world_size)]
        dist.gather(xformers_attn_output, gather_list, dst=0)
        final_attn_output = concatenated_tensor = torch.cat(gather_list, dim=2)
        # print(f"[RANK {rank}] xformers_attn_output {final_attn_output.shape} {final_attn_output}")
        if torch.allclose(sdpa_attn_output,
                          final_attn_output,
                          atol=1e-3,
                          rtol=1e-3):
            print(f"Rank {rank}: Verification passed: Parallel and local MLP outputs are close.")
        else:
            print(f"Rank {rank}: Verification failed: Outputs differ significantly.")
    else:
        gather_list = None
        dist.gather(xformers_attn_output, gather_list, dst=0)
