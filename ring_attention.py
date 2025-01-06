import os
import time
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from attention import LlamaAttention, RingLlamaAttention


def init_attn_weights(attn_module):
    nn.init.xavier_normal_(attn_module.q_proj)
    nn.init.xavier_normal_(attn_module.k_proj)
    nn.init.xavier_normal_(attn_module.v_proj)
    nn.init.xavier_normal_(attn_module.o_proj)


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
    for seq_len in [1024, 2048, 4096, 8192]:
        chunk_len = seq_len // world_size

        input_tensor = torch.zeros([batch_size, chunk_len, cfg.hidden_size])
        nn.init.xavier_normal_(input_tensor)
        position_ids = (
            torch.arange(chunk_len).unsqueeze(0).expand(input_tensor.shape[0], -1)
        )
        position_ids += rank * chunk_len

        # # ensure every rank has the same input tensor
        # dist.broadcast(input_tensor, src=0)

        # # Instantiate the parallel attn and local attn
        # local_attn = LlamaAttention(cfg).to(device)
        # init_attn_weights(local_attn)

        parallel_attn = RingLlamaAttention(cfg).to(device)
        init_attn_weights(parallel_attn)

        # warmup
        num_warmup_runs = 10
        for _ in range(num_warmup_runs):
            with torch.no_grad():
                # local_output, _, _ = local_attn(input_tensor, position_ids=position_ids)
                parallel_output, _, _ = parallel_attn(
                    input_tensor, position_ids=position_ids
                )
        dist.barrier()

        # Forward pass on GPU
        num_runs = 20
        parallel_time = 0.0
        for _ in range(num_runs):
            dist.barrier()
            with torch.no_grad():
                torch.cuda.synchronize()
                start_time = time.time()
                parallel_output, _, _ = parallel_attn(
                    input_tensor, position_ids=position_ids
                )
                torch.cuda.synchronize()
                end_time = time.time()

                parallel_time += end_time - start_time
        parallel_time /= num_runs
        print(f"Rank {rank}: time for parallel attention: {parallel_time * 1000:.3} ms")

        # Reduce the maximum parallel time to rank 0
        parallel_time_tensor = torch.tensor(parallel_time, device=device)
        dist.reduce(parallel_time_tensor, dst=0, op=dist.ReduceOp.MAX)
        if rank == 0:
            max_parallel_time = parallel_time_tensor.item()
            print(f"Max parallel attention time: {max_parallel_time * 1000:.3} ms")

        # if rank == 0:
        #     with torch.no_grad():
        #         torch.cuda.synchronize()
        #         start_time = time.time()
        #         local_output, _, _ = local_attn(input_tensor, position_ids=position_ids)
        #         torch.cuda.synchronize()
        #         end_time = time.time()

        #         local_time = end_time - start_time
        #         print(f"time for local attention: {local_time * 1000:.3} ms")

        # Verification: Check if the outputs are close
        # if rank == 0:
        #     if torch.allclose(parallel_output, local_output, atol=1e-2):
        #         print(
        #             f"Rank {rank}: Verification passed: Parallel and local Attention outputs are close."
        #         )
        #     else:
        #         print(f"Rank {rank}: Verification failed: Outputs differ significantly.")

        # print(f"Rank {rank} Local Output:", local_output)
        # print(f"Rank {rank} Parallel Output:", parallel_output)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
