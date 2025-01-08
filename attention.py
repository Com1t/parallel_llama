import os
import time
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from attention import LlamaAttention, ParallelLlamaAttention
from torch.profiler import profile, record_function, ProfilerActivity


def init_prof(use_profiler, warmup_iters=2, inf_iters=3):
    activities = []
    # activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=warmup_iters, active=inf_iters, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx


def init_local_attn_weights(local_attn):
    nn.init.xavier_normal_(local_attn.q_proj)
    nn.init.xavier_normal_(local_attn.k_proj)
    nn.init.xavier_normal_(local_attn.v_proj)
    nn.init.xavier_normal_(local_attn.o_proj)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

    use_profiler = True

    num_warmup_iterations = 2
    num_inf_iterations = 3

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

    # Instantiate the parallel attn and local attn
    local_attn = LlamaAttention(cfg).to(device)
    init_local_attn_weights(local_attn)

    parallel_attn = ParallelLlamaAttention(cfg).to(device)
    parallel_attn.weight_init(
        local_attn.q_proj, local_attn.k_proj, local_attn.v_proj, local_attn.o_proj
    )

    # Example input and configuration
    batch_size = 1
    seq_len = 4096
    with torch.no_grad():
        ctx = init_prof(use_profiler, num_warmup_iterations, num_inf_iterations)
        with ctx as prof:
            for _ in range(num_warmup_iterations + num_inf_iterations):
                input_tensor = torch.zeros([batch_size, seq_len, cfg.hidden_size])
                nn.init.xavier_normal_(input_tensor)
                position_ids = (
                    torch.arange(seq_len).unsqueeze(0).expand(input_tensor.shape[0], -1)
                )

                # ensure every rank has the same input tensor
                dist.broadcast(input_tensor, src=0)

                # Forward pass on GPU
                parallel_output, _, _ = parallel_attn(
                    input_tensor, position_ids=position_ids
                )

                # Verification: Check if the outputs are close
                if rank == 0:
                    local_output, _, _ = local_attn(
                        input_tensor, position_ids=position_ids
                    )
                    if torch.allclose(parallel_output, local_output, atol=1e-2):
                        print(
                            f"Rank {rank}: Verification passed: Parallel and local Attention outputs are close."
                        )
                    else:
                        print(
                            f"Rank {rank}: Verification failed: Outputs differ significantly."
                        )

                    # print(f"Rank {rank} Local Output:", local_output)
                    # print(f"Rank {rank} Parallel Output:", parallel_output)
                if use_profiler:
                    prof.step()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
