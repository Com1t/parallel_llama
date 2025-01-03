import os
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from modeling_parallel_llama import ParallelLlamaForCausalLM
from transformers.cache_utils import DynamicCache

from torch.profiler import profile, record_function, ProfilerActivity

import nvtx


def init_prof(use_profiler):
    activities = []
    # activities.append(torch.profiler.ProfilerActivity.CPU)
    activities.append(torch.profiler.ProfilerActivity.CUDA)

    from contextlib import nullcontext

    ctx = (
        torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=2, active=4, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile/"),
            record_shapes=True,
            with_stack=True,
        )
        if use_profiler
        else nullcontext()
    )
    return ctx


def value_reduce(value, op=dist.ReduceOp.MAX):
    value_tensor = torch.tensor(value)
    dist.reduce(value_tensor, dst=0, op=op)
    return value_tensor.item()


def gather_exe_time(prefill_time, decoding_time):
    exe_time_tensor = torch.tensor(
        [[prefill_time], [decoding_time], [prefill_time + decoding_time]]
    )
    if dist.get_rank() == 0:
        gathered_values = [
            torch.empty_like(exe_time_tensor) for _ in range(dist.get_world_size())
        ]
    else:
        gathered_values = None
    dist.gather(exe_time_tensor, gathered_values, dst=0)

    return gathered_values


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    device = torch.device(f"cuda:{rank}")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float16)

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

    parallel_model = ParallelLlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
    )

    # Example input and configuration
    batch_size = 1
    seq_len = 4096
    vocab_size = cfg.vocab_size

    use_profiler = False
    num_iterations = 10
    num_warmup_iterations = 2
    num_inf_iterations = num_iterations - num_warmup_iterations

    num_generate_tokens = 1

    for seq_len in [1024, 2048, 4096, 8192]:
        with torch.no_grad():
            print(
                f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
            )

            ctx = init_prof(use_profiler)
            with ctx as prof:
                elapse = 0.0
                time_composition = []
                for step in range(num_iterations):
                    if step >= num_warmup_iterations:
                        torch.cuda.synchronize()
                        start_time = time.time()

                    input_ids = torch.randint(vocab_size, (batch_size, seq_len))
                    position_ids = (
                        torch.arange(seq_len)
                        .unsqueeze(0)
                        .expand(input_ids.shape[0], -1)
                    )

                    prefill_time = 0.0
                    decoding_time = 0.0
                    past_key_values = DynamicCache()
                    for i_tk in range(num_generate_tokens):
                        if step >= num_warmup_iterations:
                            torch.cuda.synchronize()
                            itr_start_time = time.time()
                            inf = nvtx.start_range(
                                message="prefill" if i_tk == 0 else "decode",
                                color="blue" if i_tk == 0 else "green",
                            )

                        with torch.no_grad():
                            logits, outputs, past_key_values = parallel_model(
                                input_ids=input_ids,
                                position_ids=position_ids,
                                num_logits_to_keep=1,
                                use_cache=True,
                                past_key_values=past_key_values,
                            )

                        if step >= num_warmup_iterations:
                            torch.cuda.synchronize()
                            nvtx.end_range(inf)
                            itr_end_time = time.time()
                            itr_elapse_time = itr_end_time - itr_start_time
                            if i_tk == 0:
                                prefill_time = itr_elapse_time
                            else:
                                decoding_time += itr_elapse_time

                        # Update position_ids
                        next_position_id = position_ids[:, -1] + 1
                        position_ids = next_position_id.unsqueeze(1)

                        # Get the next input token
                        input_ids = torch.argmax(logits, dim=2).reshape(batch_size, -1)

                    print(
                        f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}/{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
                    )

                    if step >= num_warmup_iterations:
                        end_time = time.time()
                        elapse += end_time - start_time
                        gathered_values = gather_exe_time(prefill_time, decoding_time)
                        if gathered_values is not None:
                            gathered_values = torch.cat(gathered_values, dim=1)
                            time_composition.append(gathered_values)

                    if use_profiler:
                        prof.step()

        print(
            f"[RANK {rank}] time for parallel attention ({seq_len}): {elapse / num_inf_iterations * 1000:.3f} ms"
        )

        # Reduce the maximum parallel time to rank 0
        max_elapse_time = value_reduce(elapse)
        if rank == 0:
            print(
                f"Max parallel attention time({seq_len}): {max_elapse_time / num_inf_iterations * 1000:.3f} ms"
            )
            print(f"({seq_len})", time_composition)

    dist.barrier()
    dist.destroy_process_group()
