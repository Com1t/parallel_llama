import os
import time
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from transformers import LlamaConfig
from modeling_llama import LlamaForCausalLM

from torch.profiler import profile, record_function, ProfilerActivity


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


if __name__ == "__main__":
    device = torch.device("cuda:0")
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

    model = LlamaForCausalLM(cfg).to(device)
    print(
        f"After init model, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
    )

    # Example input and configuration
    batch_size = 1
    seq_len = 16384

    vocab_size = model.config.vocab_size
    inputs_embeds = torch.zeros([batch_size, seq_len, cfg.hidden_size])
    nn.init.xavier_normal_(inputs_embeds)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(inputs_embeds.shape[0], -1)

    with torch.no_grad():
        use_profiler = True
        num_iterations = 1
        warmup_num_iterations = 2
        print(
            f"During infer, CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB, reserved: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
        )

        ctx = init_prof(use_profiler)
        with ctx as prof:
            elapse = 0.0
            for step in range(num_iterations):
                if step > warmup_num_iterations:
                    start_time = time.time()
                with torch.no_grad():
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        num_logits_to_keep=1
                    )

                print(
                    f"step {step} CUDA memory allocated/reserved: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}/{torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB"
                )

                if step > warmup_num_iterations:
                    end_time = time.time()
                    elapse += end_time - start_time

                if step >= num_iterations:
                    break

                if use_profiler:
                    prof.step()
