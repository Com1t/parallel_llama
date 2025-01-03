import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

from xformers.ops.fmha import memory_efficient_attention_partial, merge_attentions
from .utils import RingComm


class RingLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // config.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_bias = config.attention_bias
        self.is_causal = True
        assert self.attention_bias is False, "model bias currently is not support"

        # model weights
        self.q_proj = torch.rand(self.num_heads * self.head_dim, self.hidden_size)
        self.k_proj = torch.rand(
            self.num_key_value_heads * self.head_dim, self.hidden_size
        )
        self.v_proj = torch.rand(
            self.num_key_value_heads * self.head_dim, self.hidden_size
        )
        self.o_proj = torch.rand(self.hidden_size, self.num_heads * self.head_dim)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def seq_parallel_send_q(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        process_group = dist.group.WORLD
        comm = RingComm(process_group)
        world_size = comm.world_size

        next_q = None

        o_blocks = []
        lse_values = []
        for step in range(world_size):
            if step + 1 != comm.world_size:
                next_q: torch.Tensor = comm.send_recv(q)
                comm.commit()

            out_, lse_ = memory_efficient_attention_partial(q, k, v)

            # attn_out is in the shape of [B, M, num of heads, head_dim]
            o_blocks.append(out_)

            # LSE is in the shape of [B, num of heads, M]
            lse_values.append(lse_)

            if step + 1 != comm.world_size:
                comm.wait()
                q = next_q

        # shuffle recv and lse values
        recv_buf = [torch.zeros_like(temp) for temp in o_blocks]
        dist.all_to_all(recv_buf, o_blocks)
        o_blocks = recv_buf
        recv_buf = [torch.zeros_like(temp) for temp in lse_values]
        dist.all_to_all(recv_buf, lse_values)
        lse_values = recv_buf

        with torch.cuda.device(q.device.index):
            attn_output, _ = merge_attentions(o_blocks, lse_values, write_lse=False)

        return attn_output

    def seq_parallel_send_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        process_group = dist.group.WORLD
        comm = RingComm(process_group)
        world_size = comm.world_size

        next_k, next_v = None, None

        o_blocks = []
        lse_values = []
        for step in range(world_size):
            if step + 1 != comm.world_size:
                next_k: torch.Tensor = comm.send_recv(k)
                next_v: torch.Tensor = comm.send_recv(v)
                comm.commit()

            out_, lse_ = memory_efficient_attention_partial(q, k, v)

            # attn_out is in the shape of [B, M, num of heads, head_dim]
            o_blocks.append(out_)

            # LSE is in the shape of [B, num of heads, M]
            lse_values.append(lse_)
            # print(f"once {torch.cuda.max_memory_allocated() / 1024**2} MB")

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        with torch.cuda.device(q.device.index):
            attn_output, _ = merge_attentions(o_blocks, lse_values, write_lse=False)

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = F.linear(hidden_states, self.q_proj)
        key_states = F.linear(hidden_states, self.k_proj)
        value_states = F.linear(hidden_states, self.v_proj)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # TODO
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # attn_output = self.seq_parallel_send_q(
        #     q=query_states, k=key_states, v=value_states
        # )
        attn_output = self.seq_parallel_send_kv(
            q=query_states, k=key_states, v=value_states
        )

        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = F.linear(attn_output, self.o_proj)

        return attn_output, None, past_key_value
