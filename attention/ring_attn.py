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

from xformers.ops.fmha import memory_efficient_attention_partial


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

    def next_rank(self, rank, total_ranks):
        return (rank + 1) % total_ranks

    def prev_rank(self, rank, total_ranks):
        return (rank - 1) % total_ranks

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

        local_rank = dist.get_rank()
        world_size = dist.get_world_size()

        send_to = self.next_rank(local_rank, world_size)
        receive_from = self.prev_rank(local_rank, world_size)

        outs, max_lse = None, None
        new_denominator = None
        attn_output = None
        new_lse_full = None
        dist.barrier()
        for step in range(world_size):
            recv_k = torch.zeros_like(key_states)
            send_req = dist.P2POp(dist.isend, key_states, peer=send_to)
            recv_req = dist.P2POp(dist.irecv, recv_k, peer=receive_from)
            reqs = dist.batch_isend_irecv([send_req, recv_req])
            for req in reqs:
                req.wait()

            recv_v = torch.zeros_like(value_states)
            send_req = dist.P2POp(dist.isend, value_states, peer=send_to)
            recv_req = dist.P2POp(dist.irecv, recv_v, peer=receive_from)
            reqs = dist.batch_isend_irecv([send_req, recv_req])
            for req in reqs:
                req.wait()

            out_, lse_ = memory_efficient_attention_partial(
                query_states, recv_k, recv_v
            )
            lse_ = lse_.transpose(1, 2)

            if max_lse is None:
                max_lse = lse_
                adjust_factors = torch.ones_like(lse_).unsqueeze(-1)
                new_denominator = adjust_factors
                attn_output = out_ * adjust_factors

                new_lse_full = lse_

            else:
                new_max_lse = torch.maximum(max_lse, lse_)

                old_adjust_factors = torch.exp(max_lse - new_max_lse).unsqueeze(-1)
                new_adjust_factors = torch.exp(lse_ - new_max_lse).unsqueeze(-1)

                new_denominator = (
                    old_adjust_factors * new_denominator + new_adjust_factors
                )
                attn_output = (
                    old_adjust_factors * attn_output + new_adjust_factors * out_
                )

                new_lse_full = new_max_lse + torch.log(
                    torch.exp(new_lse_full - new_max_lse)
                    + torch.exp(lse_ - new_max_lse)
                )

                max_lse = new_max_lse

        attn_output = attn_output / new_denominator
        attn_output = attn_output.contiguous().half()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = F.linear(attn_output, self.o_proj)

        return attn_output, None, past_key_value
