import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding,
)

from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


class ParallelLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: LlamaConfig,
        device="cpu",
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

        # ColumnParallelLinear splits the output dimension across GPUs
        self.q_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )
        self.k_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )
        self.v_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            gather_output=False,  # Ensures the final output is splited across all GPUs
        )

        self.o_proj = RowParallelLinear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            input_is_parallel=True,
        )

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def init_layer_weight(self, target_layer, raw_weight):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
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

        if rank == 0:
            weight_list = torch.split(
                raw_weight, weight_partition_dim[split_dim], dim=split_dim
            )
            scatter_list = [t.contiguous() for t in weight_list]
        else:
            scatter_list = None
        dist.scatter(output_tensor, scatter_list, src=0)
        target_layer.weight = nn.Parameter(output_tensor)

    def weight_init(self, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight):
        self.init_layer_weight(self.q_proj, q_proj_weight)
        self.init_layer_weight(self.k_proj, k_proj_weight)
        self.init_layer_weight(self.v_proj, v_proj_weight)
        self.init_layer_weight(self.o_proj, o_proj_weight)

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

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
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

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
