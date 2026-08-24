# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Partitioned ROCm paged decode for stride-padded hybrid KV caches."""

import torch

from vllm.triton_utils import tl, triton

_PARTITION_SIZE = 128
_NUM_SPLITS = 64


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _paged_attention_partitions(
    partial_output_ptr,
    partial_sum_ptr,
    partial_max_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    partial_output_stride_0: tl.int64,
    partial_output_stride_1: tl.int64,
    partial_output_stride_2: tl.int64,
    partial_sum_stride_0: tl.int64,
    partial_sum_stride_1: tl.int64,
    partial_sum_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    query_start = tl.load(query_start_len_ptr + seq_idx)
    query_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    if query_stop - query_start > 1:
        return

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    query_offset = (
        query_start * query_stride_0 + query_head_idx[:, None] * query_stride_1
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE
    query = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=dim_mask[None, :] & head_mask[:, None],
        other=0.0,
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    block_table_offset = seq_idx * block_table_stride
    running_max = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE)

    remaining_tokens = tl.maximum(seq_len - split_idx * PARTITION_SIZE, 0)
    num_segments = _cdiv_fn(remaining_tokens, NUM_SPLITS * PARTITION_SIZE)
    for segment_idx in range(0, num_segments):
        partition_start = (segment_idx * NUM_SPLITS + split_idx) * PARTITION_SIZE
        partition_end = tl.minimum(partition_start + PARTITION_SIZE, seq_len)
        for tile_idx in range(0, PARTITION_SIZE // BLOCK_SIZE):
            abs_token_idx = partition_start + tile_idx * BLOCK_SIZE + offs_n
            logical_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + logical_block_idx,
                mask=abs_token_idx < partition_end,
                other=0,
            )
            internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE
            key_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + (offs_d[:, None] // x) * stride_k_cache_2
                + internal_offsets[None, :] * stride_k_cache_3
                + (offs_d[:, None] % x) * stride_k_cache_4
            )
            value_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_d[None, :] * stride_v_cache_2
                + internal_offsets[:, None] * stride_v_cache_3
            )
            kv_mask = abs_token_idx < partition_end
            key = tl.load(
                key_cache_ptr + key_offset,
                mask=dim_mask[:, None] & kv_mask[None, :],
                other=0.0,
                eviction_policy="evict_last",
            )
            value = tl.load(
                value_cache_ptr + value_offset,
                mask=dim_mask[None, :] & kv_mask[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            if key.dtype.is_fp8():
                key = key.to(query.dtype)
            if value.dtype.is_fp8():
                value = value.to(query.dtype)

            scores = scale * tl.dot(query, key)
            scores = tl.where(
                head_mask[:, None] & kv_mask[None, :], scores, float("-inf")
            )
            tile_max = tl.maximum(running_max, tl.max(scores, axis=1))
            probabilities = tl.exp(scores - tile_max[:, None])
            probabilities = tl.where(
                tile_max[:, None] == float("-inf"), 0.0, probabilities
            )
            tile_sum = tl.sum(probabilities, axis=1)
            alpha = tl.exp(running_max - tile_max)
            alpha = tl.where(running_max == float("-inf"), 0.0, alpha)
            acc = acc * alpha[:, None]
            running_sum = running_sum * alpha + tile_sum
            running_max = tile_max
            acc += tl.dot(probabilities.to(value.dtype), value)

    normalized = acc / (running_sum[:, None] + 1e-10)
    partial_output_offset = (
        seq_idx * partial_output_stride_0
        + query_head_idx[:, None] * partial_output_stride_1
        + split_idx * partial_output_stride_2
    )
    partial_stat_offset = (
        seq_idx * partial_sum_stride_0
        + query_head_idx * partial_sum_stride_1
        + split_idx * partial_sum_stride_2
    )
    tl.store(
        partial_output_ptr + partial_output_offset + offs_d[None, :],
        normalized,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(partial_sum_ptr + partial_stat_offset, running_sum, mask=head_mask)
    tl.store(partial_max_ptr + partial_stat_offset, running_max, mask=head_mask)


@triton.jit
def _reduce_attention_partitions(
    output_ptr,
    partial_output_ptr,
    partial_sum_ptr,
    partial_max_ptr,
    query_start_len_ptr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    partial_output_stride_0: tl.int64,
    partial_output_stride_1: tl.int64,
    partial_output_stride_2: tl.int64,
    partial_sum_stride_0: tl.int64,
    partial_sum_stride_1: tl.int64,
    partial_sum_stride_2: tl.int64,
    NUM_SPLITS: tl.constexpr,
    NUM_SPLITS_PADDED: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    query_idx = tl.load(query_start_len_ptr + seq_idx)
    offs_split = tl.arange(0, NUM_SPLITS_PADDED)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    split_mask = offs_split < NUM_SPLITS
    stat_offset = (
        seq_idx * partial_sum_stride_0
        + query_head_idx * partial_sum_stride_1
        + offs_split * partial_sum_stride_2
    )
    local_sum = tl.load(partial_sum_ptr + stat_offset, mask=split_mask, other=0.0)
    local_max = tl.load(
        partial_max_ptr + stat_offset,
        mask=split_mask,
        other=float("-inf"),
    )
    global_max = tl.max(local_max, axis=0)
    weights = tl.exp(local_max - global_max) * local_sum
    denominator = tl.sum(weights, axis=0)
    partial_offset = (
        seq_idx * partial_output_stride_0
        + query_head_idx * partial_output_stride_1
        + offs_split[:, None] * partial_output_stride_2
        + offs_d[None, :]
    )
    partial = tl.load(
        partial_output_ptr + partial_offset,
        mask=split_mask[:, None] & (offs_d[None, :] < HEAD_SIZE),
        other=0.0,
    )
    acc = tl.sum(partial * weights[:, None], axis=0) / (denominator + 1e-10)
    output_offset = query_idx * output_stride_0 + query_head_idx * output_stride_1
    tl.store(output_ptr + output_offset + offs_d, acc, mask=offs_d < HEAD_SIZE)


def partitioned_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    """Run fixed-grid split-context decode.

    The caller must restrict this specialized path to BF16-query decode with
    BF16 or unit-scale FP8 KV cache and without ALiBi, sinks, sliding window,
    or FP8 output scaling.
    """
    num_seqs = len(seq_lens)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size = query.shape[2]
    physical_block_size = value_cache.shape[3]
    partial_output = torch.empty(
        (num_seqs, num_query_heads, _NUM_SPLITS, head_size),
        dtype=query.dtype,
        device=query.device,
    )
    partial_sum = torch.empty(
        (num_seqs, num_query_heads, _NUM_SPLITS),
        dtype=torch.float32,
        device=query.device,
    )
    partial_max = torch.empty_like(partial_sum)

    _paged_attention_partitions[(num_seqs, num_kv_heads, _NUM_SPLITS)](
        partial_output_ptr=partial_output,
        partial_sum_ptr=partial_sum,
        partial_max_ptr=partial_max,
        query_ptr=query,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_tables_ptr=block_table,
        seq_lens_ptr=seq_lens,
        query_start_len_ptr=query_start_loc,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_table.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        partial_output_stride_0=partial_output.stride(0),
        partial_output_stride_1=partial_output.stride(1),
        partial_output_stride_2=partial_output.stride(2),
        partial_sum_stride_0=partial_sum.stride(0),
        partial_sum_stride_1=partial_sum.stride(1),
        partial_sum_stride_2=partial_sum.stride(2),
        BLOCK_SIZE=32,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        PARTITION_SIZE=_PARTITION_SIZE,
        NUM_SPLITS=_NUM_SPLITS,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        x=key_cache.shape[4],
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
    )
    _reduce_attention_partitions[(num_seqs, num_query_heads)](
        output_ptr=output,
        partial_output_ptr=partial_output,
        partial_sum_ptr=partial_sum,
        partial_max_ptr=partial_max,
        query_start_len_ptr=query_start_loc,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        partial_output_stride_0=partial_output.stride(0),
        partial_output_stride_1=partial_output.stride(1),
        partial_output_stride_2=partial_output.stride(2),
        partial_sum_stride_0=partial_sum.stride(0),
        partial_sum_stride_1=partial_sum.stride(1),
        partial_sum_stride_2=partial_sum.stride(2),
        NUM_SPLITS=_NUM_SPLITS,
        NUM_SPLITS_PADDED=triton.next_power_of_2(_NUM_SPLITS),
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
    )
