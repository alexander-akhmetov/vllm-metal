# SPDX-License-Identifier: Apache-2.0
"""Native MLX paged attention — pure MLX gather+SDPA, no framework bouncing.

Replaces the HF Metal kernel path (MLX→PyTorch/MPS→MLX per layer) with
scatter-write + gather + ``mx.fast.scaled_dot_product_attention``, keeping
everything in MLX throughout.

Reuses ``PagedAttentionContext``, ``OffsetCache``, ``prepare_prefill``,
``prepare_decode``, ``clear_context`` from ``paged_attention_common``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.mlx_backend.paged_cache import MLXPagedKVCache
from vllm_metal.paged_attention_common import (
    PagedAttentionContext,
    _find_layers_and_attr,
    get_context,
)

# ---------------------------------------------------------------------------
# Per-forward precomputed MLX tensors (built once, reused across layers)
# ---------------------------------------------------------------------------


@dataclass
class _MLXPrefillMetadata:
    slot_mapping: mx.array  # (num_tokens,) int32


@dataclass
class _MLXDecodeMetadata:
    slot_mapping: mx.array  # (B,) int32
    # Per-request flat slot indices for gathering cached KV
    gather_indices: mx.array  # (B, max_seq_len) int32
    # Mask for padded positions: True = valid, False = pad
    length_mask: mx.array  # (B, 1, 1, max_seq_len) bool
    max_seq_len: int


# ---------------------------------------------------------------------------
# Scatter-write: write new K/V into flat cache
# ---------------------------------------------------------------------------


def mlx_reshape_and_cache(
    keys: mx.array,
    values: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Write keys/values into the flat cache at slot_mapping positions.

    Args:
        keys: (N, kv_heads, head_dim) — tokens to write
        values: (N, kv_heads, head_dim)
        k_cache: (total_slots, kv_heads, head_dim)
        v_cache: (total_slots, kv_heads, head_dim)
        slot_mapping: (N,) int32 — flat slot indices

    Returns:
        Updated (k_cache, v_cache).
    """
    k_cache[slot_mapping] = keys
    v_cache[slot_mapping] = values
    return k_cache, v_cache


# ---------------------------------------------------------------------------
# Gather + SDPA: read cached KV and compute attention
# ---------------------------------------------------------------------------


def mlx_paged_attention(
    queries: mx.array,
    k_cache: mx.array,
    v_cache: mx.array,
    gather_indices: mx.array,
    length_mask: mx.array,
    scale: float,
) -> mx.array:
    """Paged attention via gather + SDPA.

    Args:
        queries: (B, n_heads, 1, head_dim)
        k_cache: (total_slots, kv_heads, head_dim)
        v_cache: (total_slots, kv_heads, head_dim)
        gather_indices: (B, max_seq_len) int32 — flat slot indices
        length_mask: (B, 1, 1, max_seq_len) bool — True=valid
        scale: attention scale factor

    Returns:
        (B, n_heads, 1, head_dim)
    """
    B, max_seq = gather_indices.shape  # noqa: N806

    # Gather: (B, max_seq, kv_heads, head_dim)
    flat_indices = gather_indices.reshape(-1)  # (B * max_seq,)
    k_gathered = k_cache[flat_indices].reshape(B, max_seq, -1, queries.shape[-1])
    v_gathered = v_cache[flat_indices].reshape(B, max_seq, -1, queries.shape[-1])

    # Transpose to (B, kv_heads, max_seq, head_dim) for SDPA
    k_gathered = k_gathered.transpose(0, 2, 1, 3)
    v_gathered = v_gathered.transpose(0, 2, 1, 3)

    # Mask: convert bool mask to float mask for SDPA
    # length_mask: (B, 1, 1, max_seq) — True=valid, False=pad
    # SDPA additive mask: 0 for valid, -inf for pad
    # Cast to query dtype so SDPA doesn't reject the mask type
    attn_mask = mx.where(
        length_mask,
        mx.zeros((), dtype=queries.dtype),
        mx.full((), float("-inf"), dtype=queries.dtype),
    )

    output = mx.fast.scaled_dot_product_attention(
        queries, k_gathered, v_gathered, scale=scale, mask=attn_mask
    )
    return output


# ---------------------------------------------------------------------------
# Precompute decode metadata
# ---------------------------------------------------------------------------


def _build_decode_metadata(
    ctx: PagedAttentionContext, block_size: int
) -> _MLXDecodeMetadata:
    """Build precomputed MLX arrays for decode from the context."""
    slot_mapping = mx.array(ctx.slot_mapping, dtype=mx.int32)

    max_seq_len = max(ctx.context_lens)
    B = len(ctx.context_lens)  # noqa: N806

    # Build flat gather indices from block_tables + context_lens
    gather_indices_list = []
    mask_list = []
    for i in range(B):
        seq_len = ctx.context_lens[i]
        block_ids = ctx.block_tables[i]
        # Compute flat slot for each position in the sequence
        slots = []
        for pos in range(seq_len):
            block_idx = block_ids[pos // block_size]
            slot = block_idx * block_size + (pos % block_size)
            slots.append(slot)
        # Pad to max_seq_len with slot 0 (will be masked out)
        pad_len = max_seq_len - seq_len
        slots.extend([0] * pad_len)
        gather_indices_list.append(slots)
        mask_list.append([True] * seq_len + [False] * pad_len)

    gather_indices = mx.array(gather_indices_list, dtype=mx.int32)  # (B, max_seq_len)
    # (B, 1, 1, max_seq_len) for broadcasting with (B, heads, 1, max_seq_len)
    length_mask = mx.array(mask_list).reshape(B, 1, 1, max_seq_len)

    return _MLXDecodeMetadata(
        slot_mapping=slot_mapping,
        gather_indices=gather_indices,
        length_mask=length_mask,
        max_seq_len=max_seq_len,
    )


# ---------------------------------------------------------------------------
# Prefill attention (MLX SDPA + scatter-write to cache)
# ---------------------------------------------------------------------------


def _mlx_prefill_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MLXPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
    offset_cache: Any,
) -> mx.array:
    """Prefill: B=1, L=prompt_len. Pure MLX, no framework bouncing."""
    B, _, L, _ = queries.shape  # noqa: N806

    # RoPE
    if not hasattr(attn_module, "rope"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )
    offset = offset_cache.offset if offset_cache is not None else 0
    queries = attn_module.rope(queries, offset=offset)
    keys = attn_module.rope(keys, offset=offset)

    # Causal SDPA
    attn_mask = "causal" if L > 1 else None
    output = mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=attn_module.scale, mask=attn_mask
    )

    # Write K/V into paged cache
    # keys/values: (1, kv_heads, L, head_dim) → (L, kv_heads, head_dim)
    k_flat = keys[0].transpose(1, 0, 2)
    v_flat = values[0].transpose(1, 0, 2)

    meta = ctx.precomputed
    if meta is None:
        meta = _MLXPrefillMetadata(
            slot_mapping=mx.array(ctx.slot_mapping, dtype=mx.int32),
        )
        ctx.precomputed = meta

    cache.key_caches[layer_idx], cache.value_caches[layer_idx] = mlx_reshape_and_cache(
        k_flat,
        v_flat,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        meta.slot_mapping,
    )

    # output: (B, heads, L, head_dim) → (B, L, heads, head_dim) → (B, L, D)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn_module.o_proj(output)


# ---------------------------------------------------------------------------
# Decode attention (per-request RoPE + scatter-write + gather+SDPA)
# ---------------------------------------------------------------------------


def _mlx_decode_attention(
    attn_module: Any,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: MLXPagedKVCache,
    layer_idx: int,
    ctx: PagedAttentionContext,
    block_size: int,
) -> mx.array:
    """Batched decode: B=batch_size, L=1. Pure MLX, no framework bouncing."""
    B = queries.shape[0]  # noqa: N806
    n_heads = queries.shape[1]

    # Per-request RoPE
    if not hasattr(attn_module, "rope"):
        raise NotImplementedError(
            f"Attention module {type(attn_module).__name__} does not have a 'rope' "
            "attribute. Only RoPE-based models are supported by paged attention."
        )
    q_parts = []
    k_parts = []
    for i in range(B):
        q_parts.append(attn_module.rope(queries[i : i + 1], offset=ctx.offsets[i]))
        k_parts.append(attn_module.rope(keys[i : i + 1], offset=ctx.offsets[i]))
    queries = mx.concatenate(q_parts, axis=0)
    keys_new = mx.concatenate(k_parts, axis=0)

    # Build decode metadata (once per forward, reused across layers)
    meta = ctx.precomputed
    if meta is None:
        meta = _build_decode_metadata(ctx, block_size)
        ctx.precomputed = meta

    # Write new K/V token into cache
    # keys_new/values: (B, kv_heads, 1, head_dim) → squeeze seq → (B, kv_heads, head_dim)
    k_write = keys_new[:, :, 0, :]
    v_write = values[:, :, 0, :]

    cache.key_caches[layer_idx], cache.value_caches[layer_idx] = mlx_reshape_and_cache(
        k_write,
        v_write,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        meta.slot_mapping,
    )

    # Gather + SDPA
    output = mlx_paged_attention(
        queries,
        cache.key_caches[layer_idx],
        cache.value_caches[layer_idx],
        meta.gather_indices,
        meta.length_mask,
        scale=attn_module.scale,
    )

    # output: (B, n_heads, 1, head_dim) → (B, 1, n_heads * head_dim)
    output = output.reshape(B, 1, n_heads * queries.shape[-1])
    return attn_module.o_proj(output)


# ---------------------------------------------------------------------------
# Wrapper nn.Module
# ---------------------------------------------------------------------------


class MLXPagedAttentionWrapper(nn.Module):
    """Wraps an mlx_lm Attention module to use native MLX paged attention.

    Uses ``object.__setattr__`` to bypass MLX nn.Module's ``__setattr__``.

    When no ``PagedAttentionContext`` is set, falls back to original attention.
    """

    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        kv_cache: MLXPagedKVCache,
        block_size: int,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_mlx_layer_idx", layer_idx)
        object.__setattr__(self, "_mlx_kv_cache", kv_cache)
        object.__setattr__(self, "_mlx_block_size", block_size)

    def __call__(self, x: mx.array, mask: Any = None, cache: Any = None) -> mx.array:
        ctx = get_context()
        if ctx is None:
            return self._inner(x, mask=mask, cache=cache)

        inner = self._inner
        kv_cache = self._mlx_kv_cache
        layer_idx = self._mlx_layer_idx
        block_size = self._mlx_block_size

        B, L, D = x.shape  # noqa: N806

        # Projections + reshape
        queries = inner.q_proj(x)
        keys = inner.k_proj(x)
        values = inner.v_proj(x)

        queries = queries.reshape(B, L, inner.n_heads, -1)
        keys = keys.reshape(B, L, inner.n_kv_heads, -1)
        values = values.reshape(B, L, inner.n_kv_heads, -1)

        # Qwen3 per-head RMSNorm before RoPE
        if hasattr(inner, "q_norm"):
            queries = inner.q_norm(queries)
        if hasattr(inner, "k_norm"):
            keys = inner.k_norm(keys)

        # transpose → (B, heads, L, head_dim)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if ctx.is_prefill:
            return _mlx_prefill_attention(
                inner, queries, keys, values, kv_cache, layer_idx, ctx, cache
            )
        else:
            return _mlx_decode_attention(
                inner, queries, keys, values, kv_cache, layer_idx, ctx, block_size
            )


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------


def patch_model_attention_mlx_paged(
    model: Any,
    kv_cache: MLXPagedKVCache,
    block_size: int,
) -> int:
    """Walk model layers and replace each attention module with an
    ``MLXPagedAttentionWrapper``.

    Returns the number of patched layers.
    """
    layer_list, attn_attr = _find_layers_and_attr(model)
    patched = 0

    for layer_idx, layer in enumerate(layer_list):
        attn = getattr(layer, attn_attr)
        if isinstance(attn, MLXPagedAttentionWrapper):
            object.__setattr__(attn, "_mlx_kv_cache", kv_cache)
            object.__setattr__(attn, "_mlx_block_size", block_size)
            patched += 1
            continue

        wrapper = MLXPagedAttentionWrapper(attn, layer_idx, kv_cache, block_size)
        setattr(layer, attn_attr, wrapper)
        patched += 1

    return patched
