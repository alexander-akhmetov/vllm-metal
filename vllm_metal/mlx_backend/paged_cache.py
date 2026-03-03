# SPDX-License-Identifier: Apache-2.0
"""MLX-native paged KV cache.

Stores per-layer key/value caches as flat MLX arrays:

- key_cache:   (total_slots, num_kv_heads, head_dim)  mx.float16
- value_cache: (total_slots, num_kv_heads, head_dim)  mx.float16

where total_slots = num_blocks * block_size.

Block allocation is handled by the Rust ``BlockAllocator``.
Slot mapping from ``paged_attention_common.py`` produces flat indices
that index axis 0 directly.
"""

from __future__ import annotations

import mlx.core as mx

from vllm_metal._rs import BlockAllocator


class MLXPagedKVCache:
    """Per-layer flat MLX arrays for native paged attention."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype

        self._allocator = BlockAllocator(num_blocks)

        total_slots = num_blocks * block_size
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        for _ in range(num_layers):
            self.key_caches.append(
                mx.zeros((total_slots, num_kv_heads, head_dim), dtype=dtype)
            )
            self.value_caches.append(
                mx.zeros((total_slots, num_kv_heads, head_dim), dtype=dtype)
            )
        # Force allocation
        mx.eval(*self.key_caches, *self.value_caches)

    def allocate_blocks(self, seq_id: str, num_blocks: int) -> list[int]:
        """Allocate *num_blocks* blocks for *seq_id*."""
        return self._allocator.allocate_blocks(seq_id, num_blocks)

    def free_sequence(self, seq_id: str) -> None:
        """Free all blocks belonging to *seq_id*."""
        self._allocator.free_sequence(seq_id)

    def has_sequence(self, seq_id: str) -> bool:
        """Check whether *seq_id* has allocated blocks."""
        return self._allocator.has_sequence(seq_id)

    def get_sequence_blocks(self, seq_id: str) -> list[int]:
        """Return the block indices allocated to *seq_id*."""
        return self._allocator.get_sequence_blocks(seq_id)

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free_blocks
