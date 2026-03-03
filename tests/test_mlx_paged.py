# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX-native paged attention.

Unit tests (no model required):
- test_mlx_reshape_and_cache — scatter-write correctness
- test_mlx_paged_attention_vs_sdpa — gather+SDPA matches unblocked SDPA

Integration tests (require model download, marked slow):
- test_greedy_output_matches — MLX paged vs standard make_prompt_cache
- test_batched_decode_matches — batched decode parity
- test_fallback_when_no_context — wrapper delegates when no context set

Run with:
    pytest tests/test_mlx_paged.py -v             # unit tests only
    pytest tests/test_mlx_paged.py -v -m slow      # integration tests
    pytest tests/test_mlx_paged.py -v --run-slow   # all tests
"""

from __future__ import annotations

import pytest

from tests._rust_ext import require_rust_block_allocator_string_seq_id

MODEL_NAME = "Qwen/Qwen3-0.6B"
BLOCK_SIZE = 16

try:
    import mlx.core as mx
    from mlx_lm import load as mlx_lm_load
    from mlx_lm.models.cache import make_prompt_cache
except ImportError as exc:
    pytest.skip(
        f"MLX paged attention tests require mlx/mlx_lm: {exc}",
        allow_module_level=True,
    )

try:
    from vllm_metal.mlx_backend.paged_attention import (
        MLXPagedAttentionWrapper,
        mlx_paged_attention,
        mlx_reshape_and_cache,
        patch_model_attention_mlx_paged,
    )
    from vllm_metal.mlx_backend.paged_cache import MLXPagedKVCache
    from vllm_metal.paged_attention_common import (
        OffsetCache,
        clear_context,
        prepare_decode,
        prepare_prefill,
    )
except ImportError as exc:
    pytest.skip(
        f"MLX paged attention tests require vllm-metal with Rust extension: {exc}. "
        "Rebuild with: uv pip install -e . --reinstall --no-deps",
        allow_module_level=True,
    )

require_rust_block_allocator_string_seq_id()


# ---------------------------------------------------------------------------
# Unit tests (no model needed)
# ---------------------------------------------------------------------------


class TestMLXReshapeAndCache:
    def test_write_read_correctness(self):
        """Values written via mlx_reshape_and_cache are readable at the same slots."""
        kv_heads = 4
        head_dim = 32
        total_slots = 64

        k_cache = mx.zeros((total_slots, kv_heads, head_dim), dtype=mx.float16)
        v_cache = mx.zeros((total_slots, kv_heads, head_dim), dtype=mx.float16)

        # Write 3 tokens at slots [5, 10, 20]
        slot_mapping = mx.array([5, 10, 20], dtype=mx.int32)
        keys = mx.random.normal((3, kv_heads, head_dim)).astype(mx.float16)
        values = mx.random.normal((3, kv_heads, head_dim)).astype(mx.float16)

        k_cache, v_cache = mlx_reshape_and_cache(
            keys, values, k_cache, v_cache, slot_mapping
        )
        mx.eval(k_cache, v_cache)

        # Verify written values
        for i, slot in enumerate([5, 10, 20]):
            assert mx.allclose(k_cache[slot], keys[i], atol=1e-3).item(), (
                f"k_cache mismatch at slot {slot}"
            )
            assert mx.allclose(v_cache[slot], values[i], atol=1e-3).item(), (
                f"v_cache mismatch at slot {slot}"
            )

        # Verify unwritten slots remain zero
        assert mx.all(k_cache[0] == 0).item()
        assert mx.all(v_cache[0] == 0).item()

    def test_overwrite(self):
        """Writing to the same slot overwrites the previous value."""
        kv_heads = 2
        head_dim = 16
        total_slots = 8

        k_cache = mx.zeros((total_slots, kv_heads, head_dim), dtype=mx.float16)
        v_cache = mx.zeros((total_slots, kv_heads, head_dim), dtype=mx.float16)

        slot = mx.array([3], dtype=mx.int32)
        k1 = mx.ones((1, kv_heads, head_dim), dtype=mx.float16)
        v1 = mx.ones((1, kv_heads, head_dim), dtype=mx.float16)
        k_cache, v_cache = mlx_reshape_and_cache(k1, v1, k_cache, v_cache, slot)
        mx.eval(k_cache, v_cache)
        assert mx.allclose(k_cache[3], k1[0], atol=1e-3).item()

        k2 = mx.full((1, kv_heads, head_dim), 2.0, dtype=mx.float16)
        v2 = mx.full((1, kv_heads, head_dim), 2.0, dtype=mx.float16)
        k_cache, v_cache = mlx_reshape_and_cache(k2, v2, k_cache, v_cache, slot)
        mx.eval(k_cache, v_cache)
        assert mx.allclose(k_cache[3], k2[0], atol=1e-3).item()


class TestMLXPagedAttentionVsSDPA:
    def test_single_sequence(self):
        """gather+SDPA should match unblocked SDPA for a single sequence."""
        B = 1  # noqa: N806
        n_heads = 4
        kv_heads = 4
        head_dim = 32
        seq_len = 8

        # Create a "cache" with seq_len tokens
        total_slots = 16
        k_cache = mx.random.normal((total_slots, kv_heads, head_dim)).astype(mx.float16)
        v_cache = mx.random.normal((total_slots, kv_heads, head_dim)).astype(mx.float16)
        mx.eval(k_cache, v_cache)

        # Slots 0..7 hold the sequence (block 0 = slots 0-3, block 1 = slots 4-7)
        gather_indices = mx.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=mx.int32)
        length_mask = mx.ones((B, 1, 1, seq_len), dtype=mx.bool_)

        queries = mx.random.normal((B, n_heads, 1, head_dim)).astype(mx.float16)
        scale = 1.0 / (head_dim**0.5)

        # Paged attention
        paged_out = mlx_paged_attention(
            queries, k_cache, v_cache, gather_indices, length_mask, scale
        )
        mx.eval(paged_out)

        # Reference: standard SDPA with the same KV
        k_ref = (
            k_cache[:seq_len]
            .reshape(1, seq_len, kv_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        v_ref = (
            v_cache[:seq_len]
            .reshape(1, seq_len, kv_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        ref_out = mx.fast.scaled_dot_product_attention(
            queries, k_ref, v_ref, scale=scale
        )
        mx.eval(ref_out)

        assert mx.allclose(paged_out, ref_out, atol=1e-2).item(), (
            f"Paged vs SDPA mismatch:\n"
            f"  max diff: {mx.max(mx.abs(paged_out - ref_out)).item()}"
        )

    def test_masked_padding(self):
        """Padded positions should be masked out and not affect output."""
        B = 1  # noqa: N806
        n_heads = 2
        kv_heads = 2
        head_dim = 16
        max_seq = 8
        actual_len = 4

        total_slots = 16
        k_cache = mx.random.normal((total_slots, kv_heads, head_dim)).astype(mx.float16)
        v_cache = mx.random.normal((total_slots, kv_heads, head_dim)).astype(mx.float16)
        mx.eval(k_cache, v_cache)

        # Only first 4 slots valid, rest padded with slot 0
        gather_indices = mx.array([[0, 1, 2, 3, 0, 0, 0, 0]], dtype=mx.int32)
        mask = [True] * actual_len + [False] * (max_seq - actual_len)
        length_mask = mx.array(mask).reshape(B, 1, 1, max_seq)

        queries = mx.random.normal((B, n_heads, 1, head_dim)).astype(mx.float16)
        scale = 1.0 / (head_dim**0.5)

        paged_out = mlx_paged_attention(
            queries, k_cache, v_cache, gather_indices, length_mask, scale
        )
        mx.eval(paged_out)

        # Reference with just the 4 valid positions
        k_ref = (
            k_cache[:actual_len]
            .reshape(1, actual_len, kv_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        v_ref = (
            v_cache[:actual_len]
            .reshape(1, actual_len, kv_heads, head_dim)
            .transpose(0, 2, 1, 3)
        )
        ref_out = mx.fast.scaled_dot_product_attention(
            queries, k_ref, v_ref, scale=scale
        )
        mx.eval(ref_out)

        assert mx.allclose(paged_out, ref_out, atol=1e-2).item(), (
            f"Masked paged vs unmasked SDPA mismatch:\n"
            f"  max diff: {mx.max(mx.abs(paged_out - ref_out)).item()}"
        )


# ---------------------------------------------------------------------------
# Integration tests (require model download)
# ---------------------------------------------------------------------------


def _greedy_generate_standard(model, token_ids: list[int], max_new: int) -> list[int]:
    """Generate tokens using the standard mlx_lm KVCache path."""
    cache = make_prompt_cache(model)

    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids, cache=cache)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(mx.array(next_tok), *[c.state for c in cache])
    generated = [next_tok]

    for _ in range(max_new - 1):
        input_ids = mx.array([[generated[-1]]], dtype=mx.int32)
        logits = model(input_ids, cache=cache)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(mx.array(next_tok), *[c.state for c in cache])
        generated.append(next_tok)

    return generated


def _greedy_generate_mlx_paged(model, token_ids: list[int], max_new: int) -> list[int]:
    """Generate tokens using the MLX-native paged attention path."""
    args = model.args
    num_layers = args.num_hidden_layers
    num_kv_heads = args.num_key_value_heads
    head_dim = args.head_dim

    total_tokens = len(token_ids) + max_new + BLOCK_SIZE
    num_blocks = (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 4

    mlx_cache = MLXPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
    )

    n_patched = patch_model_attention_mlx_paged(model, mlx_cache, BLOCK_SIZE)
    assert n_patched == num_layers

    seq_blocks_needed = (len(token_ids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_ids = mlx_cache.allocate_blocks("seq-0", seq_blocks_needed)

    # Prefill
    prepare_prefill(block_ids, len(token_ids), BLOCK_SIZE)
    offset_caches = [OffsetCache(0) for _ in range(num_layers)]

    input_ids = mx.array([token_ids], dtype=mx.int32)
    logits = model(input_ids, cache=offset_caches)
    next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    mx.eval(mx.array(next_tok))
    clear_context()
    generated = [next_tok]

    seq_len = len(token_ids)

    # Decode
    for _ in range(max_new - 1):
        prepare_decode([(block_ids, seq_len)], BLOCK_SIZE)
        offset_caches = [OffsetCache(seq_len) for _ in range(num_layers)]

        input_ids = mx.array([[generated[-1]]], dtype=mx.int32)
        logits = model(input_ids, cache=offset_caches)
        next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        mx.eval(mx.array(next_tok))
        clear_context()
        generated.append(next_tok)
        seq_len += 1

    return generated


@pytest.fixture(scope="module")
def qwen3_model():
    """Load Qwen3-0.6B once for all tests in this module."""
    model, tokenizer = mlx_lm_load(MODEL_NAME)
    return model, tokenizer


class TestMLXPagedVsStandard:
    @pytest.mark.slow
    def test_greedy_output_matches(self, qwen3_model):
        """MLX paged attention greedy decode must match standard path."""
        model, tokenizer = qwen3_model
        prompt = "The capital of France is"
        token_ids = tokenizer.encode(prompt)
        max_new = 20

        ref_tokens = _greedy_generate_standard(model, token_ids, max_new)
        mlx_tokens = _greedy_generate_mlx_paged(model, token_ids, max_new)

        assert ref_tokens == mlx_tokens, (
            f"Token mismatch!\n  Standard:  {ref_tokens}\n  MLX paged: {mlx_tokens}"
        )

    @pytest.mark.slow
    def test_batched_decode_matches(self, qwen3_model):
        """Batched MLX paged decode must match per-request sequential."""
        model, tokenizer = qwen3_model
        prompts = [
            "The capital of France is",
            "Machine learning is",
        ]
        max_new = 10

        # Reference: standard path per prompt
        ref_all = []
        for prompt in prompts:
            token_ids = tokenizer.encode(prompt)
            ref_all.append(_greedy_generate_standard(model, token_ids, max_new))

        # MLX paged path: prefill each, then batched decode
        args = model.args
        num_layers = args.num_hidden_layers
        num_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim

        total_max = (
            max(len(tokenizer.encode(p)) for p in prompts) + max_new + BLOCK_SIZE
        )
        num_blocks = ((total_max + BLOCK_SIZE - 1) // BLOCK_SIZE) * len(prompts) + 8

        mlx_cache = MLXPagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
        )
        patch_model_attention_mlx_paged(model, mlx_cache, BLOCK_SIZE)

        all_token_ids = []
        all_block_ids = []
        all_seq_lens = []
        all_generated: list[list[int]] = []

        for i, prompt in enumerate(prompts):
            tids = tokenizer.encode(prompt)
            all_token_ids.append(tids)
            needed = (len(tids) + max_new + BLOCK_SIZE - 1) // BLOCK_SIZE
            bids = mlx_cache.allocate_blocks(f"seq-{i}", needed)
            all_block_ids.append(bids)

            prepare_prefill(bids, len(tids), BLOCK_SIZE)
            offset_caches = [OffsetCache(0) for _ in range(num_layers)]
            input_ids = mx.array([tids], dtype=mx.int32)
            logits = model(input_ids, cache=offset_caches)
            next_tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            mx.eval(mx.array(next_tok))
            clear_context()

            all_generated.append([next_tok])
            all_seq_lens.append(len(tids))

        # Batched decode steps
        for _step in range(max_new - 1):
            requests_info = []
            for i in range(len(prompts)):
                requests_info.append((all_block_ids[i], all_seq_lens[i]))

            prepare_decode(requests_info, BLOCK_SIZE)
            max_offset = max(all_seq_lens)
            offset_caches = [OffsetCache(max_offset) for _ in range(num_layers)]

            last_tokens = [gen[-1] for gen in all_generated]
            batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]
            logits = model(batched_input, cache=offset_caches)
            next_toks = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_toks)
            clear_context()

            for i in range(len(prompts)):
                tok = int(next_toks[i].item())
                all_generated[i].append(tok)
                all_seq_lens[i] += 1

        for i, prompt in enumerate(prompts):
            assert all_generated[i] == ref_all[i], (
                f"Mismatch for prompt {i} ({prompt!r}):\n"
                f"  Standard:  {ref_all[i]}\n"
                f"  MLX paged: {all_generated[i]}"
            )


class TestMLXPagedPatchRouting:
    @pytest.mark.slow
    def test_patch_replaces_self_attn(self, qwen3_model):
        """After patching, each layer's self_attn should be a wrapper."""
        model, _ = qwen3_model
        args = model.args

        mlx_cache = MLXPagedKVCache(
            num_layers=args.num_hidden_layers,
            num_kv_heads=args.num_key_value_heads,
            head_dim=args.head_dim,
            num_blocks=32,
            block_size=BLOCK_SIZE,
        )
        patch_model_attention_mlx_paged(model, mlx_cache, BLOCK_SIZE)

        layers = model.model.layers
        for i, layer in enumerate(layers):
            assert isinstance(layer.self_attn, MLXPagedAttentionWrapper), (
                f"Layer {i} self_attn is {type(layer.self_attn).__name__}, "
                f"expected MLXPagedAttentionWrapper"
            )

    @pytest.mark.slow
    def test_fallback_when_no_context(self, qwen3_model):
        """Without paged context, calls must fall back to original attention."""
        model, _ = qwen3_model
        args = model.args

        mlx_cache = MLXPagedKVCache(
            num_layers=args.num_hidden_layers,
            num_kv_heads=args.num_key_value_heads,
            head_dim=args.head_dim,
            num_blocks=32,
            block_size=BLOCK_SIZE,
        )
        patch_model_attention_mlx_paged(model, mlx_cache, BLOCK_SIZE)

        cache = make_prompt_cache(model)
        input_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        logits = model(input_ids, cache=cache)
        mx.eval(logits)
