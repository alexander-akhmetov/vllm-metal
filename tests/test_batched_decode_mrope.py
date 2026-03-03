# SPDX-License-Identifier: Apache-2.0
"""Tests for mrope model detection and batched decode fallback.

Qwen3.5 and other models using multimodal rotary position embeddings (mrope)
store per-request offsets in BatchKVCache.offset as mx.array, which is
incompatible with mlx_vlm's attention code that uses cache.offset as a
Python int for mask slicing. This causes:

    ValueError: Slice indices must be integers or None.

These tests verify that:
1. mrope models are correctly detected
2. Batched decode falls back to sequential decode for mrope models
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import BatchKVCache, KVCache

import vllm_metal.v1.model_runner as mr


class TestBatchKVCacheOffsetType:
    """Verify the root cause: BatchKVCache.offset is mx.array, not int."""

    def test_kvcache_offset_is_int(self):
        cache = KVCache()
        assert isinstance(cache.offset, int)

    def test_batch_kvcache_offset_is_mx_array(self):
        """BatchKVCache.offset is mx.array — the root cause of the crash."""
        c1, c2 = KVCache(), KVCache()
        # Simulate cached tokens: request 1 has 100 tokens, request 2 has 200
        c1.keys = mx.zeros((1, 4, 100, 64))
        c1.values = mx.zeros((1, 4, 100, 64))
        c1.offset = 100
        c2.keys = mx.zeros((1, 4, 200, 64))
        c2.values = mx.zeros((1, 4, 200, 64))
        c2.offset = 200

        batch = BatchKVCache.merge([c1, c2])
        # This IS the problem: offset is mx.array, not int
        assert isinstance(batch.offset, mx.array), (
            "Expected mx.array offset (the root cause of the Qwen3.5 crash)"
        )
        # Slicing with mx.array raises ValueError in mlx_vlm attention code:
        #   mask[..., :kv_seq_len] where kv_seq_len = keys.shape[-2] + cache.offset + 1
        kv_seq_len = 1 + batch.offset + 1  # mx.array, not int
        assert isinstance(kv_seq_len, mx.array)


class TestMropeDetection:
    """Test that mrope models are correctly detected as incompatible with batched decode."""

    def test_detects_mrope_from_rope_parameters(self):
        model_args = {
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "type": "default",
            },
            "model_type": "qwen3_5_moe_text",
        }
        assert mr._has_mrope(model_args) is True

    def test_no_mrope_for_standard_models(self):
        model_args = {
            "rope_theta": 10000,
            "model_type": "llama",
        }
        assert mr._has_mrope(model_args) is False

    def test_no_mrope_for_empty_args(self):
        assert mr._has_mrope({}) is False

    def test_no_mrope_when_rope_parameters_lacks_mrope_section(self):
        model_args = {
            "rope_parameters": {
                "rope_theta": 10000,
                "type": "default",
            },
        }
        assert mr._has_mrope(model_args) is False


class TestBatchedDecodeDispatch:
    """Test that mrope models fall back to sequential decode."""

    def _make_runner(self, *, supports_batched: bool) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner._supports_batched_decode = supports_batched
        return runner

    def test_supports_batched_decode_true_for_standard_models(self):
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model_args = {"model_type": "llama"}
        runner._init_batched_decode_support()
        assert runner._supports_batched_decode is True

    def test_supports_batched_decode_false_for_mrope_models(self):
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model_args = {
            "rope_parameters": {
                "mrope_section": [11, 11, 10],
            },
        }
        runner._init_batched_decode_support()
        assert runner._supports_batched_decode is False
