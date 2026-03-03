# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

import vllm_metal.v1.model_runner as mr


class TestWarmUp:
    """Tests for MetalModelRunner.warm_up()."""

    def _make_runner(self, *, is_vlm: bool = False) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = MagicMock()
        runner._is_vlm = is_vlm
        # No paged KV cache by default
        runner._paged_kv_cache = None
        return runner

    def test_warm_up_runs_prefill_and_decode(self, monkeypatch) -> None:
        """warm_up() should run a multi-token prefill followed by a 1-token decode."""
        forward_calls: list[tuple] = []

        def fake_forward(tokens, cache=None):
            forward_calls.append((tokens.shape, cache is not None))
            return mx.zeros((1, tokens.shape[1], 10))

        fake_cache = [MagicMock()]
        monkeypatch.setattr(mr, "make_prompt_cache", lambda model: fake_cache)

        runner = self._make_runner()
        runner.model.side_effect = fake_forward

        def fake_extract_logits(output):
            return output

        monkeypatch.setattr(runner, "_extract_logits", fake_extract_logits)

        runner.warm_up()

        assert len(forward_calls) == 2, (
            f"Expected 2 forward calls, got {len(forward_calls)}"
        )
        # First call: prefill with ~128 tokens and cache
        prefill_shape, prefill_has_cache = forward_calls[0]
        assert prefill_shape[1] == 128, (
            f"Prefill should use 128 tokens, got {prefill_shape[1]}"
        )
        assert prefill_has_cache, "Prefill should use a cache"
        # Second call: decode with 1 token and same cache
        decode_shape, decode_has_cache = forward_calls[1]
        assert decode_shape[1] == 1, f"Decode should use 1 token, got {decode_shape[1]}"
        assert decode_has_cache, "Decode should use a cache"

    def test_warm_up_uses_vlm_language_model_for_cache(self, monkeypatch) -> None:
        """For VLM models, cache should be created from model.language_model."""
        cache_model_used = []

        def fake_make_prompt_cache(model):
            cache_model_used.append(model)
            return [MagicMock()]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner(is_vlm=True)
        language_model = MagicMock()
        runner.model.language_model = language_model
        runner.model.side_effect = lambda tokens, cache=None: mx.zeros(
            (1, tokens.shape[1], 10)
        )

        monkeypatch.setattr(runner, "_extract_logits", lambda output: output)

        runner.warm_up()

        assert len(cache_model_used) == 1
        assert cache_model_used[0] is language_model

    def test_warm_up_uses_model_directly_for_non_vlm(self, monkeypatch) -> None:
        """For non-VLM models, cache should be created from the model directly."""
        cache_model_used = []

        def fake_make_prompt_cache(model):
            cache_model_used.append(model)
            return [MagicMock()]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner(is_vlm=False)
        runner.model.side_effect = lambda tokens, cache=None: mx.zeros(
            (1, tokens.shape[1], 10)
        )

        monkeypatch.setattr(runner, "_extract_logits", lambda output: output)

        runner.warm_up()

        assert len(cache_model_used) == 1
        assert cache_model_used[0] is runner.model

    def test_warm_up_calls_clear_cache(self, monkeypatch) -> None:
        """warm_up() should call mx.clear_cache() after completing."""
        clear_cache_called = []

        monkeypatch.setattr(mr, "make_prompt_cache", lambda model: [MagicMock()])

        runner = self._make_runner()
        runner.model.side_effect = lambda tokens, cache=None: mx.zeros(
            (1, tokens.shape[1], 10)
        )
        monkeypatch.setattr(runner, "_extract_logits", lambda output: output)

        monkeypatch.setattr(mx, "clear_cache", lambda: clear_cache_called.append(True))

        runner.warm_up()

        assert len(clear_cache_called) >= 1, (
            "mx.clear_cache() should be called after warm-up"
        )

    def test_warm_up_skips_when_model_is_none(self) -> None:
        """warm_up() should skip gracefully when model is not loaded."""
        runner = self._make_runner()
        runner.model = None

        # Should not raise
        runner.warm_up()

    def test_warm_up_handles_forward_failure(self, monkeypatch) -> None:
        """warm_up() should catch and log exceptions from model forward pass."""
        monkeypatch.setattr(mr, "make_prompt_cache", lambda model: [MagicMock()])

        runner = self._make_runner()
        runner.model.side_effect = RuntimeError("shader compilation failed")
        monkeypatch.setattr(runner, "_extract_logits", lambda output: output)

        # Should not raise
        runner.warm_up()

    def test_warm_up_preserves_paged_attention_kernel_warmup(self, monkeypatch) -> None:
        """warm_up() should still call _warm_up_paged_attention_kernel when paged cache exists."""
        monkeypatch.setattr(mr, "make_prompt_cache", lambda model: [MagicMock()])

        runner = self._make_runner()
        runner.model.side_effect = lambda tokens, cache=None: mx.zeros(
            (1, tokens.shape[1], 10)
        )
        monkeypatch.setattr(runner, "_extract_logits", lambda output: output)

        paged_warmup_called = []
        runner._paged_kv_cache = MagicMock()
        monkeypatch.setattr(
            runner,
            "_warm_up_paged_attention_kernel",
            lambda: paged_warmup_called.append(True),
        )

        runner.warm_up()

        assert len(paged_warmup_called) == 1, (
            "Paged attention kernel warm-up should still be called"
        )
