# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import UserDict
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

import vllm_metal.v1.model_runner as mr


class TestPrefixCacheHybridGuard:
    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = MagicMock()
        runner._is_vlm = False
        runner._prefix_cache = mr.PrefixCacheManager(max_bytes=1024 * 1024)
        runner._gen_prompt_suffix = ()
        return runner

    def test_hybrid_model_uses_prefix_cache(self, monkeypatch) -> None:
        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.keys = mx.zeros((1, 8, 0, 64))
            kv.values = mx.zeros((1, 8, 0, 64))
            kv.offset = 0
            ac = mr.ArraysCache(2)
            return [kv, ac, kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner()
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        token_ids = [1, 2, 3, 4, 5]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        lookup_spy.assert_called_once_with([1, 2, 3, 4])
        insert_spy.assert_called_once()

    def test_pure_kvcache_uses_prefix_cache(self, monkeypatch) -> None:
        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.state = [mx.zeros((1, 4, 8, 64)), mx.zeros((1, 4, 8, 64))]
            return [kv, kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        runner = self._make_runner()
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 1, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        token_ids = [1, 2, 3, 4, 5]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        lookup_spy.assert_called_once_with([1, 2, 3, 4])
        insert_spy.assert_called_once()


class TestHybridPrefixCacheRoundtrip:
    """Verify insert/restore roundtrip for mixed KVCache+ArraysCache."""

    def _make_kv(self, seq_len: int = 2, value: float = 1.0) -> mr.KVCache:
        kv = mr.KVCache()
        kv.state = [
            mx.full((1, 1, seq_len, 8), value, dtype=mx.float32),
            mx.full((1, 1, seq_len, 8), value + 0.5, dtype=mx.float32),
        ]
        return kv

    def _make_arrays_cache(
        self, entries: int = 2, value: float = 1.0
    ) -> mr.ArraysCache:
        ac = mr.ArraysCache(entries)
        for i in range(entries):
            ac[i] = mx.full((1, 4), value + i, dtype=mx.float32)
        return ac

    def test_insert_restore_roundtrip(self) -> None:
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)

        kv = self._make_kv(seq_len=3, value=2.0)
        ac = self._make_arrays_cache(entries=2, value=5.0)
        cache = [kv, ac, self._make_kv(seq_len=3, value=3.0)]

        token_ids = [10, 20, 30]
        mgr.insert(token_ids, cache)

        assert len(mgr._entries) == 1

        result = mgr.lookup(token_ids)
        assert result is not None
        cached, match_len = result
        assert match_len == len(token_ids)

        # Verify KVCache entries
        assert isinstance(cached.cache_state[0], tuple)
        assert bool(mx.allclose(cached.cache_state[0][0], kv.state[0]))
        assert bool(mx.allclose(cached.cache_state[0][1], kv.state[1]))

        # Verify ArraysCache entry
        assert isinstance(cached.cache_state[1], list)
        assert len(cached.cache_state[1]) == 2
        assert bool(mx.allclose(cached.cache_state[1][0], ac.state[0]))
        assert bool(mx.allclose(cached.cache_state[1][1], ac.state[1]))

    def test_compute_entry_bytes_includes_arrays_cache(self) -> None:
        kv = self._make_kv(seq_len=2, value=1.0)
        ac = self._make_arrays_cache(entries=2, value=1.0)

        kv_state = (mx.array(kv.state[0]), mx.array(kv.state[1]))
        ac_state = [mx.array(s) for s in ac.state]
        cache_state = [kv_state, ac_state]

        total = mr._compute_entry_bytes(cache_state)
        expected = kv.state[0].nbytes + kv.state[1].nbytes
        expected += sum(s.nbytes for s in ac.state)
        assert total == expected

    def test_compute_entry_bytes_handles_none_arrays(self) -> None:
        ac_state = [mx.zeros((1, 4), dtype=mx.float32), None]
        cache_state = [ac_state]

        total = mr._compute_entry_bytes(cache_state)
        assert total == mx.zeros((1, 4), dtype=mx.float32).nbytes


class TestHybridCacheMergeExtract:
    """Regression tests for hybrid (KV + ArraysCache) batching.

    Background:
    - `mlx-lm==0.30.6` removed `MambaCache` and hybrid models now use `ArraysCache`.
    - Older mlx-lm versions don't provide `ArraysCache.merge()` / `extract()`.

    These tests validate that vllm-metal can merge per-request caches into a batched
    cache, run a batched forward pass, and then extract per-request caches back,
    without depending on `MambaCache` or new mlx-lm APIs.
    """

    _ARRAYS_CACHE_ENTRIES = 2
    _ARRAYS_CACHE_FEATURES = 4

    _KV_NUM_HEADS = 1
    _KV_HEAD_DIM = 2

    def _make_arrays_cache(self, v0: float | None, v1: float | None) -> mr.ArraysCache:
        cache = mr.ArraysCache(self._ARRAYS_CACHE_ENTRIES)
        if v0 is not None:
            cache[0] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v0, dtype=mx.float32)
        if v1 is not None:
            cache[1] = mx.full((1, self._ARRAYS_CACHE_FEATURES), v1, dtype=mx.float32)
        return cache

    def _make_kv_cache(self, seq_len: int, value: float) -> mr.KVCache:
        kv = mr.KVCache()
        kv.keys = mx.full(
            (1, self._KV_NUM_HEADS, seq_len, self._KV_HEAD_DIM),
            value,
            dtype=mx.float32,
        )
        kv.values = mx.full(
            (1, self._KV_NUM_HEADS, seq_len, self._KV_HEAD_DIM),
            value + 0.5,
            dtype=mx.float32,
        )
        kv.offset = seq_len
        return kv

    def _make_rotating_kv_cache(
        self, *, max_size: int, total_tokens: int, value: float
    ) -> mr.RotatingKVCache:
        cache = mr.RotatingKVCache(max_size=max_size)
        keys = mx.full(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM),
            value,
            dtype=mx.float32,
        )
        values = mx.full(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM),
            value + 0.5,
            dtype=mx.float32,
        )
        for _ in range(total_tokens):
            cache.update_and_fetch(keys, values)
        return cache

    def test_arrays_cache_merge_extract_roundtrip(self) -> None:
        """Merging then extracting ArraysCache round-trips per request."""
        arrays_cache_req0 = self._make_arrays_cache(1.0, 11.0)
        arrays_cache_req1 = self._make_arrays_cache(2.0, 22.0)

        merged = mr._merge_kv_caches([[arrays_cache_req0], [arrays_cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.ArraysCache)
        assert isinstance(extracted_req0, mr.ArraysCache)
        assert isinstance(extracted_req1, mr.ArraysCache)
        assert bool(mx.allclose(extracted_req0.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(extracted_req0.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(extracted_req1.state[0], arrays_cache_req1.state[0]))
        assert bool(mx.allclose(extracted_req1.state[1], arrays_cache_req1.state[1]))

    def test_arrays_cache_merge_extract_handles_missing_entries(self) -> None:
        """Missing per-request entries become zeros after merging.

        ArraysCache merging densifies per-entry state into a batch array when at
        least one request has that entry populated. Requests that had `None`
        for the entry are represented as zeros in the merged state.
        """
        arrays_cache_req0 = self._make_arrays_cache(1.0, 11.0)
        arrays_cache_req1 = self._make_arrays_cache(2.0, None)

        merged = mr._merge_kv_caches([[arrays_cache_req0], [arrays_cache_req1]])

        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(extracted_req0, mr.ArraysCache)
        assert isinstance(extracted_req1, mr.ArraysCache)

        assert bool(mx.allclose(extracted_req0.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(extracted_req0.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(extracted_req1.state[0], arrays_cache_req1.state[0]))

        missing = extracted_req1.state[1]
        assert missing is not None
        assert missing.shape == (1, self._ARRAYS_CACHE_FEATURES)
        assert bool(mx.allclose(missing, mx.zeros_like(missing)))

    def test_mixed_kv_and_arrays_cache_merge_extract_roundtrip(self) -> None:
        """Merging/extracting preserves both KVCache and ArraysCache layers."""
        kv_cache_req0 = self._make_kv_cache(seq_len=2, value=1.0)
        kv_cache_req1 = self._make_kv_cache(seq_len=4, value=2.0)
        arrays_cache_req0 = self._make_arrays_cache(3.0, 33.0)
        arrays_cache_req1 = self._make_arrays_cache(4.0, 44.0)

        merged = mr._merge_kv_caches(
            [[kv_cache_req0, arrays_cache_req0], [kv_cache_req1, arrays_cache_req1]]
        )
        extracted_req0 = mr._extract_kv_cache(merged, 0)
        extracted_req1 = mr._extract_kv_cache(merged, 1)

        assert isinstance(merged[0], mr.BatchKVCache)
        assert isinstance(merged[1], mr.ArraysCache)

        kv_req0_out, arrays_req0_out = extracted_req0
        kv_req1_out, arrays_req1_out = extracted_req1

        assert isinstance(kv_req0_out, mr.KVCache)
        assert isinstance(kv_req1_out, mr.KVCache)
        assert isinstance(arrays_req0_out, mr.ArraysCache)
        assert isinstance(arrays_req1_out, mr.ArraysCache)

        assert kv_req0_out.offset == kv_cache_req0.offset
        assert kv_req1_out.offset == kv_cache_req1.offset
        assert bool(mx.allclose(kv_req0_out.keys, kv_cache_req0.keys))
        assert bool(mx.allclose(kv_req0_out.values, kv_cache_req0.values))
        assert bool(mx.allclose(kv_req1_out.keys, kv_cache_req1.keys))
        assert bool(mx.allclose(kv_req1_out.values, kv_cache_req1.values))

        assert bool(mx.allclose(arrays_req0_out.state[0], arrays_cache_req0.state[0]))
        assert bool(mx.allclose(arrays_req0_out.state[1], arrays_cache_req0.state[1]))
        assert bool(mx.allclose(arrays_req1_out.state[0], arrays_cache_req1.state[0]))
        assert bool(mx.allclose(arrays_req1_out.state[1], arrays_cache_req1.state[1]))

    def test_rotating_kvcache_merge_extract_preserves_offsets(self) -> None:
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=20, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(max_size=8, total_tokens=5, value=2.0)

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_handles_offset_exceeding_max_size(self) -> None:
        """Merging works when offset > max_size (cache has rotated).

        This is a regression test for a bug in ``BatchRotatingKVCache.merge``
        (mlx-lm <= 0.29.1) where using ``c.offset`` instead of ``len(c)`` caused
        a broadcast shape error after the cache rotated past its maximum size.
        """
        # offset=300 >> max_size=8 — the cache has rotated many times
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=300, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=150, value=2.0
        )

        assert cache_req0.offset > cache_req0.max_size
        assert cache_req1.offset > cache_req1.max_size

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_handles_prefill_exceeding_max_size(self) -> None:
        """Merging works when prefill length exceeds max_size.

        After a large prefill the internal buffer may temporarily be larger than
        ``max_size``.  The merge must trim to the effective sliding-window length.
        """
        # Prefill 128 tokens into a cache with max_size=70
        cache_req0 = mr.RotatingKVCache(max_size=70)
        big_k = mx.full(
            (1, self._KV_NUM_HEADS, 128, self._KV_HEAD_DIM), 1.0, dtype=mx.float32
        )
        big_v = mx.full(
            (1, self._KV_NUM_HEADS, 128, self._KV_HEAD_DIM), 1.5, dtype=mx.float32
        )
        cache_req0.update_and_fetch(big_k, big_v)

        cache_req1 = self._make_rotating_kv_cache(
            max_size=70, total_tokens=30, value=2.0
        )

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]
        extracted_req1 = mr._extract_kv_cache(merged, 1)[0]

        assert isinstance(merged[0], mr.BatchRotatingKVCache)
        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset
        assert extracted_req1.offset == cache_req1.offset

    def test_rotating_kvcache_merge_decode_extract_roundtrip(self) -> None:
        """Merged cache can be used for a batched decode step and extracted back.

        This verifies that the internal state (_idx, _offset) set by
        ``_merge_rotating_kv_caches`` is compatible with
        ``BatchRotatingKVCache.update_and_fetch`` and ``extract``.
        """
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=20, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(max_size=8, total_tokens=5, value=2.0)

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        batch_cache = merged[0]
        assert isinstance(batch_cache, mr.BatchRotatingKVCache)

        # Simulate one batched decode step (S=1)
        decode_k = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        decode_v = mx.ones((2, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM))
        batch_cache.update_and_fetch(decode_k, decode_v)

        # Extract back to per-request caches
        extracted_req0 = batch_cache.extract(0)
        extracted_req1 = batch_cache.extract(1)

        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert isinstance(extracted_req1, mr.RotatingKVCache)
        assert extracted_req0.offset == cache_req0.offset + 1
        assert extracted_req1.offset == cache_req1.offset + 1

    def test_extracted_rotating_cache_can_decode_after_rotation(self) -> None:
        """Extracted RotatingKVCache can continue decoding after offset > max_size.

        After merge -> extract, the extracted cache may have offset > max_size
        but keys.shape[2] < max_size (buffer sliced by extract).  Without the
        buffer padding fix in ``_extract_kv_cache``, the next
        ``_update_in_place`` call would compute a negative ``new_size`` and
        crash with ``ValueError: [full] Negative dimensions not allowed``.
        """
        cache_req0 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=300, value=1.0
        )
        cache_req1 = self._make_rotating_kv_cache(
            max_size=8, total_tokens=150, value=2.0
        )

        merged = mr._merge_kv_caches([[cache_req0], [cache_req1]])
        extracted_req0 = mr._extract_kv_cache(merged, 0)[0]

        assert isinstance(extracted_req0, mr.RotatingKVCache)
        assert extracted_req0.offset > extracted_req0.max_size
        assert extracted_req0.keys.shape[2] == extracted_req0.max_size

        # This would crash without the buffer padding fix
        decode_k = mx.ones(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM), dtype=mx.float32
        )
        decode_v = mx.ones(
            (1, self._KV_NUM_HEADS, 1, self._KV_HEAD_DIM), dtype=mx.float32
        )
        extracted_req0.update_and_fetch(decode_k, decode_v)

        assert extracted_req0.offset == cache_req0.offset + 1

    def test_merge_kv_caches_rejects_mixed_cache_types_within_layer(self) -> None:
        arrays_cache = self._make_arrays_cache(1.0, 2.0)
        kv_cache = mr.KVCache()
        with pytest.raises(TypeError, match="Mixed cache types in a single layer"):
            mr._merge_kv_caches([[arrays_cache], [kv_cache]])


class TestPrefixCacheEviction:
    def test_eviction_under_max_bytes(self) -> None:
        # 1KB limit
        mgr = mr.PrefixCacheManager(max_bytes=1024)

        # Create fake KVCache with known size
        kv = mr.KVCache()
        k = mx.zeros((1, 4, 8, 64))  # 8192 bytes (float32)
        v = mx.zeros((1, 4, 8, 64))
        kv.state = [k, v]

        # Insert should be skipped (entry > max_bytes)
        mgr.insert([1, 2, 3], [kv])
        assert len(mgr._entries) == 0
        assert mgr._current_bytes == 0

    def _make_kv(self, seq_len: int = 1) -> mr.KVCache:
        kv = mr.KVCache()
        kv.state = [
            mx.zeros((1, 1, seq_len, 8)),
            mx.zeros((1, 1, seq_len, 8)),
        ]
        return kv

    def test_eviction_triggers_on_full(self) -> None:
        kv = self._make_kv()
        entry_bytes = kv.state[0].nbytes + kv.state[1].nbytes
        mgr = mr.PrefixCacheManager(max_bytes=entry_bytes * 2 + 1)

        mgr.insert([1], [self._make_kv()])
        mgr.insert([2], [self._make_kv()])
        assert len(mgr._entries) == 2

        # Third insert should evict one entry
        mgr.insert([3], [self._make_kv()])
        assert len(mgr._entries) == 2

    def test_duplicate_insert_skipped(self) -> None:
        mgr = mr.PrefixCacheManager(max_bytes=1024 * 1024)

        mgr.insert([1, 2], [self._make_kv()])
        bytes_after_first = mgr._current_bytes

        mgr.insert([1, 2], [self._make_kv()])
        assert mgr._current_bytes == bytes_after_first

    def test_eviction_prefers_least_recently_used(self) -> None:
        """When ref_count is tied, evict the entry accessed least recently."""
        kv = self._make_kv()
        entry_bytes = kv.state[0].nbytes + kv.state[1].nbytes
        # Room for exactly 2 entries
        mgr = mr.PrefixCacheManager(max_bytes=entry_bytes * 2 + 1)

        mgr.insert([1], [self._make_kv()])
        mgr.insert([2], [self._make_kv()])

        # Both start with ref_count=1. Lookup both to bring to ref_count=2,
        # but [2] is looked up second so it's more recent.
        mgr.lookup([1])
        mgr.lookup([2])

        # Insert third entry — should evict [1] (older last_used, same ref_count)
        mgr.insert([3], [self._make_kv()])
        assert (1,) not in mgr._entries
        assert (2,) in mgr._entries
        assert (3,) in mgr._entries


class TestLongestPrefixMatch:
    """Tests for longest-prefix matching in PrefixCacheManager."""

    def _make_kv(self, seq_len: int = 1) -> mr.KVCache:
        kv = mr.KVCache()
        kv.state = [
            mx.zeros((1, 1, seq_len, 8)),
            mx.zeros((1, 1, seq_len, 8)),
        ]
        return kv

    def test_partial_prefix_hit(self) -> None:
        """Insert 100-token prefix, lookup 150-token query → full entry match."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        short_prefix = list(range(100))
        mgr.insert(short_prefix, [self._make_kv()])

        long_query = list(range(150))
        result = mgr.lookup(long_query)
        assert result is not None
        cached, match_len = result
        assert list(cached.token_ids) == short_prefix
        assert match_len == 100

    def test_longest_match(self) -> None:
        """Insert 100 and 150 tokens, lookup 170 → returns 150-token entry."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        prefix_100 = list(range(100))
        prefix_150 = list(range(150))
        mgr.insert(prefix_100, [self._make_kv()])
        mgr.insert(prefix_150, [self._make_kv()])

        query_170 = list(range(170))
        result = mgr.lookup(query_170)
        assert result is not None
        cached, match_len = result
        assert list(cached.token_ids) == prefix_150
        assert match_len == 150

    def test_no_match(self) -> None:
        """Completely different prefix returns miss."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        mgr.insert([1, 2, 3, 4, 5], [self._make_kv()])

        assert mgr.lookup([10, 20, 30]) is None

    def test_exact_match_still_works(self) -> None:
        """Exact match (fast path) still returns the entry."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        prefix = [1, 2, 3, 4, 5]
        mgr.insert(prefix, [self._make_kv()])

        result = mgr.lookup(prefix)
        assert result is not None
        cached, match_len = result
        assert list(cached.token_ids) == prefix
        assert match_len == 5

    def test_branch_divergence(self) -> None:
        """Cached [0..99] and [0..49], lookup [0..49]+[999..] → LCP is 50 tokens."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        mgr.insert(list(range(100)), [self._make_kv()])
        mgr.insert(list(range(50)), [self._make_kv()])

        # Query shares first 50 tokens then diverges
        query = list(range(50)) + list(range(999, 1050))
        result = mgr.lookup(query)
        assert result is not None
        _, match_len = result
        assert match_len == 50

    def test_insert_does_not_subsume_shorter_prefixes(self) -> None:
        """Inserting a longer prefix does not remove a shorter one."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        short = list(range(50))
        long = list(range(100))
        mgr.insert(short, [self._make_kv()])
        mgr.insert(long, [self._make_kv()])

        # Both entries should exist
        result_short = mgr.lookup(short)
        result_long = mgr.lookup(long)
        assert result_short is not None
        assert result_long is not None
        assert list(result_short[0].token_ids) == short
        assert list(result_long[0].token_ids) == long

    def test_eviction_removes_from_both_structures(self) -> None:
        """After eviction, entry is gone from both dict and sorted list."""
        kv = self._make_kv()
        entry_bytes = kv.state[0].nbytes + kv.state[1].nbytes
        # Room for 2 entries only
        mgr = mr.PrefixCacheManager(max_bytes=entry_bytes * 2 + 1)

        mgr.insert([1, 2], [self._make_kv()])
        mgr.insert([3, 4], [self._make_kv()])

        # Third insert evicts one
        mgr.insert([5, 6, 7], [self._make_kv()])
        assert len(mgr._entries) == 2
        assert len(mgr._sorted) == 2

        # Evicted entry should not be found via prefix match either
        all_token_ids = {tuple(e.token_ids) for e in mgr._sorted}
        for entry_key in mgr._entries:
            assert entry_key in all_token_ids

    def test_insert_after_partial_hit(self) -> None:
        """After a partial hit + forward, the full prefix should be cacheable."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        short_prefix = list(range(50))
        mgr.insert(short_prefix, [self._make_kv()])

        # Simulate: lookup 100 tokens → partial hit for first 50
        result = mgr.lookup(list(range(100)))
        assert result is not None
        cached, match_len = result
        assert list(cached.token_ids) == short_prefix
        assert match_len == 50

        # After processing remaining tokens, insert the full prefix
        full_prefix = list(range(100))
        mgr.insert(full_prefix, [self._make_kv()])

        # Now lookup should return the longer match
        result2 = mgr.lookup(list(range(120)))
        assert result2 is not None
        cached2, match_len2 = result2
        assert list(cached2.token_ids) == full_prefix
        assert match_len2 == 100

    def test_lookup_query_shorter_than_all_entries(self) -> None:
        """Query shorter than any cached entry but sharing prefix → miss (below threshold)."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        mgr.insert([1, 2, 3, 4, 5], [self._make_kv()])

        # Only 2 tokens match, below _MIN_PREFIX_MATCH threshold
        assert mgr.lookup([1, 2]) is None

    def test_hit_rate_counts_partial_hits(self) -> None:
        """Partial prefix hits are counted as hits."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        prefix = list(range(100))
        mgr.insert(prefix, [self._make_kv()])

        result = mgr.lookup(list(range(150)))  # partial hit
        assert result is not None
        assert mgr._hits == 1
        assert mgr._misses == 0

    def test_sorted_order_maintained(self) -> None:
        """Entries in _sorted are in descending length order after inserts."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        mgr.insert(list(range(40)), [self._make_kv()])
        mgr.insert(list(range(100)), [self._make_kv()])
        mgr.insert(list(range(70)), [self._make_kv()])

        lengths = [len(e.token_ids) for e in mgr._sorted]
        assert lengths == sorted(lengths, reverse=True)

    def test_empty_cache_lookup(self) -> None:
        """Lookup on empty cache returns None."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        assert mgr.lookup(list(range(50))) is None
        assert mgr._misses == 1

    def test_ref_count_incremented_on_hit(self) -> None:
        """ref_count bumps on both exact and partial hits."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        prefix = list(range(100))
        mgr.insert(prefix, [self._make_kv()])

        result = mgr.lookup(prefix)  # exact hit
        assert result is not None
        cached, _ = result
        assert cached.ref_count == 2  # 1 from insert + 1 from lookup

        result2 = mgr.lookup(list(range(150)))  # partial hit (same entry)
        assert result2 is not None
        cached2, match_len2 = result2
        assert cached2 is cached
        assert match_len2 == 100
        assert cached.ref_count == 3

    def test_equal_length_entries(self) -> None:
        """Two entries with same length but different tokens are both reachable."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        mgr.insert(list(range(100)), [self._make_kv()])
        mgr.insert(list(range(100, 200)), [self._make_kv()])

        result_a = mgr.lookup(list(range(100)))
        result_b = mgr.lookup(list(range(100, 200)))
        assert result_a is not None
        assert result_b is not None
        assert list(result_a[0].token_ids) == list(range(100))
        assert list(result_b[0].token_ids) == list(range(100, 200))

    def test_lcp_trailing_mismatch(self) -> None:
        """Entry of 100 tokens where last 2 diverge → match_len=98."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        # Cached entry: [0..99]
        entry_tokens = list(range(100))
        mgr.insert(entry_tokens, [self._make_kv()])

        # Query: first 98 tokens match, then diverges
        query = list(range(98)) + [999, 998] + list(range(200, 250))
        result = mgr.lookup(query)
        assert result is not None
        cached, match_len = result
        assert match_len == 98
        assert list(cached.token_ids) == entry_tokens

    def test_lcp_generation_prompt_scenario(self) -> None:
        """Simulates the generation prompt contamination bug.

        Turn 1 caches [shared..., <think>] (100 tokens).
        Turn 2 queries [shared..., actual_content...] (150 tokens).
        LCP finds 99 matching tokens (all except the trailing <think>).
        """
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        think_token = 248068

        # Turn 1: shared prefix + generation prompt token
        shared = list(range(99))
        cached_prefix = shared + [think_token]
        mgr.insert(cached_prefix, [self._make_kv()])

        # Turn 2: shared prefix + actual content
        query = shared + [8915, 678, 1234] + list(range(500, 550))
        result = mgr.lookup(query)
        assert result is not None
        cached, match_len = result
        assert match_len == 99  # all but the <think> token
        assert list(cached.token_ids) == cached_prefix

    def test_lcp_below_min_threshold_is_miss(self) -> None:
        """LCP shorter than _MIN_PREFIX_MATCH → miss."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        # 20-token common prefix, below the 32-token threshold
        entry = list(range(50))
        mgr.insert(entry, [self._make_kv()])

        query = list(range(20)) + list(range(900, 950))
        assert mgr.lookup(query) is None
        assert mgr._misses == 1

    def test_restore_cache_truncates_kv(self) -> None:
        """restore_cache with match_len < entry length truncates KV state."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)

        kv = mr.KVCache()
        seq_len = 100
        kv.state = [
            mx.ones((1, 4, seq_len, 64)),
            mx.ones((1, 4, seq_len, 64)),
        ]
        mgr.insert(list(range(seq_len)), [kv])

        result = mgr.lookup(list(range(seq_len)))
        assert result is not None
        cached, _ = result

        # Create a mock model that returns a fresh KVCache
        from unittest.mock import MagicMock

        mock_model = MagicMock()

        def fake_make_prompt_cache(model):
            fresh_kv = mr.KVCache()
            fresh_kv.state = [mx.zeros((1, 4, 0, 64)), mx.zeros((1, 4, 0, 64))]
            return [fresh_kv]

        import vllm_metal.v1.model_runner as mr_mod

        original_make = mr_mod.make_prompt_cache
        mr_mod.make_prompt_cache = fake_make_prompt_cache
        try:
            restored = mgr.restore_cache(cached, mock_model, False, match_len=80)
            assert restored[0].state[0].shape[2] == 80
            assert restored[0].state[1].shape[2] == 80

            # Full restore should keep all tokens
            restored_full = mgr.restore_cache(cached, mock_model, False)
            assert restored_full[0].state[0].shape[2] == 100
        finally:
            mr_mod.make_prompt_cache = original_make

    def test_partial_match_rejected_for_arrays_cache_entry(self) -> None:
        """Partial LCP match is rejected when entry contains ArraysCache state."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)

        # Simulate a hybrid entry: KVCache tuple + ArraysCache list
        kv_state = (mx.zeros((1, 1, 100, 8)), mx.zeros((1, 1, 100, 8)))
        arrays_state = [mx.zeros((1, 4)), mx.zeros((1, 4))]
        entry = mr.CachedPrefix(
            token_ids=tuple(range(100)),
            cache_state=[kv_state, arrays_state],
            size_bytes=1024,
            ref_count=1,
        )
        mgr._entries[entry.token_ids] = entry
        mgr._sorted.append(entry)

        # Query shares 98 tokens (partial match) — should be rejected
        query = list(range(98)) + [999, 998] + list(range(200, 250))
        assert mgr.lookup(query) is None
        assert mgr._misses == 1

    def test_full_match_allowed_for_arrays_cache_entry(self) -> None:
        """Full entry match is allowed even when entry contains ArraysCache state."""
        mgr = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)

        kv_state = (mx.zeros((1, 1, 100, 8)), mx.zeros((1, 1, 100, 8)))
        arrays_state = [mx.zeros((1, 4)), mx.zeros((1, 4))]
        entry = mr.CachedPrefix(
            token_ids=tuple(range(100)),
            cache_state=[kv_state, arrays_state],
            size_bytes=1024,
            ref_count=1,
        )
        mgr._entries[entry.token_ids] = entry
        mgr._sorted.append(entry)

        # Query extends the full entry — full match, should succeed
        query = list(range(150))
        result = mgr.lookup(query)
        assert result is not None
        cached, match_len = result
        assert match_len == 100


class TestPrefixCacheEnableFlag:
    """Verify VLLM_METAL_PREFIX_CACHE presence-based enabling."""

    def test_enabled_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE", "1")
        assert mr._prefix_cache_enabled() is True

    def test_disabled_when_env_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE", raising=False)
        assert mr._prefix_cache_enabled() is False


_TEN_GB = 10 * 1024**3


def _mock_device_info():
    return {"max_recommended_working_set_size": _TEN_GB}


class TestPrefixCacheFractionParsing:
    def test_valid_fraction(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "0.1")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * 0.1)

    def test_default_fraction(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE_FRACTION", raising=False)
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_invalid_string_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "abc")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_out_of_range_zero_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "0")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_out_of_range_above_one_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "2")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_nan_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "nan")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_inf_uses_default(self, monkeypatch) -> None:
        monkeypatch.setenv("VLLM_METAL_PREFIX_CACHE_FRACTION", "inf")
        monkeypatch.setattr(mr.mx.metal, "device_info", _mock_device_info)
        result = mr._get_prefix_cache_max_bytes()
        assert result == int(_TEN_GB * mr._PREFIX_CACHE_DEFAULT_FRACTION)

    def test_device_info_fallback(self, monkeypatch) -> None:
        monkeypatch.delenv("VLLM_METAL_PREFIX_CACHE_FRACTION", raising=False)
        monkeypatch.setattr(
            mr.mx.metal,
            "device_info",
            lambda: {},
        )
        result = mr._get_prefix_cache_max_bytes()
        fallback = 8 * 1024**3
        assert result == int(fallback * mr._PREFIX_CACHE_DEFAULT_FRACTION)


class TestDetectGenPromptSuffix:
    """Tests for _detect_gen_prompt_suffix() suffix detection."""

    def _make_runner(self) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.tokenizer = None
        runner._gen_prompt_suffix = ()
        return runner

    def _make_tokenizer(self, side_effect):
        """Create a mock tokenizer that won't be unwrapped as a VLM processor."""
        tokenizer = MagicMock(spec=["apply_chat_template"])
        tokenizer.apply_chat_template = MagicMock(side_effect=side_effect)
        return tokenizer

    def test_working_chat_template(self) -> None:
        """Detects suffix when template adds generation prompt tokens."""
        runner = self._make_runner()
        runner.tokenizer = self._make_tokenizer(
            lambda msgs, add_generation_prompt, tokenize: (
                [1, 2, 3, 50, 51, 52] if add_generation_prompt else [1, 2, 3]
            )
        )
        result = runner._detect_gen_prompt_suffix()
        assert result == (50, 51, 52)

    def test_working_chat_template_dict_return(self) -> None:
        """Detects suffix when apply_chat_template returns a dict (transformers >=5.x)."""
        runner = self._make_runner()
        runner.tokenizer = self._make_tokenizer(
            lambda msgs, add_generation_prompt, tokenize: (
                {"input_ids": [1, 2, 3, 50, 51, 52], "attention_mask": [1] * 6}
                if add_generation_prompt
                else {"input_ids": [1, 2, 3], "attention_mask": [1] * 3}
            )
        )
        result = runner._detect_gen_prompt_suffix()
        assert result == (50, 51, 52)

    def test_working_chat_template_userdict_return(self) -> None:
        """Detects suffix when apply_chat_template returns a UserDict (BatchEncoding)."""
        runner = self._make_runner()

        def side_effect(msgs, add_generation_prompt, tokenize):
            d = UserDict()
            if add_generation_prompt:
                d["input_ids"] = [1, 2, 3, 50, 51, 52]
                d["attention_mask"] = [1] * 6
            else:
                d["input_ids"] = [1, 2, 3]
                d["attention_mask"] = [1] * 3
            return d

        runner.tokenizer = self._make_tokenizer(side_effect)
        result = runner._detect_gen_prompt_suffix()
        assert result == (50, 51, 52)

    def test_vlm_processor_unwrap(self) -> None:
        """Unwraps VLM processor to inner tokenizer for suffix detection."""
        runner = self._make_runner()
        inner_tok = self._make_tokenizer(
            lambda msgs, add_generation_prompt, tokenize: (
                [1, 2, 3, 50, 51] if add_generation_prompt else [1, 2, 3]
            )
        )
        processor = MagicMock(spec=["tokenizer"])
        processor.tokenizer = inner_tok
        runner.tokenizer = processor
        result = runner._detect_gen_prompt_suffix()
        assert result == (50, 51)

    def test_no_tokenizer(self) -> None:
        """Returns empty tuple when tokenizer is None."""
        runner = self._make_runner()
        runner.tokenizer = None
        result = runner._detect_gen_prompt_suffix()
        assert result == ()

    def test_no_chat_template(self) -> None:
        """Returns empty tuple when apply_chat_template raises."""
        runner = self._make_runner()
        runner.tokenizer = self._make_tokenizer(
            MagicMock(side_effect=Exception("no template"))
        )
        result = runner._detect_gen_prompt_suffix()
        assert result == ()

    def test_incompatible_templates(self) -> None:
        """Returns empty tuple when with_gen is not a prefix of without_gen."""
        runner = self._make_runner()
        runner.tokenizer = self._make_tokenizer(
            lambda msgs, add_generation_prompt, tokenize: (
                [10, 20, 30, 40] if add_generation_prompt else [1, 2, 3]
            )
        )
        result = runner._detect_gen_prompt_suffix()
        assert result == ()

    def test_no_suffix_added(self) -> None:
        """Returns empty tuple when gen prompt adds no tokens."""
        runner = self._make_runner()
        runner.tokenizer = self._make_tokenizer(
            lambda msgs, add_generation_prompt, tokenize: [1, 2, 3]
        )
        result = runner._detect_gen_prompt_suffix()
        assert result == ()


class TestPrefillStripGenPrompt:
    """Tests for gen prompt stripping in _prefill_single() cache boundary."""

    def _make_runner(self, gen_suffix: tuple[int, ...] = ()) -> mr.MetalModelRunner:
        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = MagicMock()
        runner._is_vlm = False
        runner._prefix_cache = mr.PrefixCacheManager(max_bytes=1024 * 1024)
        runner._gen_prompt_suffix = gen_suffix
        return runner

    def test_strips_gen_prompt_from_cache_key(self, monkeypatch) -> None:
        """Cache key excludes gen prompt suffix when detected."""
        gen_suffix = (50, 51, 52)
        runner = self._make_runner(gen_suffix=gen_suffix)

        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.keys = mx.zeros((1, 8, 0, 64))
            kv.values = mx.zeros((1, 8, 0, 64))
            kv.offset = 0
            return [kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        # token_ids ends with gen prompt suffix
        token_ids = [1, 2, 3, 4, 5] + list(gen_suffix)
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        # Should strip suffix: lookup key is [1,2,3,4,5] not [1,2,3,4,5,50,51]
        lookup_spy.assert_called_once_with([1, 2, 3, 4, 5])

    def test_no_strip_when_suffix_not_at_end(self, monkeypatch) -> None:
        """Falls back to [:-1] when suffix is not at end of token_ids."""
        gen_suffix = (50, 51, 52)
        runner = self._make_runner(gen_suffix=gen_suffix)

        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.keys = mx.zeros((1, 8, 0, 64))
            kv.values = mx.zeros((1, 8, 0, 64))
            kv.offset = 0
            return [kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        # token_ids does NOT end with gen prompt suffix
        token_ids = [1, 2, 3, 4, 5, 6, 7]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        # Fallback: [:-1] → [1,2,3,4,5,6]
        lookup_spy.assert_called_once_with([1, 2, 3, 4, 5, 6])

    def test_no_strip_when_no_suffix(self, monkeypatch) -> None:
        """Behaves as before when _gen_prompt_suffix is empty."""
        runner = self._make_runner(gen_suffix=())

        lookup_spy = MagicMock(return_value=None)
        insert_spy = MagicMock()

        def fake_make_prompt_cache(model):
            kv = mr.KVCache()
            kv.keys = mx.zeros((1, 8, 0, 64))
            kv.values = mx.zeros((1, 8, 0, 64))
            kv.offset = 0
            return [kv]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)
        monkeypatch.setattr(runner._prefix_cache, "lookup", lookup_spy)
        monkeypatch.setattr(runner._prefix_cache, "insert", insert_spy)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        token_ids = [1, 2, 3, 4, 5]
        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        runner._prefill_single("req-1", token_ids, sampling_params)

        # Original behavior: [:-1]
        lookup_spy.assert_called_once_with([1, 2, 3, 4])


class TestHybridMultiTurnFullMatch:
    """End-to-end: hybrid entry gets full match when gen prompt is stripped."""

    def test_multi_turn_full_match(self, monkeypatch) -> None:
        """Turn 2 gets exact HIT when gen prompt is stripped from both turns."""
        gen_suffix = (50, 51, 52)

        runner = mr.MetalModelRunner.__new__(mr.MetalModelRunner)
        runner.model = MagicMock()
        runner._is_vlm = False
        runner._prefix_cache = mr.PrefixCacheManager(max_bytes=10 * 1024 * 1024)
        runner._gen_prompt_suffix = gen_suffix

        def fake_make_prompt_cache(model):
            kv1 = mr.KVCache()
            kv1.keys = mx.zeros((1, 8, 0, 64))
            kv1.values = mx.zeros((1, 8, 0, 64))
            kv1.offset = 0
            ac = mr.ArraysCache(2)
            kv2 = mr.KVCache()
            kv2.keys = mx.zeros((1, 8, 0, 64))
            kv2.values = mx.zeros((1, 8, 0, 64))
            kv2.offset = 0
            return [kv1, ac, kv2]

        monkeypatch.setattr(mr, "make_prompt_cache", fake_make_prompt_cache)

        fake_logits = mx.zeros((1, 5, 100))
        runner.model.return_value = MagicMock(logits=fake_logits)

        sampling_params = MagicMock(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1.0,
        )

        # Turn 1: [system, user1, gen_prompt_suffix]
        shared = list(range(100))
        turn1_tokens = shared + list(gen_suffix)
        runner._prefill_single("turn1", turn1_tokens, sampling_params)

        # Verify cache key is `shared` (stripped suffix)
        assert tuple(shared) in runner._prefix_cache._entries

        # Turn 2: [system, user1, assistant1, user2, gen_prompt_suffix]
        turn2_tokens = shared + [200, 201, 202, 203] + list(gen_suffix)
        runner._prefill_single("turn2", turn2_tokens, sampling_params)

        # Turn 1's entry should have been an exact HIT (full match)
        stats = runner._prefix_cache.get_stats()
        assert stats["hits"] >= 1
