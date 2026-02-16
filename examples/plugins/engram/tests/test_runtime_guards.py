# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Runtime guard and async cleanup tests for Engram runtime constraints.

from types import SimpleNamespace

import pytest
import torch
from concurrent.futures import Future

from vllm.model_executor.layers.engram import (
    EngramRuntimeState,
    validate_engram_runtime_support,
)


def _make_vllm_config(**kwargs):
    """@brief Implement  make vllm config.

    Args:
        **kwargs (Any): Kwargs input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    data = {
        "model_config": SimpleNamespace(dtype="bfloat16"),
        "parallel_config": SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
        "speculative_config": None,
        "scheduler_config": SimpleNamespace(),
        "compilation_config": SimpleNamespace(cudagraph_mode="NONE", level=0),
        "enforce_eager": True,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


def test_runtime_guard_requires_bf16():
    """@brief Validate that runtime guard requires bf16.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(model_config=SimpleNamespace(dtype="float16"))

    with pytest.raises(ValueError, match="BF16"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_accepts_torch_bfloat16_dtype():
    """@brief Validate that runtime guard accepts torch bfloat16 dtype.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(model_config=SimpleNamespace(dtype=torch.bfloat16))
    validate_engram_runtime_support(cfg)


def test_runtime_guard_rejects_pipeline_parallelism():
    """@brief Validate that runtime guard rejects pipeline parallelism.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            enable_expert_parallel=False,
        )
    )

    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_rejects_expert_parallelism():
    """@brief Validate that runtime guard rejects expert parallelism.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            enable_expert_parallel=True,
        )
    )

    with pytest.raises(ValueError, match="expert parallelism"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_rejects_speculative_decoding():
    """@brief Validate that runtime guard rejects speculative decoding.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(speculative_config=SimpleNamespace(enabled=True))

    with pytest.raises(ValueError, match="speculative decoding"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_requires_eager_mode():
    """@brief Validate that runtime guard requires eager mode.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(enforce_eager=False)

    with pytest.raises(ValueError, match="eager execution"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_rejects_cudagraph_mode():
    """@brief Validate that runtime guard rejects cudagraph mode.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(
        compilation_config=SimpleNamespace(cudagraph_mode="FULL", level=0)
    )

    with pytest.raises(ValueError, match="CUDA graph mode"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_rejects_torch_compile_mode():
    """@brief Validate that runtime guard rejects torch compile mode.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE", level=1)
    )

    with pytest.raises(ValueError, match="torch.compile mode"):
        validate_engram_runtime_support(cfg)


def test_runtime_guard_allows_none_compile_level():
    """@brief Validate that runtime guard allows none compile level.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_vllm_config(
        compilation_config=SimpleNamespace(cudagraph_mode="NONE", level=None)
    )
    validate_engram_runtime_support(cfg)


def test_runtime_state_requires_memory_size_at_least_max_ngram_order():
    """@brief Validate runtime state requires memory size >= max ngram order.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(
        ValueError, match="engram_memory_size.*engram_max_ngram_order"
    ):
        EngramRuntimeState(
            memory_size=2,
            max_ngram_order=3,
            heads=8,
            mem_dim=32,
            vocab_size=1024,
            compression_ratio=0.8,
            conv_kernel=4,
            conv_dilation=3,
        )


def test_cleanup_requests_removes_async_lookup_futures():
    """@brief Validate that cleanup requests removes async lookup futures.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=4,
        heads=8,
        mem_dim=1008,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=4,
    )
    f0: Future[dict[int, dict[str, list[int]]]] = Future()
    f0.set_result({1: {"rows": [1]}})
    f1: Future[dict[int, dict[str, list[int]]]] = Future()
    f1.set_result({1: {"rows": [2]}})
    state.lookup_futures[(3, "req-1")] = f0
    state.lookup_futures[(3, "req-2")] = f1
    state.lookup_prefetch_cache[(3, "req-1")] = {1: torch.zeros((1, 8))}
    state.lookup_prefetch_cache[(3, "req-2")] = {1: torch.zeros((1, 8))}
    state.lookup_prefetch_ready_events[(3, "req-1")] = None
    state.lookup_prefetch_ready_events[(3, "req-2")] = None
    decode_future: Future[dict[int, torch.Tensor]] = Future()
    decode_future.set_result({1: torch.ones((2, 8), dtype=torch.bfloat16)})
    state.decode_lookup_batch_futures[3] = decode_future
    state.decode_lookup_batch_cache[3] = {1: torch.ones((2, 8), dtype=torch.bfloat16)}
    state.decode_lookup_batch_ready_events[3] = None
    state.decode_lookup_batch_req_ids[3] = ("req-1", "req-2")
    state.decode_lookup_batch_req_pos[3] = {"req-1": 0, "req-2": 1}

    state.cleanup_requests(["req-1"])
    assert (3, "req-1") not in state.lookup_futures
    assert (3, "req-2") in state.lookup_futures
    assert (3, "req-1") not in state.lookup_prefetch_cache
    assert (3, "req-2") in state.lookup_prefetch_cache
    assert (3, "req-1") not in state.lookup_prefetch_ready_events
    assert (3, "req-2") in state.lookup_prefetch_ready_events
    assert 3 not in state.decode_lookup_batch_futures
    assert 3 not in state.decode_lookup_batch_cache
    assert 3 not in state.decode_lookup_batch_ready_events
    assert 3 not in state.decode_lookup_batch_req_ids
    assert 3 not in state.decode_lookup_batch_req_pos


def test_consume_decode_prefetched_layer_or_fallback_future_ready():
    """@brief Validate that consume decode prefetched layer or fallback future ready.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=32,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    f0: Future[dict[int, torch.Tensor]] = Future()
    payload = torch.stack(
        [
            torch.full((32,), 1, dtype=torch.bfloat16),
            torch.full((32,), 2, dtype=torch.bfloat16),
            torch.full((32,), 3, dtype=torch.bfloat16),
        ],
        dim=0,
    )
    f0.set_result({1: payload})
    state.decode_lookup_batch_futures[9] = f0
    state.decode_lookup_batch_req_ids[9] = ("req-a", "req-b", "req-c")
    state.decode_lookup_batch_req_pos[9] = {"req-a": 0, "req-b": 1, "req-c": 2}

    fallback_called = {"value": False}

    def fallback() -> torch.Tensor:
        """@brief Implement fallback.

        Args:
            None.

        Returns:
            torch.Tensor: Computed result for this helper.
                Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
        """
        fallback_called["value"] = True
        return torch.zeros((2, 32), dtype=torch.bfloat16)

    out = state.consume_decode_prefetched_layer_or_fallback(
        req_ids=["req-b", "req-c"],
        layer_idx=1,
        step_uid=9,
        fallback_fn=fallback,
        wait_timeout_s=0.0,
    )
    assert out.shape == (2, 32)
    assert out[0, 0].item() == 2
    assert out[1, 0].item() == 3
    assert fallback_called["value"] is False


def test_consume_decode_prefetched_layer_or_fallback_order_mismatch_uses_fallback():
    """@brief Validate that consume decode prefetched layer or fallback order mismatch uses fallback.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=32,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    f0: Future[dict[int, torch.Tensor]] = Future()
    f0.set_result({1: torch.ones((3, 32), dtype=torch.bfloat16)})
    state.decode_lookup_batch_futures[9] = f0
    state.decode_lookup_batch_req_ids[9] = ("req-a", "req-b", "req-c")
    state.decode_lookup_batch_req_pos[9] = {"req-a": 0, "req-b": 1, "req-c": 2}

    fallback_called = {"value": False}

    def fallback() -> torch.Tensor:
        """@brief Implement fallback.

        Args:
            None.

        Returns:
            torch.Tensor: Computed result for this helper.
                Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
        """
        fallback_called["value"] = True
        return torch.zeros((2, 32), dtype=torch.bfloat16)

    out = state.consume_decode_prefetched_layer_or_fallback(
        req_ids=["req-c", "req-a"],
        layer_idx=1,
        step_uid=9,
        fallback_fn=fallback,
        wait_timeout_s=0.0,
    )
    assert out.shape == (2, 32)
    assert fallback_called["value"] is True


def test_consume_prefetched_layer_or_fallback_with_timeout():
    """Test per-request prefetch consumption with timeout and fallback."""
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=1008,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=2026)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=2026)
    state.engram_layer_indices = (1, 15)
    state.init_async_executor(max_workers=2)

    input_ids = torch.tensor([101, 102, 103, 201, 202], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    request_ids = ["req-a", "req-b"]

    state.schedule_async_step_prefetch(
        step_uid=1,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
    )

    fallback_called = {"value": False}

    def fallback_fn():
        fallback_called["value"] = True
        return torch.zeros((3, state.mem_dim), dtype=torch.bfloat16)

    result = state.consume_prefetched_layer_or_fallback(
        req_id="req-a",
        layer_idx=1,
        step_uid=1,
        fallback_fn=fallback_fn,
        wait_for_ready_timeout_s=2.0,
    )
    assert result is not None
    assert result.shape[0] == 3
    assert result.shape[1] == state.mem_dim
    assert not fallback_called["value"]

    fallback_called["value"] = False
    result = state.consume_prefetched_layer_or_fallback(
        req_id="req-nonexistent",
        layer_idx=1,
        step_uid=1,
        fallback_fn=fallback_fn,
        wait_for_ready_timeout_s=0.1,
    )
    assert result is not None
    assert fallback_called["value"]

    state.shutdown_async_executor()


def test_shutdown_async_executor_clears_all_state():
    """Test shutdown_async_executor clears futures and stops executor."""
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=1008,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=2027)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=2027)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=2027)
    state.engram_layer_indices = (1, 15)

    state.init_async_executor(max_workers=2)
    assert state._async_executor is not None

    input_ids = torch.tensor([101, 102, 201], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    request_ids = ["req-a", "req-b"]

    state.schedule_async_step_prefetch(
        step_uid=5,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
    )

    assert len(state.lookup_futures) > 0

    state.shutdown_async_executor()

    assert state._async_executor is None
    assert len(state.lookup_futures) == 0
    assert len(state.decode_lookup_batch_futures) == 0

    state.shutdown_async_executor()
    assert state._async_executor is None


def test_drop_stale_lookup_futures_boundary_conditions():
    """Test stale lookup futures cleanup with boundary step_uids."""
    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=1008,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=2028)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=2028)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=2028)
    state.engram_layer_indices = (1, 15)
    state.init_async_executor(max_workers=2)

    input_ids = torch.tensor([101, 102, 103], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    request_ids = ["req-a"]

    for step_num in [1, 2, 3]:
        state.schedule_async_step_prefetch(
            step_uid=step_num,
            input_ids=input_ids,
            query_start_loc=query_start_loc,
            request_ids=request_ids,
            layer_indices=[1, 15],
        )

    state._drop_stale_lookup_futures(current_step_uid=3)

    assert (1, "req-a") not in state.lookup_futures
    assert (2, "req-a") in state.lookup_futures
    assert (3, "req-a") in state.lookup_futures

    state.shutdown_async_executor()


def test_consume_decode_prefetch_exception_handling():
    """Test decode prefetch handles future exceptions and falls back."""
    import time
    import unittest.mock

    state = EngramRuntimeState(
        memory_size=128,
        max_ngram_order=3,
        heads=8,
        mem_dim=1008,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=2030)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=2030)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=2030)
    state.engram_layer_indices = (1, 15)
    state.init_async_executor(max_workers=2)

    input_ids = torch.tensor([101, 201], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    request_ids = ["req-a", "req-b"]

    state.schedule_async_decode_step_prefetch(
        step_uid=7,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
    )

    time.sleep(0.5)

    with unittest.mock.patch.object(
        state.decode_lookup_batch_futures[7],
        'result',
        side_effect=RuntimeError("Simulated async failure")
    ):
        fallback_called = {"value": False}

        def fallback_fn():
            fallback_called["value"] = True
            return torch.zeros((2, state.mem_dim), dtype=torch.bfloat16)

        result = state.consume_decode_prefetched_layer_or_fallback(
            req_ids=["req-a", "req-b"],
            layer_idx=1,
            step_uid=7,
            fallback_fn=fallback_fn,
            wait_timeout_s=0.0,
        )
        assert result is not None
        assert result.shape == (2, state.mem_dim)
        assert fallback_called["value"]

    state.shutdown_async_executor()
