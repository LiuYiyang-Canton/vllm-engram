# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Forward-context and runtime plumbing tests for Engram integration.

from types import SimpleNamespace
import copy

import pytest
import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.engram import (
    ENGRAM_STEP_PAYLOAD_KEY,
    EngramLayer,
    EngramRuntimeState,
)


def _make_fake_vllm_config():
    """@brief Implement  make fake vllm config.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def _make_runtime_state(memory_size: int = 128, max_ngram_order: int = 3) -> EngramRuntimeState:
    """@brief Implement  make runtime state.

    Args:
        memory_size (int): Maximum request history length retained in runtime state.
            Shape: N/A. Dtype: N/A unless tensor.
        max_ngram_order (int): Maximum n-gram order used by Engram hashing.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        EngramRuntimeState: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    num_tables = (max_ngram_order - 1) * 8
    mem_dim = (1024 // num_tables) * num_tables
    return EngramRuntimeState(
        memory_size=memory_size,
        max_ngram_order=max_ngram_order,
        heads=8,
        mem_dim=mem_dim,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=max_ngram_order,
    )


def _has_usable_cuda() -> bool:
    """@brief Implement  has usable cuda.

    Args:
        None.

    Returns:
        bool: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.empty(1, device="cuda")
        return True
    except Exception:
        return False


def test_ensure_request_initializes_state():
    """@brief Validate that ensure request initializes state.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = _make_runtime_state(memory_size=128, max_ngram_order=4)
    req_state = state.ensure_request("req-1")

    assert req_state["token_history"] == []


def test_append_tokens_updates_history():
    """@brief Validate that append tokens updates history.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = _make_runtime_state(memory_size=128, max_ngram_order=4)

    state.append_tokens("req-1", [3, 4, 5])
    req_state = state.get_request_state("req-1")

    assert req_state["token_history"] == [3, 4, 5]


def test_append_tokens_respects_memory_size_cap():
    """@brief Validate that append tokens respects memory size cap.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = _make_runtime_state(memory_size=3, max_ngram_order=3)

    state.append_tokens("req-1", [1, 2, 3, 4])
    req_state = state.get_request_state("req-1")

    assert req_state["token_history"] == [2, 3, 4]


def test_cleanup_requests_removes_request_state():
    """@brief Validate that cleanup requests removes request state.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = _make_runtime_state(memory_size=128, max_ngram_order=4)
    state.append_tokens("req-1", [42])

    state.cleanup_requests(["req-1"])

    assert state.get_request_state("req-1") is None


def test_set_forward_context_merges_extra_additional_kwargs():
    """@brief Validate that set forward context merges extra additional kwargs.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    payload = {
        "step_uid": 3,
        "input_ids": torch.tensor([101], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "request_ids": ["req-a"],
    }
    with set_forward_context(
        attn_metadata=None,
        vllm_config=_make_fake_vllm_config(),
        extra_additional_kwargs={ENGRAM_STEP_PAYLOAD_KEY: payload},
    ):
        ctx = get_forward_context()
        assert ENGRAM_STEP_PAYLOAD_KEY in ctx.additional_kwargs
        assert ctx.additional_kwargs[ENGRAM_STEP_PAYLOAD_KEY]["step_uid"] == 3


def test_engram_layer_forward_updates_runtime_state_from_real_input_ids(monkeypatch):
    """@brief Validate that engram layer forward updates runtime state from real input ids.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.w_k is not None
    hidden_device = layer.w_k.device
    hidden_states = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        dtype=torch.bfloat16,
        device=hidden_device,
    )
    payload = {
        "step_uid": 7,
        "input_ids": torch.tensor([101, 102], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "request_ids": ["req-a"],
    }
    fake_context = SimpleNamespace(additional_kwargs={ENGRAM_STEP_PAYLOAD_KEY: payload})

    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.get_forward_context",
        lambda: fake_context,
    )

    output = layer(hidden_states)

    assert output.shape == hidden_states.shape
    assert output.data_ptr() == hidden_states.data_ptr()
    req_state = runtime_state.get_request_state("req-a")
    assert req_state is not None
    assert req_state["token_history"] == [101, 102]


def test_engram_layer_dedups_same_step_uid(monkeypatch):
    """@brief Validate that engram layer dedups same step uid.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.w_k is not None
    hidden_states = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4]],
        dtype=torch.bfloat16,
        device=layer.w_k.device,
    )
    payload = {
        "step_uid": 11,
        "input_ids": torch.tensor([123], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "request_ids": ["req-a"],
    }
    fake_context = SimpleNamespace(additional_kwargs={ENGRAM_STEP_PAYLOAD_KEY: payload})

    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.get_forward_context",
        lambda: fake_context,
    )

    layer(hidden_states)
    layer(hidden_states)

    req_state = runtime_state.get_request_state("req-a")
    assert req_state is not None
    assert req_state["token_history"] == [123]


def test_engram_layer_requires_engram_step_payload(monkeypatch):
    """@brief Validate that engram layer requires engram step payload.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    hidden_states = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    fake_context = SimpleNamespace(additional_kwargs={})

    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.is_forward_context_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_layer.get_forward_context",
        lambda: fake_context,
    )

    output = layer(hidden_states)
    assert output.shape == hidden_states.shape
    assert runtime_state.get_request_state("req-a") is None


def test_engram_layer_forward_dispatches_decode_path(monkeypatch):
    """@brief Validate that engram layer forward dispatches decode path.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.w_k is not None
    hidden_states = torch.randn((1, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    payload = {
        "step_uid": 13,
        "input_ids": torch.tensor([42], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 1], dtype=torch.int32),
        "request_ids": ["req-a"],
    }

    calls = {"decode": 0, "prefill": 0}
    dummy_segments = [("req-a", {}, 0, 1, torch.zeros((1, 1), dtype=torch.bfloat16))]

    monkeypatch.setattr(layer, "_get_engram_step_payload", lambda: payload)
    monkeypatch.setattr(
        layer,
        "_collect_active_segments",
        lambda **_: (dummy_segments, True),
    )
    monkeypatch.setattr(
        layer,
        "_decode_forward",
        lambda hs, segments, step_uid=None: calls.__setitem__(
            "decode", calls["decode"] + 1
        )
        or hs,
    )
    monkeypatch.setattr(
        layer,
        "_prefill_forward",
        lambda hs, segments, step_uid=None: calls.__setitem__(
            "prefill", calls["prefill"] + 1
        )
        or hs,
    )

    output = layer(hidden_states)
    assert output.data_ptr() == hidden_states.data_ptr()
    assert calls["decode"] == 1
    assert calls["prefill"] == 0


def test_engram_layer_forward_dispatches_prefill_path(monkeypatch):
    """@brief Validate that engram layer forward dispatches prefill path.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.w_k is not None
    hidden_states = torch.randn((2, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    payload = {
        "step_uid": 14,
        "input_ids": torch.tensor([42, 43], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
        "request_ids": ["req-a"],
    }

    calls = {"decode": 0, "prefill": 0}
    dummy_segments = [("req-a", {}, 0, 2, torch.zeros((2, 1), dtype=torch.bfloat16))]

    monkeypatch.setattr(layer, "_get_engram_step_payload", lambda: payload)
    monkeypatch.setattr(
        layer,
        "_collect_active_segments",
        lambda **_: (dummy_segments, False),
    )
    monkeypatch.setattr(
        layer,
        "_decode_forward",
        lambda hs, segments, step_uid=None: calls.__setitem__(
            "decode", calls["decode"] + 1
        )
        or hs,
    )
    monkeypatch.setattr(
        layer,
        "_prefill_forward",
        lambda hs, segments, step_uid=None: calls.__setitem__(
            "prefill", calls["prefill"] + 1
        )
        or hs,
    )

    output = layer(hidden_states)
    assert output.data_ptr() == hidden_states.data_ptr()
    assert calls["decode"] == 0
    assert calls["prefill"] == 1


def test_copy_lookup_to_device_async_pins_non_pinned_source():
    """@brief Validate async lookup H2D helper repins non-pinned CPU tensors."""
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for H2D helper test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    hidden_states = torch.empty((1, 4), device="cuda", dtype=torch.bfloat16)
    lookup_cpu = torch.zeros((1, 1024), dtype=torch.bfloat16, device="cpu")
    lookup = layer._copy_lookup_to_device_async(lookup_cpu, hidden_states)
    assert lookup.device == hidden_states.device
    assert lookup.dtype == hidden_states.dtype
    assert lookup.shape == lookup_cpu.shape


def test_copy_lookup_to_device_async_returns_cuda_tensor():
    """@brief Validate async lookup H2D helper returns correct CUDA tensor."""
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for H2D helper test.")
    runtime_state = _make_runtime_state(memory_size=8, max_ngram_order=3)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    hidden_states = torch.empty((2, 4), device="cuda", dtype=torch.bfloat16)
    lookup_cpu = torch.randn((2, 1024), dtype=torch.bfloat16, device="cpu").pin_memory()
    lookup = layer._copy_lookup_to_device_async(lookup_cpu, hidden_states)
    assert lookup.device == hidden_states.device
    assert lookup.dtype == hidden_states.dtype
    assert lookup.shape == lookup_cpu.shape
    assert torch.equal(lookup.cpu(), lookup_cpu)


def test_prefill_and_decode_use_async_lookup_copy(monkeypatch):
    """@brief Validate prefill and decode paths both use async lookup H2D helper."""
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for H2D helper test.")
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=2026)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=2026)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.w_k is not None
    hidden_states_prefill = torch.randn((2, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    hidden_states_decode = torch.randn((2, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    req_a = runtime_state.ensure_request("req-a")
    req_b = runtime_state.ensure_request("req-b")
    active_prefill = [("req-a", req_a, 0, 2, 2)]
    active_decode = [("req-a", req_a, 0, 1, 1), ("req-b", req_b, 1, 2, 1)]
    lookup_prefill_cpu = torch.randn((2, 1024), dtype=torch.bfloat16, device="cpu").pin_memory()
    lookup_decode_cpu = torch.randn((2, 1024), dtype=torch.bfloat16, device="cpu").pin_memory()

    calls = {"count": 0}

    def _copy_spy(lookup_cpu, hidden_states):
        calls["count"] += 1
        return torch.zeros(
            lookup_cpu.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    monkeypatch.setattr(
        runtime_state,
        "consume_prefetched_layer_or_fallback",
        lambda **_: lookup_prefill_cpu,
    )
    monkeypatch.setattr(
        runtime_state,
        "consume_decode_prefetched_layer_or_fallback",
        lambda **_: lookup_decode_cpu,
    )
    monkeypatch.setattr(layer, "_copy_lookup_to_device_async", _copy_spy)
    monkeypatch.setattr(
        layer,
        "_compute_preconv_tensors",
        lambda hidden_slice, lookup_embeddings: (
            torch.zeros_like(hidden_slice),
            torch.zeros_like(hidden_slice),
        ),
    )
    monkeypatch.setattr(layer, "_conv_with_state", lambda u_norm, req_state: torch.zeros_like(u_norm))
    monkeypatch.setattr(
        layer,
        "_conv_decode_batch_with_state",
        lambda u_norm_batch, req_states: torch.zeros_like(u_norm_batch),
    )

    _ = layer._prefill_forward(hidden_states_prefill, active_prefill, step_uid=11)
    _ = layer._decode_forward(hidden_states_decode, active_decode, step_uid=12)
    assert calls["count"] == 2


def test_collect_active_segments_skips_empty_and_preserves_order():
    """@brief Validate active-segment collection skips empty spans and preserves request order."""
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    input_ids = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    request_ids = ["req-a", "req-b", "req-c"]

    active_segments, decode_only = layer._collect_active_segments(
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        step_uid=101,
    )

    assert decode_only is False
    assert [(seg[0], seg[2], seg[3], seg[4]) for seg in active_segments] == [
        ("req-a", 0, 2, 2),
        ("req-c", 2, 5, 3),
    ]
    assert runtime_state.get_request_state("req-a")["token_history"] == [10, 11]
    assert runtime_state.get_request_state("req-b") is None
    assert runtime_state.get_request_state("req-c")["token_history"] == [20, 21, 22]


def test_collect_active_segments_dedups_on_duplicate_step_uid():
    """@brief Validate active-segment collection deduplicates duplicate step UIDs."""
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    input_ids = torch.tensor([30, 31], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    request_ids = ["req-a"]

    first_segments, first_decode_only = layer._collect_active_segments(
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        step_uid=202,
    )
    second_segments, second_decode_only = layer._collect_active_segments(
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        step_uid=202,
    )

    assert first_decode_only is False
    assert len(first_segments) == 1
    assert second_segments == []
    assert second_decode_only is True
    assert runtime_state.get_request_state("req-a")["token_history"] == [30, 31]


def test_collect_active_segments_decode_only_classification():
    """@brief Validate decode-only classification based on accepted segment lengths."""
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )

    decode_segments, decode_only = layer._collect_active_segments(
        input_ids=torch.tensor([1, 2], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        request_ids=["req-a", "req-b"],
        step_uid=303,
    )
    mixed_segments, mixed_decode_only = layer._collect_active_segments(
        input_ids=torch.tensor([7, 8, 9], dtype=torch.int64),
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        request_ids=["req-c", "req-d"],
        step_uid=304,
    )

    assert len(decode_segments) == 2
    assert decode_only is True
    assert len(mixed_segments) == 2
    assert mixed_decode_only is False


def test_decode_forward_uses_batched_decode_lookup_once(monkeypatch):
    """@brief Validate that decode forward uses batched decode lookup once.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=19)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=29)
    assert layer.w_k is not None

    runtime_state.append_tokens("req-a", [101])
    runtime_state.append_tokens("req-b", [201])
    req_a = runtime_state.get_request_state("req-a")
    req_b = runtime_state.get_request_state("req-b")
    assert req_a is not None
    assert req_b is not None

    calls = {"batched_lookup": 0}
    original = runtime_state.build_lookup_for_requests_layer_decode

    def wrapped(*, req_states, layer_idx):
        """@brief Wrap and count calls to the intercepted function under test.

        Args:
            req_states (Any): List of per-request runtime state dictionaries.
                Shape: N/A. Dtype: N/A unless tensor.
            layer_idx (Any): Transformer layer index.
                Shape: N/A. Dtype: N/A unless tensor.

        Returns:
            Any: Computed result for this helper.
                Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
        """
        calls["batched_lookup"] += 1
        return original(req_states=req_states, layer_idx=layer_idx)

    monkeypatch.setattr(
        runtime_state, "build_lookup_for_requests_layer_decode", wrapped
    )
    hidden_states = torch.randn((2, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    active_segments = [
        ("req-a", req_a, 0, 1, 1),
        ("req-b", req_b, 1, 2, 1),
    ]

    output = layer._decode_forward(hidden_states, active_segments, step_uid=7)
    assert output.shape == hidden_states.shape
    assert calls["batched_lookup"] == 1


def test_decode_forward_matches_reference_path(monkeypatch):
    """@brief Validate that decode forward matches reference path.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if not _has_usable_cuda():
        pytest.skip("CUDA is required for EngramLayer forward-path test.")
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=19)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=8,
        engram_mem_dim=1024,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=runtime_state,
    )
    layer.init_synthetic_layer_parameters(seed=29)
    assert layer.w_k is not None

    runtime_state.append_tokens("req-a", [101])
    runtime_state.append_tokens("req-b", [201])
    req_a = runtime_state.get_request_state("req-a")
    req_b = runtime_state.get_request_state("req-b")
    assert req_a is not None
    assert req_b is not None
    req_a_fast = copy.deepcopy(req_a)
    req_b_fast = copy.deepcopy(req_b)
    req_a_ref = copy.deepcopy(req_a)
    req_b_ref = copy.deepcopy(req_b)

    hidden = torch.randn((2, 4), dtype=torch.bfloat16, device=layer.w_k.device)
    fast_in = hidden.clone()
    ref_in = hidden.clone()

    fast_segments = [
        ("req-a", req_a_fast, 0, 1, 1),
        ("req-b", req_b_fast, 1, 2, 1),
    ]
    ref_segments = [
        ("req-a", req_a_ref, 0, 1, 1),
        ("req-b", req_b_ref, 1, 2, 1),
    ]

    fast_out = layer._decode_forward(fast_in, fast_segments, step_uid=11)

    start_indices = torch.tensor([0, 1], device=ref_in.device, dtype=torch.long)
    hidden_cat = ref_in.index_select(0, start_indices)
    lookup_cpu = runtime_state.build_lookup_for_requests_layer_decode(
        [req_a_ref, req_b_ref], layer_idx=layer.layer_idx
    )
    lookup = lookup_cpu.to(
        device=ref_in.device, dtype=ref_in.dtype, non_blocking=True
    )
    u_all, u_norm_all = layer._compute_preconv_tensors(
        hidden_slice=hidden_cat, lookup_embeddings=lookup
    )
    delta = torch.empty_like(hidden_cat)
    for idx, (_, req_state, _, _, _) in enumerate(ref_segments):
        conv_out = layer._conv_one_token_with_state(
            u_norm_all[idx : idx + 1], req_state=req_state
        )
        delta[idx : idx + 1] = F.silu(conv_out) + u_all[idx : idx + 1]
    ref_in.index_add_(0, start_indices, delta)

    assert torch.allclose(fast_out, ref_in, rtol=0, atol=0)


def test_schedule_async_step_prefetch_creates_lookup_futures():
    """@brief Validate that schedule async step prefetch creates lookup futures.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=41)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=41)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=15, seed=41)
    runtime_state.engram_layer_indices = (1, 15)
    runtime_state.init_async_executor(max_workers=2)

    input_ids = torch.tensor([11, 12, 21], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    request_ids = ["req-a", "req-b"]
    runtime_state.schedule_async_step_prefetch(
        step_uid=5,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
        target_device=torch.device("cuda:0") if torch.cuda.is_available() else None,
        target_dtype=torch.bfloat16,
    )

    keys = set(runtime_state.lookup_futures.keys())
    assert (5, "req-a") in keys
    assert (5, "req-b") in keys
    payload_a = runtime_state.lookup_futures[(5, "req-a")].result(timeout=1.0)
    payload_b = runtime_state.lookup_futures[(5, "req-b")].result(timeout=1.0)
    assert 1 in payload_a and 15 in payload_a
    assert 1 in payload_b and 15 in payload_b
    if torch.cuda.is_available():
        assert payload_a[1].is_cuda
        assert payload_a[15].is_cuda
        assert payload_b[1].is_cuda
        assert payload_b[15].is_cuda
        assert runtime_state.lookup_prefetch_ready_events[(5, "req-a")] is not None
        assert runtime_state.lookup_prefetch_ready_events[(5, "req-b")] is not None
    else:
        assert payload_a[1].device.type == "cpu"
        assert payload_b[1].device.type == "cpu"

    req_a = runtime_state.get_request_state("req-a")
    req_b = runtime_state.get_request_state("req-b")
    assert req_a is not None and req_b is not None
    assert req_a["token_history"] == [11, 12]
    assert req_b["token_history"] == [21]

    runtime_state.schedule_async_step_prefetch(
        step_uid=5,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
    )
    req_a = runtime_state.get_request_state("req-a")
    req_b = runtime_state.get_request_state("req-b")
    assert req_a is not None and req_b is not None
    assert req_a["token_history"] == [11, 12]
    assert req_b["token_history"] == [21]

    runtime_state.shutdown_async_executor()


def test_schedule_async_decode_step_prefetch_creates_batched_future():
    """@brief Validate that schedule async decode step prefetch creates batched future.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    runtime_state = _make_runtime_state(memory_size=16, max_ngram_order=3)
    runtime_state.init_synthetic_compression_map(seed=53)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=1, seed=53)
    runtime_state.init_synthetic_layer_host_tables(layer_idx=15, seed=53)
    runtime_state.engram_layer_indices = (1, 15)
    runtime_state.init_async_executor(max_workers=2)

    input_ids = torch.tensor([11, 21], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    request_ids = ["req-a", "req-b"]
    runtime_state.schedule_async_decode_step_prefetch(
        step_uid=7,
        input_ids=input_ids,
        query_start_loc=query_start_loc,
        request_ids=request_ids,
        layer_indices=[1, 15],
        target_device=torch.device("cuda:0") if torch.cuda.is_available() else None,
        target_dtype=torch.bfloat16,
    )

    assert 7 in runtime_state.decode_lookup_batch_futures
    assert runtime_state.decode_lookup_batch_req_ids[7] == ("req-a", "req-b")
    payload = runtime_state.decode_lookup_batch_futures[7].result(timeout=1.0)
    assert 1 in payload and 15 in payload
    assert payload[1].shape[0] == 2
    assert payload[15].shape[0] == 2
    if torch.cuda.is_available():
        assert payload[1].is_cuda
        assert payload[15].is_cuda
        assert runtime_state.decode_lookup_batch_ready_events[7] is not None
    else:
        assert payload[1].device.type == "cpu"
        assert payload[15].device.type == "cpu"

    runtime_state.shutdown_async_executor()
