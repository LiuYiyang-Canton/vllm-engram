# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Hash, lookup, and batching helper tests for Engram runtime behavior.

import torch
import pytest

from vllm.model_executor.layers.engram import (
    EngramLayer,
    EngramRuntimeState,
    _compute_deepseek_hash_ids_last_token,
    _compute_deepseek_hash_ids_for_sequence,
    compute_deepseek_hash_ids_last_token_batch,
    build_engram_step_payload,
    slice_engram_step_payload,
    validate_engram_step_payload,
)


def _history_entry_as_logical_tensor(entry):
    """@brief Implement  history entry as logical tensor.

    Args:
        entry (Any): Entry input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if isinstance(entry, dict):
        history = entry["history"]
        if history.numel() == 0:
            return history
        head = int(entry.get("head", 0)) % history.shape[0]
        if head == 0:
            return history
        return torch.cat([history[head:], history[:head]], dim=0)
    return entry


def _compute_deepseek_hash_ids_scalar_reference(
    *,
    compressed_tokens: torch.Tensor,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes_by_order: dict[int, torch.Tensor],
    max_ngram_order: int,
    pad_id: int,
) -> torch.Tensor:
    """@brief Implement  compute deepseek hash ids scalar reference.

    Args:
        compressed_tokens (torch.Tensor): Compressed token IDs for a single request.
            Shape: [T]. Dtype: torch.int64.
        layer_multipliers (torch.Tensor): Per-order multipliers used by DeepSeek hash mixing.
            Shape: [max_ngram_order]. Dtype: torch.int64.
        head_vocab_sizes_by_order (dict[int, torch.Tensor]): Per-order modulus tensors for each head.
            Shape: dict: each value [heads]. Dtype: torch.int64 tensors.
        max_ngram_order (int): Maximum n-gram order used by Engram hashing.
            Shape: N/A. Dtype: N/A unless tensor.
        pad_id (int): Pad ID used for unavailable history positions.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        torch.Tensor: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    tokens = compressed_tokens.to(dtype=torch.int64, device="cpu")
    multipliers = layer_multipliers.to(dtype=torch.int64, device="cpu")
    t = int(tokens.shape[0])
    all_hashes: list[torch.Tensor] = []
    for order in range(2, max_ngram_order + 1):
        head_vocab_sizes = head_vocab_sizes_by_order[order]
        for head_idx in range(head_vocab_sizes.shape[0]):
            mod = int(head_vocab_sizes[head_idx].item())
            column = torch.zeros(t, dtype=torch.int64)
            for pos in range(t):
                mix = int(tokens[pos].item()) * int(multipliers[0].item())
                for k in range(1, order):
                    prev = int(tokens[pos - k].item()) if pos - k >= 0 else int(pad_id)
                    mix = mix ^ (prev * int(multipliers[k].item()))
                column[pos] = mix % mod
            all_hashes.append(column)
    return torch.stack(all_hashes, dim=1)


def test_deepseek_hash_is_deterministic():
    """@brief Validate that deepseek hash is deterministic.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    compressed_tokens = torch.tensor([101, 22, 303, 77, 88], dtype=torch.int64)
    layer_multipliers = torch.tensor([17, 31, 43, 59], dtype=torch.int64)
    head_vocab_sizes_by_order = {
        2: torch.tensor([64, 64], dtype=torch.int64),
        3: torch.tensor([64, 64], dtype=torch.int64),
        4: torch.tensor([64, 64], dtype=torch.int64),
    }
    h0 = _compute_deepseek_hash_ids_for_sequence(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    h1 = _compute_deepseek_hash_ids_for_sequence(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    assert torch.equal(h0, h1)


def test_deepseek_hash_vectorized_matches_scalar_reference():
    """@brief Validate that deepseek hash vectorized matches scalar reference.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    compressed_tokens = torch.tensor([101, 22, 303, 77, 88], dtype=torch.int64)
    layer_multipliers = torch.tensor([17, 31, 43, 59], dtype=torch.int64)
    head_vocab_sizes_by_order = {
        2: torch.tensor([64, 64], dtype=torch.int64),
        3: torch.tensor([64, 64], dtype=torch.int64),
        4: torch.tensor([64, 64], dtype=torch.int64),
    }
    vectorized = _compute_deepseek_hash_ids_for_sequence(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    scalar = _compute_deepseek_hash_ids_scalar_reference(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    assert torch.equal(vectorized, scalar)


def test_deepseek_hash_last_token_matches_sequence_tail():
    """@brief Validate that deepseek hash last token matches sequence tail.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    compressed_tokens = torch.tensor([101, 22, 303, 77, 88], dtype=torch.int64)
    layer_multipliers = torch.tensor([17, 31, 43, 59], dtype=torch.int64)
    head_vocab_sizes_by_order = {
        2: torch.tensor([64, 64], dtype=torch.int64),
        3: torch.tensor([64, 64], dtype=torch.int64),
        4: torch.tensor([64, 64], dtype=torch.int64),
    }
    full = _compute_deepseek_hash_ids_for_sequence(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    last = _compute_deepseek_hash_ids_last_token(
        compressed_tokens=compressed_tokens,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=4,
        pad_id=0,
    )
    assert torch.equal(last, full[-1])


def test_build_shifted_tokens_direct():
    """Test _build_shifted_tokens edge cases directly."""
    from vllm.model_executor.layers.engram_hash import _build_shifted_tokens

    # Test 1: Normal case
    tokens = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    shifted = _build_shifted_tokens(tokens, max_ngram_order=4, pad_id=0)

    assert len(shifted) == 4  # orders 1, 2, 3, 4 (k=0,1,2,3)
    assert torch.equal(shifted[0], tokens)  # k=0, no shift
    assert shifted[1][0] == 0  # k=1, first position padded
    assert torch.equal(shifted[1][1:], tokens[:3])

    # Test 2: Empty tensor
    empty_tokens = torch.tensor([], dtype=torch.int64)
    shifted_empty = _build_shifted_tokens(empty_tokens, max_ngram_order=3, pad_id=0)
    assert len(shifted_empty) == 3
    assert all(t.shape[0] == 0 for t in shifted_empty)

    # Test 3: Single token
    single = torch.tensor([42], dtype=torch.int64)
    shifted_single = _build_shifted_tokens(single, max_ngram_order=4, pad_id=99)
    assert shifted_single[0][0] == 42
    assert shifted_single[1][0] == 99  # Padded
    assert shifted_single[2][0] == 99  # Padded
    assert shifted_single[3][0] == 99  # Padded


def test_hash_computation_workspace_undersized():
    """Test hash computation reallocates undersized workspaces."""
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
    state.init_synthetic_compression_map(seed=2032)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=2032)

    state._decode_tokens_by_offset_workspace = torch.empty(
        (state.max_ngram_order, 2),
        dtype=torch.int64,
        device="cpu"
    )

    req_ids = [f"req-{i}" for i in range(5)]
    req_states = []
    for i, req_id in enumerate(req_ids):
        state.ensure_request(req_id)
        state.append_tokens(req_id, torch.tensor([100 + i], dtype=torch.int64))
        req_states.append(state.get_request_state(req_id))

    result = state.build_lookup_for_requests_layer_decode(
        req_states=req_states,
        layer_idx=1,
    )

    assert result.shape[0] == 5
    assert result.shape[1] == state.mem_dim
    assert state._decode_tokens_by_offset_workspace.shape[1] >= 5


def _build_lookup_embeddings_reference(
    state: EngramRuntimeState,
    req_state: dict[str, object],
    table_layout: list[tuple[int, int, int, int]],
    *,
    layer_idx: int,
    mem_dim: int,
    new_token_count: int,
) -> torch.Tensor:
    """@brief Implement  build lookup embeddings reference.

    Args:
        state (EngramRuntimeState): State input.
            Shape: N/A. Dtype: N/A unless tensor.
        req_state (dict[str, object]): Mutable per-request runtime state dictionary.
            Shape: N/A. Dtype: N/A unless tensor.
        table_layout (list[tuple[int, int, int, int]]): Table slice layout for (order, head) lookup tables.
            Shape: list of (order, head, start, end). Dtype: N/A unless tensor.
        layer_idx (int): Transformer layer index.
            Shape: N/A. Dtype: N/A unless tensor.
        mem_dim (int): Total lookup embedding width.
            Shape: N/A. Dtype: N/A unless tensor.
        new_token_count (int): Number of newly appended tokens for this request.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        torch.Tensor: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    if new_token_count <= 0:
        return torch.zeros((0, mem_dim), dtype=torch.bfloat16)
    host = state.host_tables_by_layer[layer_idx]
    compression = state.token_compression_map
    assert compression is not None

    token_history = req_state["token_history"]
    history_tensor = torch.as_tensor(token_history, dtype=torch.int64)
    clamped = torch.remainder(history_tensor, compression.shape[0])
    compressed_history = compression[clamped]

    total_tokens = len(token_history)
    start_token = max(0, total_tokens - new_token_count)
    embeddings = torch.zeros((new_token_count, mem_dim), dtype=torch.bfloat16)
    hash_ids = _compute_deepseek_hash_ids_scalar_reference(
        compressed_tokens=compressed_history,
        layer_multipliers=host.layer_multipliers,
        head_vocab_sizes_by_order=host.head_vocab_sizes_by_order,
        max_ngram_order=state.max_ngram_order,
        pad_id=state.compressed_pad_id,
    )
    output_hashes = hash_ids[start_token:total_tokens]
    for out_idx in range(output_hashes.shape[0]):
        for col_idx, (order, head, start, end) in enumerate(table_layout):
            _ = order
            row_id = int(output_hashes[out_idx, col_idx].item())
            _ = head
            embeddings[out_idx, start:end] = host.tables[col_idx, row_id]
    return embeddings


def test_lookup_embeddings_vectorized_matches_scalar_reference():
    """@brief Validate that lookup embeddings vectorized matches scalar reference.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    layer = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )

    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    req_state = state.get_request_state("req-a")
    assert req_state is not None

    new_token_count = 5
    vectorized = layer._build_lookup_embeddings(
        req_state=req_state,
        new_token_count=new_token_count,
    )
    scalar = _build_lookup_embeddings_reference(
        state=state,
        req_state=req_state,
        table_layout=layer.table_layout,
        layer_idx=1,
        mem_dim=128,
        new_token_count=new_token_count,
    )
    assert torch.equal(vectorized, scalar)


def test_decode_lookup_fast_path_matches_full_slice():
    """@brief Validate that decode lookup fast path matches full slice.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    req_state = state.get_request_state("req-a")
    assert req_state is not None

    full = state.build_lookup_for_request_layer(
        req_state=req_state, layer_idx=1, new_token_count=1
    )
    fast = state.build_lookup_for_request_layer_decode(req_state=req_state, layer_idx=1)
    assert full.shape == (1, 128)
    assert fast.shape == (1, 128)
    assert torch.equal(full, fast)


def test_batch_decode_hash_matches_stacked_single_decode_hash():
    """@brief Validate that batch decode hash matches stacked single decode hash.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    histories = [
        torch.tensor([101, 102, 7, 8, 9, 10, 11], dtype=torch.int64),
        torch.tensor([701, 702, 703], dtype=torch.int64),
        torch.tensor([42], dtype=torch.int64),
    ]
    layer_multipliers = torch.tensor([17, 31, 43], dtype=torch.int64)
    head_vocab_sizes_by_order = {
        2: torch.tensor([64, 64, 64, 64], dtype=torch.int64),
        3: torch.tensor([64, 64, 64, 64], dtype=torch.int64),
    }
    batched = compute_deepseek_hash_ids_last_token_batch(
        compressed_histories=histories,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=3,
        pad_id=0,
    )
    stacked = torch.stack(
        [
            _compute_deepseek_hash_ids_last_token(
                compressed_tokens=h,
                layer_multipliers=layer_multipliers,
                head_vocab_sizes_by_order=head_vocab_sizes_by_order,
                max_ngram_order=3,
                pad_id=0,
            )
            for h in histories
        ],
        dim=0,
    )
    assert torch.equal(batched, stacked)


def test_batch_decode_hash_workspace_matches_default_path():
    """@brief Validate that batch decode hash workspace matches default path.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    histories = [
        torch.tensor([101, 102, 7, 8, 9, 10, 11], dtype=torch.int64),
        torch.tensor([701, 702, 703], dtype=torch.int64),
        torch.tensor([42], dtype=torch.int64),
    ]
    layer_multipliers = torch.tensor([17, 31, 43], dtype=torch.int64)
    head_vocab_sizes_by_order = {
        2: torch.tensor([64, 64, 64, 64], dtype=torch.int64),
        3: torch.tensor([64, 64, 64, 64], dtype=torch.int64),
    }
    expected = compute_deepseek_hash_ids_last_token_batch(
        compressed_histories=histories,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=3,
        pad_id=0,
    )
    tokens_ws = torch.empty((3, len(histories)), dtype=torch.int64, device="cpu")
    hash_ws = torch.empty(
        (len(histories), expected.shape[1]), dtype=torch.int64, device="cpu"
    )
    with_workspace = compute_deepseek_hash_ids_last_token_batch(
        compressed_histories=histories,
        layer_multipliers=layer_multipliers,
        head_vocab_sizes_by_order=head_vocab_sizes_by_order,
        max_ngram_order=3,
        pad_id=0,
        tokens_by_offset_workspace=tokens_ws,
        hash_out_workspace=hash_ws,
    )
    assert torch.equal(expected, with_workspace)


def test_batch_decode_lookup_matches_stacked_single_request_decode_lookup():
    """@brief Validate that batch decode lookup matches stacked single request decode lookup.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)

    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    state.append_tokens("req-b", [701, 702, 703, 704, 705])
    state.append_tokens("req-c", [42])
    req_a = state.get_request_state("req-a")
    req_b = state.get_request_state("req-b")
    req_c = state.get_request_state("req-c")
    assert req_a is not None
    assert req_b is not None
    assert req_c is not None
    req_states = [req_a, req_b, req_c]

    batched = state.build_lookup_for_requests_layer_decode(
        req_states=req_states,
        layer_idx=1,
    )
    stacked = torch.cat(
        [
            state.build_lookup_for_request_layer_decode(req_state, layer_idx=1)
            for req_state in req_states
        ],
        dim=0,
    )
    assert batched.shape == stacked.shape
    assert torch.equal(batched, stacked)


def test_joint_lookup_builder_matches_single_layer_builders():
    """@brief Validate that joint lookup builder matches single layer builders.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=17)
    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    req_state = state.get_request_state("req-a")
    assert req_state is not None

    snapshot = state._build_lookup_snapshot(req_state, new_token_count=3)
    joint = state.build_lookup_for_layers_from_snapshot(snapshot, (1, 15))
    single_1 = state.build_lookup_for_layer_from_snapshot(snapshot, 1)
    single_15 = state.build_lookup_for_layer_from_snapshot(snapshot, 15)
    assert torch.equal(joint[1], single_1)
    assert torch.equal(joint[15], single_15)


def test_runtime_decode_lookup_reuses_cpu_workspace_buffers():
    """@brief Validate that runtime decode lookup reuses cpu workspace buffers.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    state.append_tokens("req-b", [701, 702, 703, 704, 705])
    req_a = state.get_request_state("req-a")
    req_b = state.get_request_state("req-b")
    assert req_a is not None
    assert req_b is not None

    _ = state.build_lookup_for_requests_layer_decode([req_a, req_b], layer_idx=1)
    tokens_ws0 = state._decode_tokens_by_offset_workspace
    hash_ws0 = state._decode_hash_out_workspace
    assert tokens_ws0 is not None
    assert hash_ws0 is not None

    _ = state.build_lookup_for_requests_layer_decode([req_a, req_b], layer_idx=1)
    tokens_ws1 = state._decode_tokens_by_offset_workspace
    hash_ws1 = state._decode_hash_out_workspace
    assert tokens_ws1 is not None
    assert hash_ws1 is not None
    assert tokens_ws0.data_ptr() == tokens_ws1.data_ptr()
    assert hash_ws0.data_ptr() == hash_ws1.data_ptr()


def test_lookup_embeddings_short_history_uses_pad_shift_hashes():
    """@brief Validate that lookup embeddings short history uses pad shift hashes.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=32,
        max_ngram_order=4,
        heads=2,
        mem_dim=60,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=4,
    )
    state.init_synthetic_compression_map(seed=3)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=5)
    layer = EngramLayer(
        hidden_size=8,
        layer_idx=1,
        max_ngram_order=4,
        engram_heads=2,
        engram_mem_dim=60,
        conv_kernel=4,
        conv_dilation=4,
        runtime_state=state,
    )

    state.append_tokens("req-a", [42])
    req_state = state.get_request_state("req-a")
    assert req_state is not None

    result = layer._build_lookup_embeddings(req_state=req_state, new_token_count=1)
    assert result.shape == (1, 60)
    reference = _build_lookup_embeddings_reference(
        state=state,
        req_state=req_state,
        table_layout=layer.table_layout,
        layer_idx=1,
        mem_dim=60,
        new_token_count=1,
    )
    assert torch.equal(result, reference)


def test_append_tokens_updates_compressed_token_history_cache():
    """@brief Validate that append tokens updates compressed token history cache.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=8,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=19)
    state.append_tokens("req-a", [1, 2, 3])
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    assert len(req_state["compressed_token_history"]) == len(req_state["token_history"])

    token_tensor = torch.tensor([4, 5], dtype=torch.int64)
    state.append_tokens("req-a", token_tensor)
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    assert len(req_state["compressed_token_history"]) == len(req_state["token_history"])
    assert req_state["compressed_token_history_tensor_valid"] is False


def test_decode_ring_tracks_recent_compressed_tokens():
    """@brief Validate that decode ring tracks recent compressed tokens.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=4,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=31)
    state.append_tokens("req-a", [10, 11, 12, 13, 14, 15])
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    ring = req_state["compressed_decode_ring"]
    decode_len = int(req_state["compressed_decode_len"])
    head = int(req_state["compressed_decode_head"])
    assert decode_len == state.max_ngram_order
    recent = []
    for k in range(decode_len):
        recent.append(int(ring[(head - decode_len + k) % ring.shape[0]].item()))
    assert recent == req_state["compressed_token_history"][-state.max_ngram_order :]


def test_decode_lookup_uses_decode_ring_when_tensor_cache_stale():
    """@brief Validate that decode lookup uses decode ring when tensor cache stale.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    expected = state.build_lookup_for_request_layer_decode(req_state, layer_idx=1)

    # Force legacy tensor/list cache stale while keeping token history intact.
    req_state["compressed_token_history"] = []
    req_state["compressed_token_history_tensor"] = torch.empty(
        0, dtype=torch.int64, device="cpu"
    )
    req_state["compressed_token_history_tensor_valid"] = False

    out = state.build_lookup_for_requests_layer_decode([req_state], layer_idx=1)
    assert torch.equal(out, expected)


def test_build_lookup_snapshot_decode_uses_ring_tail_when_cache_is_stale():
    """@brief Validate that build lookup snapshot decode uses ring tail when cache is stale.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=41)
    state.append_tokens("req-a", [1, 2, 3, 4, 5, 6, 7])
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    ring = req_state["compressed_decode_ring"]
    head = int(req_state["compressed_decode_head"])
    expected = torch.empty((state.max_ngram_order,), dtype=torch.int64, device="cpu")
    for i in range(state.max_ngram_order):
        expected[state.max_ngram_order - 1 - i] = ring[(head - 1 - i) % ring.shape[0]]

    req_state["compressed_token_history"] = []
    req_state["compressed_token_history_tensor"] = torch.empty(
        0, dtype=torch.int64, device="cpu"
    )
    req_state["compressed_token_history_tensor_valid"] = False
    snapshot = state._build_lookup_snapshot(req_state, new_token_count=1)
    assert snapshot.compressed_history.shape[0] == state.max_ngram_order
    assert torch.equal(snapshot.compressed_history, expected)


def test_ring_buffer_large_append_wrap_around():
    """Test decode ring handles large appends exceeding buffer size."""
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
    state.init_synthetic_compression_map(seed=2031)

    req_id = "req-large-append"
    state.ensure_request(req_id)

    ring_size = state.max_ngram_order

    large_tokens = torch.arange(100, 100 + ring_size * 2, dtype=torch.int64)
    state.append_tokens(req_id, large_tokens)

    req_state = state.get_request_state(req_id)
    ring = req_state["compressed_decode_ring"]

    # Ring stores compressed tokens, not raw tokens
    # Just verify ring size and that it has valid data
    assert ring.shape[0] == ring_size
    assert req_state["compressed_decode_len"] > 0

    more_tokens = torch.tensor([200, 201, 202], dtype=torch.int64)
    state.append_tokens(req_id, more_tokens)

    # Verify ring is updated after appending more tokens
    updated_len = req_state["compressed_decode_len"]
    assert updated_len > 0
    # Ring should have room for max_ngram_order tokens
    assert req_state["compressed_decode_ring"].shape[0] == ring_size


def test_lookup_embeddings_rebuilds_compressed_cache_when_stale():
    """@brief Validate that lookup embeddings rebuilds compressed cache when stale.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=16,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=23)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=29)
    layer = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    state.append_tokens("req-a", [10, 11, 12, 13, 14])
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    req_state["compressed_token_history"] = []

    out = layer._build_lookup_embeddings(req_state=req_state, new_token_count=3)
    assert out.shape == (3, 128)
    assert len(req_state["compressed_token_history"]) == len(req_state["token_history"])


def test_conv_one_token_with_state_matches_reference_cat_update():
    """@brief Validate that conv one token with state matches reference cat update.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=16,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    layer = EngramLayer(
        hidden_size=4,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    layer.init_synthetic_layer_parameters(seed=2026)
    assert layer.conv_weight is not None

    history_len = (layer.conv_kernel - 1) * layer.conv_dilation
    req_state = {"conv_history_by_layer": {}}
    ref_history = torch.zeros(
        history_len,
        layer.hidden_size,
        dtype=layer.conv_weight.dtype,
        device=layer.conv_weight.device,
    )
    req_state["conv_history_by_layer"][layer.layer_idx] = ref_history.clone()

    for step in range(5):
        token = torch.randn(
            (1, layer.hidden_size),
            dtype=layer.conv_weight.dtype,
            device=layer.conv_weight.device,
        )
        out = layer._conv_one_token_with_state(token, req_state=req_state)

        conv_w = layer.conv_weight[:, 0, :]
        ref_out = conv_w[:, layer.conv_kernel - 1] * token[0]
        for tap in range(layer.conv_kernel - 1):
            ref_out = ref_out + conv_w[:, tap] * ref_history[tap * layer.conv_dilation]
        assert torch.allclose(out[0], ref_out, rtol=0, atol=0)

        ref_history = torch.cat([ref_history[1:], token], dim=0)
        entry = req_state["conv_history_by_layer"][layer.layer_idx]
        assert isinstance(entry, dict)
        assert entry.get("initialized", False) is True
        history = _history_entry_as_logical_tensor(entry)
        assert torch.allclose(history, ref_history, rtol=0, atol=0)


def test_synthetic_host_tables_have_layer_specific_multipliers_and_uniform_mods():
    """@brief Validate that synthetic host tables have layer specific multipliers and uniform mods.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=256,
        vocab_size=1000,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=123)
    state.init_synthetic_layer_host_tables(layer_idx=15, seed=123)
    host_1 = state.host_tables_by_layer[1]
    host_15 = state.host_tables_by_layer[15]
    assert host_1.layer_multipliers.shape == (3,)
    assert host_15.layer_multipliers.shape == (3,)
    assert bool(torch.all((host_1.layer_multipliers & 1) == 1).item())
    assert bool(torch.all((host_15.layer_multipliers & 1) == 1).item())
    assert not torch.equal(host_1.layer_multipliers, host_15.layer_multipliers)
    assert host_1.head_vocab_sizes_by_order[2].tolist() == [64, 64, 64, 64]
    assert host_15.head_vocab_sizes_by_order[3].tolist() == [64, 64, 64, 64]


def test_lookup_embeddings_recomputes_without_shared_step_cache(monkeypatch):
    """@brief Validate that lookup embeddings recomputes without shared step cache.

    Args:
        monkeypatch (Any): Monkeypatch input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=11)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=13)
    layer = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    state.append_tokens("req-a", [101, 102, 7, 8, 9, 10, 11])
    req_state = state.get_request_state("req-a")
    assert req_state is not None

    calls = {"count": 0}
    original = _compute_deepseek_hash_ids_for_sequence

    def wrapped(*args, **kwargs):
        """@brief Wrap and count calls to the intercepted function under test.

        Args:
            *args (Any): Args input.
                Shape: N/A. Dtype: N/A unless tensor.
            **kwargs (Any): Kwargs input.
                Shape: N/A. Dtype: N/A unless tensor.

        Returns:
            Any: Computed result for this helper.
                Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
        """
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "vllm.model_executor.layers.engram_hash._compute_deepseek_hash_ids_for_sequence",
        wrapped,
    )
    layer._build_lookup_embeddings(req_state=req_state, new_token_count=5, step_uid=7)
    calls_after_first = calls["count"]
    layer._build_lookup_embeddings(req_state=req_state, new_token_count=5, step_uid=7)
    assert calls_after_first > 0
    assert calls["count"] > calls_after_first


def test_synthetic_compression_map_obeys_ratio_range():
    """@brief Validate that synthetic compression map obeys ratio range.

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
        vocab_size=1000,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=4,
    )
    state.init_synthetic_compression_map(seed=7)
    assert state.token_compression_map is not None
    assert int(state.token_compression_map.max().item()) < 800


def test_synthetic_compression_map_rejects_out_of_range_pad_token_id():
    """@brief Validate pad_token_id must already lie in compressed vocab range.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1000,
        compression_ratio=0.5,  # compressed_vocab=500
        conv_kernel=4,
        conv_dilation=3,
        pad_token_id=700,
        engram_layer_indices=(1, 15),
    )
    with pytest.raises(ValueError, match="pad_token_id"):
        state.init_synthetic_compression_map(seed=7)


def test_synthetic_host_tables_have_expected_shape():
    """@brief Validate that synthetic host tables have expected shape.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=256,
        vocab_size=1000,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=123)
    host = state.host_tables_by_layer[1]
    assert host.layer_multipliers.shape == (3,)
    assert host.head_vocab_sizes_by_order[2].shape == (4,)
    assert host.head_vocab_sizes_by_order[3].shape == (4,)
    table = host.tables[0]
    # num_tables = (3 - 1) * 4 = 8 -> d_table = 256 / 8 = 32
    assert table.shape == (64, 32)


def test_validate_engram_step_payload_accepts_valid_payload():
    """@brief Validate that validate engram step payload accepts valid payload.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    payload = {
        "step_uid": 1,
        "input_ids": torch.tensor([10, 11, 20, 21], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 2, 4], dtype=torch.int32),
        "request_ids": ["r0", "r1"],
    }
    validate_engram_step_payload(payload)


def test_validate_engram_step_payload_non_monotonic():
    """Test payload validation catches non-monotonic query_start_loc."""
    with pytest.raises(ValueError, match="query_start_loc.*must start at 0"):
        validate_engram_step_payload({
            "step_uid": 1,
            "input_ids": torch.tensor([10, 20, 30], dtype=torch.int64),
            "query_start_loc": torch.tensor([1, 3], dtype=torch.int32),
            "request_ids": ["r0"],
        })

    with pytest.raises(ValueError, match="non-decreasing"):
        validate_engram_step_payload({
            "step_uid": 1,
            "input_ids": torch.tensor([10, 20, 30], dtype=torch.int64),
            "query_start_loc": torch.tensor([0, 2, 1], dtype=torch.int32),
            "request_ids": ["r0", "r1"],
        })

    with pytest.raises(ValueError, match="`engram_step` shape mismatch"):
        validate_engram_step_payload({
            "step_uid": 1,
            "input_ids": torch.tensor([10, 20, 30], dtype=torch.int64),
            "query_start_loc": torch.tensor([0, 2], dtype=torch.int32),
            "request_ids": ["r0"],
        })


def test_slice_engram_step_payload_rebases_offsets():
    """@brief Validate that slice engram step payload rebases offsets.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    payload = {
        "step_uid": 7,
        "input_ids": torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
        "query_start_loc": torch.tensor([0, 3, 5], dtype=torch.int32),
        "request_ids": ["a", "b"],
    }
    sliced = slice_engram_step_payload(
        payload, request_slice=slice(0, 1), token_slice=slice(0, 3)
    )
    assert sliced["request_ids"] == ["a"]
    assert sliced["query_start_loc"].tolist() == [0, 3]
    assert sliced["input_ids"].tolist() == [10, 11, 12]


def test_build_engram_step_payload_returns_none_when_inputs_missing():
    """@brief Validate that build engram step payload returns none when inputs missing.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    assert build_engram_step_payload(1, None, None, None) is None


def test_lookup_embeddings_are_request_isolated():
    """@brief Validate that lookup embeddings are request isolated.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=17)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    layer = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    state.append_tokens("req-a", [1, 2, 3, 4, 5, 6])
    state.append_tokens("req-b", [700, 701, 702, 703, 704, 705])
    req_a = state.get_request_state("req-a")
    req_b = state.get_request_state("req-b")
    assert req_a is not None
    assert req_b is not None
    out_a = layer._build_lookup_embeddings(req_state=req_a, new_token_count=3)
    out_b = layer._build_lookup_embeddings(req_state=req_b, new_token_count=3)

    state_a_only = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state_a_only.init_synthetic_compression_map(seed=17)
    state_a_only.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    layer_a_only = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state_a_only,
    )
    state_a_only.append_tokens("req-a", [1, 2, 3, 4, 5, 6])
    req_a_only = state_a_only.get_request_state("req-a")
    assert req_a_only is not None
    out_a_only = layer_a_only._build_lookup_embeddings(req_state=req_a_only, new_token_count=3)
    assert torch.equal(out_a, out_a_only)

    state_b_only = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state_b_only.init_synthetic_compression_map(seed=17)
    state_b_only.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    layer_b_only = EngramLayer(
        hidden_size=16,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state_b_only,
    )
    state_b_only.append_tokens("req-b", [700, 701, 702, 703, 704, 705])
    req_b_only = state_b_only.get_request_state("req-b")
    assert req_b_only is not None
    out_b_only = layer_b_only._build_lookup_embeddings(req_state=req_b_only, new_token_count=3)
    assert torch.equal(out_b, out_b_only)


def test_decode_lookup_is_request_isolated():
    """@brief Validate that decode lookup is request isolated.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=17)
    state.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    state.append_tokens("req-a", [1, 2, 3, 4, 5, 6])
    state.append_tokens("req-b", [700, 701, 702, 703, 704, 705])
    req_a = state.get_request_state("req-a")
    req_b = state.get_request_state("req-b")
    assert req_a is not None
    assert req_b is not None
    out_a = state.build_lookup_for_request_layer_decode(req_a, layer_idx=1)
    out_b = state.build_lookup_for_request_layer_decode(req_b, layer_idx=1)

    state_a_only = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state_a_only.init_synthetic_compression_map(seed=17)
    state_a_only.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    state_a_only.append_tokens("req-a", [1, 2, 3, 4, 5, 6])
    req_a_only = state_a_only.get_request_state("req-a")
    assert req_a_only is not None
    out_a_only = state_a_only.build_lookup_for_request_layer_decode(
        req_a_only, layer_idx=1
    )
    assert torch.equal(out_a, out_a_only)

    state_b_only = EngramRuntimeState(
        memory_size=64,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=2048,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state_b_only.init_synthetic_compression_map(seed=17)
    state_b_only.init_synthetic_layer_host_tables(layer_idx=1, seed=23)
    state_b_only.append_tokens("req-b", [700, 701, 702, 703, 704, 705])
    req_b_only = state_b_only.get_request_state("req-b")
    assert req_b_only is not None
    out_b_only = state_b_only.build_lookup_for_request_layer_decode(
        req_b_only, layer_idx=1
    )
    assert torch.equal(out_b, out_b_only)


def test_append_tokens_for_step_uid_dedups_same_step():
    """@brief Validate that append tokens for step uid dedups same step.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=16,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    state.init_synthetic_compression_map(seed=31)

    state.append_tokens_for_step_uid(
        "req-a",
        step_uid=9,
        token_ids=torch.tensor([11, 12], dtype=torch.int64),
    )
    state.append_tokens_for_step_uid(
        "req-a",
        step_uid=9,
        token_ids=torch.tensor([11, 12], dtype=torch.int64),
    )
    req_state = state.get_request_state("req-a")
    assert req_state is not None
    assert req_state["token_history"] == [11, 12]


def test_single_token_conv_path_matches_generic_conv_without_history():
    """@brief Validate that single token conv path matches generic conv without history.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=32,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    layer = EngramLayer(
        hidden_size=8,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    layer.init_synthetic_layer_parameters(seed=123)
    assert layer.conv_weight is not None
    device = layer.conv_weight.device
    dtype = layer.conv_weight.dtype

    token = torch.randn(1, 8, device=device, dtype=dtype)
    req_state_fast = {"conv_history_by_layer": {}}
    req_state_ref = {"conv_history_by_layer": {}}
    out_fast = layer._conv_one_token_with_state(token, req_state_fast)
    out_ref = layer._conv_with_state(token, req_state_ref)
    assert torch.allclose(out_fast, out_ref, rtol=1e-2, atol=5e-3)
    fast_entry = req_state_fast["conv_history_by_layer"][layer.layer_idx]
    ref_entry = req_state_ref["conv_history_by_layer"][layer.layer_idx]
    fast_history = _history_entry_as_logical_tensor(fast_entry)
    ref_history = _history_entry_as_logical_tensor(ref_entry)
    assert torch.equal(
        fast_history,
        ref_history,
    )


def test_single_token_conv_path_matches_generic_conv_with_history():
    """@brief Validate that single token conv path matches generic conv with history.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    state = EngramRuntimeState(
        memory_size=32,
        max_ngram_order=3,
        heads=4,
        mem_dim=128,
        vocab_size=1024,
        compression_ratio=0.8,
        conv_kernel=4,
        conv_dilation=3,
    )
    layer = EngramLayer(
        hidden_size=8,
        layer_idx=1,
        max_ngram_order=3,
        engram_heads=4,
        engram_mem_dim=128,
        conv_kernel=4,
        conv_dilation=3,
        runtime_state=state,
    )
    layer.init_synthetic_layer_parameters(seed=456)
    assert layer.conv_weight is not None
    device = layer.conv_weight.device
    dtype = layer.conv_weight.dtype

    history_len = (layer.conv_kernel - 1) * layer.conv_dilation
    history = torch.randn(history_len, 8, device=device, dtype=dtype)
    token = torch.randn(1, 8, device=device, dtype=dtype)
    req_state_fast = {"conv_history_by_layer": {layer.layer_idx: history.clone()}}
    req_state_ref = {"conv_history_by_layer": {layer.layer_idx: history.clone()}}
    out_fast = layer._conv_one_token_with_state(token, req_state_fast)
    out_ref = layer._conv_with_state(token, req_state_ref)
    assert torch.allclose(out_fast, out_ref, rtol=1e-2, atol=5e-3)
    fast_entry = req_state_fast["conv_history_by_layer"][layer.layer_idx]
    ref_entry = req_state_ref["conv_history_by_layer"][layer.layer_idx]
    fast_history = _history_entry_as_logical_tensor(fast_entry)
    ref_history = _history_entry_as_logical_tensor(ref_entry)
    assert torch.equal(
        fast_history,
        ref_history,
    )

