# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Engram hash and table lookup primitives for sequence and decode paths.

from __future__ import annotations

from dataclasses import dataclass

import torch

def _build_table_layout(
    max_ngram_order: int, heads: int, mem_dim: int
) -> list[tuple[int, int, int, int]]:
    """Build contiguous table slices for every (order, head) lookup table.

    Args:
        max_ngram_order: Maximum n-gram order used by Engram hashing.

        heads: Number of lookup heads per n-gram order.

        mem_dim: Total lookup embedding width.

    Returns:
        list[tuple[int, int, int, int]]: Ordered layout entries, one per lookup
            table, where each tuple is
            ``(ngram_order, head_index, slice_start, slice_end)``.
            The interval ``[slice_start, slice_end)`` selects the contiguous
            column range inside the full Engram embedding vector assigned to
            that specific ``(order, head)`` table.
            With the divisibility constraint, each table receives equal width
            ``mem_dim // num_tables``.
    """
    table_keys = [
        (order, head)
        for order in range(2, max_ngram_order + 1)
        for head in range(heads)
    ]
    num_tables = len(table_keys)
    if num_tables <= 0:
        raise ValueError("Engram requires `engram_max_ngram_order >= 2`.")
    if mem_dim % num_tables != 0:
        raise ValueError(
            "`engram_mem_dim` must be divisible by number of tables "
            "((engram_max_ngram_order - 1) * engram_heads)."
        )

    base = mem_dim // num_tables
    layout: list[tuple[int, int, int, int]] = []
    offset = 0
    for order, head in table_keys:
        width = base
        next_offset = offset + width
        layout.append((order, head, offset, next_offset))
        offset = next_offset
    return layout

@dataclass
class EngramLayerHostTables:
    layer_multipliers: torch.Tensor
    head_vocab_sizes_by_order: dict[int, torch.Tensor]
    tables: torch.Tensor


def _gather_lookup_embeddings(
    row_ids: torch.Tensor, host_tables: torch.Tensor, mem_dim: int
) -> torch.Tensor:
    """Gather per-table rows from host tables and flatten into mem_dim.

    Args:
        row_ids: Row IDs per output token and table.

        host_tables: Host lookup tables.

        mem_dim: Flattened embedding width.

    Returns:
        torch.Tensor: Gathered lookup embeddings.

    """
    row_ids_i64 = row_ids.detach().to(dtype=torch.long, device="cpu")
    num_rows, num_tables = row_ids_i64.shape
    table_ids = torch.arange(num_tables, dtype=torch.long, device="cpu").unsqueeze(0)
    table_ids = table_ids.expand(num_rows, num_tables)
    gathered = host_tables[table_ids, row_ids_i64]
    return gathered.reshape(num_rows, mem_dim)


@dataclass(frozen=True)
class EngramLookupSnapshot:
    compressed_history: torch.Tensor
    total_tokens: int
    new_token_count: int
    start_token: int


def _build_lookup_embeddings_from_snapshot(
    snapshot: EngramLookupSnapshot,
    host: EngramLayerHostTables | None,
    table_layout: list[tuple[int, int, int, int]],
    *,
    mem_dim: int,
    max_ngram_order: int,
    pad_id: int,
) -> torch.Tensor:
    """Build prefill lookup embeddings from one request snapshot.

    Args:
        snapshot: Immutable lookup snapshot built from request state.

        host: Host-side lookup table bundle for one layer.

        table_layout: Table slice layout for (order, head) lookup tables.

        mem_dim: Total lookup embedding width.

        max_ngram_order: Maximum n-gram order used by Engram hashing.

        pad_id: Pad ID used for unavailable history positions.

    Returns:
        torch.Tensor: Computed result for this helper.

    """
    embeddings = torch.zeros(
        (snapshot.new_token_count, mem_dim), dtype=torch.bfloat16, device="cpu"
    )
    if (
        snapshot.new_token_count <= 0
        or host is None
        or snapshot.total_tokens == 0
        or snapshot.start_token >= snapshot.total_tokens
    ):
        return embeddings

    compressed_history = snapshot.compressed_history
    hash_ids = _compute_deepseek_hash_ids_for_sequence(
        compressed_tokens=compressed_history,
        layer_multipliers=host.layer_multipliers,
        head_vocab_sizes_by_order=host.head_vocab_sizes_by_order,
        max_ngram_order=max_ngram_order,
        pad_id=pad_id,
    )
    output_hashes = hash_ids[snapshot.start_token : snapshot.total_tokens]
    if output_hashes.numel() == 0:
        return embeddings
    if output_hashes.shape[1] != len(table_layout):
        raise ValueError(
            "Hash table count mismatch between output hashes and table layout."
        )
    return _gather_lookup_embeddings(output_hashes, host.tables, mem_dim)


def _build_decode_lookup_embedding_from_snapshot(
    snapshot: EngramLookupSnapshot,
    host: EngramLayerHostTables | None,
    table_layout: list[tuple[int, int, int, int]],
    *,
    mem_dim: int,
    max_ngram_order: int,
    pad_id: int,
) -> torch.Tensor:
    """Build one-token decode lookup embeddings from one snapshot.

    Args:
        snapshot: Immutable lookup snapshot built from request state.

        host: Host-side lookup table bundle for one layer.

        table_layout: Table slice layout for (order, head) lookup tables.

        mem_dim: Total lookup embedding width.

        max_ngram_order: Maximum n-gram order used by Engram hashing.

        pad_id: Pad ID used for unavailable history positions.

    Returns:
        torch.Tensor: Computed result for this helper.

    """
    if snapshot.new_token_count != 1:
        return _build_lookup_embeddings_from_snapshot(
            snapshot=snapshot,
            host=host,
            table_layout=table_layout,
            mem_dim=mem_dim,
            max_ngram_order=max_ngram_order,
            pad_id=pad_id,
        )
    embeddings = torch.zeros((1, mem_dim), dtype=torch.bfloat16, device="cpu")
    if (
        host is None
        or snapshot.total_tokens == 0
        or snapshot.start_token >= snapshot.total_tokens
    ):
        return embeddings
    row_ids = _compute_deepseek_hash_ids_last_token(
        compressed_tokens=snapshot.compressed_history,
        layer_multipliers=host.layer_multipliers,
        head_vocab_sizes_by_order=host.head_vocab_sizes_by_order,
        max_ngram_order=max_ngram_order,
        pad_id=pad_id,
    )
    if row_ids.numel() == 0:
        return embeddings
    row_ids_2d = row_ids.reshape(1, -1)
    if row_ids_2d.shape[1] != len(table_layout):
        raise ValueError("Hash table count mismatch between decode hashes and layout.")
    return _gather_lookup_embeddings(row_ids_2d, host.tables, mem_dim)


def _build_decode_lookup_embeddings_from_histories(
    *,
    compressed_histories: list[torch.Tensor],
    host: EngramLayerHostTables | None,
    table_layout: list[tuple[int, int, int, int]],
    mem_dim: int,
    max_ngram_order: int,
    pad_id: int,
    tokens_by_offset_workspace: torch.Tensor | None = None,
    hash_out_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build batched decode lookup embeddings from request histories.

    Args:
        compressed_histories: Compressed token history tensors for request batch.

        host: Host-side lookup table bundle for one layer.

        table_layout: Table slice layout for (order, head) lookup tables.

        mem_dim: Total lookup embedding width.

        max_ngram_order: Maximum n-gram order used by Engram hashing.

        pad_id: Pad ID used for unavailable history positions.

        tokens_by_offset_workspace: Optional reusable offset-token workspace buffer.

        hash_out_workspace: Optional reusable hash-output workspace buffer.

    Returns:
        torch.Tensor: Computed result for this helper.

    """
    batch = len(compressed_histories)
    embeddings = torch.zeros((batch, mem_dim), dtype=torch.bfloat16, device="cpu")
    if batch == 0 or host is None:
        return embeddings
    hash_ids = compute_deepseek_hash_ids_last_token_batch(
        compressed_histories=compressed_histories,
        layer_multipliers=host.layer_multipliers,
        head_vocab_sizes_by_order=host.head_vocab_sizes_by_order,
        max_ngram_order=max_ngram_order,
        pad_id=pad_id,
        tokens_by_offset_workspace=tokens_by_offset_workspace,
        hash_out_workspace=hash_out_workspace,
    )
    if hash_ids.numel() == 0:
        return embeddings
    if hash_ids.shape[1] != len(table_layout):
        raise ValueError("Hash table count mismatch between batched hashes and layout.")
    return _gather_lookup_embeddings(hash_ids, host.tables, mem_dim)

def _build_shifted_tokens(
    compressed_tokens: torch.Tensor,
    max_ngram_order: int,
    pad_id: int,
) -> list[torch.Tensor]:
    """Build shifted token views with left padding for n-gram hashing.

    Args:
        compressed_tokens: Compressed token IDs for a single request.

        max_ngram_order: Maximum n-gram order used by Engram hashing.

        pad_id: Pad ID used for unavailable history positions.

    Returns:
        list[torch.Tensor]: Computed result for this helper.

    """
    tokens = compressed_tokens.detach().to(dtype=torch.int64, device="cpu")
    if tokens.ndim != 1:
        raise ValueError("`compressed_tokens` must be a 1D tensor.")
    t = int(tokens.shape[0])
    shifted: list[torch.Tensor] = [tokens]
    for k in range(1, max_ngram_order):
        padded = torch.full((k + t,), pad_id, dtype=torch.int64, device="cpu")
        padded[k:] = tokens
        shifted.append(padded[:t])
    return shifted


def _compute_deepseek_hash_ids_for_sequence(
    *,
    compressed_tokens: torch.Tensor,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes_by_order: dict[int, torch.Tensor],
    max_ngram_order: int,
    pad_id: int,
) -> torch.Tensor:
    """Compute DeepSeek-style n-gram hash ids for every token position.

    Args:
        compressed_tokens: Compressed token ids for one request.

        layer_multipliers: Per-order hash multipliers.

        head_vocab_sizes_by_order: Per-order modulus
            values for each head.

        max_ngram_order: Maximum n-gram order to hash.

        pad_id: Pad token id used for shifted prefix positions.

    Returns:
        torch.Tensor: Hash ids for every token and table column.

    """
    shifted = _build_shifted_tokens(
        compressed_tokens=compressed_tokens,
        max_ngram_order=max_ngram_order,
        pad_id=pad_id,
    )
    t = compressed_tokens.shape[0]
    all_hashes: list[torch.Tensor] = []
    multipliers = layer_multipliers.detach().to(dtype=torch.int64, device="cpu")
    for order in range(2, max_ngram_order + 1):
        tokens = shifted[:order]
        mix = tokens[0] * multipliers[0]
        for k in range(1, order):
            mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])
        head_vocab_sizes = head_vocab_sizes_by_order[order]
        for head_idx in range(head_vocab_sizes.shape[0]):
            mod = int(head_vocab_sizes[head_idx].item())
            all_hashes.append(torch.remainder(mix, mod).to(dtype=torch.int64))
    if not all_hashes:
        return torch.empty((t, 0), dtype=torch.int64, device="cpu")
    return torch.stack(all_hashes, dim=1)


def _compute_deepseek_hash_ids_last_token(
    *,
    compressed_tokens: torch.Tensor,
    layer_multipliers: torch.Tensor,
    head_vocab_sizes_by_order: dict[int, torch.Tensor],
    max_ngram_order: int,
    pad_id: int,
) -> torch.Tensor:
    """Compute DeepSeek-style hash ids for the final token only.

    Args:
        compressed_tokens: Compressed token ids for one request.

        layer_multipliers: Per-order hash multipliers.

        head_vocab_sizes_by_order: Per-order modulus
            values for each head.

        max_ngram_order: Maximum n-gram order to hash.

        pad_id: Pad token id used for missing history offsets.

    Returns:
        torch.Tensor: Hash ids for the last token over all tables.

    """
    tokens = compressed_tokens.detach().to(dtype=torch.int64, device="cpu")
    if tokens.ndim != 1:
        raise ValueError("`compressed_tokens` must be a 1D tensor.")
    if tokens.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device="cpu")

    last_idx = int(tokens.shape[0]) - 1
    multipliers = layer_multipliers.detach().to(dtype=torch.int64, device="cpu")
    pad_scalar = torch.tensor(pad_id, dtype=torch.int64, device="cpu")
    all_hashes: list[torch.Tensor] = []
    for order in range(2, max_ngram_order + 1):
        mix = tokens[last_idx] * multipliers[0]
        for k in range(1, order):
            idx = last_idx - k
            prev = tokens[idx] if idx >= 0 else pad_scalar
            mix = torch.bitwise_xor(mix, prev * multipliers[k])
        head_vocab_sizes = head_vocab_sizes_by_order[order]
        for head_idx in range(head_vocab_sizes.shape[0]):
            mod = int(head_vocab_sizes[head_idx].item())
            all_hashes.append(torch.remainder(mix, mod).to(dtype=torch.int64))
    if not all_hashes:
        return torch.empty(0, dtype=torch.int64, device="cpu")
    return torch.stack(all_hashes, dim=0)


def compute_deepseek_hash_ids_last_token_batch(
    *,
    compressed_histories: list[torch.Tensor],
    layer_multipliers: torch.Tensor,
    head_vocab_sizes_by_order: dict[int, torch.Tensor],
    max_ngram_order: int,
    pad_id: int,
    tokens_by_offset_workspace: torch.Tensor | None = None,
    hash_out_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute last-token DeepSeek hashes for a batch of histories.

    Args:
        compressed_histories: Per-request compressed token
            histories.

        layer_multipliers: Per-order hash multipliers.

        head_vocab_sizes_by_order: Per-order modulus
            values for each head.

        max_ngram_order: Maximum n-gram order to hash.

        pad_id: Pad token id used for missing history offsets.

        tokens_by_offset_workspace: Optional workspace
            reused to hold offset tokens.

        hash_out_workspace: Optional workspace reused to
            hold hash outputs.

    Returns:
        torch.Tensor: Batched hash ids for the final token of each request.

    """
    batch = len(compressed_histories)
    if batch == 0:
        return torch.empty((0, 0), dtype=torch.int64, device="cpu")

    multipliers = layer_multipliers.detach().to(dtype=torch.int64, device="cpu")
    pad_scalar = int(pad_id)
    num_tables = sum(
        int(head_vocab_sizes_by_order[order].shape[0])
        for order in range(2, max_ngram_order + 1)
    )
    if (
        tokens_by_offset_workspace is not None
        and tokens_by_offset_workspace.dtype == torch.int64
        and tokens_by_offset_workspace.device.type == "cpu"
        and tokens_by_offset_workspace.shape[0] >= max_ngram_order
        and tokens_by_offset_workspace.shape[1] >= batch
    ):
        tokens_by_offset = tokens_by_offset_workspace[:max_ngram_order, :batch]
    else:
        tokens_by_offset = torch.empty(
            (max_ngram_order, batch), dtype=torch.int64, device="cpu"
        )
    if (
        hash_out_workspace is not None
        and hash_out_workspace.dtype == torch.int64
        and hash_out_workspace.device.type == "cpu"
        and hash_out_workspace.shape[0] >= batch
        and hash_out_workspace.shape[1] >= num_tables
    ):
        hash_out = hash_out_workspace[:batch, :num_tables]
    else:
        hash_out = torch.empty((batch, num_tables), dtype=torch.int64, device="cpu")
    for req_idx, history in enumerate(compressed_histories):
        tokens = history.detach().to(dtype=torch.int64, device="cpu")
        last_idx = int(tokens.numel()) - 1
        for k in range(max_ngram_order):
            idx = last_idx - k
            if idx >= 0:
                tokens_by_offset[k, req_idx] = tokens[idx]
            else:
                tokens_by_offset[k, req_idx] = pad_scalar

    col_idx = 0
    for order in range(2, max_ngram_order + 1):
        mix = tokens_by_offset[0] * multipliers[0]
        for k in range(1, order):
            mix = torch.bitwise_xor(mix, tokens_by_offset[k] * multipliers[k])
        head_vocab_sizes = head_vocab_sizes_by_order[order]
        for head_idx in range(head_vocab_sizes.shape[0]):
            mod = int(head_vocab_sizes[head_idx].item())
            hash_out[:, col_idx].copy_(torch.remainder(mix, mod).to(dtype=torch.int64))
            col_idx += 1
    return hash_out
