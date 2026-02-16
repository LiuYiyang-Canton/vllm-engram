# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Per-request Engram runtime state, lookup building, and async prefetch management.

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch
import torch.profiler

from vllm.logger import init_logger
from vllm.model_executor.layers.engram_hash import (
    EngramLayerHostTables,
    EngramLookupSnapshot,
    _build_decode_lookup_embedding_from_snapshot,
    _build_decode_lookup_embeddings_from_histories,
    _build_lookup_embeddings_from_snapshot,
    _build_table_layout,
)

logger = init_logger(__name__)

@dataclass
class EngramRuntimeState:
    """Per-model runtime state/cache manager for Engram inference.

    This plays a role similar to a cache manager for Engram features:
    it tracks per-request histories, lookup prefetch state, and request
    lifecycle cleanup so Engram layers can reuse precomputed data safely.

    Comparison to KV cache:
    - KV cache stores attention K/V tensors used by transformer attention.
    - EngramRuntimeState stores Engram-specific CPU-side lookup/compression
      state and async prefetch artifacts used by Engram layers.
    """

    memory_size: int
    max_ngram_order: int
    heads: int
    mem_dim: int
    vocab_size: int
    compression_ratio: float
    conv_kernel: int
    conv_dilation: int
    pad_token_id: int = 0
    async_workers: int = 1
    engram_layer_indices: tuple[int, ...] = ()
    request_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    host_tables_by_layer: dict[int, EngramLayerHostTables] = field(default_factory=dict)
    token_compression_map: torch.Tensor | None = None
    compressed_pad_id: int = 0

    # Per-request prefill/mixed-step async lookup state.
    # Keyed by (step_uid, req_id): futures are in-flight jobs; cache stores
    # completed per-layer lookup tensors for later consume.
    lookup_futures: dict[tuple[int, str], Future[dict[int, torch.Tensor]]] = field(
        default_factory=dict
    )
    lookup_prefetch_cache: dict[tuple[int, str], dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    lookup_prefetch_ready_events: dict[tuple[int, str], torch.cuda.Event | None] = field(
        default_factory=dict
    )

    # Decode-step batched async lookup state.
    # Keyed by step_uid: futures/cache hold batched per-layer tensors.
    # req_ids/req_pos preserve row ordering so active subsets can be sliced safely.
    decode_lookup_batch_futures: dict[int, Future[dict[int, torch.Tensor]]] = field(
        default_factory=dict
    )
    decode_lookup_batch_cache: dict[int, dict[int, torch.Tensor]] = field(
        default_factory=dict
    )
    decode_lookup_batch_ready_events: dict[int, torch.cuda.Event | None] = field(
        default_factory=dict
    )
    decode_lookup_batch_req_ids: dict[int, tuple[str, ...]] = field(
        default_factory=dict
    )
    decode_lookup_batch_req_pos: dict[int, dict[str, int]] = field(
        default_factory=dict
    )
    decode_lookup_batch_row_idx_cache: dict[
        int, dict[tuple[str, ...], torch.Tensor]
    ] = field(default_factory=dict)

    table_layout: list[tuple[int, int, int, int]] = field(default_factory=list)
    _async_executor: ThreadPoolExecutor | None = field(default=None, init=False)
    _async_lock: Lock = field(default_factory=Lock, init=False)
    _async_warned_failure_keys: set[tuple[int, str]] = field(
        default_factory=set, init=False
    )
    _async_warned_decode_steps: set[int] = field(default_factory=set, init=False)
    _decode_tokens_by_offset_workspace: torch.Tensor | None = field(
        default=None, init=False
    )
    _decode_hash_out_workspace: torch.Tensor | None = field(default=None, init=False)
    _decode_h2d_streams: dict[str, torch.cuda.Stream] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Initialize derived runtime tables from constructor configuration.

        Args:
            self (Any): Module or runtime instance.

        Returns:
            None: Function returns no value.
        """
        if self.memory_size < self.max_ngram_order:
            raise ValueError(
                "`engram_memory_size` must be >= `engram_max_ngram_order`, "
                f"got memory_size={self.memory_size}, "
                f"max_ngram_order={self.max_ngram_order}."
            )
        self.table_layout = _build_table_layout(
            max_ngram_order=self.max_ngram_order,
            heads=self.heads,
            mem_dim=self.mem_dim,
        )

    def _ensure_decode_cpu_workspace(
        self, *, batch: int, num_tables: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate or reuse CPU workspaces used by decode hash computation.

        Args:
            self (Any): Module or runtime instance.
            batch (int): Active decode batch size.
            num_tables (int): Number of lookup tables in current layout.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Computed result for this helper.
        """
        tokens_ws = self._decode_tokens_by_offset_workspace
        if (
            tokens_ws is None
            or tokens_ws.device.type != "cpu"
            or tokens_ws.dtype != torch.int64
            or tokens_ws.shape[0] != self.max_ngram_order
            or tokens_ws.shape[1] < batch
        ):
            tokens_ws = torch.empty(
                (self.max_ngram_order, batch), dtype=torch.int64, device="cpu"
            )
            self._decode_tokens_by_offset_workspace = tokens_ws

        hash_ws = self._decode_hash_out_workspace
        if (
            hash_ws is None
            or hash_ws.device.type != "cpu"
            or hash_ws.dtype != torch.int64
            or hash_ws.shape[1] != num_tables
            or hash_ws.shape[0] < batch
        ):
            hash_ws = torch.empty((batch, num_tables), dtype=torch.int64, device="cpu")
            self._decode_hash_out_workspace = hash_ws

        return tokens_ws[:, :batch], hash_ws[:batch, :]

    def _pin_cpu_lookup_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Best-effort pinning for CPU lookup tensors.

        Args:
            self (Any): Module or runtime instance.
            tensor (torch.Tensor): Lookup tensor expected to reside on CPU.

        Returns:
            torch.Tensor: Pinned CPU tensor when pinning succeeds;
                otherwise the original CPU tensor.
        """
        if tensor.device.type != "cpu":
            raise RuntimeError(
                "Engram lookup tensor must be on CPU before pinning. "
                f"Got device={tensor.device}."
            )
        if tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except Exception:
            return tensor

    def _ensure_decode_ring(self, req_state: dict[str, Any]) -> torch.Tensor:
        """Allocate or reuse the decode n-gram tail ring buffer.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        ring = req_state.get("compressed_decode_ring")
        # Decode hashing only needs the latest n-gram window.
        ring_size = max(1, int(self.max_ngram_order))
        if (
            not isinstance(ring, torch.Tensor)
            or ring.device.type != "cpu"
            or ring.dtype != torch.int64
            or ring.ndim != 1
            or ring.shape[0] != ring_size
        ):
            ring = torch.zeros((ring_size,), dtype=torch.int64, device="cpu")
            req_state["compressed_decode_ring"] = ring
            req_state["compressed_decode_len"] = 0
            req_state["compressed_decode_head"] = 0
        return ring

    def ensure_request(self, req_id: str) -> dict[str, Any]:
        """Create or return request-local Engram runtime state.

        Args:
            self (Any): Module or runtime instance.
            req_id (str): Single request identifier.

        Returns:
            dict[str, Any]: Computed result for this helper.
        """
        if req_id not in self.request_states:
            self.request_states[req_id] = {
                "token_history": [],
                "compressed_token_history": [],
                "compressed_token_history_tensor": torch.empty(
                    0, dtype=torch.int64, device="cpu"
                ),
                "compressed_token_history_tensor_valid": True,
                "compressed_decode_ring": torch.empty(0, dtype=torch.int64, device="cpu"),
                "compressed_decode_len": 0,
                "compressed_decode_head": 0,
                "last_step_uid_by_layer": {},
                "last_appended_step_uid": None,
                "conv_history_by_layer": {},
            }
        return self.request_states[req_id]

    def append_tokens(self, req_id: str, token_ids: list[int] | torch.Tensor) -> None:
        """Append new token IDs and update compressed caches/ring buffers.

        Args:
            self (Any): Module or runtime instance.
            req_id (str): Single request identifier.
            token_ids (list[int] | torch.Tensor): New token IDs to append to request history.

        Returns:
            None: Function returns no value.
        """
        req_state = self.ensure_request(req_id)
        if isinstance(token_ids, torch.Tensor):
            token_ids_cpu = token_ids.detach().to(device="cpu", dtype=torch.int64)
            token_list = token_ids_cpu.tolist()
        else:
            token_list = list(token_ids)
            token_ids_cpu = torch.as_tensor(token_list, dtype=torch.int64, device="cpu")

        if len(token_list) == 0:
            return

        req_state["token_history"].extend(token_list)

        compression = self.token_compression_map
        if compression is not None:
            clamped = torch.remainder(token_ids_cpu, compression.shape[0])
            compressed_appended = compression[clamped].to(dtype=torch.int64, device="cpu")
            req_state["compressed_token_history"].extend(compressed_appended.tolist())
            req_state["compressed_token_history_tensor_valid"] = False

            ring = self._ensure_decode_ring(req_state)
            ring_size = int(ring.shape[0])
            append_count = int(compressed_appended.numel())
            if append_count > 0:
                if append_count >= ring_size:
                    ring.copy_(compressed_appended[-ring_size:])
                    req_state["compressed_decode_head"] = 0
                    req_state["compressed_decode_len"] = ring_size
                else:
                    head = int(req_state.get("compressed_decode_head", 0)) % ring_size
                    existing_len = int(req_state.get("compressed_decode_len", 0))
                    room_to_end = ring_size - head
                    if append_count <= room_to_end:
                        ring[head : head + append_count].copy_(compressed_appended)
                    else:
                        ring[head:].copy_(compressed_appended[:room_to_end])
                        ring[: append_count - room_to_end].copy_(
                            compressed_appended[room_to_end:]
                        )
                    req_state["compressed_decode_head"] = (head + append_count) % ring_size
                    req_state["compressed_decode_len"] = min(
                        ring_size, existing_len + append_count
                    )
        else:
            # Keep cache invalidated until compression map is available.
            req_state["compressed_token_history"] = []
            req_state["compressed_token_history_tensor"] = torch.empty(
                0, dtype=torch.int64, device="cpu"
            )
            req_state["compressed_token_history_tensor_valid"] = True
            req_state["compressed_decode_ring"] = torch.empty(
                0, dtype=torch.int64, device="cpu"
            )
            req_state["compressed_decode_len"] = 0
            req_state["compressed_decode_head"] = 0

        if len(req_state["token_history"]) > self.memory_size:
            # Prefill/mixed-token lookups rely on memory_size-bounded full history.
            req_state["token_history"] = req_state["token_history"][-self.memory_size :]
        if len(req_state["compressed_token_history"]) > self.memory_size:
            req_state["compressed_token_history"] = req_state["compressed_token_history"][
                -self.memory_size :
            ]
        cached_tensor = req_state.get("compressed_token_history_tensor")
        if isinstance(cached_tensor, torch.Tensor) and cached_tensor.numel() > self.memory_size:
            req_state["compressed_token_history_tensor"] = cached_tensor[-self.memory_size :]
            req_state["compressed_token_history_tensor_valid"] = False

    def append_tokens_for_step_uid(
        self, req_id: str, step_uid: int, token_ids: list[int] | torch.Tensor
    ) -> None:
        """Append tokens at most once per request per step UID.

        Args:
            self (Any): Module or runtime instance.
            req_id (str): Single request identifier.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.
            token_ids (list[int] | torch.Tensor): New token IDs to append to request history.

        Returns:
            None: Function returns no value.
        """
        req_state = self.ensure_request(req_id)
        if req_state.get("last_appended_step_uid") == step_uid:
            return
        self.append_tokens(req_id, token_ids)
        req_state["last_appended_step_uid"] = step_uid

    def get_request_state(self, req_id: str) -> dict[str, Any] | None:
        """Fetch request-local runtime state if present.

        Args:
            self (Any): Module or runtime instance.
            req_id (str): Single request identifier.

        Returns:
            dict[str, Any] | None: Computed result for this helper.
        """
        return self.request_states.get(req_id)

    def cleanup_requests(self, finished_request_ids: list[str]) -> None:
        """Remove completed requests and associated async lookup artifacts.

        Args:
            self (Any): Module or runtime instance.
            finished_request_ids (list[str]): Completed request IDs to cleanup.

        Returns:
            None: Function returns no value.
        """
        req_id_set = set(finished_request_ids)
        for req_id in finished_request_ids:
            self.request_states.pop(req_id, None)
        with self._async_lock:
            for key, future in list(self.lookup_futures.items()):
                if key[1] in req_id_set:
                    if not future.done():
                        future.cancel()
                    self.lookup_futures.pop(key, None)
                    self.lookup_prefetch_ready_events.pop(key, None)
            for key in list(self.lookup_prefetch_cache.keys()):
                if key[1] in req_id_set:
                    self.lookup_prefetch_cache.pop(key, None)
                    self.lookup_prefetch_ready_events.pop(key, None)
            if self.decode_lookup_batch_req_ids:
                for step_uid, req_ids in list(self.decode_lookup_batch_req_ids.items()):
                    if not req_id_set.isdisjoint(req_ids):
                        self._drop_decode_lookup_batch_step_locked(step_uid)
            self._async_warned_failure_keys = {
                key for key in self._async_warned_failure_keys if key[1] not in req_id_set
            }
            self._async_warned_decode_steps = {
                step_uid
                for step_uid in self._async_warned_decode_steps
                if step_uid in self.decode_lookup_batch_req_ids
            }

    def init_async_executor(self, max_workers: int | None = None) -> None:
        """Initialize the async lookup thread pool executor.

        Args:
            self (Any): Module or runtime instance.
            max_workers (int | None): Requested async worker count override.

        Returns:
            None: Function returns no value.
        """
        workers = int(self.async_workers if max_workers is None else max_workers)
        if workers <= 0:
            raise ValueError(
                "`engram_async_workers` must be a positive integer, "
                f"got: {workers!r}"
            )
        with self._async_lock:
            if self._async_executor is not None:
                self._async_executor.shutdown(wait=False, cancel_futures=True)
            self._async_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="engram-cpu",
            )
            self.async_workers = workers

    def shutdown_async_executor(self) -> None:
        """Stop async executor and clear async caches/futures.

        Args:
            self (Any): Module or runtime instance.

        Returns:
            None: Function returns no value.
        """
        with self._async_lock:
            if self._async_executor is not None:
                self._async_executor.shutdown(wait=False, cancel_futures=True)
                self._async_executor = None
            self.lookup_futures.clear()
            self.lookup_prefetch_cache.clear()
            self.lookup_prefetch_ready_events.clear()
            self.decode_lookup_batch_futures.clear()
            self.decode_lookup_batch_cache.clear()
            self.decode_lookup_batch_ready_events.clear()
            self.decode_lookup_batch_req_ids.clear()
            self.decode_lookup_batch_req_pos.clear()
            self.decode_lookup_batch_row_idx_cache.clear()
            self._decode_h2d_streams.clear()
            self._async_warned_failure_keys.clear()
            self._async_warned_decode_steps.clear()

    def _drop_decode_lookup_batch_step_locked(self, step_uid: int) -> None:
        """Drop decode batch async artifacts for one step (lock must be held)."""
        future = self.decode_lookup_batch_futures.pop(step_uid, None)
        if future is not None and not future.done():
            future.cancel()
        self.decode_lookup_batch_cache.pop(step_uid, None)
        self.decode_lookup_batch_ready_events.pop(step_uid, None)
        self.decode_lookup_batch_req_ids.pop(step_uid, None)
        self.decode_lookup_batch_req_pos.pop(step_uid, None)
        self.decode_lookup_batch_row_idx_cache.pop(step_uid, None)

    def _get_decode_h2d_stream(self, device: torch.device) -> torch.cuda.Stream:
        """Get or create a decode lookup H2D copy stream for ``device``."""
        key = str(device)
        stream = self._decode_h2d_streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self._decode_h2d_streams[key] = stream
        return stream

    def _stage_lookup_payload_to_device(
        self,
        *,
        payload_by_layer: dict[int, torch.Tensor],
        target_device: torch.device | None,
        target_dtype: torch.dtype | None,
    ) -> tuple[dict[int, torch.Tensor], torch.cuda.Event | None]:
        """Stage lookup payload to CUDA during async prefetch."""
        if target_device is None or target_device.type != "cuda":
            return payload_by_layer, None
        if not torch.cuda.is_available():
            return payload_by_layer, None

        staged_payload: dict[int, torch.Tensor] = {}
        dtype = torch.bfloat16 if target_dtype is None else target_dtype
        stream = self._get_decode_h2d_stream(target_device)
        with torch.cuda.device(target_device), torch.cuda.stream(stream):
            for layer_idx, lookup_cpu in payload_by_layer.items():
                src = lookup_cpu
                if src.device.type != "cpu":
                    src = src.to(device="cpu")
                if not src.is_pinned():
                    try:
                        src = src.pin_memory()
                    except Exception:
                        pass
                dst = torch.empty(src.shape, device=target_device, dtype=dtype)
                dst.copy_(src, non_blocking=True)
                staged_payload[int(layer_idx)] = dst
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
        return staged_payload, ready_event

    def _drop_stale_lookup_futures(self, current_step_uid: int) -> None:
        """Drop stale async lookup futures and caches from older steps.

        Args:
            self (Any): Module or runtime instance.
            current_step_uid (int): Current step UID used for stale-cache cleanup.

        Returns:
            None: Function returns no value.
        """
        with self._async_lock:
            for key, future in list(self.lookup_futures.items()):
                step_uid, _ = key
                if step_uid >= current_step_uid - 1:
                    continue
                if not future.done():
                    future.cancel()
                self.lookup_futures.pop(key, None)
                self.lookup_prefetch_ready_events.pop(key, None)
            for key in list(self.lookup_prefetch_cache.keys()):
                step_uid, _ = key
                if step_uid < current_step_uid - 1:
                    self.lookup_prefetch_cache.pop(key, None)
                    self.lookup_prefetch_ready_events.pop(key, None)
            for step_uid, future in list(self.decode_lookup_batch_futures.items()):
                if step_uid >= current_step_uid - 1:
                    continue
                self._drop_decode_lookup_batch_step_locked(step_uid)
            for step_uid in list(self.decode_lookup_batch_cache.keys()):
                if step_uid < current_step_uid - 1:
                    self._drop_decode_lookup_batch_step_locked(step_uid)
            for step_uid in list(self.decode_lookup_batch_req_ids.keys()):
                if step_uid < current_step_uid - 1:
                    self._drop_decode_lookup_batch_step_locked(step_uid)
            self._async_warned_failure_keys = {
                key
                for key in self._async_warned_failure_keys
                if key[0] >= current_step_uid - 1
            }
            self._async_warned_decode_steps = {
                step_uid
                for step_uid in self._async_warned_decode_steps
                if step_uid >= current_step_uid - 1
            }

    def _ensure_compressed_history(
        self, req_state: dict[str, Any]
    ) -> torch.Tensor:
        """Return a consistent compressed-history tensor for one request.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        token_history = req_state["token_history"]
        cached_compressed = req_state["compressed_token_history"]
        cached_tensor = req_state.get("compressed_token_history_tensor")
        cached_tensor_valid = bool(
            req_state.get("compressed_token_history_tensor_valid", False)
        )
        compression = self.token_compression_map
        if compression is None:
            return torch.empty(0, dtype=torch.int64, device="cpu")

        if (
            isinstance(cached_tensor, torch.Tensor)
            and cached_tensor_valid
            and cached_tensor.ndim == 1
            and int(cached_tensor.shape[0]) == len(token_history)
        ):
            if len(cached_compressed) != len(token_history):
                req_state["compressed_token_history"] = cached_tensor.tolist()
            return cached_tensor
        if len(cached_compressed) != len(token_history):
            history_tensor = torch.as_tensor(token_history, dtype=torch.int64, device="cpu")
            clamped = torch.remainder(history_tensor, compression.shape[0])
            compressed_history = compression[clamped]
            req_state["compressed_token_history"] = compressed_history.tolist()
            req_state["compressed_token_history_tensor"] = compressed_history
            req_state["compressed_token_history_tensor_valid"] = True
            return compressed_history
        compressed_history = torch.as_tensor(cached_compressed, dtype=torch.int64, device="cpu")
        req_state["compressed_token_history_tensor"] = compressed_history
        req_state["compressed_token_history_tensor_valid"] = True
        return compressed_history

    def _build_lookup_snapshot(
        self, req_state: dict[str, Any], new_token_count: int
    ) -> EngramLookupSnapshot:
        """Build an immutable lookup snapshot for one request step.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            new_token_count (int): Number of newly appended tokens for this request.

        Returns:
            EngramLookupSnapshot: Computed result for this helper.
        """
        token_history = req_state["token_history"]
        total_tokens = len(token_history)
        start_token = max(0, total_tokens - max(0, new_token_count))
        if int(new_token_count) == 1:
            ring = req_state.get("compressed_decode_ring")
            decode_len = int(req_state.get("compressed_decode_len", 0))
            decode_head = int(req_state.get("compressed_decode_head", 0))
            if (
                isinstance(ring, torch.Tensor)
                and ring.device.type == "cpu"
                and ring.dtype == torch.int64
                and ring.ndim == 1
                and ring.numel() > 0
                and decode_len > 0
            ):
                # Decode path consumes only the latest n-gram window from the ring.
                keep = min(decode_len, self.max_ngram_order)
                tail = torch.empty((keep,), dtype=torch.int64, device="cpu")
                ring_size = int(ring.shape[0])
                for i in range(keep):
                    tail_idx = keep - 1 - i
                    ring_pos = (decode_head - 1 - i) % ring_size
                    tail[tail_idx] = ring[ring_pos]
                compressed_history = tail
            else:
                compressed_history = self._ensure_compressed_history(req_state)
        else:
            compressed_history = self._ensure_compressed_history(req_state)
        return EngramLookupSnapshot(
            compressed_history=compressed_history,
            total_tokens=total_tokens,
            new_token_count=max(0, new_token_count),
            start_token=start_token,
        )

    def build_lookup_for_layer_from_snapshot(
        self, snapshot: EngramLookupSnapshot, layer_idx: int
    ) -> torch.Tensor:
        """Build lookup embeddings for one layer from one snapshot.

        Args:
            self (Any): Module or runtime instance.
            snapshot (EngramLookupSnapshot): Immutable lookup snapshot built from request state.
            layer_idx (int): Transformer layer index.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        if snapshot.new_token_count == 1:
            lookup = _build_decode_lookup_embedding_from_snapshot(
                snapshot=snapshot,
                host=self.host_tables_by_layer.get(layer_idx),
                table_layout=self.table_layout,
                mem_dim=self.mem_dim,
                max_ngram_order=self.max_ngram_order,
                pad_id=self.compressed_pad_id,
            )
            return self._pin_cpu_lookup_tensor(lookup)
        lookup = _build_lookup_embeddings_from_snapshot(
            snapshot=snapshot,
            host=self.host_tables_by_layer.get(layer_idx),
            table_layout=self.table_layout,
            mem_dim=self.mem_dim,
            max_ngram_order=self.max_ngram_order,
            pad_id=self.compressed_pad_id,
        )
        return self._pin_cpu_lookup_tensor(lookup)

    def build_lookup_for_layers_from_snapshot(
        self, snapshot: EngramLookupSnapshot, layer_indices: tuple[int, ...]
    ) -> dict[int, torch.Tensor]:
        """Build lookup embeddings for multiple layers from one snapshot.

        Args:
            self (Any): Module or runtime instance.
            snapshot (EngramLookupSnapshot): Immutable lookup snapshot built from request state.
            layer_indices (tuple[int, ...]): Tuple/list of Engram-enabled layer indices.

        Returns:
            dict[int, torch.Tensor]: Computed result for this helper.
        """
        return {
            int(layer_idx): self.build_lookup_for_layer_from_snapshot(
                snapshot=snapshot,
                layer_idx=int(layer_idx),
            )
            for layer_idx in layer_indices
        }

    def build_lookup_for_request_layer(
        self, req_state: dict[str, Any], layer_idx: int, new_token_count: int
    ) -> torch.Tensor:
        """Build lookup embeddings for one request/layer pair.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            layer_idx (int): Transformer layer index.
            new_token_count (int): Number of newly appended tokens for this request.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        snapshot = self._build_lookup_snapshot(req_state, new_token_count)
        return self.build_lookup_for_layer_from_snapshot(snapshot, layer_idx)

    def build_lookup_for_request_layer_decode(
        self, req_state: dict[str, Any], layer_idx: int
    ) -> torch.Tensor:
        """Build decode lookup embeddings for one request/layer pair.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            layer_idx (int): Transformer layer index.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        snapshot = self._build_lookup_snapshot(req_state, 1)
        lookup = _build_decode_lookup_embedding_from_snapshot(
            snapshot=snapshot,
            host=self.host_tables_by_layer.get(layer_idx),
            table_layout=self.table_layout,
            mem_dim=self.mem_dim,
            max_ngram_order=self.max_ngram_order,
            pad_id=self.compressed_pad_id,
        )
        return self._pin_cpu_lookup_tensor(lookup)

    def build_lookup_for_requests_layer_decode(
        self, req_states: list[dict[str, Any]], layer_idx: int
    ) -> torch.Tensor:
        """Build decode lookup embeddings for a request batch on one layer.

        Args:
            self (Any): Module or runtime instance.
            req_states (list[dict[str, Any]]): List of per-request runtime state dictionaries.
            layer_idx (int): Transformer layer index.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        if len(req_states) == 0:
            return self._pin_cpu_lookup_tensor(
                torch.zeros((0, self.mem_dim), dtype=torch.bfloat16, device="cpu")
            )
        host = self.host_tables_by_layer.get(layer_idx)
        if host is None:
            return self._pin_cpu_lookup_tensor(
                torch.zeros((len(req_states), self.mem_dim), dtype=torch.bfloat16, device="cpu")
            )
        tokens_by_offset_ws, hash_out_ws = self._ensure_decode_cpu_workspace(
            batch=len(req_states),
            num_tables=len(self.table_layout),
        )
        pad_id = int(self.compressed_pad_id)
        for req_idx, req_state in enumerate(req_states):
            ring = self._ensure_decode_ring(req_state)
            ring_size = int(ring.shape[0])
            decode_len = int(req_state.get("compressed_decode_len", 0))
            head = int(req_state.get("compressed_decode_head", 0)) % ring_size
            if decode_len <= 0:
                tokens_by_offset_ws[:, req_idx].fill_(pad_id)
                continue
            for k in range(self.max_ngram_order):
                if k < decode_len:
                    pos = (head - 1 - k) % ring_size
                    tokens_by_offset_ws[k, req_idx] = ring[pos]
                else:
                    tokens_by_offset_ws[k, req_idx] = pad_id

        multipliers = host.layer_multipliers.detach().to(dtype=torch.int64, device="cpu")
        col_idx = 0
        for order in range(2, self.max_ngram_order + 1):
            mix = tokens_by_offset_ws[0] * multipliers[0]
            for k in range(1, order):
                mix = torch.bitwise_xor(mix, tokens_by_offset_ws[k] * multipliers[k])
            head_vocab_sizes = host.head_vocab_sizes_by_order[order].to(
                dtype=torch.int64, device="cpu"
            )
            for head_idx in range(head_vocab_sizes.shape[0]):
                mod = int(head_vocab_sizes[head_idx].item())
                hash_out_ws[:, col_idx].copy_(
                    torch.remainder(mix, mod).to(dtype=torch.int64)
                )
                col_idx += 1

        embeddings = torch.zeros(
            (len(req_states), self.mem_dim), dtype=torch.bfloat16, device="cpu"
        )
        if hash_out_ws.numel() == 0:
            return self._pin_cpu_lookup_tensor(embeddings)
        row_ids = hash_out_ws.to(dtype=torch.long, device="cpu")
        num_tables = row_ids.shape[1]
        table_ids = torch.arange(num_tables, dtype=torch.long, device="cpu").unsqueeze(0)
        table_ids = table_ids.expand(row_ids.shape[0], num_tables)
        gathered = host.tables[table_ids, row_ids]
        return self._pin_cpu_lookup_tensor(gathered.reshape(len(req_states), self.mem_dim))

    def build_lookup_for_requests_layers_decode(
        self,
        *,
        compressed_histories: list[torch.Tensor],
        layer_indices: tuple[int, ...],
    ) -> dict[int, torch.Tensor]:
        """Build decode lookup embeddings for a request batch across layers.

        Args:
            self (Any): Module or runtime instance.
            compressed_histories (list[torch.Tensor]): Compressed token history tensors for request batch.
                Shape: list of [T_i]. Dtype: torch.int64 tensors.
            layer_indices (tuple[int, ...]): Tuple/list of Engram-enabled layer indices.

        Returns:
            dict[int, torch.Tensor]: Computed result for this helper.
        """
        outputs: dict[int, torch.Tensor] = {}
        for layer_idx in layer_indices:
            lookup = _build_decode_lookup_embeddings_from_histories(
                compressed_histories=compressed_histories,
                host=self.host_tables_by_layer.get(int(layer_idx)),
                table_layout=self.table_layout,
                mem_dim=self.mem_dim,
                max_ngram_order=self.max_ngram_order,
                pad_id=self.compressed_pad_id,
            )
            outputs[int(layer_idx)] = self._pin_cpu_lookup_tensor(lookup)
        return outputs

    def schedule_async_step_prefetch(
        self,
        *,
        step_uid: int,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_ids: list[str] | tuple[str, ...],
        layer_indices: list[int] | tuple[int, ...],
        target_device: torch.device | str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> None:
        """Schedule per-request async lookup prefetch for a step.

        Args:
            self (Any): Module or runtime instance.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.
            input_ids (torch.Tensor): Flattened token IDs before embedding lookup.
                Shape: [num_step_tokens]. Dtype: torch.int64.
            query_start_loc (torch.Tensor): Prefix-sum token offsets per request.
                Shape: [num_requests + 1]. Dtype: torch.int32/torch.int64.
            request_ids (list[str] | tuple[str, ...]): Request identifiers aligned with query_start_loc.
                Shape: [num_requests]. Dtype: list[str].
            layer_indices (list[int] | tuple[int, ...]): Tuple/list of Engram-enabled layer indices.
            target_device (torch.device | str | None): Optional prefill lookup target device.
            target_dtype (torch.dtype | None): Optional prefill lookup target dtype.

        Returns:
            None: Function returns no value.
        """
        with torch.profiler.record_function("engram:schedule_async_prefetch"):
            if len(request_ids) == 0 or len(layer_indices) == 0:
                return
            if self._async_executor is None:
                self.init_async_executor(self.async_workers)
            assert self._async_executor is not None

            self._drop_stale_lookup_futures(step_uid)
            for req_idx, req_id in enumerate(request_ids):
                start = query_start_loc[req_idx].item()
                end = query_start_loc[req_idx + 1].item()
                if end <= start:
                    continue
                token_ids = input_ids[start:end]
                self.append_tokens_for_step_uid(req_id, step_uid, token_ids)
                req_state = self.ensure_request(req_id)
                snapshot = self._build_lookup_snapshot(req_state, end - start)
                key = (step_uid, req_id)
                prefill_target_device = (
                    torch.device(target_device) if target_device is not None else None
                )

                # Create wrapper function with profiler annotation for CPU thread
                def _do_prefill_lookup(
                    snapshot=snapshot,
                    layer_indices=layer_indices,
                    prefill_key=key,
                    staged_device=prefill_target_device,
                    staged_dtype=target_dtype,
                ):
                    with torch.profiler.record_function("engram:cpu_lookup_prefill"):
                        payload_by_layer = self.build_lookup_for_layers_from_snapshot(
                            snapshot,
                            tuple(int(layer_idx) for layer_idx in layer_indices),
                        )
                        staged_payload, ready_event = (
                            self._stage_lookup_payload_to_device(
                                payload_by_layer=payload_by_layer,
                                target_device=staged_device,
                                target_dtype=staged_dtype,
                            )
                        )
                        with self._async_lock:
                            if (
                                prefill_key in self.lookup_futures
                                or prefill_key in self.lookup_prefetch_cache
                            ):
                                self.lookup_prefetch_ready_events[prefill_key] = ready_event
                        return staged_payload

                future = self._async_executor.submit(_do_prefill_lookup)
                with self._async_lock:
                    self.lookup_futures[key] = future
                    self.lookup_prefetch_cache.pop(key, None)
                    self.lookup_prefetch_ready_events.pop(key, None)

    def schedule_async_decode_step_prefetch(
        self,
        *,
        step_uid: int,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_ids: list[str] | tuple[str, ...],
        layer_indices: list[int] | tuple[int, ...],
        target_device: torch.device | str | None = None,
        target_dtype: torch.dtype | None = None,
    ) -> None:
        """Schedule batched decode async prefetch for a step.

        Args:
            self (Any): Module or runtime instance.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.
            input_ids (torch.Tensor): Flattened token IDs before embedding lookup.
                Shape: [num_step_tokens]. Dtype: torch.int64.
            query_start_loc (torch.Tensor): Prefix-sum token offsets per request.
                Shape: [num_requests + 1]. Dtype: torch.int32/torch.int64.
            request_ids (list[str] | tuple[str, ...]): Request identifiers aligned with query_start_loc.
                Shape: [num_requests]. Dtype: list[str].
            layer_indices (list[int] | tuple[int, ...]): Tuple/list of Engram-enabled layer indices.
            target_device (torch.device | str | None): Optional decode lookup target device.
            target_dtype (torch.dtype | None): Optional decode lookup target dtype.

        Returns:
            None: Function returns no value.
        """
        with torch.profiler.record_function("engram:schedule_async_decode"):
            if len(request_ids) == 0 or len(layer_indices) == 0:
                return
            if self._async_executor is None:
                self.init_async_executor(self.async_workers)
            assert self._async_executor is not None

            self._drop_stale_lookup_futures(step_uid)

            decode_req_ids: list[str] = []
            compressed_histories: list[torch.Tensor] = []
            for req_idx, req_id in enumerate(request_ids):
                start = query_start_loc[req_idx].item()
                end = query_start_loc[req_idx + 1].item()
                if end <= start:
                    continue
                token_ids = input_ids[start:end]
                self.append_tokens_for_step_uid(req_id, step_uid, token_ids)
                req_state = self.ensure_request(req_id)
                snapshot = self._build_lookup_snapshot(req_state, end - start)
                compressed_histories.append(snapshot.compressed_history.clone())
                decode_req_ids.append(req_id)

            if len(decode_req_ids) == 0:
                return

            key = int(step_uid)
            layer_indices_tuple = tuple(int(layer_idx) for layer_idx in layer_indices)
            decode_target_device = (
                torch.device(target_device) if target_device is not None else None
            )

            # Create wrapper function with profiler annotation for CPU thread
            def _do_decode_lookup(
                histories=compressed_histories,
                layer_indices=layer_indices_tuple,
                step_key=key,
                staged_device=decode_target_device,
                staged_dtype=target_dtype,
            ):
                with torch.profiler.record_function("engram:cpu_lookup_decode"):
                    payload_by_layer = self.build_lookup_for_requests_layers_decode(
                        compressed_histories=histories,
                        layer_indices=layer_indices,
                    )
                    staged_payload, ready_event = (
                        self._stage_lookup_payload_to_device(
                            payload_by_layer=payload_by_layer,
                            target_device=staged_device,
                            target_dtype=staged_dtype,
                        )
                    )
                    with self._async_lock:
                        if (
                            step_key in self.decode_lookup_batch_futures
                            or step_key in self.decode_lookup_batch_req_ids
                        ):
                            self.decode_lookup_batch_ready_events[step_key] = ready_event
                    return staged_payload

            future = self._async_executor.submit(_do_decode_lookup)
            with self._async_lock:
                self._drop_decode_lookup_batch_step_locked(key)
                self.decode_lookup_batch_futures[key] = future
                req_ids_tuple = tuple(decode_req_ids)
                self.decode_lookup_batch_req_ids[key] = req_ids_tuple
                self.decode_lookup_batch_req_pos[key] = {
                    req_id: idx for idx, req_id in enumerate(req_ids_tuple)
                }
                self.decode_lookup_batch_row_idx_cache[key] = {}

    def consume_decode_prefetched_layer_or_fallback(
        self,
        *,
        req_ids: list[str],
        layer_idx: int,
        step_uid: int,
        fallback_fn,
        wait_timeout_s: float = 0.0,
    ) -> torch.Tensor:
        """Consume batched decode prefetch or fallback to sync lookup.

        Args:
            self (Any): Module or runtime instance.
            req_ids (list[str]): Req ids input.
            layer_idx (int): Transformer layer index.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.
            fallback_fn (Any): Callable used when prefetched data is unavailable.
            wait_timeout_s (float): Optional wait timeout before sync fallback.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        with torch.profiler.record_function("engram:consume_future_decode"):
            if len(req_ids) == 0:
                return fallback_fn()
            layer_idx = int(layer_idx)
            step_uid = int(step_uid)
            req_ids_tuple = tuple(req_ids)

            with self._async_lock:
                expected_req_ids = self.decode_lookup_batch_req_ids.get(step_uid)
                req_pos = self.decode_lookup_batch_req_pos.get(step_uid)
                row_idx_cache = self.decode_lookup_batch_row_idx_cache.get(step_uid)
                cached = self.decode_lookup_batch_cache.get(step_uid)
                future = self.decode_lookup_batch_futures.get(step_uid)
                ready_event = self.decode_lookup_batch_ready_events.get(step_uid)

            if expected_req_ids is None or req_pos is None:
                return fallback_fn()

            cached_row_idx: torch.Tensor | None = None
            if expected_req_ids != req_ids_tuple:
                if row_idx_cache is None:
                    with self._async_lock:
                        row_idx_cache = self.decode_lookup_batch_row_idx_cache.get(step_uid)
                        if row_idx_cache is None:
                            row_idx_cache = {}
                            self.decode_lookup_batch_row_idx_cache[step_uid] = row_idx_cache
                cached_row_idx = row_idx_cache.get(req_ids_tuple)
                if cached_row_idx is None:
                    try:
                        req_rows = [int(req_pos[req_id]) for req_id in req_ids_tuple]
                    except KeyError:
                        return fallback_fn()
                    if req_rows != sorted(req_rows):
                        return fallback_fn()
                    cached_row_idx = torch.tensor(req_rows, dtype=torch.long, device="cpu")
                    with self._async_lock:
                        cache_for_step = self.decode_lookup_batch_row_idx_cache.get(step_uid)
                        if cache_for_step is None:
                            cache_for_step = {}
                            self.decode_lookup_batch_row_idx_cache[step_uid] = cache_for_step
                        cache_for_step[req_ids_tuple] = cached_row_idx

            def _slice_or_full(tensor: torch.Tensor) -> torch.Tensor:
                """Return full tensor or subset rows for active decode request IDs.

                Args:
                    tensor (torch.Tensor): Decode lookup tensor for one Engram layer.
                        Shape: [batch, mem_dim]. Dtype: torch.bfloat16.

                Returns:
                    torch.Tensor: Full tensor when request order matches prefetch order,
                        otherwise a row-selected subset.
                        Shape: [active_batch, mem_dim]. Dtype: matches ``tensor``.
                """
                if expected_req_ids == req_ids_tuple:
                    if ready_event is not None and tensor.is_cuda:
                        torch.cuda.current_stream(device=tensor.device).wait_event(
                            ready_event
                        )
                    return tensor
                assert cached_row_idx is not None
                if ready_event is not None and tensor.is_cuda:
                    torch.cuda.current_stream(device=tensor.device).wait_event(
                        ready_event
                    )
                row_idx = cached_row_idx
                if row_idx.device != tensor.device:
                    row_idx = row_idx.to(device=tensor.device)
                return tensor.index_select(0, row_idx)

            if cached is not None:
                tensor = cached.get(layer_idx)
                if tensor is not None:
                    return _slice_or_full(tensor)
                return fallback_fn()

            if future is None:
                return fallback_fn()

            if not future.done() and wait_timeout_s > 0.0:
                try:
                    payload_by_layer = future.result(timeout=wait_timeout_s)
                    with self._async_lock:
                        self.decode_lookup_batch_futures.pop(step_uid, None)
                        self.decode_lookup_batch_cache[step_uid] = payload_by_layer
                    tensor = payload_by_layer.get(layer_idx)
                    if tensor is not None:
                        return _slice_or_full(tensor)
                    return fallback_fn()
                except FutureTimeoutError:
                    return fallback_fn()
                except Exception:
                    with self._async_lock:
                        self.decode_lookup_batch_futures.pop(step_uid, None)
                        self.decode_lookup_batch_ready_events.pop(step_uid, None)
                        should_log = step_uid not in self._async_warned_decode_steps
                        if should_log:
                            self._async_warned_decode_steps.add(step_uid)
                    if should_log:
                        logger.warning(
                            "Engram async decode lookup failed for step=%d; "
                            "falling back to sync lookup.",
                            step_uid,
                            exc_info=True,
                        )
                    return fallback_fn()

            if not future.done():
                return fallback_fn()

            try:
                payload_by_layer = future.result()
                with self._async_lock:
                    self.decode_lookup_batch_futures.pop(step_uid, None)
                    self.decode_lookup_batch_cache[step_uid] = payload_by_layer
                tensor = payload_by_layer.get(layer_idx)
                if tensor is not None:
                    return _slice_or_full(tensor)
                return fallback_fn()
            except Exception:
                with self._async_lock:
                    self.decode_lookup_batch_futures.pop(step_uid, None)
                    self.decode_lookup_batch_ready_events.pop(step_uid, None)
                    should_log = step_uid not in self._async_warned_decode_steps
                    if should_log:
                        self._async_warned_decode_steps.add(step_uid)
                if should_log:
                    logger.warning(
                        "Engram async decode lookup failed for step=%d; "
                        "falling back to sync lookup.",
                        step_uid,
                        exc_info=True,
                    )
                return fallback_fn()

    def consume_prefetched_layer_or_fallback(
        self,
        req_id: str,
        layer_idx: int,
        step_uid: int,
        fallback_fn,
        *,
        wait_for_ready_timeout_s: float = 0.0,
    ) -> Any:
        """Consume per-request prefetch or fallback to sync lookup.

        Args:
            self (Any): Module or runtime instance.
            req_id (str): Single request identifier.
            layer_idx (int): Transformer layer index.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.
            fallback_fn (Any): Callable used when prefetched data is unavailable.
            wait_for_ready_timeout_s (float): Timeout for per-request prefetch readiness.

        Returns:
            Any: Computed result for this helper.
        """
        with torch.profiler.record_function("engram:consume_future_prefill"):
            key = (step_uid, req_id)
            layer_idx = int(layer_idx)
            with self._async_lock:
                cached_payload = self.lookup_prefetch_cache.get(key)
                ready_event = self.lookup_prefetch_ready_events.get(key)
                if cached_payload is not None and layer_idx in cached_payload:
                    out = cached_payload[layer_idx]
                    if ready_event is not None and isinstance(out, torch.Tensor) and out.is_cuda:
                        torch.cuda.current_stream(device=out.device).wait_event(ready_event)
                    return out
                future = self.lookup_futures.get(key)
            if future is None:
                return fallback_fn()
            if not future.done() and wait_for_ready_timeout_s > 0:
                try:
                    payload_by_layer = future.result(timeout=wait_for_ready_timeout_s)
                    with self._async_lock:
                        self.lookup_futures.pop(key, None)
                        self.lookup_prefetch_cache[key] = payload_by_layer
                        ready_event = self.lookup_prefetch_ready_events.get(key)
                    if layer_idx in payload_by_layer:
                        out = payload_by_layer[layer_idx]
                        if ready_event is not None and isinstance(out, torch.Tensor) and out.is_cuda:
                            torch.cuda.current_stream(device=out.device).wait_event(ready_event)
                        return out
                    return fallback_fn()
                except FutureTimeoutError:
                    return fallback_fn()
                except Exception:
                    with self._async_lock:
                        self.lookup_futures.pop(key, None)
                        self.lookup_prefetch_ready_events.pop(key, None)
                        should_log = key not in self._async_warned_failure_keys
                        if should_log:
                            self._async_warned_failure_keys.add(key)
                    if should_log:
                        logger.warning(
                            "Engram async lookup failed for step=%d req_id=%s; "
                            "falling back to sync lookup.",
                            step_uid,
                            req_id,
                            exc_info=True,
                        )
                    return fallback_fn()
            if not future.done():
                return fallback_fn()
            try:
                payload_by_layer = future.result()
                with self._async_lock:
                    self.lookup_futures.pop(key, None)
                    self.lookup_prefetch_cache[key] = payload_by_layer
                    ready_event = self.lookup_prefetch_ready_events.get(key)
                if layer_idx in payload_by_layer:
                    out = payload_by_layer[layer_idx]
                    if ready_event is not None and isinstance(out, torch.Tensor) and out.is_cuda:
                        torch.cuda.current_stream(device=out.device).wait_event(ready_event)
                    return out
                return fallback_fn()
            except Exception:
                with self._async_lock:
                    self.lookup_futures.pop(key, None)
                    self.lookup_prefetch_ready_events.pop(key, None)
                    should_log = key not in self._async_warned_failure_keys
                    if should_log:
                        self._async_warned_failure_keys.add(key)
                if should_log:
                    logger.warning(
                        "Engram async lookup failed for step=%d req_id=%s; "
                        "falling back to sync lookup.",
                        step_uid,
                        req_id,
                        exc_info=True,
                    )
                return fallback_fn()

    def init_synthetic_compression_map(self, seed: int) -> None:
        """Initialize synthetic token-compression mapping.

        Args:
            self (Any): Module or runtime instance.
            seed (int): Seed for deterministic synthetic parameter initialization.

        Returns:
            None: Function returns no value.
        """
        compressed_vocab = max(1, int(self.vocab_size * self.compression_ratio))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.token_compression_map = torch.randint(
            low=0,
            high=compressed_vocab,
            size=(self.vocab_size,),
            dtype=torch.int64,
            generator=generator,
            device="cpu",
        )
        pad_token_id = int(self.pad_token_id)
        if not (0 <= pad_token_id < compressed_vocab):
            raise ValueError(
                "`pad_token_id` must be in range [0, compressed_vocab), "
                f"got: {pad_token_id} with compressed_vocab={compressed_vocab}."
            )
        self.compressed_pad_id = pad_token_id

    def init_synthetic_layer_host_tables(self, layer_idx: int, seed: int) -> None:
        """Initialize synthetic host lookup tables for one Engram layer.

        Args:
            self (Any): Module or runtime instance.
            layer_idx (int): Transformer layer index.
            seed (int): Seed for deterministic synthetic parameter initialization.

        Returns:
            None: Function returns no value.
        """
        table_layout = _build_table_layout(
            max_ngram_order=self.max_ngram_order,
            heads=self.heads,
            mem_dim=self.mem_dim,
        )

        mhash_generator = torch.Generator(device="cpu")
        mhash_generator.manual_seed(seed + (layer_idx * 997))
        multipliers = torch.randint(
            low=1,
            high=2**31 - 1,
            size=(self.max_ngram_order,),
            dtype=torch.int64,
            generator=mhash_generator,
            device="cpu",
        )
        layer_multipliers = multipliers * 2 + 1
        head_vocab_sizes_by_order = {
            order: torch.full(
                (self.heads,),
                self.memory_size,
                dtype=torch.int64,
                device="cpu",
            )
            for order in range(2, self.max_ngram_order + 1)
        }
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + (layer_idx * 1009))
        num_tables = len(table_layout)
        table_dim = self.mem_dim // num_tables
        tables = torch.randn(
            num_tables,
            self.memory_size,
            table_dim,
            dtype=torch.bfloat16,
            generator=generator,
            device="cpu",
        ) * 0.02
        self.host_tables_by_layer[layer_idx] = EngramLayerHostTables(
            layer_multipliers=layer_multipliers,
            head_vocab_sizes_by_order=head_vocab_sizes_by_order,
            tables=tables,
        )
