# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Engram neural layer forward paths for prefill and decode execution.

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.engram_hash import _build_table_layout
from vllm.model_executor.layers.engram_payload import (
    ENGRAM_STEP_PAYLOAD_KEY,
    validate_engram_step_payload,
)
from vllm.model_executor.layers.engram_runtime import EngramRuntimeState

DEFAULT_ENGRAM_DECODE_PREFETCH_WAIT_S = 0.0

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization with a learned per-channel scale.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        weight (torch.Tensor): Per-channel scale tensor.
        eps (float): Numerical epsilon for stability.

    Returns:
        torch.Tensor: Computed result for this helper.
    """
    inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x * inv_rms * weight

class EngramLayer(nn.Module):
    """Inference-only Engram layer for Qwen3 integration."""

    def __init__(
        self,
        hidden_size: int,
        layer_idx: int,
        *,
        max_ngram_order: int,
        engram_heads: int,
        engram_mem_dim: int,
        conv_kernel: int,
        conv_dilation: int | None,
        runtime_state: EngramRuntimeState | None = None,
    ):
        """Implement   init  .

        Args:
            self (Any): Module or runtime instance.
            hidden_size (int): Hidden size input.
            layer_idx (int): Transformer layer index.
            max_ngram_order (int): Maximum n-gram order used by Engram hashing.
            engram_heads (int): Engram heads input.
            engram_mem_dim (int): Engram mem dim input.
            conv_kernel (int): Depthwise convolution kernel size.
            conv_dilation (int | None): Depthwise convolution dilation factor.
            runtime_state (EngramRuntimeState | None): Runtime state input.

        Returns:
            Any: Computed result for this helper.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.scale = 1.0 / math.sqrt(float(hidden_size))
        self.layer_idx = layer_idx
        self.max_ngram_order = max_ngram_order
        self.engram_heads = engram_heads
        self.engram_mem_dim = engram_mem_dim
        self.conv_kernel = conv_kernel
        self.conv_dilation = conv_dilation or max_ngram_order

        self.table_layout = _build_table_layout(
            max_ngram_order=self.max_ngram_order,
            heads=self.engram_heads,
            mem_dim=self.engram_mem_dim,
        )

        self.runtime_state = runtime_state
        self.register_buffer("w_k", None, persistent=False)
        self.register_buffer("w_v", None, persistent=False)
        self.register_buffer("rms_weight_h", None, persistent=False)
        self.register_buffer("rms_weight_k", None, persistent=False)
        self.register_buffer("rms_weight_u", None, persistent=False)
        # Keep conv as a runtime-only object (not a registered submodule),
        # so checkpoint loading does not expect Engram synthetic weights.
        object.__setattr__(self, "_conv", None)
        self._kv_fused_weight_cache: dict[
            tuple[torch.device, torch.dtype, int, int], torch.Tensor
        ] = {}
        self._lookup_h2d_stream: torch.cuda.Stream | None = None
        self._lookup_h2d_stream_device: torch.device | None = None

    def _get_kv_fused_weight(self) -> torch.Tensor:
        """Return cached fused projection weight [mem_dim, 2 * hidden_size]."""
        assert self.w_k is not None
        assert self.w_v is not None
        w_k = self.w_k[0]
        w_v = self.w_v
        key = (w_k.device, w_k.dtype, w_k.data_ptr(), w_v.data_ptr())
        cached = self._kv_fused_weight_cache.get(key)
        if (
            cached is None
            or cached.device != w_k.device
            or cached.dtype != w_k.dtype
            or cached.shape != (w_k.shape[0], w_k.shape[1] + w_v.shape[1])
        ):
            cached = torch.cat((w_k, w_v), dim=1).contiguous()
            self._kv_fused_weight_cache = {key: cached}
        return cached

    @property
    def conv_weight(self) -> torch.Tensor:
        """Return depthwise convolution weight tensor.

        Args:
            self (Any): Module or runtime instance.

        Returns:
            torch.Tensor: Depthwise convolution filter weights.
                Shape: [hidden_size, 1, conv_kernel]. Dtype: torch.bfloat16.
        """
        conv = getattr(self, "_conv", None)
        if conv is None:
            raise RuntimeError("Engram conv is not initialized.")
        return conv.weight

    def init_synthetic_layer_parameters(self, seed: int) -> None:
        """Initialize synthetic Engram compute parameters for this layer.

        Args:
            self (Any): Module or runtime instance.
            seed (int): Seed for deterministic synthetic parameter initialization.

        Returns:
            None: Function returns no value.
        """
        device = torch.device("cuda")
        generator = torch.Generator(device=device)
        sample_device = device
        generator.manual_seed(seed + (self.layer_idx * 6151))
        self.w_k = torch.randn(
            1,
            self.engram_mem_dim,
            self.hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
            device=sample_device,
        ) * 0.02
        self.w_v = torch.randn(
            self.engram_mem_dim,
            self.hidden_size,
            dtype=torch.bfloat16,
            generator=generator,
            device=sample_device,
        ) * 0.02
        conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            bias=False,
            padding=0,
            dilation=self.conv_dilation,
        ).to(device=sample_device, dtype=torch.bfloat16)
        conv.weight.data.copy_(
            torch.randn(
                self.hidden_size,
                1,
                self.conv_kernel,
                dtype=torch.bfloat16,
                generator=generator,
                device=sample_device,
            )
            * 0.02
        )
        object.__setattr__(self, "_conv", conv)
        self.rms_weight_h = torch.ones(
            self.hidden_size, dtype=torch.bfloat16, device=sample_device
        )
        self.rms_weight_k = torch.ones(
            self.hidden_size, dtype=torch.bfloat16, device=sample_device
        )
        self.rms_weight_u = torch.ones(
            self.hidden_size, dtype=torch.bfloat16, device=sample_device
        )

    def _get_engram_step_payload(self) -> dict[str, Any] | None:
        """Read and validate the Engram step payload from forward context.

        Args:
            self (Any): Module or runtime instance.

        Returns:
            dict[str, Any] | None: Computed result for this helper.
        """
        if not is_forward_context_available():
            return None
        forward_context = get_forward_context()
        payload = forward_context.additional_kwargs.get(ENGRAM_STEP_PAYLOAD_KEY)
        if payload is None:
            return None
        validate_engram_step_payload(payload)
        return payload

    def _build_lookup_embeddings(
        self, req_state: dict[str, Any], new_token_count: int, step_uid: int | None = None
    ) -> torch.Tensor:
        """Build lookup embeddings for prefill tokens of one request.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            new_token_count (int): Number of newly appended tokens for this request.
            step_uid (int | None): Monotonic step identifier for deduplication/prefetch.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        assert self.runtime_state is not None
        return self.runtime_state.build_lookup_for_request_layer(
            req_state=req_state,
            layer_idx=self.layer_idx,
            new_token_count=new_token_count,
        )

    def _build_decode_lookup_embedding(
        self, req_state: dict[str, Any], step_uid: int | None = None
    ) -> torch.Tensor:
        """Build one-token decode lookup embeddings for one request.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            step_uid (int | None): Monotonic step identifier for deduplication/prefetch.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        assert self.runtime_state is not None
        return self.runtime_state.build_lookup_for_request_layer_decode(
            req_state=req_state, layer_idx=self.layer_idx
        )

    def _copy_lookup_to_device_async(
        self, lookup_cpu: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Copy CPU lookup tensor to CUDA using a dedicated transfer stream.

        Args:
            self (Any): Module or runtime instance.
            lookup_cpu (torch.Tensor): CPU lookup tensor.
            hidden_states (torch.Tensor): CUDA hidden states for target device/dtype.

        Returns:
            torch.Tensor: Lookup tensor on CUDA.
        """
        if not hidden_states.is_cuda:
            raise ValueError("Engram lookup transfer requires CUDA hidden_states.")
        if lookup_cpu.device.type != "cpu":
            raise RuntimeError(
                "Engram lookup source must be on CPU before H2D copy. "
                f"Got device={lookup_cpu.device}."
            )
        if not lookup_cpu.is_pinned():
            # Some intermediate CPU ops (e.g. cat/index_select) may return
            # non-pinned tensors even when upstream buffers were pinned.
            # Try to pin for better overlap; continue if pinning is unavailable.
            try:
                lookup_cpu = lookup_cpu.pin_memory()
            except Exception:
                pass

        target_device = hidden_states.device
        if (
            self._lookup_h2d_stream is None
            or self._lookup_h2d_stream_device is None
            or self._lookup_h2d_stream_device != target_device
        ):
            self._lookup_h2d_stream = torch.cuda.Stream(device=target_device)
            self._lookup_h2d_stream_device = target_device

        lookup = torch.empty(
            lookup_cpu.shape,
            device=target_device,
            dtype=hidden_states.dtype,
        )
        copy_stream = self._lookup_h2d_stream
        assert copy_stream is not None
        with torch.cuda.stream(copy_stream):
            lookup.copy_(lookup_cpu, non_blocking=True)
        torch.cuda.current_stream(device=target_device).wait_stream(copy_stream)
        return lookup

    def _get_or_init_conv_history_entry(
        self,
        req_state: dict[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        history_len: int,
    ) -> dict[str, Any]:
        """Get or initialize request-local convolution history for this layer.

        Args:
            self (Any): Module or runtime instance.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.
            device (torch.device): Target torch device.
            dtype (torch.dtype): Target torch dtype.
            history_len (int): Cached decode-history length in tokens.

        Returns:
            dict[str, Any]: Computed result for this helper.
        """
        conv_history_by_layer = req_state["conv_history_by_layer"]
        history_entry = conv_history_by_layer.get(self.layer_idx)
        history = (
            history_entry.get("history") if isinstance(history_entry, dict) else history_entry
        )
        if (
            history is None
            or history.shape != (history_len, self.hidden_size)
            or history.device != device
            or history.dtype != dtype
        ):
            history = torch.zeros(
                history_len,
                self.hidden_size,
                dtype=dtype,
                device=device,
            )
            history_entry = {"history": history, "initialized": False, "head": 0}
            conv_history_by_layer[self.layer_idx] = history_entry
            return history_entry
        if not isinstance(history_entry, dict):
            history_entry = {"history": history, "initialized": True, "head": 0}
            conv_history_by_layer[self.layer_idx] = history_entry
        return history_entry

    def _conv_with_state(
        self,
        u_norm: torch.Tensor,
        req_state: dict[str, Any],
    ) -> torch.Tensor:
        """Run depthwise convolution over a segment and update request history state.

        Args:
            self (Any): Module or runtime instance.
            u_norm (torch.Tensor): Normalized Engram u tensor before depthwise convolution.
                Shape: [segment_tokens, hidden_size]. Dtype: torch.bfloat16.
            req_state (dict[str, Any]): Mutable per-request runtime state dictionary.

        Returns:
            torch.Tensor: Computed result for this helper.
        """

        conv = getattr(self, "_conv", None)
        assert conv is not None
        history_len = (self.conv_kernel - 1) * self.conv_dilation
        conv_history_by_layer = req_state["conv_history_by_layer"]

        if history_len > 0:
            history_entry = self._get_or_init_conv_history_entry(
                req_state,
                device=u_norm.device,
                dtype=u_norm.dtype,
                history_len=history_len,
            )
            history = history_entry["history"]
            if isinstance(history_entry, dict):
                head = int(history_entry.get("head", 0)) % history_len
            else:
                head = 0
            if head != 0:
                history = torch.cat([history[head:], history[:head]], dim=0)
            conv_input = torch.cat([history, u_norm], dim=0)
        else:
            conv_input = u_norm

        conv_input_t = conv_input.transpose(0, 1).unsqueeze(0)
        conv_out = conv(conv_input_t)
        conv_out = conv_out.squeeze(0).transpose(0, 1)

        if history_len > 0:
            conv_history_by_layer[self.layer_idx] = {
                "history": conv_input[-history_len:].detach(),
                "initialized": True,
                "head": 0,
            }
        return conv_out

    def _conv_one_token_with_state(
        self,
        u_norm_token: torch.Tensor,
        req_state: dict[str, Any],
    ) -> torch.Tensor:
        """Run one-token depthwise conv with circular history state.

        Args:
            self (Any): Engram layer instance.
                Shape: scalar object. Dtype: Python object.
            u_norm_token (torch.Tensor): Normalized Engram hidden token for one
                decode step.
                Shape: [1, hidden_size]. Dtype: follows model hidden dtype.
            req_state (dict[str, Any]): Per-request runtime state dictionary.
                Shape: mapping object. Dtype: Python dict.

        Returns:
            torch.Tensor: Depthwise-conv output for the single decode token.
                Shape: [1, hidden_size]. Dtype: same as ``u_norm_token``.
        """
        conv = getattr(self, "_conv", None)
        assert conv is not None
        if u_norm_token.shape[0] != 1:
            raise ValueError(
                f"`u_norm_token` must have seq_len=1, got: {tuple(u_norm_token.shape)}"
            )

        history_len = (self.conv_kernel - 1) * self.conv_dilation
        conv_history_by_layer = req_state["conv_history_by_layer"]
        token = u_norm_token[0]
        history: torch.Tensor | None = None
        if history_len > 0:
            history_entry = conv_history_by_layer.get(self.layer_idx)
            history = (
                history_entry.get("history")
                if isinstance(history_entry, dict)
                else history_entry
            )
            if (
                history is None
                or history.shape != (history_len, self.hidden_size)
                or history.device != u_norm_token.device
                or history.dtype != u_norm_token.dtype
            ):
                history = torch.zeros(
                    history_len,
                    self.hidden_size,
                    dtype=u_norm_token.dtype,
                    device=u_norm_token.device,
                )
                history_entry = {"history": history, "initialized": False, "head": 0}
                conv_history_by_layer[self.layer_idx] = history_entry
            elif not isinstance(history_entry, dict):
                history_entry = {"history": history, "initialized": True, "head": 0}
                conv_history_by_layer[self.layer_idx] = history_entry

        conv_w = conv.weight[:, 0, :]
        out = conv_w[:, self.conv_kernel - 1] * token
        if history is not None:
            assert isinstance(history_entry, dict)
            head = int(history_entry.get("head", 0)) % history_len
            for tap in range(self.conv_kernel - 1):
                hist_idx = tap * self.conv_dilation
                out = out + conv_w[:, tap] * history[(head + hist_idx) % history_len]

            head = (head + 1) % history_len
            write_pos = (head + history_len - 1) % history_len
            history[write_pos].copy_(token)
            history_entry["initialized"] = True
            history_entry["head"] = head
        return out.unsqueeze(0)

    def _conv_decode_batch_with_state(
        self,
        u_norm_batch: torch.Tensor,
        req_states: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Run one-token convolution for a decode batch and update per-request state.

        Args:
            self (Any): Module or runtime instance.
            u_norm_batch (torch.Tensor): Single-token normalized Engram u batch tensor.
                Shape: [batch, hidden_size]. Dtype: torch.bfloat16.
            req_states (list[dict[str, Any]]): List of per-request runtime state dictionaries.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        conv = getattr(self, "_conv", None)
        assert conv is not None
        if u_norm_batch.shape[0] != len(req_states):
            raise ValueError(
                "`u_norm_batch` batch size must match `req_states` length, got "
                f"{u_norm_batch.shape[0]} vs {len(req_states)}"
            )

        conv_w = conv.weight[:, 0, :]
        out = conv_w[:, self.conv_kernel - 1].unsqueeze(0) * u_norm_batch
        history_len = (self.conv_kernel - 1) * self.conv_dilation

        entries: list[dict[str, Any]] = []
        for req_state in req_states:
            entry = self._get_or_init_conv_history_entry(
                req_state,
                device=u_norm_batch.device,
                dtype=u_norm_batch.dtype,
                history_len=history_len,
            )
            entries.append(entry)

        heads = [int(entry.get("head", 0)) % history_len for entry in entries]
        for tap in range(self.conv_kernel - 1):
            hist_idx = tap * self.conv_dilation
            tapped = torch.stack(
                [
                    entry["history"][(heads[row_idx] + hist_idx) % history_len]
                    for row_idx, entry in enumerate(entries)
                ],
                dim=0,
            )
            out = out + tapped * conv_w[:, tap].unsqueeze(0)

        new_heads = [(head + 1) % history_len for head in heads]
        write_positions = [
            (head + history_len - 1) % history_len for head in new_heads
        ]
        for row_idx, entry in enumerate(entries):
            entry["history"][write_positions[row_idx]].copy_(u_norm_batch[row_idx])
            entry["head"] = new_heads[row_idx]
            entry["initialized"] = True
        return out

    def _compute_preconv_tensors(
        self,
        hidden_slice: torch.Tensor,
        lookup_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Engram projection, gating, and normalized tensors before convolution.

        Args:
            self (Any): Module or runtime instance.
            hidden_slice (torch.Tensor): Hidden-state slice for active tokens.
                Shape: [num_active_tokens, hidden_size]. Dtype: torch.bfloat16.
            lookup_embeddings (torch.Tensor): Lookup embedding tensor read from host tables.
                Shape: [num_active_tokens, engram_mem_dim]. Dtype: torch.bfloat16.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Computed result for this helper.
        """
        assert self.w_k is not None
        assert self.w_v is not None
        assert self.rms_weight_h is not None
        assert self.rms_weight_k is not None
        assert self.rms_weight_u is not None

        e = lookup_embeddings
        kv = e @ self._get_kv_fused_weight()
        k, v = kv.split(self.hidden_size, dim=-1)

        norm_h = _rms_norm(hidden_slice, self.rms_weight_h)
        norm_k = _rms_norm(k, self.rms_weight_k)
        alpha = torch.sigmoid(
            torch.sum(norm_h * norm_k, dim=-1, keepdim=True) * self.scale
        )
        u = alpha * v
        u_norm = _rms_norm(u, self.rms_weight_u)
        return u, u_norm

    def _collect_active_segments(
        self,
        *,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_ids: list[str],
        step_uid: int,
    ) -> tuple[list[tuple[str, dict[str, Any], int, int, int]], bool]:
        """Collect active request token segments and decode/prefill mode information.

        Args:
            self (Any): Module or runtime instance.
            input_ids (torch.Tensor): Flattened token IDs before embedding lookup.
                Shape: [num_step_tokens]. Dtype: torch.int64.
            query_start_loc (torch.Tensor): Prefix-sum token offsets per request.
                Shape: [num_requests + 1]. Dtype: torch.int32/torch.int64.
            request_ids (list[str]): Request identifiers aligned with query_start_loc.
                Shape: [num_requests]. Dtype: list[str].
            step_uid (int): Monotonic step identifier for deduplication/prefetch.

        Returns:
            tuple[list[tuple[str, dict[str, Any], int, int, int]], bool]: Computed result for this helper.
        """
        assert self.runtime_state is not None
        starts = query_start_loc[:-1]
        ends = query_start_loc[1:]
        lengths = ends - starts

        non_empty_indices = (lengths > 0).nonzero(as_tuple=False).flatten().tolist()
        accepted_indices: list[int] = []
        req_states_by_index: dict[int, dict[str, Any]] = {}
        for req_idx in non_empty_indices:
            req_id = request_ids[req_idx]
            req_state = self.runtime_state.ensure_request(req_id)
            last_step_uid_by_layer = req_state["last_step_uid_by_layer"]
            if last_step_uid_by_layer.get(self.layer_idx) == step_uid:
                continue
            last_step_uid_by_layer[self.layer_idx] = step_uid
            req_states_by_index[req_idx] = req_state
            accepted_indices.append(req_idx)

        active_segments: list[tuple[str, dict[str, Any], int, int, int]] = []
        if not accepted_indices:
            return active_segments, True

        accepted_starts = starts[accepted_indices].tolist()
        accepted_ends = ends[accepted_indices].tolist()
        accepted_lengths = lengths[accepted_indices]
        decode_only = bool(torch.all(accepted_lengths == 1).item())

        for req_idx, start, end in zip(accepted_indices, accepted_starts, accepted_ends):
            req_id = request_ids[req_idx]
            req_state = req_states_by_index[req_idx]
            if req_state.get("last_appended_step_uid") != step_uid:
                token_ids = input_ids[start:end]
                self.runtime_state.append_tokens_for_step_uid(
                    req_id, step_uid, token_ids
                )
            new_token_count = end - start
            active_segments.append((req_id, req_state, start, end, new_token_count))
        return active_segments, decode_only

    def _prefill_forward(
        self,
        hidden_states: torch.Tensor,
        active_segments: list[tuple[str, dict[str, Any], int, int, int]],
        *,
        step_uid: int,
    ) -> torch.Tensor:
        """Execute Engram prefill path for active request segments.

        Args:
            self (Any): Module or runtime instance.
            hidden_states (torch.Tensor): Flattened hidden states for current model step.
                Shape: [num_tokens, hidden_size]. Dtype: torch.bfloat16 in Engram path.
            active_segments (list[tuple[str, dict[str, Any], int, int, int]]):
                Active request segments as (req_id, req_state, start, end, new_token_count),
                where [start:end] indexes this request in flattened step tensors.
            step_uid (int): Monotonic step identifier for deduplication/prefetch.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        assert self.runtime_state is not None
        lookup_segments: list[tuple[str, dict[str, Any], int, int, torch.Tensor]] = []
        for req_id, req_state, start, end, new_token_count in active_segments:
            wait_for_ready_timeout_s = 0.0

            def _lookup_fallback(
                req_state=req_state,
                new_token_count=new_token_count,
                step_uid=step_uid,
            ) -> torch.Tensor:
                """Implement  lookup fallback.

                Args:
                    req_state (Any): Mutable per-request runtime state dictionary.
                    new_token_count (Any): Number of newly appended tokens for this request.
                    step_uid (Any): Monotonic step identifier for deduplication/prefetch.

                Returns:
                    torch.Tensor: Computed result for this helper.
                """
                if new_token_count == 1:
                    return self._build_decode_lookup_embedding(
                        req_state=req_state, step_uid=step_uid
                    )
                return self._build_lookup_embeddings(
                    req_state=req_state,
                    new_token_count=new_token_count,
                    step_uid=step_uid,
                )

            # CPU lookup embeddings produced from Engram table lookup (prefetched or fallback).
            lookup_cpu = self.runtime_state.consume_prefetched_layer_or_fallback(
                req_id=req_id,
                layer_idx=self.layer_idx,
                step_uid=step_uid,
                fallback_fn=_lookup_fallback,
                wait_for_ready_timeout_s=wait_for_ready_timeout_s,
            )
            lookup_segments.append((req_id, req_state, start, end, lookup_cpu))

        if len(lookup_segments) == 1:
            _, _, start, end, lookup_cpu = lookup_segments[0]
            hidden_cat = hidden_states[start:end]
            lookup_cpu_cat = lookup_cpu
        else:
            hidden_cat = torch.cat(
                [hidden_states[start:end] for _, _, start, end, _ in lookup_segments], dim=0
            )
            lookup_parts = [lookup_cpu for _, _, _, _, lookup_cpu in lookup_segments]
            lookup_devices = {part.device for part in lookup_parts}
            if len(lookup_devices) == 1:
                lookup_cpu_cat = torch.cat(lookup_parts, dim=0)
            else:
                normalized_parts: list[torch.Tensor] = []
                for part in lookup_parts:
                    if part.device == hidden_states.device and part.dtype == hidden_states.dtype:
                        normalized_parts.append(part)
                    elif part.device.type == "cpu":
                        normalized_parts.append(
                            self._copy_lookup_to_device_async(part, hidden_states)
                        )
                    else:
                        normalized_parts.append(
                            part.to(
                                device=hidden_states.device,
                                dtype=hidden_states.dtype,
                                non_blocking=True,
                            )
                        )
                lookup_cpu_cat = torch.cat(normalized_parts, dim=0)
        if (
            lookup_cpu_cat.device == hidden_states.device
            and lookup_cpu_cat.dtype == hidden_states.dtype
        ):
            lookup = lookup_cpu_cat
        elif lookup_cpu_cat.device.type == "cpu":
            lookup = self._copy_lookup_to_device_async(lookup_cpu_cat, hidden_states)
        else:
            lookup = lookup_cpu_cat.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True,
            )
        u_all, u_norm_all = self._compute_preconv_tensors(
            hidden_slice=hidden_cat,
            lookup_embeddings=lookup,
        )

        cursor = 0
        for _, req_state, start, end, _ in lookup_segments:
            seg_len = end - start
            u_seg = u_all[cursor : cursor + seg_len]
            u_norm_seg = u_norm_all[cursor : cursor + seg_len]
            conv_out = self._conv_with_state(u_norm=u_norm_seg, req_state=req_state)
            hidden_states[start:end] = hidden_states[start:end] + F.silu(conv_out) + u_seg
            cursor += seg_len
        return hidden_states

    def _decode_forward(
        self,
        hidden_states: torch.Tensor,
        active_segments: list[tuple[str, dict[str, Any], int, int, int]],
        *,
        step_uid: int,
    ) -> torch.Tensor:
        """Execute Engram decode path for active request segments.

        Args:
            self (Any): Module or runtime instance.
            hidden_states (torch.Tensor): Flattened hidden states for current model step.
                Shape: [num_tokens, hidden_size]. Dtype: torch.bfloat16 in Engram path.
            active_segments (list[tuple[str, dict[str, Any], int, int, int]]): Active request segments as (req_id, req_state, start, end, seg_len).
            step_uid (int): Monotonic step identifier for deduplication/prefetch.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        assert self.runtime_state is not None
        batch = len(active_segments)
        start_values = [start for _, _, start, _, _ in active_segments]
        contiguous_decode = all(start == idx for idx, start in enumerate(start_values))
        start_indices: torch.Tensor | None = None
        if contiguous_decode:
            hidden_cat = hidden_states[:batch]
        else:
            # Build start indices in one shot to avoid per-item CUDA writes in Python.
            start_indices = torch.tensor(
                start_values, dtype=torch.long, device=hidden_states.device
            )
            hidden_cat = hidden_states.index_select(0, start_indices)
        req_ids = [req_id for req_id, _, _, _, _ in active_segments]
        req_states = [req_state for _, req_state, _, _, _ in active_segments]

        def _batch_lookup_fallback() -> torch.Tensor:
            """Implement  batch lookup fallback.

            Args:
                None.

            Returns:
                torch.Tensor: Computed result for this helper.
            """
            return self.runtime_state.build_lookup_for_requests_layer_decode(
                req_states=req_states, layer_idx=self.layer_idx
            )

        lookup_cpu_cat = self.runtime_state.consume_decode_prefetched_layer_or_fallback(
            req_ids=req_ids,
            layer_idx=self.layer_idx,
            step_uid=step_uid,
            fallback_fn=_batch_lookup_fallback,
            wait_timeout_s=DEFAULT_ENGRAM_DECODE_PREFETCH_WAIT_S,
        )

        # Decode prefetch may already stage lookup on target CUDA device.
        if (
            lookup_cpu_cat.device == hidden_states.device
            and lookup_cpu_cat.dtype == hidden_states.dtype
        ):
            lookup = lookup_cpu_cat
        elif lookup_cpu_cat.device.type == "cpu":
            lookup = self._copy_lookup_to_device_async(lookup_cpu_cat, hidden_states)
        else:
            lookup = lookup_cpu_cat.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                non_blocking=True,
            )
        u_all, u_norm_all = self._compute_preconv_tensors(
            hidden_slice=hidden_cat,
            lookup_embeddings=lookup,
        )

        conv_out_all = self._conv_decode_batch_with_state(
            u_norm_batch=u_norm_all,
            req_states=req_states,
        )
        conv_out_all = F.silu(conv_out_all).add_(u_all)

        # Wiki invariant: residual update uses silu(conv_out) + u.
        if contiguous_decode:
            hidden_states[:batch].add_(conv_out_all)
        else:
            assert start_indices is not None
            hidden_states.index_add_(0, start_indices, conv_out_all)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run Engram forward by dispatching to decode or prefill path.

        Args:
            self (Any): Module or runtime instance.
            hidden_states (torch.Tensor): Flattened hidden states for current model step.
                Shape: [num_tokens, hidden_size]. Dtype: torch.bfloat16 in Engram path.

        Returns:
            torch.Tensor: Computed result for this helper.
        """
        if hidden_states.numel() == 0:
            return hidden_states
        payload = self._get_engram_step_payload()
        if payload is None:
            return hidden_states
        if self.runtime_state is None:
            return hidden_states
        if (
            self.w_k is None
            or self.w_v is None
            or getattr(self, "_conv", None) is None
        ):
            return hidden_states
        if not hidden_states.is_cuda:
            raise ValueError(
                "EngramLayer requires CUDA hidden_states when Engram is enabled."
            )
        input_ids = payload["input_ids"]
        query_start_loc = payload["query_start_loc"]
        request_ids = payload["request_ids"]
        step_uid = payload["step_uid"]

        active_segments, decode_only = self._collect_active_segments(
            input_ids=input_ids,
            query_start_loc=query_start_loc,
            request_ids=request_ids,
            step_uid=step_uid,
        )
        if not active_segments:
            return hidden_states
        if decode_only:
            return self._decode_forward(
                hidden_states, active_segments, step_uid=step_uid
            )
        return self._prefill_forward(hidden_states, active_segments, step_uid=step_uid)
