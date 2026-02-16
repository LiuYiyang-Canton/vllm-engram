# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Engram forward payload build, validation, and slicing utilities.

from __future__ import annotations

from typing import Any

import torch

ENGRAM_STEP_PAYLOAD_KEY = "engram_step"

def build_engram_step_payload(
    step_uid: int,
    input_ids: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    request_ids: list[str] | tuple[str, ...] | None,
) -> dict[str, Any] | None:
    """Build and validate an Engram step payload dictionary.

    Args:
        step_uid (int): Monotonic step identifier for deduplication/prefetch.
        input_ids (torch.Tensor | None): Flattened token IDs before embedding lookup.
            Shape: [num_step_tokens]. Dtype: torch.int64.
        query_start_loc (torch.Tensor | None): Prefix-sum token offsets per request.
            Shape: [num_requests + 1]. Dtype: torch.int32 on CPU.
        request_ids (list[str] | tuple[str, ...] | None): Request identifiers aligned with query_start_loc.
            Shape: [num_requests]. Dtype: list[str].

    Returns:
        dict[str, Any] | None: Computed result for this helper.
    """
    if input_ids is None or query_start_loc is None or request_ids is None:
        return None
    payload = {
        "step_uid": int(step_uid),
        "input_ids": input_ids,
        "query_start_loc": query_start_loc,
        "request_ids": list(request_ids),
    }
    validate_engram_step_payload(payload)
    return payload


def validate_engram_step_payload(payload: Any) -> None:
    """Validate schema, types, and shape constraints for an Engram payload.

    Args:
        payload (Any): Engram step payload dictionary.

    Returns:
        None: Function returns no value.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            "`engram_step` payload must be a dict with step/token metadata."
        )

    required_keys = ("step_uid", "input_ids", "query_start_loc", "request_ids")
    for key in required_keys:
        if key not in payload:
            raise ValueError(f"`engram_step` payload missing required field `{key}`.")

    step_uid = payload["step_uid"]
    if not isinstance(step_uid, int):
        raise ValueError(
            f"`engram_step.step_uid` must be int, got: {type(step_uid).__name__}."
        )

    input_ids = payload["input_ids"]
    query_start_loc = payload["query_start_loc"]
    request_ids = payload["request_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("`engram_step.input_ids` must be a torch.Tensor.")
    if not isinstance(query_start_loc, torch.Tensor):
        raise ValueError("`engram_step.query_start_loc` must be a torch.Tensor.")
    if not isinstance(request_ids, (list, tuple)):
        raise ValueError("`engram_step.request_ids` must be a list or tuple of request IDs.")
    if input_ids.ndim != 1:
        raise ValueError("`engram_step.input_ids` must be a 1D tensor.")
    if input_ids.device.type != "cpu":
        raise ValueError("`engram_step.input_ids` must be on CPU.")
    if input_ids.dtype != torch.int64:
        raise ValueError("`engram_step.input_ids` must have dtype torch.int64.")
    if query_start_loc.ndim != 1:
        raise ValueError("`engram_step.query_start_loc` must be a 1D tensor.")
    if query_start_loc.numel() == 0:
        raise ValueError("`engram_step.query_start_loc` must not be empty.")
    if query_start_loc.device.type != "cpu":
        raise ValueError("`engram_step.query_start_loc` must be on CPU.")
    if query_start_loc.dtype != torch.int32:
        raise ValueError("`engram_step.query_start_loc` must have dtype torch.int32.")

    if query_start_loc[0] != 0:
        raise ValueError("`engram_step.query_start_loc` must start at 0.")
    if not torch.all(query_start_loc[1:] >= query_start_loc[:-1]).item():
        raise ValueError("`engram_step.query_start_loc` must be non-decreasing.")

    num_reqs = len(request_ids)
    if query_start_loc.numel() != num_reqs + 1:
        raise ValueError(
            "`engram_step` shape mismatch: len(query_start_loc) must equal "
            "len(request_ids) + 1."
        )
    total_tokens = query_start_loc[-1].item()
    if total_tokens != input_ids.numel():
        raise ValueError(
            "`engram_step` shape mismatch: query_start_loc[-1] must equal "
            "len(input_ids)."
        )

def slice_engram_step_payload(
    payload: dict[str, Any], request_slice: slice, token_slice: slice
) -> dict[str, Any]:
    """Slice an Engram payload for ubatching and rebase offsets.

    Args:
        payload (dict[str, Any]): Engram step payload dictionary.
        request_slice (slice): Slice selecting requests from payload.
        token_slice (slice): Slice selecting token range from payload input_ids.

    Returns:
        dict[str, Any]: Computed result for this helper.
    """
    validate_engram_step_payload(payload)
    request_ids = payload["request_ids"][request_slice]
    start = request_slice.start if request_slice.start is not None else 0
    stop = request_slice.stop if request_slice.stop is not None else len(
        payload["request_ids"]
    )
    query_start_loc = payload["query_start_loc"][start : stop + 1].clone()
    query_start_loc = query_start_loc - query_start_loc[0]
    sliced = {
        "step_uid": payload["step_uid"],
        "input_ids": payload["input_ids"][token_slice],
        "query_start_loc": query_start_loc,
        "request_ids": list(request_ids),
    }
    validate_engram_step_payload(sliced)
    return sliced
