# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Facade re-exports for Engram config, runtime, payload, and layer APIs.

from vllm.model_executor.layers.engram_config import (
    EngramHFConfig,
    build_engram_layer_flags,
    parse_and_validate_engram_hf_config,
    validate_engram_runtime_support,
)
from vllm.model_executor.layers.engram_hash import (
    _compute_deepseek_hash_ids_for_sequence,
    _compute_deepseek_hash_ids_last_token,
    compute_deepseek_hash_ids_last_token_batch,
)
from vllm.model_executor.layers.engram_layer import EngramLayer
from vllm.model_executor.layers.engram_payload import (
    ENGRAM_STEP_PAYLOAD_KEY,
    build_engram_step_payload,
    slice_engram_step_payload,
    validate_engram_step_payload,
)
from vllm.model_executor.layers.engram_runtime import EngramRuntimeState

__all__ = [
    "ENGRAM_STEP_PAYLOAD_KEY",
    "EngramHFConfig",
    "EngramLayer",
    "EngramRuntimeState",
    "_compute_deepseek_hash_ids_for_sequence",
    "_compute_deepseek_hash_ids_last_token",
    "build_engram_layer_flags",
    "build_engram_step_payload",
    "compute_deepseek_hash_ids_last_token_batch",
    "parse_and_validate_engram_hf_config",
    "slice_engram_step_payload",
    "validate_engram_runtime_support",
    "validate_engram_step_payload",
]
