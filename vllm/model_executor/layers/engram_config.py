# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Engram configuration defaults and runtime support validation helpers.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.engram_hash import _build_table_layout


@dataclass(frozen=True)
class EngramHFConfig:
    layer_indices: list[int]
    memory_size: int
    max_ngram_order: int
    heads: int
    mem_dim: int
    compression_ratio: float
    async_workers: int
    conv_kernel: int
    conv_dilation: int
    use_synthetic_weights: bool
    synthetic_seed: int
    vocab_size: int
    pad_token_id: int

def _validate_positive_int(name: str, value: Any) -> int:
    """Validate that a config value is a positive integer.

    Args:
        name: Configuration field name
        value: Configuration field value

    Returns:
        Validated positive integer value
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer, got: {value!r}")
    return value


def _require_field(hf_config: Any, field_name: str) -> Any:
    value = getattr(hf_config, field_name, None)
    if value is None:
        raise ValueError(f"Engram requires HF config field `{field_name}`.")
    return value


def _normalize_engram_layers(raw_layers: Any, num_hidden_layers: int) -> list[int]:
    """Normalize and validate the configured Engram layer indices.

    Args:
        raw_layers: Raw Engram layer list from config
        num_hidden_layers: Number of transformer hidden layers

    Returns:
        Sorted and validated list of Engram layer indices

    Examples:
        Valid normalize/sort behavior:
            raw_layers=[15, 1], num_hidden_layers=30 -> [1, 15]

        Out-of-range rejection:
            raw_layers=[1, 30], num_hidden_layers=30 -> ValueError

        Duplicate rejection:
            raw_layers=[1, 1], num_hidden_layers=30 -> ValueError

        Empty rejection:
            raw_layers=[], num_hidden_layers=30 -> ValueError
    """
    if raw_layers is None:
        raise ValueError("Engram requires HF config field `engram_layers`.")
    if not isinstance(raw_layers, (list, tuple)):
        raise ValueError(
            f"`engram_layers` must be a list/tuple of layer indices, got: {type(raw_layers)}"
        )
    if len(raw_layers) == 0:
        raise ValueError("`engram_layers` must not be empty.")
    if not all(isinstance(layer_idx, int) for layer_idx in raw_layers):
        raise ValueError("`engram_layers` must contain only integers.")

    normalized_layers = sorted(raw_layers)
    if len(set(normalized_layers)) != len(normalized_layers):
        raise ValueError("`engram_layers` must not contain duplicates.")

    min_layer = normalized_layers[0]
    max_layer = normalized_layers[-1]
    if min_layer < 0 or max_layer >= num_hidden_layers:
        raise ValueError(
            "`engram_layers` indices are out of range. "
            f"Expected each index in [0, {num_hidden_layers - 1}], got: {normalized_layers}"
        )

    return normalized_layers


def build_engram_layer_flags(num_hidden_layers: int, layers: list[int]) -> list[bool]:
    """Build a boolean per-layer mask for Engram-enabled layers.

    Args:
        num_hidden_layers: Number of transformer hidden layers
        layers: Sorted Engram layer index list

    Returns:
        Boolean mask indicating which layers have Engram enabled
    """
    flags = [False] * num_hidden_layers
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= num_hidden_layers:
            raise ValueError(
                "`engram_layers` indices are out of range. "
                f"Expected each index in [0, {num_hidden_layers - 1}], got: {layers}"
            )
        flags[layer_idx] = True
    return flags


def parse_and_validate_engram_hf_config(hf_config: Any) -> EngramHFConfig:
    """Parse and validate Engram settings from the HF config object.

    Args:
        hf_config: HF config object carrying Engram override fields

    Returns:
        Validated Engram configuration object
    """
    engram_enable = bool(getattr(hf_config, "engram_enable", False))
    if not engram_enable:
        return EngramHFConfig(
            layer_indices=[],
            memory_size=0,
            max_ngram_order=0,
            heads=0,
            mem_dim=0,
            compression_ratio=0.0,
            async_workers=0,
            conv_kernel=0,
            conv_dilation=0,
            use_synthetic_weights=False,
            synthetic_seed=0,
            vocab_size=_validate_positive_int("vocab_size", getattr(hf_config, "vocab_size", 1)),
            pad_token_id=int(getattr(hf_config, "pad_token_id", 0) or 0),
        )

    num_hidden_layers = _validate_positive_int(
        "num_hidden_layers", _require_field(hf_config, "num_hidden_layers")
    )
    layers = _normalize_engram_layers(
        _require_field(hf_config, "engram_layers"),
        num_hidden_layers,
    )
    memory_size = _validate_positive_int(
        "engram_memory_size",
        _require_field(hf_config, "engram_memory_size"),
    )
    max_ngram_order = _validate_positive_int(
        "engram_max_ngram_order",
        _require_field(hf_config, "engram_max_ngram_order"),
    )
    if max_ngram_order < 2:
        raise ValueError("`engram_max_ngram_order` must be >= 2.")
    if memory_size < max_ngram_order:
        raise ValueError(
            "`engram_memory_size` must be >= `engram_max_ngram_order`, "
            f"got memory_size={memory_size}, max_ngram_order={max_ngram_order}."
        )
    heads = _validate_positive_int(
        "engram_heads",
        _require_field(hf_config, "engram_heads"),
    )
    mem_dim = _validate_positive_int(
        "engram_mem_dim",
        _require_field(hf_config, "engram_mem_dim"),
    )
    _build_table_layout(
        max_ngram_order=max_ngram_order,
        heads=heads,
        mem_dim=mem_dim,
    )

    compression_ratio = float(_require_field(hf_config, "engram_compression_ratio"))
    if not (0.0 < compression_ratio <= 1.0):
        raise ValueError(
            "`engram_compression_ratio` must be in range (0, 1], "
            f"got: {compression_ratio!r}"
        )
    async_workers = _validate_positive_int(
        "engram_async_workers",
        _require_field(hf_config, "engram_async_workers"),
    )
    conv_kernel = _validate_positive_int(
        "engram_conv_kernel",
        _require_field(hf_config, "engram_conv_kernel"),
    )
    conv_dilation = _validate_positive_int(
        "engram_conv_dilation",
        _require_field(hf_config, "engram_conv_dilation"),
    )
    use_synthetic_weights = bool(
        getattr(hf_config, "engram_use_synthetic_weights", True)
    )
    if not use_synthetic_weights:
        raise ValueError(
            "Engram v2 currently supports synthetic Engram weights only. "
            "Set `engram_use_synthetic_weights=true`."
        )
    synthetic_seed = int(_require_field(hf_config, "engram_synthetic_seed"))
    vocab_size = _validate_positive_int(
        "vocab_size",
        _require_field(hf_config, "vocab_size"),
    )
    pad_token_id = int(getattr(hf_config, "pad_token_id", 0) or 0)

    return EngramHFConfig(
        layer_indices=layers,
        memory_size=memory_size,
        max_ngram_order=max_ngram_order,
        heads=heads,
        mem_dim=mem_dim,
        compression_ratio=compression_ratio,
        async_workers=async_workers,
        conv_kernel=conv_kernel,
        conv_dilation=conv_dilation,
        use_synthetic_weights=use_synthetic_weights,
        synthetic_seed=synthetic_seed,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
    )


def validate_engram_runtime_support(vllm_config: Any) -> None:
    """Validate runtime constraints required by Engram execution.

    Args:
        vllm_config: vLLM runtime configuration object
    """
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    compilation_config = getattr(vllm_config, "compilation_config", None)

    model_dtype = getattr(model_config, "dtype", None)
    model_dtype_name = str(model_dtype).lower()
    if model_dtype not in {torch.bfloat16} and model_dtype_name not in {
        "bf16",
        "bfloat16",
        "torch.bfloat16",
    }:
        raise ValueError(f"Engram v1 only supports BF16. Got dtype={model_dtype}.")
    if getattr(parallel_config, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Engram v1 does not support pipeline parallelism.")
    if bool(getattr(parallel_config, "enable_expert_parallel", False)):
        raise ValueError("Engram v1 does not support expert parallelism.")
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError("Engram v1 does not support speculative decoding.")
    enforce_eager = bool(
        getattr(model_config, "enforce_eager", getattr(vllm_config, "enforce_eager", False))
    )
    if not enforce_eager:
        raise ValueError("Engram v1 requires eager execution.")
    if str(getattr(compilation_config, "cudagraph_mode", "NONE")).upper() != "NONE":
        raise ValueError("Engram v1 does not support CUDA graph mode.")
    compile_level = getattr(compilation_config, "level", 0)
    if compile_level is not None and int(compile_level) != 0:
        raise ValueError("Engram v1 does not support torch.compile mode.")
