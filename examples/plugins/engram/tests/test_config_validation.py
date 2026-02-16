# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Validation tests for Engram HF config parsing and layer selection helpers.

from types import SimpleNamespace

import pytest

from vllm.model_executor.models.qwen3 import maybe_create_engram_layer
from vllm.model_executor.layers.engram import (
    build_engram_layer_flags,
    parse_and_validate_engram_hf_config,
)


def _make_cfg(**kwargs):
    """@brief Implement  make cfg.

    Args:
        **kwargs (Any): Kwargs input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = {
        "num_hidden_layers": 30,
        "hidden_size": 2048,
        "vocab_size": 151936,
        "engram_enable": True,
        "engram_layers": [1, 15],
        "engram_memory_size": 8192,
        "engram_max_ngram_order": 4,
        "engram_heads": 8,
        "engram_mem_dim": 1008,
        "engram_compression_ratio": 0.8,
        "engram_conv_kernel": 4,
        "engram_conv_dilation": 4,
        "engram_async_workers": 1,
        "engram_use_synthetic_weights": True,
        "engram_synthetic_seed": 2026,
    }
    cfg.update(kwargs)
    return SimpleNamespace(**cfg)


def test_parse_valid_engram_config():
    """@brief Validate that parse valid engram config.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    parsed = parse_and_validate_engram_hf_config(_make_cfg())
    assert parsed.layer_indices == [1, 15]
    assert parsed.memory_size == 8192
    assert parsed.max_ngram_order == 4
    assert parsed.heads == 8
    assert parsed.mem_dim == 1008
    assert parsed.async_workers == 1


def test_layer_indices_must_be_in_bounds():
    """@brief Validate that layer indices must be in bounds.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="out of range"):
        parse_and_validate_engram_hf_config(_make_cfg(engram_layers=[1, 30]))


def test_missing_engram_layers_when_enabled():
    """@brief Validate that missing engram layers when enabled.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="engram_layers"):
        parse_and_validate_engram_hf_config(_make_cfg(engram_layers=None))


def test_build_engram_layer_flags():
    """@brief Validate that build engram layer flags.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    flags = build_engram_layer_flags(30, [1, 15])
    assert len(flags) == 30
    assert flags[1] is True
    assert flags[15] is True
    assert flags[0] is False


def test_build_engram_layer_flags_rejects_invalid_index():
    """@brief Validate that build engram layer flags rejects invalid index.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="out of range"):
        build_engram_layer_flags(30, [30])


def test_disabled_engram_returns_empty_layers():
    """@brief Validate that disabled engram returns empty layers.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    parsed = parse_and_validate_engram_hf_config(_make_cfg(engram_enable=False))
    assert parsed.layer_indices == []


@pytest.mark.parametrize(
    "field",
    [
        "engram_memory_size",
        "engram_max_ngram_order",
        "engram_heads",
        "engram_mem_dim",
        "engram_async_workers",
        "engram_conv_kernel",
        "engram_conv_dilation",
    ],
)
def test_invalid_positive_fields(field: str):
    """@brief Validate that invalid positive fields.

    Args:
        field (str): Field input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="positive integer"):
        parse_and_validate_engram_hf_config(_make_cfg(**{field: 0}))


def test_memory_size_must_be_at_least_max_ngram_order():
    """@brief Validate that memory size must be at least max ngram order.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(
        ValueError,
        match="engram_memory_size.*engram_max_ngram_order",
    ):
        parse_and_validate_engram_hf_config(
            _make_cfg(engram_memory_size=3, engram_max_ngram_order=4)
        )


def test_compression_ratio_must_be_in_valid_range():
    """@brief Validate that compression ratio must be in valid range.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="compression_ratio"):
        parse_and_validate_engram_hf_config(_make_cfg(engram_compression_ratio=0.0))


def test_mem_dim_must_cover_all_tables():
    """@brief Validate that mem dim must divide evenly across table count.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="divisible by number of tables"):
        parse_and_validate_engram_hf_config(
            _make_cfg(engram_mem_dim=8, engram_heads=8, engram_max_ngram_order=4)
        )


def test_mem_dim_must_be_divisible_by_num_tables():
    """@brief Validate that mem dim must divide evenly across all tables.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="divisible by number of tables"):
        parse_and_validate_engram_hf_config(
            _make_cfg(engram_mem_dim=1025, engram_heads=8, engram_max_ngram_order=4)
        )


def test_non_synthetic_weights_are_rejected():
    """@brief Validate that non-synthetic Engram weights are rejected.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    with pytest.raises(ValueError, match="synthetic Engram weights only"):
        parse_and_validate_engram_hf_config(
            _make_cfg(engram_use_synthetic_weights=False)
        )


def test_qwen3_helper_populates_layer_flag_cache_via_layer_creation():
    """@brief Validate that qwen3 helper populates layer flag cache via layer creation.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_cfg()
    assert getattr(cfg, "_vllm_engram_layer_flags_cache", None) is None
    layer = maybe_create_engram_layer(cfg, layer_idx=1)
    assert layer is not None
    flags = getattr(cfg, "_vllm_engram_layer_flags_cache", None)
    assert flags is not None
    assert len(flags) == 30
    assert sum(flags) == 2
    assert flags[1] is True
    assert flags[15] is True


def test_qwen3_helper_returns_none_layer_when_engram_disabled():
    """@brief Validate that qwen3 helper returns none layer when engram disabled.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    assert maybe_create_engram_layer(_make_cfg(engram_enable=False), layer_idx=1) is None


def test_qwen3_helper_creates_engram_layer_only_for_configured_indices():
    """@brief Validate that qwen3 helper creates engram layer only for configured indices.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    layer_1 = maybe_create_engram_layer(_make_cfg(), layer_idx=1)
    layer_14 = maybe_create_engram_layer(_make_cfg(), layer_idx=14)
    layer_15 = maybe_create_engram_layer(_make_cfg(), layer_idx=15)

    assert layer_1 is not None
    assert layer_14 is None
    assert layer_15 is not None


def test_qwen3_helper_respects_disable_after_cache_population():
    """@brief Validate that qwen3 helper respects disable after cache population.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    cfg = _make_cfg()
    assert maybe_create_engram_layer(cfg, layer_idx=1) is not None
    assert getattr(cfg, "_vllm_engram_layer_flags_cache", None) is not None
    assert getattr(cfg, "_vllm_engram_hf_config_cache", None) is not None

    cfg.engram_enable = False
    assert maybe_create_engram_layer(cfg, layer_idx=1) is None
    assert getattr(cfg, "_vllm_engram_layer_flags_cache", None) is None
    assert getattr(cfg, "_vllm_engram_hf_config_cache", None) is None
