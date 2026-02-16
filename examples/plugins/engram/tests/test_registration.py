# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Plugin registration tests for Engram architecture wiring.

import importlib
import pathlib
import sys
from types import SimpleNamespace

import pytest
from vllm import ModelRegistry

PLUGIN_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def _resolve_qualname(qualname: str):
    """@brief Implement  resolve qualname.

    Args:
        qualname (str): Qualname input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    module_name, class_name = qualname.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


@pytest.fixture
def clear_engram_arch_registration():
    """@brief Implement clear engram arch registration.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    arch = "Qwen3EngramForCausalLM"
    original = ModelRegistry.models.pop(arch, None)
    try:
        yield
    finally:
        if original is not None:
            ModelRegistry.models[arch] = original
        else:
            ModelRegistry.models.pop(arch, None)


def test_engram_plugin_registers_architecture(clear_engram_arch_registration):
    """@brief Validate that engram plugin registers architecture.

    Args:
        clear_engram_arch_registration (Any): Clear engram arch registration input.
            Shape: N/A. Dtype: N/A unless tensor.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    from vllm_engram_plugin import register

    assert "Qwen3EngramForCausalLM" not in ModelRegistry.get_supported_archs()

    register()

    assert "Qwen3EngramForCausalLM" in ModelRegistry.get_supported_archs()


def test_registered_qualname_is_importable():
    """@brief Validate that registered qualname is importable.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    from vllm_engram_plugin.register import ENGRAM_QUALNAME

    resolved = _resolve_qualname(ENGRAM_QUALNAME)

    assert resolved.__name__ == "Qwen3EngramForCausalLM"


def test_engram_default_field_normalization_behavior():
    """@brief Validate normalized default behavior for Engram activation fields.

    Args:
        None.

    Returns:
        Any: Computed result for this helper.
            Shape: N/A unless otherwise specified. Dtype: N/A unless tensor.
    """
    hf_config = SimpleNamespace(
        num_hidden_layers=61,
        hidden_size=7168,
        num_attention_heads=56,
        intermediate_size=28672,
    )
    hf_config.engram_enable = bool(getattr(hf_config, "engram_enable", False))
    hf_config.engram_layers = list(getattr(hf_config, "engram_layers", []))

    assert hf_config.num_hidden_layers == 61
    assert hf_config.hidden_size == 7168
    assert hf_config.num_attention_heads == 56
    assert hf_config.intermediate_size == 28672
    assert hf_config.engram_enable is False
    assert hf_config.engram_layers == []
