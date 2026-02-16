# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Plugin registration entrypoint for Qwen3 Engram architecture.

from vllm import ModelRegistry

ENGRAM_ARCH = "Qwen3EngramForCausalLM"
ENGRAM_QUALNAME = (
    "vllm_engram_plugin.modeling_qwen3_engram:Qwen3EngramForCausalLM"
)


def register() -> None:
    """Register the Engram Qwen3 architecture with vLLM plugin system.

    Args:
        None.

    Returns:
        None: Function returns no value.
    """
    if ENGRAM_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ENGRAM_ARCH, ENGRAM_QUALNAME)
