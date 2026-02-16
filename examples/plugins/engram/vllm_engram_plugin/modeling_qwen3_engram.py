# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Engram-enabled Qwen3 model wrapper and runtime initialization glue.

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.layers.engram import (
    EngramRuntimeState,
    parse_and_validate_engram_hf_config,
    validate_engram_runtime_support,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM


class Qwen3EngramForCausalLM(Qwen3ForCausalLM):
    """Top-level architecture class for Engram-enabled Qwen3."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        """Implement   init  .

        Args:
            self (Any): Module or runtime instance.
            vllm_config (VllmConfig): vLLM runtime configuration object.
            prefix (str): Optional module prefix string.

        Returns:
            Any: Computed result for this helper.
        """
        hf_config = vllm_config.model_config.hf_config
        hf_config.engram_enable = bool(getattr(hf_config, "engram_enable", False))
        hf_config.engram_layers = list(getattr(hf_config, "engram_layers", []))
        if hf_config.engram_enable:
            validate_engram_runtime_support(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.engram_runtime_state: EngramRuntimeState | None = None
        if not hf_config.engram_enable:
            return

        engram_cfg = parse_and_validate_engram_hf_config(hf_config)

        runtime_state = EngramRuntimeState(
            memory_size=engram_cfg.memory_size,
            max_ngram_order=engram_cfg.max_ngram_order,
            heads=engram_cfg.heads,
            mem_dim=engram_cfg.mem_dim,
            vocab_size=engram_cfg.vocab_size,
            compression_ratio=engram_cfg.compression_ratio,
            conv_kernel=engram_cfg.conv_kernel,
            conv_dilation=engram_cfg.conv_dilation,
            pad_token_id=engram_cfg.pad_token_id,
            async_workers=engram_cfg.async_workers,
            engram_layer_indices=tuple(engram_cfg.layer_indices),
        )
        runtime_state.init_async_executor(engram_cfg.async_workers)
        runtime_state.init_synthetic_compression_map(engram_cfg.synthetic_seed)
        for layer_idx in engram_cfg.layer_indices:
            runtime_state.init_synthetic_layer_host_tables(
                layer_idx=layer_idx,
                seed=engram_cfg.synthetic_seed,
            )

        for layer in self.model.layers:
            if not hasattr(layer, "engram") or layer.engram is None:
                continue
            layer.engram.runtime_state = runtime_state
            layer.engram.init_synthetic_layer_parameters(engram_cfg.synthetic_seed)

        self.engram_runtime_state = runtime_state
