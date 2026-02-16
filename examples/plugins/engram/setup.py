# SPDX-License-Identifier: Apache-2.0
# Created: 2026-02-18
# Author: Liu Yiyang
# Purpose: Package configuration for the Engram example plugin.

from setuptools import setup

setup(
    name="vllm_engram_plugin",
    version="0.1.0",
    packages=["vllm_engram_plugin"],
    install_requires=["vllm"],
    entry_points={
        "vllm.general_plugins": [
            "register_engram_plugin = vllm_engram_plugin:register",
        ]
    },
)
