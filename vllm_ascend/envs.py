#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # Opt in to external FlashMLA for supported dense MLA decode layers.
    # Prefill retains FIA and the existing BBND cache. Default: 0 (disabled).
    # Valid values: 0 or 1. Not sensitive. Unsupported configurations fail early.
    "VLLM_ASCEND_ENABLE_FLASH_MLA": lambda: _strict_binary_env("VLLM_ASCEND_ENABLE_FLASH_MLA"),
    # Diagnostic logging for FlashMLA metadata and graph submission. Default: 0.
    # Valid: 0/1. No tensor contents or credentials; emits local device addresses.
    # Debug only: per-step host logging affects performance; disable for benchmarks.
    "VLLM_ASCEND_FLASH_MLA_TRACE": lambda: _strict_binary_env("VLLM_ASCEND_FLASH_MLA_TRACE"),
    # Default 0; strict 0/1. Add bounded eager attention/KV references to SAMPLE_DIAG.
    # Requires SAMPLE_DIAG_DIR and the same selected ranks. Not a credential;
    # captured request/tensor data is private. Deliberately synchronizes the hot path.
    "VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG": lambda: _strict_binary_env("VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG"),
    # Default {}. JSON: max_layers=2 (1..8), requests_per_phase=2 (1..8),
    # query_rows=2 (1..4), max_kv_tokens=4096 (1..16384), max_saved_mib=128 (1..1024),
    # layer_names=[] (<=8), request_ids=[] (<=16), atol/rtol=0.05 (finite 0..1).
    # start_after_forwards=0 (0..1000000), scan_all_mla_layers=false (JSON boolean).
    # Only matching forwards after the delay consume the attention/sampling budget.
    # Names/IDs are optional exact-match filters, not credentials. Diagnostic only.
    "VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG_CONFIG": lambda: os.getenv("VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG_CONFIG", "{}"),
    # Eager-only numerical sampling diagnostics. Default empty disables all hooks.
    # Nonempty path enables private per-rank reports; artifacts contain request
    # IDs/token IDs/logits (sensitive run data), never credentials. Debug copies
    # synchronize selected workers and must not be used for performance results.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR": lambda: os.getenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR", ""),
    # Default 64, valid 1..128 real sampling calls per worker. Not sensitive.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS": lambda: int(os.getenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS", "64")),
    # Default 2, valid 1..8 full-logits rows saved per call. Not sensitive.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS": lambda: int(os.getenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS", "2")),
    # Default -1 selects every DP group; otherwise a configured DP rank. Not sensitive.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK": lambda: int(
        os.getenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK", "-1")
    ),
    # Default 0, valid nonnegative configured TP rank in each selected DP. Not sensitive.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK": lambda: int(
        os.getenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK", "0")
    ),
    # Default "0", comma-separated 1..16 nonnegative vocabulary IDs. Not sensitive.
    # IDs must be verified with the deployed tokenizer; 0 is not assumed to mean !.
    "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS": lambda: os.getenv(
        "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS", "0"
    ),
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
