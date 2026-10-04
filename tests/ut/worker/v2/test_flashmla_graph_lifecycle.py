# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the actual scope and capability gate, without NPU claims."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


def _load_function(path, name, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    node = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace[name]


@pytest.mark.parametrize("fail", [False, True])
def test_scope_owns_release_and_restores_builder_after_nested_consumers(fail):
    trace = []
    state = SimpleNamespace(executor=None, defer=False)
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: SimpleNamespace(flashmla_state=state))]]
    executor = SimpleNamespace(submission_in_flight=True, release=lambda: trace.append("release"))
    scope = _load_function(
        "vllm_ascend/worker/v2/attn_utils.py", "flashmla_metadata_scope", {"contextmanager": contextmanager}
    )
    try:
        with scope(groups, executor):
            assert state.executor is executor and state.defer
            with scope(groups, executor):
                trace.append("consumer")
            assert trace == ["consumer"]
            if fail:
                raise ValueError("consumer failed")
    except ValueError:
        assert fail
    assert trace == ["consumer", "release"]
    assert state.executor is None and not state.defer


def test_disabled_scope_does_not_inspect_uninitialized_builders():
    scope = _load_function(
        "vllm_ascend/worker/v2/attn_utils.py", "flashmla_metadata_scope", {"contextmanager": contextmanager}
    )
    with scope(None, None):
        pass


@pytest.mark.parametrize(
    "change, message",
    [
        ({}, None),
        ({"use_v2_model_runner": False}, "model runner V2"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "without KV transfer"),
        ({"decode_context_parallel_size": 2}, "DCP=1"),
        ({"enable_kv_nz": True}, "BBND"),
        ({"num_kv_heads": 2}, "one KV head"),
        ({"pcp_enabled": True}, "PCP=1"),
    ],
)
def test_target_support_gate_rejects_unwired_modes(change, message):
    gate = _load_function(
        "vllm_ascend/attention/mla_v1.py",
        "_validate_external_flashmla",
        dict(
            torch=torch,
            MLA_FLASH_SUPPORTED_Q_HEADS=(8, 12, 64, 96),
            FLASHMLA_QK_DIM=576,
            FLASHMLA_V_DIM=512,
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: True),
            HardwareCapability=SimpleNamespace(MLA_FLASH="flash"),
        ),
        class_name="AscendMLAImpl",
    )
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=64,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
    )
    for name, value in change.items():
        target = (
            config.parallel_config
            if name == "decode_context_parallel_size"
            else config
            if hasattr(config, name)
            else impl
        )
        setattr(target, name, value)
    if message:
        with pytest.raises(ValueError, match=message):
            gate(impl)
    else:
        gate(impl)
