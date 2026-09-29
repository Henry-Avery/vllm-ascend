# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU dispatch/bias regressions; native NPU operators are explicit substitutes.

Run with --confcutdir=tests/ut/ops without importing vLLM or NPU extensions.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def forward(monkeypatch):
    source = ROOT / "vllm_ascend/ops/layernorm.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendRMSNorm")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward_oot")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])), str(source), "exec"),
        namespace,
    )
    native = ModuleType("torch_npu")
    native.npu_add_rms_norm = MagicMock()
    native.npu_rms_norm = MagicMock()
    monkeypatch.setitem(sys.modules, "torch_npu", native)
    # Permit the old eager import so the negative control reaches its wrong
    # custom-kernel dispatch even when the capability flag is false.
    package = ModuleType("vllm_ascend")
    extension = ModuleType("vllm_ascend.vllm_ascend_C")
    package.vllm_ascend_C = extension
    monkeypatch.setitem(sys.modules, "vllm_ascend", package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.vllm_ascend_C", extension)
    custom = MagicMock()
    namespace["torch"] = SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_add_rms_norm_bias=custom)))
    namespace["enable_custom_op"] = MagicMock()
    return namespace, native, custom


@pytest.mark.parametrize("custom_enabled", [False, True])
@pytest.mark.parametrize("bias_state", ["none", "allocated", "loaded"])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rmsnorm_dispatch_preserves_bias_and_residual(forward, custom_enabled, bias_state, has_residual, dtype):
    namespace, native, custom = forward
    namespace["enable_custom_op"].return_value = custom_enabled
    x = torch.arange(1, 9, dtype=dtype).reshape(2, 4)
    residual = torch.full_like(x, 0.5) if has_residual else None
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=dtype)
    bias = None if bias_state == "none" else torch.zeros(4, dtype=dtype)
    if bias_state == "loaded":
        bias.copy_(torch.tensor([-1.0, 0.5, 1.0, 2.0], dtype=dtype))
    layer = SimpleNamespace(weight=weight, bias=bias, bias_loaded=bias_state == "loaded", variance_epsilon=1e-6)
    summed = x + residual if residual is not None else x
    # The test substitutes the NPU outputs; the production function must select
    # the correct operator and add bias once without changing the residual.
    normalized = torch.nn.functional.rms_norm(summed, (4,), weight, layer.variance_epsilon)
    expected = normalized + bias if layer.bias_loaded else normalized.clone()
    returned_residual = summed.clone()
    native.npu_add_rms_norm.return_value = (normalized.clone(), None, returned_residual)
    native.npu_rms_norm.return_value = (normalized.clone(), None)
    custom.return_value = (expected.clone(), None, returned_residual)
    actual = namespace["forward_oot"](layer, x, residual)
    actual_x = actual[0] if has_residual else actual
    torch.testing.assert_close(actual_x, expected, rtol=0, atol=0)
    if has_residual:
        assert actual[1] is returned_residual
        torch.testing.assert_close(actual[1], summed, rtol=0, atol=0)
        namespace["enable_custom_op"].assert_called_once_with()
        native.npu_rms_norm.assert_not_called()
        if custom_enabled:
            custom.assert_called_once_with(x, residual, weight, bias, layer.variance_epsilon)
            native.npu_add_rms_norm.assert_not_called()
        else:
            native.npu_add_rms_norm.assert_called_once_with(x, residual, weight, layer.variance_epsilon)
            custom.assert_not_called()
    else:
        native.npu_rms_norm.assert_called_once_with(x, weight, layer.variance_epsilon)
        namespace["enable_custom_op"].assert_not_called()
        native.npu_add_rms_norm.assert_not_called()
        custom.assert_not_called()


def test_enabled_custom_kernel_error_is_not_silently_retried(forward):
    namespace, native, custom = forward
    namespace["enable_custom_op"].return_value = True
    custom.side_effect = RuntimeError("kernel execution failed")
    layer = SimpleNamespace(weight=None, bias=None, bias_loaded=False, variance_epsilon=1e-6)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        namespace["forward_oot"](layer, torch.ones(4), torch.ones(4))
    native.npu_add_rms_norm.assert_not_called()
