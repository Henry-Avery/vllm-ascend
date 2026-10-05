# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for FlashMLA's KV-transfer configuration gate.

Exercise the production validator without importing the NPU engine. These
checks do not validate network transfers or device execution.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from . import test_flashmla_contract as contract

api = contract.api


def _load_validator(namespace):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    method = next(node for node in original.body if getattr(node, "name", None) == "_validate_external_flashmla")
    cls = ast.ClassDef(name="AscendMLAImpl", bases=[], keywords=[], body=[method], decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["AscendMLAImpl"]


@pytest.fixture
def impl(api):
    cls = _load_validator(
        dict(
            torch=torch,
            HardwareCapability=SimpleNamespace(MLA_FLASH="MLA_FLASH"),
            get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: True),
            MLA_FLASH_SUPPORTED_Q_HEADS=api.MLA_FLASH_SUPPORTED_Q_HEADS,
            FLASHMLA_QK_DIM=api.FLASHMLA_QK_DIM,
            FLASHMLA_V_DIM=api.FLASHMLA_V_DIM,
        ),
    )
    result = cls()
    result.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        kv_transfer_config=SimpleNamespace(kv_connector="MooncakeConnectorV2", kv_role="kv_consumer"),
    )
    result.num_heads, result.num_kv_heads = 64, 1
    result.kv_lora_rank, result.qk_rope_head_dim = 512, 64
    result.dtype = torch.bfloat16
    result.fa_quant_layer = result.enable_kv_nz = result.pcp_enabled = False
    return result


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flashmla_accepts_colocated_and_mooncake_v2_roles(impl, role, dtype):
    impl.dtype = dtype
    impl.vllm_config.kv_transfer_config = (
        SimpleNamespace(kv_connector="MooncakeConnectorV2", kv_role=role) if role is not None else None
    )
    impl._validate_external_flashmla()


@pytest.mark.parametrize(
    "attribute, value, message",
    [
        ("num_heads", 16, "local Q heads"),
        ("kv_lora_rank", 256, "latent512"),
        ("dtype", torch.float32, "BF16/FP16"),
        ("fa_quant_layer", True, "unquantized"),
        ("enable_kv_nz", True, "BBND cache"),
        ("pcp_enabled", True, "PCP=1 and DCP=1"),
    ],
)
def test_kv_transfer_does_not_bypass_flashmla_input_constraints(impl, attribute, value, message):
    setattr(impl, attribute, value)
    with pytest.raises(ValueError, match=message):
        impl._validate_external_flashmla()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kv_transfer_admission_does_not_whitelist_connector_names(impl, dtype):
    # Admission is distinct from validating a connector's physical cache layout.
    impl.dtype = dtype
    impl.vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector="CustomConnector", kv_connector_module_path="custom.connector", kv_role="kv_consumer"
    )
    impl._validate_external_flashmla()
