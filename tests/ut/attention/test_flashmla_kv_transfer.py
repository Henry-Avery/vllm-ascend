# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for FlashMLA's KV-transfer configuration gate.

Exercise the production validator without importing the NPU engine. These
checks do not validate network transfers or device execution.
"""

from types import SimpleNamespace

import pytest
import torch

from . import test_flashmla_contract as contract
from .test_flashmla_dspark_lifecycle import MLA, load_class

api = contract.api


@pytest.fixture
def impl(api):
    cls = load_class(
        MLA,
        "AscendMLAImpl",
        ["_validate_external_flashmla"],
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
        speculative_config=SimpleNamespace(method="dspark"),
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
