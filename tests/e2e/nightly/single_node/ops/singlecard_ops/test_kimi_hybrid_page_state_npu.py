# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual padded-page conv/prefill/recurrent/COW/zero integration on NPU."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.hardware_profile import HardwareCapability
from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.utils import AscendV2KVBlockZeroer

enable_custom_op()

HEADS = 12
HEAD_DIM = 128
CONV_WIDTH = 4
CHANNELS = 3 * HEADS * HEAD_DIM
NUM_BLOCKS = 8
KERNEL_BLOCK_SIZE = 128


def _allocate_hybrid(monkeypatch, manager_block_size):
    state_shapes = ((CONV_WIDTH - 1, CHANNELS), (HEADS, HEAD_DIM, HEAD_DIM))
    state_bytes = (CONV_WIDTH - 1) * CHANNELS * 2 + HEADS * HEAD_DIM * HEAD_DIM * 4
    page_bytes = max(state_bytes, manager_block_size * 576 * 2)
    mla_spec = AscendMLAAttentionSpec(
        block_size=manager_block_size,
        num_kv_heads=1,
        head_size=576,
        num_query_heads=HEADS,
        dtype=torch.bfloat16,
        page_size_padded=page_bytes,
    )
    mamba_spec = MambaSpec(
        block_size=manager_block_size,
        shapes=state_shapes,
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=page_bytes,
    )
    groups = [
        KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=mla_spec),
        KVCacheGroupSpec(layer_names=["kda"], kv_cache_spec=mamba_spec),
    ]
    # Nonzero raw offsets, with prefix and suffix bytes outside all views.
    layer_bytes = NUM_BLOCKS * page_bytes
    descriptors = [
        KVCacheTensor(
            size=layer_bytes + 128, layers=[name], layer_stride=layer_bytes, block_stride=page_bytes, offset=64
        )
        for name in ("mla", "kda")
    ]
    cache_config = KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    config = SimpleNamespace(kv_transfer_config=None, additional_config={}, model_config=None)
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = SimpleNamespace(fa_quant_layer=False)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 512, 64
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attn_utils, "get_layers_from_vllm_config", lambda *_args: {"mla": layer})
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _: False)
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: capability == HardwareCapability.MLA_FLASH),
    )
    raw = attn_utils._allocate_kv_cache(cache_config, {}, torch.device("npu"))
    attn_groups = [
        AttentionGroup(
            backend=AscendMLABackend,
            layer_names=group.layer_names,
            kv_cache_spec=group.kv_cache_spec,
            kv_cache_group_id=index,
        )
        for index, group in enumerate(groups)
    ]
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups, raw, "auto", [KERNEL_BLOCK_SIZE, manager_block_size], {}, cache_config
    )
    return raw, caches, mla_spec, attn_groups


@pytest.mark.parametrize("manager_block_size", [128, 384, 768])
@torch.inference_mode()
def test_hybrid_selected_state_ops_preserve_live_mla_pages(monkeypatch, manager_block_size):
    """Compare padded states to dense references, with real device operators."""
    torch.manual_seed(29)
    raw, caches, mla_spec, groups = _allocate_hybrid(monkeypatch, manager_block_size)
    conv, recurrent = caches["kda"]
    mla = caches["mla"]
    ratio = manager_block_size // KERNEL_BLOCK_SIZE
    mla_guard = mla[4 * ratio : 5 * ratio]
    mla_guard.fill_(1)
    guard_before = mla_guard.clone()
    before_state_updates = raw["mla"].cpu().clone()
    assert conv.storage_offset() > 0 and recurrent.storage_offset() > 0
    assert conv.stride(0) * conv.element_size() == mla_spec.page_size_bytes
    assert recurrent.stride(0) * recurrent.element_size() == mla_spec.page_size_bytes
    ids = torch.tensor([3, 1], dtype=torch.int32, device="npu")
    flags = torch.tensor([True, False], device="npu")
    query_start = torch.tensor([0, 2, 5], dtype=torch.int32, device="npu")
    weights = torch.randn(CONV_WIDTH, CHANNELS, dtype=torch.bfloat16, device="npu") * 0.01
    mixed = torch.randn(5, CHANNELS, dtype=torch.bfloat16, device="npu") * 0.1
    dense_conv = conv.clone()
    output = AscendKimiK3DeltaAttention._run_causal_conv1d(
        mixed.clone(), weights, conv, query_start, ids, flags, run_mode=0
    )
    reference = AscendKimiK3DeltaAttention._run_causal_conv1d(
        mixed.clone(), weights, dense_conv, query_start, ids, flags, run_mode=0
    )
    torch.testing.assert_close(output, reference, rtol=0.02, atol=0.02)
    torch.testing.assert_close(conv, dense_conv, rtol=0, atol=0)
    decode_input = mixed[:2].clone()
    decode_start = torch.tensor([0, 1, 2], dtype=torch.int32, device="npu")
    for pool in (conv, dense_conv):
        decoded = AscendKimiK3DeltaAttention._run_causal_conv1d(
            decode_input.clone(), weights, pool, decode_start, ids, None, run_mode=1, max_query_len=1
        )
        assert torch.isfinite(decoded).all()
    torch.testing.assert_close(conv, dense_conv, rtol=0, atol=0)

    attention = SimpleNamespace(
        A_log=torch.zeros(HEADS, dtype=torch.float32, device="npu"),
        dt_bias=torch.zeros(HEADS * HEAD_DIM, dtype=torch.float32, device="npu"),
        gate_lower_bound=-5.0,
    )
    q, k, v = [part.reshape(1, 5, HEADS, HEAD_DIM) for part in output.chunk(3, dim=-1)]
    gate = torch.zeros_like(q)
    beta = torch.full((1, 5, HEADS), 0.5, dtype=torch.float32, device="npu")
    metadata = SimpleNamespace(
        cu_seqlens_host=(0, 2, 5), cu_seqlens_kern=None, keep_meta=None, chunk_indices_chunk64_host=(0, 0, 1, 0)
    )
    dense_state = recurrent.clone()
    outputs = [
        AscendKimiK3DeltaAttention._run_prefill(attention, q, k, v, gate, beta, pool, ids, flags, metadata)
        for pool in (recurrent, dense_state)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.02)
    torch.testing.assert_close(recurrent, dense_state, rtol=0.02, atol=0.02)
    assert torch.isfinite(outputs[0]).all()
    decode_args = (attention, q[:, :2], k[:, :2], v[:, :2], gate[:, :2], beta[:, :2])
    outputs = [
        AscendKimiK3DeltaAttention._run_recurrent(*decode_args, pool, decode_start, ids)
        for pool in (recurrent, dense_state)
    ]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.02, atol=0.02)
    torch.testing.assert_close(recurrent, dense_state, rtol=0.02, atol=0.02)
    assert torch.isfinite(outputs[0]).all()
    torch.testing.assert_close(mla_guard, guard_before, rtol=0, atol=0)

    invalid_ids = torch.tensor([-1, NUM_BLOCKS, 3], dtype=torch.int64, device="npu")
    state_changed = raw["mla"].cpu() != before_state_updates
    state_allowed = torch.zeros_like(state_changed)
    state_payload_bytes = conv[0].numel() * conv.element_size() + recurrent[0].numel() * recurrent.element_size()
    for state_id in (1, 3):
        start = state_id * mla_spec.page_size_bytes
        state_allowed[start : start + state_payload_bytes] = True
    assert torch.all(~state_changed | state_allowed)
    packed = gather_ssm_states(recurrent, invalid_ids, torch.ones(3, dtype=torch.bool, device="npu"))
    assert torch.all(packed[:2] == 0)
    scatter_ssm_states_(recurrent, invalid_ids, packed)
    before_copy = raw["mla"].cpu().clone()
    copy_kv_cache_blocks_inplace([caches["kda"], mla], NUM_BLOCKS, [SimpleNamespace(src_block_id=3, dst_block_id=2)])
    torch.testing.assert_close(conv[2], conv[3], rtol=0, atol=0)
    torch.testing.assert_close(recurrent[2], recurrent[3], rtol=0, atol=0)
    torch.testing.assert_close(mla[2 * ratio : 3 * ratio], mla[3 * ratio : 4 * ratio], rtol=0, atol=0)
    torch.testing.assert_close(mla_guard, guard_before, rtol=0, atol=0)

    copy_changed = raw["mla"].cpu() != before_copy
    copy_allowed = torch.zeros_like(copy_changed)
    copy_start = 2 * mla_spec.page_size_bytes
    copy_allowed[copy_start : copy_start + state_payload_bytes] = True
    for kernel_index in range(2 * ratio, 3 * ratio):
        start = kernel_index * mla.stride(0) * mla.element_size()
        copy_allowed[start : start + mla[0].numel() * mla.element_size()] = True
    assert torch.all(~copy_changed | copy_allowed)

    # Zero only a free MLA block. It may intentionally alias KDA at that same
    # block ID, but cannot touch live IDs 1/3/4 or page padding.
    zeroer = AscendV2KVBlockZeroer(
        torch.device("npu"),
        attn_groups_iter=groups[:1],
        kernel_block_sizes=[KERNEL_BLOCK_SIZE, manager_block_size],
        static_forward_context={"mla": SimpleNamespace(kv_cache=mla)},
        num_blocks=NUM_BLOCKS,
        cache_dtype="auto",
    )
    before_zero = raw["mla"].cpu().clone()
    zeroer.zero_block_ids([2])
    torch.npu.synchronize()
    assert torch.all(mla[2 * ratio : 3 * ratio] == 0)
    torch.testing.assert_close(mla_guard, guard_before, rtol=0, atol=0)
    changed = raw["mla"].cpu() != before_zero
    allowed = torch.zeros_like(changed)
    kernel_stride_bytes = mla.stride(0) * mla.element_size()
    kernel_payload_bytes = KERNEL_BLOCK_SIZE * 576 * mla.element_size()
    for kernel_index in range(2 * ratio, 3 * ratio):
        start = kernel_index * kernel_stride_bytes
        allowed[start : start + kernel_payload_bytes] = True
    assert torch.all(~changed | allowed)
    assert raw["mla"].untyped_storage().data_ptr() == raw["kda"].untyped_storage().data_ptr()
