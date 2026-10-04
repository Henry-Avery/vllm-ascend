# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, compute_layer_kv_cache_shape_bytes

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

KVCacheLayout = pytest.importorskip("vllm.v1.kv_cache_layout").KVCacheLayout


@pytest.mark.parametrize("query_heads", [8, 12, 48, 64, 96])
@pytest.mark.parametrize("page_padding", [None, 488448])
def test_query_heads_do_not_change_planner_storage(query_heads, page_padding):
    # Exercise the planner before allocation, rather than handcrafting a
    # descriptor that already has the desired single-latent-head strides.
    spec = AscendMLAAttentionSpec(
        block_size=384,
        num_query_heads=query_heads,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        page_size_padded=page_padding,
    )
    spec = AscendMLAAttentionSpec.merge([spec, spec])
    assert spec.num_query_heads == query_heads
    assert spec.num_heads == spec.num_kv_heads == 1
    assert compute_layer_kv_cache_shape_bytes(spec, 2) == (2, 1, 384, 1152)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBNHC,
            prefix_cache_retention_interval=None,
            num_gpu_blocks_override=None,
        ),
        kv_transfer_config=None,
    )
    group = KVCacheGroupSpec(layer_names=["layer.0", "layer.1"], kv_cache_spec=spec)
    num_blocks = 2
    pool_bytes = num_blocks * len(group.layer_names) * spec.page_size_bytes
    result = get_kv_cache_config_from_groups(config, [group], pool_bytes)
    assert result.num_blocks == num_blocks
    (descriptor,) = result.kv_cache_tensors
    assert descriptor.size == pool_bytes
    assert descriptor.layer_stride == num_blocks * spec.page_size_bytes
    assert descriptor.block_stride == spec.page_size_bytes
    assert descriptor.offset == 0


@pytest.mark.parametrize("num_head_slots,expected_heads", [(None, 1), (2, 2)])
def test_query_heads_preserve_upstream_physical_head_slots(num_head_slots, expected_heads):
    # The backend may explicitly pack more than one physical head slot.
    # Restoring the inherited property must preserve that contract too.
    content_bytes = 1152 // expected_heads
    spec = AscendMLAAttentionSpec(
        block_size=384,
        num_query_heads=64,
        num_kv_heads=1,
        num_head_slots=num_head_slots,
        state_content_bytes=content_bytes,
        head_size=576,
        dtype=torch.bfloat16,
    )
    spec = AscendMLAAttentionSpec.merge([spec, spec])
    assert spec.num_query_heads == 64
    assert spec.num_heads == expected_heads
    assert compute_layer_kv_cache_shape_bytes(spec, 2) == (2, expected_heads, 384, content_bytes)


def test_merge_rejects_incompatible_query_head_layouts():
    def spec(query_heads):
        return AscendMLAAttentionSpec(
            block_size=128,
            num_query_heads=query_heads,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
        )

    with pytest.raises(AssertionError, match="same Ascend KV cache layout"):
        AscendMLAAttentionSpec.merge([spec(64), spec(48)])
