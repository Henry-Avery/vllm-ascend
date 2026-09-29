# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Actual ced685 planner -> Ascend allocator/reshape CPU regressions.

Upstream excerpts are unmodified, locked test data. Model spec/config shells
and device launches are substitutes; descriptors, layout stride calculation,
allocation/reshape and zeroer metadata construction execute production bodies.
"""

import ast
import dataclasses
import itertools
import json
import math
import sys
from collections import defaultdict
from enum import Enum, IntEnum
from functools import cached_property
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest
import torch
from test_hybrid_state_page_layout import ROOT, MambaSpec, _load_functions
from test_hybrid_state_page_layout import api as page_api

HEADS = 12
HEAD_DIM = 128
LATENT_DIM = 512
ROPE_DIM = 64
KERNEL_TOKENS = 128
NUM_BLOCKS = 4
NUM_LAYERS = 2
ALLOCATION_PREFIX = 64


class _MambaSpec(MambaSpec):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


@pytest.fixture
def descriptor_api(monkeypatch):
    namespace = page_api.__wrapped__()
    namespace["async_tensor_h2d"] = lambda array, device: torch.from_numpy(array).to(device)
    module = ModuleType("ced685_descriptor_fixture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    upstream = module.__dict__
    upstream.update(namespace)
    upstream.update(
        Enum=Enum,
        IntEnum=IntEnum,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        fields=dataclasses.fields,
        replace=dataclasses.replace,
        cached_property=cached_property,
        prod=math.prod,
        math=math,
        defaultdict=defaultdict,
        cast=cast,
        iprod=itertools.product,
        _DIM_L=0,
        _DIM_B=1,
        HiSparseHotSpec=type("HiSparseHotSpec", (), {}),
        _glm5_next_tensor_layout=lambda _: None,
    )
    fixture = json.loads((Path(__file__).parent / "fixtures/ced685_descriptor_sources.json").read_text())
    assert fixture["upstream_sha"] == "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
    for name in ("mla_spec", "layout", "interface", "planner", "zeroer"):
        exec("from __future__ import annotations\n" + fixture["sources"][name]["verbatim_source"], upstream)
    core_tree = ast.parse((ROOT / "vllm_ascend/core/kv_cache_interface.py").read_text())
    core_class = next(
        node for node in core_tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAAttentionSpec"
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, core_class], type_ignores=[])), "cache_spec", "exec"
        ),
        upstream,
    )
    _load_functions(
        "vllm_ascend/core/kv_cache_interface.py", ["get_storage_block_size", "get_kv_cache_compression_ratio"], upstream
    )
    namespace.update({name: upstream[name] for name in ("AttentionSpec", "MLAAttentionSpec", "AscendMLAAttentionSpec")})
    namespace["get_kv_cache_compression_ratio"] = upstream["get_kv_cache_compression_ratio"]
    namespace["get_storage_block_size"] = upstream["get_storage_block_size"]
    attention_utils = ast.parse((ROOT / "vllm_ascend/attention/utils.py").read_text())
    heads = next(
        node
        for node in attention_utils.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MLA_FLASH_SUPPORTED_Q_HEADS" for target in node.targets)
    )
    exec(compile(ast.Module(body=[heads], type_ignores=[]), "query_heads", "exec"), namespace)
    namespace["enable_sfa_dcp_replicated_indexer"] = lambda _: False
    namespace["kv_cache_dtype_str_to_dtype"] = lambda *_: torch.int8
    imported = ModuleType("vllm.model_executor.models.deepseek_v2")
    imported.DeepseekV32IndexerCache = type("DeepseekV32IndexerCache", (), {})
    monkeypatch.setitem(sys.modules, imported.__name__, imported)
    _load_functions("vllm_ascend/worker/v2/attn_utils.py", ["get_kv_cache_spec"], namespace)
    production = ast.parse((ROOT / "vllm_ascend/worker/v2/attn_utils.py").read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == "_mla_kernel_page_geometry" for node in production.body):
        _load_functions("vllm_ascend/worker/v2/attn_utils.py", ["_mla_kernel_page_geometry"], namespace)
    namespace.update(
        KVBlockZeroer=upstream["KVBlockZeroer"],
        AttentionGroup=upstream["AttentionGroup"],
        replace=dataclasses.replace,
        SimpleNamespace=SimpleNamespace,
    )
    _load_functions("vllm_ascend/worker/v2/attn_utils.py", ["allocate_kv_cache_main"], namespace)
    tree = ast.parse((ROOT / "vllm_ascend/worker/v2/utils.py").read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendV2KVBlockZeroer")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), "zeroer", "exec"),
        namespace,
    )
    return namespace, upstream


def _planner_config(namespace, upstream, layout, ratio, num_blocks=NUM_BLOCKS):
    manager_tokens = KERNEL_TOKENS * ratio
    shapes = ((3, 3 * HEADS * HEAD_DIM), (HEADS, HEAD_DIM, HEAD_DIM))
    state_bytes = math.prod(shapes[0]) * 2 + math.prod(shapes[1]) * 4
    page_bytes = max(state_bytes, manager_tokens * (LATENT_DIM + ROPE_DIM) * 2)
    common = dict(block_size=manager_tokens, page_size_bytes=page_bytes, page_size_padded=page_bytes)
    base_spec = upstream["MLAAttentionSpec"](
        block_size=manager_tokens, num_kv_heads=1, head_size=LATENT_DIM + ROPE_DIM, dtype=torch.bfloat16
    )
    mamba = _MambaSpec(
        **common,
        shapes=shapes,
        dtypes=(torch.bfloat16, torch.float32),
        num_heads=1,
        tokens_per_state=-1,
        state_content_size_bytes=state_bytes,
        get_num_kernel_states=lambda _: 1,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
            prefix_cache_retention_interval=None,
            cache_dtype="auto",
            num_gpu_blocks_override=None,
        ),
        attention_config=SimpleNamespace(hisparse_config=None, indexer_kv_dtype="int8"),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        kv_transfer_config=None,
    )
    layer = namespace["MLAAttention"](
        num_heads=HEADS, impl=SimpleNamespace(fa_quant_layer=False), get_kv_cache_spec=lambda _: base_spec
    )
    namespace["get_layers_from_vllm_config"] = lambda *_: {
        "mla": layer,
        **{f"mla{i}": layer for i in range(NUM_LAYERS)},
    }
    mla = dataclasses.replace(namespace["get_kv_cache_spec"](config)["mla"], page_size_padded=page_bytes)
    groups = [
        upstream["KVCacheGroupSpec"](layer_names=[f"mla{i}" for i in range(NUM_LAYERS)], kv_cache_spec=mla),
        upstream["KVCacheGroupSpec"](layer_names=[f"kda{i}" for i in range(NUM_LAYERS)], kv_cache_spec=mamba),
    ]
    plan = upstream["get_kv_cache_config_from_groups"](config, groups, num_blocks * NUM_LAYERS * page_bytes)
    return plan, config, mla, mamba


def _writer_reader_seam(namespace, fused, ratio):
    """Execute real Python consumers with explicit CPU operator substitutes."""
    slots = torch.tensor([ratio * KERNEL_TOKENS, 2 * ratio * KERNEL_TOKENS - 1])
    payload = torch.arange(2 * (LATENT_DIM + ROPE_DIM)).remainder(53).reshape(2, -1).to(torch.bfloat16)
    expected = payload.clone()
    expected[:, :LATENT_DIM] *= 2
    calls = []

    def writer(**kwargs):
        for cache, values in ((kwargs["key_cache"], kwargs["key"]), (kwargs["value_cache"], kwargs["value"])):
            assert cache.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
            assert cache.stride(0) == fused.stride(0)
            cache[slots // KERNEL_TOKENS, slots % KERNEL_TOKENS] = values
        calls.append("writer substitute")

    def reader(query, cache, **kwargs):
        assert cache is fused
        torch.testing.assert_close(cache[slots // KERNEL_TOKENS, slots % KERNEL_TOKENS, 0], expected)
        calls.append("reader substitute")
        return torch.ones((1, 2, LATENT_DIM)), None

    namespace.update(
        DeviceOperator=SimpleNamespace(reshape_and_cache=writer),
        envs=SimpleNamespace(),
        wait_for_device_metadata=lambda *_: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        record_attention_compute_start=lambda: None,
    )
    _load_functions(
        "vllm_ascend/attention/mla_v1.py",
        ["_exec_kv_no_rope", "_forward_external_flashmla"],
        namespace,
        class_name="AscendMLAImpl",
    )
    layer = SimpleNamespace(
        kv_a_layernorm=lambda value: value * 2,
        num_kv_heads=1,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        _v_up_proj=lambda value: value,
    )
    namespace["_exec_kv_no_rope"](layer, payload, (fused[..., :LATENT_DIM], fused[..., LATENT_DIM:]), slots)

    def read():
        flash = SimpleNamespace(
            schedule=torch.zeros(1),
            query=torch.zeros(2, 1, LATENT_DIM + ROPE_DIM),
            adapter=SimpleNamespace(attention=reader),
            block_table=torch.tensor([[ratio, 2 * ratio - 1]]),
            cache_lens=torch.tensor([2]),
            cu=torch.tensor([0, 2]),
            used_q=torch.tensor([2]),
            attn_mask=None,
            token_live=torch.ones(2, dtype=torch.bool),
        )
        namespace["_forward_external_flashmla"](
            layer,
            SimpleNamespace(ql_nope=torch.zeros(2, 1, LATENT_DIM), q_pe=torch.zeros(2, 1, ROPE_DIM)),
            fused,
            SimpleNamespace(external_flashmla=flash),
        )
        assert calls == ["writer substitute", "reader substitute"]

    return read


def test_reported_lbnhc_query_heads_do_not_expand_planner_geometry(descriptor_api):
    namespace, upstream = descriptor_api
    blocks = 53733
    layout = upstream["KVCacheLayout"].LBNHC
    plan, config, spec, _ = _planner_config(namespace, upstream, layout, 6, num_blocks=blocks)
    namespace["get_current_vllm_config"] = lambda: config
    descriptor = plan.kv_cache_tensors[0]
    assert spec.num_query_heads == 12 and spec.num_heads == 1
    assert spec.page_size_bytes == 884736
    assert descriptor.block_stride == 884736
    assert descriptor.layer_stride == 47539519488
    # Reproduce the two exact startup-log numbers using the old erroneous
    # H=12 planner input. No large allocation is needed for this calculation.
    legacy_shape = SimpleNamespace(
        block_size=spec.block_size,
        page_size_padded=spec.page_size_padded,
        num_heads=12,
        get_num_kernel_states=spec.get_num_kernel_states,
        state_content_size_bytes=spec.state_content_size_bytes,
    )
    legacy_strides = upstream["compute_layout_strides"](legacy_shape, blocks, NUM_LAYERS, layout)
    assert legacy_strides[:2] == (570474233856, 10616832)
    # Exercise the full-size production allocation bounds without reserving
    # hundreds of GiB of physical CPU memory.
    raw = namespace["_allocate_kv_cache"](plan, {}, torch.device("meta"))
    assert raw["mla0"].numel() == blocks * spec.page_size_bytes
    assert raw["mla1"].storage_offset() == descriptor.layer_stride


def test_real_spec_merge_preserves_query_heads_and_kv_geometry(descriptor_api):
    namespace, upstream = descriptor_api
    _, _, spec, _ = _planner_config(namespace, upstream, upstream["KVCacheLayout"].LBNHC, 6)
    merged = type(spec).merge([spec, dataclasses.replace(spec)])
    assert merged.num_query_heads == 12
    assert merged.num_heads == merged.num_kv_heads == 1
    assert merged.page_size_bytes == spec.page_size_bytes
    with pytest.raises(AssertionError, match="same Ascend KV cache layout"):
        type(spec).merge([spec, dataclasses.replace(spec, num_query_heads=48)])


@pytest.mark.parametrize("query_heads", [8, 12, 48, 64, 96])
@pytest.mark.parametrize("flash_hardware", [False, True])
def test_real_spec_query_heads_select_backend_without_changing_kv_heads(descriptor_api, query_heads, flash_hardware):
    namespace, upstream = descriptor_api
    layout = upstream["KVCacheLayout"].BLHNC
    plan, config, spec, mamba = _planner_config(namespace, upstream, layout, 3)
    spec = dataclasses.replace(spec, num_query_heads=query_heads)
    plan.kv_cache_groups[0].kv_cache_spec = spec
    namespace["get_current_vllm_config"] = lambda: config
    namespace["get_current_hardware_profile"] = lambda: SimpleNamespace(supports=lambda _: flash_hardware)
    namespace["_get_attention_kv_cache_dims"] = lambda *_: (LATENT_DIM, ROPE_DIM)
    raw = namespace["_allocate_kv_cache"](plan, {}, torch.device("cpu"))
    groups = [
        upstream["AttentionGroup"](
            backend=None, layer_names=group.layer_names, kv_cache_spec=group.kv_cache_spec, kv_cache_group_id=index
        )
        for index, group in enumerate(plan.kv_cache_groups)
    ]
    caches = namespace["_reshape_kv_cache_v2"](
        groups, raw, "auto", [KERNEL_TOKENS, mamba.block_size], {}, plan, kv_cache_layout=layout
    )
    assert spec.num_heads == spec.num_kv_heads == 1
    fused = flash_hardware and query_heads in (8, 12, 64, 96)
    assert isinstance(caches["mla0"], torch.Tensor) is fused
    if fused:
        assert caches["mla0"].shape == (NUM_BLOCKS * 3, KERNEL_TOKENS, 1, LATENT_DIM + ROPE_DIM)
    else:
        nope, rope = caches["mla0"]
        assert nope.shape == (NUM_BLOCKS * 3, KERNEL_TOKENS, 1, LATENT_DIM)
        assert rope.shape == (NUM_BLOCKS * 3, KERNEL_TOKENS, 1, ROPE_DIM)
        assert nope.stride(0) * nope.element_size() == plan.kv_cache_tensors[0].block_stride // 3
        assert rope.stride(0) == nope.stride(0)


@pytest.mark.parametrize("invalid", ["short_pitch", "negative_offset", "allocation_bounds"])
def test_allocator_rejects_invalid_descriptor_geometry(descriptor_api, invalid):
    namespace, upstream = descriptor_api
    plan, config, spec, _ = _planner_config(namespace, upstream, upstream["KVCacheLayout"].BLHNC, 6)
    namespace["get_current_vllm_config"] = lambda: config
    descriptor = plan.kv_cache_tensors[0]
    if invalid == "short_pitch":
        descriptor.block_stride = spec.page_size_bytes - 1
    elif invalid == "negative_offset":
        descriptor.offset = -1
    else:
        descriptor.offset = descriptor.size
    with pytest.raises(ValueError, match="exceeds the backing allocation"):
        namespace["_allocate_kv_cache"](plan, {}, torch.device("cpu"))


@pytest.mark.parametrize("invalid", ["fractional_pitch", "layer_offset", "layer_region", "layout"])
def test_mla_split_rejects_non_affine_descriptor_geometry(descriptor_api, invalid):
    namespace, upstream = descriptor_api
    layout = upstream["KVCacheLayout"].BLHNC
    plan, config, spec, _ = _planner_config(namespace, upstream, layout, 6)
    namespace["get_current_vllm_config"] = lambda: config
    raw = namespace["_allocate_kv_cache"](plan, {}, torch.device("cpu"))["mla1"]
    descriptor = dataclasses.replace(plan.kv_cache_tensors[0])
    if invalid == "fractional_pitch":
        raw = torch.as_strided(raw, raw.shape, (raw.stride(0) - 1, 1))
    elif invalid == "layer_offset":
        descriptor.offset = 1
    elif invalid == "layer_region":
        descriptor.layer_stride = 2 * spec.page_size_bytes
    else:
        layout = upstream["KVCacheLayout"].LHBNC
    with pytest.raises(ValueError, match="MLA"):
        namespace["_mla_kernel_page_geometry"](raw, spec, 6, descriptor, 1, layout)


class _CpuZeroKernel:
    """Substitute the device launch, using the real zeroer's address metadata."""

    def __init__(self, backing):
        self.backing = backing

    def __getitem__(self, grid):
        def launch(addresses, strides, sizes, ids, **kwargs):
            assert grid[:2] == (len(ids), len(addresses))
            for address, stride, size in zip(addresses.tolist(), strides.tolist(), sizes.tolist()):
                for block in ids.tolist():
                    start = address - self.backing.data_ptr() + block * stride * 4
                    assert 0 <= start < start + size * 4 <= self.backing.numel()
                    self.backing[start : start + size * 4].zero_()

        return launch


@pytest.mark.parametrize("layout_name", ["BLHNC", "LBHNC", "LBNHC"])
@pytest.mark.parametrize("ratio", [1, 3, 6])
@pytest.mark.parametrize("flash_enabled", [False, True])
def test_real_planner_descriptors_allocate_reshape_and_zero_geometry(
    descriptor_api, monkeypatch, layout_name, ratio, flash_enabled
):
    namespace, upstream = descriptor_api
    layout = upstream["KVCacheLayout"][layout_name]
    plan, config, mla, mamba = _planner_config(namespace, upstream, layout, ratio)
    assert mla.num_heads == mla.num_kv_heads == 1
    assert mla.num_query_heads == HEADS
    namespace["get_current_vllm_config"] = lambda: config
    namespace["envs"] = SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash_enabled)
    namespace["_get_attention_kv_cache_dims"] = lambda *_: (LATENT_DIM, ROPE_DIM)
    layer_class = namespace["MLAAttention"]
    backend = SimpleNamespace(full_cls_name=lambda: "test.backend")
    layers = {
        f"mla{i}": layer_class(impl=SimpleNamespace(fa_quant_layer=False), get_attn_backend=lambda: backend)
        for i in range(NUM_LAYERS)
    }
    layers.update({f"kda{i}": SimpleNamespace(get_attn_backend=lambda: backend) for i in range(NUM_LAYERS)})
    namespace["get_layers_from_vllm_config"] = lambda *_: layers
    allocation = []
    zeros = torch.zeros

    def shifted_zeros(size, *extra_sizes, **kwargs):
        if kwargs.get("dtype") == torch.int8:
            backing = zeros(size + 2 * ALLOCATION_PREFIX, **kwargs)
            backing[:ALLOCATION_PREFIX] = 19
            backing[-ALLOCATION_PREFIX:] = 23
            allocation.append(backing)
            return backing[ALLOCATION_PREFIX:-ALLOCATION_PREFIX]
        return zeros(size, *extra_sizes, **kwargs)

    monkeypatch.setattr(torch, "zeros", shifted_zeros)
    caches = namespace["allocate_kv_cache_main"](plan, torch.device("cpu"), layout, [KERNEL_TOKENS, mamba.block_size])
    assert len(allocation) == 1
    groups = [
        upstream["AttentionGroup"](
            backend=None, layer_names=group.layer_names, kv_cache_spec=group.kv_cache_spec, kv_cache_group_id=index
        )
        for index, group in enumerate(plan.kv_cache_groups)
    ]
    page_bytes = mamba.page_size_bytes
    mla_descriptor, state_descriptor = plan.kv_cache_tensors
    for layer in range(NUM_LAYERS):
        conv, state = caches[f"kda{layer}"]
        expected = ALLOCATION_PREFIX + state_descriptor.offset + layer * state_descriptor.layer_stride
        assert conv.data_ptr() - allocation[0].data_ptr() == expected
        assert state.data_ptr() - allocation[0].data_ptr() == expected + conv[0].numel() * conv.element_size()
        assert conv.stride(0) * conv.element_size() == state_descriptor.block_stride
        assert state.stride(0) * state.element_size() == state_descriptor.block_stride
        view = caches[f"mla{layer}"]
        assert isinstance(view, torch.Tensor)
        for block in range(NUM_BLOCKS):
            for subpage in range(ratio):
                if layout_name == "BLHNC":
                    expected = ALLOCATION_PREFIX + block * mla_descriptor.block_stride
                    expected += subpage * (mla_descriptor.block_stride // ratio)
                    expected += (mla_descriptor.offset + layer * mla_descriptor.layer_stride) // ratio
                else:
                    expected = ALLOCATION_PREFIX + mla_descriptor.offset + layer * mla_descriptor.layer_stride
                    expected += block * page_bytes + subpage * (page_bytes // ratio)
                assert view[block * ratio + subpage].data_ptr() - allocation[0].data_ptr() == expected
                view[block * ratio + subpage].fill_(layer + block + 1)

    # Same-block cross-group alias is intentional; different owned manager
    # IDs must stay isolated even though BLHNC MLA subpages are repacked.
    readers = [_writer_reader_seam(namespace, caches[f"mla{i}"], ratio) for i in range(NUM_LAYERS)]
    live = [caches[f"mla{i}"][ratio : 2 * ratio].clone() for i in range(NUM_LAYERS)]
    for layer in range(NUM_LAYERS):
        for state in caches[f"kda{layer}"]:
            state[2].fill_(torch.nan)
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[f"mla{layer}"][ratio : 2 * ratio], live[layer])
    for read in readers:
        read()
    before_copy = allocation[0].clone()
    namespace["copy_kv_cache_blocks_inplace"](
        [caches[f"mla{i}"] for i in range(NUM_LAYERS)], NUM_BLOCKS, [SimpleNamespace(src_block_id=1, dst_block_id=3)]
    )
    for layer in range(NUM_LAYERS):
        torch.testing.assert_close(caches[f"mla{layer}"][3 * ratio : 4 * ratio], live[layer])
    allowed = torch.zeros_like(allocation[0], dtype=torch.bool)
    for layer in range(NUM_LAYERS):
        for subpage in range(ratio):
            view = caches[f"mla{layer}"][3 * ratio + subpage]
            start = view.data_ptr() - allocation[0].data_ptr()
            allowed[start : start + view.numel() * view.element_size()] = True
    assert torch.all((allocation[0] == before_copy) | allowed)
    zeroer = namespace["AscendV2KVBlockZeroer"](
        torch.device("cpu"),
        groups[:1],
        [KERNEL_TOKENS, mamba.block_size],
        {f"mla{i}": SimpleNamespace(kv_cache=caches[f"mla{i}"]) for i in range(NUM_LAYERS)},
        num_blocks=NUM_BLOCKS,
        cache_dtype="auto",
    )
    addresses, strides, sizes, *_ = zeroer._zeroers[0]._meta
    expected_addresses = [
        caches[f"mla{i}"].data_ptr() + subpage * caches[f"mla{i}"].stride(0) * 2
        for i in range(NUM_LAYERS)
        for subpage in range(ratio)
    ]
    assert addresses.tolist() == expected_addresses
    assert strides.tolist() == [mla_descriptor.block_stride // 4] * (ratio * NUM_LAYERS)
    assert sizes.tolist() == [KERNEL_TOKENS * (LATENT_DIM + ROPE_DIM) * 2 // 4] * (ratio * NUM_LAYERS)
    upstream["async_tensor_h2d"] = lambda ids, **kwargs: torch.tensor(ids, **kwargs)
    upstream["_zero_kv_blocks_kernel"] = _CpuZeroKernel(allocation[0])
    before_zero = allocation[0].clone()
    zeroer.zero_block_ids([3])
    for layer in range(NUM_LAYERS):
        assert torch.all(caches[f"mla{layer}"][3 * ratio : 4 * ratio] == 0)
        torch.testing.assert_close(caches[f"mla{layer}"][ratio : 2 * ratio], live[layer])
    assert torch.all((allocation[0] == before_zero) | allowed)
    assert torch.all(allocation[0][:ALLOCATION_PREFIX] == 19)
    assert torch.all(allocation[0][-ALLOCATION_PREFIX:] == 23)
