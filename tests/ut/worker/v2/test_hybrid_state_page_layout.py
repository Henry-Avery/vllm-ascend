# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU production-function regressions, without loading vLLM/NPU extensions.

Run with --confcutdir=tests/ut/worker/v2. AST loading executes the production
function bodies; only external spec classes and device kernels are shimmed.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]

MLA_LATENT_DIM = 512
MLA_ROPE_DIM = 64
MLA_KERNEL_BLOCK_SIZE = 128
KDA_STATE_PAYLOAD_BYTES = 814080


def _load_functions(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    for node in selected:
        node.decorator_list = []
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)


@pytest.mark.parametrize("manager_block_size", [128, 384, 768])
def test_writer_to_external_reader_keeps_shared_backing_across_kda_updates(api, manager_block_size):
    """CPU Python seam only; writer/attention launches are explicit substitutes."""
    api["_get_attention_kv_cache_dims"] = lambda *_: (MLA_LATENT_DIM, MLA_ROPE_DIM)
    page_bytes = max(KDA_STATE_PAYLOAD_BYTES, manager_block_size * (MLA_LATENT_DIM + MLA_ROPE_DIM) * 2)
    mamba_spec = MambaSpec(
        page_size_bytes=page_bytes,
        page_size_padded=page_bytes,
        shapes=((3, 4608), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=manager_block_size,
    )
    mla_spec = AscendMLAAttentionSpec(
        page_size_bytes=page_bytes,
        page_size_padded=page_bytes,
        block_size=manager_block_size,
        num_kv_heads=1,
        head_size=MLA_LATENT_DIM + MLA_ROPE_DIM,
        dtype=torch.bfloat16,
        model_version=None,
        indexes_kv_by_block_stride=False,
        num_query_heads=12,
    )
    groups = [
        SimpleNamespace(layer_names=[name], kv_cache_spec=spec)
        for name, spec in (("mla", mla_spec), ("kda", mamba_spec))
    ]
    descriptors = [
        SimpleNamespace(
            size=7 * page_bytes + 128,
            layers=group.layer_names,
            layer_stride=7 * page_bytes,
            block_stride=page_bytes,
            offset=64,
        )
        for group in groups
    ]
    config = SimpleNamespace(num_blocks=7, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    raw = api["_allocate_kv_cache"](config, {}, torch.device("cpu"))
    caches = api["_reshape_kv_cache_v2"](
        [SimpleNamespace(**vars(group), kv_cache_group_id=index, backend=None) for index, group in enumerate(groups)],
        raw,
        "auto",
        [MLA_KERNEL_BLOCK_SIZE, manager_block_size],
        {},
        config,
    )
    fused = caches["mla"]
    components = fused[..., :MLA_LATENT_DIM], fused[..., MLA_LATENT_DIM:]
    slots = torch.tensor(
        [4 * manager_block_size, 4 * manager_block_size + 127, 5 * manager_block_size, 6 * manager_block_size - 1]
    )
    payload = torch.arange(4 * (MLA_LATENT_DIM + MLA_ROPE_DIM)).remainder(53).reshape(4, -1).to(torch.bfloat16)
    expected = payload.clone()
    expected[:, :MLA_LATENT_DIM] *= 2
    calls = []

    def writer(**kwargs):
        for cache, values in ((kwargs["key_cache"], kwargs["key"]), (kwargs["value_cache"], kwargs["value"])):
            assert cache.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
            cache[slots // MLA_KERNEL_BLOCK_SIZE, slots % MLA_KERNEL_BLOCK_SIZE] = values
        calls.append("writer substitute")

    def reader(query, cache, **kwargs):
        assert cache is fused
        torch.testing.assert_close(cache[slots // MLA_KERNEL_BLOCK_SIZE, slots % MLA_KERNEL_BLOCK_SIZE, 0], expected)
        calls.append("reader substitute")
        return torch.ones((1, 4, MLA_LATENT_DIM)), None

    api.update(
        DeviceOperator=SimpleNamespace(reshape_and_cache=writer),
        envs=SimpleNamespace(VLLM_ASCEND_FLASH_MLA_TRACE=False),
        wait_for_device_metadata=lambda *_: None,
        DeviceMetadataStage=SimpleNamespace(ATTENTION=1),
        record_attention_compute_start=lambda: None,
    )
    _load_functions(
        "vllm_ascend/attention/mla_v1.py",
        ["_exec_kv_no_rope", "_forward_external_flashmla"],
        api,
        class_name="AscendMLAImpl",
    )
    layer = SimpleNamespace(
        kv_a_layernorm=lambda value: value * 2,
        num_kv_heads=1,
        kv_lora_rank=MLA_LATENT_DIM,
        qk_rope_head_dim=MLA_ROPE_DIM,
        _logged_flashmla_decode=True,
        _v_up_proj=lambda value: value,
    )
    api["_exec_kv_no_rope"](layer, payload, components, slots)
    before = raw["mla"].clone()
    conv, state = caches["kda"]
    conv[1].fill_(torch.nan)
    state[3].fill_(torch.nan)
    allowed = torch.zeros_like(before, dtype=torch.bool)
    for view, slot in ((conv, 1), (state, 3)):
        start = view.data_ptr() - raw["mla"].data_ptr() + slot * view.stride(0) * view.element_size()
        allowed[start : start + view[0].numel() * view.element_size()] = True
    assert torch.all((raw["mla"] == before) | allowed)
    flash = SimpleNamespace(
        schedule=torch.zeros(1),
        query=torch.zeros(4, 1, MLA_LATENT_DIM + MLA_ROPE_DIM),
        adapter=SimpleNamespace(attention=reader),
        block_table=torch.tensor([[4, 5]]),
        cache_lens=torch.tensor([4]),
        cu=torch.tensor([0, 4]),
        used_q=torch.tensor([4]),
        attn_mask=None,
        token_live=torch.ones(4, dtype=torch.bool),
    )
    api["_forward_external_flashmla"](
        layer,
        SimpleNamespace(ql_nope=torch.zeros(4, 1, MLA_LATENT_DIM), q_pe=torch.zeros(4, 1, MLA_ROPE_DIM)),
        fused,
        SimpleNamespace(external_flashmla=flash),
    )
    assert calls == ["writer substitute", "reader substitute"]


class MambaSpec(SimpleNamespace):
    pass


class AttentionSpec(SimpleNamespace):
    pass


class MLAAttentionSpec(AttentionSpec):
    pass


class AscendMLAAttentionSpec(MLAAttentionSpec):
    @property
    def supports_single_raw_backing(self):
        return self.model_version is None and not self.indexes_kv_by_block_stride


class MLAAttention(SimpleNamespace):
    pass


@pytest.fixture
def api():
    config = SimpleNamespace(kv_transfer_config=None)
    layer = MLAAttention(num_heads=12, impl=SimpleNamespace(fa_quant_layer=False))
    namespace = {
        "torch": torch,
        "np": np,
        "math": math,
        "logger": MagicMock(),
        "MambaSpec": MambaSpec,
        "AttentionSpec": AttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "AscendMLAAttentionSpec": AscendMLAAttentionSpec,
        "MLAAttention": MLAAttention,
        "UniformTypeKVCacheSpecs": type("UniformTypeKVCacheSpecs", (), {}),
        "AscendIndexerKPoolTailSpec": type("AscendIndexerKPoolTailSpec", (), {}),
        "AscendSFAIndexerCacheSpec": type("AscendSFAIndexerCacheSpec", (), {}),
        "AttentionLayerBase": object,
        "KVPPConfig": SimpleNamespace(from_vllm_config=lambda _: SimpleNamespace(size=1)),
        "get_current_vllm_config": lambda: config,
        "get_layers_from_vllm_config": lambda *_args: {"mla": layer},
        "get_dtype_size": lambda dtype: torch.empty((), dtype=dtype).element_size(),
        "get_kv_cache_compression_ratio": lambda _: 1,
        "get_storage_block_size": lambda spec: spec.block_size,
        "get_kv_cache_tensor_layers": lambda descriptor: descriptor.layers,
        "vllm_version_is": lambda _: False,
        "enable_sfa": lambda _: False,
        "is_hidden_state_cache_spec": lambda _: False,
        "_is_dsv4_model": lambda _: False,
        "_get_attention_kv_cache_dims": lambda *_: (4, 0),
        "HardwareCapability": SimpleNamespace(MLA_FLASH="flash"),
        "get_current_hardware_profile": lambda: SimpleNamespace(supports=lambda _: True),
        "MLA_FLASH_SUPPORTED_Q_HEADS": (12,),
        "PAD_SLOT_ID": -1,
        "_STATE_COPY_BLOCK_SIZE": 8192,
        # Baseline comparison may load the old prefill consumer. Supply its
        # external clear helper so failures are behavioral, not missing names.
        "clear_ssm_states": lambda state, flags: state.masked_fill_(~flags[:, None, None, None], 0),
    }
    _load_functions(
        "vllm_ascend/worker/utils.py",
        [
            "row_major_strides",
            "make_page_strided_cache_view",
            "get_single_raw_mla_backing",
            "copy_kv_cache_blocks_inplace",
        ],
        namespace,
    )
    _load_functions(
        "vllm_ascend/worker/v2/attn_utils.py",
        [
            "_get_layer_kv_cache_specs",
            "_uses_single_raw_mla_cache",
            "_allocate_kv_cache",
            "_adjust_dsv4_kv_layout",
            "_reshape_mamba_kv_cache",
            "_reshape_kv_cache_v2",
        ],
        namespace,
    )
    _load_functions("vllm_ascend/ops/kimi_kda.py", ["_normalize_causal_cache_indices"], namespace)
    _install_state_kernels(namespace)
    _load_functions(
        "vllm_ascend/ops/kimi_kda.py",
        ["_run_causal_conv1d", "_run_prefill"],
        namespace,
        class_name="AscendKimiK3DeltaAttention",
    )
    return namespace


def _mamba_spec(page_bytes=64):
    return MambaSpec(
        page_size_bytes=page_bytes,
        page_size_padded=page_bytes,
        shapes=((2, 4), (1, 2, 2)),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=4,
    )


def _states(api, blocks=5):
    spec = _mamba_spec()
    backing = torch.full((64 + blocks * spec.page_size_bytes + 64,), 17, dtype=torch.uint8)
    raw = backing[64:-64]
    states = api["_reshape_mamba_kv_cache"](raw, spec)
    return backing, raw, states


def test_reported_slot3_overlaps_mla_page174_only_in_old_layout(api):
    # Exact reported state offset/payload and bad-element address, relative
    # to the reported storage base. Meta avoids allocating the 849-page pool.
    blocks, conv_bytes, recurrent_bytes = 849, 27648, 786432
    spec = MambaSpec(
        page_size_bytes=912384,
        page_size_padded=912384,
        shapes=((conv_bytes // 2,), (12, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    raw = torch.empty(blocks * spec.page_size_bytes, dtype=torch.uint8, device="meta")
    # Reconstruct d7 state-major geometry from the user-provided addresses;
    # this is not a new run of the production incident.
    old = (
        raw[blocks * conv_bytes : blocks * (conv_bytes + recurrent_bytes)]
        .view(torch.float32)
        .view(blocks, 12, 128, 128)
    )
    new = api["_reshape_mamba_kv_cache"](raw, spec)[1]
    assert old.storage_offset() * 4 == 23473152
    bad_byte = 70434475949812 - 70434449489920
    old_start = (old.storage_offset() + 3 * old.stride(0)) * 4
    new_start = (new.storage_offset() + 3 * new.stride(0)) * 4
    assert old_start == 25832448
    assert old_start <= bad_byte < old_start + recurrent_bytes
    assert not new_start <= bad_byte < new_start + recurrent_bytes
    assert new_start == 3 * spec.page_size_bytes + conv_bytes
    # Old manager block 768 splits into six 128-token kernel pages.
    assert bad_byte == 174 * (spec.page_size_bytes // 6) + 378 * 2


def test_page_states_preserve_offsets_dtypes_and_all_distinct_block_ranges(api):
    backing, raw, states = _states(api)
    conv, recurrent = states
    assert conv.stride() == (32, 4, 1)
    assert recurrent.stride() == (16, 4, 2, 1)
    assert conv.data_ptr() == raw.data_ptr()
    assert recurrent.data_ptr() == raw.data_ptr() + 16
    for block in range(5):
        for state in states:
            start = state[block].data_ptr() - raw.data_ptr()
            end = start + state[block].numel() * state.element_size()
            assert block * 64 <= start < end <= (block + 1) * 64
    before = backing.clone()
    conv[3].fill_(2)
    recurrent[3].fill_(5)
    assert torch.all(conv[3] == 2) and torch.all(recurrent[3] == 5)
    changed = (backing != before).nonzero().flatten()
    assert torch.all((changed >= 64 + 3 * 64) & (changed < 64 + 3 * 64 + 32))
    # An MLA history view over another owned physical block remains intact.
    mla = api["make_page_strided_cache_view"](raw, (5, 4, 1, 4), torch.bfloat16, 64)
    torch.testing.assert_close(mla[4].view(torch.uint8), before[64 + 4 * 64 : 64 + 4 * 64 + 32].view(4, 1, 8))


@pytest.mark.parametrize("page_bytes", [32, 64])
def test_independent_mamba_keeps_pr13_page_major_contract(api, page_bytes):
    spec = _mamba_spec(page_bytes)
    raw = torch.zeros(5 * page_bytes, dtype=torch.uint8)
    conv, recurrent = api["_reshape_mamba_kv_cache"](raw, spec)
    assert conv.stride(0) * conv.element_size() == page_bytes
    assert recurrent.stride(0) * recurrent.element_size() == page_bytes
    assert recurrent.data_ptr() - raw.data_ptr() == 16


@pytest.mark.parametrize("page_bytes", [28, 31])
def test_invalid_page_geometry_rejected(api, page_bytes):
    with pytest.raises(ValueError, match="aligned|exceed"):
        api["_reshape_mamba_kv_cache"](torch.zeros(5 * page_bytes, dtype=torch.uint8), _mamba_spec(page_bytes))


def test_page_builder_checks_slice_bounds_without_allocating_shape(api):
    backing = torch.zeros(512, dtype=torch.uint8)
    with pytest.raises(ValueError, match="allocation"):
        api["_adjust_dsv4_kv_layout"](backing[64:128], [(3, 4)], [torch.float32], 64)
    with pytest.raises(ValueError, match="aligned"):
        api["_adjust_dsv4_kv_layout"](backing[2:130], [(2, 4)], [torch.float32], 64)
    # In particular, no torch.empty(shape) may reserve the complete CPU pool.
    original_empty = torch.empty

    def scalar_empty(shape, *args, **kwargs):
        assert shape == ()
        return original_empty(shape, *args, **kwargs)

    with patch.object(torch, "empty", scalar_empty):
        states = api["_reshape_mamba_kv_cache"](backing[64:384], _mamba_spec())
    assert states[1].storage_offset() == 20


@pytest.mark.parametrize("kernel_block_size", [2, 4])
def test_production_allocator_and_reshape_select_pool_layout(api, kernel_block_size):
    spec = _mamba_spec()
    mla_spec = AscendMLAAttentionSpec(
        page_size_bytes=64,
        page_size_padded=64,
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
        model_version=None,
        indexes_kv_by_block_stride=False,
        num_query_heads=12,
    )
    groups = [
        SimpleNamespace(layer_names=[name], kv_cache_spec=layer_spec)
        for name, layer_spec in (("mla", mla_spec), ("kda", spec))
    ]
    descriptors = [
        SimpleNamespace(size=5 * 64, layers=[name], layer_stride=5 * 64, block_stride=64, offset=0)
        for name in ("mla", "kda")
    ]
    config = SimpleNamespace(num_blocks=5, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    raw = api["_allocate_kv_cache"](config, {}, torch.device("cpu"))
    assert raw["mla"].data_ptr() == raw["kda"].data_ptr()
    attn_groups = [
        SimpleNamespace(**vars(group), kv_cache_group_id=index, backend=None) for index, group in enumerate(groups)
    ]
    caches = api["_reshape_kv_cache_v2"](attn_groups, raw, "auto", [kernel_block_size, 4], {}, config)
    conv, recurrent = caches["kda"]
    assert conv.stride(0) * conv.element_size() == 64
    assert recurrent.stride(0) * recurrent.element_size() == 64
    ratio = 4 // kernel_block_size
    mla = caches["mla"]
    assert mla.shape[0] == 5 * ratio
    assert mla.stride(0) * mla.element_size() == 64 // ratio
    before = mla[4 * ratio : 5 * ratio].clone()
    recurrent[3].fill_(float("nan"))
    torch.testing.assert_close(mla[4 * ratio : 5 * ratio], before)


def test_allocator_multiple_layers_preserves_shared_aliases_and_layer_offsets(api):
    state_spec = _mamba_spec()
    mla_spec = AscendMLAAttentionSpec(
        page_size_bytes=64,
        block_size=4,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
        model_version=None,
        indexes_kv_by_block_stride=False,
        num_query_heads=12,
    )
    layer_bytes = 5 * 64
    groups = [
        SimpleNamespace(layer_names=["mla", "mla2"], kv_cache_spec=mla_spec),
        SimpleNamespace(layer_names=["kda", "kda2"], kv_cache_spec=state_spec),
    ]
    descriptors = [
        SimpleNamespace(
            size=2 * layer_bytes + 128, layers=group.layer_names, layer_stride=layer_bytes, block_stride=64, offset=64
        )
        for group in groups
    ]
    config = SimpleNamespace(num_blocks=5, kv_cache_tensors=descriptors, kv_cache_groups=groups)
    raw = api["_allocate_kv_cache"](config, {}, torch.device("cpu"))
    assert raw["mla"].data_ptr() == raw["kda"].data_ptr()
    assert raw["mla2"].data_ptr() == raw["kda2"].data_ptr() == raw["kda"].data_ptr() + layer_bytes
    state_groups = [SimpleNamespace(**vars(groups[1]), kv_cache_group_id=1)]
    caches = api["_reshape_kv_cache_v2"](state_groups, raw, "auto", [4, 4], {}, config)
    for name in ["kda", "kda2"]:
        conv, recurrent = caches[name]
        assert conv.data_ptr() == raw[name].data_ptr()
        assert recurrent.data_ptr() == raw[name].data_ptr() + 16
    before = raw["mla2"].clone()
    caches["kda"][1][3].fill_(float("nan"))
    torch.testing.assert_close(raw["mla2"], before)
    descriptors[0].offset = -64
    with pytest.raises(ValueError, match="allocation"):
        api["_allocate_kv_cache"](config, {}, torch.device("cpu"))


@pytest.mark.parametrize("manager_size,page_bytes,kernel_size", [(5, 64, 2), (4, 63, 2), (4, 16, 4)])
def test_mla_reshape_rejects_fractional_or_overlapping_kernel_pages(api, manager_size, page_bytes, kernel_size):
    spec = AscendMLAAttentionSpec(
        page_size_bytes=page_bytes,
        block_size=manager_size,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.bfloat16,
        model_version=None,
        indexes_kv_by_block_stride=False,
        num_query_heads=12,
    )
    group = SimpleNamespace(layer_names=["mla"], kv_cache_spec=spec, kv_cache_group_id=0)
    config = SimpleNamespace(num_blocks=5, kv_cache_groups=[group])
    with pytest.raises(ValueError, match="whole|fit"):
        api["_reshape_kv_cache_v2"](
            [group], {"mla": torch.zeros(5 * page_bytes, dtype=torch.uint8)}, "auto", [kernel_size], {}, config
        )


class _Pointer:
    def __init__(self, tensor, offset=None):
        self.flat = torch.empty(0, dtype=tensor.dtype).set_(
            tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,)
        )
        self.offset = torch.as_tensor(tensor.storage_offset() if offset is None else offset)

    def __add__(self, offset):
        pointer = object.__new__(_Pointer)
        pointer.flat = self.flat
        pointer.offset = self.offset + offset
        return pointer


class _TorchLanguage:
    """Execute actual Triton kernel bodies using CPU Torch load/store ops."""

    int64 = torch.int64
    int1 = torch.bool

    def __init__(self):
        self.program = (0, 0)

    def program_id(self, dim):
        return self.program[dim]

    def arange(self, start, end):
        return torch.arange(start, end)

    def load(self, pointer, mask=True, other=0):
        mask = torch.broadcast_to(torch.as_tensor(mask, dtype=torch.bool), pointer.offset.shape)
        # Mask before indexing: invalid or PAD pointers must never be read.
        active_offsets = pointer.offset[mask]
        assert torch.all((active_offsets >= 0) & (active_offsets < pointer.flat.numel()))
        output = torch.full(pointer.offset.shape, other, dtype=pointer.flat.dtype)
        output[mask] = pointer.flat[active_offsets]
        return output

    def store(self, pointer, values, mask=True):
        mask = torch.broadcast_to(torch.as_tensor(mask, dtype=torch.bool), pointer.offset.shape)
        active_offsets = pointer.offset[mask]
        assert torch.all((active_offsets >= 0) & (active_offsets < pointer.flat.numel()))
        pointer.flat[active_offsets] = values.to(pointer.flat.dtype)[mask]


class _Kernel:
    def __init__(self, function, language):
        self.function = function
        self.language = language

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.pop("num_warps", None)
            args = tuple(_Pointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args)
            for block in range(grid[0]):
                for batch in range(grid[1]):
                    self.language.program = (block, batch)
                    self.function(*args, **kwargs)

        return launch


def _install_state_kernels(namespace):
    language = _TorchLanguage()
    namespace["tl"] = language
    namespace["triton"] = SimpleNamespace(
        cdiv=lambda numerator, denominator: (numerator + denominator - 1) // denominator
    )
    namespace["STATE_IO_BLOCK_SIZE"] = 1024
    names = [
        "_gather_ssm_states_kernel",
        "_scatter_ssm_states_kernel",
        "_validate_state_and_indices",
        "_validate_gather_inputs",
        "_validate_scatter_inputs",
        "gather_ssm_states",
        "scatter_ssm_states_",
    ]
    _load_functions("vllm_ascend/ops/triton/mamba/state_index.py", names, namespace)
    for name in names[:2]:
        namespace[name] = _Kernel(namespace[name], language)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_actual_state_kernels_mask_invalid_rows_preserve_offsets_and_copy_dtype(api, dtype):
    backing, _, (_, state) = _states(api)
    state.fill_(2)
    before = backing.clone()
    index_storage = torch.tensor([3, 9, -1, 9, 5, 9, 0, 9, 2], dtype=dtype)
    indices = index_storage[::2]
    flags = torch.tensor([True, True, True, True, False])
    gathered = api["gather_ssm_states"](state, indices, flags, output_dtype=torch.bfloat16)
    assert torch.all(gathered[0] == 2) and torch.all(gathered[3] == 2)
    assert torch.all(gathered[1:3] == 0) and torch.all(gathered[4] == 0)
    # Same physical slot may be gathered repeatedly; scatter IDs are unique.
    duplicate_gather = api["gather_ssm_states"](state, torch.tensor([3, 3]), torch.tensor([True, True]))
    torch.testing.assert_close(duplicate_gather[0], duplicate_gather[1])
    source = gathered + 5
    api["scatter_ssm_states_"](state, indices, source)
    assert torch.all(state[3] == 7) and torch.all(state[0] == 7) and torch.all(state[2] == 5)
    assert torch.all(state[1] == 2) and torch.all(state[4] == 2)
    changed = (backing != before).nonzero().flatten()
    allowed = torch.zeros_like(backing, dtype=torch.bool)
    for slot in [3, 0, 2]:
        offset = state[slot].data_ptr() - backing.data_ptr()
        allowed[offset : offset + 16] = True
    assert torch.all(allowed[changed])


def test_empty_batch_and_state_validation(api):
    _, _, (_, state) = _states(api)
    indices = torch.empty(0, dtype=torch.int32)
    flags = torch.empty(0, dtype=torch.bool)
    gathered = api["gather_ssm_states"](state, indices, flags)
    assert gathered.shape == (0, *state.shape[1:])
    api["scatter_ssm_states_"](state, indices, gathered)
    with pytest.raises(TypeError, match="int32 or int64"):
        api["gather_ssm_states"](state, indices.float(), flags)
    with pytest.raises(ValueError, match="row must be contiguous"):
        api["gather_ssm_states"](state.transpose(-1, -2), indices, flags)


@pytest.mark.parametrize("keep", [None, torch.tensor([0, 2])])
def test_prefill_production_path_reads_only_selected_payloads_and_scatter_guards(api, keep):
    backing, _, (_, state) = _states(api)
    state.fill_(2)
    before = backing.clone()
    selected_indices = torch.tensor([3, -1, 1])
    flags = torch.tensor([True, True, False])

    def chunk(*args, **kwargs):
        initial = args[5]
        assert initial.is_contiguous() and initial.shape[0] == (3 if keep is None else 2)
        assert torch.all(initial[0] == 2) and torch.all(initial[1:] == 0)
        return args[0], initial.to(torch.bfloat16) + 5

    api["run_chunk_kda"] = chunk
    metadata = SimpleNamespace(
        cu_seqlens_host=[0, 1, 2], cu_seqlens_kern=None, keep_meta=keep, chunk_indices_chunk64_host=None
    )
    attention = SimpleNamespace(A_log=None, dt_bias=None, gate_lower_bound=-5)
    q = torch.zeros(1, 2, 1, 2)
    api["_run_prefill"](attention, q, q, q, q, q, state, selected_indices, flags, metadata)
    assert torch.all(state[3] == 7) and torch.all(state[1] == 5)
    assert torch.all(state[0] == 2) and torch.all(state[4] == 2)
    changed = (backing != before).nonzero().flatten()
    assert torch.all(((changed >= 144) & (changed < 160)) | ((changed >= 272) & (changed < 288)))


@pytest.mark.parametrize("run_mode", [0, 1])
def test_conv_passes_real_page_view_to_pinned_fla_backend(api, run_mode):
    _, _, (conv, _) = _states(api)
    seen = []

    def convolution(*args, **kwargs):
        actual_state = kwargs["conv_states"] if run_mode == 0 else args[1]
        assert actual_state is conv and actual_state.stride(0) == 32
        seen.append(kwargs)
        return args[0]

    api["causal_conv1d_fn"] = api["causal_conv1d_update"] = convolution
    x = torch.zeros(2, 4)
    api["_run_causal_conv1d"](
        x,
        torch.zeros(2, 4),
        conv,
        torch.tensor([0, 1, 2]),
        torch.tensor([[3, 4], [1, 2]]),
        None,
        run_mode=run_mode,
        max_query_len=1,
    )
    indices = seen[0]["cache_indices" if run_mode == 0 else "conv_state_indices"]
    assert indices.tolist() == [3, 1]


def test_production_cow_gathers_before_scatter_and_preserves_padding(api):
    backing, _, states = _states(api)
    for tensor in states:
        tensor[1].fill_(3)
        tensor[3].fill_(5)
    before = backing.clone()
    api["async_tensor_h2d"] = lambda array, device: torch.from_numpy(array).to(device)
    copies = [SimpleNamespace(src_block_id=1, dst_block_id=3), SimpleNamespace(src_block_id=3, dst_block_id=1)]
    api["copy_kv_cache_blocks_inplace"]([states, states], 5, copies)
    for state in states:
        assert torch.all(state[1] == 5) and torch.all(state[3] == 3)
    changed = (backing != before).nonzero().flatten()
    assert torch.all(((changed >= 128) & (changed < 160)) | ((changed >= 256) & (changed < 288)))


def test_cow_same_pointer_distinct_views_copy_union_once_from_original_sources(api):
    backing, raw, (conv, _) = _states(api)
    larger = api["make_page_strided_cache_view"](raw, (5, 4, 4), torch.bfloat16, 64)
    assert larger.data_ptr() == conv.data_ptr()
    larger[1].fill_(3)
    larger[3].fill_(5)
    before = backing.clone()
    api["async_tensor_h2d"] = lambda array, device: torch.from_numpy(array).to(device)
    copies = [SimpleNamespace(src_block_id=1, dst_block_id=3), SimpleNamespace(src_block_id=3, dst_block_id=1)]
    api["copy_kv_cache_blocks_inplace"]([[conv, larger], [conv, larger]], 5, copies)
    assert torch.all(larger[1] == 5) and torch.all(larger[3] == 3)
    changed = (backing != before).nonzero().flatten()
    assert torch.all(((changed >= 128) & (changed < 160)) | ((changed >= 256) & (changed < 288)))


def test_state_copy_preserves_nonfinite_payload_bits_and_cold_gather_masks_them(api):
    # Distinct NaN payload, infinity and signed zero must survive pure copies;
    # the same reused row must be ignored entirely for a cold request.
    backing, raw, (_, state) = _states(api)
    bits = torch.tensor([0x7FC00001, 0x7FA00001, -2147483648, -8388608], dtype=torch.int32)
    state[3].view(torch.int32).copy_(bits.view_as(state[3]))
    before = backing.clone()
    ids = torch.tensor([3])
    gathered = api["gather_ssm_states"](state, ids, torch.tensor([True]))
    assert torch.equal(gathered[0].view(torch.int32), bits.view_as(state[3]))
    cold = api["gather_ssm_states"](state, ids, torch.tensor([False]))
    assert torch.all(cold == 0) and torch.isfinite(cold).all()
    api["scatter_ssm_states_"](state, torch.tensor([1]), gathered)
    assert torch.equal(state[1].view(torch.int32), state[3].view(torch.int32))
    allowed = torch.zeros_like(backing, dtype=torch.bool)
    offset = state[1].data_ptr() - backing.data_ptr()
    allowed[offset : offset + state[1].numel() * state.element_size()] = True
    assert torch.all((backing == before) | allowed)
