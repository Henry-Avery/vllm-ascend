# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests, runnable without vLLM or the NPU extension installed.

Use --confcutdir=tests/ut/attention for this standalone adapter suite. Loading
the leaf module isolates it from the engine/plugin initialization in __init__.
The ordinary repository UT configuration can also run these tests.
"""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/flashmla.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


@pytest.fixture
def inputs():
    return dict(
        q=torch.empty(3, 64, 576, dtype=torch.bfloat16),
        k_cache=torch.empty(2, 128, 1, 576, dtype=torch.bfloat16),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cache_seqlens=torch.tensor([12, 16], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 3], dtype=torch.int32),
        seqused_q=torch.tensor([1, 2], dtype=torch.int32),
        metadata=torch.empty(4096, dtype=torch.int32),
    )


def make_adapter(api, **kwargs):
    config = api.FlashMLAConfig(num_heads=64, softmax_scale=0.125, mask_mode=0, **kwargs)
    return api.FlashMLAAdapter(config, Mock(), Mock())


@pytest.mark.parametrize("heads", (8, 12, 64, 96))
def test_local_tp_head_counts_pass_to_both_operators(api, inputs, heads):
    adapter = api.FlashMLAAdapter(api.FlashMLAConfig(heads, 0.125, mask_mode=0), Mock(), Mock())
    inputs["q"] = torch.empty(3, heads, 576, dtype=torch.bfloat16)
    adapter.metadata_op.return_value = inputs["metadata"]
    adapter.build_metadata(inputs["cache_seqlens"], inputs["cu_seqlens_q"], inputs["seqused_q"])
    adapter.attention(**inputs)
    assert adapter.metadata_op.call_args.kwargs["num_heads_q"] == heads
    assert adapter.attention_op.call_args.args[0] is inputs["q"]


def test_lengths_and_attributes_match_both_operators(api, inputs):
    adapter = make_adapter(api, return_softmax_lse=True)
    adapter.metadata_op.return_value = inputs["metadata"]
    generated = adapter.build_metadata(inputs["cache_seqlens"], inputs["cu_seqlens_q"], inputs["seqused_q"])
    assert generated is inputs["metadata"]
    result = adapter.attention(**inputs)
    assert result is adapter.attention_op.return_value
    meta_args, meta_kwargs = adapter.metadata_op.call_args
    attn_args, attn_kwargs = adapter.attention_op.call_args
    assert meta_args[0] is attn_kwargs["cache_seqlens"] is inputs["cache_seqlens"]
    assert attn_args[0] is inputs["q"]
    assert attn_args[1] is inputs["k_cache"]
    for key in ("cu_seqlens_q", "seqused_q"):
        assert meta_kwargs[key] is attn_kwargs[key] is inputs[key]
    for key, value in {"max_seqlen_q": -1, "max_seqlen_kv": -1, "mask_mode": 0, "layout_q": "TND"}.items():
        assert meta_kwargs[key] == attn_kwargs[key] == value
    assert meta_kwargs["num_heads_q"] == 64
    assert meta_kwargs["num_heads_kv"] == 1
    assert meta_kwargs["head_dim_qk"] == 576
    assert meta_kwargs["head_dim_v"] == attn_kwargs["head_dim_v"] == 512
    assert attn_kwargs["layout_out"] == "NTD"
    assert attn_kwargs["layout_kv"] == "PA_BBND"
    assert attn_kwargs["softmax_scale"] == 0.125
    assert attn_kwargs["return_softmax_lse"] is True
    assert attn_kwargs["metadata"] is generated


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("token_stride", [576, 608])
def test_noncontiguous_cache_is_forwarded_with_offset_and_storage_intact(api, inputs, dtype, token_stride):
    adapter = make_adapter(api)
    page_stride = 128 * token_stride + 256
    backing = torch.full((2 * page_stride + 64,), 7, dtype=dtype)
    cache = backing.as_strided((2, 128, 1, 576), (page_stride, token_stride, 576, 1), 32)
    inputs.update(q=inputs["q"].to(dtype), k_cache=cache)
    adapter.attention(**inputs)
    passed_cache = adapter.attention_op.call_args.args[1]
    assert passed_cache is cache
    assert passed_cache.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert passed_cache.storage_offset() == 32
    assert passed_cache.stride() == (page_stride, token_stride, 576, 1)
    assert torch.all(backing == 7)


def test_optional_query_used_lengths_are_not_synthesized_on_host(api, inputs):
    adapter = make_adapter(api)
    inputs["seqused_q"] = None
    adapter.build_metadata(inputs["cache_seqlens"], inputs["cu_seqlens_q"])
    adapter.attention(**inputs)
    assert adapter.metadata_op.call_args.kwargs["seqused_q"] is None
    assert adapter.attention_op.call_args.kwargs["seqused_q"] is None


def test_meta_capacity_query_uses_package_instead_of_a_hardcoded_formula(api):
    adapter = make_adapter(api)
    schedule = torch.empty(8192, dtype=torch.int32, device="meta")
    adapter.metadata_op.return_value = schedule
    result = adapter.build_metadata(
        torch.empty(2, dtype=torch.int32, device="meta"),
        torch.empty(3, dtype=torch.int32, device="meta"),
    )
    assert result is schedule


def test_nz_layout_is_explicit_and_never_reshapes_cache(api, inputs):
    adapter = make_adapter(api, layout_kv="PA_NZ")
    inputs["k_cache"] = torch.empty(2, 1, 36, 128, 16, dtype=torch.bfloat16)
    adapter.attention(**inputs)
    assert adapter.attention_op.call_args.args[1] is inputs["k_cache"]
    assert adapter.attention_op.call_args.kwargs["layout_kv"] == "PA_NZ"


@pytest.mark.parametrize(
    "override, match",
    [({"num_heads": 16}, "local Q heads"), ({"mask_mode": 1}, "mask_mode"), ({"layout_kv": "PA_Nz"}, "layout_kv")],
)
def test_unsupported_config_is_rejected(api, override, match):
    config = dict(num_heads=64, softmax_scale=1.0)
    config.update(override)
    with pytest.raises(ValueError, match=match):
        api.FlashMLAConfig(**config)


@pytest.mark.parametrize(
    "name, replacement, match",
    [
        ("q", lambda: torch.empty(3, 64, 512, dtype=torch.bfloat16), "q must"),
        ("q", lambda: torch.empty(3, 64, 576), "BF16 or FP16"),
        ("k_cache", lambda: torch.empty(2, 1, 128, 576, dtype=torch.bfloat16), "cache must"),
        ("k_cache", lambda: torch.empty(2, 128, 1, 576, dtype=torch.float16), "same dtype"),
        ("block_table", lambda: torch.empty(2, 0, dtype=torch.int32), "table_capacity"),
        ("block_table", lambda: torch.empty(2, 1, dtype=torch.int64), "block_table must be int32"),
        ("cache_seqlens", lambda: torch.empty(2, dtype=torch.int64), "cache_seqlens requires"),
        ("cu_seqlens_q", lambda: torch.empty(2, dtype=torch.int32), "cu_seqlens_q requires"),
        ("seqused_q", lambda: torch.empty(3, dtype=torch.int32), "seqused_q requires"),
        ("metadata", lambda: torch.empty(0, dtype=torch.int32), "metadata must"),
        ("metadata", lambda: torch.empty(8, dtype=torch.int64), "metadata must"),
        ("metadata", lambda: torch.empty(8, dtype=torch.int32, device="meta"), "same device"),
        ("attn_mask", lambda: torch.empty(2048, 2048, dtype=torch.int8), "attn_mask=None"),
    ],
)
def test_invalid_input_fails_before_calling_operator(api, inputs, name, replacement, match):
    adapter = make_adapter(api)
    inputs[name] = replacement()
    with pytest.raises(ValueError, match=match):
        adapter.attention(**inputs)
    adapter.attention_op.assert_not_called()


def test_causal_mask_is_required_and_kept_identical(api, inputs):
    adapter = api.FlashMLAAdapter(api.FlashMLAConfig(64, 0.125), Mock(), Mock())
    with pytest.raises(ValueError, match="requires the upper-triangular"):
        adapter.attention(**inputs)
    with pytest.raises(ValueError, match="attn_mask requires"):
        adapter.attention(**inputs, attn_mask=torch.empty(2048, 2048, dtype=torch.bool))
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)
    adapter.attention(**inputs, attn_mask=mask)
    assert adapter.attention_op.call_args.kwargs["attn_mask"] is mask
    assert adapter.attention_op.call_args.kwargs["mask_mode"] == 3


def test_loader_imports_public_package_only_when_requested(api):
    ops = SimpleNamespace(flash_mla_with_kvcache=Mock(), flash_mla_with_kvcache_metadata=Mock())
    importer = Mock(return_value=ops)
    with patch.dict(api.FlashMLAAdapter.load.__func__.__globals__, import_module=importer):
        config = api.FlashMLAConfig(96, 1.0)
        importer.assert_not_called()
        adapter = api.FlashMLAAdapter.load(config)
    importer.assert_called_once_with("cann_ops_transformer.ops")
    assert adapter.attention_op is ops.flash_mla_with_kvcache
    assert adapter.metadata_op is ops.flash_mla_with_kvcache_metadata


@pytest.mark.parametrize("missing", ["package", "symbol"])
def test_missing_package_or_symbol_has_actionable_error(api, missing):
    importer = Mock(side_effect=ImportError("absent")) if missing == "package" else Mock(return_value=SimpleNamespace())
    with (
        patch.dict(api.FlashMLAAdapter.load.__func__.__globals__, import_module=importer),
        pytest.raises(RuntimeError, match="Install a package matching"),
    ):
        api.FlashMLAAdapter.load(api.FlashMLAConfig(64, 1.0))


def test_kernel_error_propagates_without_fallback_or_retry(api, inputs):
    adapter = make_adapter(api)
    adapter.attention_op.side_effect = RuntimeError("unsupported token stride")
    with pytest.raises(RuntimeError, match="unsupported token stride"):
        adapter.attention(**inputs)
    assert adapter.attention_op.call_count == 1


def test_short_prefill_uses_real_phase_and_rejects_interleaved_requests(api):
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        is_prefilling=torch.tensor([False, True, True]),
    )
    assert api.split_flashmla_requests(common) == (1, 2, 1, 2)
    common.is_prefilling = torch.tensor([True, False, True])
    with pytest.raises(RuntimeError, match="runner ordering"):
        api.split_flashmla_requests(common)


def test_graph_padding_does_not_add_requests_or_require_stage_flags(api):
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 8], dtype=torch.int32),
        is_prefilling=torch.tensor([False, False]),
    )
    assert api.split_flashmla_requests(common) == (2, 0, 2, 0)
    common.is_prefilling = torch.tensor([False, True])
    assert api.split_flashmla_requests(common) == (1, 1, 1, 1)
    common.num_actual_tokens = 8
    with pytest.raises(RuntimeError, match="active request batch"):
        api.split_flashmla_requests(common)


def test_diagnostic_switch_is_explicit_and_off_by_default(monkeypatch):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/envs.py"
    env = runpy.run_path(str(path))
    name = "VLLM_ASCEND_FLASH_MLA_TRACE"
    read = env["env_variables"][name]
    monkeypatch.delenv(name, raising=False)
    assert read() is False
    for value, expected in (("0", False), ("1", True)):
        monkeypatch.setenv(name, value)
        assert read() is expected
    for invalid in ("true", "-1", "2", ""):
        monkeypatch.setenv(name, invalid)
        with pytest.raises(ValueError):
            read()
