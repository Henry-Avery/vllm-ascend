# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of DSpark's actual integration methods; no NPU numerical claim."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from . import test_flashmla_metadata_lifecycle as lifecycle
from .test_flashmla_metadata_lifecycle import common, tensor_fields

runtime = lifecycle.runtime

ROOT = Path(__file__).resolve().parents[3]
SPECULATOR = "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"
MLA = "vllm_ascend/attention/mla_v1.py"


def load_class(path, name, methods, namespace, base=object):
    original = next(n for n in ast.parse((ROOT / path).read_text()).body if getattr(n, "name", None) == name)
    selected = [n for n in original.body if getattr(n, "name", None) in methods]
    assert len(selected) == len(methods)
    cls = ast.ClassDef(
        name=name, bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=selected, decorator_list=[]
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace["Parent"] = base
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), path, "exec"), namespace)
    return namespace[name]


def load_scope():
    path = ROOT / "vllm_ascend/worker/v2/attn_utils.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) == "flashmla_metadata_scope")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"contextmanager": contextmanager}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace["flashmla_metadata_scope"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_draft_capture_binds_and_releases_only_its_metadata_executor(enabled, fail):
    state = SimpleNamespace(executor=None, defer=False)
    groups = [[SimpleNamespace(get_metadata_builder=lambda _: SimpleNamespace(flashmla_state=state))]]
    released = []
    executor = SimpleNamespace(submission_in_flight=True, release=lambda: released.append(True)) if enabled else None

    class Parent:
        def capture(self):
            assert state.executor is executor and state.defer == enabled
            if fail:
                raise ValueError("capture failed")

    cls = load_class(
        SPECULATOR, "AscendDSparkSpeculator", ["capture"], {"flashmla_metadata_scope": load_scope()}, Parent
    )
    speculator = cls()
    speculator.attn_groups, speculator.flashmla_executor = groups, executor
    if fail:
        with pytest.raises(ValueError, match="capture failed"):
            speculator.capture()
    else:
        speculator.capture()
    assert state.executor is None and not state.defer
    assert released == ([True] if enabled else [])


def test_fia_query_length_updates_leave_external_metadata_untouched():
    cls = load_class(SPECULATOR, "AscendDSparkSpeculator", ["_update_draft_attn_metadata"], {})
    speculator = cls()
    speculator.num_query_per_req = 7
    speculator.attn_architecture = "MLA"
    external = SimpleNamespace(external_flashmla=object(), decode=SimpleNamespace(actual_seq_lengths_q=None))
    fia = SimpleNamespace(decode=SimpleNamespace(actual_seq_lengths_q=[7]))
    speculator._update_draft_attn_metadata({"external": external, "fia": fia}, 2)
    assert external.decode.actual_seq_lengths_q is None
    assert fia.decode.actual_seq_lengths_q == [7, 14]


def test_external_draft_never_enters_fia_graph_parameter_updates():
    cls = load_class(
        MLA,
        "AscendMLAImpl",
        ["update_graph_params"],
        dict(
            _EXTRA_CTX=SimpleNamespace(is_draft_model=True, is_draft_model_prefill=False),
            get_draft_graph_params=lambda: object(),
        ),
    )
    metadata = {"draft": SimpleNamespace(decode=object(), external_flashmla=object())}
    # No FIA graph handles or stream are needed when every draft layer is external.
    cls.update_graph_params(None, None, 14, draft_attn_metadatas=[metadata])


@pytest.mark.parametrize("causal,width", [(True, 8), (False, 7)])
def test_verification_and_draft_replay_refresh_captured_addresses(runtime, causal, width):
    builder = runtime.builder()
    first = common([31, 47], [0, width, 2 * width], tokens=2 * width)
    first.causal = causal
    captured = builder.build(first, 2, 2 * width, False, retain_for_graph=True)
    addresses = {name: value.data_ptr() for name, value in tensor_fields(captured).items()}
    old_schedule = captured.schedule.clone()
    # Only one live request on replay; device query boundaries retain padded width.
    next_batch = common(
        [93, 0],
        [0, width, 2 * width],
        tokens=2 * width,
        blocks=[[7, 9], [0, 0]],
        positions=list(range(70, 70 + 2 * width)),
    )
    next_batch.causal = causal
    builder.defer = True
    replay = builder.build(next_batch, 1, width, False)
    assert replay is captured
    torch.testing.assert_close(replay.schedule, old_schedule)
    (task,) = builder.take_tasks()
    task.run()
    assert addresses == {name: value.data_ptr() for name, value in tensor_fields(replay).items()}
    assert replay.used_q.tolist() == [width, 0, 0, 0, 0]
    assert replay.cache_lens.tolist() == [93, 0, 0, 0, 0]
    assert replay.block_table[0].tolist() == [7, 9]
    assert replay.slots[width:].tolist() == [-1] * width
    assert replay.adapter.config.mask_mode == (3 if causal else 0)
    assert (replay.attn_mask is None) == (not causal)
    assert runtime.metadata.call_args.kwargs["mask_mode"] == (3 if causal else 0)


@pytest.mark.parametrize("fused", [False, True])
def test_k3_draft_context_writer_receives_component_views_of_same_backing(fused):
    # Same page-strided BBND geometry produced by the pinned dependency allocator.
    page_stride, pages, block_size, latent_dim, rope_dim = 81408, 6, 128, 512, 64
    backing = torch.zeros(pages * page_stride, dtype=torch.bfloat16)
    cache = backing.as_strided(
        (pages, block_size, 1, latent_dim + rope_dim), (page_stride, latent_dim + rope_dim, latent_dim + rope_dim, 1)
    )
    expected_nope, expected_rope = cache[..., :latent_dim], cache[..., latent_dim:]
    calls = []

    def writer(kv, weight, cos, sin, slots, rope_cache, nope_cache, **kwargs):
        assert rope_cache.shape == expected_rope.shape
        assert nope_cache.shape == expected_nope.shape
        for actual, expected in [(rope_cache, expected_rope), (nope_cache, expected_nope)]:
            assert actual.data_ptr() == expected.data_ptr() and actual.stride() == expected.stride()
        page, token = divmod(int(slots[0]), block_size)
        nope_cache[page, token].fill_(3)
        rope_cache[page, token].fill_(5)
        calls.append(True)
        return None, None, kv[..., latent_dim:], kv[..., :latent_dim]

    impl_cls = load_class(
        MLA,
        "AscendMLAImpl",
        ["exec_kv_prefill"],
        dict(
            torch=torch,
            torch_npu=SimpleNamespace(npu_kv_rmsnorm_rope_cache=writer),
        ),
    )
    impl = SimpleNamespace(
        use_mla_rope=True,
        pcp_enabled=False,
        num_kv_heads=1,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        support_fp8_attention=False,
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(latent_dim), variance_epsilon=1e-6),
    )
    impl.exec_kv_prefill = MethodType(impl_cls.exec_kv_prefill, impl)
    attn = SimpleNamespace(
        q_lora_rank=1,
        impl=impl,
        kv_cache=cache if fused else (expected_nope, expected_rope),
        fused_qkv_a_proj=lambda x: (torch.cat((torch.zeros(1, 1), x), dim=-1),),
    )
    model_cls = load_class(
        "vllm_ascend/models/kimi_k3_dspark.py",
        "AscendK3DSparkModel",
        ["precompute_and_store_context_kv"],
        dict(torch=torch, get_cos_and_sin_mla=lambda pos: (torch.zeros(1, 1, 1, rope_dim),) * 2),
    )
    model = model_cls()
    model.layers = [SimpleNamespace(self_attn=attn)]
    model.precompute_and_store_context_kv(torch.ones(1, latent_dim + rope_dim), torch.tensor([8]), torch.tensor([257]))
    assert calls == [True]
    assert torch.all(cache[2, 1, 0, :latent_dim] == 3)
    assert torch.all(cache[2, 1, 0, latent_dim:] == 5)
