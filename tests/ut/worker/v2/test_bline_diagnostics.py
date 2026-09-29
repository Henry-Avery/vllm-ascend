# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_hybrid_descriptor_layout import _planner_config
from test_hybrid_descriptor_layout import descriptor_api as _descriptor_api


@pytest.fixture
def descriptor_api(monkeypatch):
    return _descriptor_api.__wrapped__(monkeypatch)


ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("bline", ROOT / "vllm_ascend/worker/v2/bline_diagnostics.py")
bline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bline)


def probe_for(cache, manager_tokens, state, pages=(1,), history=2, **limits):
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1]), seq_lens=torch.tensor([history + 1]), block_tables=torch.tensor([pages])
    )
    events = []
    probe = bline.BLineDiagnostics(
        limits,
        rank=0,
        context=lambda: SimpleNamespace(attn_metadata={"mla": metadata}),
        emit=lambda line: events.append(json.loads(line)),
    )
    probe.mla["mla"] = (cache, manager_tokens, 0)
    probe.begin(False)
    probe.batch(SimpleNamespace(req_ids=["req0"]))
    return probe, events


@pytest.mark.parametrize("bad", [False, True])
def test_finite_old_state_major_overwrite_is_detected(bad):
    # Isolated scaled analogue of the old state-major mechanism: slot3 is
    # intended for manager3 but a displaced dense state view lands in MLA1.
    backing = torch.zeros(1024, dtype=torch.uint8)
    cache = torch.as_strided(backing[64:].view(torch.float32), (4, 4, 1, 4), (32, 4, 4, 1))
    state = torch.as_strided(backing[64:].view(torch.float32), (4, 4), (32, 1))
    bad_state = torch.as_strided(backing[64:].view(torch.float32), (4, 4), (4, 1), storage_offset=36)
    probe, events = probe_for(cache, 4, state)

    def update(recurrent_state, state_indices):
        recurrent_state[state_indices[0]].fill_(7)  # finite corruption, not NaN

    call = probe.wrap_kda("kda0", "kda_recurrent", update, "recurrent_state", "state_indices")
    call(bad_state if bad else state, torch.tensor([3]))
    assert any(e["status"] == "FAIL" for e in events) == bad
    assert any(e["check"] == "kda_recurrent" and e["status"] == "CHECK" for e in events) != bad


@pytest.mark.parametrize("layout_name", ["BLHNC", "LBHNC", "LBNHC"])
@pytest.mark.parametrize("ratio", [1, 3, 6])
def test_probe_consumes_real_planner_stride_and_nonzero_offset(descriptor_api, monkeypatch, layout_name, ratio):
    namespace, upstream = descriptor_api
    layout = upstream["KVCacheLayout"][layout_name]
    plan, config, mla, mamba = _planner_config(namespace, upstream, layout, ratio)
    namespace["get_current_vllm_config"] = lambda: config
    namespace["_get_attention_kv_cache_dims"] = lambda *_: (512, 64)
    layer_class = namespace["MLAAttention"]
    layers = {f"mla{i}": layer_class(impl=SimpleNamespace(fa_quant_layer=False)) for i in range(2)}
    namespace["get_layers_from_vllm_config"] = lambda *_: layers
    original = torch.zeros

    def shifted(size, *args, **kwargs):
        return (
            original(size + 128, **kwargs)[64:-64]
            if kwargs.get("dtype") == torch.int8
            else original(size, *args, **kwargs)
        )

    monkeypatch.setattr(torch, "zeros", shifted)
    raw = namespace["_allocate_kv_cache"](plan, {}, torch.device("cpu"))
    groups = [SimpleNamespace(**vars(g), kv_cache_group_id=i, backend=None) for i, g in enumerate(plan.kv_cache_groups)]
    caches = namespace["_reshape_kv_cache_v2"](groups, raw, "auto", [128, mamba.block_size], {}, plan, layout)
    cache, state = caches["mla1"], caches["kda0"][1]
    cache[ratio].fill_(2)
    probe, events = probe_for(cache, mla.block_size, state, pages=(ratio,), history=2)
    snapshots = probe.protect("kda0", "kda_recurrent", state, torch.tensor([3]))
    state[3].fill_(7)
    probe.compare("kda_recurrent", snapshots)
    assert events[-1]["status"] == "CHECK"
    # A single finite byte change in real protected history must be caught.
    cache[ratio, 0, 0, 0] = 9
    probe.compare("kda_recurrent", snapshots)
    assert events[-1]["status"] == "FAIL"
    assert events[-1]["first_bad_address"] >= cache[ratio].data_ptr()


@pytest.mark.parametrize("history,limits,reason", [(0, {}, "no history"), (2, {"max_bytes": 1}, "budget")])
def test_empty_history_and_budget_never_pass(history, limits, reason):
    backing = torch.zeros(512)
    cache = backing[:256].view(4, 4, 1, 16)
    state = backing.view(4, 128)
    probe, _ = probe_for(cache, 4, state, history=history, **limits)
    with pytest.raises(bline.NotCovered, match=reason):
        probe.protect("kda", "kda_recurrent", state, torch.tensor([3]))


def test_invalid_ownership_pad_and_invalid_page():
    backing = torch.zeros(512)
    cache = backing[:256].view(4, 4, 1, 16)
    state = backing.view(4, 128)
    probe, _ = probe_for(cache, 4, state)
    with pytest.raises(ValueError, match="ownership"):
        probe.protect("kda", "kda_recurrent", state, torch.tensor([1]))
    with pytest.raises(bline.NotApplicable, match="no writable"):
        probe.protect("kda", "kda_conv", state, torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match="state index"):
        probe.protect("kda", "kda_recurrent", state, torch.tensor([4]))
    probe, _ = probe_for(cache, 4, state, pages=(-1,))
    with pytest.raises(ValueError, match="MLA page"):
        probe.protect("kda", "kda_recurrent", state, torch.tensor([3]))


@pytest.mark.parametrize("bad", [False, True])
def test_writer_compares_real_post_normalization_values(bad):
    cache = torch.zeros(4, 4, 1, 4)
    probe, events = probe_for(cache, 4, cache)
    slots = torch.tensor([5, -1])

    def writer(values, caches, slots):
        latent, rope = values[..., :2] * 3, values[..., 2:]
        caches[0][1, 1] = latent[0] + int(bad)
        caches[1][1, 1] = rope[0]
        return rope, latent

    wrapped = probe.wrap_writer("mla", writer)
    wrapped(torch.ones(2, 1, 4), (cache[..., :2], cache[..., 2:]), slots)
    assert events[-1]["status"] == ("FAIL" if bad else "CHECK")


@pytest.mark.parametrize("padded", [False, True])
def test_raw_logits_before_processor_both_row_paths(padded):
    probe, events = probe_for(torch.zeros(4, 4, 1, 4), 4, None)
    model = SimpleNamespace(compute_logits=lambda h: h.clone())
    batch = SimpleNamespace(logits_indices=torch.tensor([0, 1]))
    hidden = torch.ones(4 if padded else 2, 8)
    hidden[2:] = torch.nan  # ignored LM-head capacity padding
    with probe.logits_scope(model, batch):
        logits = model.compute_logits(hidden)
        logits[:2].fill_(torch.nan)  # simulate a later processor, outside the probe
    assert any(e["check"] == "raw_logits" and e["status"] == "CHECK" for e in events)
    assert not any(e["status"] == "FAIL" for e in events)
    probe.begin(False)
    with probe.logits_scope(model, batch):
        model.compute_logits(torch.full((2, 8), torch.inf))
    assert any(e["check"] == "raw_logits" and e["status"] == "FAIL" for e in events)


def test_dummy_does_not_spend_budget_and_step_limit_is_uncovered():
    probe = bline.BLineDiagnostics({"max_steps": 1}, rank=0, context=lambda: None, emit=lambda _: None)
    probe.begin(True)
    assert probe.step == 0 and not probe.active
    probe.begin(False)
    assert probe.step == 1
    probe.begin(False)
    assert not probe.active and probe.counts["step_budget:UNCOVERED"] == 1


def test_disabled_install_has_no_mutation():
    runner = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={}))
    before = vars(runner).copy()
    bline.install_bline_diagnostics(runner, lambda: None)
    assert vars(runner) == before


def test_event_budget_retains_incomplete_marker():
    events = []
    probe = bline.BLineDiagnostics(
        {"max_events": 1}, rank=0, context=lambda: None, emit=lambda line: events.append(json.loads(line))
    )
    probe.begin(False)
    probe.batch(SimpleNamespace(req_ids=["req"]))
    assert events[-1]["check"] == "event_budget" and events[-1]["incomplete_step"]
    assert not probe.active


@pytest.mark.parametrize("padded", [False, True])
def test_instance_hooks_cover_request_order_and_both_sample_paths(caplog, padded):
    """Run complete installed hooks across cold, history-prefill and decode steps."""
    backing = torch.zeros(512)
    cache = backing[:256].view(4, 4, 1, 16)
    state = backing.view(4, 128)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1]),
        seq_lens=torch.tensor([1]),
        block_tables=torch.tensor([[1]]),
        num_prefills=1,
        num_decodes=0,
        keep_meta=None,
    )
    context = lambda: SimpleNamespace(attn_metadata={"mla": metadata, "kda": metadata})
    spec_class = type("AscendMLAAttentionSpec", (), {})
    mla_spec = spec_class()
    mla_spec.block_size, mla_spec.num_heads, mla_spec.num_query_heads = 4, 1, 12

    class KDA:
        kv_cache = (state, state)

        @staticmethod
        def _run_causal_conv1d(
            mixed_qkv,
            conv_weights_t,
            conv_state,
            query_start_loc,
            cache_indices,
            initial_state_mode,
            *,
            run_mode,
            max_query_len=-1,
            num_accepted_tokens=None,
        ):
            conv_state[cache_indices[0]].fill_(7)
            return mixed_qkv

        def _run_prefill(
            self, q, k, v, raw_gate, beta, recurrent_state, state_indices, has_initial_state, prebuilt_metadata
        ):
            recurrent_state[state_indices[0]].fill_(7)
            return q

        def _run_recurrent(
            self, q, k, v, raw_gate, beta, recurrent_state, cu_seqlens, state_indices, *, num_accepted_tokens=None
        ):
            recurrent_state[state_indices[0]].fill_(7)
            return q

    def writer(kv_no_split, kv_cache, slots):
        latent, rope = kv_no_split[..., :8] * 3, kv_no_split[..., 8:]
        page, token = divmod(int(slots[0]), 4)
        kv_cache[0][page, token] = latent[0]
        kv_cache[1][page, token] = rope[0]
        return rope, latent

    def prefill(q_nope, q_pe, k_nope, k_pe, value, kv_c_and_k_pe_cache, attn_metadata):
        return q_nope

    def external(preprocessed, fused_cache, metadata):
        return preprocessed

    kda = KDA()
    impl = SimpleNamespace(_exec_kv_no_rope=writer, _forward_prefill=prefill, _forward_external_flashmla=external)
    mla = SimpleNamespace(kv_cache=cache, impl=impl)
    parallel = SimpleNamespace(
        rank=0, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, tensor_parallel_size=1
    )
    config = SimpleNamespace(
        additional_config={"bline_diagnostics": {"enabled": True}},
        parallel_config=parallel,
        speculative_config=None,
        kv_transfer_config=None,
    )
    model = SimpleNamespace(compute_logits=lambda x: x.clone())
    original_logits = model.compute_logits
    batch = SimpleNamespace(req_ids=["real-request"], logits_indices=torch.tensor([0]))
    runner = SimpleNamespace(
        vllm_config=config,
        model_config=SimpleNamespace(enforce_eager=True),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "LBNHC"),
        compilation_config=SimpleNamespace(static_forward_context={"mla": mla, "kda": kda}),
        kv_cache_config=SimpleNamespace(
            kv_cache_tensors=[],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["mla"], kv_cache_spec=mla_spec),
                SimpleNamespace(layer_names=["kda"], kv_cache_spec=SimpleNamespace()),
            ],
        ),
        model=model,
    )
    runner.prepare_inputs = lambda: batch

    def execute(scheduler_output, dummy_run=False, is_profile=False):
        runner.prepare_inputs()
        q, ids, starts = torch.ones(1, 8), torch.tensor([3]), torch.tensor([0, 1])
        flags = torch.tensor([True])
        kda._run_causal_conv1d(q, q, state, starts, ids, flags, run_mode=int(not metadata.num_prefills))
        if metadata.num_prefills:
            kda._run_prefill(q, q, q, q, q, state, ids, flags, metadata)
        else:
            kda._run_recurrent(q, q, q, q, q, state, starts, ids)
        views = (cache[..., :8], cache[..., 8:])
        impl._exec_kv_no_rope(torch.ones(1, 1, 16), views, torch.tensor([4 + int(metadata.seq_lens[0]) - 1]))
        if metadata.num_prefills:
            impl._forward_prefill(q, q, q, q, q, views, metadata)
        else:
            impl._forward_external_flashmla(q, cache, metadata)
        return "unchanged return"

    def sample(hidden, input_batch, grammar_output):
        rows = torch.cat((hidden, torch.full((3, 8), torch.nan))) if padded else hidden
        logits = model.compute_logits(rows)[:1]
        return logits

    runner.execute_model, runner.sample = execute, sample
    bline.install_bline_diagnostics(runner, context)
    schedule = SimpleNamespace(total_num_scheduled_tokens=1)
    runner.execute_model(schedule, dummy_run=True)
    runner.execute_model(schedule, is_profile=True)
    assert runner._bline_diagnostics.step == 0
    for history, prefill_phase in ((0, True), (2, True), (3, False)):
        metadata.seq_lens[0] = history + 1
        metadata.num_prefills, metadata.num_decodes = int(prefill_phase), int(not prefill_phase)
        assert runner.execute_model(schedule) == "unchanged return"
        assert torch.equal(runner.sample(torch.ones(1, 8), batch, None), torch.ones(1, 8))
        assert model.compute_logits is original_logits
    events = [
        json.loads(record.message.split("BLINE ", 1)[1]) for record in caplog.records if "BLINE " in record.message
    ]
    assert any(e["status"] == "NOT_APPLICABLE" for e in events)
    assert not any(e["status"] in ("FAIL", "UNCOVERED") for e in events)
    result = load_cli().summarize(events, [0])
    assert result["status"] == "CHECK"
    assert result["ranks"][0]["checked_steps"] == [1, 2, 3]


def load_cli():
    spec = importlib.util.spec_from_file_location("bline_cli", ROOT / "tools/bline/validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prefix_keeps_nested_dimensions_quantization_and_one_based_mapping():
    cli = load_cli()
    config = dict(
        text_config=dict(
            num_hidden_layers=8,
            hidden_size=777,
            quantization_config={"bits": 4},
            linear_attn_config={"kda_layers": [1, 2, 3, 5, 6, 7]},
        ),
        vision_config={"dim": 33},
    )
    override, expected, types = cli.prefix_override(config, 4)
    assert override == {"text_config": {"num_hidden_layers": 4}}
    assert types[:4] == ["KDA", "KDA", "KDA", "MLA"]
    assert config["text_config"]["num_hidden_layers"] == 8
    assert expected["text_config"]["hidden_size"] == 777
    assert expected["text_config"]["quantization_config"] == {"bits": 4}
    config["text_config"]["linear_attn_config"]["kda_layers"] = [0, 1, 2]
    with pytest.raises(ValueError, match="prefix"):
        cli.prefix_override(config, 4)


def test_summary_requires_ranks_routes_and_retains_uncovered():
    cli = load_cli()
    events = [dict(rank=0, run_id="test", step=1, status="CHECK", check=check) for check in cli.REQUIRED_CHECKS]
    assert cli.summarize(events, [0])["status"] == "UNCOVERED"  # Truncated records lack binding and step boundaries.
    assert cli.summarize(events, [0, 1])["status"] == "UNCOVERED"
    events.append(dict(rank=0, run_id="test", step=1, status="UNCOVERED", check="step_budget"))
    assert cli.summarize(events, [0])["status"] == "UNCOVERED"
    events.append(dict(rank=0, run_id="test", step=1, status="FAIL", check="writer"))
    assert cli.summarize(events, [0, 1])["status"] == "FAIL"


def test_prefill_keep_meta_limits_actual_state_ownership():
    backing = torch.zeros(512)
    cache = backing[:256].view(4, 4, 1, 16)
    state = backing.view(4, 128)
    probe, events = probe_for(cache, 4, state)
    metadata = SimpleNamespace(keep_meta=torch.tensor([1]))

    def prefill(recurrent_state, state_indices, prebuilt_metadata):
        recurrent_state[state_indices[prebuilt_metadata.keep_meta][0]].fill_(7)

    wrapped = probe.wrap_kda("kda", "kda_prefill", prefill, "recurrent_state", "state_indices")
    wrapped(state, torch.tensor([1, 3]), metadata)
    assert not any(event["status"] == "FAIL" for event in events)
    assert events[-1]["status"] == "CHECK"
    mapping = next(event for event in events if event["check"] == "protected_mapping")
    assert mapping["protected"][0]["state_ids"] == [3]


def test_publisher_cli_direct_execution_avoids_tools_bisect_shadow():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/bline/validate.py"), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "manifest" in result.stdout and "summary" in result.stdout


def test_delayed_arm_preserves_budget_and_same_cache_until_target_round(tmp_path):
    marker = tmp_path / "round7.arm"
    events = []
    probe = bline.BLineDiagnostics(
        {"arm_file": str(marker), "max_steps": 2},
        rank=0,
        context=lambda: None,
        emit=lambda line: events.append(json.loads(line)),
    )
    initial_bytes = probe.remaining
    for _ in range(500):
        probe.begin(False)
        probe.finite("raw_logits", torch.tensor([float("nan")]))
        assert not probe.active
    assert probe.step == 0 and probe.remaining == initial_bytes and not events
    marker.touch()
    probe.begin(True)  # warmup/idle must not arm or spend a step
    assert not probe.armed
    probe.begin(False)
    assert probe.active and probe.step == 1
    probe.finite("raw_logits", torch.tensor([float("nan")]))
    assert any(e["check"] == "raw_logits" and e["status"] == "FAIL" for e in events)
    marker.unlink()  # arming is latched for this run; never resets budgets
    probe.begin(False)
    assert probe.active and probe.step == 2
    probe.begin(False)
    assert not probe.active and probe.step == 2
    assert sum(e["check"] == "armed" for e in events) == 1
    assert events[-1]["check"] == "step_budget"


@pytest.mark.parametrize("path", ["relative.arm", "", 12, False])
def test_arm_path_must_be_absolute(path):
    with pytest.raises(ValueError, match="absolute"):
        bline.BLineDiagnostics({"arm_file": path}, rank=0, context=lambda: None)
