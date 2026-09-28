# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions: pytest --confcutdir=tests/ut/worker/v2 <this file>."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def api():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/attention/flashmla_chunk_diagnostics.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


def make_case(api, tmp_path, **config):
    sampling = SimpleNamespace(
        directory=tmp_path,
        identity={"dp_rank": 0, "tp_rank": 0},
        config=SimpleNamespace(steps=2),
        capture_check=lambda: False,
    )
    owner = api.ChunkDiagnostics(api.ChunkDiagnosticConfig(**config), sampling)
    owner.step = 1
    batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=0,
        req_ids=["decode", "continued", "new"],
        query_start_loc_np=np.array([0, 1, 3, 4]),
        num_computed_prefill_tokens_np=np.array([2, 130, 0]),
        is_prefilling_np=np.array([False, True, True]),
        idx_mapping_np=np.array([4, 2, 7]),
        idx_mapping=torch.tensor([4, 2, 7]),
        seq_lens_np=np.array([3, 132, 1]),
        prefill_len_np=np.array([2, 132, 1]),
        positions=torch.tensor([2, 130, 131, 0]),
    )
    fused = torch.randn(4, 128, 1, 5, generator=torch.Generator().manual_seed(5))
    cache = (fused[..., :3], fused[..., 3:])
    table = torch.tensor([[3, 0], [1, 2], [0, 0]])
    chunk = SimpleNamespace(
        starts=torch.tensor([[0, 0], [128, 128]]),
        chunk_seq_lens_npu=torch.tensor([[128, 0], [2, 0]]),
        chunk_seq_lens=torch.tensor([[128, 0], [2, 0]]),
        chunk_actual_seq_lengths_kv_list=[[128, 128], [2, 2]],
    )
    meta = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        num_actual_tokens=4,
        num_prefills=2,
        query_start_loc=torch.tensor([0, 1, 3, 4]),
        seq_lens=torch.tensor([3, 132, 1]),
        slot_mapping=torch.tensor([386, 258, 259, 0]),
        block_tables=table,
        causal=True,
        prefill=SimpleNamespace(chunked_context=chunk, block_table=table[1:], actual_seq_lengths_q=[2, 3]),
    )
    observed = api.DiagnosticBatch(owner, batch)
    layer = observed.begin_layer(SimpleNamespace(scale=0.4), "model.layers.0.attn", meta, cache)
    return owner, batch, layer


def records(owner):
    return [json.loads(line) for line in (owner.directory / "events.jsonl").read_text().splitlines()]


def exercise_prefill(api, layer, corrupt=None):
    slots = layer.meta.slot_mapping[1:]
    latent, positional = [cache[slots // 128, slots % 128].clone() for cache in layer.cache]
    if corrupt == "writer":
        latent[0, 0, 0] += 1
    layer.writer(latent, positional)
    query = torch.randn(3, 2, 5, generator=torch.Generator().manual_seed(9))
    key, rope = latent.expand(-1, 2, -1), positional.expand(-1, 2, -1)
    current = [
        api.reference_attention(
            query[a:b], torch.cat((key[a:b], rope[a:b]), -1), key[a:b], 0.4, causal_positions=list(range(b - a))
        )
        for a, b in ((0, 2), (2, 3))
    ]
    out, lse = (torch.cat([branch[j] for branch in current]) for j in (0, 1))
    if corrupt == "current":
        out[0, 0, 0] += 1
    layer.current(query[..., :3], query[..., 3:], key, rope, key, out, lse.T)
    branches = [current]
    for index, (start, length) in enumerate(((0, 128), (128, 2))):
        gathered = [api.read_paged(cache, layer.table[1], start, length) for cache in layer.cache]
        if corrupt == "gather" and index == 1:
            gathered[0][0, 0, 0] += 1
        layer.gather(index, *gathered)
        k, p = [value.expand(-1, 2, -1) for value in gathered]
        result = api.reference_attention(query[:2], torch.cat((k, p), -1), k, 0.4)
        empty = (torch.full((1, 2, 3), torch.nan), torch.full((1, 2), -torch.inf))
        h_out = torch.cat((result[0], empty[0]))
        h_lse = torch.cat((result[1], empty[1]))
        if corrupt == "history" and index == 0:
            h_out[0, 0, 0] += 1
        layer.history(index, k, p, k, h_out, h_lse[..., None])
        branches.append([result, empty])
    merged = torch.cat([api.reference_merge([part[req] for part in branches])[0] for req in range(2)])
    if corrupt == "merge":
        merged[0, 0, 0] += 1
    layer.merged(merged)
    projection = torch.ones(4, 6)
    if corrupt == "projection":
        projection[3, 0] = torch.nan
    layer.finish(projection, torch.zeros(4, 10))


@pytest.mark.parametrize(
    "corrupt,stage",
    [
        (None, None),
        ("writer", "writer/"),
        ("gather", "gather/"),
        ("current", "current/"),
        ("history", "history/"),
        ("merge", "merge/"),
        ("projection", "projection_input/"),
    ],
)
def test_mixed_prefill_cross_page_and_empty_history(api, tmp_path, corrupt, stage):
    owner, _, layer = make_case(api, tmp_path)
    before = [value.clone() for value in layer.cache]
    exercise_prefill(api, layer, corrupt)
    events = records(owner)
    failures = [event for event in events if event["event"] == "comparison" and not event["passed"]]
    if stage is None:
        assert failures == []
        assert events[-1]["skipped"] == 0
    else:
        assert any(event["stage"].startswith(stage) for event in failures)
    assert events[-1]["failed"] == (stage is not None)
    assert events[-1]["tensor_file"]
    assert torch.isnan(layer.tensors["history/0/2/empty_output"]).all()
    assert any(event.get("stage") == "merge/2/empty_history_identity" for event in events)
    assert all(torch.equal(old, new) for old, new in zip(before, layer.cache))
    assert layer.cache[1].storage_offset() == 3
    assert not layer.cache[0].is_contiguous()


def test_independent_paged_reader_and_exact_large_integers(api):
    cache = torch.arange(3 * 128 * 5).view(3, 128, 1, 5)[..., 3:]
    read = api.read_paged(cache, [2, 0], 127, 2)
    assert torch.equal(read, torch.stack((cache[2, 127], cache[0, 0])))
    with pytest.raises(ValueError, match="page ID"):
        api.read_paged(cache, [-1], 0, 1)
    with pytest.raises(ValueError, match="range"):
        api.read_paged(cache, [0], 127, 2)
    assert not api.compare_tensors(torch.tensor(2**28), torch.tensor(2**28 + 1), 0, 0)["passed"]


def test_partition_merge_matches_whole_causal_attention(api):
    generator = torch.Generator().manual_seed(3)
    q, k, v = [torch.randn(*shape, generator=generator) for shape in ((2, 3, 4), (7, 1, 4), (7, 1, 5))]
    full = api.reference_attention(q, k, v, 0.5, causal_positions=[5, 6])
    partitions = [
        api.reference_attention(q, k[:5], v[:5], 0.5),
        api.reference_attention(q, k[5:], v[5:], 0.5, causal_positions=[0, 1]),
    ]
    merged = api.reference_merge(partitions)
    torch.testing.assert_close(merged[0], full[0])
    torch.testing.assert_close(merged[1], full[1])


def test_reference_budget_is_explicit_coverage_loss(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path, max_kv_tokens=1)
    exercise_prefill(api, layer)
    events = records(owner)
    assert any(event.get("reason") == "reference_input_budget" for event in events)
    assert any(event.get("reason") == "incomplete_reference" for event in events)
    assert events[-1]["skipped"] > 0


def test_save_budget_and_capture_guard(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path, max_saved_mib=1)
    assert owner.save({"large": torch.zeros(300000)}, 0) is None
    owner.sampling.capture_check = lambda: True
    with pytest.raises(RuntimeError, match="capture"):
        layer.batch.begin_layer(layer.impl, "another", layer.meta, layer.cache)
    assert records(owner)[-1]["event"] == "snapshot_skipped"


@pytest.mark.parametrize(
    "config",
    [
        '{"max_layers":0}',
        '{"query_rows":true}',
        '{"max_kv_tokens":20000}',
        '{"rtol":NaN}',
        '{"request_ids":[1]}',
        "[]",
        '{"unknown":1}',
    ],
)
def test_bad_config_fails_explicitly(api, config):
    with pytest.raises((TypeError, ValueError)):
        api.ChunkDiagnosticConfig.from_json(config)


def test_decode_reference_and_refreshed_metadata(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path)
    query = torch.ones(1, 2, 5)
    k, p = [api.read_paged(cache, layer.table[0], 0, 3) for cache in layer.cache]
    expected, _ = api.reference_attention(query, torch.cat((k, p), -1), k, 0.4)
    flash = SimpleNamespace(
        query=query,
        cache_lens=torch.tensor([3, 0]),
        cu=torch.tensor([0, 1, 1]),
        used_q=torch.tensor([1, 0]),
        block_table=layer.meta.block_tables[:1],
        slots=torch.tensor([386]),
        positions=torch.tensor([2]),
        schedule=torch.zeros(8),
    )
    layer.decode(flash, expected.transpose(0, 1))
    assert not layer.failed
    flash.slots[0] = 0
    layer.decode(flash, expected.transpose(0, 1) + 1)
    failures = [event["stage"] for event in records(owner) if event.get("passed") is False]
    assert "flash_metadata/slots" in failures and "flash/0/out" in failures


def test_lazy_prepare_attn_hook_preserves_batch_identity_and_budget(api, tmp_path):
    owner, batch, _ = make_case(api, tmp_path)
    owner.step = 0
    runner = SimpleNamespace(
        prepare_inputs=lambda: batch,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=512),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    owner.install(runner)  # ModelState does not exist until load_model.
    metadata = SimpleNamespace(chunk_diagnostics=None)
    runner.model_state = SimpleNamespace(prepare_attn=lambda input_batch: {"layer": metadata})
    for index in range(1, 4):
        assert runner.prepare_inputs() is batch
        # Real metadata builders create fresh metadata every forward.
        metadata = SimpleNamespace(chunk_diagnostics=None)
        result = runner.model_state.prepare_attn(input_batch=batch)
        assert result["layer"] is metadata
        assert batch.flashmla_diagnostic_id == index
        assert (metadata.chunk_diagnostics is not None) == (index <= 2)
    assert sum(event["event"] == "budget_exhausted" for event in records(owner)) == 1


@pytest.fixture(scope="module")
def checker():
    path = Path(__file__).resolve().parents[4] / "tools/flashmla_check_diagnostics.py"
    return SimpleNamespace(**runpy.run_path(str(path)))


def write_sample_evidence(owner, batch):
    owner.emit("runtime", chunked_prefill=True, prefix_caching=False, max_num_batched_tokens=512)
    rows = [
        dict(request_id=req, raw=dict(nan=0, positive_inf=0, finite=5), processed=dict(nan=0, positive_inf=0, finite=4))
        for req in batch.req_ids
    ]
    events = [dict(event="armed"), dict(event="end", forward_id=1, rows=rows, tensor_file="sample.pt")]
    (owner.directory.parent / "sample.pt").write_bytes(b"test")
    (owner.directory.parent / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))


def add_decode_evidence(api, layer):
    query = torch.ones(1, 2, 5)
    k, p = [api.read_paged(cache, layer.table[0], 0, 3) for cache in layer.cache]
    expected, _ = api.reference_attention(query, torch.cat((k, p), -1), k, 0.4)
    flash = SimpleNamespace(
        query=query,
        cache_lens=torch.tensor([3, 0]),
        cu=torch.tensor([0, 1, 1]),
        used_q=torch.tensor([1, 0]),
        block_table=layer.meta.block_tables[:1],
        slots=torch.tensor([386]),
        positions=torch.tensor([2]),
        schedule=torch.zeros(8),
    )
    layer.decode(flash, expected.transpose(0, 1))


def test_checker_complete_evidence_and_missing_rank(api, checker, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    add_decode_evidence(api, layer)
    exercise_prefill(api, layer)
    write_sample_evidence(owner, batch)
    report = checker.check_directory(tmp_path, [0], require_history=True)
    assert report["status"] == "SAMPLED_CHECKS_COMPLETE", report
    assert report["workers"][0]["zero_history_observed"]
    assert checker.check_directory(tmp_path, [0, 1], require_history=True)["exit_code"] == 2
    (owner.directory.parent / "sample.pt").unlink()
    assert checker.check_directory(tmp_path, [0])["exit_code"] == 2


@pytest.mark.parametrize(
    "problem", ["armed_only", "missing_end", "budget", "partial_json", "no_history", "missing_file"]
)
def test_checker_refuses_false_pass(api, checker, tmp_path, problem):
    owner, batch, layer = make_case(api, tmp_path)
    add_decode_evidence(api, layer)
    exercise_prefill(api, layer)
    write_sample_evidence(owner, batch)
    assert checker.check_directory(tmp_path, [0], require_history=True)["exit_code"] == 0
    events = records(owner)
    if problem == "armed_only":
        events = events[:1]
    if problem == "missing_end":
        events = [event for event in events if event["event"] != "layer_end"]
    if problem == "budget":
        events.append(dict(event="budget_exhausted"))
    if problem == "no_history":
        events = [event for event in events if event["event"] != "history_chunk"]
    if problem == "missing_file":
        next(owner.directory.glob("*.pt")).unlink()
    text = "".join(json.dumps(event) + "\n" for event in events)
    if problem == "partial_json":
        text += '{"event":'
    (owner.directory / "events.jsonl").write_text(text)
    assert checker.check_directory(tmp_path, [0], require_history=True)["exit_code"] == 2


def test_checker_reports_finite_wrong_value_and_request_order(api, checker, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    exercise_prefill(api, layer, "gather")
    batch.req_ids.reverse()
    write_sample_evidence(owner, batch)
    report = checker.check_directory(tmp_path, [0], require_history=True)
    assert report["exit_code"] == 1
    failures = report["workers"][0]["failures"]
    assert any("gather/" in value for value in failures)
    assert any("request order" in value for value in failures)


def test_bad_partition_and_positions_are_detected(api, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    owner.layers.clear()
    layer.meta.prefill.chunked_context.starts[1, 0] += 1
    batch.positions[1] -= 1
    layer.batch.visited.clear()
    observed = layer.batch.begin_layer(layer.impl, layer.name, layer.meta, layer.cache)
    assert observed.failed
    failures = [event["stage"] for event in records(owner) if event.get("passed") is False]
    assert "history_contiguous" in failures and "positions_vs_extents" in failures


def test_one_token_prefill_tail_keeps_full_history(api, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    batch.query_start_loc_np = np.array([0, 1, 2, 3])
    batch.seq_lens_np = np.array([3, 131, 1])
    batch.positions = torch.tensor([2, 130, 0])
    meta = layer.meta
    meta.num_actual_tokens = 3
    meta.query_start_loc = torch.tensor([0, 1, 2, 3])
    meta.seq_lens = torch.tensor([3, 131, 1])
    meta.slot_mapping = torch.tensor([386, 258, 0])
    meta.prefill.actual_seq_lengths_q = [1, 2]
    observed = api.DiagnosticBatch(owner, batch).begin_layer(layer.impl, layer.name, meta, layer.cache)
    assert not observed.failed
    assert observed.history_lengths == [[128, 0], [2, 0]]
    extents = [event for event in records(owner) if event["event"] == "prefill_extents"][-1]
    assert extents["requests"][0] == dict(request_id="continued", start=130, end=131)


def test_head_tiled_reference_and_partial_projection(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path)
    q = torch.ones(2, 7, 5)
    k, v = torch.ones(3, 7, 5), torch.ones(3, 7, 3)
    out, lse = api.reference_attention(q, k, v, 0.4)
    layer.attention_branch("tiled", q, k[..., :3], k[..., 3:], v, out, lse)
    assert not layer.failed
    out[:, -1] += 1
    layer.attention_branch("tiled", q, k[..., :3], k[..., 3:], v, out, lse)
    assert layer.failed
    assert int(layer.tensors["tiled/input_head"]) == 6
    layer.finish(torch.ones(4, 6), torch.ones(1, 10))
    assert any(event.get("reason") == "projection_rows_not_local" for event in records(owner))


def test_chunk_switch_is_strict_and_disabled_by_default(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/envs.py"
    env = runpy.run_path(str(path))
    name = "VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG"
    monkeypatch.delenv(name, raising=False)
    assert env["env_variables"][name]() is False
    monkeypatch.setenv(name, "1")
    assert env["env_variables"][name]() is True
    monkeypatch.setenv(name, "true")
    with pytest.raises(ValueError):
        env["env_variables"][name]()


def test_target_window_and_reused_metadata_are_not_consumed_by_smoke(api, tmp_path):
    owner, batch, _ = make_case(api, tmp_path, request_ids=("target",), start_after_forwards=2)
    owner.step = 0
    metadata = SimpleNamespace(chunk_diagnostics=None)
    runner = SimpleNamespace(
        prepare_inputs=lambda: batch,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=512),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_state=SimpleNamespace(prepare_attn=lambda input_batch: {"layer": metadata}),
    )
    owner.install(runner)
    batch.req_ids[0] = "target"
    for _ in range(2):
        runner.prepare_inputs()
        assert not batch.flashmla_diagnostic_selected
    batch.req_ids[0] = "decode"
    for _ in range(998):
        runner.prepare_inputs()
        runner.model_state.prepare_attn(batch)
        assert not batch.flashmla_diagnostic_selected and metadata.chunk_diagnostics is None
    assert owner.observed_steps == 0
    batch.req_ids[0] = "target"
    runner.prepare_inputs()
    runner.model_state.prepare_attn(batch)
    assert batch.flashmla_diagnostic_id == 1001 and owner.observed_steps == 1
    assert metadata.chunk_diagnostics is owner.active_batch
    batch.req_ids[0] = "unrelated"
    runner.prepare_inputs()
    runner.model_state.prepare_attn(batch)
    assert metadata.chunk_diagnostics is None and owner.observed_steps == 1
    batch.req_ids[0] = "target"
    runner.prepare_inputs()
    assert owner.observed_steps == 2 and batch.flashmla_diagnostic_selected
    runner.prepare_inputs()
    assert not batch.flashmla_diagnostic_selected
    assert sum(event["event"] == "budget_exhausted" for event in records(owner)) == 1


def test_all_layer_scan_covers_layer_outside_reference_budget(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path, max_layers=1, scan_all_mla_layers=True)
    observed = layer.batch
    assert observed.begin_layer(layer.impl, "later.layer", layer.meta, layer.cache) is None
    value = torch.ones(5, 7)
    value[4] = torch.nan  # Padding never belongs to a request.
    observed.boundary("later.layer", "mla_input", value)
    value[3, 0] = torch.nan
    observed.boundary("later.layer", "mla_output", value)
    observed.boundary("last.layer", "mla_input", value)
    events = [event for event in records(owner) if event["event"] == "boundary"]
    assert [event["passed"] for event in events] == [True, False, False]
    assert events[1]["rows"][-1]["request_id"] == "new"
    assert events[1]["tensor_file"] and not events[2]["tensor_file"]
    assert len(list(owner.directory.glob("*first-boundary-failure.pt"))) == 1
    observed.boundary("sharded.layer", "mla_input", value[:1])
    assert records(owner)[-1]["reason"] == "boundary_rows_not_global_or_width"


@pytest.mark.parametrize("config", [{"start_after_forwards": -1}, {"scan_all_mla_layers": 1}])
def test_invalid_capture_window_config(api, config):
    with pytest.raises(ValueError):
        api.ChunkDiagnosticConfig(**config)


def test_checker_requires_new_boundaries_and_identifies_first_observed_fault(api, checker, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    add_decode_evidence(api, layer)
    exercise_prefill(api, layer)
    write_sample_evidence(owner, batch)
    sample_path = owner.directory.parent / "events.jsonl"
    sampling = [json.loads(line) for line in sample_path.read_text().splitlines()]
    sampling[0]["hidden_boundaries"] = True
    sample_path.write_text("".join(json.dumps(event) + "\n" for event in sampling))
    report = checker.check_directory(tmp_path, [0])
    assert report["exit_code"] == 2 and any("model_output" in gap for gap in report["workers"][0]["gaps"])
    sampling.extend(
        [
            dict(
                event="hidden_boundary",
                forward_id=1,
                stage=stage,
                utc="2026-09-28T00:00:00Z",
                rows=[dict(request_id=req, nan=0, positive_inf=0, negative_inf=0) for req in batch.req_ids],
            )
            for stage in ("model_output", "lm_head_input")
        ]
    )
    sampling.append(dict(event="hidden_mapping", forward_id=1, passed=True))
    sample_path.write_text("".join(json.dumps(event) + "\n" for event in sampling))
    assert checker.check_directory(tmp_path, [0])["exit_code"] == 0
    sampling[-3]["rows"][1]["nan"] = 1
    sample_path.write_text("".join(json.dumps(event) + "\n" for event in sampling))
    report = checker.check_directory(tmp_path, [0])
    assert report["exit_code"] == 1
    assert report["workers"][0]["first_observed_anomaly"]["stage"] == "model_output"


def test_checker_requires_all_mla_boundaries_and_every_target(api, checker, tmp_path):
    owner, batch, layer = make_case(api, tmp_path, scan_all_mla_layers=True, request_ids=("continued", "new"))
    add_decode_evidence(api, layer)
    exercise_prefill(api, layer)
    write_sample_evidence(owner, batch)
    owner.emit("boundary_layers", layers=[layer.name, "later.layer"])
    for name in (layer.name, "later.layer"):
        for stage in ("mla_input", "mla_output"):
            layer.batch.boundary(name, stage, torch.ones(4, 8))
    # No decode request was selected, so the full run still lacks Flash evidence.
    report = checker.check_directory(tmp_path, [0])
    assert not any("mla_input" in gap or "mla_output" in gap for gap in report["workers"][0]["gaps"])
    events = records(owner)
    events = [event for event in events if not (event["event"] == "boundary" and event["layer"] == "later.layer")]
    events[0]["configuration"]["request_ids"].append("absent")
    (owner.directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    report = checker.check_directory(tmp_path, [0])
    assert report["exit_code"] == 2
    assert any("later.layer" in gap for gap in report["workers"][0]["gaps"])
    assert any("absent" in gap for gap in report["workers"][0]["gaps"])


def test_cpu_norm_interleaved_rope_has_independent_expected_values(api):
    raw = torch.tensor([[3.0, 4.0, 1.0, 2.0, 3.0, 4.0]])
    weight = torch.tensor([2.0, 3.0])
    latent, positional = api.reference_kv(raw, weight, 0.0, True, torch.zeros(1, 4), torch.ones(1, 4))
    assert torch.allclose(latent, torch.tensor([[6.0, 12.0]]) / (12.5**0.5))
    assert torch.equal(positional, torch.tensor([[-2.0, -4.0, 1.0, 3.0]]))
    _, no_rope = api.reference_kv(raw, weight, 0.0, False)
    assert torch.equal(no_rope, raw[:, 2:])


@pytest.mark.parametrize(
    "phase,use_rope,fault",
    [
        ("decode", False, None),
        ("decode", True, None),
        ("prefill", True, None),
        ("decode", False, "target"),
        ("prefill", False, "neighbor"),
    ],
)
def test_writer_reference_and_untouched_sentinels(api, tmp_path, phase, use_rope, fault):
    owner, _, layer = make_case(api, tmp_path)
    layer.impl.kv_a_layernorm = SimpleNamespace(weight=torch.tensor([2.0, 3.0, 4.0]), variance_epsilon=0.5)
    layer.impl.use_mla_rope = use_rope
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]) if phase == "decode" else torch.ones(3, 5)
    slots = layer.meta.slot_mapping[:1] if phase == "decode" else layer.meta.slot_mapping[1:]
    cos, sin = torch.zeros(len(raw), 1, 1, 2), torch.ones(len(raw), 1, 1, 2)
    # Unwritten cache can contain NaN; unchanged bits must pass guard checking.
    layer.cache[0][3, 1, 0, 0] = torch.nan
    with layer.watch_writer(phase, raw, cos, sin, slots):
        latent = raw[:, :3] / (raw[:, :3].square().mean(-1, keepdim=True) + 0.5).sqrt() * torch.tensor([2.0, 3.0, 4.0])
        positional = torch.stack((-raw[:, 4], raw[:, 3]), -1) if use_rope else raw[:, 3:]
        layer.cache[0][slots // 128, slots % 128] = latent[:, None]
        layer.cache[1][slots // 128, slots % 128] = positional[:, None]
        if fault == "target":
            layer.cache[0][slots[0] // 128, slots[0] % 128, 0, 0] += 3
        if fault == "neighbor":
            guard = next(event["guard_slots"][0] for event in records(owner) if event["event"] == "writer_inputs")
            layer.cache[0][guard // 128, guard % 128, 0, 0] += 3
    failures = [event for event in records(owner) if event.get("passed") is False]
    assert bool(failures) == (fault is not None)
    if fault == "neighbor":
        assert any(event["stage"].startswith("writer_guard/") for event in failures)
    if fault == "target":
        assert any(event["stage"].startswith("decode_writer/") for event in failures)


def test_writer_exception_preserves_original_failure(api, tmp_path):
    owner, _, layer = make_case(api, tmp_path)
    with (
        pytest.raises(ValueError, match="device failure"),
        layer.watch_writer("decode", torch.ones(1, 5), None, None, torch.tensor([386])),
    ):
        raise ValueError("device failure")
    assert records(owner)[-1]["event"] == "writer_error"


def test_registry_detects_shared_cpu_device_mapping_error(api, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    states = SimpleNamespace(
        req_id_to_index=dict(zip(batch.req_ids, [4, 2, 7])),
        index_to_req_id={4: "decode", 2: "continued", 7: "new"},
        num_computed_prefill_tokens=np.array([0, 0, 130, 0, 2, 0, 0, 0]),
        prefill_len=SimpleNamespace(np=np.array([0, 0, 132, 0, 2, 0, 0, 1])),
    )
    scheduler = SimpleNamespace(num_scheduled_tokens={"continued": 2, "new": 1, "decode": 1})
    sources = api.snapshot_request_sources(SimpleNamespace(req_states=states), scheduler)
    layer.batch.check_request_sources(sources)
    assert records(owner)[-1]["passed"]
    batch.idx_mapping_np[0] = 7
    batch.idx_mapping[0] = 7
    observed = api.DiagnosticBatch(owner, batch)
    observed.check_request_sources(sources)
    assert records(owner)[-1]["passed"] is False
    assert not records(owner)[-1]["checks"][0]["passed"]
    states.index_to_req_id[4] = "wrong_owner"
    layer.batch.check_request_sources(api.snapshot_request_sources(SimpleNamespace(req_states=states), scheduler))
    assert records(owner)[-1]["passed"] is False


def test_forward_context_serializes_padding_and_dispatch_once(api, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    batch.is_padding = torch.tensor([False, False, False, False, True])
    batch.num_tokens_after_padding = 5
    extra = SimpleNamespace(moe_comm_type=SimpleNamespace(name="ALLTOALL"))
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([4, 0, 3, 0])))
    layer.batch.record_context(extra, context)
    layer.batch.record_context(extra, context)
    events = [event for event in records(owner) if event["event"] == "forward_context"]
    assert len(events) == 1
    assert events[0]["dp_tokens"] == [4, 0, 3, 0]
    assert events[0]["padding"] == {"tokens": 1, "first_indices": [4]}
    assert events[0]["moe_comm_type"] == "ALLTOALL"


def test_scratch_descriptor_variants_and_full_backing_mask():
    path = Path(__file__).resolve().parents[4] / "tools/flashmla_writer_probe.py"
    api = SimpleNamespace(**runpy.run_path(str(path)))
    backing, cache, direct = api.make_views(81408, "direct")
    _, _, rebuilt = api.make_views(81408, "singleton_rebuilt")
    assert direct[0].stride()[2] == 576 and rebuilt[0].stride()[2] == 512
    assert direct[1].stride()[2] == 576 and rebuilt[1].stride()[2] == 64
    mask = api.untouched_mask(backing.numel(), [255, 0, 1], 81408)
    before = backing.clone()
    cache[1, 127] = 3
    cache[0, :2] = 4
    assert torch.equal(before[mask], backing[mask])
    assert int((~mask).sum()) == 3 * 576
    backing[-1] = 9
    assert not torch.equal(before[mask], backing[mask])


def test_installed_registry_hook_freezes_before_sort_and_dummy_does_not_consume(api, tmp_path):
    owner, batch, _ = make_case(api, tmp_path)
    owner.step = 0
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"continued": 2, "new": 1, "decode": 1}, total_num_scheduled_tokens=4
    )
    runner = SimpleNamespace(
        prepare_inputs=lambda: batch,
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=512),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        req_states=SimpleNamespace(
            req_id_to_index={"decode": 4, "continued": 2, "new": 7},
            index_to_req_id={4: "decode", 2: "continued", 7: "new"},
            num_computed_prefill_tokens=np.array([0, 0, 130, 0, 2, 0, 0, 0]),
            prefill_len=SimpleNamespace(np=np.array([0, 0, 132, 0, 2, 0, 0, 1])),
        ),
        model_state=SimpleNamespace(prepare_attn=lambda input_batch: {}),
    )

    def gather(scheduler_output, dummy_run=False):
        # Simulate the same incorrect index getting copied to CPU and device.
        if not dummy_run:
            batch.idx_mapping_np[0] = 7
            batch.idx_mapping[0] = 7
        return batch

    sentinel = object()

    def execute(scheduler_output, dummy_run=False, is_profile=False):
        runner.gather_batch_req_state(scheduler_output, dummy_run)
        if not dummy_run:
            runner.prepare_inputs()
        return sentinel

    runner.gather_batch_req_state, runner.execute_model = gather, execute
    owner.install(runner)
    assert runner.execute_model(scheduler, dummy_run=True) is sentinel
    assert owner.observed_steps == 0
    assert runner.execute_model(scheduler) is sentinel
    assert owner.observed_steps == 1
    events = records(owner)
    assert [event["dummy"] for event in events if event["event"] == "execute_call"] == [True, False]
    registry = [event for event in events if event["event"] == "request_registry"]
    assert len(registry) == 1 and not registry[0]["passed"]
    assert registry[0]["scheduler_order"][0]["request_id"] == "continued"


def test_checker_requires_new_writer_registry_and_context_evidence(api, checker, tmp_path):
    owner, batch, layer = make_case(api, tmp_path)
    add_decode_evidence(api, layer)
    exercise_prefill(api, layer)
    write_sample_evidence(owner, batch)
    events = records(owner)
    runtime = next(event for event in events if event["event"] == "runtime")
    runtime.update(writer_references=True, independent_mapping=True, dispatch_evidence=True)
    (owner.directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    report = checker.check_directory(tmp_path, [0])
    assert report["exit_code"] == 2
    gaps = report["workers"][0]["gaps"]
    assert any("registry" in gap for gap in gaps) and any("dispatch" in gap for gap in gaps)
    assert any("decode_writer" in gap for gap in gaps)
    owner.emit("request_registry", passed=False)
    assert checker.check_directory(tmp_path, [0])["exit_code"] == 1
