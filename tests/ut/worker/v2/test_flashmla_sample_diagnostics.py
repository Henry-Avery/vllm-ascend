# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolated CPU checks: pytest --confcutdir=tests/ut/worker/v2 <this file>."""

import ast
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def api():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/flashmla_sample_diagnostics.py"
    return SimpleNamespace(**runpy.run_path(str(source)))


class Model:
    def compute_logits(self, hidden):
        # Third row simulates LM-head collective padding, not a real request.
        return torch.tensor([[1.0, 5.0, 3.0], [4.0, 2.0, 1.0], [float("nan")] * 3])


class Sampler:
    use_flashinfer = False

    def __init__(self):
        self.sampling_states = SimpleNamespace(
            temperature=SimpleNamespace(gpu=torch.tensor([0.8, 0.0])),
            seeds=SimpleNamespace(gpu=torch.tensor([123, 456])),
        )

    def sample(
        self,
        logits,
        expanded_idx_mapping,
        idx_mapping,
        idx_mapping_np,
        pos,
        input_ids,
        expanded_local_pos,
        return_logprobs=False,
    ):
        # Mutates the same tensor: raw evidence must have been copied already.
        logits[:, 0] = -torch.inf
        return logits.argmax(1), logits


@pytest.fixture
def case():
    batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=3,
        req_ids=["request-b", "request-a"],
        num_draft_tokens=0,
        logits_indices=torch.tensor([2, 0]),
        cu_num_logits_np=np.array([0, 1, 2]),
        positions=torch.tensor([15, 16, 25]),
        idx_mapping_np=np.array([1, 0]),
        expanded_idx_mapping=torch.tensor([1, 0]),
        is_prefilling_np=np.array([False, True]),
    )
    runner = SimpleNamespace(model=Model(), sampler=Sampler(), speculative_config=None, batch_sharder=None)
    seen = {}

    def call():
        logits = runner.model.compute_logits(None)[:2]
        # Grammar-style mutation occurs after compute_logits, before sampler.
        logits[0, 1] = -torch.inf
        sampled, _ = runner.sampler.sample(
            logits,
            batch.expanded_idx_mapping,
            batch.expanded_idx_mapping,
            batch.idx_mapping_np,
            batch.positions[batch.logits_indices],
            torch.tensor([7, 8]),
            torch.zeros(2),
        )
        sampled[0] = 0  # Trace-style final override must differ from internal sample.
        output = SimpleNamespace(sampled_token_ids=sampled[:, None])
        seen["result"] = (output, torch.tensor([1, 0]), torch.tensor([0, 0]))
        return seen["result"]

    return runner, batch, call, seen


def records(diagnostic):
    return [json.loads(line) for line in (diagnostic.directory / "events.jsonl").read_text().splitlines()]


def test_raw_inplace_processed_final_mapping_and_identity(api, tmp_path, case):
    runner, batch, call, seen = case
    diagnostic = api.SampleDiagnostics(api.DiagnosticConfig(str(tmp_path)), dp_rank=2, tp_rank=0, global_rank=16)
    result = diagnostic.run(runner, batch, call)
    assert result is seen["result"]
    assert "sample" not in vars(runner.sampler) and "compute_logits" not in vars(runner.model)
    report = records(diagnostic)[-1]
    assert report["event"] == "end" and report["utc"] and report["dp_rank"] == 2
    row = report["rows"][0]
    assert row["request_id"] == "request-b" and row["hidden_state_row"] == 2 and row["position"] == 25
    assert row["request_state_index"] == 1 and row["temperature"] == 0 and row["seed"] == 456
    assert row["internal_sampled_token_id"] == 2 and row["sampled_token_id"] == 0
    assert row["raw"]["finite"] == 3 and row["processed"]["negative_inf"] == 2
    assert not report["rows"][1]["emitted"]  # Chunked prefill is not an SSE token.
    saved = torch.load(diagnostic.directory / report["tensor_file"], weights_only=True)
    assert saved["raw_logits"].shape == (2, 3)  # No padded NaN row.
    assert torch.equal(saved["raw_logits"][0], torch.tensor([1.0, 5.0, 3.0]))
    assert torch.isneginf(saved["processed_logits"][0, :2]).all()


def test_failure_restores_existing_instance_overrides(api, tmp_path, case):
    runner, batch, _, _ = case
    override = lambda _: torch.ones(2, 3)
    runner.model.compute_logits = override
    sample = runner.sampler.sample
    runner.sampler.sample = sample
    diagnostic = api.SampleDiagnostics(api.DiagnosticConfig(str(tmp_path)), dp_rank=0, tp_rank=0, global_rank=0)

    def fail():
        runner.model.compute_logits(None)
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        diagnostic.run(runner, batch, fail)
    assert runner.model.compute_logits is override and runner.sampler.sample is sample
    assert not diagnostic.active and records(diagnostic)[-1]["event"] == "error"


def test_budget_stops_observation_and_marks_exhaustion(api, tmp_path, case):
    runner, batch, call, _ = case
    diagnostic = api.SampleDiagnostics(
        api.DiagnosticConfig(str(tmp_path), steps=1), dp_rank=0, tp_rank=0, global_rank=0
    )
    diagnostic.run(runner, batch, call)
    sentinel = object()
    assert diagnostic.run(runner, batch, lambda: sentinel) is sentinel
    assert diagnostic.run(runner, batch, lambda: sentinel) is sentinel
    assert [item["event"] for item in records(diagnostic)].count("budget_exhausted") == 1
    assert len(list(diagnostic.directory.glob("*.pt"))) == 1


def test_nonfinite_json_and_invalid_priority(api, tmp_path, case):
    runner, batch, call, _ = case
    runner.model.compute_logits = lambda _: torch.tensor([[1.0, 2.0, 3.0], [float("nan"), float("inf"), -float("inf")]])
    diagnostic = api.SampleDiagnostics(api.DiagnosticConfig(str(tmp_path), rows=1), dp_rank=0, tp_rank=0, global_rank=0)
    diagnostic.run(runner, batch, call)
    report = records(diagnostic)[-1]
    assert report["tensor_rows"] == [1]  # Bad second row beats watched token in row 0.
    assert report["rows"][1]["raw"]["nan"] == 1
    assert report["rows"][1]["raw"]["positive_inf"] == 1
    assert report["rows"][1]["raw"]["negative_inf"] == 1
    assert report["rows"][1]["raw"]["finite_min"] == "inf"


@pytest.mark.parametrize("failure", ["capture", "spec", "shard", "interface", "mapping"])
def test_unsupported_paths_fail_visibly(api, tmp_path, case, failure):
    runner, batch, call, _ = case
    if failure == "spec":
        batch.num_draft_tokens = 1
    elif failure == "shard":
        runner.batch_sharder = object()
    elif failure == "interface":
        runner.sampler.sample = lambda logits: logits
    elif failure == "mapping":
        batch.expanded_idx_mapping = torch.tensor([0, 1])
    diagnostic = api.SampleDiagnostics(
        api.DiagnosticConfig(str(tmp_path)),
        dp_rank=0,
        tp_rank=0,
        global_rank=0,
        capture_check=lambda: failure == "capture",
    )
    with pytest.raises(RuntimeError):
        diagnostic.run(runner, batch, call)
    assert records(diagnostic)[-1]["event"] == "error"


def test_default_disabled_runner_init_never_installs_hook(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    monkeypatch.delenv("VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR", raising=False)
    env = runpy.run_path(str(root / "vllm_ascend/envs.py"))
    tree = ast.parse((root / "vllm_ascend/worker/v2/model_runner.py").read_text())
    gate = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR"
    )
    called = []
    exec(
        compile(ast.Module(body=[gate], type_ignores=[]), "<runner diagnostic gate>", "exec"),
        {
            "ascend_envs": SimpleNamespace(
                VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR=env["env_variables"]["VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR"]()
            ),
            "self": object(),
            "install_sample_diagnostics": lambda *args: called.append(args),
        },
    )
    assert not called


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_installer_rank_and_eager_gates(api, tmp_path, monkeypatch, enforce_eager):
    distributed = ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: SimpleNamespace(world_size=8, rank_in_group=1, rank=1)
    sampler_module = ModuleType("vllm.v1.worker.gpu.sample.sampler")
    sampler_module.Sampler = Sampler
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.sampler", sampler_module)
    # Selecting TP0 leaves this TP1 worker entirely unmodified.
    envs = SimpleNamespace(
        VLLM_ASCEND_ENABLE_FLASH_MLA=True,
        VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG=False,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR=str(tmp_path),
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS=64,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS=2,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK=-1,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK=0,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS="0",
    )
    original = lambda *args: None
    runner = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        use_aclgraph=False,
        speculative_config=None,
        dp_rank=0,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=4),
        sample=original,
    )
    if enforce_eager:
        api.install_sample_diagnostics(runner, envs)
    else:
        with pytest.raises(RuntimeError, match="enforce_eager"):
            api.install_sample_diagnostics(runner, envs)
    assert runner.sample is original and not list(tmp_path.iterdir())


def test_selected_rank_install_before_model_load_and_observe_first_sample(api, tmp_path, monkeypatch, case):
    runner, batch, call, seen = case
    distributed = ModuleType("vllm.distributed")
    distributed.get_tp_group = lambda: SimpleNamespace(world_size=8, rank_in_group=0, rank=8)
    sampler_module = ModuleType("vllm.v1.worker.gpu.sample.sampler")
    sampler_module.Sampler = Sampler
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.sampler", sampler_module)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    envs = SimpleNamespace(
        VLLM_ASCEND_ENABLE_FLASH_MLA=True,
        VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG=False,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR=str(tmp_path),
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS=64,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS=2,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK=-1,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK=0,
        VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS="0",
    )
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.use_aclgraph = False
    runner.dp_rank = 1
    runner.parallel_config = SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=4)
    original = lambda hidden_states, input_batch, grammar_output: call()
    runner.sample = original
    model, sampler = runner.model, runner.sampler
    del runner.model
    del runner.sampler
    api.install_sample_diagnostics(runner, envs)
    assert runner.sample is not original
    runner.model, runner.sampler = model, sampler
    batch.flashmla_diagnostic_id = 7
    result = runner.sample(None, batch, None)
    assert result is seen["result"]
    report = next(tmp_path.glob("*/events.jsonl"))
    events = [json.loads(line) for line in report.read_text().splitlines()]
    assert [event["event"] for event in events] == ["armed", "batch", "end"]
    assert all(event["dp_rank"] == 1 and event["tp_rank"] == 0 for event in events)
    assert "compute_logits" not in vars(model) and "sample" not in vars(sampler)
    assert events[-1]["forward_id"] == 7
