# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline report accounting; no endpoint or device is contacted."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("failure", [False, True])
def test_comparison_collects_every_case_even_after_a_failed_request(tmp_path, monkeypatch, failure):
    module = runpy.run_path(str(ROOT / "tests/e2e/manual/flashmla_pd_correctness.py"))
    output = tmp_path / "report.json"
    args = SimpleNamespace(repeat=2, concurrency=1, output=output)
    cases = [{"name": "first", "prompt": [1]}, {"name": "second", "prompt": [2]}]

    def compare(case, _args):
        # The ledger exists before the first HTTP operation could begin.
        initial = json.loads(output.read_text())
        assert initial["expected_cases"] == 4
        if failure and case["name"] == "first":
            raise TimeoutError("response not received")
        return {"name": case["name"], "max_logprob_error": 0}

    monkeypatch.setitem(module["run_cases"].__globals__, "compare_case", compare)
    report = module["run_cases"](cases, args)
    assert report["completed_cases"] == report["expected_cases"] == 4
    assert report["failed_cases"] == (2 if failure else 0)
    assert report["passed_cases"] == (2 if failure else 4)
    assert len({case["case_id"] for case in report["cases"]}) == 4
    saved = json.loads(output.read_text())
    assert saved["cases"] == report["cases"]
    if failure:
        assert all("TimeoutError" in case["error"] for case in report["cases"] if case["status"] == "failed")


def test_comparison_interruption_keeps_pending_case_ledger(tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "tests/e2e/manual/flashmla_pd_correctness.py"))
    output = tmp_path / "report.json"
    args = SimpleNamespace(repeat=1, concurrency=1, output=output)

    def compare(*_):
        raise KeyboardInterrupt()

    monkeypatch.setitem(module["run_cases"].__globals__, "compare_case", compare)
    with pytest.raises(KeyboardInterrupt):
        module["run_cases"]([{"name": "interrupt", "prompt": [1]}], args)
    report = json.loads(output.read_text())
    assert report["expected_cases"] == 1
    assert report["completed_cases"] == 0
    assert report["cases"][0]["status"] == "pending"
