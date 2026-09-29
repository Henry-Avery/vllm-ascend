#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced-layer plan, runtime manifest and conservative BLINE log summary."""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

REQUIRED_CHECKS = (
    "kda_conv",
    "kda_prefill",
    "kda_recurrent",
    "writer",
    "fia",
    "external",
    "raw_logits",
    "final_hidden",
)


def prefix_override(config, layers):
    """Keep actual nesting/dimensions; kda_layers are one-based in KimiLinear."""
    text = config.get("text_config", config)
    total = text["num_hidden_layers"]
    linear = text.get("linear_attn_config")
    if not linear or "kda_layers" not in linear:
        raise ValueError("No explicit KDA layer mapping; inspect the actual model configuration")
    types = ["KDA" if i + 1 in linear["kda_layers"] else "MLA" for i in range(total)]
    if not 4 <= layers <= total or types[:4] != ["KDA", "KDA", "KDA", "MLA"]:
        raise ValueError("Expected original prefix KDA0/1/2 + MLA3; do not blindly truncate this model")
    override = {"num_hidden_layers": layers}
    if "text_config" in config:
        override = {"text_config": override}
    expected = copy.deepcopy(config)
    expected.get("text_config", expected)["num_hidden_layers"] = layers
    return override, expected, types


def plan(args):
    # Runtime-only import: use the publisher's actual paired vLLM config machinery.
    from vllm.config import ModelConfig

    original = ModelConfig(model=args.model, trust_remote_code=args.trust_remote_code)
    override, _, types = prefix_override(original.hf_config.to_dict(), args.layers)
    reduced = ModelConfig(model=args.model, trust_remote_code=args.trust_remote_code, hf_overrides=override)
    before, after = original.hf_text_config.to_dict(), reduced.hf_text_config.to_dict()
    expected = copy.deepcopy(before)
    expected["num_hidden_layers"] = args.layers
    if after != expected:
        changed = sorted(key for key in set(after) | set(expected) if after.get(key) != expected.get(key))
        raise ValueError(f"Unexpected effective text config changes: {changed}")
    actual_types = ["KDA" if reduced.hf_text_config.is_kda_layer(i) else "MLA" for i in range(args.layers)]
    if actual_types != types[: args.layers]:
        raise ValueError("Actual is_kda_layer disagrees with original layer order")
    additional = json.loads(Path(args.additional_config).read_text()) if args.additional_config else {}
    additional["bline_diagnostics"] = dict(
        enabled=True, max_steps=args.steps, max_bytes=args.max_bytes, max_events=2048, max_pages=256, max_requests=64
    )
    if args.arm_file:
        if not Path(args.arm_file).is_absolute():
            raise ValueError("--arm-file must be an absolute path")
        additional["bline_diagnostics"]["arm_file"] = args.arm_file
    extra = args.serve_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    forbidden = {
        "--hf-overrides",
        "--additional-config",
        "--tensor-parallel-size",
        "-tp",
        "--data-parallel-size",
        "-dp",
        "--pipeline-parallel-size",
        "-pp",
        "--speculative-config",
    }
    if any(arg.split("=")[0] in forbidden for arg in extra):
        raise ValueError(
            "Supply existing additional config via --additional-config; do not override diagnostic topology"
        )
    command = [
        "vllm",
        "serve",
        args.model,
        "--tensor-parallel-size",
        "8",
        "--data-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--enforce-eager",
        "--hf-overrides",
        json.dumps(override),
        "--additional-config",
        json.dumps(additional),
        *extra,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    result = dict(
        model=args.model,
        original_layers=types,
        effective_layers=actual_types,
        hf_overrides=override,
        text_config_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        effective_text_config=after,
        command=shlex.join(command),
        status="PLAN_ONLY",
        note="No service started. All non-layer text fields verified unchanged.",
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(result["command"])


def summarize(events, expected_ranks):
    by_rank = {rank: [] for rank in expected_ranks}
    for event in events:
        if event.get("rank") in by_rank:
            by_rank[event["rank"]].append(event)
    reports = {}
    for rank, records in by_rank.items():
        observed = {e["check"] for e in records if e["status"] == "CHECK"}
        missing = sorted(set(REQUIRED_CHECKS) - observed)
        bindings = [e for e in records if e["check"] == "binding"]
        if not bindings or any(not e.get("kda_layers") or not e.get("mla_layers") for e in bindings):
            missing.append("complete binding")
        if not any(e["check"] == "protected_mapping" and e.get("protected") for e in records):
            missing.append("actual protected history mapping")

        for binding in (e for e in records if e["check"] == "binding"):
            if binding.get("expected_ranks") != len(expected_ranks):
                missing.append("expected rank count disagrees with runtime TP topology")
            for layers, routes in (
                (binding.get("kda_layers", []), ("kda_conv", "kda_prefill", "kda_recurrent")),
                (binding.get("mla_layers", []), ("writer", "fia", "external")),
            ):
                for layer in layers:
                    for route in routes:
                        if not any(
                            e["status"] == "CHECK" and e["check"] == route and e.get("layer") == layer for e in records
                        ):
                            missing.append(f"{layer}:{route}")
        failures = [e for e in records if e["status"] == "FAIL"]
        uncovered = [e for e in records if e["status"] == "UNCOVERED"]
        run_ids = sorted({e["run_id"] for e in records})
        starts = {e["step"] for e in records if e["check"] == "begin"}
        ends = {e["step"] for e in records if e["check"] == "step_summary"}
        incomplete = sorted(starts - ends)
        if not starts or not ends or ends - starts:
            missing.append("complete begin/summary sequence")
        reports[rank] = dict(
            status="FAIL"
            if failures
            else "UNCOVERED"
            if (missing or uncovered or incomplete or len(run_ids) != 1)
            else "CHECK",
            run_ids=run_ids,
            observed=sorted(observed),
            missing=missing,
            checked_steps=sorted(ends),
            incomplete_steps=incomplete,
            first_failure=failures[0] if failures else None,
            uncovered=uncovered,
        )
    states = {report["status"] for report in reports.values()}
    unexpected_ranks = sorted({e.get("rank") for e in events} - set(expected_ranks))
    if unexpected_ranks or not expected_ranks:
        states.add("UNCOVERED")
    return dict(
        status="FAIL" if "FAIL" in states else "UNCOVERED" if "UNCOVERED" in states else "CHECK",
        ranks=reports,
        unexpected_ranks=unexpected_ranks,
        scope="Bounded local diagnostic evidence only; no numerical/full-model PASS",
    )


def summary(args):
    events = []
    for filename in args.logs:
        for line in Path(filename).read_text(errors="replace").splitlines():
            if "BLINE " in line:
                try:
                    event, _ = json.JSONDecoder().raw_decode(line.split("BLINE ", 1)[1])
                    events.append(event)
                except (ValueError, KeyError):
                    raise ValueError(f"Malformed diagnostic record in {filename}") from None
    result = summarize(events, [int(rank) for rank in args.expected_ranks.split(",")])
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    return 1 if result["status"] == "FAIL" else 2 if result["status"] == "UNCOVERED" else 0


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(args):
    packages = {}
    for module, distribution in (
        ("vllm_ascend", "vllm-ascend"),
        ("vllm", "vllm"),
        ("torch", "torch"),
        ("torch_npu", "torch-npu"),
        ("fla_npu", "fla-npu"),
        ("cann_ops_transformer", "cann-ops-transformer"),
    ):
        try:
            found = importlib.util.find_spec(module)
            path = found.origin if found else None
            packages[module] = dict(path=path, version=importlib.metadata.version(distribution))
        except (ImportError, importlib.metadata.PackageNotFoundError):
            packages[module] = dict(path=None, version=None)
    repos = {}
    for label, directory in (("va", args.va_repo), ("vllm", args.vllm_repo)):
        sha = subprocess.check_output(["git", "-C", directory, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", directory, "status", "--porcelain"], text=True)
        repos[label] = dict(path=directory, sha=sha, dirty=dirty)
    binaries = {str(Path(name).resolve()): file_hash(Path(name)) for name in args.binary}
    result = dict(
        repos=repos,
        packages=packages,
        binaries=binaries,
        cann=args.cann,
        status="RECORDED" if binaries and args.cann else "UNCOVERED",
        note="Supply all loaded FLA/native/FlashMLA OPP binaries and CANN identity; verify loaded import paths.",
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    item = commands.add_parser("plan")
    item.add_argument("--model", required=True)
    item.add_argument("--layers", type=int, default=4)
    item.add_argument("--trust-remote-code", action="store_true")
    item.add_argument("--arm-file", help="Begin capture after this shared marker file appears")
    item.add_argument("--steps", type=int, default=8)
    item.add_argument("--max-bytes", type=int, default=268435456)
    item.add_argument("--additional-config", help="Existing JSON file, merged rather than discarded")
    item.add_argument("--output", required=True)
    item.add_argument("serve_args", nargs=argparse.REMAINDER)
    item.set_defaults(func=plan)
    item = commands.add_parser("summary")
    item.add_argument("--expected-ranks", required=True, help="e.g. 0,1,2,3,4,5,6,7; no implicit single-rank PASS")
    item.add_argument("--output", required=True)
    item.add_argument("logs", nargs="+")
    item.set_defaults(func=summary)
    item = commands.add_parser("manifest")
    item.add_argument("--va-repo", required=True)
    item.add_argument("--vllm-repo", required=True)
    item.add_argument("--binary", action="append", default=[])
    item.add_argument("--cann")
    item.add_argument("--output", required=True)
    item.set_defaults(func=manifest)
    args = parser.parse_args()
    raise SystemExit(args.func(args) or 0)


if __name__ == "__main__":
    main()
