# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate collected FlashMLA JSONL evidence without torch or NPU dependencies.

Exit 0: sampled checks complete; 1: a discrepancy; 2: coverage/evidence missing.
This is an evidence check, never a model accuracy or race-freedom certification.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def read_events(path, gaps):
    if not path.is_file():
        gaps.append(f"missing {path}")
        return []
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or "event" not in event:
                raise ValueError("event object required")
            result.append(event)
        except (ValueError, TypeError):
            gaps.append(f"invalid JSONL {path}:{number}")
    return result


def inspect_worker(path, require_history):
    gaps, failures = [], []
    events = read_events(path, gaps)
    sample_path = path.parent.parent / "events.jsonl"
    sampling = read_events(sample_path, gaps)
    armed = next((event for event in events if event["event"] == "armed"), {})
    runtime = next((event for event in events if event["event"] == "runtime"), None)
    if not armed or not runtime or not any(event["event"] == "armed" for event in sampling):
        gaps.append("missing armed/runtime evidence")
    batches = {event["forward_id"]: event for event in events if event["event"] == "batch"}
    selected = {event["layer"] for event in events if event["event"] == "layer_selected"}
    requested = set(armed.get("configuration", {}).get("layer_names", []))
    if requested - selected:
        gaps.append(f"requested layers absent: {sorted(requested - selected)}")
    if not batches or not selected:
        gaps.append("no real attention batches/layers; armed alone is insufficient")
    begins = Counter(
        (event.get("forward_id"), event.get("layer")) for event in events if event["event"] == "layer_begin"
    )
    ends = Counter((event.get("forward_id"), event.get("layer")) for event in events if event["event"] == "layer_end")
    samples = {event.get("forward_id"): event for event in sampling if event["event"] == "end"}
    hidden_required = any(event.get("hidden_boundaries") for event in sampling if event["event"] == "armed")
    boundaries = Counter(
        (event.get("forward_id"), event.get("layer"), event.get("stage"))
        for event in events
        if event["event"] == "boundary"
    )
    hidden = Counter(
        (event.get("forward_id"), event.get("stage")) for event in sampling if event["event"] == "hidden_boundary"
    )
    boundary_layers = {event["forward_id"]: event["layers"] for event in events if event["event"] == "boundary_layers"}
    for forward, batch in batches.items():
        if runtime and runtime.get("independent_mapping"):
            registry = [
                event for event in events if event["event"] == "request_registry" and event.get("forward_id") == forward
            ]
            if len(registry) != 1:
                gaps.append(f"forward {forward}: missing/duplicate request registry check")
        if runtime and runtime.get("dispatch_evidence"):
            context = [
                event for event in events if event["event"] == "forward_context" and event.get("forward_id") == forward
            ]
            if len(context) != 1:
                gaps.append(f"forward {forward}: missing/duplicate dispatch context")
            elif context[0].get("moe_comm_type") is None:
                gaps.append(f"forward {forward}: unknown MoE dispatch method")
        if armed.get("configuration", {}).get("scan_all_mla_layers"):
            if not boundary_layers.get(forward):
                gaps.append(f"forward {forward}: no expected MLA boundary layer list")
            for layer in boundary_layers.get(forward, []):
                for stage in ("mla_input", "mla_output"):
                    if boundaries[forward, layer, stage] != 1:
                        gaps.append(f"forward {forward}, {layer}: missing/duplicate {stage}")
        if hidden_required:
            for stage in ("model_output", "lm_head_input"):
                if hidden[forward, stage] != 1:
                    gaps.append(f"forward {forward}: missing/duplicate {stage}")
            for event in sampling:
                if event["event"] == "hidden_boundary" and event.get("forward_id") == forward:
                    if [row.get("request_id") for row in event.get("rows", [])] != batch["request_ids"]:
                        failures.append(f"forward {forward}: hidden boundary request order differs")
            mappings = [
                event for event in sampling if event["event"] == "hidden_mapping" and event.get("forward_id") == forward
            ]
            if len(mappings) != 1:
                gaps.append(f"forward {forward}: missing/duplicate hidden row mapping")
        for layer in selected:
            if begins[forward, layer] != 1 or ends[forward, layer] != 1:
                gaps.append(f"forward {forward}, {layer}: missing/duplicate begin/end")
        sample = samples.get(forward)
        if sample is None:
            gaps.append(f"forward {forward}: no correlated sampling end")
        elif [row["request_id"] for row in sample.get("rows", [])] != batch["request_ids"]:
            failures.append(f"forward {forward}: attention/sampling request order differs")
    stages = set()
    for event in events:
        kind = event["event"]
        if kind == "comparison":
            stages.add(event["stage"].split("/")[0])
            if event.get("passed") is not True:
                failures.append(f"forward {event.get('forward_id')} {event.get('layer')} {event['stage']}")
        if kind == "request_registry" and event.get("passed") is not True:
            failures.append(f"request registry mismatch: forward {event.get('forward_id')}")
        if kind == "writer_error":
            failures.append(f"writer exception: forward {event.get('forward_id')} {event.get('stage')}")
        if kind == "boundary" and event.get("passed") is not True:
            failures.append(f"nonfinite {event['stage']}: forward {event.get('forward_id')} {event.get('layer')}")
        if kind == "boundary" and event.get("tensor_file") and not (path.parent / event["tensor_file"]).is_file():
            gaps.append("missing first nonfinite boundary snapshot")
        if kind == "layer_begin" and not event.get("selected_requests"):
            gaps.append("layer has no selected request")
        if kind == "layer_end" and (not event.get("comparisons") or event.get("skipped")):
            gaps.append(f"incomplete layer {event.get('forward_id')} {event.get('layer')}")
        if kind == "layer_end" and event.get("failed"):
            failures.append(f"failed layer {event.get('forward_id')} {event.get('layer')}")
        if kind in ("check_skipped", "snapshot_skipped", "budget_exhausted"):
            gaps.append(f"{kind}: {event.get('stage', '')} {event.get('reason', '')}".strip())
        if kind == "layer_end":
            tensor_file = event.get("tensor_file")
            if not tensor_file or not (path.parent / tensor_file).is_file():
                gaps.append(f"missing attention snapshot for forward {event.get('forward_id')}")
    for event in sampling:
        if event["event"] == "boundary_skipped":
            gaps.append(f"sampling boundary skipped: {event.get('stage')}")
        if event["event"] == "hidden_mapping" and event.get("passed") is not True:
            failures.append(f"model output/LM-head row mapping differs: forward {event.get('forward_id')}")
        if event["event"] == "hidden_boundary":
            for row in event.get("rows", []):
                if row.get("nan") or row.get("positive_inf") or row.get("negative_inf"):
                    failures.append(
                        f"nonfinite {event['stage']}: forward {event.get('forward_id')} {row['request_id']}"
                    )
        if event["event"] in ("error", "mapping_mismatch"):
            failures.append(f"sampling {event['event']}: {event.get('message', '')}")
        if event["event"] == "budget_exhausted":
            gaps.append("sampling budget exhausted; later outputs unobserved")
        if event["event"] != "end":
            continue
        file = event.get("tensor_file")
        if not file or not (sample_path.parent / file).is_file():
            gaps.append(f"missing sampling snapshot for forward {event.get('forward_id')}")
        for row in event.get("rows", []):
            for stage in ("raw", "processed"):
                summary = row.get(stage, {})
                if summary.get("nan") or summary.get("positive_inf") or not summary.get("finite"):
                    failures.append(f"invalid {stage} logits: forward {event.get('forward_id')} {row['request_id']}")
    required = {"writer", "current", "merge", "flash"}
    if runtime and runtime.get("writer_references"):
        required |= {"prefill_writer", "decode_writer", "writer_guard"}
    if require_history:
        required |= {"gather", "history"}
    if required - stages:
        gaps.append(f"unobserved stages: {sorted(required - stages)}")
    chunks = [event for event in events if event["event"] == "history_chunk"]
    if require_history and not any(event["length"] > 0 for event in chunks):
        gaps.append("no positive history chunk observed")
    target_ids = set(armed.get("configuration", {}).get("request_ids", []))
    observed_ids = {req for batch in batches.values() for req in batch["request_ids"]}
    if target_ids - observed_ids:
        gaps.append(f"target request IDs never observed: {sorted(target_ids - observed_ids)}")
    anomalies = []
    for event in events + sampling:
        bad = event.get("passed") is False
        if event["event"] in ("hidden_boundary", "end"):
            for row in event.get("rows", []):
                if event["event"] == "hidden_boundary":
                    bad |= bool(row.get("nan") or row.get("positive_inf") or row.get("negative_inf"))
                else:
                    bad |= any(
                        row.get(stage, {}).get("nan")
                        or row.get(stage, {}).get("positive_inf")
                        or not row.get(stage, {}).get("finite")
                        for stage in ("raw", "processed")
                    )
        if bad:
            anomalies.append(
                {key: event[key] for key in ("utc", "forward_id", "layer", "stage", "event", "rows") if key in event}
            )
    anomalies.sort(key=lambda event: event.get("utc", ""))
    return dict(
        directory=str(path.parent.parent),
        dp_rank=armed.get("dp_rank"),
        tp_rank=armed.get("tp_rank"),
        batches=len(batches),
        layers=sorted(selected),
        stages=sorted(stages),
        runtime=runtime,
        first_observed_anomaly=anomalies[0] if anomalies else None,
        target_request_ids=sorted(target_ids),
        all_mla_boundary_scan=bool(armed.get("configuration", {}).get("scan_all_mla_layers")),
        zero_history_observed=any(event["length"] == 0 for event in chunks),
        positive_history_observed=any(event["length"] > 0 for event in chunks),
        failures=failures,
        gaps=gaps,
    )


def check_directory(directory, expected_dp_ranks, tp_rank=0, require_history=False):
    workers = [
        inspect_worker(path, require_history) for path in sorted(Path(directory).glob("**/attention/events.jsonl"))
    ]
    gaps = []
    observed = Counter((worker["dp_rank"], worker["tp_rank"]) for worker in workers)
    for rank in expected_dp_ranks:
        count = observed[rank, tp_rank]
        if count != 1:
            gaps.append(
                f"DP {rank}/TP {tp_rank}: expected one worker, found {count}; collect every node into one run directory"
            )
    if not workers:
        gaps.append("no attention evidence found")
    failed = any(worker["failures"] for worker in workers)
    incomplete = bool(gaps) or any(worker["gaps"] for worker in workers)
    code = 1 if failed else 2 if incomplete else 0
    return dict(
        status=("DISCREPANCY" if failed else "INCOMPLETE" if incomplete else "SAMPLED_CHECKS_COMPLETE"),
        exit_code=code,
        workers=workers,
        gaps=gaps,
        scope="Selected layers/requests/query rows/ranks only; finite outputs do not prove model accuracy. "
        "Synchronization may mask timing bugs. Budget exhaustion leaves later forwards unobserved.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-dp-ranks", required=True, help="Comma-separated selected DP ranks, e.g. 0,1,2,3")
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--require-history", action="store_true")
    args = parser.parse_args()
    try:
        ranks = [int(value) for value in args.expected_dp_ranks.split(",")]
        if not ranks or min(ranks) < 0 or len(set(ranks)) != len(ranks) or args.tp_rank < 0:
            raise ValueError
    except ValueError:
        parser.error("rank arguments must contain distinct nonnegative integers")
    report = check_directory(args.directory, ranks, args.tp_rank, args.require_history)
    print(json.dumps(report, indent=2))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
