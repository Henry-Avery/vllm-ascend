# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare an already running colocated baseline and PD proxy; starts no servers."""

import argparse
import concurrent.futures
import json
import math
import urllib.request
from pathlib import Path


def complete(url, model, prompt, max_tokens):
    data = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 1,
            "logprobs": 1,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    choice = result["choices"][0]
    assert choice.get("logprobs"), "Endpoint must return token logprobs for numerical comparison"
    return choice


def compare_case(case, args):
    baseline = complete(args.baseline, args.model, case["prompt"], args.max_tokens)
    pd = complete(args.pd, args.model, case["prompt"], args.max_tokens)
    left, right = baseline["logprobs"], pd["logprobs"]
    assert baseline["text"] == pd["text"], f"{case['name']}: generated text differs"
    assert baseline["finish_reason"] == pd["finish_reason"], f"{case['name']}: finish reason differs"
    assert left["tokens"] == right["tokens"], f"{case['name']}: generated tokens differ"
    a, b = left["token_logprobs"], right["token_logprobs"]
    assert len(a) == len(b) and a, f"{case['name']}: missing token scores"
    assert all(x is not None and math.isfinite(x) for x in a + b), "Non-finite/missing token logprobs"
    error = max(abs(x - y) for x, y in zip(a, b))
    assert error <= args.atol, f"{case['name']}: maximum logprob error {error} exceeds {args.atol}"
    return {"name": case["name"], "max_logprob_error": error, "baseline": baseline, "pd": pd}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Already running colocated server URL")
    parser.add_argument("--pd", required=True, help="Already running PD proxy URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True, help="JSON list of {name,prompt}; prompt may be token IDs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--atol", type=float, required=True, help="Pre-agreed absolute logprob tolerance")
    args = parser.parse_args()
    if args.atol < 0 or not math.isfinite(args.atol) or min(args.concurrency, args.repeat, args.max_tokens) < 1:
        parser.error("Tolerance must be finite and nonnegative; count options must be positive")
    cases = json.loads(args.cases.read_text())
    if not cases or not all(isinstance(case.get("name"), str) and case.get("prompt") for case in cases):
        parser.error("Provide nonempty named prompt cases")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(compare_case, case, args) for _ in range(args.repeat) for case in cases]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    finally:
        args.output.write_text(json.dumps({"config": vars(args), "completed_cases": results}, default=str, indent=2))
    print(f"Compared {len(results)} cases; separately verify logs show real remote READs, not local recomputation.")


if __name__ == "__main__":
    main()
