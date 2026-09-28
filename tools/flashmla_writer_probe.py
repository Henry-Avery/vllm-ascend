# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scratch-only writer/gather descriptor A/B. No device work without --execute.

Run with python -m tools.flashmla_writer_probe. Production KV is never loaded.
"""

import argparse
import json
import runpy
from pathlib import Path

import torch

BLOCK_SIZE = 128
LATENT = 512
ROPE = 64
WIDTH = LATENT + ROPE
PAGES = 3
OFFSET = 32


def make_views(page_stride, variant, device="cpu", dtype=torch.bfloat16):
    if page_stride < BLOCK_SIZE * WIDTH:
        raise ValueError("page stride must hold a full block")
    backing = torch.full((PAGES * page_stride + OFFSET,), 7, dtype=dtype, device=device)
    cache = backing.as_strided((PAGES, BLOCK_SIZE, 1, WIDTH), (page_stride, WIDTH, WIDTH, 1), OFFSET)
    views = (cache[..., :LATENT], cache[..., LATENT:])
    if variant == "singleton_rebuilt":
        views = tuple(view.squeeze(2).unsqueeze(2) for view in views)
    return backing, cache, views


def untouched_mask(size, slots, page_stride):
    mask = torch.ones(size, dtype=torch.bool)
    for slot in slots:
        start = OFFSET + slot // BLOCK_SIZE * page_stride + slot % BLOCK_SIZE * WIDTH
        mask[start : start + WIDTH] = False
    return mask


def execute_case(args, variant, use_rope, reference):
    # Lazy imports: metadata-only invocation must not initialize an NPU/runtime.
    import torch_npu

    from vllm_ascend.device.device_op import DeviceOperator

    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(3, WIDTH, generator=generator).to(torch.bfloat16)
    weight = torch.randn(LATENT, generator=generator).to(torch.bfloat16)
    angle = torch.randn(3, ROPE // 2, generator=generator).repeat(1, 2)
    cos, sin = angle.cos().to(raw.dtype), angle.sin().to(raw.dtype)
    expected = reference(raw, weight, 1e-6, use_rope, cos, sin)
    backing, cache, views = make_views(args.page_stride, variant, args.device)
    before = backing.cpu().clone()
    slots_cpu = [255, 0, 1]  # table [1,0], logical history [127,130).
    slots = torch.tensor(slots_cpu, dtype=torch.int64, device=args.device)
    if use_rope:
        torch_npu.npu_kv_rmsnorm_rope_cache(
            raw.to(args.device).view(3, 1, 1, WIDTH),
            weight.to(args.device),
            cos.to(args.device).view(3, 1, 1, ROPE),
            sin.to(args.device).view(3, 1, 1, ROPE),
            slots,
            views[1],
            views[0],
            epsilon=1e-6,
            cache_mode="PA",
            is_output_kv=False,
        )
    else:
        # Isolate scatter descriptor behavior from RMSNorm rounding.
        DeviceOperator.reshape_and_cache(
            key=expected[0].to(dtype=raw.dtype, device=args.device).view(3, 1, LATENT),
            value=expected[1].to(dtype=raw.dtype, device=args.device).view(3, 1, ROPE),
            key_cache=views[0],
            value_cache=views[1],
            slot_mapping=slots,
        )
    torch.npu.synchronize()
    actual = cache[slots // BLOCK_SIZE, slots % BLOCK_SIZE].cpu().reshape(3, WIDTH).float()
    expected_full = torch.cat(expected, -1)
    mask = untouched_mask(backing.numel(), slots_cpu, args.page_stride)
    untouched = torch.equal(backing.cpu()[mask], before[mask])
    written = torch.allclose(actual, expected_full, atol=args.atol, rtol=args.rtol)
    after_writer = backing.cpu().clone()
    gather_cases = []
    table_cpu = torch.tensor([1, 0])
    expected_history = cache.cpu()[table_cpu].reshape(-1, WIDTH)
    for start, length in ((0, 130), (128, 2), (128, 1)):
        key = torch.empty((length, 1, LATENT), dtype=raw.dtype, device=args.device)
        value = torch.empty((length, 1, ROPE), dtype=raw.dtype, device=args.device)
        DeviceOperator.kv_cache_load(
            views[0],
            views[1],
            table_cpu.to(device=args.device, dtype=torch.int32)[None],
            torch.tensor([length], dtype=torch.int32, device=args.device),
            torch.tensor([start], dtype=torch.int32, device=args.device),
            key=key,
            value=value,
        )
        torch.npu.synchronize()
        gathered = torch.cat((key.cpu().reshape(length, LATENT), value.cpu().reshape(length, ROPE)), -1)
        gather_cases.append(
            dict(start=start, length=length, passed=torch.equal(gathered, expected_history[start : start + length]))
        )
    gather_correct = all(case["passed"] for case in gather_cases)
    gather_readonly = torch.equal(after_writer, backing.cpu())
    return dict(
        variant=variant,
        use_rope=use_rope,
        component_strides=[list(view.stride()) for view in views],
        component_offsets=[view.storage_offset() for view in views],
        writer_matches_reference=bool(written),
        untouched_backing=untouched,
        gather_matches_written=gather_correct,
        gather_readonly=gather_readonly,
        gather_cases=gather_cases,
        max_abs=float((actual - expected_full).abs().max()),
        passed=bool(written and untouched and gather_correct and gather_readonly),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--page-stride", type=int, default=81408)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("flashmla-writer-probe.json"))
    args = parser.parse_args()
    if not BLOCK_SIZE * WIDTH <= args.page_stride <= 2 * BLOCK_SIZE * WIDTH:
        parser.error("page stride must be 73728..147456 elements")
    if not 0 <= args.atol <= 1 or not 0 <= args.rtol <= 1:
        parser.error("finite tolerances must be 0..1")
    variants = ("direct", "singleton_rebuilt")
    report = dict(
        executed=args.execute,
        page_stride=args.page_stride,
        offset=OFFSET,
        cases=[],
        scope="Synthetic BF16 scratch only; does not certify production writer, allocator or concurrency",
    )
    if args.execute:
        api = runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "vllm_ascend/attention/flashmla_chunk_diagnostics.py")
        )
        try:
            for use_rope in (False, True):
                for variant in variants:
                    report["active_case"] = dict(variant=variant, use_rope=use_rope)
                    report["cases"].append(execute_case(args, variant, use_rope, api["reference_kv"]))
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1 if report.get("error") or any(not case["passed"] for case in report["cases"]) else 0
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
