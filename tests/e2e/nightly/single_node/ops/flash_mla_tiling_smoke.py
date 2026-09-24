# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone real-package eager/graph smoke; run on an authorized A5 environment.

This tests metadata/main integration against CPU float32 attention, not a model
or DSpark model. It does not mock or fall back when the package is absent.
"""

import argparse
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.flash_mla import (
    build_flash_mla_metadata,
    init_flash_mla_metadata,
    run_flash_mla,
)
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor


def reference(q_cpu, expected_cache, table_cpu, lens_cpu, query_len, causal, scale):
    heads, tokens = q_cpu.shape[1], len(lens_cpu) * query_len
    expected = torch.zeros(heads, tokens, 512, dtype=torch.float32)
    for request, length in enumerate(lens_cpu.tolist()):
        keys = expected_cache[table_cpu[request].long(), :, 0].reshape(-1, 576)[:length].float()
        q = q_cpu[request * query_len : (request + 1) * query_len].float().transpose(0, 1)
        logits = torch.matmul(q, keys.t()) * scale
        if causal:
            visible = torch.arange(length)[None, :] <= length - query_len + torch.arange(query_len)[:, None]
            logits.masked_fill_(~visible[None], float("-inf"))
        expected[:, request * query_len : (request + 1) * query_len] = torch.matmul(logits.softmax(-1), keys[:, :512])
    return expected


def check_graph_replays(builder, common, q_cpu, cache, expected_cache, table_cpu, lens_cpu, query_len, scale):
    """Capture only the main operator; update the real schedule outside capture."""
    executor = DeviceMetadataExecutor()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        q = q_cpu.to("npu")
        builder._flash_capture = True
        builder._device_metadata_enabled = True
        metadata = build_flash_mla_metadata(builder, common)
        builder._flash_capture = False

        def submit():
            tasks = builder._device_metadata_tasks
            executor.submit(tasks)
            for task in tasks:
                executor.wait(task.stage, task.group_id)

        submit()
        for _ in range(3):
            run_flash_mla(q, cache, metadata, scale)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            output, _ = run_flash_mla(q, cache, metadata, scale)
        executor.release()
        pointers = {name: value.data_ptr() for name, value in vars(metadata).items() if isinstance(value, torch.Tensor)}
        for step in range(2):
            lengths = lens_cpu - step * 3
            table = table_cpu if step == 0 else table_cpu.flip(1)
            query = q_cpu if step == 0 else q_cpu * 0.5
            common.seq_lens.copy_(lengths)
            common.block_table_tensor.copy_(table)
            q.copy_(query)
            current = build_flash_mla_metadata(builder, common)
            assert current is metadata
            submit()
            graph.replay()
            executor.release()
            stream.synchronize()
            expected = reference(query, expected_cache, table, lengths, query_len, common.causal, scale)
            torch.testing.assert_close(output[:, : expected.shape[1]].float().cpu(), expected, atol=0.02, rtol=0.02)
            assert pointers == {
                name: value.data_ptr() for name, value in vars(metadata).items() if isinstance(value, torch.Tensor)
            }


def run_case(heads: int, token_stride: int, query_len: int, kv_len: int, causal: bool = True, graph: bool = False):
    batch, padding, pages_per_req = 2, 2, 3
    pages = batch * pages_per_req
    page_stride = 128 * token_stride + 64
    offset = 32
    torch.manual_seed(20260924)
    backing = torch.full((offset + pages * page_stride,), -7, dtype=torch.bfloat16, device="npu")
    cache = torch.as_strided(backing, (pages, 128, 1, 576), (page_stride, token_stride, 576, 1), offset)
    payload = torch.randn(pages, 128, 1, 576, dtype=torch.bfloat16) * 0.1
    cache.copy_(payload.to("npu"))
    # Reorder pages: logical block number must not be treated as physical ID.
    table_cpu = torch.tensor([[2, 0, 4], [5, 1, 3]], dtype=torch.int32)
    lens_cpu = torch.tensor([kv_len, kv_len - 3], dtype=torch.int32)
    tokens = batch * query_len
    q_cpu = torch.randn(tokens + padding, heads, 576, dtype=torch.bfloat16) * 0.1
    slots_cpu = torch.empty(tokens, dtype=torch.int64)
    for request, length in enumerate(lens_cpu.tolist()):
        for i in range(query_len):
            position = length - query_len + i
            slots_cpu[request * query_len + i] = table_cpu[request, position // 128] * 128 + position % 128
    common = SimpleNamespace(
        num_reqs=batch,
        num_actual_tokens=tokens,
        num_input_tokens=tokens + padding,
        max_query_len=query_len,
        causal=causal,
        query_start_loc=torch.tensor([0, query_len, tokens], dtype=torch.int32, device="npu"),
        seq_lens=lens_cpu.to("npu"),
        block_table_tensor=table_cpu.to("npu"),
        slot_mapping=slots_cpu.to("npu"),
        positions=torch.arange(tokens, device="npu"),
    )
    builder = SimpleNamespace(device="npu", decode_threshold=1)
    init_flash_mla_metadata(builder, SimpleNamespace(num_heads=heads, num_kv_heads=1))
    metadata = build_flash_mla_metadata(builder, common)
    current = torch.randn(tokens, 1, 576, dtype=torch.bfloat16) * 0.1
    current_npu = current.to("npu")
    expected_backing = backing.cpu()
    expected_cache = torch.as_strided(expected_backing, cache.shape, cache.stride(), cache.storage_offset())
    for i, slot in enumerate(slots_cpu.tolist()):
        expected_cache[slot // 128, slot % 128] = current[i]
    torch_npu.npu_scatter_pa_kv_cache(
        key=current_npu[..., :512].contiguous(),
        value=current_npu[..., 512:].contiguous(),
        key_cache=cache[..., :512],
        value_cache=cache[..., 512:],
        slot_mapping=metadata.slots[:tokens],
        cache_mode="Norm",
    )
    scale = 576**-0.5
    actual, _ = run_flash_mla(q_cpu.to("npu"), cache, metadata, scale)
    torch.npu.synchronize()
    # Includes page gaps and leading/trailing guards; all untouched bytes exact.
    torch.testing.assert_close(backing.cpu(), expected_backing, atol=0, rtol=0)
    expected = reference(q_cpu, expected_cache, table_cpu, lens_cpu, query_len, causal, scale)
    # The integration masks physical padding before output projection; compare live rows here.
    torch.testing.assert_close(actual[:, :tokens].float().cpu(), expected, atol=0.02, rtol=0.02)
    if graph:
        check_graph_replays(builder, common, q_cpu, cache, expected_cache, table_cpu, lens_cpu, query_len, scale)
        torch.testing.assert_close(backing.cpu(), expected_backing, atol=0, rtol=0)
    print(f"PASS heads={heads} token_stride={token_stride} q={query_len} kv={kv_len} causal={causal} graph={graph}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, choices=(8, 12, 64, 96), default=8)
    parser.add_argument("--non-causal", action="store_true", help="Check the non-causal MLA DSpark query contract")
    parser.add_argument(
        "--graph", action="store_true", help="Also capture main and replay with updated device metadata"
    )
    args = parser.parse_args()
    torch.npu.set_device(0)
    for stride in (576, 1152):
        for q_len, kv_len in ((1, 127), (2, 129), (16, 257)):
            run_case(args.heads, stride, q_len, kv_len, causal=not args.non_causal, graph=args.graph)
