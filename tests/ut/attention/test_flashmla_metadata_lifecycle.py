# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression checks for the real FlashMLA metadata buffer lifecycle.

Run with --confcutdir=tests/ut/attention without installing vLLM/torch_npu.
Only external dependencies are stubbed; the builder, adapter and task classes
are loaded from production source. NPU stream fences and numerical attention
still require the release machine.
"""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).resolve().parents[3]

    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, relative_path):
        return install(name, **runpy.run_path(str(root / relative_path)))

    install(
        "vllm.forward_context", BatchDescriptor=object, get_forward_context=Mock(), is_forward_context_available=Mock()
    )
    install("vllm_ascend")
    adapter_module = load("vllm_ascend.attention.flashmla", "vllm_ascend/attention/flashmla.py")
    task_module = load("vllm_ascend.worker.device_metadata", "vllm_ascend/worker/device_metadata.py")

    def metadata_op(lengths, *, cu_seqlens_q, seqused_q, **kwargs):
        # Encode inputs so stale lengths/boundaries remain visible in schedule.
        # This also works on Meta and supplies capacity via the real adapter.
        return torch.cat((lengths, cu_seqlens_q, seqused_q))

    metadata = Mock(side_effect=metadata_op)
    monkeypatch.setattr(
        adapter_module.FlashMLAAdapter,
        "load",
        classmethod(lambda cls, config: cls(config, Mock(), metadata)),
    )

    def rope(positions):
        values = positions[:, None, None, None].expand(-1, 1, 1, 64).to(torch.bfloat16)
        return values, -values

    install("vllm_ascend.ops.rotary_embedding", get_cos_and_sin_mla=rope)
    module = load("flashmla_lifecycle_under_test", "vllm_ascend/attention/flashmla_metadata.py")

    def builder():
        impl = SimpleNamespace(num_heads=64, scale=0.125, dtype=torch.bfloat16, use_mla_rope=True)
        return module.FlashMLAMetadataBuilder(impl, torch.device("cpu"), max_num_reqs=4)

    return SimpleNamespace(builder=builder, metadata=metadata, stage=task_module.DeviceMetadataStage)


def common(lengths, boundaries, *, tokens=4, slots=None, positions=None, blocks=None):
    actual = boundaries[-1]
    return SimpleNamespace(
        num_actual_tokens=actual,
        num_input_tokens=tokens,
        causal=True,
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        query_start_loc=torch.tensor(boundaries, dtype=torch.int32),
        slot_mapping=torch.tensor(slots if slots is not None else list(range(actual)), dtype=torch.int64),
        positions=torch.tensor(positions if positions is not None else list(range(actual)), dtype=torch.int64),
        block_table_tensor=torch.tensor(blocks if blocks is not None else [[1, 2]] * len(lengths), dtype=torch.int32),
    )


def tensor_fields(flash):
    return {name: tensor for name, tensor in vars(flash).items() if isinstance(tensor, torch.Tensor)}


def test_metadata_refresh_never_reads_tensor_values_back_to_host(runtime, monkeypatch):
    builder = runtime.builder()
    inputs = common([9], [0, 1])

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata refresh must not read device tensor values back to the host")

    # Allocate before guarding refresh, so PyTorch's lazy Meta imports are outside
    # the check. This CPU test does not test NPU stream ordering.
    builder.defer = True
    builder.build(inputs, 1, 1, False)
    (task,) = builder.take_tasks()
    runtime.metadata.reset_mock()
    with monkeypatch.context() as patch:
        for method in ("cpu", "item", "tolist", "numpy", "__bool__"):
            patch.setattr(torch.Tensor, method, forbidden)
        task.run()
    runtime.metadata.assert_called_once()


def test_same_capacity_refreshes_new_requests_without_changing_addresses(runtime):
    builder = runtime.builder()
    first = builder.build(common([11, 22], [0, 1, 3]), 2, 3, False, retain_for_graph=True)
    addresses = {name: value.data_ptr() for name, value in tensor_fields(first).items()}
    previous_schedule = first.schedule.clone()
    second = builder.build(common([91], [0, 2], slots=[51, 52], positions=[89, 90], blocks=[[7, 8]]), 1, 2, False)
    assert first is second
    assert addresses == {name: value.data_ptr() for name, value in tensor_fields(second).items()}
    assert second.cache_lens.tolist() == [91, 0, 0, 0, 0]
    assert second.cu.tolist() == [0, 2, 4, 4, 4, 4]
    assert second.used_q.tolist() == [2, 0, 0, 0, 0]
    assert second.slots.tolist() == [51, 52, -1, -1]
    assert second.positions.tolist() == [89, 90, 0, 0]
    assert second.block_table.tolist() == [[7, 8], [0, 0], [0, 0], [0, 0], [0, 0]]
    assert second.cos[:, 0, 0, 0].tolist() == [89, 90, 0, 0]
    assert second.sin[:, 0, 0, 0].tolist() == [-89, -90, 0, 0]
    assert not torch.equal(previous_schedule, second.schedule)
    torch.testing.assert_close(second.schedule, torch.cat((second.cache_lens, second.cu, second.used_q)))
    assert runtime.metadata.call_count == 3  # One Meta allocation, two refreshes.


def test_defer_keeps_previous_step_intact_until_task_runs_and_take_is_once(runtime):
    builder = runtime.builder()
    flash = builder.build(common([11, 22], [0, 1, 3]), 2, 3, False, retain_for_graph=True)
    # query is owned by forward; initialize it for a deterministic snapshot.
    flash.query.zero_()
    snapshot = {name: value.clone() for name, value in tensor_fields(flash).items()}
    builder.defer = True
    runtime.metadata.reset_mock()
    same = builder.build(common([41], [0, 1], slots=[200]), 1, 1, False)
    assert same is flash
    runtime.metadata.assert_not_called()
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(flash, name), expected)
    (task,) = builder.take_tasks()
    assert task.stage == runtime.stage.ATTENTION
    assert task.group_id == id(flash.schedule)
    assert builder.take_tasks() == ()
    task.run()
    runtime.metadata.assert_called_once()
    assert flash.cache_lens.tolist() == [41, 0, 0, 0, 0]
    assert flash.slots.tolist() == [200, -1, -1, -1]
    torch.testing.assert_close(flash.schedule, torch.cat((flash.cache_lens, flash.cu, flash.used_q)))


def test_inactive_request_and_physical_padding_cannot_write_cache(runtime):
    builder = runtime.builder()
    flash = builder.build(common([19, 0], [0, 1, 3], slots=[100, 200, 201]), 2, 3, False)
    assert flash.cu.tolist() == [0, 1, 3, 4, 4, 4]
    assert flash.used_q.tolist() == [1, 0, 0, 0, 0]
    assert flash.cache_lens.tolist() == [19, 0, 0, 0, 0]
    assert flash.token_live.tolist() == [True, False, False, False]
    assert flash.slots.tolist() == [100, -1, -1, -1]


def test_mixed_batch_refresh_does_not_overwrite_retained_decode_buffers(runtime):
    builder = runtime.builder()
    decode = builder.build(common([12], [0, 1]), 1, 1, False, retain_for_graph=True)
    decode.query.zero_()
    snapshot = {name: value.clone() for name, value in tensor_fields(decode).items()}
    retained = dict(builder.buffers)
    mixed = common([44, 100], [0, 2, 5], tokens=5, slots=[80, 81, 200, 201, 202])
    first = builder.build(mixed, 1, 2, True)
    second = builder.build(mixed, 1, 2, True)
    assert first is not second and first is not decode
    assert first.schedule.data_ptr() != second.schedule.data_ptr()
    assert first.query.shape[0] == 2
    assert first.slots.tolist() == [80, 81]
    assert first.cache_lens.tolist() == [44, 0]
    assert first.cu.tolist() == [0, 2, 2]
    assert builder.buffers.keys() == retained.keys()
    assert all(builder.buffers[key] is value for key, value in retained.items())
    for name, expected in snapshot.items():
        torch.testing.assert_close(getattr(decode, name), expected)


def test_varying_eager_shapes_do_not_accumulate_permanent_buffers(runtime):
    builder = runtime.builder()
    for tokens in range(1, 17):
        flash = builder.build(common([40], [0, tokens], tokens=tokens), 1, tokens, False)
        assert flash.query.shape[0] == tokens
        assert builder.buffers == {}


def test_only_explicit_graph_shapes_are_retained_and_reused_by_eager(runtime):
    builder = runtime.builder()
    captured = builder.build(common([20], [0, 1]), 1, 1, False, retain_for_graph=True)
    retained = dict(builder.buffers)
    assert len(retained) == 1
    addresses = {name: value.data_ptr() for name, value in tensor_fields(captured).items()}
    for tokens in range(1, 9):
        flash = builder.build(common([30], [0, 1], tokens=tokens), 1, 1, False)
        assert builder.buffers.keys() == retained.keys()
        assert all(builder.buffers[key] is value for key, value in retained.items())
        if tokens == captured.query.shape[0]:
            assert flash is captured
            assert addresses == {name: value.data_ptr() for name, value in tensor_fields(flash).items()}
            assert flash.cache_lens.tolist() == [30, 0, 0, 0, 0]
        else:
            assert flash is not captured
