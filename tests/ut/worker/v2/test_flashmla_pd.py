# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU allocator/connector seam; device, network and model configs are substitutes.

Run with --confcutdir=tests/ut/worker/v2. Production classes and fixed ced685
collector/scheduler excerpts execute unchanged; ctypes stands in for RDMA.
"""

import ast
import ctypes
import dataclasses
import json
import math
import queue
import runpy
import threading
from collections.abc import Sequence
from concurrent.futures import Future, as_completed
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
import torch
from test_hybrid_descriptor_layout import _planner_config
from test_hybrid_descriptor_layout import descriptor_api as _descriptor_api
from test_hybrid_state_page_layout import ROOT, _load_functions

MOONCAKE = "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/"


def load_classes(path, names, ns):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), path, "exec"), ns)


@pytest.fixture
def pd_api(monkeypatch):
    ns, upstream = _descriptor_api.__wrapped__(monkeypatch)
    ns.update(runpy.run_path(str(ROOT / MOONCAKE / "layout.py")))
    ns.update(
        __name__=__name__,
        dataclass=dataclasses.dataclass,
        field=dataclasses.field,
        threading=threading,
        queue=queue,
        Sequence=Sequence,
        math=math,
        KVConnectorHandshakeMetadata=object,
        KVConnectorMetadata=object,
        KVConnectorBase_V1=type("ConnectorBase", (), {}),
        SupportsHMA=type("HMA", (), {}),
        envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        global_te=MagicMock(),
        tensor_storage_key=lambda t: t.untyped_storage().data_ptr(),
        validate_register_region_count=MagicMock(),
        SlidingWindowMLASpec=type("SlidingWindowMLASpec", (), {}),
        RegisterRegions=NS,
        KV_CACHE_BUFFER_ALIGNMENT=2 * 1024 * 1024,
    )
    ns["FullAttentionSpec"] = upstream["FullAttentionSpec"]
    ns["SlidingWindowSpec"] = type("SlidingWindowSpec", (), {})
    ns["CircularBufferSpec"] = type("CircularBufferSpec", (), {})
    _load_functions("vllm_ascend/core/kv_cache_interface.py", ["is_circular_kv_cache_spec"], ns)
    _load_functions("vllm_ascend/worker/v2/attn_utils.py", ["_align_memory"], ns)
    _load_functions(
        MOONCAKE + "utils.py",
        [
            "as_kv_cache_tensors",
            "collect_configured_register_regions",
            "_get_storage_nbytes",
            "_get_tensor_span_nbytes",
            "group_concurrent_contiguous",
        ],
        ns,
    )
    load_classes(
        MOONCAKE + "metadata.py",
        [
            "MooncakeTransferMetadata",
            "MooncakeTPTransferMetadata",
            "MooncakePCPTransferMetadata",
            "MooncakePPTransferMetadata",
            "MooncakeTransferMetadataGroups",
            "ReqMeta",
            "MooncakeConnectorMetadata",
        ],
        ns,
    )
    fixture = json.loads((Path(__file__).parent / "fixtures/ced685_pd_sources.json").read_text())
    assert fixture["upstream_sha"] == "ced6857afa0ea7b2e3f0846a62e1394e90f15607"
    for source in fixture["sources"].values():
        exec("from __future__ import annotations\n" + source["verbatim_source"], ns)
    load_classes(MOONCAKE + "base_worker.py", ["MooncakeBaseConnectorWorker"], ns)
    load_classes(MOONCAKE + "pull_worker.py", ["MooncakePullRecvingThread", "MooncakePullConnectorWorker"], ns)
    load_classes(MOONCAKE + "base_scheduler.py", ["MooncakeBaseConnectorScheduler"], ns)
    load_classes(
        MOONCAKE + "pull_scheduler.py", ["MooncakeSchedulerSendingThread", "MooncakePullConnectorScheduler"], ns
    )
    load_classes(MOONCAKE + "connector.py", ["MooncakeBaseConnector"], ns)
    return ns, upstream


def allocate_and_register(api, layout, ratio, num_blocks=4):
    ns, upstream = api
    plan, config, _, mamba = _planner_config(
        ns, upstream, upstream["KVCacheLayout"][layout], ratio, num_blocks=num_blocks
    )
    mamba.mamba_type = "KDA"
    config.kv_transfer_config = NS(kv_connector="MooncakeConnectorV2", kv_connector_module_path=None)
    config.parallel_config.prefill_context_parallel_size = 1
    config.speculative_config = None
    ns["get_current_vllm_config"] = lambda: config
    ns["_get_attention_kv_cache_dims"] = lambda *_: (512, 64)
    raw = ns["_allocate_kv_cache"](plan, {}, torch.device("cpu"))
    groups = [
        upstream["AttentionGroup"](
            backend=None, layer_names=g.layer_names, kv_cache_spec=g.kv_cache_spec, kv_cache_group_id=i
        )
        for i, g in enumerate(plan.kv_cache_groups)
    ]
    caches = ns["_reshape_kv_cache_v2"](
        groups, raw, "auto", [128, mamba.block_size], {}, plan, kv_cache_layout=upstream["KVCacheLayout"][layout]
    )
    cls = ns["MooncakeBaseConnectorWorker"]
    worker = cls.__new__(cls)
    worker.kv_cache_config = plan
    worker.ascend_config = NS(kvpp_config=NS(size=1))
    worker.pcp_rank = worker.tp_rank = 0
    worker.tp_size = 1
    worker.engine_id, worker.te_rpc_port = "engine", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 128 * ratio, "localhost", 5000
    worker.register_kv_caches(caches)
    return worker, raw, caches


@pytest.mark.parametrize("layout", ["BLHNC", "LBHNC", "LBNHC"])
@pytest.mark.parametrize("ratio", [1, 3, 6])
def test_planner_allocator_registration_to_different_blocks_preserves_neighbors(pd_api, layout, ratio):
    ns, _ = pd_api
    producer, _, p_caches = allocate_and_register(pd_api, layout, ratio)
    consumer, d_raw, d_caches = allocate_and_register(pd_api, layout, ratio, num_blocks=5)
    p, d = producer.transfer_metadata, consumer.transfer_metadata
    assert p.fused_mla_protocol == d.fused_mla_protocol == 1
    assert p.layer_layouts == d.layer_layouts
    backing = next(iter(d_raw.values())).untyped_storage()
    target = torch.empty(0, dtype=torch.uint8).set_(backing, 0, (backing.nbytes(),), (1,))
    target.fill_(73)
    before = target.clone()
    allowed = torch.zeros_like(target, dtype=torch.bool)
    thread, peer = make_peer(pd_api, d)
    thread.tp_rank = 0
    thread.num_speculative_tokens = 0
    thread.kv_cache_specs = consumer.kv_cache_specs
    for field in ("layer_names", "kv_caches_base_addr", "block_strides", "block_lens", "block_shapes"):
        setattr(thread, field, getattr(d, field))
    merge_cls = ns["MooncakeSchedulerSendingThread"]
    merger = merge_cls.__new__(merge_cls)
    merger.engine_id, merger.pp_size, merger.pcp_size, merger.tp_size = "engine", 1, 1, 1
    merger.use_kv_pp = False
    remote = merger._merge_metadata_by_pp_rank({0: p})[0]
    thread._validate_fused_mla_layout(peer, remote)
    for name in ("mla0", "kda0"):
        li, ri = d.layer_names.index(name), remote.layer_names.index(name)
        spec_index = consumer.layer_name_to_spec_index[name]
        spec = consumer.kv_cache_specs[spec_index]
        scale = d.block_size_scales[li][0]
        ranks = thread._compute_group_block_ids(
            "req",
            [[0]],
            1,
            spec_index,
            spec.block_size,
            spec.block_size,
            [4],
            [4],
            [0],
            spec.block_size,
            spec.block_size,
            0,
            scale,
            scale,
            spec,
            0,
        )
        pc = ns["as_kv_cache_tensors"](p_caches[name])
        dc = ns["as_kv_cache_tensors"](d_caches[name])
        for pv in pc:
            pv[:scale].fill_(5)
        src, dst, lengths = [], [], []
        thread._append_spec_transfer_addresses(
            spec_index,
            0,
            0,
            1,
            1,
            1,
            {(li, ri): [("req", ranks[0][1], ranks[0][2])]},
            remote,
            src,
            dst,
            lengths,
        )
        # Mooncake READ uses local addresses as src and remote addresses as dst.
        for local_addr, remote_addr, length in zip(src, dst, lengths):
            ctypes.memmove(local_addr, remote_addr, length)
            start = local_addr - target.data_ptr()
            assert 0 <= start < target.numel() and start + length <= target.numel()
            allowed[start : start + length] = True
        for pv, dv in zip(pc, dc):
            for local_block, remote_block in zip(ranks[0][1], ranks[0][2]):
                assert torch.equal(pv[remote_block], dv[local_block])
    assert torch.equal(target[~allowed], before[~allowed])
    assert ns["global_te"].register_buffer.call_count == 2


def test_packed_components_do_not_transfer_neighbor_or_padding(pd_api):
    ns, _ = pd_api
    worker = ns["MooncakeBaseConnectorWorker"].__new__(ns["MooncakeBaseConnectorWorker"])
    worker.num_blocks = 4
    raw = torch.zeros(4 * 128, dtype=torch.float16)
    a = raw.as_strided((4, 32), (128, 1))
    b = raw.as_strided((4, 8), (128, 1), 32)
    assert worker._get_shared_page_metadata((a,)) is None
    assert worker._get_shared_page_metadata((a, b)) is not None
    # A gap may be owned by another layer: do not collapse it.
    c = raw.as_strided((4, 8), (128, 1), 64)
    assert worker._get_shared_page_metadata((a, c)) is None
    assert not ns["block_is_contiguous"](raw.as_strided((4, 4, 4), (128, 8, 1)))


def make_peer(api, local):
    ns, _ = api
    cls = ns["MooncakePullRecvingThread"]
    thread = cls.__new__(cls)
    thread.local_metadata = local
    thread.tp_size = thread.pp_size = thread.pcp_size = thread.dcp_size = 1
    peer = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, use_kv_pp=False)
    return thread, peer


@pytest.mark.parametrize("bad", ["old_peer", "dtype", "shape", "format", "scale", "length", "tp", "cp"])
def test_layout_mismatch_is_rejected_before_transfer(pd_api, bad):
    ns, _ = pd_api
    worker, _, _ = allocate_and_register(pd_api, "BLHNC", 3)
    local = worker.transfer_metadata
    thread, peer = make_peer(pd_api, local)
    remote = dataclasses.replace(local)
    if bad == "old_peer":
        remote = dataclasses.replace(remote, fused_mla_protocol=0)
    elif bad in ("dtype", "shape", "format"):
        fields = {
            "dtype": {"dtypes": ("torch.float16",)},
            "shape": {"shapes": ((1, 128, 576),)},
            "format": {"kind": "component-major"},
        }
        remote = dataclasses.replace(
            remote, layer_layouts=[dataclasses.replace(local.layer_layouts[0], **fields[bad]), *local.layer_layouts[1:]]
        )
    elif bad == "scale":
        remote = dataclasses.replace(remote, block_size_scales=[[1], *local.block_size_scales[1:]])
    elif bad == "length":
        remote = dataclasses.replace(remote, block_lens=[[1], *local.block_lens[1:]])
    elif bad == "tp":
        peer.tp_size = 2
    else:
        peer.pcp_size = 2
    peer.metadata_by_pp_rank = {0: remote}
    thread.executor = MagicMock()
    with pytest.raises(ValueError):
        thread._build_remote_transfer_layout(peer)
    thread.executor.submit.assert_not_called()


def test_layout_allows_different_capacity_and_physical_pitch(pd_api):
    _, _ = pd_api
    worker, _, _ = allocate_and_register(pd_api, "BLHNC", 3)
    local = worker.transfer_metadata
    thread, peer = make_peer(pd_api, local)
    remote = dataclasses.replace(
        local, num_blocks=100, block_strides=[[2 * x for x in row] for row in local.block_strides]
    )
    thread._validate_fused_mla_layout(peer, remote)


@pytest.mark.parametrize("failure", ["none", "partial", "exception"])
@pytest.mark.parametrize("recompute", [True, False])
def test_hybrid_failure_reaches_paired_collector_and_scheduler(pd_api, monkeypatch, failure, recompute):
    ns, _ = pd_api
    cls = ns["MooncakePullRecvingThread"]
    thread = cls.__new__(cls)
    thread.device = "cpu"
    thread.ready_event = threading.Event()
    thread.can_report_invalid_block_ids = False
    thread.finished_requests = queue.SimpleQueue()
    thread.request_queue = MagicMock()
    requests = {r: NS(remote_host="host", remote_port=1, local_block_ids=([2], [2])) for r in ("a", "b")}
    thread.request_queue.get.side_effect = [("p", requests), StopIteration]
    thread._handle_requests = MagicMock(return_value={"b"} if failure == "partial" else set())
    if failure == "exception":
        thread._handle_requests.side_effect = RuntimeError("transfer failed")
    monkeypatch.setattr(torch, "npu", NS(set_device=lambda _: None), raising=False)
    with pytest.raises(StopIteration):
        thread.run()
    worker_cls = ns["MooncakePullConnectorWorker"]
    worker = worker_cls.__new__(worker_cls)
    worker._recving_thread = thread
    facade_cls = ns["MooncakeBaseConnector"]
    facade = facade_cls.__new__(facade_cls)
    facade.connector_worker = worker
    connector = MagicMock()
    connector.get_transfer_results.side_effect = facade.get_transfer_results
    connector.get_block_ids_with_load_errors.return_value = set()
    output = ns["post_forward"](NS(_disabled=False, _pending_load_kwargs=None, kv_connector=connector), set())
    expected = set() if failure == "none" else ({"b"} if failure == "partial" else {"a", "b"})
    assert output.finished_recving == {"a", "b"}
    assert output.failed_recving == expected
    assert not facade.get_transfer_results(set()).finished_recving
    ns["RequestStatus"] = NS(WAITING_FOR_REMOTE_KVS="waiting")
    states = {r: NS(request_id=r, status="waiting", num_computed_tokens=32) for r in requests}
    scheduler = NS(
        requests=states,
        recompute_kv_load_failures=recompute,
        failed_recving_kv_req_ids=set(),
        finished_recving_kv_req_ids={"a", "b"},
        connector=object(),
        kv_cache_manager=MagicMock(),
    )
    errors = ns["_handle_failed_recving"](scheduler, output.failed_recving)
    assert errors == (set() if recompute else expected)
    if recompute:
        for req in expected:
            assert states[req].num_computed_tokens == 0
            ns["_update_waiting_for_remote_kv"](scheduler, states[req])
        assert scheduler.kv_cache_manager.free.call_count == len(expected)


@pytest.mark.parametrize(
    "connector,custom,speculative,pcp,allowed",
    [
        ("MooncakeConnectorV2", None, None, 1, True),
        ("MooncakePullConnector", None, None, 1, True),
        ("MooncakeConnectorV1", None, None, 1, False),
        ("MooncakeConnectorV2", "custom", None, 1, False),
        ("MooncakeConnectorV2", None, object(), 1, False),
        ("MooncakeConnectorV2", None, None, 2, False),
    ],
)
def test_fused_pd_gate(pd_api, connector, custom, speculative, pcp, allowed):
    ns, _ = pd_api
    config = NS(
        kv_transfer_config=NS(kv_connector=connector, kv_connector_module_path=custom),
        parallel_config=NS(prefill_context_parallel_size=pcp, decode_context_parallel_size=1),
        speculative_config=speculative,
    )
    assert ns["supports_flashmla_pd"](config) is allowed


@pytest.mark.parametrize("mismatch", ["protocol", "dtype", "none"])
def test_scheduler_preserves_layout_and_rejects_inconsistent_tp(pd_api, mismatch):
    ns, _ = pd_api
    worker, _, _ = allocate_and_register(pd_api, "BLHNC", 3)
    local = worker.transfer_metadata
    other = dataclasses.replace(local)
    if mismatch == "protocol":
        other = dataclasses.replace(other, fused_mla_protocol=0)
    elif mismatch == "dtype":
        other = dataclasses.replace(
            other,
            layer_layouts=[
                dataclasses.replace(local.layer_layouts[0], dtypes=("torch.float16",)),
                *local.layer_layouts[1:],
            ],
        )
    cls = ns["MooncakeSchedulerSendingThread"]
    scheduler = cls.__new__(cls)
    scheduler.engine_id = "engine"
    scheduler.pp_size = scheduler.pcp_size = 1
    scheduler.tp_size = 2
    scheduler.use_kv_pp = False
    if mismatch != "none":
        with pytest.raises(ValueError, match="differs across"):
            scheduler._merge_metadata_by_pp_rank({0: local, 1: other})
    else:
        merged = scheduler._merge_metadata_by_pp_rank({0: local, 1: other})[0]
        assert merged.fused_mla_protocol == 1
        for name, layout in zip(merged.layer_names, merged.layer_layouts):
            assert layout == local.layer_layouts[local.layer_names.index(name)]


def test_failed_receive_is_acknowledged_once_to_release_producer(pd_api):
    ns, _ = pd_api
    cls = ns["MooncakePullConnectorScheduler"]
    scheduler = cls.__new__(cls)
    scheduler._reqs_recv_info = {"failed": ("p", 5000, "remote-id")}
    scheduler._recving_thread = MagicMock()
    scheduler._sending_thread = None
    output = ns["KVConnectorOutput"](finished_recving={"failed"}, failed_recving={"failed"})
    scheduler.update_connector_output(output)
    scheduler.update_connector_output(output)
    scheduler._recving_thread.add_request.assert_called_once_with("p", 5000, "remote-id")
    assert not scheduler._reqs_recv_info


def test_hybrid_failed_transfer_bucket_is_not_ignored(pd_api):
    ns, _ = pd_api
    ns["as_completed"] = as_completed
    cls = ns["MooncakePullRecvingThread"]
    thread = cls.__new__(cls)
    thread.can_report_invalid_block_ids = False
    thread._get_remote_metadata = MagicMock(
        return_value=NS(metadata_by_pp_rank={0: object()}, pcp_size=1, tp_size=2, dcp_size=1)
    )
    thread.remote_tp_rank_groups = {"p": {0: {(0, 0): [[0], [1]]}}}
    thread.remote_layer_index_pairs = {"p": {0: [(0, 0)]}}
    thread._build_transfer_block_buckets = MagicMock(
        return_value=({(0, 0): {}, (0, 1): {}}, {(0, 0): {"ok"}, (0, 1): {"failed"}})
    )
    success, failure = Future(), Future()
    success.set_result(None)
    failure.set_exception(TimeoutError("READ timed out"))
    thread.executor = MagicMock()
    thread.executor.submit.side_effect = [success, failure]
    assert thread._handle_requests("p", "host", 1, {}) == {"failed"}


def test_paired_aggregator_retains_failure_until_all_ranks_finish(pd_api):
    ns, _ = pd_api
    aggregator = ns["KVOutputAggregator"](2)
    output_cls = ns["KVConnectorOutput"]
    first = aggregator.aggregate(
        [
            NS(kv_connector_output=output_cls(finished_recving={"r"}, failed_recving={"r"})),
            NS(kv_connector_output=output_cls()),
        ]
    )
    assert not first.kv_connector_output.finished_recving
    assert not first.kv_connector_output.failed_recving
    second = aggregator.aggregate(
        [
            NS(kv_connector_output=output_cls()),
            NS(kv_connector_output=output_cls(finished_recving={"r"})),
        ]
    )
    assert second.kv_connector_output.finished_recving == {"r"}
    assert second.kv_connector_output.failed_recving == {"r"}
    assert not aggregator._failed_recving_pending


def test_fused_pd_rejects_v1_runner(pd_api):
    ns, _ = pd_api
    worker, _, _ = allocate_and_register(pd_api, "BLHNC", 1)
    _load_functions(
        "vllm_ascend/worker/model_runner_v1.py", ["_uses_single_raw_mla_cache"], ns, class_name="NPUModelRunner"
    )
    ns["ascend_envs"] = NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
    spec = worker.kv_cache_specs[worker.layer_name_to_spec_index["mla0"]]
    with pytest.raises(ValueError, match="Model Runner V2"):
        ns["_uses_single_raw_mla_cache"](NS(vllm_config=NS(kv_transfer_config=object())), "mla0", spec)
