# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; run this file directly to avoid the NPU UT conftest.

The external operators and streams are spies, not kernel/ACL Graph validation.
Real CPU tensors exercise strides, offsets, values, padding and buffer reuse.
"""

import ast
import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
EXEC_SPEC = importlib.util.spec_from_file_location(
    "vllm_ascend.worker.v2.flash_mla_metadata", ROOT / "vllm_ascend/worker/v2/flash_mla_metadata.py"
)
EXECUTOR = importlib.util.module_from_spec(EXEC_SPEC)
sys.modules[EXEC_SPEC.name] = EXECUTOR
EXEC_SPEC.loader.exec_module(EXECUTOR)
SPEC = importlib.util.spec_from_file_location(
    "external_flash_mla_test_module", ROOT / "vllm_ascend/attention/flash_mla.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_method(path, class_name, method_name, namespace):
    """Execute the actual integration method without importing NPU packages."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name) if class_name else tree
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(ast.unparse(module), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag),
        namespace,
    )
    return namespace[method_name]


class FakeStream:
    def __init__(self, name="consumer", calls=None):
        self.name = name
        self.calls = calls if calls is not None else []
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)
        self.calls.append((self.name, "wait_stream", other.name))

    def wait_event(self, event):
        self.calls.append((self.name, "wait", event.name))


class FakeEvent:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def record(self, stream):
        self.calls.append((stream.name, "record", self.name))


class FlashMLATest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.events = []
        self.stream = FakeStream(calls=self.calls)

        @contextmanager
        def stream_context(stream):
            old, self.stream = self.stream, stream
            try:
                yield
            finally:
                self.stream = old

        def event():
            value = FakeEvent(f"event{len(self.events)}", self.calls)
            self.events.append(value)
            return value

        self.npu = types.SimpleNamespace(
            current_stream=Mock(side_effect=lambda: self.stream),
            is_current_stream_capturing=Mock(return_value=False),
            Stream=lambda: FakeStream("metadata", self.calls),
            Event=event,
            stream=stream_context,
        )
        self.metadata_calls = []
        self.main_calls = []
        self.bad_capacity = False

        def metadata_op(lengths, heads, kv_heads, **kw):
            self.metadata_calls.append((lengths, heads, kv_heads, kw))
            size = 32 + lengths.numel() * 16
            if lengths.device.type == "meta":
                return torch.empty(size, dtype=torch.int32, device="meta")
            self.calls.append((self.stream.name, "schedule"))
            return torch.full((size + int(self.bad_capacity),), int(lengths.sum()), dtype=torch.int32)

        def main_op(q, cache, **kw):
            self.main_calls.append((q, cache, kw))
            out = torch.ones(q.shape[1], q.shape[0], 512, dtype=q.dtype)
            live = int(kw["seqused_q"].sum())
            out[:, live:] = float("nan")
            return out, torch.empty(1)

        self.ops = types.SimpleNamespace(flash_mla_with_kvcache_metadata=metadata_op, flash_mla_with_kvcache=main_op)
        self.patches = [
            patch.object(torch, "npu", self.npu, create=True),
            patch.dict(sys.modules, cann_ops_transformer=self.ops),
            patch.object(
                torch.Tensor, "record_stream", lambda tensor, stream: self.calls.append((stream.name, "retain"))
            ),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.executor = EXECUTOR.DeviceMetadataExecutor()
        self.context = EXECUTOR.device_metadata_context(self.executor)
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def prepare(self, builder, common, meta, actual):
        if self.executor.submission_in_flight:
            self.executor.release()
        tasks = []
        flash = builder.build(common, meta, actual, tasks)
        self.executor.submit(tasks)
        for task in tasks:
            self.executor.wait(task.group_id)
        return flash

    def inputs(self, *, batch=2, tokens=2, live=2, causal=True, prefills=0):
        common = types.SimpleNamespace(
            num_input_tokens=tokens,
            block_table_tensor=torch.tensor([[2, 0, 1]] * batch, dtype=torch.int32),
            query_start_loc=torch.arange(batch + 1, dtype=torch.int32),
            seq_lens=torch.arange(129, 129 + batch, dtype=torch.int32),
            causal=causal,
        )
        meta = types.SimpleNamespace(num_decodes=batch, num_decode_tokens=live, num_prefills=prefills)
        return common, meta

    def cache(self, dtype=torch.bfloat16, token_gap=0):
        token_stride = 576 + token_gap
        page_stride = 128 * token_stride + 1024
        backing = torch.arange(3 * page_stride + 23, dtype=torch.float32).to(dtype)
        cache = backing.as_strided((3, 128, 1, 576), (page_stride, token_stride, 576, 1), 23)
        return backing, cache

    def test_contract_heads_dtypes_and_original_strided_cache(self):
        for heads in (8, 12, 64, 96):
            for dtype in (torch.bfloat16, torch.float16):
                for gap in (0, 32):
                    with self.subTest(heads=heads, dtype=dtype, gap=gap):
                        builder = MODULE.FlashMLABuilder(heads, "cpu")
                        common, meta = self.inputs()
                        flash = self.prepare(builder, common, meta, 2)
                        backing, cache = self.cache(dtype, gap)
                        before = backing.clone()
                        nope = torch.ones(2, heads, 512, dtype=dtype)
                        pe = torch.full((2, heads, 64), 2, dtype=dtype)
                        out = MODULE.flash_mla_decode(nope, pe, cache, flash, 0.125)
                        q, received, kw = self.main_calls[-1]
                        self.assertIs(received, cache)
                        self.assertEqual(received.storage_offset(), 23)
                        self.assertTrue(torch.equal(backing, before))
                        self.assertTrue(torch.equal(q[..., :512], nope))
                        self.assertTrue(torch.equal(q[..., 512:], pe))
                        self.assertEqual(out.shape, (heads, 2, 512))
                        self.assertEqual((kw["layout_q"], kw["layout_kv"], kw["layout_out"]), ("TND", "PA_BBND", "NTD"))
                        self.assertEqual((kw["max_seqlen_q"], kw["max_seqlen_kv"]), (-1, -1))
                        self.assertEqual(kw["softmax_scale"], 0.125)
                        lengths, h, kh, mk = self.metadata_calls[-1]
                        self.assertEqual((h, kh), (heads, 1))
                        self.assertIs(lengths, kw["cache_seqlens"])
                        for name in ("cu_seqlens_q", "seqused_q"):
                            self.assertIs(mk[name], kw[name])
                        for name in ("max_seqlen_q", "max_seqlen_kv", "mask_mode", "layout_q", "head_dim_v"):
                            self.assertEqual(mk[name], kw[name])
                        self.assertIs(kw["metadata"], flash.schedule)

    def test_dynamic_replay_values_and_stable_addresses(self):
        builder = MODULE.FlashMLABuilder(12, "cpu")
        common, meta = self.inputs(batch=4, tokens=4, live=4)
        first = self.prepare(builder, common, meta, 4)
        addresses = {k: v.data_ptr() for k, v in vars(first).items() if isinstance(v, torch.Tensor)}
        common.seq_lens = torch.tensor([100_000, 128_000, 999, 999], dtype=torch.int32)
        common.block_table_tensor[0] = torch.tensor([1, 2, 0])
        meta.num_decode_tokens = 2
        second = self.prepare(builder, common, meta, 2)
        self.assertIs(first, second)
        self.assertEqual(addresses, {k: v.data_ptr() for k, v in vars(second).items() if isinstance(v, torch.Tensor)})
        self.assertEqual(second.cache_lens.tolist(), [100_000, 128_000, 0, 0])
        self.assertEqual(second.used_q.tolist(), [1, 1, 0, 0])
        self.assertEqual(second.token_live.tolist(), [True, True, False, False])
        self.assertEqual(second.block_table[0].tolist(), [1, 2, 0])
        self.assertTrue(torch.all(second.schedule == 228_000))
        _, cache = self.cache()
        out = MODULE.flash_mla_decode(
            torch.ones(4, 12, 512, dtype=cache.dtype), torch.ones(4, 12, 64, dtype=cache.dtype), cache, second, 0.1
        )
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(int(out[:, 2:].count_nonzero()), 0)

    def test_short_query_and_mixed_prefill_prefix(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs(tokens=37, live=17, prefills=1)
        common.query_start_loc = torch.tensor([0, 1, 17, 37], dtype=torch.int32)
        flash = self.prepare(builder, common, meta, 3)
        self.assertEqual(flash.cu.tolist(), [0, 1, 17])
        self.assertEqual(flash.used_q.tolist(), [1, 16])
        self.assertEqual(flash.token_live.shape, (17,))

    def test_mask_and_bucket_identity(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        causal = self.prepare(builder, common, meta, 2)
        self.assertEqual(causal.attn_mask.dtype, torch.int8)
        self.assertEqual(causal.attn_mask.shape, (2048, 2048))
        self.assertEqual(int(causal.attn_mask[0, 1]), 1)
        self.assertEqual(int(causal.attn_mask[1, 0]), 0)
        common.causal = False
        unmasked = self.prepare(builder, common, meta, 2)
        self.assertIsNot(causal, unmasked)
        self.assertIsNone(unmasked.attn_mask)
        self.assertEqual(unmasked.mask_mode, 0)
        common.causal = True
        self.assertIs(self.prepare(builder, common, meta, 2), causal)

    def test_meta_capacity_mismatch_fails(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        self.bad_capacity = True
        with self.assertRaisesRegex(ValueError, "Meta capacity"):
            self.prepare(builder, common, meta, 2)

    def test_capture_refresh_and_consumer_stream_guards(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = self.prepare(builder, common, meta, 2)
        self.stream = FakeStream()
        _, cache = self.cache()
        with self.assertRaisesRegex(RuntimeError, "stream that waits"):
            MODULE.flash_mla_decode(None, None, cache, flash, 1.0)
        self.prepare(builder, common, meta, 2)
        self.assertIs(flash.execution_stream, self.stream)
        self.assertIsNot(self.executor.stream, self.stream)
        self.npu.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "outside graph capture"):
            self.prepare(builder, common, meta, 2)

    def test_invalid_cache_never_falls_back(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = self.prepare(builder, common, meta, 2)
        _, cache = self.cache()
        q = torch.ones(2, 8, 512, dtype=cache.dtype)
        pe = torch.ones(2, 8, 64, dtype=cache.dtype)
        for bad in (
            (cache[..., :512], cache[..., 512:]),
            cache.squeeze(2),
            cache[..., :512],
            cache[:1].expand(3, -1, -1, -1),
        ):
            with self.subTest(shape=getattr(bad, "shape", "tuple")), self.assertRaises(ValueError):
                MODULE.flash_mla_decode(q, pe, bad, flash, 1.0)
        self.assertEqual(self.main_calls, [])

    def test_real_forward_keeps_prefill_gate_projection_and_raw_cache(self):
        ns = {
            "torch": torch,
            "_EXTRA_CTX": types.SimpleNamespace(num_tokens=3),
            "flash_mla_decode": Mock(return_value=torch.ones(8, 1, 512)),
            "get_current_hardware_profile": Mock(),
            "HardwareCapability": types.SimpleNamespace(MLA_DECODE_PROLOG_WITHOUT_ROPE=0),
            "MLAPO_MAX_SUPPORTED_TOKENS": 128,
            "maybe_save_kv_layer_to_connector": Mock(),
        }
        forward = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "forward", ns)
        _, cache = self.cache()
        flash = types.SimpleNamespace(token_live=torch.ones(1, dtype=torch.bool))
        meta = types.SimpleNamespace(
            num_actual_tokens=3, num_decode_tokens=1, num_decodes=1, num_prefills=1, flash_mla=flash
        )
        decode = types.SimpleNamespace(ql_nope="q_nope", q_pe="q_pe")
        prefill = types.SimpleNamespace(q_nope=0, q_pe=0, k_nope=0, k_pe=0, value=0)
        impl = types.SimpleNamespace(
            get_num_actual_tokens=lambda m: m.num_actual_tokens,
            kv_lora_rank=512,
            num_heads=8,
            v_head_dim=2,
            use_flash_mla=True,
            use_output_gate=True,
            use_mla_rope=True,
            fa_quant_layer=False,
            enable_mlapo=False,
            scale=0.125,
            _mla_preprocess=Mock(return_value=(decode, prefill)),
            _v_up_proj=Mock(return_value=torch.full((1, 16), 2.0)),
            _forward_prefill=Mock(return_value=torch.full((2, 16), 4.0)),
            g_proj=lambda x: (torch.zeros(3, 16),),
            o_proj=Mock(side_effect=lambda x, **kw: (x.clone(),)),
            _forward_decode=Mock(side_effect=AssertionError("legacy decode called")),
        )
        output = torch.empty(3, 16)
        self.assertIs(forward(impl, "layer", torch.ones(3, 4), cache, meta, output), output)
        self.assertIs(ns["flash_mla_decode"].call_args.args[2], cache)
        self.assertTrue(torch.equal(output[0], torch.ones(16)))
        self.assertTrue(torch.equal(output[1:], torch.full((2, 16), 2.0)))
        impl.o_proj.assert_called_once()
        self.assertTrue(impl.o_proj.call_args.kwargs["is_prefill"])
        logical = impl._mla_preprocess.call_args.args[2]
        self.assertEqual(logical[0].untyped_storage().data_ptr(), cache.untyped_storage().data_ptr())
        self.assertEqual(logical[1].storage_offset(), cache.storage_offset() + 512)

    def test_config_scope_guards(self):
        config = types.SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=None,
            parallel_config=types.SimpleNamespace(
                decode_context_parallel_size=1, prefill_context_parallel_size=1, enable_dbo=False
            ),
        )
        impl = types.SimpleNamespace(
            vllm_config=config,
            is_draft_model=False,
            num_heads=12,
            num_kv_heads=1,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            fa_quant_layer=False,
            enable_kv_nz=False,
            dtype=torch.bfloat16,
        )
        MODULE.validate_flash_mla_config(impl)
        cases = [
            (config, "use_v2_model_runner", False),
            (config, "speculative_config", object()),
            (config.parallel_config, "decode_context_parallel_size", 2),
            (config.parallel_config, "prefill_context_parallel_size", 2),
            (config.parallel_config, "enable_dbo", True),
            (impl, "is_draft_model", True),
            (impl, "num_heads", 24),
            (impl, "num_kv_heads", 2),
            (impl, "kv_lora_rank", 256),
            (impl, "qk_rope_head_dim", 0),
            (impl, "fa_quant_layer", True),
            (impl, "enable_kv_nz", True),
            (impl, "dtype", torch.float32),
        ]
        for obj, field, value in cases:
            with self.subTest(field=field), patch.object(obj, field, value), self.assertRaises(ValueError):
                MODULE.validate_flash_mla_config(impl)

    def test_batch_and_short_query_metadata_matrix(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        buckets = {}
        for batch in (1, 29, 30, 31, 32):
            for query_len in (1, 2, 16):
                with self.subTest(batch=batch, query_len=query_len):
                    common, meta = self.inputs(batch=batch, tokens=batch * query_len, live=batch * query_len)
                    common.query_start_loc *= query_len
                    flash = self.prepare(builder, common, meta, batch)
                    self.assertEqual(flash.used_q.tolist(), [query_len] * batch)
                    buckets[batch, query_len] = flash
        common, meta = self.inputs(batch=1, tokens=1, live=1)
        self.assertIs(self.prepare(builder, common, meta, 1), buckets[1, 1])

    def test_real_graph_capture_and_replay_binding_guards(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = self.prepare(builder, common, meta, 2)
        metadata = {"layer": types.SimpleNamespace(flash_mla=flash)}
        context = types.SimpleNamespace(attn_metadata=metadata, cudagraph_runtime_mode="FULL")
        extra = types.SimpleNamespace(capturing=False)
        ns = {
            "torch": torch,
            "get_forward_context": lambda: context,
            "_EXTRA_CTX": extra,
            "CUDAGraphMode": types.SimpleNamespace(PIECEWISE="PIECEWISE"),
            "set_current_vllm_config": lambda _: nullcontext(),
            "_get_graph_update_backend": lambda _: "mla",
            "use_updatable_graph": lambda _: False,
        }
        capture = load_method("vllm_ascend/worker/v2/aclgraph_utils.py", "ModelWithContext", "forward", ns)
        replay = load_method("vllm_ascend/worker/v2/aclgraph_utils.py", "ModelAclGraphManager", "run_fullgraph", ns)
        registry = {}
        wrapper = types.SimpleNamespace(
            is_draft_model=False,
            is_draft_model_prefill=False,
            flash_mla_graph_inputs=registry,
            original_model=Mock(return_value="capture"),
        )
        self.npu.is_current_stream_capturing.return_value = True
        self.assertEqual(capture(wrapper), "capture")
        self.assertIs(registry[2, 2]["layer"][0], flash)
        self.npu.is_current_stream_capturing.return_value = False
        runner = types.SimpleNamespace(attn_groups=[], model_state=types.SimpleNamespace(attn_metadata=metadata))
        manager = types.SimpleNamespace(
            update_stream=object(),
            vllm_config=object(),
            model_runner=runner,
            flash_mla_graph_inputs=registry,
            _graph_relay=Mock(return_value="replay"),
        )
        desc = types.SimpleNamespace(num_tokens=2, num_reqs=2)
        common.seq_lens += 17
        self.assertIs(self.prepare(builder, common, meta, 2), flash)
        self.assertEqual(replay(manager, desc), "replay")
        metadata["layer"].flash_mla = types.SimpleNamespace(execution_stream=self.stream)
        with self.assertRaisesRegex(RuntimeError, "not replace"):
            replay(manager, desc)
        metadata["layer"].flash_mla = None
        with self.assertRaisesRegex(RuntimeError, "captured attention groups"):
            replay(manager, desc)
        metadata["layer"].flash_mla = flash
        original_lens = flash.cache_lens
        flash.cache_lens = original_lens.clone()
        with self.assertRaisesRegex(RuntimeError, "not replace"):
            replay(manager, desc)
        flash.cache_lens = original_lens
        self.stream = FakeStream()
        with self.assertRaisesRegex(RuntimeError, "stream that waits"):
            replay(manager, desc)

    def test_graph_update_skips_only_flash_target_entries(self):
        # A FULL FlashMLA graph contains no FIA task handles/events. Its update
        # method must return before touching those lists; draft branch is intact.
        ns = {
            "torch": torch,
            "_EXTRA_CTX": types.SimpleNamespace(is_draft_model=False),
            "get_graph_params": lambda: object(),
        }
        update = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "update_graph_params", ns)
        metadata = {"layer": types.SimpleNamespace(decode=object(), flash_mla=object())}
        update(None, types.SimpleNamespace(attn_metadata=metadata), 2)
        event = Mock()
        params = types.SimpleNamespace(
            attn_params={2: [tuple([None] * 18)]}, handles={2: ["fia-handle"]}, events={2: [event]}, workspaces={}
        )
        ns["get_graph_params"] = lambda: params
        ns["torch_npu"] = types.SimpleNamespace(npu_fused_infer_attention_score_v2=types.SimpleNamespace(out=Mock()))
        self.npu.graph_task_update_begin = Mock()
        self.npu.graph_task_update_end = Mock()
        metadata["legacy"] = types.SimpleNamespace(
            flash_mla=None, decode=types.SimpleNamespace(seq_lens_list=[7, 8], actual_seq_lengths_q=[1, 2])
        )
        update(self.executor.stream, types.SimpleNamespace(attn_metadata=metadata), 2)
        self.assertEqual(
            ns["torch_npu"].npu_fused_infer_attention_score_v2.out.call_args.kwargs["actual_seq_kvlen"], [7, 8]
        )
        self.npu.graph_task_update_begin.assert_called_once_with(self.executor.stream, "fia-handle")

    def test_bad_metadata_dtype_is_rejected(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        common.block_table_tensor = common.block_table_tensor.long()
        with self.assertRaisesRegex(ValueError, "int32"):
            self.prepare(builder, common, meta, 2)

    def test_executor_reuse_fence_and_partial_failure(self):
        def task():
            self.calls.append((self.stream.name, "task"))

        tasks = [EXECUTOR.DeviceMetadataTask(task, 123)]
        self.executor.submit(tasks)
        with self.assertRaisesRegex(RuntimeError, "not been released"):
            self.executor.submit(tasks)
        self.executor.wait(123)
        self.calls.append((self.stream.name, "consumer_queued"))
        self.executor.release()
        self.executor.submit(tasks)
        self.executor.wait(123)
        self.assertEqual(
            self.calls,
            [
                ("consumer", "record", "event0"),
                ("metadata", "wait", "event0"),
                ("metadata", "task"),
                ("metadata", "record", "event2"),
                ("consumer", "wait", "event2"),
                ("consumer", "consumer_queued"),
                ("consumer", "record", "event1"),
                ("consumer", "record", "event0"),
                ("metadata", "wait", "event0"),
                ("metadata", "wait", "event1"),
                ("metadata", "task"),
                ("metadata", "record", "event2"),
                ("consumer", "wait", "event2"),
            ],
        )
        failing = EXECUTOR.DeviceMetadataExecutor()

        def fail():
            self.calls.append((self.stream.name, "partial_task"))
            raise ValueError("partial submission")

        with self.assertRaisesRegex(ValueError, "partial submission"), EXECUTOR.device_metadata_context(failing):
            failing.submit([EXECUTOR.DeviceMetadataTask(fail, 1)])
        self.assertFalse(failing.submission_in_flight)
        self.assertIs(EXECUTOR.get_device_metadata_executor(), self.executor)
        self.assertEqual(self.calls[-2][1:], ("wait_stream", "metadata"))
        with EXECUTOR.device_metadata_context(failing):
            failing.submit(tasks)
            failing.wait(123)
        self.assertFalse(failing.submission_in_flight)

    def test_real_runner_padding_and_metadata_entry_across_batches(self):
        split = load_method(
            "vllm_ascend/attention/utils.py",
            None,
            "split_decodes_and_prefills",
            {
                "torch": torch,
                "is_pd_decode_recompute_scheduler_enabled": lambda: False,
            },
        )
        pad = load_method(
            "vllm_ascend/worker/v2/model_runner.py",
            "NPUModelRunner",
            "_pad_query_start_loc_for_fia",
            {
                "np": np,
                "CUDAGraphMode": types.SimpleNamespace(FULL="FULL"),
            },
        )
        config = types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(
                static_forward_context={"layer": types.SimpleNamespace(impl=types.SimpleNamespace(num_heads=8))}
            )
        )

        class Builder:
            vllm_config = config

            def build(self, common_prefix_len, common_attn_metadata):
                self.common = common_attn_metadata
                common_attn_metadata.context_parallel_metadata = None
                d, p, dt, pt = split(common_attn_metadata, decode_threshold=1)
                return types.SimpleNamespace(num_decodes=d, num_prefills=p, num_decode_tokens=dt, num_prefill_tokens=pt)

            def build_for_cudagraph_capture(self, common):
                return self.build(0, common)

        builder = Builder()
        other = type("OtherBackend", (), {})
        ns = {
            "torch": torch,
            "np": np,
            "envs": types.SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            "get_device_metadata_executor": EXECUTOR.get_device_metadata_executor,
            "AscendMLAMetadataBuilder": Builder,
            "FlashMLABuilder": MODULE.FlashMLABuilder,
            "AscendCommonAttentionMetadata": lambda **kw: types.SimpleNamespace(**kw),
            "AscendDSAMetadataBuilder": other,
            "AscendSFAMetadataBuilder": other,
            "GDNAttentionMetadataBuilder": other,
        }
        entry = load_method("vllm_ascend/worker/v2/attn_utils.py", None, "build_attn_metadata", ns)
        runner = types.SimpleNamespace(
            decode_query_len=1, compilation_config=types.SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY")
        )
        group = types.SimpleNamespace(layer_names=["layer"], get_metadata_builder=lambda _: builder)
        buckets = {}
        for capacity, live in (
            (32, 32),
            (32, 29),
            (32, 31),
            (32, 30),
            (32, 32),
            (8, 8),
            (8, 5),
            (8, 7),
            (8, 8),
            (32, 29),
        ):
            cu = np.full(34, live, dtype=np.int32)
            cu[: live + 1] = np.arange(live + 1)
            cu, batch = pad(runner, capacity, capacity, live, cu, "FULL", capacity)
            cu = torch.from_numpy(cu[: batch + 1])
            table = torch.arange(capacity * 3, dtype=torch.int32).reshape(capacity, 3).flip(0)
            lengths = torch.arange(127, 127 + capacity, dtype=torch.int32)
            result = entry(
                attn_groups=[[group]],
                num_reqs=batch,
                num_actual_reqs=live,
                num_tokens=capacity,
                num_actual_tokens=live,
                num_input_tokens=capacity,
                query_start_loc_gpu=cu,
                query_start_loc_cpu=cu,
                max_query_len=1,
                seq_lens=lengths,
                # Deliberately different: FlashMLA must read the device input,
                # while the existing builder must retain its CPU length mirror.
                seq_lens_np=np.full(capacity, 999, dtype=np.int32),
                max_seq_len=384,
                block_tables=[table],
                slot_mappings=torch.full((1, capacity), -1, dtype=torch.int64),
                positions=torch.arange(capacity),
                kv_cache_config=types.SimpleNamespace(kv_cache_groups=[object()]),
            )["layer"]
            flash = result.flash_mla
            if capacity in buckets:
                self.assertIs(flash, buckets[capacity][0])
                self.assertEqual(flash.binding(), buckets[capacity][1])
            buckets[capacity] = flash, flash.binding()
            self.assertEqual(flash.cu[-1].item(), capacity)
            self.assertEqual(flash.used_q.tolist(), [1] * live + [0] * (capacity - live))
            self.assertTrue(torch.equal(flash.block_table[:live], table[:live]))
            self.assertEqual(flash.block_table[live:].count_nonzero(), 0)
            self.assertEqual(flash.cache_lens[live:].count_nonzero(), 0)
            self.assertTrue(torch.equal(flash.cache_lens[:live], lengths[:live]))
            self.assertEqual(builder.common.seq_lens_cpu.tolist(), [999] * capacity)
            self.assertEqual(flash.token_live.sum(), live)
            self.calls.append((self.stream.name, "consumer_queued"))
        self.assertIn(("metadata", "schedule"), self.calls)
        self.assertNotIn(("consumer", "schedule"), self.calls)

    def test_mainline_short_prefill_classification_is_preserved(self):
        split = load_method(
            "vllm_ascend/attention/utils.py",
            None,
            "split_decodes_and_prefills",
            {
                "torch": torch,
                "is_pd_decode_recompute_scheduler_enabled": lambda: False,
            },
        )
        # Run the actual builder initialization statements that own the threshold.
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAMetadataBuilder")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        start = next(i for i, n in enumerate(init.body) if ast.unparse(n).startswith("self.decode_threshold ="))
        end = next(i for i, n in enumerate(init.body) if ast.unparse(n).startswith("self.reorder_batch_threshold ="))
        obj = types.SimpleNamespace(speculative_config=None)
        exec(
            compile(ast.Module(body=init.body[start : end + 1], type_ignores=[]), "threshold", "exec"),
            {"self": obj, "envs": types.SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True)},
        )
        common = types.SimpleNamespace(
            context_parallel_metadata=None,
            num_reqs=3,
            num_actual_tokens=11,
            max_query_len=8,
            query_start_loc_cpu=torch.tensor([0, 1, 3, 11]),
            is_prefilling=torch.tensor([False, True, True]),
        )
        self.assertEqual(split(common, decode_threshold=obj.decode_threshold), (1, 2, 1, 10))

    def test_runner_and_capture_context_release_on_exception(self):
        owned = EXECUTOR.DeviceMetadataExecutor()
        parent = Mock()

        def submit_then_fail(*args, **kwargs):
            self.assertIs(EXECUTOR.get_device_metadata_executor(), owned)
            owned.submit([EXECUTOR.DeviceMetadataTask(lambda: None, 7)])
            owned.wait(7)
            self.calls.append((self.stream.name, "consumer_queued"))
            raise ValueError("consumer failed")

        parent.execute_model = submit_then_fail
        runner_ns = {
            "super": lambda: parent,
            "device_metadata_context": EXECUTOR.device_metadata_context,
            "_start_profiling_chunk_timing": lambda *a: None,
            "has_kv_transfer_group": lambda: False,
        }
        execute = load_method("vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner", "execute_model", runner_ns)
        runner = types.SimpleNamespace(
            device_metadata_executor=owned,
            ascend_config=types.SimpleNamespace(scheduler_config=types.SimpleNamespace(profiling_chunk_config=None)),
            model_state=types.SimpleNamespace(),
        )
        with self.assertRaisesRegex(ValueError, "consumer failed"):
            execute(runner, object())
        self.assertFalse(owned.submission_in_flight)
        self.assertIs(EXECUTOR.get_device_metadata_executor(), self.executor)
        parent.capture = submit_then_fail
        capture_ns = {
            "super": lambda: parent,
            "device_metadata_context": EXECUTOR.device_metadata_context,
            "ModelWithContext": lambda model, **kw: model,
            "communicator_switch": nullcontext,
        }
        capture = load_method("vllm_ascend/worker/v2/aclgraph_utils.py", "ModelAclGraphManager", "capture", capture_ns)
        manager = types.SimpleNamespace(model_runner=runner, flash_mla_graph_inputs={})
        with self.assertRaisesRegex(ValueError, "consumer failed"):
            capture(manager, object(), object(), object(), None, object(), [], object())
        self.assertFalse(owned.submission_in_flight)
        self.assertIs(EXECUTOR.get_device_metadata_executor(), self.executor)

    def test_frozen_capacity_forward_reordered_pages_and_output_reuse(self):
        # Simulate a captured forward with fixed Python counts and pointers.
        # The writer/main kernels are CPU test doubles; no ACL replay is claimed.
        backing, cache = self.cache(token_gap=32)

        def scatter(*, key, value, key_cache, value_cache, slot_mapping):
            for row, slot in enumerate(slot_mapping.tolist()):
                if slot >= 0:
                    key_cache[slot // 128, slot % 128] = key[row]
                    value_cache[slot // 128, slot % 128] = value[row]

        writer = load_method(
            "vllm_ascend/attention/mla_v1.py",
            "AscendMLAImpl",
            "_exec_kv_no_rope",
            {
                "DeviceOperator": types.SimpleNamespace(reshape_and_cache=scatter),
            },
        )

        def read_current(q, received, **kw):
            self.assertIs(received, cache)
            self.assertTrue(torch.all(kw["metadata"] == int(kw["cache_seqlens"].sum())))
            result = torch.full((8, 4, 512), float("nan"), dtype=q.dtype)
            for row in range(4):
                if kw["seqused_q"][row] > 0:
                    pos = int(kw["cache_seqlens"][row]) - 1
                    page = int(kw["block_table"][row, pos // 128])
                    result[:, row] = received[page, pos % 128, 0, :512] + q[row, :, :512]
            return result, torch.empty(0)

        self.ops.flash_mla_with_kvcache = read_current
        ns = {
            "torch": torch,
            "_EXTRA_CTX": types.SimpleNamespace(num_tokens=4),
            "flash_mla_decode": MODULE.flash_mla_decode,
            "get_current_hardware_profile": Mock(),
            "HardwareCapability": types.SimpleNamespace(MLA_DECODE_PROLOG_WITHOUT_ROPE=0),
            "MLAPO_MAX_SUPPORTED_TOKENS": 128,
            "maybe_save_kv_layer_to_connector": Mock(),
        }
        forward = load_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "forward", ns)
        slots = torch.full((4,), -1, dtype=torch.int64)
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, runtime = self.inputs(batch=4, tokens=4, live=4)
        captured = types.SimpleNamespace(num_actual_tokens=4, num_decode_tokens=4, num_decodes=4, num_prefills=0)
        impl = types.SimpleNamespace(
            get_num_actual_tokens=lambda m: m.num_actual_tokens,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_kv_heads=1,
            num_heads=8,
            v_head_dim=2,
            kv_a_layernorm=lambda x: x,
            use_flash_mla=True,
            use_output_gate=True,
            use_mla_rope=True,
            fa_quant_layer=False,
            enable_mlapo=False,
            scale=0.125,
            _v_up_proj=lambda x: x[0, :, :1].expand(4, 16).clone(),
            o_proj=lambda x, **kw: (x.clone(),),
        )

        def preprocess(layer, hidden, logical, meta):
            writer(impl, hidden[:, :1].expand(4, 576).contiguous(), logical, slots)
            return types.SimpleNamespace(
                ql_nope=hidden[:, :1, None].expand(4, 8, 512), q_pe=hidden[:, :1, None].expand(4, 8, 64)
            ), None

        impl._mla_preprocess = preprocess
        output = torch.full((4, 16), float("nan"), dtype=cache.dtype)
        binding = None
        for step, ids in enumerate(([0, 1, 2, 3], [3, 1], [2, 0, 3], [1, 3, 0, 2])):
            live = len(ids)
            runtime.num_decode_tokens = live
            hidden = torch.full((4, 1), float("nan"), dtype=cache.dtype)
            slots.fill_(-1)
            common.seq_lens.fill_(999)  # Deliberately stale inactive metadata.
            common.block_table_tensor.fill_(2)
            touched = torch.zeros(backing.numel(), dtype=torch.bool)
            expected = []
            for row, req in enumerate(ids):
                length = 127 + step + req
                pages = [(req + step) % 3, (req + step + 1) % 3, (req + step + 2) % 3]
                common.seq_lens[row] = length
                common.block_table_tensor[row] = torch.tensor(pages)
                pos = length - 1
                page = pages[pos // 128]
                slots[row] = page * 128 + pos % 128
                value = 10 * step + req + 1
                hidden[row] = value
                expected.append(value)
                offset = cache.storage_offset() + page * cache.stride(0) + (pos % 128) * cache.stride(1)
                touched[offset : offset + 576] = True
            before = backing.clone()
            flash = self.prepare(builder, common, runtime, live)
            if binding is not None:
                self.assertEqual(flash.binding(), binding)
            binding = flash.binding()
            captured.flash_mla = flash
            impl.g_proj = lambda x, flash=flash: (
                torch.where(flash.token_live[:, None], 0.0, float("nan")).expand(4, 16).to(cache.dtype),
            )
            forward(impl, "layer", hidden, cache, captured, output)
            self.assertTrue(torch.equal(output[:live, 0], torch.tensor(expected, dtype=cache.dtype)))
            self.assertEqual(output[live:].count_nonzero(), 0)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(torch.equal(backing[~touched], before[~touched]))

    def test_dependency_cow_remains_visible_through_original_cache(self):
        path = ROOT / "vllm_ascend/worker/utils.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "copy_kv_cache_blocks_inplace")
        ns = {"torch": torch, "np": np, "async_tensor_h2d": lambda a, device: torch.from_numpy(a).to(device)}
        exec(compile(ast.unparse(fn), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag), ns)
        backing, cache = self.cache()
        before = backing.clone()
        cache[2].fill_(7)
        ns[fn.name]([cache], 3, [types.SimpleNamespace(src_block_id=2, dst_block_id=0)])
        self.assertTrue(torch.all(cache[0] == 7))
        self.assertTrue(torch.equal(cache[1], before.as_strided(cache.shape, cache.stride(), 23)[1]))
        self.assertTrue(torch.equal(backing[:23], before[:23]))
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = self.prepare(builder, common, meta, 2)
        MODULE.flash_mla_decode(
            torch.ones(2, 8, 512, dtype=cache.dtype), torch.ones(2, 8, 64, dtype=cache.dtype), cache, flash, 1.0
        )
        self.assertIs(self.main_calls[-1][1], cache)
        self.assertTrue(torch.all(self.main_calls[-1][1][0] == 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
