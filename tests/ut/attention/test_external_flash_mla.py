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
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "external_flash_mla_test_module", ROOT / "vllm_ascend/attention/flash_mla.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def load_method(path, class_name, method_name, namespace):
    """Execute the actual integration method without importing NPU packages."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(ast.unparse(module), str(path), "exec", flags=__import__("__future__").annotations.compiler_flag),
        namespace,
    )
    return namespace[method_name]


class FakeStream:
    def __init__(self):
        self.waits = []

    def wait_stream(self, other):
        self.waits.append(other)


class FlashMLATest(unittest.TestCase):
    def setUp(self):
        self.stream = FakeStream()
        self.npu = types.SimpleNamespace(
            current_stream=Mock(side_effect=lambda: self.stream), is_current_stream_capturing=Mock(return_value=False)
        )
        self.metadata_calls = []
        self.main_calls = []
        self.bad_capacity = False

        def metadata_op(lengths, heads, kv_heads, **kw):
            self.metadata_calls.append((lengths, heads, kv_heads, kw))
            size = 32 + lengths.numel() * 16
            if lengths.device.type == "meta":
                return torch.empty(size, dtype=torch.int32, device="meta")
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
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

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
                        flash = builder.build(common, meta, 2)
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
        first = builder.build(common, meta, 4)
        addresses = {k: v.data_ptr() for k, v in vars(first).items() if isinstance(v, torch.Tensor)}
        common.seq_lens = torch.tensor([100_000, 128_000, 999, 999], dtype=torch.int32)
        common.block_table_tensor[0] = torch.tensor([1, 2, 0])
        meta.num_decode_tokens = 2
        second = builder.build(common, meta, 2)
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
        flash = builder.build(common, meta, 3)
        self.assertEqual(flash.cu.tolist(), [0, 1, 17])
        self.assertEqual(flash.used_q.tolist(), [1, 16])
        self.assertEqual(flash.token_live.shape, (17,))

    def test_mask_and_bucket_identity(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        causal = builder.build(common, meta, 2)
        self.assertEqual(causal.attn_mask.dtype, torch.int8)
        self.assertEqual(causal.attn_mask.shape, (2048, 2048))
        self.assertEqual(int(causal.attn_mask[0, 1]), 1)
        self.assertEqual(int(causal.attn_mask[1, 0]), 0)
        common.causal = False
        unmasked = builder.build(common, meta, 2)
        self.assertIsNot(causal, unmasked)
        self.assertIsNone(unmasked.attn_mask)
        self.assertEqual(unmasked.mask_mode, 0)
        common.causal = True
        self.assertIs(builder.build(common, meta, 2), causal)

    def test_meta_capacity_mismatch_fails(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        self.bad_capacity = True
        with self.assertRaisesRegex(ValueError, "Meta capacity"):
            builder.build(common, meta, 2)

    def test_capture_refresh_and_consumer_stream_guards(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = builder.build(common, meta, 2)
        old_stream = self.stream
        self.stream = FakeStream()
        _, cache = self.cache()
        with self.assertRaisesRegex(RuntimeError, "preparation stream"):
            MODULE.flash_mla_decode(None, None, cache, flash, 1.0)
        builder.build(common, meta, 2)
        self.assertEqual(self.stream.waits, [old_stream])
        self.npu.is_current_stream_capturing.return_value = True
        with self.assertRaisesRegex(RuntimeError, "outside graph capture"):
            builder.build(common, meta, 2)

    def test_invalid_cache_never_falls_back(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = builder.build(common, meta, 2)
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
                    flash = builder.build(common, meta, batch)
                    self.assertEqual(flash.used_q.tolist(), [query_len] * batch)
                    buckets[batch, query_len] = flash
        common, meta = self.inputs(batch=1, tokens=1, live=1)
        self.assertIs(builder.build(common, meta, 1), buckets[1, 1])

    def test_real_graph_capture_and_replay_binding_guards(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        flash = builder.build(common, meta, 2)
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
        self.assertIs(registry[2, 2]["layer"], flash)
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
        self.assertIs(builder.build(common, meta, 2), flash)
        self.assertEqual(replay(manager, desc), "replay")
        metadata["layer"].flash_mla = types.SimpleNamespace(execution_stream=self.stream)
        with self.assertRaisesRegex(RuntimeError, "not replace"):
            replay(manager, desc)
        metadata["layer"].flash_mla = None
        with self.assertRaisesRegex(RuntimeError, "captured attention groups"):
            replay(manager, desc)
        metadata["layer"].flash_mla = flash
        self.stream = FakeStream()
        with self.assertRaisesRegex(RuntimeError, "preparation stream"):
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

    def test_bad_metadata_dtype_is_rejected(self):
        builder = MODULE.FlashMLABuilder(8, "cpu")
        common, meta = self.inputs()
        common.block_table_tensor = common.block_table_tensor.long()
        with self.assertRaisesRegex(ValueError, "int32"):
            builder.build(common, meta, 2)

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
        flash = builder.build(common, meta, 2)
        MODULE.flash_mla_decode(
            torch.ones(2, 8, 512, dtype=cache.dtype), torch.ones(2, 8, 64, dtype=cache.dtype), cache, flash, 1.0
        )
        self.assertIs(self.main_calls[-1][1], cache)
        self.assertTrue(torch.all(self.main_calls[-1][1][0] == 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
