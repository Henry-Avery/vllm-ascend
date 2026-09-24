# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly on CPU; mocked operators do not establish NPU correctness."""

import ast
import sys
import types
import unittest
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load_definitions(path, names=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    module = types.ModuleType("_flash_host_test")
    module.__dict__.update(namespace)
    with patch.dict(sys.modules, {module.__name__: module}):
        exec(compile(tree, str(ROOT / path), "exec"), module.__dict__)
    return module


def load_methods(path, class_name, methods, **namespace):
    """Execute selected methods unchanged; heavyweight imports stay stubbed."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    tree.body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls]
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return namespace[class_name]


class FlashMLAHostContract(unittest.TestCase):
    def setUp(self):
        self.calls = []
        package = types.ModuleType("cann_ops_transformer.ops")

        def metadata(lengths, heads, kv_heads, **kwargs):
            self.calls.append(("metadata", lengths, heads, kv_heads, kwargs))
            return torch.full((lengths.numel() * 8,), len(self.calls), dtype=torch.int32, device=lengths.device)

        def main(query, cache, **kwargs):
            self.calls.append(("main", query, cache, kwargs))
            return torch.zeros(query.shape[1], query.shape[0], 512, dtype=query.dtype), torch.empty(0)

        package.flash_mla_with_kvcache_metadata = metadata
        package.flash_mla_with_kvcache = main
        self.modules = patch.dict(sys.modules, {"cann_ops_transformer.ops": package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.device_api = load_definitions(
            "vllm_ascend/worker/device_metadata.py",
            {"DeviceMetadataStage", "DeviceMetadataTask", "DeviceMetadataExecutor"},
            torch=torch,
            dataclass=dataclass,
            IntEnum=IntEnum,
            Callable=Callable,
            Iterable=Iterable,
            BatchDescriptor=object,
        )
        self.api = load_definitions(
            "vllm_ascend/attention/flash_mla.py",
            torch=torch,
            dataclass=dataclass,
            MLA_FLASH_SUPPORTED_Q_HEADS={8, 12, 64, 96},
            FLASH_MLA_BLOCK_SIZE=128,
            FLASH_MLA_QK_DIM=576,
            FLASH_MLA_V_DIM=512,
            FLASH_MLA_MASK_SIZE=2048,
            DeviceMetadataStage=self.device_api.DeviceMetadataStage,
            DeviceMetadataTask=self.device_api.DeviceMetadataTask,
        )

    def builder(self, heads=8):
        builder = NS(device="cpu", decode_threshold=1)
        self.api.init_flash_mla_metadata(builder, NS(num_heads=heads, num_kv_heads=1))
        return builder

    @staticmethod
    def common(batch=1, query=2, padding=2, causal=True, kv=128000):
        tokens = batch * query
        return NS(
            num_reqs=batch,
            num_actual_tokens=tokens,
            num_input_tokens=tokens + padding,
            max_query_len=query,
            causal=causal,
            block_table_tensor=torch.arange(batch * 1000, dtype=torch.int32).view(batch, 1000),
            query_start_loc=torch.arange(0, tokens + 1, query, dtype=torch.int32),
            seq_lens=torch.full((batch,), kv, dtype=torch.int32),
            slot_mapping=torch.arange(tokens, dtype=torch.int64),
            positions=torch.arange(tokens),
        )

    @staticmethod
    def cache():
        backing = torch.full((4 * 128 * 2 * 576 + 37,), -9, dtype=torch.bfloat16)
        return backing, torch.as_strided(backing, (2, 128, 1, 576), (2 * 128 * 2 * 576, 2 * 576, 576, 1), 37)

    def test_head_batch_length_matrix(self):
        for heads in (8, 12, 64, 96):
            builder = self.builder(heads)
            for batch, query, kv in (
                (1, 1, 100000),
                (29, 2, 128000),
                (30, 4, 100000),
                (31, 8, 128000),
                (32, 16, 128000),
            ):
                with self.subTest(heads=heads, batch=batch, query=query, kv=kv):
                    b = self.api.build_flash_mla_metadata(builder, self.common(batch, query, kv=kv))
                    self.assertEqual(b.num_tokens, batch * query + 2)
                    self.assertEqual(b.cu[-1], b.num_tokens)
                    self.assertEqual(b.used_q[-1], 0)
                    self.assertTrue(torch.equal(b.cache_lens[:-1], torch.full((batch,), kv)))
                    self.assertTrue((b.slots[-2:] == -1).all())
                    call = self.calls[-1]
                    self.assertEqual(call[2:4], (heads, 1))
                    self.assertEqual((call[-1]["max_seqlen_q"], call[-1]["max_seqlen_kv"]), (-1, -1))

    def test_fresh_schedule_and_metadata_each_step(self):
        builder, common = self.builder(), self.common()
        first = self.api.build_flash_mla_metadata(builder, common)
        common.seq_lens.fill_(129)
        common.block_table_tensor.add_(7)
        second = self.api.build_flash_mla_metadata(builder, common)
        self.assertIsNot(first.schedule, second.schedule)
        self.assertEqual(first.cache_lens[0], 128000)
        self.assertEqual(second.cache_lens[0], 129)
        self.assertEqual(first.block_table[0, 0], 0)
        self.assertEqual(second.block_table[0, 0], 7)
        self.assertEqual(builder._flash_buffers, {})

    def test_main_consumes_exact_metadata_and_original_cache(self):
        for causal in (True, False):
            b = self.api.build_flash_mla_metadata(self.builder(), self.common(causal=causal))
            metadata_call = self.calls[-1]
            backing, cache = self.cache()
            before = backing.clone()
            q = torch.zeros(b.num_tokens, b.num_heads, 576, dtype=torch.bfloat16)
            output, lse = self.api.run_flash_mla(q, cache, b, 0.125)
            _, actual_q, actual_cache, args = self.calls[-1]
            self.assertIs(actual_q, q)
            self.assertIs(actual_cache, cache)
            self.assertEqual(cache.storage_offset(), 37)
            self.assertTrue(torch.equal(before, backing))
            self.assertEqual(output.shape, (8, 4, 512))
            self.assertEqual(lse.numel(), 0)
            self.assertEqual((args["layout_q"], args["layout_kv"], args["layout_out"]), ("TND", "PA_BBND", "NTD"))
            self.assertEqual((args["max_seqlen_q"], args["max_seqlen_kv"]), (-1, -1))
            self.assertIs(args["metadata"], b.schedule)
            self.assertIs(args["cache_seqlens"], metadata_call[1])
            for name in ("cu_seqlens_q", "seqused_q"):
                self.assertIs(args[name], metadata_call[-1][name])
            self.assertEqual(args["mask_mode"], metadata_call[-1]["mask_mode"])
            self.assertFalse(args["return_softmax_lse"])
            if causal:
                self.assertEqual(b.attn_mask.shape, (2048, 2048))
                self.assertEqual(b.attn_mask[0, 1], 1)
                self.assertEqual(b.attn_mask[1, 0], 0)
                self.assertEqual(b.attn_mask[0, 0], 0)
            else:
                self.assertIsNone(args["attn_mask"])

    def test_inactive_request_padding_and_variable_queries(self):
        common = self.common(batch=3, query=2)
        common.query_start_loc = torch.tensor([0, 1, 4, 6], dtype=torch.int32)
        common.seq_lens[1] = 0
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertEqual(b.used_q.tolist(), [1, 0, 2, 0])
        self.assertEqual(b.slots.tolist(), [0, -1, -1, -1, 4, 5, -1, -1])
        self.assertEqual(b.token_live.tolist(), [True, False, False, False, True, True, False, False])

    def test_uses_device_lengths_not_cpu_upper_bounds(self):
        common = self.common()
        common.seq_lens_cpu = torch.tensor([17])
        common.seq_lens_cpu_upper_bound = torch.tensor([128007])
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertEqual(b.cache_lens[0], 128000)

    def test_reject_invalid_cache_heads_or_query(self):
        for cache in (torch.empty(2, 1, 128, 576), torch.empty(2, 16, 1, 576), torch.empty(2, 128, 1, 576)):
            with self.assertRaises(ValueError):
                self.api.validate_flash_cache(cache)
        with self.assertRaises(ValueError):
            self.builder(heads=24)
        b = self.api.build_flash_mla_metadata(self.builder(), self.common())
        _, cache = self.cache()
        with self.assertRaises(ValueError):
            self.api.run_flash_mla(torch.zeros(4, 12, 576, dtype=torch.bfloat16), cache, b, 1)

    def test_no_external_package_fails_without_fallback(self):
        with patch.dict(sys.modules, {"cann_ops_transformer.ops": None}), self.assertRaises(ModuleNotFoundError):
            self.api.build_flash_mla_metadata(self.builder(), self.common())

    def test_builder_routes_before_legacy_cpu_length_handling(self):
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAMetadataBuilder")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
        method.body = [method.body[0]]  # Execute the real opt-in builder branch.
        method.args.args = [ast.arg(arg=a.arg) for a in method.args.args]
        method.returns = None
        scope = dict(
            envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            split_decodes_and_prefills=lambda *a, **kw: (1, 0, 1, 0),
            build_flash_mla_metadata=self.api.build_flash_mla_metadata,
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "builder", "exec"), scope)
        builder = self.builder()
        builder.metadata_cls = NS
        common = self.common(query=1)
        common.attn_state = "decode"
        result = scope["build"](builder, 0, common)
        self.assertEqual(result.num_decode_tokens, 1)
        self.assertIsNone(result.seq_lens_cpu)
        self.assertIs(result.seq_lens, common.seq_lens)
        self.assertEqual(result.flash.num_tokens, 3)

    def test_empty_padding_only_metadata_and_overlapping_cache(self):
        common = self.common()
        common.seq_lens.zero_()
        b = self.api.build_flash_mla_metadata(self.builder(), common)
        self.assertFalse(b.token_live.any())
        self.assertTrue((b.slots == -1).all())
        backing, cache = self.cache()
        overlap = torch.as_strided(backing, cache.shape, (576, 576, 576, 1))
        with self.assertRaises(ValueError):
            self.api.validate_flash_cache(overlap)

    def test_forward_writer_rope_nope_and_output(self):
        tree = ast.parse((ROOT / "vllm_ascend/attention/mla_v1.py").read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMLAImpl")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_forward_flash")
        for use_rope in (False, True):
            for gate in (False, True):
                common = self.common()
                common.slot_mapping = torch.tensor([127, 128])
                b = self.api.build_flash_mla_metadata(self.builder(), common)
                backing, cache = self.cache()
                expected = backing.clone()
                expected_cache = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
                data = torch.arange(4 * 576).to(torch.bfloat16).view(4, 576) / 1024
                expected_cache[0, 127, 0] = data[0]
                expected_cache[1, 0, 0] = data[1]
                if use_rope:
                    expected_cache[0, 127, 0, 512:] += 1
                    expected_cache[1, 0, 0, 512:] += 1
                events = []

                def scatter(*, events=events, cache=cache, **kwargs):
                    events.append("write")
                    self.assertEqual(kwargs["key_cache"].stride(), cache.stride())
                    for index, slot in enumerate(kwargs["slot_mapping"].tolist()):
                        if slot >= 0:
                            kwargs["key_cache"][slot // 128, slot % 128] = kwargs["key"][index]
                            kwargs["value_cache"][slot // 128, slot % 128] = kwargs["value"][index]

                def run(q, kv, flash, scale, events=events, backing=backing, expected=expected, cache=cache):
                    events.append("attention")
                    self.assertEqual(events, ["write", "notify", "attention"])
                    self.assertTrue(torch.equal(backing, expected))
                    self.assertIs(kv, cache)
                    return torch.ones(8, 4, 512, dtype=torch.bfloat16), None

                def project(latent):
                    self.assertEqual(latent.shape, (8, 4, 512))
                    projected = torch.ones(4, 8, dtype=torch.bfloat16)
                    projected[2:] = float("nan")
                    return projected

                scope = dict(
                    torch=torch,
                    torch_npu=NS(npu_scatter_pa_kv_cache=scatter),
                    validate_flash_cache=self.api.validate_flash_cache,
                    run_flash_mla=run,
                    get_cos_and_sin_mla=lambda *a, **k: (None, None),
                    notify_kv_cache_written=lambda name, events=events: events.append("notify"),
                )
                exec(compile(ast.Module(body=[method], type_ignores=[]), "forward", "exec"), scope)
                impl = NS(
                    layerwise_kv_cache_hook=None,
                    fused_qkv_a_proj=None,
                    kv_a_proj_with_mqa=lambda x, data=data: (data, None),
                    kv_a_layernorm=lambda x: x,
                    _q_proj_and_k_up_proj=lambda x: (torch.zeros(4, 8, 512), torch.zeros(4, 8, 64)),
                    use_mla_rope=use_rope,
                    rope_single=lambda x, cos, sin: x + 1,
                    scale=1,
                    _v_up_proj=project,
                    use_output_gate=gate,
                    g_proj=lambda x: (torch.zeros(4, 8), None),
                    o_proj=lambda x, **kwargs: (x + 2, None),
                )
                output = torch.full((6, 8), -99, dtype=torch.bfloat16)
                scope["_forward_flash"](impl, "layer", data, cache, NS(flash=b), output)
                torch.testing.assert_close(output[:2], torch.full((2, 8), 2.5 if gate else 3, dtype=torch.bfloat16))
                self.assertTrue((output[2:] == 0).all())

    def test_scope_guards_and_disabled_default(self):
        hardware = types.ModuleType("vllm_ascend.device.device_config")
        hardware.is_950 = lambda: True
        enabled = NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        api = load_definitions(
            "vllm_ascend/platform.py",
            {"_validate_flash_mla_config"},
            VllmConfig=object,
            torch=torch,
            envs=enabled,
            model_uses_sfa_sparse=lambda model: False,
            KVPPConfig=NS(from_vllm_config=lambda cfg: NS(size=1)),
        )
        cfg = NS(
            model_config=NS(enforce_eager=True, use_mla=True, dtype=torch.bfloat16),
            use_v2_model_runner=True,
            cache_config=NS(cache_dtype="auto"),
            parallel_config=NS(
                prefill_context_parallel_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1
            ),
            speculative_config=None,
            kv_transfer_config=None,
        )
        with patch.dict(sys.modules, {hardware.__name__: hardware}):
            # Graph and eager target-only configurations must not need a draft.
            for enforce_eager in (True, False):
                cfg.model_config.enforce_eager = enforce_eager
                api._validate_flash_mla_config(cfg)
            for obj, field, value in (
                (cfg, "speculative_config", NS(method="dspark")),
                (cfg.parallel_config, "decode_context_parallel_size", 2),
                (cfg.parallel_config, "prefill_context_parallel_size", 2),
                (cfg.cache_config, "cache_dtype", "fp8"),
                (cfg, "use_v2_model_runner", False),
            ):
                previous = getattr(obj, field)
                setattr(obj, field, value)
                with self.assertRaises(ValueError):
                    api._validate_flash_mla_config(cfg)
                setattr(obj, field, previous)
            enabled.VLLM_ASCEND_ENABLE_FLASH_MLA = False
            api._validate_flash_mla_config(None)

    def dspark_speculator(self):
        module = NS(build_attn_metadata=object())
        builder = self.builder(heads=12)

        def build(**kwargs):
            spec.last_factory = kwargs
            if spec.fail:
                raise RuntimeError("metadata failed")
            if not enabled.VLLM_ASCEND_ENABLE_FLASH_MLA:
                query = NS(actual_seq_lengths_q=[])
                return {"draft": NS(decode=query) if spec.attn_architecture == "MLA" else query}
            common = self.common(batch=kwargs["num_reqs"], query=spec.num_query_per_req, padding=0)
            common.num_input_tokens = common.num_actual_tokens = kwargs["num_tokens"]
            common.query_start_loc = spec.input_buffers.query_start_loc[: kwargs["num_reqs"] + 1]
            common.seq_lens = spec.input_buffers.seq_lens[: kwargs["num_reqs"]]
            common.block_table_tensor = spec.table[: kwargs["num_reqs"]]
            common.slot_mapping = spec.slots[: kwargs["num_tokens"]]
            common.positions = kwargs["positions"]
            common.causal = kwargs["causal"]
            return {"draft": NS(flash=self.api.build_flash_mla_metadata(builder, common), decode=None)}

        wrappers = load_definitions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"build_attn_metadata_wrapper", "build_draft_attn_metadata_factory"},
            contextmanager=contextmanager,
            _BUILD_ATTN_METADATA_MODULE=module,
            build_attn_metadata=build,
        )

        class Parent:
            def _build_uniform_attn_metadata(self, **kwargs):
                # Current vLLM calls the general hook from its uniform hook.
                return self._build_attn_metadata(**kwargs)

            def _build_attn_metadata(self, **kwargs):
                self.parent_kwargs = kwargs
                desc = kwargs["batch_desc"]
                tokens = desc.num_tokens if desc.cg_mode == "FULL" else kwargs["num_reqs"] * self.num_query_per_req
                return module.build_attn_metadata(
                    num_reqs=desc.num_reqs or kwargs["num_reqs"],
                    num_tokens=tokens,
                    causal=kwargs.get("causal", False),
                )

            def propose(self, *args, **kwargs):
                self.parent_propose_args = args
                return self._build_uniform_attn_metadata(
                    num_reqs=1,
                    batch_desc=NS(cg_mode="FULL", num_reqs=1, num_tokens=7),
                    num_query_per_req=3,
                    seq_lens_cpu_upper_bound=None,
                    step=3,
                    causal=False,
                )

        enabled = NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        cls = load_methods(
            "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
            "AscendDSparkSpeculator",
            {
                "_draft_query_attn_state",
                "_prepare_draft_dcp_metadata_inputs",
                "_build_draft_attn_metadata",
                "_build_uniform_attn_metadata",
                "_update_draft_attn_metadata",
                "propose",
            },
            DSparkSpeculator=Parent,
            BatchExecutionDescriptor=NS,
            CUDAGraphMode=NS(FULL="FULL"),
            torch=torch,
            ascend_envs=enabled,
            AscendAttentionState=NS(SpecDecoding="spec", ChunkedPrefill="prefill"),
            build_attn_metadata_wrapper=wrappers.build_attn_metadata_wrapper,
            build_draft_attn_metadata_factory=wrappers.build_draft_attn_metadata_factory,
            device_metadata_context=lambda _: nullcontext(),
        )
        spec = cls()
        spec.attn_architecture = "MLA"
        spec.use_dcp = False
        spec.num_query_per_req = 3
        spec.max_num_reqs = 4
        spec.max_num_tokens = 12
        spec.attn_vllm_config = NS(parallel_config=NS(decode_context_parallel_size=1))
        spec.input_buffers = NS(
            positions=torch.arange(12) + 126,
            query_start_loc=torch.tensor([0, 3, 3, 3, 3], dtype=torch.int32),
            seq_lens=torch.tensor([129, 0, 0, 0], dtype=torch.int32),
        )
        spec.table = torch.tensor([[1, 0], [0, 0], [0, 0], [0, 0]], dtype=torch.int32)
        spec.slots = torch.tensor([254, 255, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        spec.fail = False
        spec.device_metadata_executor = None
        return spec, enabled, module, wrappers

    def test_dspark_query_device_lengths_padding_and_refresh(self):
        spec, _, _, _ = self.dspark_speculator()
        for causal in (False, True):
            first = spec._build_draft_attn_metadata(
                num_reqs=1, num_reqs_padded=1, num_tokens_padded=7, seq_lens_cpu_upper_bound=None, step=3, causal=causal
            )["draft"].flash
            # Seven physical tokens need not be a whole number of draft groups.
            self.assertEqual(spec.parent_kwargs["batch_desc"].num_reqs, 1)
            self.assertEqual(first.cu.tolist(), [0, 3, 7])
            self.assertEqual(first.used_q.tolist(), [3, 0])
            self.assertEqual(first.cache_lens.tolist(), [129, 0])
            self.assertEqual(first.slots.tolist(), [254, 255, 0, -1, -1, -1, -1])
            self.assertEqual(spec.last_factory["attn_state"], "spec")
            self.assertEqual(spec.last_factory["is_prefilling"].tolist(), [False])
            self.assertTrue(torch.equal(first.positions, spec.input_buffers.positions[:7]))
            self.assertEqual(first.num_heads, 12)
            spec.input_buffers.seq_lens[0] = 130
            spec.table[0] = torch.tensor([0, 1])
            second = spec._build_draft_attn_metadata(
                num_reqs=1, num_reqs_padded=2, num_tokens_padded=7, seq_lens_cpu_upper_bound=None, step=3, causal=causal
            )["draft"].flash
            self.assertIsNot(first.schedule, second.schedule)
            self.assertEqual(first.cache_lens.tolist(), [129, 0])
            self.assertEqual(second.cache_lens.tolist(), [130, 0, 0])
            self.assertEqual(second.used_q.tolist(), [3, 0, 0])
            self.assertEqual(first.block_table[0].tolist(), [1, 0])
            self.assertEqual(second.block_table[0].tolist(), [0, 1])
            _, cache = self.cache()
            self.api.run_flash_mla(torch.zeros(7, 12, 576, dtype=torch.bfloat16), cache, second, 0.125)
            self.assertIs(self.calls[-1][-1]["metadata"], second.schedule)
            self.assertEqual(self.calls[-1][-1]["mask_mode"], 3 if causal else 0)
            if not causal:
                self.assertIsNone(self.calls[-1][-1]["attn_mask"])
            spec.input_buffers.seq_lens[0] = 129
            spec.table[0] = torch.tensor([1, 0])

    def test_dspark_keeps_fia_padding_and_restores_factory_on_failure(self):
        spec, enabled, module, wrappers = self.dspark_speculator()
        original = module.build_attn_metadata
        for architecture in ("MLA", "GQA"):
            enabled.VLLM_ASCEND_ENABLE_FLASH_MLA = False
            spec.attn_architecture = architecture
            result = spec._build_draft_attn_metadata(
                num_reqs=1, num_reqs_padded=2, num_tokens_padded=6, seq_lens_cpu_upper_bound=None, step=3
            )["draft"]
            query = result.decode if architecture == "MLA" else result
            self.assertEqual(query.actual_seq_lengths_q, [3, 6])
            self.assertEqual(spec.parent_kwargs["batch_desc"].num_reqs, 2)
            self.assertEqual(spec.last_factory["attn_state"], "prefill")
        enabled.VLLM_ASCEND_ENABLE_FLASH_MLA = True
        spec.attn_architecture = "MLA"
        spec.fail = True
        with wrappers.build_attn_metadata_wrapper():
            outer = module.build_attn_metadata
            with self.assertRaisesRegex(RuntimeError, "metadata failed"):
                spec._build_draft_attn_metadata(
                    num_reqs=1, num_reqs_padded=1, num_tokens_padded=7, seq_lens_cpu_upper_bound=None, step=3
                )
            self.assertIs(module.build_attn_metadata, outer)
        self.assertIs(module.build_attn_metadata, original)

    def test_dspark_current_uniform_hook_eager_vs_full(self):
        spec, _, _, wrappers = self.dspark_speculator()
        for mode, expected_tokens in (("NONE", 3), ("PIECEWISE", 3), ("FULL", 7)):
            spec._flash_query_metadata = None
            with (
                wrappers.build_attn_metadata_wrapper(),
                wrappers.build_draft_attn_metadata_factory(
                    spec.input_buffers.positions,
                    spec.max_num_tokens,
                    torch.zeros(spec.max_num_reqs, dtype=torch.bool),
                    attn_state="spec",
                ),
            ):
                metadata = spec._build_uniform_attn_metadata(
                    batch_desc=NS(cg_mode=mode, num_tokens=7, num_reqs=2),
                    num_reqs=1,
                    num_query_per_req=3,
                    seq_lens_cpu_upper_bound=None,
                    step=3,
                    causal=False,
                )
            self.assertIs(spec._flash_query_metadata, metadata)
            self.assertEqual(metadata["draft"].flash.num_tokens, expected_tokens)
            self.assertEqual(metadata["draft"].flash.used_q.tolist(), [3, 0, 0])

    def test_dspark_propose_builds_own_query_metadata(self):
        spec, _, module, _ = self.dspark_speculator()
        original = module.build_attn_metadata
        # FlashMLA must not use target prefill flags for the draft query batch.
        batch = NS(is_prefilling_np=object(), num_reqs=1)
        target_metadata = {"target": object()}
        dp_sync = object()
        result = spec.propose(batch, target_metadata, {}, None, None, None, None, None, None, None, None, dp_sync)
        self.assertEqual(result["draft"].flash.cache_lens.tolist(), [129, 0])
        self.assertIs(spec._flash_query_metadata, result)
        self.assertEqual(spec.last_factory["is_prefilling"].tolist(), [False] * spec.max_num_reqs)
        self.assertIs(spec.parent_propose_args[1], target_metadata)
        self.assertIs(spec.parent_propose_args[11], dp_sync)
        self.assertIs(module.build_attn_metadata, original)

    def test_dspark_context_writer_preserves_cache_and_rejected_slots(self):
        for use_rope in (False, True):
            backing, cache = self.cache()
            expected = backing.clone()
            expected_cache = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
            data = torch.arange(3 * 576).to(torch.bfloat16).view(3, 576) / 1024
            slots = torch.tensor([127, 99, -1, 99, 128, 99], dtype=torch.int32)[::2]
            transformed = data.clone()
            transformed[:, :512] *= 2
            if use_rope:
                transformed[:, 512:] += 1
            expected_cache[0, 127, 0] = transformed[0]
            expected_cache[1, 0, 0] = transformed[2]

            def scatter(cache=cache, **kwargs):
                self.assertEqual(kwargs["key_cache"].stride(), cache.stride())
                self.assertEqual(kwargs["key_cache"].storage_offset(), cache.storage_offset())
                self.assertEqual(kwargs["value_cache"].storage_offset(), cache.storage_offset() + 512)
                self.assertTrue(kwargs["slot_mapping"].is_contiguous())
                self.assertEqual(kwargs["slot_mapping"].dtype, torch.int64)
                for i, slot in enumerate(kwargs["slot_mapping"].tolist()):
                    if slot >= 0:
                        kwargs["key_cache"][slot // 128, slot % 128] = kwargs["key"][i]
                        kwargs["value_cache"][slot // 128, slot % 128] = kwargs["value"][i]

            cls = load_methods(
                "vllm_ascend/attention/mla_v1.py",
                "AscendMLAImpl",
                {"exec_kv_prefill"},
                MLAAttentionImpl=object,
                torch=torch,
                envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
                validate_flash_cache=self.api.validate_flash_cache,
                torch_npu=NS(npu_scatter_pa_kv_cache=scatter),
            )
            impl = cls()
            impl.kv_a_layernorm = lambda x: x * 2
            impl.use_mla_rope = use_rope
            impl.rope_single = lambda x, cos, sin: x + 1
            rope, latent = impl.exec_kv_prefill(data, None, None, cache, slots)
            self.assertTrue(torch.equal(backing, expected))
            self.assertTrue(torch.equal(latent[:, 0], transformed[:, :512]))
            self.assertTrue(torch.equal(rope[:, 0], transformed[:, 512:]))

    def test_dspark_scope_allows_eager_and_graph_configuration(self):
        hardware = types.ModuleType("vllm_ascend.device.device_config")
        hardware.is_950 = lambda: True
        api = load_definitions(
            "vllm_ascend/platform.py",
            {"_validate_flash_mla_config"},
            VllmConfig=object,
            torch=torch,
            envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            model_uses_sfa_sparse=lambda model: getattr(model, "sparse", False),
            KVPPConfig=NS(from_vllm_config=lambda cfg: NS(size=1)),
        )
        draft = NS(use_mla=True, dtype=torch.bfloat16, sparse=False)
        spec = NS(method="dspark", enforce_eager=True, draft_model_config=draft, enable_adaptive_verification=False)
        cfg = NS(
            model_config=NS(enforce_eager=True, use_mla=True, dtype=torch.bfloat16),
            use_v2_model_runner=True,
            cache_config=NS(cache_dtype="auto"),
            parallel_config=NS(
                prefill_context_parallel_size=1, decode_context_parallel_size=1, pipeline_parallel_size=1
            ),
            speculative_config=spec,
            kv_transfer_config=None,
        )
        with patch.dict(sys.modules, {hardware.__name__: hardware}):
            for target_eager in (False, True):
                for draft_eager in (False, True):
                    cfg.model_config.enforce_eager = target_eager
                    spec.enforce_eager = draft_eager
                    api._validate_flash_mla_config(cfg)
            for obj, field, value in (
                (spec, "method", "eagle"),
                (spec, "enable_adaptive_verification", True),
                (spec, "draft_model_config", None),
                (draft, "use_mla", False),
                (draft, "sparse", True),
                (draft, "dtype", torch.float16),
                (cfg.parallel_config, "pipeline_parallel_size", 2),
            ):
                old = getattr(obj, field)
                setattr(obj, field, value)
                with self.assertRaises(ValueError):
                    api._validate_flash_mla_config(cfg)
                setattr(obj, field, old)

    def captured(self, builder, common):
        builder._flash_capture = True
        builder._device_metadata_enabled = True
        result = self.api.build_flash_mla_metadata(builder, common)
        builder._flash_capture = False
        return result

    def test_graph_multiquery_stable_addresses_and_deferred_updates(self):
        for causal in (False, True):
            builder, common = self.builder(), self.common(batch=2, query=3, causal=causal)
            first = self.captured(builder, common)
            self.assertTrue(first.graph_buffer)
            self.assertEqual(first.cache_lens.tolist(), [0, 0, 0])
            builder._device_metadata_tasks[0].run()
            pointers = {
                name: value.data_ptr() for name, value in vars(first).items() if isinstance(value, torch.Tensor)
            }
            original_schedule = first.schedule.clone()
            common.seq_lens[:] = torch.tensor([130, 0])
            common.block_table_tensor.add_(9)
            common.positions.add_(8)
            second = self.api.build_flash_mla_metadata(builder, common)
            self.assertIs(first, second)
            self.assertEqual(first.cache_lens[0], 128000)
            builder._device_metadata_tasks[0].run()
            self.assertEqual(second.cache_lens.tolist(), [130, 0, 0])
            self.assertEqual(second.used_q.tolist(), [3, 0, 0])
            self.assertEqual(second.slots.tolist(), [0, 1, 2, -1, -1, -1, -1, -1])
            self.assertEqual(second.block_table[0, 0], 9)
            self.assertEqual(second.positions[0], 8)
            self.assertFalse(torch.equal(original_schedule, second.schedule))
            self.assertEqual(
                pointers, {n: v.data_ptr() for n, v in vars(second).items() if isinstance(v, torch.Tensor)}
            )
            self.api.validate_flash_graph_metadata({"draft": NS(flash=second)})

    def test_graph_buckets_and_target_draft_are_isolated(self):
        target, draft = self.builder(), self.builder()
        common = self.common()
        first = self.captured(target, common)
        second = self.captured(draft, common)
        third = self.captured(target, self.common(batch=2))
        self.assertEqual(len({b.schedule.data_ptr() for b in (first, second, third)}), 3)
        self.assertEqual(len(target._flash_buffers), 2)
        target._device_metadata_enabled = False
        unseen = self.api.build_flash_mla_metadata(target, self.common(batch=3))
        self.assertFalse(unseen.graph_buffer)
        self.assertEqual(len(target._flash_buffers), 2)
        with self.assertRaisesRegex(RuntimeError, "No captured"):
            self.api.validate_flash_graph_metadata({"layer": NS(flash=unseen)})
        with self.assertRaisesRegex(RuntimeError, "executor"):
            self.api.build_flash_mla_metadata(target, common)

    def test_graph_requires_meta_schema_and_matching_runtime_capacity(self):
        package = sys.modules["cann_ops_transformer.ops"]
        for bad in (
            torch.empty(8, dtype=torch.int32),
            torch.empty(0, device="meta", dtype=torch.int32),
            torch.empty(8, device="meta"),
            torch.empty(2, 4, device="meta", dtype=torch.int32),
        ):
            with (
                patch.object(package, "flash_mla_with_kvcache_metadata", return_value=bad),
                self.assertRaises(RuntimeError),
            ):
                self.captured(self.builder(), self.common())
        builder = self.builder()
        first = self.captured(builder, self.common())
        pointer = first.schedule.data_ptr()
        with (
            patch.object(package, "flash_mla_with_kvcache_metadata", return_value=torch.zeros(1, dtype=torch.int32)),
            self.assertRaisesRegex(RuntimeError, "capacity"),
        ):
            builder._device_metadata_tasks[0].run()
        self.assertEqual(pointer, first.schedule.data_ptr())

    def test_graph_builder_capture_and_task_flags_are_scoped(self):
        cls = load_methods(
            "vllm_ascend/attention/mla_v1.py",
            "AscendMLAMetadataBuilder",
            {"build_for_cudagraph_capture", "enable_device_metadata", "take_device_metadata_tasks"},
            MLACommonMetadataBuilder=type("Base", (), {"__class_getitem__": classmethod(lambda cls, _: cls)}),
            AscendMLAMetadata=object,
            envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
        )
        builder = cls()
        builder._device_metadata_tasks = ()
        builder.enable_device_metadata()
        self.assertTrue(builder._device_metadata_enabled)
        self.assertEqual(builder.take_device_metadata_tasks(), ())
        self.assertFalse(builder._device_metadata_enabled)

        def fail(*args):
            self.assertTrue(builder._flash_capture)
            raise RuntimeError("capture failed")

        builder.build = fail
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            builder.build_for_cudagraph_capture(self.common())
        self.assertFalse(builder._flash_capture)

    def metadata_context_api(self):
        variable = ContextVar("test_executor", default=None)
        api = load_definitions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"device_metadata_context"},
            contextmanager=contextmanager,
            DeviceMetadataExecutor=object,
            _device_metadata_executor=variable,
        )
        return variable, api.device_metadata_context

    def test_metadata_context_nested_executors_and_exception_cleanup(self):
        variable, context = self.metadata_context_api()
        released = []
        target = NS(submission_in_flight=True, release=lambda: released.append("target"))
        draft = NS(submission_in_flight=True, release=lambda: released.append("draft"))
        with self.assertRaisesRegex(RuntimeError, "consumer failed"), context(target):
            with context(target):
                self.assertIs(variable.get(), target)
            self.assertEqual(released, [])
            with context(draft):
                self.assertIs(variable.get(), draft)
            self.assertIs(variable.get(), target)
            raise RuntimeError("consumer failed")
        self.assertIsNone(variable.get())
        self.assertEqual(released, ["draft", "target"])

    def test_target_runner_does_not_require_dspark(self):
        variable, context = self.metadata_context_api()
        calls = []
        executor = NS(submission_in_flight=True, release=lambda: calls.append("release"))

        class Parent:
            def execute_model(self, scheduler_output, **kwargs):
                self.observed_kwargs = kwargs
                self.observed_executor = variable.get()
                calls.append("target")
                return scheduler_output

        cls = load_methods(
            "vllm_ascend/worker/v2/model_runner.py",
            "NPUModelRunner",
            {"execute_model"},
            GPUModelRunner=Parent,
            torch=torch,
            device_metadata_context=context,
            _start_profiling_chunk_timing=lambda *args: None,
            _finish_profiling_chunk_timing=lambda *args: None,
            has_kv_transfer_group=lambda: False,
        )
        runner = cls()
        runner.ascend_config = NS(scheduler_config=NS(profiling_chunk_config=None))
        runner.model_state = NS()
        runner.kvpp = NS(complete_forward=lambda: None)
        # No speculator object or draft configuration is supplied.
        for active_executor in (None, executor):
            runner.device_metadata_executor = active_executor
            self.assertEqual(runner.execute_model("target", valid_dummy_state_slots=True), "target")
            self.assertIs(runner.observed_executor, active_executor)
            self.assertTrue(runner.observed_kwargs["valid_dummy_state_slots"])
            self.assertIsNone(variable.get())
        self.assertEqual(calls, ["target", "target", "release"])

    def test_target_graph_replay_does_not_require_dspark(self):
        cls = load_methods(
            "vllm_ascend/worker/v2/aclgraph_utils.py",
            "ModelAclGraphManager",
            {"run_fullgraph"},
            ModelCudaGraphManager=object,
            set_current_vllm_config=lambda _: nullcontext(),
            _get_graph_update_backend=lambda _: object(),
            use_updatable_graph=lambda _: False,
            validate_flash_graph_metadata=self.api.validate_flash_graph_metadata,
        )
        manager = cls()
        manager.update_stream = object()
        manager.vllm_config = NS(speculative_config=None)
        flash = self.captured(self.builder(), self.common(query=1))
        metadata = {"target": NS(flash=flash)}
        manager.model_runner = NS(attn_groups=[], model_state=NS(attn_metadata=metadata))
        manager._graph_relay = lambda backend, desc, tokens, attn: attn
        self.assertIs(manager.run_fullgraph(NS(num_tokens=3)), metadata)
        flash.graph_buffer = False
        with self.assertRaisesRegex(RuntimeError, "No captured"):
            manager.run_fullgraph(NS(num_tokens=3))

    def test_executor_orders_producer_consumer_and_reuse_fence(self):
        events = []

        def stream(name):
            return NS(name=name, wait_event=lambda event: events.append((name, "wait", event.name)))

        consumer, producer = stream("consumer"), stream("producer")
        names = iter(("inputs", "reuse", "ready"))

        def event():
            name = next(names)
            return NS(name=name, record=lambda s: events.append((s.name, "record", name)))

        npu = NS(Stream=lambda: producer, Event=event, current_stream=lambda: consumer, stream=lambda _: nullcontext())
        with patch.object(torch, "npu", npu, create=True):
            executor = self.device_api.DeviceMetadataExecutor()
            builder = self.builder()
            self.captured(builder, self.common())
            task = builder._device_metadata_tasks[0]
            executor.submit((task,))
            executor.wait(task.stage, task.group_id)
            executor.wait(task.stage, task.group_id)
            self.assertEqual(
                events,
                [
                    ("consumer", "record", "inputs"),
                    ("producer", "wait", "inputs"),
                    ("producer", "record", "ready"),
                    ("consumer", "wait", "ready"),
                ],
            )
            with self.assertRaisesRegex(RuntimeError, "not been released"):
                executor.submit((task,))
            events.append(("consumer", "replay"))
            executor.release()
            executor.submit((task,))
            self.assertEqual(
                events[-5:],
                [
                    ("consumer", "record", "reuse"),
                    ("consumer", "record", "inputs"),
                    ("producer", "wait", "inputs"),
                    ("producer", "wait", "reuse"),
                    ("producer", "record", "ready"),
                ],
            )
            executor.release()

    def test_dspark_graph_consumes_fresh_metadata_once_without_fia_update(self):
        class Parent:
            def run_fullgraph(self, desc):
                return desc

        cls = load_methods(
            "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
            "DFlashAclGraphManager",
            {"run_fullgraph"},
            DFlashCudaGraphManager=Parent,
            ascend_envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            validate_flash_graph_metadata=self.api.validate_flash_graph_metadata,
        )
        manager = cls()
        manager.speculator = NS(attn_architecture="MLA", _flash_query_metadata=None)
        with self.assertRaisesRegex(RuntimeError, "freshly built"):
            manager.run_fullgraph("replay")
        metadata = self.captured(self.builder(), self.common(query=3))
        manager.speculator._flash_query_metadata = {"draft": NS(flash=metadata)}
        self.assertEqual(manager.run_fullgraph("replay"), "replay")
        with self.assertRaisesRegex(RuntimeError, "freshly built"):
            manager.run_fullgraph("replay")

    def test_dspark_capture_factory_preserves_causality_and_restores(self):
        calls = []
        module = NS(build_attn_metadata=object())
        original = module.build_attn_metadata
        cls = load_methods(
            "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
            "AscendDSparkSpeculator",
            {"draft_capture_context", "_draft_query_attn_state"},
            DSparkSpeculator=object,
            ascend_envs=NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            contextmanager=contextmanager,
            dflash_cudagraph=module,
            build_attn_metadata=lambda **kw: calls.append(kw),
            torch=torch,
            AscendAttentionState=NS(SpecDecoding="spec"),
        )
        spec = cls()
        spec.attn_architecture = "MLA"
        spec.input_buffers = NS(positions=torch.arange(8))
        spec.attn_vllm_config = NS(parallel_config=object())
        with self.assertRaisesRegex(RuntimeError, "capture failed"), spec.draft_capture_context():
            module.build_attn_metadata(num_tokens=6, num_reqs=2, causal={0: False}, for_cudagraph_capture=True)
            raise RuntimeError("capture failed")
        self.assertIs(module.build_attn_metadata, original)
        self.assertEqual(calls[0]["causal"], {0: False})
        self.assertTrue(calls[0]["for_cudagraph_capture"])
        self.assertEqual(calls[0]["positions"].tolist(), list(range(6)))
        self.assertEqual(calls[0]["is_prefilling"].tolist(), [False, False])

    def test_mrv2_metadata_factory_submits_and_waits_before_consumer(self):
        variable, context = self.metadata_context_api()
        events = []
        api = self.api
        test = self

        class Builder:
            def enable_device_metadata(self):
                self._device_metadata_enabled = True

            def take_device_metadata_tasks(self):
                self._device_metadata_enabled = False
                tasks, self._device_metadata_tasks = self._device_metadata_tasks, ()
                return tasks

            def build(self, common_prefix_len, common_attn_metadata):
                return NS(flash=api.build_flash_mla_metadata(self, common_attn_metadata))

            def build_for_cudagraph_capture(self, common):
                self._flash_capture = True
                try:
                    return self.build(0, common)
                finally:
                    self._flash_capture = False

        class Executor:
            submission_in_flight = False

            def release(self):
                events.append("release")
                self.submission_in_flight = False

            def submit(self, tasks):
                test.assertFalse(self.submission_in_flight)
                self.submission_in_flight = True
                events.append("submit")
                for task in tasks:
                    task.run()

            def wait(self, *args):
                events.append("wait")

        never = type("OtherBuilder", (), {})
        npu = NS(is_current_stream_capturing=lambda: False)
        fake_torch = NS(Tensor=torch.Tensor, from_numpy=lambda value: value, npu=npu)
        factory = load_definitions(
            "vllm_ascend/worker/v2/attn_utils.py",
            {"build_attn_metadata"},
            torch=fake_torch,
            np=NS(ndarray=torch.Tensor),
            AttentionGroup=object,
            Sequence=Sequence,
            Mapping=Mapping,
            Any=Any,
            KVCacheConfig=object,
            ParallelConfig=object,
            ModelSpecificAttnMetadata=object,
            AscendCommonAttentionMetadata=NS,
            AscendDSAMetadataBuilder=never,
            AscendSFAMetadataBuilder=never,
            GDNAttentionMetadataBuilder=never,
            DeviceMetadataTaskProvider=Builder,
            _device_metadata_executor=variable,
        )
        builder = Builder()
        builder.device, builder.decode_threshold = "cpu", 1
        api.init_flash_mla_metadata(builder, NS(num_heads=8, num_kv_heads=1))
        common = self.common(query=3)
        kwargs = dict(
            attn_groups=[[NS(get_metadata_builder=lambda _: builder, layer_names=["a", "b"])]],
            num_reqs=common.num_reqs,
            num_tokens=common.num_input_tokens,
            query_start_loc_gpu=common.query_start_loc,
            query_start_loc_cpu=common.query_start_loc,
            max_query_len=3,
            seq_lens=common.seq_lens,
            seq_lens_np=common.seq_lens,
            max_seq_len=128000,
            block_tables=[common.block_table_tensor],
            slot_mappings=[common.slot_mapping],
            positions=common.positions,
            kv_cache_config=NS(kv_cache_groups=[object()]),
        )
        executor = Executor()
        with context(executor):
            first = factory.build_attn_metadata(**kwargs, for_cudagraph_capture=True)
            self.assertIs(first["a"], first["b"])
            self.assertEqual(events, ["submit", "wait"])
            events.append("consumer")
            common.seq_lens.fill_(130)
            second = factory.build_attn_metadata(**kwargs)
            self.assertIs(first["a"].flash, second["a"].flash)
            self.assertEqual(second["a"].flash.cache_lens.tolist(), [130, 0])
            self.assertEqual(events, ["submit", "wait", "consumer", "release", "submit", "wait"])
            events.append("replay")
        self.assertEqual(events[-2:], ["replay", "release"])
        self.assertFalse(builder._device_metadata_enabled)
        npu.is_current_stream_capturing = lambda: True
        with context(executor), self.assertRaisesRegex(RuntimeError, "outside"):
            factory.build_attn_metadata(**kwargs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
