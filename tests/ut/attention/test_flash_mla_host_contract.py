# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run directly on CPU; mocked operators do not establish NPU correctness."""

import ast
import sys
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
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
            return torch.full((lengths.numel() * 8,), len(self.calls), dtype=torch.int32)

        def main(query, cache, **kwargs):
            self.calls.append(("main", query, cache, kwargs))
            return torch.zeros(query.shape[1], query.shape[0], 512, dtype=query.dtype), torch.empty(0)

        package.flash_mla_with_kvcache_metadata = metadata
        package.flash_mla_with_kvcache = main
        self.modules = patch.dict(sys.modules, {"cann_ops_transformer.ops": package})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.api = load_definitions(
            "vllm_ascend/attention/flash_mla.py",
            torch=torch,
            dataclass=dataclass,
            MLA_FLASH_SUPPORTED_Q_HEADS={8, 12, 64, 96},
            FLASH_MLA_BLOCK_SIZE=128,
            FLASH_MLA_QK_DIM=576,
            FLASH_MLA_V_DIM=512,
            FLASH_MLA_MASK_SIZE=2048,
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
        self.assertFalse(hasattr(builder, "_flash_buffers"))

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
            api._validate_flash_mla_config(cfg)
            for obj, field, value in (
                (cfg.model_config, "enforce_eager", False),
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
            def _build_draft_attn_metadata(self, **kwargs):
                self.parent_kwargs = kwargs
                return module.build_attn_metadata(
                    num_reqs=kwargs["num_reqs_padded"],
                    num_tokens=kwargs["num_tokens_padded"],
                    causal=kwargs.get("causal", False),
                )

            def propose(self, *args, **kwargs):
                self.parent_propose_args = args
                return self._build_draft_attn_metadata(
                    num_reqs=1, num_reqs_padded=1, num_tokens_padded=7, step=3, causal=False
                )

        enabled = NS(VLLM_ASCEND_ENABLE_FLASH_MLA=True)
        cls = load_methods(
            "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py",
            "AscendDSparkSpeculator",
            {
                "_draft_query_attn_state",
                "_prepare_draft_dcp_metadata_inputs",
                "_build_draft_attn_metadata",
                "_update_draft_attn_metadata",
                "propose",
            },
            DSparkSpeculator=Parent,
            torch=torch,
            ascend_envs=enabled,
            AscendAttentionState=NS(SpecDecoding="spec", ChunkedPrefill="prefill"),
            build_attn_metadata_wrapper=wrappers.build_attn_metadata_wrapper,
            build_draft_attn_metadata_factory=wrappers.build_draft_attn_metadata_factory,
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
        return spec, enabled, module, wrappers

    def test_dspark_query_device_lengths_padding_and_refresh(self):
        spec, _, _, _ = self.dspark_speculator()
        for causal in (False, True):
            first = spec._build_draft_attn_metadata(
                num_reqs=1, num_reqs_padded=1, num_tokens_padded=7, step=3, causal=causal
            )["draft"].flash
            # Seven physical tokens need not be a whole number of draft groups.
            self.assertEqual(spec.parent_kwargs["num_reqs_padded"], 1)
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
                num_reqs=1, num_reqs_padded=2, num_tokens_padded=7, step=3, causal=causal
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
            result = spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=6, step=3)[
                "draft"
            ]
            query = result.decode if architecture == "MLA" else result
            self.assertEqual(query.actual_seq_lengths_q, [3, 6])
            self.assertEqual(spec.parent_kwargs["num_reqs_padded"], 2)
            self.assertEqual(spec.last_factory["attn_state"], "prefill")
        enabled.VLLM_ASCEND_ENABLE_FLASH_MLA = True
        spec.attn_architecture = "MLA"
        spec.fail = True
        with wrappers.build_attn_metadata_wrapper():
            outer = module.build_attn_metadata
            with self.assertRaisesRegex(RuntimeError, "metadata failed"):
                spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=7, step=3)
            self.assertIs(module.build_attn_metadata, outer)
        self.assertIs(module.build_attn_metadata, original)

    def test_dspark_propose_builds_own_query_metadata(self):
        spec, _, module, _ = self.dspark_speculator()
        original = module.build_attn_metadata
        # FlashMLA must not use target prefill flags for the draft query batch.
        batch = NS(is_prefilling_np=object(), num_reqs=1)
        target_metadata = {"target": object()}
        dp_sync = object()
        result = spec.propose(batch, target_metadata, {}, None, None, None, None, None, None, None, None, dp_sync)
        self.assertEqual(result["draft"].flash.cache_lens.tolist(), [129, 0])
        self.assertEqual(spec.last_factory["is_prefilling"].tolist(), [False])
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

    def test_dspark_scope_allows_only_supported_eager_configuration(self):
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
            api._validate_flash_mla_config(cfg)
            for obj, field, value in (
                (spec, "method", "eagle"),
                (spec, "enforce_eager", False),
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


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
