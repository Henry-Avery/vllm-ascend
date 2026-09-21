# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute DSpark Python contracts on CPU, without NPU/upstream imports.

Run directly. Selected repository methods execute unchanged; only heavyweight
construction, upstream dispatch and package dependencies are stubbed. This is
not model loading, NPU operator correctness, or graph execution evidence.
"""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = "vllm_ascend/models/kimi_k3_dspark.py"
SPEC = "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"


def load_selected(path, functions=(), classes=None, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in (classes or {}):
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in classes[node.name]]
            selected.append(node)
    tree.body = selected
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class DSparkHostContract(unittest.TestCase):
    def model_scope(self):
        def load(owner, **kwargs):
            def consume(weights, **kwargs):
                owner.loaded = dict(weights)
                return set(owner.loaded)

            return SimpleNamespace(load_weights=consume)

        process = load_selected("vllm_ascend/models/qwen3_dspark.py", ["process_weight"], torch=torch).process_weight
        return load_selected(
            MODEL,
            ["_get_target_rotation_path"],
            {"AscendK3DSparkForCausalLM": {"__init__", "configure_target_aux_hidden_capture", "load_weights"}},
            torch=torch,
            nn=torch.nn,
            UpstreamK3DSparkForCausalLM=torch.nn.Module,
            get_rotation_path=lambda cfg: getattr(cfg, "rotation", None),
            AscendK3DSparkModel=Mock(return_value=SimpleNamespace(embed_tokens=None)),
            LogitsProcessor=Mock(),
            VocabParallelEmbedding=Mock(return_value=object()),
            ParallelLMHead=Mock(return_value=object()),
            maybe_prefix=lambda prefix, name: f"{prefix}.{name}" if prefix else name,
            AutoWeightsLoader=load,
            vllm_version_is=lambda version: False,
            get_rotation_matrix=Mock(return_value=torch.tensor([[0.0, 1.0], [-1.0, 0.0]])),
            process_weight=process,
            load_quarot_target_layer=Mock(),
            TARGET_EMBED_WEIGHT_NAMES=("embed",),
            TARGET_LM_HEAD_WEIGHT_NAMES=("head",),
        )

    @staticmethod
    def config(rotation=None):
        draft = SimpleNamespace(draft_vocab_size=8, _ascend_target_rotation_path=rotation)
        return SimpleNamespace(
            speculative_config=SimpleNamespace(draft_model_config=SimpleNamespace(hf_config=draft)),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=93, hidden_size=2, vocab_size=8),
                get_num_layers=lambda parallel: 31,
                model="target-weights",
            ),
            parallel_config=object(),
        )

    def test_rotation_handoff_and_no_rotation(self):
        scope = self.model_scope()
        for direct, inherited, expected in [
            (None, None, None),
            (None, "target", "target"),
            ("direct", "target", "direct"),
        ]:
            with self.subTest(direct=direct, inherited=inherited):
                cfg = self.config(inherited)
                cfg.rotation = direct
                self.assertEqual(scope._get_target_rotation_path(cfg), expected)

    def test_constructor_uses_full_target_layers_and_own_rotated_weights(self):
        scope = self.model_scope()
        model = scope.AscendK3DSparkForCausalLM(vllm_config=self.config("target"))
        self.assertEqual(scope.AscendK3DSparkModel.call_args.kwargs["start_layer_id"], 93)
        self.assertEqual(model.rotation_path, "target")
        self.assertIsNotNone(model.model.embed_tokens)
        scope.VocabParallelEmbedding.assert_called_once()
        scope.ParallelLMHead.assert_called_once()

    def test_rotation_applied_once_before_weight_sharing(self):
        scope = self.model_scope()
        model = scope.AscendK3DSparkForCausalLM(vllm_config=self.config("target"))
        model.hf_to_vllm_mapper = object()
        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        untouched = torch.ones(1, 2)
        model.load_weights(iter([("model.context_proj.weight", weight), ("model.layers.0.weight", untouched)]))
        torch.testing.assert_close(model.loaded["model.context_proj.weight"], torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))
        self.assertIs(model.loaded["model.layers.0.weight"], untouched)
        scope.get_rotation_matrix.assert_called_once_with("target")
        self.assertEqual(scope.load_quarot_target_layer.call_count, 2)
        self.assertTrue(model.has_own_embed_tokens)
        self.assertTrue(model.has_own_lm_head)

    def test_unrotated_loading_preserves_weight_identity(self):
        scope = self.model_scope()
        model = scope.AscendK3DSparkForCausalLM(vllm_config=self.config())
        model.hf_to_vllm_mapper = object()
        weight = torch.ones(1, 4)
        model.load_weights(iter([("model.context_proj.weight", weight)]))
        self.assertIs(model.loaded["model.context_proj.weight"], weight)
        scope.get_rotation_matrix.assert_not_called()
        scope.load_quarot_target_layer.assert_not_called()

    def aux_pair(self):
        model = SimpleNamespace(
            config=SimpleNamespace(target_layer_ids=[0, 2], target_hidden_size=4, num_target_layers=2)
        )
        target = SimpleNamespace(
            model=SimpleNamespace(
                config=SimpleNamespace(num_hidden_layers=4, hidden_size=4), aux_hidden_state_layers=(1, 3)
            ),
            set_dspark_aux_capture_materialized=Mock(),
        )
        return model, target

    def test_aux_raw_capture_and_multimodal_wrapper(self):
        hook = self.model_scope().AscendK3DSparkForCausalLM.configure_target_aux_hidden_capture
        for wrapped in (False, True):
            model, target = self.aux_pair()
            hook(model, SimpleNamespace(get_language_model=lambda target=target: target) if wrapped else target)
            target.set_dspark_aux_capture_materialized.assert_called_once_with(False)

    def test_aux_rejects_bad_contract_before_setting_mode(self):
        hook = self.model_scope().AscendK3DSparkForCausalLM.configure_target_aux_hidden_capture
        for field, value in [
            ("target_layer_ids", []),
            ("target_layer_ids", [0, 0]),
            ("target_layer_ids", [-1, 2]),
            ("target_layer_ids", [0, 4]),
            ("target_layer_ids", [1, 2]),
            ("target_hidden_size", 8),
            ("num_target_layers", 3),
        ]:
            with self.subTest(field=field, value=value):
                model, target = self.aux_pair()
                setattr(model.config, field, value)
                with self.assertRaises(ValueError):
                    hook(model, target)
                target.set_dspark_aux_capture_materialized.assert_not_called()
        model, target = self.aux_pair()
        del target.set_dspark_aux_capture_materialized
        with self.assertRaises(ValueError):
            hook(model, target)

    def spec_scope(self, architecture="MLA", flash=True):
        class Upstream:
            def _build_draft_attn_metadata(self, **kwargs):
                self.calls.append(kwargs)
                return self.metadata

            def set_attn(self, *args):
                pass

        contexts = []

        @contextmanager
        def factory(positions, pad, is_prefilling, **kwargs):
            contexts.append((pad, is_prefilling.clone(), kwargs))
            yield

        scope = load_selected(
            SPEC,
            classes={
                "AscendDSparkSpeculator": {
                    "set_attn",
                    "_draft_query_attn_state",
                    "draft_capture_context",
                    "_build_draft_attn_metadata",
                    "build_draft_attn_metadatas",
                    "_update_draft_attn_metadata",
                }
            },
            DSparkSpeculator=Upstream,
            torch=torch,
            contextmanager=contextmanager,
            ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash),
            AscendAttentionState=SimpleNamespace(SpecDecoding="spec", ChunkedPrefill="chunked"),
            build_attn_metadata_wrapper=nullcontext,
            build_draft_attn_metadata_factory=factory,
            dflash_cudagraph=SimpleNamespace(build_attn_metadata=Mock()),
            build_attn_metadata=Mock(return_value={}),
            set_current_vllm_config=lambda cfg: nullcontext(),
            _get_graph_update_backend=lambda groups: groups,
            AscendMLABackend=type("MLA", (), {}),
            AscendAttentionBackend=type("GQA", (), {}),
        )
        spec = scope.AscendDSparkSpeculator()
        spec.attn_architecture = architecture
        spec.num_query_per_req = 5
        spec.input_buffers = SimpleNamespace(positions=torch.arange(32))
        spec.input_batch = SimpleNamespace(num_reqs=1)
        spec._group_causal = {0: False}
        spec.calls = []
        spec.metadata = {}
        spec.contexts = contexts
        return scope, spec

    def test_backend_selection_uses_draft_not_target_configuration(self):
        scope, spec = self.spec_scope()
        spec.attn_vllm_config = object()
        spec._context_slot_mappings = torch.arange(4)
        spec.draft_attn_layer_names = set()
        for backend, expected in [
            (scope.AscendMLABackend, "MLA"),
            (scope.AscendAttentionBackend, "GQA"),
            (object, None),
        ]:
            spec.attn_groups = backend
            spec.set_attn(None, SimpleNamespace(kv_cache_groups=[]), None, None, None)
            self.assertEqual(spec.attn_architecture, expected)
            self.assertEqual(spec._context_slot_mappings.dtype, torch.int32)

    def test_fia_dp_padding_and_state(self):
        for architecture in ("MLA", "GQA"):
            with self.subTest(architecture=architecture):
                _, spec = self.spec_scope(architecture, flash=False)
                query = SimpleNamespace(actual_seq_lengths_q=[5])
                spec.metadata = {
                    "draft": SimpleNamespace(decode=query, attn_state="chunked") if architecture == "MLA" else query
                }
                spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=10, step=5)
                self.assertEqual(spec.calls[0]["num_reqs_padded"], 2)
                self.assertEqual(query.actual_seq_lengths_q, [5, 10])
                pad, flags, kwargs = spec.contexts[0]
                self.assertEqual(pad, 10)
                self.assertEqual(flags.tolist(), [False, False])
                self.assertEqual(kwargs["attn_state"], "chunked")
                with self.assertRaisesRegex(AssertionError, "whole query groups"):
                    spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=9)
                self.assertEqual(len(spec.calls), 1)

    def test_flash_keeps_device_boundaries_and_zero_used_padding(self):
        _, spec = self.spec_scope()
        flash = SimpleNamespace(cu=torch.tensor([0, 5, 12]), used_q=torch.tensor([5, 0]))
        spec.metadata = {"draft": SimpleNamespace(flash=flash, decode=None)}
        result = spec._build_draft_attn_metadata(num_reqs=1, num_reqs_padded=1, num_tokens_padded=12, step=5)
        self.assertIs(result["draft"].flash, flash)
        self.assertEqual(spec.calls[0]["num_reqs_padded"], 1)
        self.assertEqual(flash.cu.tolist(), [0, 5, 12])
        self.assertEqual(flash.used_q.tolist(), [5, 0])
        self.assertEqual(spec.contexts[0][2]["attn_state"], "spec")

    def test_full_builder_calls_parent_once_and_keeps_group_causality(self):
        _, spec = self.spec_scope()
        spec.metadata = {"draft": SimpleNamespace(flash=object(), decode=None)}
        self.assertEqual(spec.build_draft_attn_metadatas(4, torch.tensor([20])), [spec.metadata])
        self.assertEqual(len(spec.calls), 1)
        self.assertEqual(spec.calls[0]["num_tokens_padded"], 20)
        self.assertEqual(spec.calls[0]["causal"], {0: False})
        self.assertEqual(spec.contexts[0][1].tolist(), [False] * 4)

    def test_sparse_metadata_is_not_normalized_as_dense(self):
        _, spec = self.spec_scope(None)
        spec.metadata = {"sparse": SimpleNamespace(actual_seq_lengths_q=[5, 5])}
        self.assertIs(spec._build_draft_attn_metadata(num_reqs_padded=2), spec.metadata)
        self.assertEqual(spec.metadata["sparse"].actual_seq_lengths_q, [5, 5])
        self.assertEqual(spec.contexts, [])

    def test_capture_factory_matches_runtime_and_restores_on_failure(self):
        for flash in (False, True):
            with self.subTest(flash=flash):
                scope, spec = self.spec_scope(flash=flash)
                original = scope.dflash_cudagraph.build_attn_metadata
                with self.assertRaisesRegex(RuntimeError, "capture failed"), spec.draft_capture_context():
                    scope.dflash_cudagraph.build_attn_metadata(num_tokens=10, num_reqs=2)
                    kwargs = scope.build_attn_metadata.call_args.kwargs
                    self.assertEqual(kwargs["attn_state"], spec._draft_query_attn_state())
                    self.assertEqual(kwargs["positions"].numel(), 10)
                    self.assertEqual(kwargs["is_prefilling"].tolist(), [False, False])
                    raise RuntimeError("capture failed")
                self.assertIs(scope.dflash_cudagraph.build_attn_metadata, original)


if __name__ == "__main__":
    unittest.main()
