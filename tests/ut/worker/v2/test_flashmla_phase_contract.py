# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of actual runner phase ordering with upstream API shells."""

import ast
from collections import namedtuple
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]


def _class_method(path, class_name, method_name, base, namespace):
    tree = ast.parse((ROOT / path).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    node = ast.ClassDef(
        name=class_name, bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace["Parent"] = base
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])), str(ROOT / path), "exec"),
        namespace,
    )
    return namespace[class_name]


@pytest.mark.parametrize("prompt_width", [1, 2])
def test_runner_sorts_short_prefill_identity_and_preserves_upstream_graph_selection(prompt_width):
    # Upstream supplies actual phase and heterogeneous indexed request fields.
    batch_type = namedtuple(
        "Batch",
        "req_ids num_scheduled_tokens idx_mapping_np prefill_len_np "
        "num_computed_prefill_tokens_np is_prefilling_np has_prefill num_tokens",
    )
    batch = batch_type(
        ["suffix", "decode-a", "cold", "decode-b"],
        np.array([prompt_width, 1, prompt_width, 1]),
        np.array([8, 3, 6, 2]),
        np.array([100 + prompt_width, 20, prompt_width, 30]),
        np.array([100, 20, 0, 30]),
        np.array([True, False, True, False]),
        True,
        2 * prompt_width + 2,
    )

    class Parent:
        def gather_batch_req_state(self, *_):
            return batch, None

    runner_class = _class_method(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        "gather_batch_req_state",
        Parent,
        dict(
            np=np,
            ascend_envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=True),
            is_pd_decode_recompute_scheduler_enabled=lambda _: False,
        ),
    )
    runner = runner_class()
    runner.vllm_config = object()
    reordered, uniform = runner.gather_batch_req_state(object(), False)
    assert reordered.req_ids == ["decode-a", "decode-b", "suffix", "cold"]
    for name in (
        "num_scheduled_tokens",
        "idx_mapping_np",
        "prefill_len_np",
        "num_computed_prefill_tokens_np",
        "is_prefilling_np",
    ):
        np.testing.assert_array_equal(getattr(reordered, name), getattr(batch, name)[[1, 3, 0, 2]])
    assert uniform is None and reordered.has_prefill
    # Pure decode and initial profiling preserve the parent's batch contract.
    batch = batch._replace(is_prefilling_np=np.zeros(4, dtype=bool), has_prefill=False)
    reordered, uniform = runner.gather_batch_req_state(object(), False)
    assert reordered.req_ids == batch.req_ids
    assert not reordered.has_prefill
    batch = None
    assert runner.gather_batch_req_state(object(), True) == (None, None)


@pytest.mark.parametrize(
    "enabled, profiling, rejects",
    [(False, False, False), (False, True, False), (True, True, False), (True, False, True)],
)
def test_initial_profiling_guard_does_not_change_disabled_execution(enabled, profiling, rejects):
    sentinel = object()

    class Parent:
        def execute_model(self, *_args, **kwargs):
            self.parent_called = True
            assert kwargs["is_profile"] == profiling
            return sentinel

    runner_class = _class_method(
        "vllm_ascend/worker/v2/model_runner.py",
        "NPUModelRunner",
        "execute_model",
        Parent,
        dict(
            _start_profiling_chunk_timing=lambda *_: None,
            _finish_profiling_chunk_timing=lambda *_: None,
            has_kv_transfer_group=lambda: False,
            flashmla_metadata_scope=lambda *_: nullcontext(),
        ),
    )
    runner = runner_class()
    runner.parent_called = False
    runner.ascend_config = SimpleNamespace(scheduler_config=SimpleNamespace(profiling_chunk_config=None))
    runner.model_state = SimpleNamespace()
    runner.kvpp = SimpleNamespace(complete_forward=lambda: None)
    runner.flashmla_executor = object() if enabled else None
    args = dict(dummy_run=profiling, is_profile=profiling, skip_attn_for_dummy_run=profiling)
    if rejects:
        with pytest.raises(RuntimeError, match="attention groups"):
            runner.execute_model(object(), **args)
        assert not runner.parent_called
    else:
        assert runner.execute_model(object(), **args) is sentinel
        assert runner.parent_called
        assert not runner.model_state.kvpp_is_dummy_run
