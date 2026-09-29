# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of actual runner sorting/graph dispatch with upstream API shells."""

import ast
from collections import namedtuple
from dataclasses import dataclass, replace
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
def test_runner_sorts_short_prefill_identity_before_metadata_and_disables_full_graph(prompt_width):
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
    runner.cudagraph_manager = SimpleNamespace(flashmla_has_prefill=False)
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
    assert runner.cudagraph_manager.flashmla_has_prefill

    modes = SimpleNamespace(FULL="full", NONE="none")

    @dataclass
    class Descriptor:
        cg_mode: str = modes.FULL
        num_reqs: int = 8
        num_tokens: int = 16

    class GraphParent:
        def dispatch(self, *_args, **_kwargs):
            return Descriptor()

    manager_class = _class_method(
        "vllm_ascend/worker/v2/aclgraph_utils.py",
        "ModelAclGraphManager",
        "dispatch",
        GraphParent,
        dict(replace=replace, CUDAGraphMode=modes),
    )
    manager = manager_class()
    manager.flashmla_has_prefill = runner.cudagraph_manager.flashmla_has_prefill
    descriptor = manager.dispatch(len(reordered.req_ids), reordered.num_tokens)
    assert descriptor.cg_mode == modes.NONE
    assert (descriptor.num_reqs, descriptor.num_tokens) == (4, 2 * prompt_width + 2)

    # The next pure-decode step and profiling must clear the stale phase gate.
    batch = batch._replace(is_prefilling_np=np.zeros(4, dtype=bool), has_prefill=False)
    runner.gather_batch_req_state(object(), False)
    manager.flashmla_has_prefill = runner.cudagraph_manager.flashmla_has_prefill
    assert manager.dispatch(4, 4).cg_mode == modes.FULL

    batch = None
    runner.gather_batch_req_state(object(), True)
    assert not runner.cudagraph_manager.flashmla_has_prefill
