# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU zeroer assembly smoke using a supplied paired vLLM worker/utils.py.

Only the device zero launch is simulated; production metadata construction
comes from the supplied vLLM source and this checkout.
"""

import argparse
import ast
import dataclasses
import importlib.util
import itertools
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


class _ZeroLaunch:
    def __init__(self, raw):
        self.raw = raw

    def __getitem__(self, grid):
        def launch(segment_addresses, segment_strides, segment_sizes, block_ids, **kwargs):
            for addr, stride, size in zip(segment_addresses.tolist(), segment_strides.tolist(), segment_sizes.tolist()):
                for block in block_ids.tolist():
                    start = addr + block * stride * 4 - self.raw.data_ptr()
                    self.raw[start : start + size * 4] = 0

        return launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_worker_utils", help="Path to paired vLLM v1/worker/utils.py")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        "page_tests", root / "tests/ut/worker/v2/test_hybrid_state_page_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = module.api.__wrapped__()
    api["SimpleNamespace"] = SimpleNamespace
    paired = types.ModuleType("pr13_paired_zeroer")
    sys.modules[paired.__name__] = paired
    paired.__dict__.update(api)
    paired.__dict__.update(dataclass=dataclasses.dataclass, field=dataclasses.field, iprod=itertools.product)
    tree = ast.parse(Path(arguments.vllm_worker_utils).read_text())
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("KVBlockZeroer", "AttentionGroup")]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.Module(body=[future, *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), arguments.vllm_worker_utils, "exec"), paired.__dict__)
    api["KVBlockZeroer"] = paired.KVBlockZeroer
    api["replace"] = dataclasses.replace
    class_tree = ast.parse((root / "vllm_ascend/worker/v2/utils.py").read_text())
    node = next(n for n in class_tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendV2KVBlockZeroer")
    code.body = [future, node]
    exec(compile(ast.fix_missing_locations(code), "vllm_ascend/worker/v2/utils.py", "exec"), api)
    for ratio in (1, 3):
        raw = torch.full((64 + 5 * 192 + 64,), 19, dtype=torch.uint8)
        view = api["make_page_strided_cache_view"](raw[64:-64], (5 * ratio, 4, 1, 4), torch.bfloat16, 192 // ratio)
        group = paired.AttentionGroup(
            backend=object,
            layer_names=["mla"],
            kv_cache_spec=module.AttentionSpec(block_size=4 * ratio),
            kv_cache_group_id=0,
        )
        zeroer = api["AscendV2KVBlockZeroer"](
            torch.device("cpu"), [group], [4], {"mla": SimpleNamespace(kv_cache=view)}, num_blocks=5, cache_dtype="auto"
        )
        addresses, strides, sizes, *_ = zeroer._zeroers[0]._meta
        assert addresses.tolist() == [view.data_ptr() + i * (192 // ratio) for i in range(ratio)]
        assert strides.tolist() == [192 // 4] * ratio
        assert sizes.tolist() == [32 // 4] * ratio

        paired._zero_kv_blocks_kernel = _ZeroLaunch(raw)
        paired.async_tensor_h2d = lambda ids, device, dtype: torch.tensor(ids, device=device, dtype=dtype)
        before = raw.clone()
        zeroer.zero_block_ids([3])
        allowed = torch.zeros_like(raw, dtype=torch.bool)
        for kernel in range(ratio):
            start = 64 + 3 * 192 + kernel * (192 // ratio)
            allowed[start : start + 32] = True
        assert torch.all((raw == before) | allowed)
        assert torch.all(raw[allowed] == 0)
    print(
        "PASS: supplied upstream AttentionGroup + V2/upstream zeroer; "
        "offset64, ratios1/3, payload-only zero/guard checks; CPU device launch shim"
    )


if __name__ == "__main__":
    main()
