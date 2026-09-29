# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Byte ownership and semantic layout of page-strided transfer buffers."""

from dataclasses import dataclass

import torch

FUSED_MLA_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class MooncakeLayerLayout:
    kind: str
    dtypes: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    strides: tuple[tuple[int, ...], ...]


def block_is_contiguous(cache: torch.Tensor) -> bool:
    """Allow gaps between blocks, never inside a transferred payload."""
    expected_stride = 1
    for size, stride in zip(reversed(cache.shape[1:]), reversed(cache.stride()[1:])):
        if size > 1 and stride != expected_stride:
            return False
        expected_stride *= size
    return True


def describe_layer_layout(kind: str, caches: tuple[torch.Tensor, ...]) -> MooncakeLayerLayout:
    return MooncakeLayerLayout(
        kind=kind,
        dtypes=tuple(str(cache.dtype) for cache in caches),
        shapes=tuple(tuple(cache.shape[1:]) for cache in caches),
        strides=tuple(tuple(cache.stride()[1:]) for cache in caches),
    )
