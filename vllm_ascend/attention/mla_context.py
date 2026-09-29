# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-built query selection for nonempty MLA history chunks."""

from dataclasses import dataclass
from itertools import accumulate

import torch


@dataclass
class HistoryQuerySelection:
    query_indices: torch.Tensor | None
    cu_query_lengths: list[int]
    cu_kv_lengths: list[int]


def build_history_query_selections(
    query_lengths: list[int], chunk_lengths: list[list[int]], device: torch.device | str
) -> list[HistoryQuerySelection]:
    """Exclude empty history rows before FIA, as in the Plan1 prefill path.

    Lengths are CPU metadata. Build indices once per batch, not per layer, and
    keep the gathered KV order: removing zero-length segments moves no KV.
    """
    if any(length <= 0 for length in query_lengths):
        raise ValueError("History prefill queries must have positive lengths")
    starts = [0, *accumulate(query_lengths)]
    selections = []
    for lengths in chunk_lengths:
        if len(lengths) != len(query_lengths) or any(length < 0 for length in lengths):
            raise ValueError("Invalid history chunk lengths")
        active = [i for i, length in enumerate(lengths) if length > 0]
        indices = None
        if len(active) != len(query_lengths):
            indices = torch.tensor(
                [token for i in active for token in range(starts[i], starts[i + 1])],
                dtype=torch.int64,
                device=device,
            )
        selections.append(
            HistoryQuerySelection(
                indices,
                list(accumulate(query_lengths[i] for i in active)),
                list(accumulate(lengths[i] for i in active)),
            )
        )
    return selections
