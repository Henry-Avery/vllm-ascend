# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production history orchestration with CPU attention and paged loads.

No NPU kernel claim: the fake FIA deliberately poisons empty KV segments.
The reference independently attends over each request's complete logical KV.
"""

import runpy
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_flashmla_phase_contract import _class_method

ROOT = Path(__file__).resolve().parents[4]
api = runpy.run_path(str(ROOT / "vllm_ascend/attention/mla_context.py"))
build = api["build_history_query_selections"]


@pytest.mark.parametrize("histories", [[0, 129], [1, 257], [257, 1], [129, 129], [0, 0]])
@pytest.mark.parametrize("rope_dim,head_padding", [(0, 0), (2, 0), (2, 1)])
def test_history_merge_matches_dense_reference_and_skips_empty_rows(histories, rope_dim, head_padding):
    torch.manual_seed(29)
    queries = [1, 2]
    ends = list(accumulate(queries))
    q = torch.randn(sum(queries), 1, 2)
    qp = torch.randn(sum(queries), 1, rope_dim)
    kv = [torch.randn(h + n, 4 + rope_dim) for h, n in zip(histories, queries)]
    chunk_lengths = [[max(0, min(128, h - start)) for h in histories] for start in range(0, max(histories), 128)]
    # Include an entirely empty trailing chunk to exercise the no-call branch.
    chunk_lengths.append([0, 0])
    cache = torch.zeros(6, 128, 1, 4)
    rope = torch.zeros(6, 128, 1, rope_dim)
    table = torch.tensor([[0, 1, 2], [3, 4, 5]])
    for i, h in enumerate(histories):
        cache[i * 3 : (i + 1) * 3].reshape(-1, 4)[:h].copy_(kv[i][:h, :4])
        if rope_dim:
            rope[i * 3 : (i + 1) * 3].reshape(-1, rope_dim)[:h].copy_(kv[i][:h, 4:])

    def attend(query, query_pe, payload):
        score = query @ payload[:, :2].T
        if rope_dim:
            score = score + query_pe @ payload[:, 4:].T
        return score.softmax(-1) @ payload[:, 2:4], torch.logsumexp(score, -1)

    initial, initial_lse, expected = [], [], []
    for i, (h, count) in enumerate(zip(histories, queries)):
        start = 0 if i == 0 else ends[i - 1]
        for j in range(count):
            out, lse = attend(q[start + j, 0], qp[start + j, 0], kv[i][h : h + j + 1])
            initial.append(out)
            initial_lse.append(lse)
            expected.append(attend(q[start + j, 0], qp[start + j, 0], kv[i][: h + j + 1])[0])
    calls = []
    merge_modes = []

    def fia(query, keys, values, **kwargs):
        outputs, lses = [], []
        qs, ks = 0, 0
        for qe, ke in zip(kwargs["actual_seq_lengths"], kwargs["actual_seq_lengths_kv"]):
            calls.append((qe - qs, ke - ks))
            if ke == ks:  # legacy empty-segment behavior is deliberately adversarial
                outputs.append(torch.full((qe - qs, 1, 2), float("nan")))
                lses.append(torch.full((qe - qs, 1, 1), float("nan")))
            else:
                scores = torch.einsum("thd,shd->hts", query[qs:qe], keys[ks:ke])
                if "query_rope" in kwargs:
                    scores += torch.einsum("thd,shd->hts", kwargs["query_rope"][qs:qe], kwargs["key_rope"][ks:ke])
                outputs.append(torch.einsum("hts,shv->thv", scores.softmax(-1), values[ks:ke]))
                lses.append(scores.logsumexp(-1).T.unsqueeze(-1))
            qs, ks = qe, ke
        return torch.cat(outputs), torch.cat(lses)

    def merge(lses, outputs, mode):
        assert mode in (0, 1)
        merge_modes.append(mode)
        log_z = torch.logsumexp(torch.stack(lses), 0)
        out = sum(torch.exp(lse - log_z)[:, None] * value for lse, value in zip(lses, outputs))
        return out, log_z if mode == 1 else None

    def load(latent_cache, position_cache, block_table, lengths, starts, *, key, value):
        latent, position = [], []
        for row, length, start in zip(block_table, lengths.tolist(), starts.tolist()):
            latent.append(latent_cache[row].flatten(0, 1)[start : start + length])
            position.append(position_cache[row].flatten(0, 1)[start : start + length])
        key.copy_(torch.cat(latent))
        value.copy_(torch.cat(position))

    ns = dict(
        torch=torch,
        torch_npu=SimpleNamespace(npu_fused_infer_attention_score=fia, npu_attention_update=merge),
        DeviceOperator=SimpleNamespace(kv_cache_load=load),
    )
    cls = _class_method("vllm_ascend/attention/mla_v1.py", "AscendMLAImpl", "_compute_prefill_context", object, ns)
    layer = cls()
    layer.num_heads = 1
    layer.v_head_dim = layer.qk_nope_head_dim = 2
    layer.qk_rope_head_dim = rope_dim
    layer.head_padding = head_padding
    layer.scale = 1.0
    layer.fa_quant_layer = False
    layer.kv_b_proj = lambda x: (x, None)
    layer.get_context_seq_len_npu = lambda i, meta: torch.tensor(chunk_lengths[i])
    layer._reorg_kvcache = lambda latent, position, **_: (latent, position)
    chunk = SimpleNamespace(
        seq_tot=[sum(x) for x in chunk_lengths],
        starts=torch.arange(len(chunk_lengths))[:, None].expand(-1, 2) * 128,
        chunk_actual_seq_lengths_kv_list=[list(accumulate(x)) for x in chunk_lengths],
        history_query_selections=build(queries, chunk_lengths, "cpu"),
    )
    meta = SimpleNamespace(prefill=SimpleNamespace(actual_seq_lengths_q=ends, chunked_context=chunk, block_table=table))
    result, _ = layer._compute_prefill_context(
        q, qp, (cache, rope), rope_dim, meta, torch.stack(initial)[:, None], torch.stack(initial_lse)[:, None, None]
    )
    torch.testing.assert_close(result[:, 0], torch.stack(expected), rtol=1e-5, atol=1e-6)
    assert all(qn > 0 and kn > 0 for qn, kn in calls)
    if any(histories):
        assert merge_modes and set(merge_modes) == {1}
    if any(0 in lengths and sum(lengths) > 0 for lengths in chunk_lengths):
        # Without selection, the same production method consumes poisoned FIA
        # rows. This control shows the test would detect the prior behavior.
        chunk.history_query_selections = None
        chunk.seq_tot.pop()
        result, _ = layer._compute_prefill_context(
            q, qp, (cache, rope), rope_dim, meta, torch.stack(initial)[:, None], torch.stack(initial_lse)[:, None, None]
        )
        assert not torch.isfinite(result).all()


@pytest.mark.parametrize("queries,chunks", [([0], [[1]]), ([1], [[-1]]), ([1, 1], [[1]])])
def test_invalid_history_metadata(queries, chunks):
    with pytest.raises(ValueError):
        build(queries, chunks, "cpu")


@pytest.mark.parametrize("flash,pcp,dcp", [(True, 1, False), (False, 1, False), (True, 2, False), (True, 1, True)])
def test_metadata_builder_constructs_selection_from_prefill_rows(monkeypatch, flash, pcp, dcp):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda t: t)
    monkeypatch.setattr(torch.Tensor, "npu", lambda t: t, raising=False)
    ns = dict(
        torch=torch,
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash),
        enable_dcp=lambda: dcp,
        ChunkedContextMetadata=SimpleNamespace,
        build_history_query_selections=build,
        round_down=lambda x, y: x // y * y,
        cdiv=lambda x, y: (x + y - 1) // y,
    )
    cls = _class_method(
        "vllm_ascend/attention/mla_v1.py", "AscendMLAMetadataBuilder", "build_chunked_metadata", object, ns
    )
    builder = cls()
    builder.chunked_prefill_enabled = True
    builder.seq_lens = torch.tensor([41, 1, 131])
    builder.query_lens = torch.tensor([1, 1, 2])
    builder.num_decodes, builder.num_prefills = 1, 2
    builder.pcp_size = pcp
    builder.chunked_prefill_workspace_size = 128
    builder.block_size = 128
    builder.device = "cpu"
    builder.chunked_prefill_workspace = None
    meta = builder.build_chunked_metadata(0, SimpleNamespace(num_reqs=3))
    if not flash or pcp > 1 or dcp:
        assert meta.history_query_selections is None
        return
    assert [s.query_indices.tolist() for s in meta.history_query_selections] == [[1, 2], [1, 2]]
    assert [s.cu_kv_lengths for s in meta.history_query_selections] == [[128], [1]]
