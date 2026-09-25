# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External FlashMLA for the MRV2 target's unquantized BBND decode cache."""

from dataclasses import dataclass
from typing import Any

import torch

SUPPORTED_HEADS = (8, 12, 64, 96)
HEAD_DIM_QK = 576
HEAD_DIM_V = 512
BLOCK_SIZE = 128
MASK_SIZE = 2048


def validate_flash_mla_config(impl):
    config = impl.vllm_config
    parallel = config.parallel_config
    if not config.use_v2_model_runner:
        raise ValueError("External FlashMLA requires MRV2")
    if config.speculative_config is not None or impl.is_draft_model:
        raise ValueError("External FlashMLA currently requires speculative decoding disabled")
    if parallel.decode_context_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("External FlashMLA currently requires DCP=PCP=1")
    if parallel.enable_dbo:
        raise ValueError("External FlashMLA metadata requires a single model execution stream")
    if impl.num_heads not in SUPPORTED_HEADS or impl.num_kv_heads != 1:
        raise ValueError("External FlashMLA requires 8/12/64/96 local query heads and one KV head")
    if (impl.kv_lora_rank, impl.qk_rope_head_dim) != (HEAD_DIM_V, HEAD_DIM_QK - HEAD_DIM_V):
        raise ValueError("External FlashMLA requires 512 latent + 64 positional lanes")
    if impl.fa_quant_layer or impl.enable_kv_nz or impl.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("External FlashMLA requires unquantized FP16/BF16 BBND cache")


@dataclass
class FlashMLAMetadata:
    cache_lens: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    block_table: torch.Tensor
    token_live: torch.Tensor
    schedule: torch.Tensor | None
    attn_mask: torch.Tensor | None
    num_heads: int
    mask_mode: int
    execution_stream: Any = None

    def make_schedule(self, metadata_op, *, meta=False):
        def tensor(value):
            return torch.empty_like(value, device="meta") if meta else value

        return metadata_op(
            tensor(self.cache_lens),
            self.num_heads,
            1,
            cu_seqlens_q=tensor(self.cu),
            seqused_q=tensor(self.used_q),
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            head_dim_qk=HEAD_DIM_QK,
            head_dim_v=HEAD_DIM_V,
            mask_mode=self.mask_mode,
            layout_q="TND",
        )


class FlashMLABuilder:
    """Refresh stable graph inputs outside capture, on the model execution stream.

    MRV2 calls build_attn_metadata before both forward and FULL replay. Using
    that same stream orders input preparation -> metadata -> consumer -> next
    update, including reuse of a bucket. No CPU sequence-length mirror is used.
    Capture preparation also happens outside the graph; the capture context
    joins the preparation stream before reading these buffers.
    """

    def __init__(self, num_heads, device):
        # Worker-only import: the package registers its own kernels and Meta.
        from cann_ops_transformer import flash_mla_with_kvcache_metadata

        if num_heads not in SUPPORTED_HEADS:
            raise ValueError("FlashMLA requires 8/12/64/96 actual local query heads")
        self.metadata_op = flash_mla_with_kvcache_metadata
        self.num_heads = num_heads
        self.device = device
        self.buffers = {}
        self.stream = torch.npu.current_stream()
        self.mask = torch.triu(torch.ones((MASK_SIZE, MASK_SIZE), dtype=torch.int8, device=device), diagonal=1)

    def build(self, common, metadata, num_actual_reqs):
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("FlashMLA metadata must be refreshed outside graph capture")
        stream = torch.npu.current_stream()
        if stream != self.stream:
            # vLLM prepares capture inputs on its capture stream, then switches
            # back to the execution stream. Join the previous consumers before
            # any stable buffer can be overwritten on the new stream.
            stream.wait_stream(self.stream)
            self.stream = stream
        batch = metadata.num_decodes
        # Pure decode FULL graphs use padded input capacity, including when
        # replay has fewer live requests than capture. Mixed prefill is eager.
        tokens = metadata.num_decode_tokens if metadata.num_prefills else common.num_input_tokens
        table = common.block_table_tensor[:batch]
        if any(t.dtype != torch.int32 for t in (table, common.query_start_loc, common.seq_lens)):
            raise ValueError("FlashMLA requires int32 page tables and device lengths")
        mask_mode = 3 if common.causal else 0
        key = (batch, tokens, table.shape[1], mask_mode)
        flash = self.buffers.get(key)
        if flash is None:
            flash = FlashMLAMetadata(
                cache_lens=torch.empty(batch, dtype=torch.int32, device=self.device),
                cu=torch.empty(batch + 1, dtype=torch.int32, device=self.device),
                used_q=torch.empty(batch, dtype=torch.int32, device=self.device),
                block_table=torch.empty_like(table),
                token_live=torch.empty(tokens, dtype=torch.bool, device=self.device),
                schedule=None,
                attn_mask=self.mask if mask_mode else None,
                num_heads=self.num_heads,
                mask_mode=mask_mode,
            )
            # Never infer capacity from core count or a previous batch. Require
            # the external package's Meta contract and retain its exact shape.
            schedule_meta = flash.make_schedule(self.metadata_op, meta=True)
            if schedule_meta.dtype != torch.int32 or schedule_meta.ndim != 1:
                raise ValueError("FlashMLA Meta must return a one-dimensional int32 schedule")
            flash.schedule = torch.empty_like(schedule_meta, device=self.device)
            self.buffers[key] = flash
        live = torch.arange(batch, device=self.device) < min(batch, num_actual_reqs)
        flash.cu.copy_(common.query_start_loc[: batch + 1])
        flash.used_q.copy_(torch.where(live, flash.cu[1:] - flash.cu[:-1], 0))
        flash.cache_lens.copy_(torch.where(live, common.seq_lens[:batch], 0))
        flash.block_table.copy_(table)
        flash.token_live.copy_(torch.arange(tokens, device=self.device) < metadata.num_decode_tokens)
        schedule = flash.make_schedule(self.metadata_op)
        if schedule.shape != flash.schedule.shape or schedule.dtype != flash.schedule.dtype:
            raise ValueError("FlashMLA runtime schedule disagrees with Meta capacity/dtype")
        flash.schedule.copy_(schedule)
        flash.execution_stream = stream
        return flash


def flash_mla_decode(q_nope, q_pe, cache, flash, scale):
    from cann_ops_transformer import flash_mla_with_kvcache

    if torch.npu.current_stream() != flash.execution_stream:
        raise RuntimeError("FlashMLA consumer must run on its metadata preparation stream")

    if not isinstance(cache, torch.Tensor) or cache.ndim != 4:
        raise ValueError("FlashMLA requires the original fused BBND cache from the runner")
    if tuple(cache.shape[1:]) != (BLOCK_SIZE, 1, HEAD_DIM_QK):
        raise ValueError("FlashMLA requires [pages, 128, 1, 576] BBND cache")
    # Axis 0/1 may have gaps; never reshape/compact/rebase the cache. Accept
    # nonzero offsets and retain the exact object passed by the allocator.
    if cache.stride(-1) != 1 or cache.stride(1) < HEAD_DIM_QK:
        raise ValueError("FlashMLA requires packed head lanes and nonoverlapping token rows")
    if cache.stride(0) < (BLOCK_SIZE - 1) * cache.stride(1) + HEAD_DIM_QK:
        raise ValueError("FlashMLA requires nonoverlapping physical pages")
    tokens = q_nope.shape[0]
    q = torch.cat(
        (q_nope.reshape(tokens, flash.num_heads, HEAD_DIM_V), q_pe.reshape(tokens, flash.num_heads, 64)),
        dim=-1,
    )
    if q.dtype != cache.dtype or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FlashMLA query and cache must have the same FP16/BF16 dtype")
    latent, _ = flash_mla_with_kvcache(
        q,
        cache,
        block_table=flash.block_table,
        cache_seqlens=flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        attn_mask=flash.attn_mask,
        metadata=flash.schedule,
        head_dim_v=HEAD_DIM_V,
        softmax_scale=scale,
        mask_mode=flash.mask_mode,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="NTD",
        return_softmax_lse=False,
    )
    if tuple(latent.shape) != (flash.num_heads, tokens, HEAD_DIM_V) or latent.dtype != q.dtype:
        raise ValueError("FlashMLA must return NTD latent output with the query dtype")
    # Padding output is unspecified by the operator (and may be NaN). Select
    # zeros rather than multiply by zero before the existing V/gate/O path.
    return latent.masked_fill(~flash.token_live[:tokens][None, :, None], 0)
