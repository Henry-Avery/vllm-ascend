# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External FlashMLA metadata/main adapter for MRV2 eager and ACL graphs.

The runner owns the cache. This adapter keeps its views and page identities;
only metadata is materialized. Captured inputs keep stable addresses, while the
worker updates their contents and schedule outside model graphs before replay.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, DeviceMetadataTask

FLASH_MLA_BLOCK_SIZE = 128
FLASH_MLA_QK_DIM = 576
FLASH_MLA_V_DIM = 512
FLASH_MLA_MASK_SIZE = 2048


@dataclass
class FlashMLAMetadata:
    num_tokens: int
    num_heads: int
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    token_live: torch.Tensor
    live_boundaries: torch.Tensor
    positions: torch.Tensor
    attn_mask: torch.Tensor | None
    causal: bool
    is_prefill: bool
    graph_buffer: bool = False


def init_flash_mla_metadata(builder, impl) -> None:
    if impl.num_heads not in MLA_FLASH_SUPPORTED_Q_HEADS or impl.num_kv_heads != 1:
        raise ValueError("FlashMLA requires actual local Q heads in {8, 12, 64, 96} and one KV head")
    builder.flash_num_heads = impl.num_heads
    builder._flash_buffers = {}
    builder._flash_capture = False
    builder._device_metadata_enabled = False
    builder._device_metadata_tasks = ()
    builder.flash_attn_mask = torch.triu(
        torch.ones((FLASH_MLA_MASK_SIZE, FLASH_MLA_MASK_SIZE), dtype=torch.int8, device=builder.device), diagonal=1
    )


def _flash_schedule(flash: FlashMLAMetadata, *, meta: bool = False):
    # Use the installed package's Meta schema, never guess schedule capacity.
    from cann_ops_transformer.ops import flash_mla_with_kvcache_metadata

    def tensor(value):
        return torch.empty_like(value, device="meta") if meta else value

    return flash_mla_with_kvcache_metadata(
        tensor(flash.cache_lens),
        flash.num_heads,
        1,
        cu_seqlens_q=tensor(flash.cu),
        seqused_q=tensor(flash.used_q),
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        head_dim_qk=FLASH_MLA_QK_DIM,
        head_dim_v=FLASH_MLA_V_DIM,
        mask_mode=3 if flash.causal else 0,
        layout_q="TND",
    )


def build_flash_mla_metadata(builder, common) -> FlashMLAMetadata:
    batch = common.num_reqs
    tokens = max(common.num_actual_tokens, common.num_input_tokens)
    table = common.block_table_tensor[:batch]
    is_prefill = common.max_query_len > builder.decode_threshold
    key = (batch, tokens, table.shape[1], common.causal, is_prefill)
    # Retain only explicitly captured shapes. Multi-query/causal DSpark also
    # needs stable buffers; max_query_len alone cannot identify eager prefill.
    flash = builder._flash_buffers.get(key)
    if flash is None:
        args = {"dtype": torch.int32, "device": builder.device}
        flash = FlashMLAMetadata(
            num_tokens=tokens,
            num_heads=builder.flash_num_heads,
            schedule=torch.empty(0, **args),
            cu=torch.zeros(batch + 2, **args),
            used_q=torch.zeros(batch + 1, **args),
            cache_lens=torch.zeros(batch + 1, **args),
            block_table=torch.zeros((batch + 1, table.shape[1]), **args),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=builder.device),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=builder.device),
            live_boundaries=torch.zeros(tokens + 1, **args),
            positions=torch.zeros(tokens, dtype=torch.int64, device=builder.device),
            attn_mask=builder.flash_attn_mask if common.causal else None,
            causal=common.causal,
            is_prefill=is_prefill,
            graph_buffer=builder._flash_capture,
        )
        if flash.graph_buffer:
            schedule_meta = _flash_schedule(flash, meta=True)
            if schedule_meta.device.type != "meta" or schedule_meta.dtype != torch.int32 or schedule_meta.ndim != 1:
                raise RuntimeError("FlashMLA graphs require the package's 1-D INT32 metadata Meta implementation")
            if schedule_meta.numel() == 0:
                raise RuntimeError("FlashMLA metadata Meta returned an empty schedule capacity")
            flash.schedule = torch.empty_like(schedule_meta, device=builder.device)
            builder._flash_buffers[key] = flash

    def update() -> None:
        flash.cu[: batch + 1].copy_(common.query_start_loc[: batch + 1])
        flash.cu[-1].fill_(tokens)
        flash.used_q[:batch].copy_(flash.cu[1 : batch + 1] - flash.cu[:batch])
        flash.used_q[:batch].masked_fill_(common.seq_lens[:batch] <= 0, 0)
        flash.used_q[batch:].zero_()
        flash.cache_lens[:batch].copy_(common.seq_lens[:batch])
        flash.cache_lens[batch:].zero_()
        flash.block_table[:batch].copy_(table)
        flash.block_table[batch:].zero_()
        flash.live_boundaries.zero_()
        live_rows = (flash.used_q > 0).to(torch.int32)
        flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live_rows)
        flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live_rows)
        flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
        flash.slots.fill_(-1)
        slots = common.slot_mapping[:tokens]
        flash.slots[: slots.shape[0]].copy_(slots)
        flash.slots.masked_fill_(~flash.token_live, -1)
        flash.positions.zero_()
        positions = common.positions[:tokens]
        flash.positions[: positions.shape[0]].copy_(positions)
        schedule = _flash_schedule(flash)
        if flash.graph_buffer:
            if schedule.shape != flash.schedule.shape or schedule.dtype != flash.schedule.dtype:
                raise RuntimeError(
                    "FlashMLA runtime schedule differs from captured Meta capacity; cannot replay safely"
                )
            flash.schedule.copy_(schedule)
        else:
            flash.schedule = schedule

    if builder._device_metadata_enabled:
        builder._device_metadata_tasks = (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, update, id(builder)),)
    else:
        if flash.graph_buffer:
            raise RuntimeError("Captured FlashMLA buffers require the MRV2 device metadata executor")
        update()
    return flash


def validate_flash_graph_metadata(attn_metadata) -> None:
    """Never replay using newly allocated inputs instead of captured addresses."""
    for metadata in (attn_metadata or {}).values():
        flash = getattr(metadata, "flash", None)
        if flash is not None and not flash.graph_buffer:
            raise RuntimeError("No captured FlashMLA metadata bucket matches this replay")


def validate_flash_cache(cache: torch.Tensor) -> None:
    """Validate the physical view without repacking or changing its ownership."""
    if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or cache.shape[1:] != (128, 1, 576):
        raise ValueError("FlashMLA requires token-fused PA_BBND cache [P,128,1,576]")
    page_span = (FLASH_MLA_BLOCK_SIZE - 1) * cache.stride(1) + FLASH_MLA_QK_DIM
    if cache.stride(-1) != 1 or cache.stride(1) < FLASH_MLA_QK_DIM or cache.stride(0) < page_span:
        raise ValueError("FlashMLA requires non-overlapping BBND pages and contiguous channels")
    if cache.dtype != torch.bfloat16:
        raise ValueError("This FlashMLA integration requires unquantized BF16 cache")


def run_flash_mla(query: torch.Tensor, cache: torch.Tensor, flash: FlashMLAMetadata, scale: float):
    from cann_ops_transformer.ops import flash_mla_with_kvcache

    validate_flash_cache(cache)
    if query.shape != (flash.num_tokens, flash.num_heads, FLASH_MLA_QK_DIM):
        raise ValueError("Actual Q shape does not match FlashMLA metadata geometry")
    if query.dtype != cache.dtype or query.device != cache.device:
        raise ValueError("FlashMLA Q and cache must have the same BF16 dtype and device")
    return flash_mla_with_kvcache(
        query,
        cache,
        block_table=flash.block_table,
        cache_seqlens=flash.cache_lens,
        cu_seqlens_q=flash.cu,
        seqused_q=flash.used_q,
        attn_mask=flash.attn_mask,
        metadata=flash.schedule,
        head_dim_v=FLASH_MLA_V_DIM,
        softmax_scale=scale,
        mask_mode=3 if flash.causal else 0,
        max_seqlen_q=-1,
        max_seqlen_kv=-1,
        layout_q="TND",
        layout_kv="PA_BBND",
        layout_out="NTD",
        return_softmax_lse=False,
    )
