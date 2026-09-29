# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builder-owned FlashMLA buffers and device-metadata tasks.

Follows the reference's stable-buffer lifecycle, using the external package's
Meta implementation and the existing executor. It never owns persistent KV.
"""

from dataclasses import dataclass, replace

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.attention.flashmla import (
    FLASHMLA_MASK_SIZE,
    FLASHMLA_QK_DIM,
    FLASHMLA_V_DIM,
    FlashMLAAdapter,
    FlashMLAConfig,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@dataclass
class FlashMLADecode:
    adapter: FlashMLAAdapter
    query: torch.Tensor
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    token_live: torch.Tensor
    live_boundaries: torch.Tensor
    attn_mask: torch.Tensor | None
    cos: torch.Tensor | None
    sin: torch.Tensor | None


class FlashMLAMetadataBuilder:
    def __init__(self, impl, device: torch.device, max_num_reqs: int):
        adapter = FlashMLAAdapter.load(FlashMLAConfig(impl.num_heads, impl.scale))
        self.adapters = {True: adapter, False: replace(adapter, config=replace(adapter.config, mask_mode=0))}
        self.device = device
        self.dtype = impl.dtype
        self.use_rope = impl.use_mla_rope
        self.max_num_reqs = max_num_reqs
        self.buffers: dict[tuple, FlashMLADecode] = {}
        self.attn_mask = torch.triu(
            torch.ones((FLASHMLA_MASK_SIZE, FLASHMLA_MASK_SIZE), dtype=torch.int8, device=device), diagonal=1
        )
        self.defer = False
        self.executor: DeviceMetadataExecutor | None = None
        self.tasks: tuple[DeviceMetadataTask, ...] = ()
        self.logged_metadata = False
        self.trace_enabled = envs.VLLM_ASCEND_FLASH_MLA_TRACE
        self.trace_step = 0

    def _trace(self, event: str, step: int, flash: FlashMLADecode, common, num_decodes: int, has_prefill: bool):
        # Inspect host-side tensor metadata only: no .cpu(), .item(), value
        # formatting or device synchronization in this diagnostic path.
        logger.info(
            "[FlashMLA TRACE] event=%s builder=%s step=%s device=%s deferred=%s "
            "decodes=%s mixed=%s tokens=%s rows=%s mask=%s "
            "source_lengths_ptr=%s source_cu_ptr=%s lengths_ptr=%s cu_ptr=%s "
            "used_q_ptr=%s table_ptr=%s schedule_ptr=%s query_ptr=%s slots_ptr=%s",
            event,
            id(self),
            step,
            flash.query.device,
            self.defer,
            num_decodes,
            has_prefill,
            flash.query.shape[0],
            flash.cache_lens.shape[0],
            flash.adapter.config.mask_mode,
            common.seq_lens.data_ptr(),
            common.query_start_loc.data_ptr(),
            flash.cache_lens.data_ptr(),
            flash.cu.data_ptr(),
            flash.used_q.data_ptr(),
            flash.block_table.data_ptr(),
            flash.schedule.data_ptr(),
            flash.query.data_ptr(),
            flash.slots.data_ptr(),
        )

    def _allocate(self, tokens: int, rows: int, columns: int, causal: bool) -> FlashMLADecode:
        ints = {"dtype": torch.int32, "device": self.device}
        adapter = self.adapters[causal]
        cu = torch.zeros(rows + 1, **ints)
        used = torch.zeros(rows, **ints)
        lengths = torch.zeros(rows, **ints)
        schedule_meta = adapter.build_metadata(
            torch.empty_like(lengths, device="meta"),
            torch.empty_like(cu, device="meta"),
            torch.empty_like(used, device="meta"),
        )
        rope_shape = (tokens, 1, 1, FLASHMLA_QK_DIM - FLASHMLA_V_DIM)
        return FlashMLADecode(
            adapter=adapter,
            query=torch.empty(
                (tokens, adapter.config.num_heads, FLASHMLA_QK_DIM), dtype=self.dtype, device=self.device
            ),
            schedule=torch.empty_like(schedule_meta, device=self.device),
            cu=cu,
            used_q=used,
            cache_lens=lengths,
            block_table=torch.zeros((rows, columns), **ints),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=self.device),
            positions=torch.zeros(tokens, dtype=torch.int64, device=self.device),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=self.device),
            live_boundaries=torch.zeros(tokens + 1, **ints),
            attn_mask=self.attn_mask if causal else None,
            cos=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
            sin=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
        )

    def build(
        self, common, num_decodes: int, num_decode_tokens: int, has_prefill: bool, *, retain_for_graph: bool = False
    ) -> FlashMLADecode:
        # Mixed batches run outside FULL graph. Do not retain every prefill
        # shape. Retain only explicitly captured decode capacities: retaining
        # every eager token count grows total query storage quadratically.
        # Replay (and eager with that capacity) reuses the captured addresses.
        tokens = num_decode_tokens if has_prefill else max(common.num_actual_tokens, common.num_input_tokens)
        rows = num_decodes + 1 if has_prefill else max(num_decodes, min(tokens, self.max_num_reqs)) + 1
        columns = common.block_table_tensor.shape[1]
        key = tokens, rows, columns, common.causal
        if has_prefill:
            if retain_for_graph:
                raise ValueError("FlashMLA graph buffers require a decode-only batch")
            flash = self._allocate(tokens, rows, columns, common.causal)
        else:
            flash = self.buffers.get(key)
            if flash is None:
                flash = self._allocate(tokens, rows, columns, common.causal)
                if retain_for_graph:
                    self.buffers[key] = flash

        trace_step = 0
        if self.trace_enabled:
            self.trace_step += 1
            trace_step = self.trace_step
            self._trace("metadata_prepared", trace_step, flash, common, num_decodes, has_prefill)

        def refresh():
            if self.trace_enabled:
                self._trace("metadata_refresh_begin", trace_step, flash, common, num_decodes, has_prefill)
            # All refreshes belong to the executor task, after its reuse fence.
            # The extra zero-used request owns physical graph/SP padding.
            flash.cu.fill_(tokens)
            flash.cu[: num_decodes + 1].copy_(common.query_start_loc[: num_decodes + 1].clamp_max(num_decode_tokens))
            flash.used_q.zero_()
            flash.used_q[:num_decodes].copy_(flash.cu[1 : num_decodes + 1] - flash.cu[:num_decodes])
            flash.used_q[:num_decodes].masked_fill_(common.seq_lens[:num_decodes] <= 0, 0)
            flash.cache_lens.zero_()
            flash.cache_lens[:num_decodes].copy_(common.seq_lens[:num_decodes])
            flash.cache_lens.masked_fill_(flash.used_q == 0, 0)
            flash.block_table.zero_()
            flash.block_table[:num_decodes].copy_(common.block_table_tensor[:num_decodes])
            flash.slots.fill_(-1)
            slots = common.slot_mapping[:num_decode_tokens]
            flash.slots[: slots.numel()].copy_(slots)
            flash.live_boundaries.zero_()
            live = (flash.used_q > 0).to(torch.int32)
            flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live)
            flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live)
            flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
            flash.slots.masked_fill_(~flash.token_live, -1)
            flash.positions.zero_()
            positions = common.positions[:num_decode_tokens]
            flash.positions[: positions.numel()].copy_(positions)
            if self.use_rope:
                cos, sin = get_cos_and_sin_mla(flash.positions)
                flash.cos.copy_(cos)
                flash.sin.copy_(sin)
            flash.schedule.copy_(flash.adapter.build_metadata(flash.cache_lens, flash.cu, flash.used_q))
            if self.trace_enabled:
                # Enqueue success is not device completion or a numerical check.
                self._trace("metadata_enqueued", trace_step, flash, common, num_decodes, has_prefill)
            if not self.logged_metadata:
                logger.info("[FlashMLA] metadata_external: device schedule refreshed; deferred=%s", self.defer)
                self.logged_metadata = True

        self.tasks = (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, refresh, id(flash.schedule)),)
        if not self.defer:
            refresh()
            self.tasks = ()
        return flash

    def take_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks, self.tasks = self.tasks, ()
        return tasks
