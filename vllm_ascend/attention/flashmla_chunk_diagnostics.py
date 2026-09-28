# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager observations of real MLA inputs, with bounded CPU references.

No device tensor is modified. CPU copies deliberately synchronize the selected
worker. A skipped reference or exhausted budget is coverage loss, never a pass.
This module has no production imports so its references can be tested on CPU.
"""

import inspect
import json
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps

import torch

MIB = 1024 * 1024
MAX_REQUESTS = 128
MAX_WORK_BYTES = 64 * MIB
BLOCK_SIZE = 128
MAX_SUCCESS_ELEMENTS = 2048
MAX_REFERENCE_HEADS = 4
MAX_HIDDEN_SIZE = 32768
MAX_GUARD_SLOTS = 32
MAX_PADDING_TOKENS = 65536


@dataclass(frozen=True)
class ChunkDiagnosticConfig:
    max_layers: int = 2
    requests_per_phase: int = 2
    query_rows: int = 2
    max_kv_tokens: int = 4096
    max_saved_mib: int = 128
    layer_names: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    start_after_forwards: int = 0
    scan_all_mla_layers: bool = False
    atol: float = 0.05
    rtol: float = 0.05

    def __post_init__(self):
        bounds = {
            "max_layers": (1, 8),
            "requests_per_phase": (1, 8),
            "query_rows": (1, 4),
            "max_kv_tokens": (1, 16384),
            "max_saved_mib": (1, 1024),
            "start_after_forwards": (0, 1000000),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Chunk diagnostics require {name}={low}..{high}")
        if type(self.scan_all_mla_layers) is not bool:
            raise ValueError("scan_all_mla_layers must be a JSON boolean")
        for name, limit in (("layer_names", 8), ("request_ids", 16)):
            values = getattr(self, name)
            if (
                not isinstance(values, (tuple, list))
                or len(values) > limit
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"Chunk diagnostics require at most {limit} nonempty {name}")
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Chunk diagnostics require finite {name}=0..1")

    @classmethod
    def from_json(cls, value):
        fields = json.loads(value)
        if not isinstance(fields, dict):
            raise ValueError("Chunk diagnostic config must be a JSON object")
        return cls(**fields)


def cpu(value):
    return value.detach().to(device="cpu").clone()


def reference_kv(raw, weight, epsilon, use_rope, cos=None, sin=None):
    """Independent FP32 RMSNorm and interleaved RoPE; no cache writes."""
    width = weight.numel()
    latent, positional = raw[..., :width].float(), raw[..., width:].float()
    latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + epsilon) * weight.float()
    if use_rope and positional.shape[-1]:
        if positional.shape[-1] % 2 or cos is None or sin is None:
            raise ValueError("RoPE reference requires even width and explicit cos/sin")
        even, odd = positional[..., ::2], positional[..., 1::2]
        positional = torch.cat((even, odd), -1) * cos.float() + torch.cat((-odd, even), -1) * sin.float()
    return latent, positional


def snapshot_request_sources(runner, scheduler):
    """Read identity from the registry before batch sorting, not idx_mapping."""
    states = runner.req_states
    result = []
    for req, scheduled in scheduler.num_scheduled_tokens.items():
        index = states.req_id_to_index.get(req)
        index = int(index) if index is not None else None
        result.append(
            dict(
                request_id=req,
                scheduled=int(scheduled),
                index=index,
                reverse_request_id=states.index_to_req_id.get(index),
                computed_prefill=int(states.num_computed_prefill_tokens[index]) if index is not None else None,
                prefill_len=int(states.prefill_len.np[index]) if index is not None else None,
            )
        )
    return result


def tensor_layout(value):
    return dict(
        shape=list(value.shape), stride=list(value.stride()), offset=value.storage_offset(), dtype=str(value.dtype)
    )


def canonical_lse(value, tokens, heads):
    if value.ndim == 2 and tuple(value.shape) == (heads, tokens):
        return value.transpose(0, 1)
    if value.ndim == 3 and tuple(value.shape) == (tokens, heads, 1):
        return value[..., 0]
    raise ValueError(f"Unrecognized LSE layout: {tuple(value.shape)} for T={tokens}, H={heads}")


def reference_attention(query, key, value, scale, *, causal_positions=None):
    """CPU FP32 reference for selected query rows; zero KV is an empty branch."""
    if key.shape[0] == 0:
        return torch.zeros((*query.shape[:2], value.shape[-1])), torch.full(query.shape[:2], -torch.inf)
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if causal_positions is not None:
        mask = torch.arange(key.shape[0])[None, :] > torch.tensor(causal_positions)[:, None]
        scores.masked_fill_(mask[None], -torch.inf)
    lse = scores.logsumexp(-1).transpose(0, 1)
    output = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value.float())
    return output, lse


def reference_merge(branches):
    """Merge softmax partitions, excluding empty (-inf LSE) contributions."""
    outputs = torch.stack([out.float() for out, _ in branches])
    lses = torch.stack([lse.float() for _, lse in branches])
    total = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - total)
    live = ~torch.isneginf(lses)
    # Empty kernels may leave output undefined; 0 * NaN is not a valid merge.
    safe_outputs = torch.where(live[..., None], outputs, 0)
    return (safe_outputs * torch.where(live, weights, 0)[..., None]).sum(0), total


def sample_rows(length, limit):
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [length - 1]
    return sorted({i * (length - 1) // (limit - 1) for i in range(limit)})


def read_paged(cache, table, start, length):
    """Independent logical page indexing, honoring tensor strides and offsets."""
    if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE:
        raise ValueError("Chunk diagnostics require a BBND block128 cache")
    if start < 0 or length < 0 or (start + length + BLOCK_SIZE - 1) // BLOCK_SIZE > len(table):
        raise ValueError("History range exceeds the supplied page table")
    positions = list(range(start, start + length))
    pages = [int(table[pos // BLOCK_SIZE]) for pos in positions]
    if any(page < 0 or page >= cache.shape[0] for page in pages):
        raise ValueError("History page ID is outside the physical cache")
    page_ids = torch.tensor(pages, dtype=torch.long, device=cache.device)
    offsets = torch.tensor([pos % BLOCK_SIZE for pos in positions], dtype=torch.long, device=cache.device)
    return cpu(cache[page_ids, offsets])


def compare_tensors(actual, expected, atol, rtol):
    passed = (
        torch.equal(actual, expected)
        if atol == rtol == 0
        else torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
        if actual.shape == expected.shape
        else False
    )
    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape:
        return dict(passed=False, reason="shape", actual=list(actual.shape), expected=list(expected.shape))
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    delta = (actual[finite] - expected[finite]).abs()
    return dict(
        passed=bool(passed),
        max_abs=float(delta.max()) if delta.numel() else None,
        max_rel=float((delta / expected[finite].abs().clamp_min(1e-6)).max()) if delta.numel() else None,
        actual_nan=int(torch.isnan(actual).sum()),
        actual_posinf=int(torch.isposinf(actual).sum()),
        actual_neginf=int(torch.isneginf(actual).sum()),
        atol=atol,
        rtol=rtol,
    )


class ChunkDiagnostics:
    def __init__(self, config, sampling_diagnostic):
        self.config = config
        self.sampling = sampling_diagnostic
        self.directory = sampling_diagnostic.directory / "attention"
        self.directory.mkdir(mode=0o700)
        self.step = 0
        self.observed_steps = 0
        self.waiting = False
        self.request_sources = None
        self.execute_calls = 0
        self.execution_events = 0
        self.saved_bytes = 0
        self.layers = []
        self.active_batch = None
        self.exhausted = False
        self.emit("armed", configuration=vars(config), max_steps=sampling_diagnostic.config.steps)

    def emit(self, event, **fields):
        record = dict(
            event=event,
            forward_id=self.step,
            utc=datetime.now(timezone.utc).isoformat(),
            **self.sampling.identity,
            **fields,
        )
        descriptor = os.open(self.directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")

    def save(self, tensors, layer_index):
        size = sum(value.numel() * value.element_size() for value in tensors.values())
        if self.saved_bytes + size > self.config.max_saved_mib * MIB:
            self.emit("snapshot_skipped", reason="save_budget", bytes=size, layer_index=layer_index)
            return None
        suffix = f"{layer_index:03d}" if isinstance(layer_index, int) else layer_index
        name = f"forward{self.step:04d}-layer{suffix}.pt"
        descriptor = os.open(self.directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            torch.save(tensors, output)
        self.saved_bytes += size
        return name

    def install(self, runner):
        # ModelState is created during load_model, after runner.__init__. Bind
        # lazily on the first real prepare_inputs; dummy/profile calls consume
        # neither the budget nor a request identity.
        original_inputs = runner.prepare_inputs
        bound_states = []
        original_gather = getattr(runner, "gather_batch_req_state", None)
        if original_gather is not None:

            @wraps(original_gather)
            def gather(*args, **kwargs):
                bound = inspect.signature(original_gather).bind(*args, **kwargs)
                bound.apply_defaults()
                scheduler = bound.arguments["scheduler_output"]
                self.request_sources = None
                if (
                    not bound.arguments["dummy_run"]
                    and len(scheduler.num_scheduled_tokens) <= MAX_REQUESTS
                    and self.step >= self.config.start_after_forwards
                    and self.observed_steps < self.sampling.config.steps
                    and (
                        not self.config.request_ids
                        or set(scheduler.num_scheduled_tokens) & set(self.config.request_ids)
                    )
                ):
                    self.request_sources = snapshot_request_sources(runner, scheduler)
                return original_gather(*args, **kwargs)

            runner.gather_batch_req_state = gather
        original_execute = getattr(runner, "execute_model", None)
        if original_execute is not None:

            @wraps(original_execute)
            def execute(*args, **kwargs):
                bound = inspect.signature(original_execute).bind(*args, **kwargs)
                bound.apply_defaults()
                self.execute_calls += 1
                # Include dummy/DP synchronization calls in a bounded host timeline.
                scheduler = bound.arguments["scheduler_output"]
                observing = (
                    self.step >= self.config.start_after_forwards
                    and self.observed_steps < self.sampling.config.steps
                    and self.execution_events < 2 * self.sampling.config.steps
                    and (
                        not self.config.request_ids
                        or set(scheduler.num_scheduled_tokens) & set(self.config.request_ids)
                        or (bound.arguments.get("dummy_run") and self.observed_steps > 0)
                    )
                )
                if observing:
                    self.execution_events += 1
                    self.emit(
                        "execute_call",
                        execute_id=self.execute_calls,
                        dummy=bool(bound.arguments.get("dummy_run", False)),
                        profile=bool(bound.arguments.get("is_profile", False)),
                        scheduled_tokens=int(scheduler.total_num_scheduled_tokens),
                    )
                with torch.profiler.record_function(f"flashmla.execute/{self.execute_calls}"):
                    return original_execute(*args, **kwargs)

            runner.execute_model = execute
        self.emit(
            "runtime",
            chunked_prefill=runner.scheduler_config.enable_chunked_prefill,
            prefix_caching=runner.cache_config.enable_prefix_caching,
            max_num_batched_tokens=runner.scheduler_config.max_num_batched_tokens,
            independent_mapping=original_gather is not None,
            writer_references=True,
            dispatch_evidence=True,
        )

        @wraps(original_inputs)
        def prepare_inputs(*args, **kwargs):
            batch = original_inputs(*args, **kwargs)
            self.step += 1
            batch.flashmla_diagnostic_id = self.step
            batch.flashmla_diagnostic_selected = False
            batch.flashmla_diagnostic_request_ids = self.config.request_ids
            self.active_batch = None
            eligible = self.step > self.config.start_after_forwards and (
                not self.config.request_ids or bool(set(batch.req_ids) & set(self.config.request_ids))
            )
            if eligible and self.observed_steps < self.sampling.config.steps:
                self.observed_steps += 1
                batch.flashmla_diagnostic_selected = True
                self.active_batch = DiagnosticBatch(self, batch)
                self.active_batch.check_request_sources(self.request_sources)
            elif eligible and not self.exhausted:
                self.emit("budget_exhausted", coverage="later forwards are unobserved")
                self.exhausted = True
            elif not eligible and not self.waiting:
                self.emit("waiting_for_window", coverage="unmatched forwards consume no capture budget")
                self.waiting = True
            state = runner.model_state
            if all(state is not previous for previous in bound_states):
                original_attn = state.prepare_attn

                @wraps(original_attn)
                def prepare_attn(*attn_args, **attn_kwargs):
                    bound = inspect.signature(original_attn).bind(*attn_args, **attn_kwargs)
                    with torch.profiler.record_function(f"flashmla.prepare_attn/{self.step}"):
                        result = original_attn(*attn_args, **attn_kwargs)
                    observed = self.active_batch
                    matches = observed is not None and bound.arguments["input_batch"] is observed.batch
                    names = []
                    for name, metadata in result.items():
                        if hasattr(metadata, "chunk_diagnostics"):
                            # Clear reused metadata on unobserved forwards too.
                            metadata.chunk_diagnostics = observed if matches else None
                            names.append(name)
                    if matches and self.config.scan_all_mla_layers:
                        self.emit("boundary_layers", layers=names)
                    return result

                state.prepare_attn = prepare_attn
                bound_states.append(state)
            return batch

        runner.prepare_inputs = prepare_inputs


class DiagnosticBatch:
    def __init__(self, owner, batch):
        if not 0 < batch.num_reqs <= MAX_REQUESTS or batch.num_draft_tokens:
            raise ValueError("Chunk diagnostics require 1..128 real requests without speculation")
        self.owner = owner
        self.batch = batch
        self.requests = list(batch.req_ids)
        self.offsets = batch.query_start_loc_np[: batch.num_reqs + 1].tolist()
        self.history = batch.num_computed_prefill_tokens_np.tolist()
        self.state_indices = batch.idx_mapping_np.tolist()
        self.seq_lens_cpu = batch.seq_lens_np[: batch.num_reqs].tolist()
        self.visited = set()
        self.boundary_dumped = False
        self.context_recorded = False
        self.owner.emit(
            "batch",
            request_ids=self.requests,
            cpu_query_offsets=self.offsets,
            computed_prefill=self.history,
            is_prefilling=batch.is_prefilling_np.tolist(),
            request_state_indices=self.state_indices,
            seq_lens_cpu=self.seq_lens_cpu,
            prefill_lengths=batch.prefill_len_np.tolist(),
            execute_id=owner.execute_calls,
            note="CPU metadata only; compare each layer's actual device inputs",
        )

    def check_request_sources(self, sources):
        if sources is None:
            self.owner.emit("check_skipped", stage="request_registry", reason="pre_sort_snapshot_unavailable")
            return
        expected = {row["request_id"]: row for row in sources}
        checks = []
        for i, req in enumerate(self.requests):
            row = expected.get(req, {})
            passed = (
                row.get("index") == self.state_indices[i]
                and row.get("reverse_request_id") == req
                and row.get("scheduled") == self.offsets[i + 1] - self.offsets[i]
                and row.get("computed_prefill") == self.history[i]
                and row.get("prefill_len") == int(self.batch.prefill_len_np[i])
                and bool(self.batch.is_prefilling_np[i]) == (self.history[i] < int(self.batch.prefill_len_np[i]))
            )
            checks.append(dict(request_id=req, passed=passed))
        phases = self.batch.is_prefilling_np.tolist()
        self.owner.emit(
            "request_registry",
            passed=all(row["passed"] for row in checks)
            and set(expected) == set(self.requests)
            and phases == sorted(phases),
            scheduler_order=sources,
            decode_before_prefill=phases == sorted(phases),
            batch_order=self.requests,
            checks=checks,
        )

    def record_context(self, extra, context):
        if self.context_recorded:
            return
        self.context_recorded = True
        dp_metadata = getattr(context, "dp_metadata", None)
        dp_tokens = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
        if dp_tokens is None:
            dp_tokens = getattr(extra, "num_tokens_across_dp", None)
        tokens = None
        if isinstance(dp_tokens, torch.Tensor) and dp_tokens.numel() <= MAX_REQUESTS:
            tokens = cpu(dp_tokens).reshape(-1).tolist()
        comm = getattr(extra, "moe_comm_type", None)
        padding = getattr(self.batch, "is_padding", None)
        padding_summary = None
        if isinstance(padding, torch.Tensor) and padding.numel() <= MAX_PADDING_TOKENS:
            padding_cpu = cpu(padding).reshape(-1)
            padding_summary = dict(
                tokens=int(padding_cpu.sum()),
                first_indices=padding_cpu.nonzero().reshape(-1)[:MAX_GUARD_SLOTS].tolist(),
            )
        self.owner.emit(
            "forward_context",
            execute_id=self.owner.execute_calls,
            num_actual_tokens=self.offsets[-1],
            num_tokens_after_padding=getattr(self.batch, "num_tokens_after_padding", None),
            padding=padding_summary,
            dp_tokens=tokens,
            moe_comm_type=getattr(comm, "name", str(comm)) if comm is not None else None,
            note="Missing fields are unknown; this is dispatch metadata, not proof of a collective kernel",
        )

    def boundary(self, name, stage, value):
        """Finite checks across MLA layers, independent of expensive references."""
        if not self.owner.config.scan_all_mla_layers:
            return
        if self.owner.sampling.capture_check():
            raise RuntimeError("Chunk diagnostics cannot execute inside graph capture")
        requests = [
            i
            for i, req in enumerate(self.requests)
            if not self.owner.config.request_ids or req in self.owner.config.request_ids
        ][:MAX_REQUESTS]
        mapping = [
            (i, self.offsets[i] + row)
            for i in requests
            for row in sample_rows(self.offsets[i + 1] - self.offsets[i], self.owner.config.query_rows)
        ]
        if value.ndim != 2 or value.shape[0] < self.offsets[-1] or not 0 < value.shape[1] <= MAX_HIDDEN_SIZE:
            self.owner.emit("check_skipped", layer=name, stage=stage, reason="boundary_rows_not_global_or_width")
            return
        selected = cpu(value[[row for _, row in mapping]])
        rows = []
        for (req, index), row in zip(mapping, selected):
            rows.append(
                dict(
                    request_id=self.requests[req],
                    token_row=index,
                    nan=int(torch.isnan(row).sum()),
                    positive_inf=int(torch.isposinf(row).sum()),
                    negative_inf=int(torch.isneginf(row).sum()),
                )
            )
        passed = bool(torch.isfinite(selected).all())
        snapshot = None
        if not passed and not self.boundary_dumped:
            snapshot = self.owner.save({"rows": selected}, "first-boundary-failure")
            self.boundary_dumped = True
        self.owner.emit("boundary", layer=name, stage=stage, passed=passed, rows=rows, tensor_file=snapshot)

    def begin_layer(self, impl, name, metadata, cache):
        owner = self.owner
        if owner.sampling.capture_check():
            raise RuntimeError("Chunk diagnostics cannot execute inside graph capture")
        if name in self.visited:
            raise RuntimeError("Chunk diagnostics observed a repeated layer in the same forward")
        self.visited.add(name)
        if owner.config.layer_names and name not in owner.config.layer_names:
            return None
        if name not in owner.layers:
            if len(owner.layers) >= owner.config.max_layers:
                return None
            owner.layers.append(name)
            owner.emit("layer_selected", layer=name, layer_index=owner.layers.index(name))
        return LayerObservation(self, impl, name, metadata, cache)


class LayerObservation:
    def __init__(self, batch, impl, name, metadata, cache):
        self.batch = batch
        self.owner = batch.owner
        self.config = self.owner.config
        self.impl = impl
        self.name = name
        self.meta = metadata
        self.cache = cache
        self.tensors = {}
        self.reference_branches = {}
        self.actual_branches = {}
        self.prefill_rows = {}
        self.failed = False
        self.skips = 0
        self.checked = 0
        self.history_starts = None
        self.history_lengths = None
        self.table = cpu(metadata.block_tables[: len(batch.requests)]).tolist()
        self.slots = cpu(metadata.slot_mapping[: metadata.num_actual_tokens]).tolist()
        self.seq_lens = cpu(metadata.seq_lens[: len(batch.requests)]).tolist()
        self.keep("seq_lens_device", torch.tensor(self.seq_lens))
        self.keep("query_offsets_device", cpu(metadata.query_start_loc[: len(batch.requests) + 1]))
        self.keep("slots", torch.tensor(self.slots))
        self.keep("block_table", torch.tensor(self.table))
        self.keep("positions", cpu(batch.batch.positions[: metadata.num_actual_tokens]))
        self.keep("request_state_indices_device", cpu(batch.batch.idx_mapping))
        self.check(
            "request_state_indices",
            self.tensors["request_state_indices_device"],
            torch.tensor(batch.state_indices),
            exact=True,
        )
        selected = []
        for start, end in ((0, metadata.num_decodes), (metadata.num_decodes, len(batch.requests))):
            candidates = [
                i
                for i in range(start, end)
                if not self.config.request_ids or batch.requests[i] in self.config.request_ids
            ]
            # Observe continued and new prefills together whenever both exist.
            positive = [i for i in candidates if batch.history[i] > 0]
            empty = [i for i in candidates if batch.history[i] == 0]
            order = ([positive.pop(0)] if positive else []) + ([empty.pop(0)] if empty else []) + positive + empty
            selected.extend(order[: self.config.requests_per_phase])
        self.selected = selected
        chunk = metadata.prefill.chunked_context if metadata.prefill is not None else None
        if chunk is not None:
            self.history_starts = cpu(chunk.starts).tolist()
            self.history_lengths = cpu(chunk.chunk_seq_lens_npu).tolist()
            self.keep("history_starts", torch.tensor(self.history_starts))
            self.keep("history_lengths_device", torch.tensor(self.history_lengths))
            self.check(
                "history_lengths_cpu_device", torch.tensor(self.history_lengths), chunk.chunk_seq_lens.cpu(), exact=True
            )
            lengths = torch.tensor(self.history_lengths)
            starts = torch.tensor(self.history_starts)
            self.check("history_total", lengths.sum(0), torch.tensor(batch.history[metadata.num_decodes :]), exact=True)
            # Zero-length partitions may start beyond a short request's history.
            self.check(
                "history_contiguous", starts[lengths > 0], (lengths.cumsum(0) - lengths)[lengths > 0], exact=True
            )
            self.check(
                "history_fia_boundaries",
                torch.tensor(chunk.chunk_actual_seq_lengths_kv_list),
                lengths.cumsum(1),
                exact=True,
            )
            self.check(
                "history_gather_table",
                cpu(metadata.prefill.block_table),
                torch.tensor(self.table[metadata.num_decodes :]),
                exact=True,
            )
        offsets = batch.offsets
        inferred = [self.seq_lens[i] - (offsets[i + 1] - offsets[i]) for i in range(len(batch.requests))]
        num_decodes = metadata.num_decodes
        self.check(
            "prefill_history_cpu_device",
            torch.tensor(inferred[num_decodes:]),
            torch.tensor(batch.history[num_decodes:]),
            exact=True,
        )
        self.check("query_offsets_cpu_device", self.tensors["query_offsets_device"], torch.tensor(offsets), exact=True)
        self.check("seq_lens_cpu_device", torch.tensor(self.seq_lens), torch.tensor(batch.seq_lens_cpu), exact=True)
        positions = [position for i in range(len(batch.requests)) for position in range(inferred[i], self.seq_lens[i])]
        self.check("positions_vs_extents", self.tensors["positions"], torch.tensor(positions), exact=True)
        if metadata.prefill is not None:
            self.check(
                "prefill_fia_boundaries",
                torch.tensor(metadata.prefill.actual_seq_lengths_q),
                torch.tensor(offsets[num_decodes + 1 :]) - offsets[num_decodes],
                exact=True,
            )
            if chunk is None:
                self.check(
                    "history_absent",
                    torch.tensor(batch.history[num_decodes:]),
                    torch.zeros(len(batch.requests) - num_decodes, dtype=torch.long),
                    exact=True,
                )
        self.emit(
            "prefill_extents",
            requests=[
                dict(
                    request_id=batch.requests[i],
                    start=batch.history[i],
                    end=batch.history[i] + offsets[i + 1] - offsets[i],
                )
                for i in range(num_decodes, len(batch.requests))
            ],
        )
        self.emit(
            "layer_begin",
            selected_requests=[batch.requests[i] for i in selected],
            num_decodes=num_decodes,
            num_decode_tokens=metadata.num_decode_tokens,
            num_prefills=metadata.num_prefills,
            history_from_device=inferred,
            cache_layouts=[tensor_layout(value) for value in cache],
            source_layouts=dict(seq=tensor_layout(metadata.seq_lens), table=tensor_layout(metadata.block_tables)),
            chunked_context=chunk is not None,
        )

    def emit(self, event, **fields):
        self.owner.emit(event, layer=self.name, **fields)

    def keep(self, name, value):
        size = value.numel() * value.element_size()
        used = sum(item.numel() * item.element_size() for item in self.tensors.values())
        if used + size <= MAX_WORK_BYTES:
            self.tensors[name] = cpu(value)
        else:
            self.skip(name, "per_layer_snapshot_bytes")

    def skip(self, stage, reason, **fields):
        self.skips += 1
        self.emit("check_skipped", stage=stage, reason=reason, **fields)

    def check(self, stage, actual, expected, *, exact=False, **fields):
        result = compare_tensors(actual, expected, 0 if exact else self.config.atol, 0 if exact else self.config.rtol)
        self.checked += 1
        self.failed |= not result["passed"]
        self.emit("comparison", stage=stage, **result, **fields)
        if result["passed"] and actual.numel() > MAX_SUCCESS_ELEMENTS:
            # Full comparison still ran; successful cache dumps retain boundary
            # samples so a long context does not consume the entire run budget.
            indices = sample_rows(actual.numel(), MAX_SUCCESS_ELEMENTS)
            self.keep(stage + "/flat_indices", torch.tensor(indices))
            self.keep(stage + "/actual", actual.reshape(-1)[indices])
            self.keep(stage + "/expected", expected.reshape(-1)[indices])
        else:
            self.keep(stage + "/actual", actual)
            self.keep(stage + "/expected", expected)
        return result["passed"]

    def allowed(self, stage, length, *tensors):
        size = sum(value.numel() * max(4, value.element_size()) for value in tensors)
        if length > self.config.max_kv_tokens or size > MAX_WORK_BYTES:
            self.skip(stage, "reference_input_budget", kv_tokens=length, estimated_bytes=size)
            return False
        return True

    @contextmanager
    def watch_writer(self, phase, raw, cos, sin, slots):
        """Freeze inputs/untouched sentinels before the real writer executes."""
        stage = f"{phase}_writer"
        prepared = self.prepare_writer_reference(phase, raw, cos, sin, slots)
        try:
            with torch.profiler.record_function(f"flashmla.writer/{self.owner.step}/{self.name}/{phase}"):
                yield
        except Exception as error:
            self.emit("writer_error", stage=stage, error_type=type(error).__name__)
            raise
        if prepared is None:
            return
        entries, guard_slots, guards = prepared
        for req, row, slot, expected in entries:
            for component, cache, reference in zip(("latent", "positional"), self.cache, expected):
                actual = cpu(cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE]).reshape(-1)
                self.check(
                    f"{stage}/{req}/{row}/{component}",
                    actual,
                    reference.reshape(-1),
                    request_id=self.batch.requests[req],
                    slot=slot,
                )
        for index, (cache, before) in enumerate(zip(self.cache, guards)):
            after = self.read_slots(cache, guard_slots)
            # Bytes preserve unchanged NaN payloads and signed zero in unused KV.
            self.check(
                f"writer_guard/{phase}/{index}",
                after.contiguous().view(torch.uint8),
                before.contiguous().view(torch.uint8),
                exact=True,
                slots=guard_slots,
            )

    @staticmethod
    def read_slots(cache, slots):
        indices = torch.tensor(slots, dtype=torch.long, device=cache.device)
        return cpu(cache[indices // BLOCK_SIZE, indices % BLOCK_SIZE])

    def prepare_writer_reference(self, phase, raw, cos, sin, slots):
        # Only sampled source rows and bounded untouched sentinels leave device.
        impl = self.impl
        stage = f"{phase}_writer"
        norm = getattr(impl, "kv_a_layernorm", None)
        if (
            norm is None
            or getattr(impl, "fa_quant_layer", False)
            or getattr(impl, "enable_kv_nz", False)
            or any(cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE or cache.shape[2] != 1 for cache in self.cache)
        ):
            self.skip(stage, "unsupported_writer_reference")
            return None
        slot_values = cpu(slots).reshape(-1).tolist()
        offset = self.meta.num_decode_tokens if phase == "prefill" else 0
        rows = [
            (req, row)
            for req in self.selected
            if (req >= self.meta.num_decodes) == (phase == "prefill")
            for row in (
                self.batch.offsets[req] + i
                for i in sample_rows(self.batch.offsets[req + 1] - self.batch.offsets[req], self.config.query_rows)
            )
        ]
        capacity = self.cache[0].shape[0] * BLOCK_SIZE
        raw_width = self.cache[0].shape[-1] + self.cache[1].shape[-1]
        if raw.ndim != 2 or raw.shape[-1] != raw_width or raw.shape[0] > len(slot_values):
            self.skip(stage, "writer_input_shape")
            return None
        weight = cpu(norm.weight).reshape(-1)
        if weight.numel() != self.cache[0].shape[-1]:
            self.skip(stage, "norm_weight_shape")
            return None
        entries = []
        for req, row in rows:
            local = row - offset
            if not 0 <= local < raw.shape[0] or not 0 <= slot_values[local] < capacity:
                self.skip(stage, "invalid_source_row_or_slot", row=row)
                continue
            source = cpu(raw[local]).reshape(1, -1)
            c = cpu(cos[local]).reshape(1, -1) if impl.use_mla_rope and cos is not None else None
            s = cpu(sin[local]).reshape(1, -1) if impl.use_mla_rope and sin is not None else None
            if (
                impl.use_mla_rope
                and self.cache[1].shape[-1]
                and (
                    c is None
                    or s is None
                    or c.shape[-1] != self.cache[1].shape[-1]
                    or s.shape != c.shape
                    or c.shape[-1] % 2
                )
            ):
                self.skip(stage, "unsupported_rope_shape")
                continue
            expected = reference_kv(source, weight, norm.variance_epsilon, impl.use_mla_rope, c, s)
            position = (
                self.seq_lens[req]
                - (self.batch.offsets[req + 1] - self.batch.offsets[req])
                + row
                - self.batch.offsets[req]
            )
            if 0 <= position // BLOCK_SIZE < len(self.table[req]):
                expected_slot = self.table[req][position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE
                self.check(
                    f"{stage}/{req}/{row}/slot",
                    torch.tensor(slot_values[local]),
                    torch.tensor(expected_slot),
                    exact=True,
                )
            else:
                self.skip(stage, "position_outside_table", request_id=self.batch.requests[req])
            entries.append((req, row, slot_values[local], expected))
            self.keep(f"{stage}/{req}/{row}/source", source)
            if c is not None:
                self.keep(f"{stage}/{req}/{row}/cos", c)
                self.keep(f"{stage}/{req}/{row}/sin", s)
        written = {int(slot) for slot in slot_values if 0 <= slot < capacity}
        candidates = {
            neighbor
            for slot in written
            for neighbor in (slot - 1, slot + 1)
            if 0 <= neighbor < capacity and neighbor not in written
        }
        written_pages = {slot // BLOCK_SIZE for slot in written}
        page_candidates = {0, self.cache[0].shape[0] - 1} | {page + 1 for page in written_pages}
        for page in sorted(page_candidates):
            if 0 <= page < self.cache[0].shape[0] and page not in written_pages:
                candidates.update(page * BLOCK_SIZE + i for i in (0, BLOCK_SIZE // 2, BLOCK_SIZE - 1))
                break
        ordered = sorted(candidates)
        guard_slots = [ordered[i] for i in sample_rows(len(ordered), MAX_GUARD_SLOTS)]
        if not guard_slots:
            self.skip("writer_guard", "no_untouched_slots")
        guards = [self.read_slots(cache, guard_slots) for cache in self.cache]
        self.emit(
            "writer_inputs",
            phase=phase,
            selected_rows=[row for _, row in rows],
            slots=slot_values,
            guard_slots=guard_slots,
            use_mla_rope=bool(impl.use_mla_rope),
            epsilon=float(norm.variance_epsilon),
            cache_layouts=[tensor_layout(cache) for cache in self.cache],
        )
        self.keep(f"{stage}/norm_weight", weight)
        return entries, guard_slots, guards

    def writer(self, latent, positional):
        num_decode_tokens = self.meta.num_decode_tokens
        for req in self.selected:
            if req < self.meta.num_decodes:
                continue
            start, end = self.batch.offsets[req : req + 2]
            rows = [start + row for row in sample_rows(end - start, self.config.query_rows)]
            for row in rows:
                slot = self.slots[row]
                position = self.batch.history[req] + row - start
                if position // BLOCK_SIZE >= len(self.table[req]):
                    self.failed = True
                    self.skip("writer", "position_outside_table", request_id=self.batch.requests[req])
                    continue
                expected_slot = self.table[req][position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE
                self.check(f"writer/{req}/{row}/slot", torch.tensor(slot), torch.tensor(expected_slot), exact=True)
                if slot < 0 or slot // BLOCK_SIZE >= self.cache[0].shape[0]:
                    self.failed = True
                    self.skip("writer", "invalid_real_slot", request_id=self.batch.requests[req], slot=slot)
                    continue
                for component, source, cache in (
                    ("latent", latent, self.cache[0]),
                    ("positional", positional, self.cache[1]),
                ):
                    expected = cpu(source[row - num_decode_tokens]).reshape(-1)
                    actual = cpu(cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE]).reshape(-1)
                    self.check(
                        f"writer/{req}/{row}/{component}",
                        actual,
                        expected,
                        exact=True,
                        request_id=self.batch.requests[req],
                        position=self.batch.history[req] + row - start,
                        slot=slot,
                    )

    def gather(self, index, latent, positional):
        lengths = self.history_lengths[index]
        starts = self.history_starts[index]
        for req in self.selected:
            local = req - self.meta.num_decodes
            if local < 0:
                continue
            length = lengths[local]
            begin = sum(lengths[:local])
            self.emit(
                "history_chunk",
                request_id=self.batch.requests[req],
                chunk_index=index,
                start=starts[local],
                length=length,
                computed_prefill=self.batch.history[req],
            )
            if length == 0:
                continue
            if not self.allowed("gather", length, latent[begin : begin + length], positional[begin : begin + length]):
                continue
            for component, source, cache in (
                ("latent", latent, self.cache[0]),
                ("positional", positional, self.cache[1]),
            ):
                try:
                    expected = read_paged(cache, self.table[req], starts[local], length)
                except ValueError as error:
                    self.failed = True
                    self.skip("gather", str(error), request_id=self.batch.requests[req])
                    continue
                actual = cpu(source[begin : begin + length])
                self.check(f"gather/{index}/{req}/{component}", actual, expected, exact=True)

    def current(self, q_nope, q_pe, key, positional, value, output, lse):
        num_decode_tokens = self.meta.num_decode_tokens
        for req in self.selected:
            if req < self.meta.num_decodes:
                continue
            start, end = [offset - num_decode_tokens for offset in self.batch.offsets[req : req + 2]]
            rows = sample_rows(end - start, self.config.query_rows)
            indices = [start + row for row in rows]
            query = torch.cat((cpu(q_nope[indices]), cpu(q_pe[indices])), -1)
            self.prefill_rows[req] = (indices, query)
            self.keep(f"query/{req}", query)
            branch = self.attention_branch(
                f"current/{req}",
                query,
                key[start:end],
                positional[start:end],
                value[start:end],
                output[indices],
                canonical_lse(lse, output.shape[0], output.shape[1])[indices],
                rows,
            )
            self.reference_branches[req] = [branch[0]] if branch is not None else None
            self.actual_branches[req] = [branch[1]] if branch is not None else None

    def attention_branch(self, stage, query, key, positional, value, output, lse, causal_positions=None):
        # Copy a few heads at a time; expanded positional views must not turn a
        # modest context into an unbounded full-head CPU allocation.
        if not self.allowed(stage, key.shape[0], key[:, :1], positional[:, :1], value[:, :1]):
            return None
        per_head = max(1, key.shape[0] * (key.shape[-1] + positional.shape[-1] + value.shape[-1]) * 4)
        heads = min(MAX_REFERENCE_HEADS, max(1, MAX_WORK_BYTES // per_head))
        references = []
        for start in range(0, query.shape[1], heads):
            end = start + heads
            packed_key = torch.cat((cpu(key[:, start:end]), cpu(positional[:, start:end])), -1)
            values = cpu(value[:, start:end])
            references.append(
                reference_attention(
                    query[:, start:end], packed_key, values, self.impl.scale, causal_positions=causal_positions
                )
            )
        expected, expected_lse = (torch.cat([result[i] for result in references], 1) for i in (0, 1))
        actual, actual_lse = cpu(output), cpu(lse)
        out_ok = True
        if key.shape[0]:
            out_ok = self.check(stage + "/out", actual, expected)
        else:
            # The empty branch output is undefined, not itself a failure. Keep
            # its actual values so a bad merge can be traced back to this input.
            self.keep(stage + "/empty_output", actual)
            self.emit(
                "empty_history_output",
                stage=stage,
                layout=tensor_layout(actual),
                nan=int(torch.isnan(actual).sum()),
                infinite=int(torch.isinf(actual).sum()),
            )
        lse_ok = self.check(stage + "/lse", actual_lse, expected_lse)
        if not out_ok or not lse_ok:
            bad = ~(
                torch.isclose(actual, expected, atol=self.config.atol, rtol=self.config.rtol).all(-1)
                & torch.isclose(actual_lse, expected_lse, atol=self.config.atol, rtol=self.config.rtol)
            )
            head = int(bad.any(0).nonzero()[0])
            self.keep(stage + "/input_head", torch.tensor(head))
            self.keep(
                stage + "/key", torch.cat((cpu(key[:, head : head + 1]), cpu(positional[:, head : head + 1])), -1)
            )
            self.keep(stage + "/value", value[:, head : head + 1])
        return (expected, expected_lse), (actual, actual_lse)

    def history(self, index, key, positional, value, output, lse):
        lengths = self.history_lengths[index]
        for req, (indices, query) in self.prefill_rows.items():
            local = req - self.meta.num_decodes
            start, length = sum(lengths[:local]), lengths[local]
            branch = self.attention_branch(
                f"history/{index}/{req}",
                query,
                key[start : start + length],
                positional[start : start + length],
                value[start : start + length],
                output[indices],
                canonical_lse(lse, output.shape[0], output.shape[1])[indices],
            )
            if branch is None or self.reference_branches[req] is None:
                self.reference_branches[req] = self.actual_branches[req] = None
            else:
                self.reference_branches[req].append(branch[0])
                self.actual_branches[req].append(branch[1])

    def merged(self, output):
        for req, (indices, _) in self.prefill_rows.items():
            if self.reference_branches[req] is None:
                self.skip("merge", "incomplete_reference", request_id=self.batch.requests[req])
                continue
            actual = cpu(output[indices])
            expected, _ = reference_merge(self.reference_branches[req])
            from_actual, _ = reference_merge(self.actual_branches[req])
            self.check(f"merge/{req}/dense_reference", actual, expected)
            self.check(f"merge/{req}/actual_branches", actual, from_actual)
            if self.batch.history[req] == 0:
                self.check(f"merge/{req}/empty_history_identity", actual, self.actual_branches[req][0][0])

    def decode(self, flash, output):
        # Compare only live rows. Flash returns NTD latent output.
        lengths, cu, used = cpu(flash.cache_lens).tolist(), cpu(flash.cu).tolist(), cpu(flash.used_q).tolist()
        table = cpu(flash.block_table).tolist()
        for name in ("cache_lens", "cu", "used_q", "block_table", "slots", "positions", "schedule"):
            self.keep("flash/" + name, getattr(flash, name))
        num_decodes, num_decode_tokens = self.meta.num_decodes, self.meta.num_decode_tokens
        for name, actual, expected in (
            ("cache_lens", lengths[:num_decodes], self.seq_lens[:num_decodes]),
            ("cu", cu[: num_decodes + 1], self.batch.offsets[: num_decodes + 1]),
            (
                "used_q",
                used[:num_decodes],
                [self.batch.offsets[i + 1] - self.batch.offsets[i] for i in range(num_decodes)],
            ),
            ("block_table", table[:num_decodes], self.table[:num_decodes]),
        ):
            self.check("flash_metadata/" + name, torch.tensor(actual), torch.tensor(expected), exact=True)
        self.check(
            "flash_metadata/slots",
            cpu(flash.slots[:num_decode_tokens]),
            torch.tensor(self.slots[:num_decode_tokens]),
            exact=True,
        )
        self.check(
            "flash_metadata/positions",
            cpu(flash.positions[:num_decode_tokens]),
            self.tensors["positions"][:num_decode_tokens],
            exact=True,
        )
        for req in self.selected:
            if req >= self.meta.num_decodes or used[req] == 0:
                continue
            rows = sample_rows(used[req], self.config.query_rows)
            indices = [cu[req] + row for row in rows]
            if not self.allowed("flash_decode", lengths[req], flash.query[indices]):
                continue
            try:
                latent = read_paged(self.cache[0], table[req], 0, lengths[req])
                positional = read_paged(self.cache[1], table[req], 0, lengths[req])
            except ValueError as error:
                self.failed = True
                self.skip("flash_decode", str(error), request_id=self.batch.requests[req])
                continue
            query = cpu(flash.query[indices])
            key = torch.cat((latent, positional), -1)
            values = latent
            positions = [lengths[req] - used[req] + row for row in rows] if self.meta.causal else None
            expected, _ = reference_attention(query, key, values, self.impl.scale, causal_positions=positions)
            self.keep(f"flash/query/{req}", query)
            self.check(f"flash/{req}/out", cpu(output[:, indices].transpose(0, 1)), expected)

    def finish(self, projected, output):
        for req in self.selected:
            start, end = self.batch.offsets[req : req + 2]
            indices = [start + row for row in sample_rows(end - start, self.config.query_rows)]
            self.keep(f"projection_rows/{req}", torch.tensor(indices))
            for name, source in (("projection_input", projected), ("projection_output", output)):
                if indices and max(indices) >= source.shape[0]:
                    self.skip(
                        name,
                        "projection_rows_not_local",
                        request_id=self.batch.requests[req],
                        layout=tensor_layout(source),
                    )
                    continue
                value = cpu(source[indices])
                self.keep(f"{name}/{req}", value)
                self.check(
                    f"{name}/{req}/finite", torch.isfinite(value), torch.ones_like(value, dtype=torch.bool), exact=True
                )
        file = self.owner.save(self.tensors, self.owner.layers.index(self.name))
        self.emit(
            "layer_end",
            tensor_file=file,
            comparisons=self.checked,
            failed=self.failed,
            skipped=self.skips,
            coverage="sampled requests/query rows and selected layers only",
        )
