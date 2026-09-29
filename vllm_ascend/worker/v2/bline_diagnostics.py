# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, bounded eager diagnostics. No device work occurs on the disabled path.

Only valid historical token payloads are copied, never the whole shared pool.
A local CHECK is evidence for one invocation, not model numerical acceptance.
"""

import functools
import hashlib
import inspect
import json
import logging
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch


class NotCovered(Exception):
    """The probe cannot establish the requested contract."""


class NotApplicable(NotCovered):
    """No historical tokens exist on a cold prefill; this supplies no evidence."""


def synchronize(tensor):
    if tensor.device.type == "npu":
        torch.npu.synchronize(tensor.device)


def tensor_layout(tensor):
    return dict(
        pointer=tensor.data_ptr(),
        storage=tensor.untyped_storage().data_ptr(),
        offset=tensor.storage_offset(),
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
    )


def history_pages(cache, metadata, num_reqs, max_pages):
    """Resolve kernel pages from actual metadata; trim the final historical page."""
    if not isinstance(cache, torch.Tensor) or cache.ndim != 4 or cache.shape[2] != 1:
        raise NotCovered("requires fused single-KV-head BBND cache")
    starts = metadata.query_start_loc[: num_reqs + 1].detach().cpu().tolist()
    lengths = metadata.seq_lens[:num_reqs].detach().cpu().tolist()
    if len(starts) != num_reqs + 1 or len(lengths) != num_reqs or starts[0] != 0:
        raise ValueError("invalid request/sequence metadata")
    if metadata.block_tables.ndim != 2 or metadata.block_tables.shape[0] < num_reqs:
        raise ValueError("invalid live block table shape")
    pages = []
    for req, length in enumerate(lengths):
        query = starts[req + 1] - starts[req]
        if query <= 0 or length < query:
            raise ValueError("invalid live query/history length")
        history = length - query
        count = (history + cache.shape[1] - 1) // cache.shape[1]
        if count > metadata.block_tables.shape[1]:
            raise ValueError("history exceeds block table")
        if len(pages) + count > max_pages:
            raise NotCovered("history page budget exceeded; no partial PASS")
        ids = metadata.block_tables[req, :count].detach().cpu().tolist()
        for logical, physical in enumerate(ids):
            if type(physical) is not int or not 0 <= physical < cache.shape[0]:
                raise ValueError("invalid live MLA page")
            live = min(cache.shape[1], history - logical * cache.shape[1])
            pages.append((req, physical, live))
    return pages


class BLineDiagnostics:
    """Per-runner state, enabled only by additional_config.bline_diagnostics."""

    def __init__(self, config, *, rank, context, emit=None):
        if type(config.get("enabled", False)) is not bool:
            raise ValueError("bline_diagnostics.enabled must be a JSON boolean")
        allowed = {"enabled", "max_steps", "max_bytes", "max_events", "max_pages", "max_requests", "arm_file"}
        if set(config) - allowed:
            raise ValueError(f"Unknown bline_diagnostics keys: {set(config) - allowed}")
        self.limits = {}
        for key, default, ceiling in (
            ("max_steps", 8, 128),
            ("max_bytes", 268435456, 1073741824),
            ("max_events", 2048, 16384),
            ("max_pages", 256, 4096),
            ("max_requests", 64, 256),
        ):
            value = config.get(key, default)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{key} must be an integer in [1, {ceiling}]")
            self.limits[key] = value
        arm_file = config.get("arm_file")
        if arm_file is not None and (not isinstance(arm_file, str) or not Path(arm_file).is_absolute()):
            raise ValueError("arm_file must be an absolute path")
        self.arm_file = Path(arm_file) if arm_file is not None else None
        self.armed = self.arm_file is None
        self.rank = rank
        self.context = context
        self.emit_line = emit or (lambda line: logging.getLogger(__name__).warning("BLINE %s", line))
        self.run_id = uuid.uuid4().hex
        self.step = 0
        self.active = False
        self.remaining = self.limits["max_bytes"]
        self.events = 0
        self.first_bad = False
        self.exhausted = False
        self.counts = Counter()
        self.req_ids = []
        self.mla = {}
        self.kda = []
        self.expected = set()
        self.hits = set()
        self.supported = True

    def emit(self, status, check, **details):
        if self.events >= self.limits["max_events"]:
            self.active = False
            if not self.exhausted:
                self.exhausted = True
                self.emit_line(
                    json.dumps(
                        dict(
                            run_id=self.run_id,
                            rank=self.rank,
                            step=self.step,
                            status="UNCOVERED",
                            check="event_budget",
                            counts=dict(self.counts),
                            requests=self.req_ids,
                            incomplete_step=True,
                        )
                    )
                )
            return
        self.events += 1
        self.counts[f"{check}:{status}"] += 1
        first = status == "FAIL" and not self.first_bad
        self.first_bad |= status == "FAIL"
        self.emit_line(
            json.dumps(
                dict(
                    run_id=self.run_id,
                    rank=self.rank,
                    step=self.step,
                    status=status,
                    check=check,
                    first_bad=first,
                    **details,
                )
            )
        )

    def reserve(self, size):
        if size > self.remaining:
            raise NotCovered("run byte budget exhausted; restart for a new measured workload")
        self.remaining -= size

    def begin(self, dummy):
        self.active = False
        if dummy or not self.supported:
            return
        if not self.armed:
            if not self.arm_file.is_file():
                return
            self.armed = True
            self.emit("TRACE", "armed", arm_file=str(self.arm_file))
        if self.step >= self.limits["max_steps"]:
            if not self.exhausted:
                self.emit("UNCOVERED", "step_budget", remaining_bytes=self.remaining)
                self.exhausted = True
            return
        self.step += 1
        self.counts.clear()
        self.expected.clear()
        self.hits.clear()
        self.req_ids = []
        self.active = True
        self.emit("TRACE", "begin", limits=self.limits, remaining_bytes=self.remaining)

    def batch(self, batch):
        if not self.active:
            return
        if len(batch.req_ids) > self.limits["max_requests"]:
            self.emit("UNCOVERED", "batch", reason="request budget exceeded")
            self.active = False
        else:
            self.req_ids = list(batch.req_ids)
            self.emit("TRACE", "batch", requests=self.req_ids)

    def finish(self):
        if not self.active:
            return
        missing = sorted(f"{layer}:{route}" for layer, route in self.expected - self.hits)
        if not any(self.counts[f"raw_logits:{status}"] for status in ("CHECK", "FAIL", "UNCOVERED")):
            missing.append("raw_logits")
        self.emit(
            "UNCOVERED" if missing else "TRACE",
            "step_summary",
            counts=dict(self.counts),
            missing=missing,
            requests=self.req_ids,
            remaining_bytes=self.remaining,
            scope="local rank/invocations only; cross-step route checks and numerical acceptance remain required",
        )
        self.active = False

    def hit(self, layer, route):
        if not self.active:
            return
        self.hits.add((layer, route))
        metadata = self.context().attn_metadata
        if not isinstance(metadata, dict):
            self.emit("UNCOVERED", "metadata", reason="per-layer metadata unavailable")
            return
        for name in self.kda + list(self.mla):
            item = metadata.get(name)
            if item is None:
                self.emit("UNCOVERED", "metadata", layer=name, reason="layer metadata missing")
                continue
            prefill, decode = getattr(item, "num_prefills", 0), getattr(item, "num_decodes", 0)
            if name in self.kda:
                if prefill or decode:
                    self.expected.add((name, "kda_conv"))
                if prefill:
                    self.expected.add((name, "kda_prefill"))
                if decode:
                    self.expected.add((name, "kda_recurrent"))
            else:
                if prefill or decode:
                    self.expected.add((name, "writer"))
                if prefill:
                    self.expected.add((name, "fia"))
                if decode:
                    self.expected.add((name, "external"))

    def finite(self, check, value, **details):
        if not self.active:
            return
        if not isinstance(value, torch.Tensor) or not value.numel():
            self.emit("UNCOVERED", check, reason="no live tensor", **details)
            return
        try:
            self.reserve(value.numel() * value.element_size())
        except NotCovered as error:
            self.emit("UNCOVERED", check, reason=str(error), **details)
            return
        finite = torch.isfinite(value).all().item()  # Diagnostic-only device synchronization.
        if not finite:
            details["first_nonfinite_flat_index"] = int(torch.isfinite(value).flatten().int().argmin().item())
        self.emit("CHECK" if finite else "FAIL", check, **details)

    def protect(self, layer, route, state, ids):
        """Snapshot only history owned by MLA, using group IDs plus actual tables."""
        snapshots = []
        if not self.active:
            return snapshots
        if not self.req_ids:
            raise NotCovered("no actual request-order mapping")
        if ids.dtype not in (torch.int32, torch.int64) or ids.ndim not in (1, 2):
            raise ValueError("invalid KDA index dtype/shape")
        indices = ids[:, 0] if ids.ndim == 2 else ids
        if indices.numel() > self.limits["max_requests"]:
            raise NotCovered("state index budget exceeded")
        state_ids = indices.detach().cpu().tolist()
        if any(index < -1 or index >= state.shape[0] for index in state_ids):
            raise ValueError("invalid live KDA state index")
        # Conv null slot0 and PAD -1 are not written; recurrence may use slot0.
        owned = {index for index in state_ids if index >= (1 if route == "kda_conv" else 0)}
        if not owned:
            raise NotApplicable("no writable state slots in this call")
        metadata = self.context().attn_metadata
        if not isinstance(metadata, dict):
            raise NotCovered("per-layer metadata unavailable")
        plans = []
        shared_metadata = False
        total_bytes = 0
        for name, (cache, manager_tokens, group) in self.mla.items():
            if name not in metadata or cache.untyped_storage().data_ptr() != state.untyped_storage().data_ptr():
                continue
            shared_metadata = True
            ratio, rem = divmod(manager_tokens, cache.shape[1])
            if rem or ratio < 1:
                raise NotCovered("unknown manager/kernel relationship")
            for req, page, live in history_pages(cache, metadata[name], len(self.req_ids), self.limits["max_pages"]):
                if page // ratio in owned:
                    raise ValueError(f"live cross-group manager ownership conflict: state={page // ratio}, MLA={name}")
                view = cache[page, :live]
                if view.stride(-1) != 1:
                    raise NotCovered("non-dense token payload")
                info = dict(
                    layer=layer,
                    mla_layer=name,
                    mla_group=group,
                    request=self.req_ids[req],
                    kernel_page=page,
                    manager_page=page // ratio,
                    live_tokens=live,
                    state_ids=state_ids,
                    address=view.data_ptr(),
                )
                total_bytes += 2 * view.numel() * view.element_size()
                plans.append((view, info))
                if len(plans) > self.limits["max_pages"]:
                    raise NotCovered("aggregate history page budget exceeded")
        if not plans:
            if shared_metadata:
                raise NotApplicable("cold prefill has no history; no protection evidence")
            raise NotCovered("no shared-pool MLA metadata; protection not covered")
        self.reserve(total_bytes)
        self.emit(
            "TRACE",
            "protected_mapping",
            layer=layer,
            route=route,
            state_layout=tensor_layout(state),
            protected=[info for _, info in plans],
        )
        for view, info in plans:
            synchronize(view)
            snapshots.append((view, view.detach().view(torch.uint8).cpu().clone(), info))
        return snapshots

    def compare(self, route, snapshots):
        failures = 0
        for view, before, info in snapshots:
            synchronize(view)
            after = view.detach().view(torch.uint8).cpu()
            changed = after != before
            if changed.any().item():
                failures += 1
                flat = int(changed.flatten().int().argmax().item())
                coordinates = []
                for dim in reversed(changed.shape):
                    coordinates.append(flat % dim)
                    flat //= dim
                coordinates.reverse()
                # Report device address via the original byte view, not the CPU copy stride.
                source = view.view(torch.uint8)
                address = source.data_ptr() + sum(i * s for i, s in zip(coordinates, source.stride()))
                self.emit("FAIL", route, mismatched_bytes=int(changed.sum()), first_bad_address=address, **info)
        if snapshots and not failures:
            self.emit("CHECK", route, layer=snapshots[0][2]["layer"], protected_payloads=len(snapshots))

    def wrap_kda(self, layer, route, original, state_name, ids_name):
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            values = signature.bind(*args, **kwargs).arguments
            self.hit(layer, route)
            self.emit("TRACE", route, layer=layer)
            snapshots = []
            try:
                ids = values[ids_name]
                if route == "kda_prefill":
                    keep = values["prebuilt_metadata"].keep_meta
                    if keep is not None:
                        ids = ids[keep]
                snapshots = self.protect(layer, route, values[state_name], ids)
            except NotApplicable as error:
                self.emit("NOT_APPLICABLE", route, layer=layer, reason=str(error))
            except NotCovered as error:
                self.emit("UNCOVERED", route, layer=layer, reason=str(error))
            except ValueError as error:
                self.emit("FAIL", route, layer=layer, reason=str(error))
            try:
                return original(*args, **kwargs)
            finally:
                self.compare(route, snapshots)

        return wrapped

    def wrap_writer(self, layer, original):
        @functools.wraps(original)
        def wrapped(kv_no_split, kv_cache, slots):
            self.hit(layer, "writer")
            result = original(kv_no_split, kv_cache, slots)
            if not self.active:
                return result
            # These are the real post-normalization values returned by the writer.
            rope, latent = result
            try:
                self.reserve(2 * (rope.numel() * rope.element_size() + latent.numel() * latent.element_size()))
                synchronize(latent)
                ids = slots[: latent.shape[0]].detach().cpu().tolist()
                if len(ids) > self.limits["max_pages"] * kv_cache[0].shape[1]:
                    raise NotCovered("writer token budget exceeded")
                self.emit(
                    "TRACE",
                    "writer_mapping",
                    layer=layer,
                    slots=ids,
                    caches=[tensor_layout(cache) for cache in kv_cache],
                )
                checked = 0
                for index, slot in enumerate(ids):
                    if slot == -1:
                        continue
                    for cache, expected in zip(kv_cache, (latent, rope)):
                        page, token = divmod(slot, cache.shape[1])
                        if not 0 <= page < cache.shape[0]:
                            raise ValueError("invalid writer slot")
                        actual = cache[page, token].detach().cpu()
                        wanted = expected[index].detach().to(device="cpu", dtype=actual.dtype)
                        if not torch.equal(actual.view(torch.uint8), wanted.view(torch.uint8)):
                            self.emit("FAIL", "writer", layer=layer, slot=slot, address=cache[page, token].data_ptr())
                            return result
                    checked += 1
                self.emit("CHECK" if checked else "UNCOVERED", "writer", layer=layer, live_slots=checked)
            except NotCovered as error:
                self.emit("UNCOVERED", "writer", layer=layer, reason=str(error))
            except ValueError as error:
                self.emit("FAIL", "writer", layer=layer, reason=str(error))
            return result

        return wrapped

    def wrap_route(self, layer, route, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.hit(layer, route)
            result = original(*args, **kwargs)
            if self.active:
                # Public eager route returns only actual output rows. LSE/masked rows are not inspected.
                self.finite(route, result, layer=layer)
            return result

        return wrapped

    @contextmanager
    def logits_scope(self, model, batch):
        original = model.compute_logits
        live_rows = batch.logits_indices.numel()

        def compute(hidden, *args, **kwargs):
            self.finite("final_hidden", hidden[:live_rows])
            result = original(hidden, *args, **kwargs)
            self.finite("raw_logits", result[:live_rows] if result is not None else None)
            return result

        model.compute_logits = compute
        try:
            yield
        finally:
            model.compute_logits = original
            self.finish()


def install_bline_diagnostics(runner, context):
    """Install instance-only hooks after cache binding; disabled runs install none."""
    config = runner.vllm_config.additional_config.get("bline_diagnostics", {})
    if not config.get("enabled", False):
        return
    if hasattr(runner, "_bline_diagnostics"):
        return
    parallel = runner.vllm_config.parallel_config
    probe = BLineDiagnostics(config, rank=parallel.rank, context=context)
    runner._bline_diagnostics = probe
    unsupported = (
        not runner.model_config.enforce_eager
        or runner.vllm_config.speculative_config is not None
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or getattr(parallel, "prefill_context_parallel_size", 1) != 1
        or runner.vllm_config.kv_transfer_config is not None
        or getattr(runner, "batch_sharder", None) is not None
    )
    if unsupported:
        probe.supported = False
        probe.emit(
            "UNCOVERED",
            "configuration",
            reason="requires V2 eager PP1/DP1/CP1, no speculative, KV transfer or batch sharder",
        )
        return
    layers = runner.compilation_config.static_forward_context
    kda = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            layer = layers[name]
            spec = group.kv_cache_spec
            if hasattr(spec, "kv_cache_specs"):
                spec = spec.kv_cache_specs[name]
            cache = layer.kv_cache
            for descriptor in runner.kv_cache_config.kv_cache_tensors:
                names = getattr(descriptor, "layers", None) or getattr(descriptor, "shared_by", [])
                if name in names:
                    probe.emit(
                        "TRACE",
                        "descriptor",
                        layer=name,
                        group=group_id,
                        layer_index=names.index(name),
                        size=descriptor.size,
                        offset=descriptor.offset,
                        layer_stride=descriptor.layer_stride,
                        block_stride=descriptor.block_stride,
                    )
            if type(spec).__name__ == "AscendMLAAttentionSpec" and isinstance(cache, torch.Tensor):
                if cache.ndim != 4 or cache.shape[2] != 1:
                    probe.emit("UNCOVERED", "layout", layer=name, reason="requires fused BBND")
                    continue
                probe.mla[name] = (cache, spec.block_size, group_id)
                probe.emit(
                    "TRACE",
                    "layout",
                    layer=name,
                    group=group_id,
                    query_heads=spec.num_query_heads,
                    kv_heads=spec.num_heads,
                    manager_tokens=spec.block_size,
                    **tensor_layout(cache),
                )
                impl = layer.impl
                if hasattr(impl, "_exec_kv_no_rope"):
                    impl._exec_kv_no_rope = probe.wrap_writer(name, impl._exec_kv_no_rope)
                for method, route in (("_forward_prefill", "fia"), ("_forward_external_flashmla", "external")):
                    if hasattr(impl, method):
                        setattr(impl, method, probe.wrap_route(name, route, getattr(impl, method)))
            elif hasattr(layer, "_run_recurrent") and hasattr(layer, "_run_causal_conv1d"):
                kda.append(name)
                for index, state in enumerate(cache):
                    probe.emit("TRACE", "layout", layer=name, group=group_id, state=index, **tensor_layout(state))
                for method, route, state_name, ids_name in (
                    ("_run_causal_conv1d", "kda_conv", "conv_state", "cache_indices"),
                    ("_run_prefill", "kda_prefill", "recurrent_state", "state_indices"),
                    ("_run_recurrent", "kda_recurrent", "recurrent_state", "state_indices"),
                ):
                    setattr(layer, method, probe.wrap_kda(name, route, getattr(layer, method), state_name, ids_name))
    if not probe.mla or not kda:
        probe.emit("UNCOVERED", "binding", reason="requires both KDA and fused MLA groups")
    probe.emit(
        "TRACE",
        "binding",
        kda_layers=kda,
        mla_layers=list(probe.mla),
        layout=str(runner.cache_config.get_resolved_kv_cache_layout()),
        limits=probe.limits,
        arm_file=str(probe.arm_file) if probe.arm_file else None,
        expected_ranks=parallel.tensor_parallel_size,
        diagnostic_path=str(Path(__file__).resolve()),
        diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    probe.kda = kda
    original_execute = runner.execute_model
    execute_signature = inspect.signature(original_execute)

    @functools.wraps(original_execute)
    def execute(*args, **kwargs):
        values = execute_signature.bind(*args, **kwargs).arguments
        scheduler = values["scheduler_output"]
        probe.finish()  # A previous step without sampling is explicitly incomplete.
        probe.begin(
            values.get("dummy_run", False)
            or values.get("is_profile", False)
            or not scheduler.total_num_scheduled_tokens
        )
        try:
            return original_execute(*args, **kwargs)
        except Exception:
            if probe.active:
                probe.emit("FAIL", "execute", reason="service exception; consult original traceback")
                probe.finish()
            raise

    original_inputs = runner.prepare_inputs

    @functools.wraps(original_inputs)
    def inputs(*args, **kwargs):
        result = original_inputs(*args, **kwargs)
        probe.batch(result)
        return result

    original_sample = runner.sample

    @functools.wraps(original_sample)
    def sample(hidden_states, input_batch, grammar_output):
        if not probe.active:
            return original_sample(hidden_states, input_batch, grammar_output)
        with probe.logits_scope(runner.model, input_batch):
            return original_sample(hidden_states, input_batch, grammar_output)

    runner.execute_model = execute
    runner.prepare_inputs = inputs
    runner.sample = sample
