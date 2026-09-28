# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, opt-in eager sampling observations; never alter sampling results.

Copies and CPU reads here deliberately synchronize the selected worker. This
is a numerical-debugging aid, not a performance measurement. Reports contain
request IDs, token IDs and logits and must be kept with private run artifacts.
"""

import inspect
import json
import logging
import math
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

import torch

MAX_BATCH_ROWS = 128
MAX_VOCAB_SIZE = 262144
MAX_HIDDEN_SIZE = 32768
TOP_K = 5


@dataclass(frozen=True)
class DiagnosticConfig:
    directory: str
    steps: int = 64
    rows: int = 2
    dp_rank: int = -1
    tp_rank: int = 0
    token_ids: tuple[int, ...] = (0,)

    def __post_init__(self):
        if not self.directory:
            raise ValueError("Sampling diagnostics require a nonempty output directory")
        if not 1 <= self.steps <= 128 or not 1 <= self.rows <= 8:
            raise ValueError("Sampling diagnostics require STEPS=1..128 and ROWS=1..8")
        if self.dp_rank < -1 or self.tp_rank < 0:
            raise ValueError("Sampling diagnostics require DP_RANK>=-1 and TP_RANK>=0")
        if not 1 <= len(self.token_ids) <= 16 or any(token < 0 for token in self.token_ids):
            raise ValueError("Sampling diagnostics require 1..16 nonnegative TOKEN_IDS")


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def summarize_logits(logits, token_ids):
    """Reduce on device, then copy only bounded per-row statistics to CPU."""
    values = logits.float()
    finite = torch.isfinite(values)
    finite_values = values.masked_fill(~finite, -torch.inf)
    top_values, top_ids = finite_values.topk(min(TOP_K, values.shape[1]), dim=1)
    statistics = (
        torch.stack(
            (
                finite.sum(1),
                torch.isnan(values).sum(1),
                torch.isposinf(values).sum(1),
                torch.isneginf(values).sum(1),
                values.masked_fill(~finite, torch.inf).amin(1),
                finite_values.amax(1),
            ),
            dim=1,
        )
        .cpu()
        .tolist()
    )
    top_values = top_values.cpu().tolist()
    top_ids = top_ids.cpu().tolist()
    watched = values[:, list(token_ids)].cpu().tolist()
    return [
        {
            "finite": int(stats[0]),
            "nan": int(stats[1]),
            "positive_inf": int(stats[2]),
            "negative_inf": int(stats[3]),
            "finite_min": _number(stats[4]),
            "finite_max": _number(stats[5]),
            "finite_topk_ids": top_ids[row],
            "finite_topk_values": [_number(value) for value in top_values[row]],
            "watched_logits": {str(token): _number(value) for token, value in zip(token_ids, watched[row])},
        }
        for row, stats in enumerate(statistics)
    ]


def _invalid_distribution(summary):
    # Negative infinity is normal after token masks/top-k; an all-masked row is not.
    return bool(summary["nan"] or summary["positive_inf"] or not summary["finite"])


class SampleDiagnostics:
    def __init__(self, config, *, dp_rank, tp_rank, global_rank, capture_check=lambda: False):
        self.config = config
        self.identity = dict(
            host=socket.gethostname(), pid=os.getpid(), dp_rank=dp_rank, tp_rank=tp_rank, global_rank=global_rank
        )
        self.capture_check = capture_check
        self.step = 0
        self.active = False
        self.exhausted = False
        self.forward_id = None
        self.directory = Path(config.directory) / (
            f"{self.identity['host']}-pid{os.getpid()}-g{global_rank}-dp{dp_rank}-tp{tp_rank}-{uuid4().hex[:8]}"
        )
        self.directory.mkdir(parents=True, mode=0o700)
        self._emit(
            "armed",
            max_steps=config.steps,
            snapshot_rows=config.rows,
            watched_token_ids=list(config.token_ids),
            watched_token_text="unverified; token 0 is not assumed to be !",
            diagnostic_synchronization=True,
            scope="eager, no speculation, unsharded sampler only",
            hidden_boundaries=True,
        )

    def _emit(self, event, **fields):
        record = dict(
            event=event,
            step=self.step,
            forward_id=self.forward_id,
            utc=datetime.now(timezone.utc).isoformat(),
            **self.identity,
            **fields,
        )
        descriptor = os.open(self.directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        logging.getLogger(__name__).info(
            "[FlashMLA SAMPLE DIAG] event=%s step=%s dp=%s tp=%s report=%s",
            event,
            self.step,
            self.identity["dp_rank"],
            self.identity["tp_rank"],
            self.directory,
        )

    def _validate_batch(self, runner, batch):
        if self.capture_check():
            raise RuntimeError("Sampling diagnostics cannot run during graph capture")
        if batch.num_draft_tokens or runner.speculative_config is not None:
            raise RuntimeError("Sampling diagnostics do not support speculative decoding")
        if getattr(runner, "batch_sharder", None) is not None:
            raise RuntimeError("Sampling diagnostics do not support sharded sampling")
        sampler = runner.sampler
        expected = (
            "logits",
            "expanded_idx_mapping",
            "idx_mapping",
            "idx_mapping_np",
            "pos",
            "input_ids",
            "expanded_local_pos",
            "return_logprobs",
        )
        if sampler is None or tuple(inspect.signature(sampler.sample).parameters) != expected:
            raise RuntimeError("Sampling diagnostics require the pinned MRv2 Sampler.sample interface")
        if getattr(sampler, "use_flashinfer", False):
            raise RuntimeError("Sampling diagnostics cannot observe FlashInfer's internal processed distribution")
        if not 0 < batch.num_reqs <= MAX_BATCH_ROWS or len(batch.req_ids) != batch.num_reqs:
            raise RuntimeError("Sampling diagnostics require 1..128 real request rows")
        if batch.logits_indices.numel() != batch.num_reqs or batch.cu_num_logits_np.tolist() != list(
            range(batch.num_reqs + 1)
        ):
            raise RuntimeError("Sampling diagnostics require exactly one logits row per request")

    def run(self, runner, batch, original_call, hidden_states=None):
        self.forward_id = getattr(batch, "flashmla_diagnostic_id", None)
        if getattr(batch, "flashmla_diagnostic_selected", True) is False:
            return original_call()
        if self.step >= self.config.steps:
            if not self.exhausted:
                self._emit(
                    "budget_exhausted", remaining=0, acceptance="not an accuracy pass; later steps are unobserved"
                )
                self.exhausted = True
            return original_call()
        if self.active:
            raise RuntimeError("Sampling diagnostics cannot observe overlapping sample calls")
        self.step += 1
        self.active = True
        snapshots = {}
        hidden_snapshots = {}
        original_logits = runner.model.compute_logits
        original_sample = runner.sampler.sample if runner.sampler is not None else None
        absent = object()
        saved_logits = vars(runner.model).get("compute_logits", absent)
        saved_sample = vars(runner.sampler).get("sample", absent) if runner.sampler is not None else absent

        def copy_logits(stage, value):
            if stage in snapshots:
                raise RuntimeError(f"Sampling diagnostics observed repeated {stage} calls")
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < batch.num_reqs:
                raise RuntimeError(f"Sampling diagnostics cannot map {stage} to request rows")
            if not 0 < value.shape[1] <= MAX_VOCAB_SIZE or max(self.config.token_ids) >= value.shape[1]:
                raise RuntimeError("Sampling diagnostics vocabulary/token IDs exceed configured bounds")
            snapshots[stage] = value[: batch.num_reqs].detach().clone()

        def copy_hidden(stage, value):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] < batch.num_reqs
                or not 0 < value.shape[1] <= MAX_HIDDEN_SIZE
            ):
                self._emit("boundary_skipped", stage=stage, reason="unmapped_hidden_rows_or_width")
                return
            if stage in hidden_snapshots:
                raise RuntimeError(f"Sampling diagnostics observed repeated {stage} calls")
            value = value[: batch.num_reqs].detach().cpu().clone()
            hidden_snapshots[stage] = value
            summaries = summarize_logits(value, ())
            self._emit(
                "hidden_boundary",
                stage=stage,
                rows=[dict(request_id=req, **summary) for req, summary in zip(batch.req_ids, summaries)],
            )
            if stage == "lm_head_input" and "model_output" in hidden_snapshots:
                expected = hidden_snapshots["model_output"]
                same = value.shape == expected.shape and bool(
                    ((value == expected) | (torch.isnan(value) & torch.isnan(expected))).all()
                )
                self._emit("hidden_mapping", passed=same, request_ids=list(batch.req_ids))

        @wraps(original_logits)
        def observe_logits(*args, **kwargs):
            bound = inspect.signature(original_logits).bind(*args, **kwargs)
            value = next((arg for arg in bound.arguments.values() if isinstance(arg, torch.Tensor)), None)
            copy_hidden("lm_head_input", value)
            value = original_logits(*args, **kwargs)
            # Before grammar masks or sampler in-place processing.
            copy_logits("raw", value)
            return value

        def observe_sample(*args, **kwargs):
            if "raw" not in snapshots:
                raise RuntimeError("Sampling diagnostics missed model.compute_logits before sampling")
            bound = inspect.signature(original_sample).bind(*args, **kwargs)
            snapshots["sampling_mapping"] = bound.arguments["expanded_idx_mapping"].detach().clone()
            result = original_sample(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("Sampling diagnostics expected (sampled, processed_logits)")
            # The real sampler has applied temperature, penalties and top-k/p.
            copy_logits("processed", result[1])
            snapshots["internal_sampled"] = result[0].detach().clone()
            return result

        try:
            self._validate_batch(runner, batch)
            self._emit(
                "batch",
                num_reqs=batch.num_reqs,
                num_tokens=batch.num_tokens,
                request_ids=list(batch.req_ids),
                diagnostic_synchronization=True,
            )
            copy_hidden("model_output", hidden_states[batch.logits_indices] if hidden_states is not None else None)
            runner.model.compute_logits = observe_logits
            runner.sampler.sample = observe_sample
            result = original_call()
            if set(snapshots) != {"raw", "processed", "internal_sampled", "sampling_mapping"}:
                raise RuntimeError("Sampling diagnostics did not observe the complete raw/processed sampler chain")
            self._finish(runner, batch, result, snapshots, hidden_snapshots)
            return result  # Preserve the original output objects and sampling decisions.
        except Exception as error:
            self._emit("error", error_type=type(error).__name__, message=str(error)[:512])
            raise
        finally:
            for target, name, previous in (
                (runner.model, "compute_logits", saved_logits),
                (runner.sampler, "sample", saved_sample),
            ):
                if target is not None:
                    if previous is absent:
                        if name in vars(target):
                            delattr(target, name)
                    else:
                        setattr(target, name, previous)
            self.active = False

    def _finish(self, runner, batch, result, snapshots, hidden_snapshots):
        output, num_sampled, _ = result
        sampled = output.sampled_token_ids.detach().cpu()
        counts = num_sampled.detach().cpu().reshape(-1)
        if sampled.shape != (batch.num_reqs, 1) or counts.numel() != batch.num_reqs:
            raise RuntimeError("Sampling diagnostics cannot map final sampled tokens to requests")
        indices = batch.logits_indices.detach().cpu().tolist()
        positions = batch.positions[batch.logits_indices].detach().cpu().tolist()
        internal_sampled = snapshots.pop("internal_sampled").cpu().reshape(-1)
        if internal_sampled.numel() != batch.num_reqs or snapshots["raw"].shape != snapshots["processed"].shape:
            raise RuntimeError("Sampling diagnostics encountered mismatched internal sampling rows")
        expanded = snapshots.pop("sampling_mapping")
        actual_mapping = expanded.detach().cpu().tolist()
        if actual_mapping != batch.idx_mapping_np.tolist():
            self._emit(
                "mapping_mismatch",
                sampler_device_mapping=actual_mapping,
                batch_cpu_mapping=batch.idx_mapping_np.tolist(),
            )
            raise RuntimeError("Sampling diagnostics CPU/device request-state mappings differ")
        states = runner.sampler.sampling_states
        temperatures = states.temperature.gpu[expanded].detach().cpu().tolist()
        seeds = states.seeds.gpu[expanded].detach().cpu().tolist()
        summaries = {stage: summarize_logits(value, self.config.token_ids) for stage, value in snapshots.items()}
        rows = []
        for row, request_id in enumerate(batch.req_ids):
            token = int(sampled[row, 0])
            if not 0 <= token < snapshots["raw"].shape[1]:
                raise RuntimeError("Sampling diagnostics received an out-of-vocabulary sampled token")
            rows.append(
                dict(
                    request_id=request_id,
                    logits_row=row,
                    hidden_state_row=indices[row],
                    position=positions[row],
                    request_state_index=int(batch.idx_mapping_np[row]),
                    is_prefilling=bool(batch.is_prefilling_np[row]),
                    temperature=_number(temperatures[row]),
                    seed=int(seeds[row]),
                    internal_sampled_token_id=int(internal_sampled[row]),
                    sampled_token_id=token,
                    num_sampled=int(counts[row]),
                    emitted=bool(counts[row]),
                    raw=summaries["raw"][row],
                    processed=summaries["processed"][row],
                )
            )
        # One bounded D2H vector per stage, not one synchronization per request.
        for stage, values in snapshots.items():
            selected = (
                values.gather(1, sampled.to(device=values.device, dtype=torch.long)).detach().cpu().reshape(-1).tolist()
            )
            for row, value in enumerate(selected):
                rows[row][stage]["sampled_token_logit"] = _number(value)
        hidden_invalid = [
            any(not bool(torch.isfinite(value[row]).all()) for value in hidden_snapshots.values())
            for row in range(len(rows))
        ]
        priority = sorted(
            range(len(rows)),
            key=lambda row: (
                bool(getattr(batch, "flashmla_diagnostic_request_ids", ()))
                and rows[row]["request_id"] not in batch.flashmla_diagnostic_request_ids,
                not (
                    hidden_invalid[row]
                    or _invalid_distribution(rows[row]["raw"])
                    or _invalid_distribution(rows[row]["processed"])
                ),
                not (rows[row]["emitted"] and rows[row]["sampled_token_id"] in self.config.token_ids),
                row,
            ),
        )[: self.config.rows]
        dump_name = f"step{self.step:04d}.pt"
        descriptor = os.open(self.directory / dump_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output_file:
            torch.save(
                {
                    "logits_rows": torch.tensor(priority),
                    "raw_logits": snapshots["raw"][priority].cpu(),
                    "processed_logits": snapshots["processed"][priority].cpu(),
                    "sampled_token_ids": sampled[priority],
                    "internal_sampled_token_ids": internal_sampled[priority],
                    "num_sampled": counts[priority],
                    **{stage: values[priority] for stage, values in hidden_snapshots.items()},
                },
                output_file,
            )
        self._emit(
            "end",
            rows=rows,
            tensor_file=dump_name,
            tensor_rows=priority,
            remaining=self.config.steps - self.step,
            diagnostic_synchronization=True,
            note="Finite values alone do not establish correctness; emitted=false rows produce no SSE token",
        )


def install_sample_diagnostics(runner, envs):
    """Install only on selected ranks, only when explicitly enabled in envs."""
    # Lazy worker imports keep the diagnostic helpers usable in isolated CPU tests.
    from vllm.distributed import get_tp_group
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    config = DiagnosticConfig(
        envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DIR,
        steps=envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_STEPS,
        rows=envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_ROWS,
        dp_rank=envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_DP_RANK,
        tp_rank=envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TP_RANK,
        token_ids=tuple(int(token.strip()) for token in envs.VLLM_ASCEND_FLASH_MLA_SAMPLE_DIAG_TOKEN_IDS.split(",")),
    )
    if not envs.VLLM_ASCEND_ENABLE_FLASH_MLA or not runner.model_config.enforce_eager or runner.use_aclgraph:
        raise RuntimeError("Sampling diagnostics require FlashMLA and enforce_eager=True without graphs")
    if runner.speculative_config is not None or runner.parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("Sampling diagnostics require no speculation and pipeline_parallel_size=1")
    group = get_tp_group()
    if config.tp_rank >= group.world_size or config.dp_rank >= runner.parallel_config.data_parallel_size:
        raise ValueError("Sampling diagnostics selected a rank outside the configured parallel groups")
    if group.rank_in_group != config.tp_rank or config.dp_rank not in (-1, runner.dp_rank):
        return
    diagnostic = SampleDiagnostics(
        config,
        dp_rank=runner.dp_rank,
        tp_rank=group.rank_in_group,
        global_rank=group.rank,
        capture_check=torch.npu.is_current_stream_capturing,
    )
    if envs.VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG:
        # Lazy worker import preserves isolated CPU use of the sampling helper.
        from vllm_ascend.attention.flashmla_chunk_diagnostics import ChunkDiagnosticConfig, ChunkDiagnostics

        chunk_diagnostic = ChunkDiagnostics(
            ChunkDiagnosticConfig.from_json(envs.VLLM_ASCEND_FLASH_MLA_CHUNK_DIAG_CONFIG), diagnostic
        )
        chunk_diagnostic.install(runner)
    original = runner.sample

    @wraps(original)
    def observed(hidden_states, input_batch, grammar_output):
        if type(runner.sampler) is not Sampler:
            raise RuntimeError("Sampling diagnostics require the pinned standard MRv2 Sampler, not a custom sampler")
        return diagnostic.run(
            runner, input_batch, lambda: original(hidden_states, input_batch, grammar_output), hidden_states
        )

    runner.sample = observed
