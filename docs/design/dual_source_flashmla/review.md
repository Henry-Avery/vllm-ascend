# Fixed dual-source cache and external FlashMLA candidate

This independent candidate appends the complete descriptor correction from
PR 13 `93dbf8f91c0ca3d94e5617b0a1903b0456edc40a` to PR 15
`d843e2e5d865926ba27c6d3222993b61db48380f`, without merging PR 13 history.
The two upstream source heads and PR base remain fixed. It is a Draft for code review;
no NPU serving, installation, deployment or automation was changed.
Local CPU checks do not establish a runtime NaN fix.

## Source locks and comparison baseline

The GitHub PR commit lists and `git ls-remote upstream refs/pull/*/head`
agreed on these heads on 2026-09-29. Git objects were fetched and verified.

| Input | Full SHA |
| --- | --- |
| [PR 16456](https://github.com/vllm-project/vllm-ascend/pull/16456/commits) | `6151d7821f3f3ff0489c4976a602f99c948d0638` |
| [PR 14340](https://github.com/vllm-project/vllm-ascend/pull/14340/commits) | `e86c7357247b44ac42613859df32890e350c1d14` |
| Their actual common ancestor | `f18d103838b831380d3590df32e0e82d4aa07a9a` |
| Pure signed-off three-way merge | `32ba99d135aa71b924600bfb10cb6ebdbe72a181` |
| Pure merge tree | `b0872f240f6209abcce680e0fb113a4428f4131d` |
| Plan-1 fixed review reference | `de31c53dc5b94ff246b17aa198404a082162c2f9` |
| Proposed paired vLLM, not installed locally | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |

The pure merge has exactly the two locked heads as parents. `merge-tree` and
the committed `ort` merge produced the same tree, without textual conflicts.
It is preserved on `codex/dual-source-cache-base-6151-e86c` as the PR base.
PR 12 and PR 13 are not ancestors. There is no additional main merge;
main history already inherited through the two source branches is retained.
The source lock and reference-document hash are in [source_lock.json](source_lock.json).

### What actually changed upstream

PR 16456 added three commits after old head
`52bf27c246e3c9c747267cd1dfce3e63e01020d1`: a fused BBND scatter writer
experiment, edge fixes, then its revert. Old and new head trees are identical:
`610504e4e2a45c4104f68f5f114d490da48e1a24`. Their net diff is empty. Thus
PR 13 did not miss a final writer change from those commits; this candidate
does not restore the reverted experiment.

PR 14340's old series base was `7e2c563f5e6ceddb5b0975753013832d356e096e`;
the new six-commit series starts at `caeb1c9cdbd3a44a9fb03495afc786fbb1018f1e`.
The [range-diff summary](pr14340_range_diff.txt) separates replayed patches from the
changed main background. Patches 2/3/4/6 are equivalent. Patch 1 additionally
uses the already-selected `kernel_block_size` in the V1 cache-view call;
patch 5's difference is test insertion context. The 324-file old-head/new-head
diff must not be presented as a 324-file cache feature rewrite.

Relative to PR 13's cache base `bccc3c9c`, the pure new merge differs in 162
files; upstream main `423e85d` to the new series base differs in 160 files.
Across the five core MLA/cache/Kimi/runner files, the additional semantic
difference is the new MLA-only, non-SFA cache-quantization dtype guard in
`_reshape_kv_cache_v2`; it is preserved. Main background changes elsewhere
remain inherited and require their own full-runtime validation.

## Full-path review against plan-1

Plan-1 was read as an execution path, not used as a patch to merge. PR 12 is
historical evidence only. Components from the reviewed PR 13 implementation
are reused after checking the contracts below against the new source tree.
This is not a rewrite from scratch. The pure baseline remains separately
reviewable before that integration.

| Stage | Plan-1 behavior examined | Candidate decision and remaining evidence |
| --- | --- | --- |
| Allocation / physical pages | Shared descriptor-backed allocation, per-layer/page offsets, manager/kernel splitting; newer 2D BLHNC/LBHNC helper | The initial review missed the actual spec/planner entry and plan-1 descriptor raw-view branch. The follow-up separates query metadata from KV geometry, allocates descriptor-strided byte rows, retains Mamba manager pitch and maps MLA kernel subpages with plan-1 ownership/offset semantics. It does not import the newer `create_kv_cache_views` API. Actual paired planner tests now cover BLHNC/LBHNC/LBNHC and ratios 1/3/6. |
| Persistent MLA writer | Writes the same durable pool later read by attention; native Flash route can use fused state copy / scatter | Retain PR 16456's final writer. K3 `_exec_kv_no_rope` passes zero-copy latent/positional component views to `DeviceOperator.reshape_and_cache`. The external decode reader receives the original fused BBND view. No replacement cache or whole-pool normalization is introduced. New device seam checks written values. |
| KDA convolution | Separate Flash conv route and native state stride support | Retain FLA NPU wrappers from the two-source baseline. Original state descriptor is forwarded; the pinned native conv source handles first-axis/state strides. Do not import older PR 10 conv workarounds or plan-1's extra operators. Device equivalence remains pending. |
| KDA recurrent | Native recurrence receives cache stride and aliases final state | Existing AscendC adapter, tiling and both kernels address actual first-axis stride. No recurrent math change. New NPU seam updates state between MLA write and decode read. |
| KDA prefill | Selected-state native/byte gather and scatter, reset flags, VK state layout | Reuse existing Triton selected-payload kernels rather than cache advanced indexing. Preserve Q/K normalization, gate/beta and `state_v_first=True`. False initial-state flags gather zero; invalid IDs are masked; valid scatter destinations must be unique. |
| Copy / zero | Copies and zeros actual payload spans using physical stride | Temporal Mamba copy remains payload-based. General COW distinguishes overlapping views and snapshots all sources before any scatter. Paired vLLM zeroer separates manager pitch from virtual-page payload spans and skips Mamba. CPU zeroer construction checks passed; actual launch remains pending. |
| FIA prefill | Reference has additional Flash/full-prefill paths, two-stage current/history merge | Keep FIA for every actual MLA prefill, including cold and history-backed 1/2-token suffixes. Existing FIA history gather/LSE merge is retained. The NPU seam exercises real current/history FIA and compares to independent logical FP32 attention. |
| External decode | Reference uses private native binding and BNBD in parts of its route | Use public `cann_ops_transformer.ops` metadata/attention with `PA_BBND`, TND query, NTD latent output and V projection. No private binding fallback. Head/dtype/layout/PCP/DCP/KV-transfer restrictions fail early. The installed binary's padded stride support remains unverified. |
| Metadata / stream lifetime | Stable captured buffers, metadata-before-main, reuse fence, inactive rows | Reuse builder-owned buffers and worker-owned executor; refresh after reuse fence, wait before writer/rope and reader, retain only captured decode capacities, use separate mixed-batch buffers and mask padding. CPU metadata contract/lifecycle checks passed; real device events remain unverified. |
| Sorting / graph | Reference has different all-Flash and graph execution choices | Sort all per-request fields by actual phase before positions/slots/tables are built. MLA phase is separate from a KDA-local one-token recurrent optimization. Real prefill bypasses FULL decode graph; decode capture buffers refresh outside replay. Full-model mixed batches and graph replay remain pending. |

The proposed FLA package is the source baseline's A5 Docker dependency
`26.9.1+deva4a7958`, release `v26.9.1-beta2`, source
[`a4a795857868729fbdbb0118c1e06c8655a7b31b`](https://github.com/flashserve/flash-linear-attention-npu/tree/a4a795857868729fbdbb0118c1e06c8655a7b31b).
Its conv wrappers preserve the view; native `IgnoreContiguous`, host input
strides and `convStateStride0/1` GM addressing are source evidence. An installed
wheel/OPP must still be identified and tested. Conv skips PAD -1 and null slot
0; generic state gather/scatter treats slot 0 as valid. Graph emptiness belongs
in metadata rather than being inferred from generic state IDs.

### PR 13 fixes checked individually

| Fix from `b4af8712a` | Present in pure new baseline? | Candidate handling |
| --- | --- | --- |
| Kimi prefill selected-payload state IO | No | Reapplied and CPU-tested |
| Invalid state-ID load/store masks with 64-bit offsets | No | Reapplied and CPU-tested |
| COW alias union and all-source snapshot before destination writes | No | Reapplied and CPU-tested |
| Page-view arithmetic, dtype alignment and raw-slice bounds | No | Reapplied and CPU-tested |
| Negative hybrid raw offsets and fractional MLA page splits | No | Reapplied and CPU-tested |
| Independent Mamba pool remains page-major | Yes | Retained, with regression coverage |
| Native stride-aware conv/recurrent and paired zeroer geometry | Yes | Retained, source/CPU construction checked; device pending |

COW temporary memory is the sum of selected payload snapshots across distinct
views and index metadata. This prevents cross-view source destruction during
swaps, but has a peak-memory cost; measure it under realistic reuse workloads.
No device `item()` or CPU tensor-content inspection is added to the decode
hot path. The existing flags are centralized in `envs.py`, default off and
documented; architectural/env review is still part of reviewing this Draft.

## Descriptor follow-up and current validation

The two independent corrections are documented in
[the source repair analysis](../pr13_descriptor_startup_fix.md).
The cache spec now uses `num_query_heads`; inherited `num_heads` remains KV
geometry through merge and V1/V2 construction. The allocator preserves legal
manager/layer strides, and MLA splitting divides only the within-manager
layer offset, never the allocation-base offset. KDA states retain their manager
pitch. Writer, reader, copy and zero consume these final views. No cache bytes
are moved to normalize the pool.

The PR 15-specific writer/reader CPU fixture also needed its obsolete
`num_heads=12` spec argument renamed; attention layer query-head parameters
keep their original meaning. The current selected-kernel-size adaptation,
MLA/non-SFA quantization guard and all phase/device seam tests are retained.

Local follow-up results: **62 cache tests + 2 phase tests passed**, and
**44 FlashMLA contract/lifecycle tests passed**. Paired-source CPU zeroer
assembly smoke also passed (offset64, ratios1/3, payload-only protection);
all-file formatting passed. The cache suite includes
the original slot-3/MLA-page-174 incident, exact publisher LBNHC values,
real spec/merge/planner, three layouts, ratios 1/3/6, KDA updates preserving
live MLA pages, backing offsets and payload-only copy/zero. CPU launches
remain explicit substitutes. Full-runtime `test_device_metadata.py` could
not collect because this environment has no installed `vllm`; the old
14-check result below is historical, not a newly repeated runtime check.

Device acceptance is **not run**. The existing NPU seam constructs dense
per-layer descriptors; it must not be presented as real device coverage of
all block-outer layouts. Follow the [publisher handoff](publisher_handoff.md)
for exact startup, actual descriptor isolation and numerical gates.

## Initial d843 validation record (historical)

Local environment: macOS CPU, Python 3.12, torch 2.8.0. No complete paired
vLLM runtime, `torch_npu` or NPU is installed in this worktree.

- Cache/consumer/geometry suite: **25 passed**. It runs actual production
  Python and Triton kernel bodies via AST/leaf loading, with explicit external
  spec/device-memory substitutes. The three added writer-to-reader seams
  check backing identity, persisted component values and KDA/MLA byte isolation
  at manager/kernel ratios 1/3/6. Their writer/attention launches are substitutes,
  so they do not validate numerical FIA/FlashMLA.
- Existing FlashMLA adapter and metadata lifecycle suite: **44 passed**.
  Length refresh, retained addresses, mixed buffers, phase split, masks and
  device-metadata task contracts are CPU checks, not real event ordering.
- Actual MRV2 request sorting and graph dispatch bodies: **2 passed** under
  upstream batch/graph API shells. One/two-token cold and suffix prefills keep
  identity across every indexed field, bypass FULL decode graph, and clear
  stale gates on the next decode/profiling step. Full scheduler execution and
  captured graph operations remain unverified.
- Existing device-metadata executor tests: **14 passed** with leaf production
  imports and forward-context/NPU stream API shells. Batch descriptor fields
  came from the local pinned vLLM source; real streams/external events and
  the complete paired runtime were not exercised.
- Paired `ced685` `AttentionGroup`/`KVBlockZeroer` plus Ascend V2 wrapper CPU
  smoke: **passed** with offset 64, ratios 1/3, payload-only zeroing and untouched
  prefix/suffix/page gaps. Device zero launch is simulated.
- All-file `bash format.sh ci`, `git diff --check` and changed-source AST parsing
  pass. The first format run detected truncated function text and trailing
  whitespace in Git's full range-diff output; the committed evidence uses its
  exact `--no-patch` summary, with full source ranges in the lock file.
  Full UT/ST, operator build, actual NPU math, serving, graph replay and
  performance remain untested. No device acceptance is claimed.

CPU commands from this repository root:

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_hybrid_state_page_layout.py -q
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_flashmla_phase_contract.py -q
python -m pytest --confcutdir=tests/ut/attention \
  tests/ut/attention/test_flashmla_contract.py \
  tests/ut/attention/test_flashmla_metadata_lifecycle.py -q
python tests/ut/worker/v2/hybrid_zeroer_cpu_smoke.py \
  /path/to/paired-vllm/vllm/v1/worker/utils.py
bash format.sh ci
```

## Prepared NPU execution and acceptance

The standalone NPU test has two separate groups:

1. Three manager sizes (128/384/768): actual conv, KDA prefill/recurrent,
   gather/scatter, COW and zeroing with byte-gap/live-page guards.
2. Eighteen eager seam cases: those three manager sizes, prefill widths 1/2/5
   and history lengths 0/129. They call actual durable KV writer, FIA current
   attention and history gather/LSE merge, then update KDA and perform first
   and subsequent external FlashMLA decode on the same backing. Independent
   CPU FP32 attention over logical KV is the numerical reference; identity
   normalization and deterministic projection weights isolate cache persistence.
   Operators are not mocked. Controlled scheduler inputs and method fixtures
   mean these cases still do not prove real scheduler or graph routing.

On an explicitly authorized compatible A5 environment:

```bash
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_hybrid_page_state_npu.py
python -m pytest -sv tests/ut/ops/test_kimi_kda.py
python -m pytest -sv tests/ut/worker/test_model_runner_v2.py
```

Record full VA/vLLM SHAs, loaded module/patch paths, torch/torch_npu/CANN,
FLA and external FlashMLA wheel/OPP identities, hardware/topology, model/input
hashes and launch flags. Preserve raw per-case results and layout metadata.
Source support is not a substitute for the loaded binary's contract.

Next, use the pure merge as A and the final integrated candidate as B, with the
same paired vLLM/software, in A/B/A/B runs. This compares the complete candidate
to its source baseline: A has no external FlashMLA integration, so actual
attention routing differs and improvements cannot be attributed solely to
the KDA fix. Record each run's actual writer, FIA/decode and state routes.
Separately compare this same integrated candidate with external FlashMLA
disabled/enabled, keeping the state fixes, software and request corpus fixed;
that isolates the added decode integration more closely. The historical PR 10
incident needs its own locked versions and reproducible preconditions.
Exercise actual model cold 1/2-token
prefill, history-backed short suffixes, long/chunked prefill, pure decode and
mixed batches; assert that MLA prefill always executes FIA even when KDA uses
local recurrent metadata. Then run eager C60/max-output-8, max sequences 64,
TP8 per node / DP4 total for at least 10 rounds, long/short mixes and block
reuse/COW/zero. Save raw logits, first nonfinite stage, request/rank/step data
and untouched-page evidence; HTTP/SSE success does not establish correctness.

After eager passes, test graph padding/empty rows, stable addresses, event
fences, replay, zero-history/LSE and speculative paths separately. Benchmark
without trace/dump overhead; record JIT warmup, COW snapshot peak allocation,
latency and throughput. This run sheet neither deploys code nor transfers
publisher ownership. Keep old PR 10 incident, PR 13 control and this new
candidate's results separate.
