# MRV2 external FlashMLA integration (Draft)

## Handoff snapshot, not a new-main integration

This branch preserves `708e1b9a1c009ae5091cc2287b165f1eeb037378` plus
four-file DSpark adaptation/test/documentation changes on the old #16456 base
`fc0580dd2d375ded3cfc7be5c81523429e5b46a6`. It is an implementation reference,
not an accepted NPU build. The PR body records the exact handoff commit and
the checks rerun for it. Existing Drafts are not superseded or modified.

To continue on another machine:

```bash
git clone --branch handoff/flashmla-dspark-708 --single-branch https://github.com/Henry-Avery/vllm-ascend.git
cd vllm-ascend
git rev-parse HEAD
```

Read this document and the handoff PR body before changing code. Historical
#16468 SHAs below explain this snapshot; they are not the required future
reference. Obtain the owner's current operator reference through an authorized
channel and pin its actual SHA before implementation. No private reference
sources, credentials, or machine configuration are bundled here.

The next integration must use main containing merged #16347 (MRV2 DSpark),
plus #16456's non-contiguous cache. If #16456 has not updated to that main,
combine them in a separate integration branch; do not overwrite this snapshot.
Reuse applicable code and tests here, not its entire obsolete commit history.
In particular, main's DSpark post-load alignment and this snapshot's load-time
rotation must not both transform the same weights. Preserve the main framework
handoff and adapt operator-specific behavior against the current reference.

The final goal remains FlashMLA tiling offload on strided MRV2 cache, including
target and draft DSpark, graph execution, and a K3 four-card service. Start with
PCP/DCP=1. No NPU/server execution is authorized by this handoff itself. After
#16456 merges into main, prepare only our operator integration increment on
that main and revalidate the final commit before requesting its merge.

## Scope and reference

This increment starts at [PR #16456](https://github.com/vllm-project/vllm-ascend/pull/16456),
`fc0580dd2d375ded3cfc7be5c81523429e5b46a6`. Its allocator, strided cache
views, physical page/slot identities, zeroing and COW implementation are retained.
The primary execution reference is [PR #16468](https://github.com/vllm-project/vllm-ascend/pull/16468)
at `3c94ce740126984b9683501dd8c3772bf88d4311`, not the earlier MRV1/MRV2 Drafts.
The paired vLLM baseline is `84030bbe3d74d99bad477a3d2e37a973ccd8865c`.

Enable with `VLLM_ASCEND_ENABLE_FLASH_MLA=1` (default: off). This Draft wires
A5 MRV2 dense, BF16 MLA, including hybrid models whose non-MLA layers retain
their existing backend. First validation uses PCP=1, DCP=1, and no PD/KVPP.
Eager and graph use the same operator path; choosing eager is a launch setting.
Only DSpark is wired for speculative execution in this increment.

This is **awaiting NPU validation**, not a runtime/performance acceptance report.
Local CPU tests use a mocked external package and cache writer.

## Producer to consumer

1. Existing TP-local projections produce absorbed Q `[T,H,576]` and latent KV.
   RoPE is applied only on RoPE layers. NoPE retains the projected 64 channels.
   Actual local heads must be one of `8,12,64,96`; there is no full-head
   replication, invented metadata head count, or implicit FIA fallback.
2. The writer slices the original `[P,128,1,576]` cache into 512/64 views,
   retaining page/token strides and storage offset. It never repacks persistent KV.
3. The metadata builder follows #16468: device query boundaries, actual KV lengths,
   padded zero-used rows, slots and positions populate stable buffers. The package's
   Meta implementation determines schedule shape; missing package/Meta support is
   an error, not an alternate algorithm.
4. `DeviceMetadataExecutor` submits and waits outside capture/replay. Target and
   DSpark own separate executors. Consumer completion is fenced before reuse;
   changing lengths or page-table entries regenerates schedule without replacing
   captured buffers. Causal eager prefill buffers are not retained across shapes.
5. Public `cann_ops_transformer.ops.flash_mla_with_kvcache_metadata` runs first;
   `flash_mla_with_kvcache` consumes the matching schedule. Both use
   `max_seqlen_q=-1,max_seqlen_kv=-1`. Positive internal capacities only size
   buffers and must not be substituted for these API attributes.
6. Main call uses `TND / PA_BBND / NTD`. Both `cu_seqlens_q` and
   `seqused_q` are explicit. Causal mode 3 passes the int8 2048-square mask
   (lower triangle including diagonal zero); noncausal mode 0 passes no mask.
   The normal single segment does not request LSE.
7. Output `[H,T,512]` feeds existing V-up, then gate/O-proj. The ordinary
   path and guarded BF16 gate/GEMM path follow #16468; padded outputs are
   zeroed without propagating inactive NaN/Inf.

## DSpark and graph

Dense MLA DSpark uses the same external interface in its target and query paths.
Its context precompute writer also accepts the strided BBND view. Draft metadata
preserves the noncausal multi-token capability when converting the upstream spec
to the Ascend spec; scheduler groups and allocator geometry are unchanged.
The query capture factory supplies positions and speculative attention state.
Replay uses the prepared buffers without a second FIA update/submission.

K3 MLA draft loading consumes the target rotation path saved before MRV2 clears
the draft quantization config. It folds rotation into context projection weights
once, before upstream embedding/LM-head sharing, and validates the target's raw
auxiliary capture boundaries and hidden size. It does not additionally run the
post-load rotation scheme used by PR #16347.

The initialized draft backend selects MLA, GQA, or the unchanged sparse path.
For FIA, DP-padded eager/PIECEWISE inputs require padded request counts and query
lengths ending at the physical token count. FlashMLA instead retains device
query boundaries and zero-used padding; it must not inherit FIA's synthetic
query lengths. Query metadata construction supplies the same attention state
for capture and runtime, with batch-sized positions and non-prefilling flags.

The September 21 adaptation reviews PR #16347 at
`910a6dcd7b5aaa21fd26ca442f3d1c6dabeaf306`, retaining #16468's earlier
`e496b86159e641010a22990ac167f3713f138d22` model-loading design. The currently
observed #16468 head `1a3728a755db7ee95ce2b911526e0e5227fcd7fa` is a closure
placeholder, not a new integration reference. The new #16456 head
`fa0be36e85b48c9b6800bc06ca1d2e09dd965c19` retains the dense BBND view but has
diverged history and other main updates; it has **not** been adopted by this
fc0580dd-based increment. Migration and revalidation must precede publication
against that new base.

GQA draft layers remain on the existing GQA/FIA backend; this is not an MLA
fallback. #16468's separate GQA FlashAttn interface and head-slot allocator are
not imported. GQA-target/draft combinations require their own graph validation.
The new flag does not disable GQA's existing graph parameter updates.

## Deliberate differences and remaining work

- #16468's BNBD cache conventions become BBND at the external boundary.
- Its positive maximum-length attributes become the delivered contract's `-1/-1`.
- The optional native NoPE prolog and expanded 192/128 FlashAttn prefill require
  additional bindings outside the delivered FlashMLA API. This increment uses
  #16468's ordinary projection and absorbed-prefill path in both execution modes.
- No changes to K3 TP weight ownership, O-projection reductions or full-head
  replication are imported from the old Drafts.
- DCP history/current preparation and LSE merge helpers are ported as preparatory
  code. Public NTD output is explicitly transposed for #16468's TND exchange;
  LSE stays `[H,T]`. Runtime remains **gated to DCP=1**. Completing DCP backend
  capabilities, draft topology (especially GQA), stream overlap and combined
  DSpark/graph tests is follow-up work, not a supported configuration here.
- #16468's unrelated PD, PP/SP, MoE and fused-communication optimizations are
  not part of this increment.

## Validation

Host contract tests:

```bash
python tests/ut/attention/test_flash_mla_host_contract.py -v
python tests/ut/spec_decode/test_flash_mla_dspark_host_contract.py -v
```

They exercise real CPU tensors for buffer identity/refresh, device-task deferral,
query padding, all four documented head counts, strided aliases/nonzero offset,
causal/noncausal arguments, and mocked DSpark context writes. They do not execute
FlashMLA, an NPU scatter, Triton kernels, capture/replay, or collectives.

The DSpark suite executes selected repository methods with heavyweight imports
and upstream construction/dispatch stubbed. It covers target rotation handoff,
single rotation and weight ownership, raw auxiliary capture validation, backend
selection, FIA DP padding, FlashMLA boundary preservation, and restoration of
the capture factory after exceptions. It does not import the complete model or
prove that the deployed package, weights, or graph can run.

Required NPU matrix (all pending):

| Layer | Cases | Acceptance |
| --- | --- | --- |
| Delivered package | BF16/FP16; H=8,12,64,96; B=1,29,30,31,32; Q=1,2,4,8,16; KV=100k/128k | Output shape/dtype, finite values, `atol=rtol=0.02`; untested values are not unsupported |
| Small numerical reference | KV=0,1,127,128,129,257; unequal query lengths; mask 0/3; LSE off/on | Reference output and LSE, defined empty-row behavior; cache/guards exact |
| Cache boundary | Block=128; axis 0 and 1 strided; nonzero offset; cross-page reads/writes; PA_BBND and package-level PA_Nz | No hidden repack; cache and protection regions exact |
| Metadata | Installed wrapper/schema and Meta sizing; both attributes -1; changing lengths/table entries in same bucket | Two stages agree; no stale schedule or changed captured addresses |
| Layer output | Ordinary/fused output; RoPE/NoPE; live/dead rows, NaN/Inf in padding | Compare to reference, padding exactly zero |
| Gate/output Triton | T=0,1,31,32,33,63,64,65; width=1023,1024,1025,1536; strided rows, all-dead rows, BF16 | Eager/graph comparisons, tail guards, NaN/Inf masks; no performance claim before correctness |
| MRV2 model | Single rank then K3 four-card, fixed weights/topology/request; prefix reuse, multi-step, COW/zeroing | Deterministic baseline comparison plus correct cache lifecycle, not merely a listening port |
| Graph | Same input as eager; repeated replay, length/table changes, buckets/padding | Numerical/result agreement and correct buffer ownership |
| DSpark | Context precompute; acceptance/rejection length changes; target/draft eager and graph | Correct slots/positions/noncausal visibility; combined four-card service |
| DCP follow-up | Enable only after backend/topology work; interleave and history/current counted once; DSpark/graph combinations | Actual head geometry, stable LSE merge, no collective hang |

The matrix is coverage-driven, not an unbounded Cartesian product. Package
FP16/PA_Nz requirements do not change this model path's BF16/BBND scope.
Every result must record candidate/vLLM SHA, package and CANN/torch_npu versions,
hardware, topology, model, command, logs and remaining gaps. Older MRV1 eager
results are not acceptance evidence for this MRV2 candidate.
