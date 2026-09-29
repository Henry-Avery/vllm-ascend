# PR 13 hybrid state cache alignment and validation

Candidate parent: `3c3cca019c889d42d43e22f2c671edf5b7c8a720`.
Paired vLLM: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
Plan-1 comparison: `de31c53dc5b94ff246b17aa198404a082162c2f9`.

This is a local code candidate. CPU checks passed; NPU execution, four-host
serving, accuracy and performance remain untested. No service, automation or
publisher environment was changed. The unpublished work originally prepared
for PR 10 remains in a separate worktree and is not part of this change.

## Incident evidence and scope

The [PR 10 incident report](https://github.com/Henry-Avery/vllm-ascend/pull/10#issuecomment-5875001626)
describes `d7e950dc2`, vLLM 0.30.0, eager TP8/DP4, concurrent 60 requests and
8 output tokens. Its KDA state slot 3 overlaps a live MLA kernel page 174;
KDA updates precede latent NaNs and raw-logit NaNs. The report is supplied
production evidence, not a measurement of this candidate. Its original log
files were not recovered or executed locally. A healthy low-concurrency run
does not establish a fix.

The byte regression reconstructs the report's old geometry: 849 blocks,
27,648 convolution bytes per block, and 786,432 recurrent bytes per block.
The old 768-token manager page has physical size 912,384 and six 128-token
kernel pages. Thus the reported bad-element byte offset is
`174 * 152064 + 378 * 2 = 26459892`. The old state-major slot 3 covers
`[849 * 27648 + 3 * 786432, 849 * 27648 + 4 * 786432)`, which contains that
address. The page-major slot starts at `3 * 912384 + 27648`, outside that
MLA page. This reconstruction is clearly separate from candidate testing.

PR 13 already has page-major Mamba views. It does not reproduce the old
state-major layout, including for independent Mamba pools. This change
completes the consumers of those views; it does not import plan-1's allocator,
new conv operators, native state-copy operator, BNBD binding, or all-Flash
prefill route. Same physical block IDs may intentionally alias across cache
groups. Simultaneously owned distinct block IDs must occupy disjoint pages.

## Changes and remaining contracts

- Kimi prefill uses the existing `gather_ssm_states` / `scatter_ssm_states_`
  kernels. They touch only selected dense `[H,V,K]` payloads using real
  `stride(0)` and the view's storage offset. False initial-state flags gather
  zero. Negative and out-of-capacity IDs gather zero and never scatter.
  Physical slot zero remains valid for this helper; callers must remove empty
  graph rows through metadata `keep_meta`, rather than infer emptiness from ID.
- Valid scatter destination IDs must be unique. Gather may repeat IDs. Parallel
  writes to the same state slot are not serialized; this preserves the existing
  scheduler requirement. Index multiplication is 64 bit on device. No device
  `item()` or whole-cache normalization is introduced.
- General V2 COW copies page-strided dense payloads through the same kernels.
  It distinguishes views by pointer, shape, stride and dtype; matching pointers
  alone cannot remove a larger overlapping view. All sources are gathered
  before any destination write, including across aliased views. Temporary
  memory is the sum of selected copied payloads across distinct layer/views,
  plus index metadata. Measure this peak during the NPU stress test; it is not
  zero overhead. Layouts with non-dense inner rows retain the existing fallback.
- Page-view construction uses arithmetic strides, preserving nonzero raw
  storage offsets without allocating a tensor of the complete cache shape on
  CPU. It rejects misaligned dtype offsets/pages, payloads exceeding a page,
  truncated raw slices, and negative hybrid allocator offsets. MLA rejects
  fractional manager/kernel splits or kernel payloads exceeding their slots.
- The allocator's existing descriptor `offset`, `layer_stride` and
  `block_stride` contract remains in force. Unsupported geometry raises an
  error; the change does not silently allocate duplicate pools.

## Execution-path comparison with plan-1

| Path | Plan-1 fixed reference | PR 13 candidate and evidence |
| --- | --- | --- |
| Allocation and state reshape | Flash early return creates descriptor-strided raw pages; per-state slicing retains physical stride | Existing PR 13 hybrid backing and page-major views retained; production allocator/reshape tested with multi-layer offsets, shared aliases, padding and mixed dtypes |
| Recurrent KDA | State stride passed to operator | Existing AscendC path retains input/output alias; adapter, tiling and both kernels use `stateIn/OutStride0`; NPU equivalence still pending |
| Convolution | Reference has a separate Flash conv route | PR 13 retains pinned FLA conv; wrapper passes original view directly and native kernel reads/writes physical state strides |
| Prefill state read/write | Selected live-state byte/native copy and clear | Existing Triton state-index kernels now used by Kimi, including clear flags and masked invalid IDs; no advanced indexing on the cache |
| Prefill math | Fused normalization and VK state layout | PR 13's existing Q/K `l2norm_fwd`, `state_v_first=True`, gate and beta contracts retained |
| COW and zero | Physical strides and selected state/page lifecycles | Temporal Mamba copy uses payload pointers; general COW now handles alias unions. Paired upstream zeroer separately stores manager stride and virtual-page payload spans |
| Metadata and mixed batches | Metadata/main dependency and stable buffers | Existing PR 13 FIA host lengths, separate mixed buffers, executor reuse fence, attention wait before writes, padding slots and inactive-row handling retained; 44 CPU contract/lifecycle tests passed |
| Graph and performance | Different operator/graph route | Existing BBND Decode / FIA Prefill route retained; stream order, graph replay, zero-history/LSE merge, JIT warmup and performance require actual NPU validation |

The pinned Docker A5 dependency is FLA `26.9.1+deva4a7958`, release
`v26.9.1-beta2`, source
[`a4a795857868729fbdbb0118c1e06c8655a7b31b`](https://github.com/flashserve/flash-linear-attention-npu/tree/a4a795857868729fbdbb0118c1e06c8655a7b31b).
At that SHA, the Python `_stable.py` conv wrappers pass the original state to
native operators. `convStates.IgnoreContiguous()`, host `GetInputStride`, and
kernel `convStateStride0/1` addressing cover the physical stride. A5 arch35
uses that common GM addressing. This is source evidence, not proof of an
installed wheel/OPP or runtime behavior. Conv treats PAD -1 and null block 0 as
inactive, unlike the generic state-index helper's valid slot zero.

Paired vLLM `ced685` zeroer uses first-axis stride times the kernel/manager
ratio for manager stepping, with separate payload spans per virtual page.
PR 13's existing `patch_model_runner.initialize_kv_cache` flattens tuple/list
views and installs the Ascend general-copy function, so COW reaches the changed
helper rather than upstream's single-Tensor assumption. Runtime imports must
still resolve to those exact patches and binaries.

## Local validation

Environment: macOS CPU, Python 3.12, torch 2.8.0; no `torch_npu`, NPU or
complete importable paired vLLM runtime. Tests execute production bodies through
AST/leaf loading; external spec classes and device launches are explicitly
shimmed. The state-index CPU suite executes the production Triton kernel
bodies with Torch load/store primitives, not a copied indexing algorithm.

- Cache regressions: 22 passed. Running the same suite against PR 13 parent
  source exported from Git gives 11 failed / 11 passed. Its old external clear
  helper is shimmed too, so these are behavioral/validation failures rather
  than missing-import errors. Failures cover the
  newly enforced geometry, invalid indices, prefill consumer and COW alias
  union; this is a CPU regression comparison, not an NPU A/B result.
- Existing FlashMLA adapter contract and metadata lifecycle: 44 passed.
- Existing device-metadata tests: 14 passed with leaf production imports and
  forward-context/NPU stream shells. Direct collection without these shells
  fails because vLLM is absent. This does not validate real streams/events.
- Zeroer CPU assembly: paired `ced685` production `AttentionGroup`,
  `KVBlockZeroer` and Ascend V2 wrapper passed offset 64, ratios 1/3,
  payload-only zero and untouched prefix/suffix/page-gap checks. Only the
  device zero launch is simulated.
- The exact new NPU test's allocation-helper body was smoke-tested on CPU with
  paired `AttentionGroup`, production allocator/reshape and spec/module shells,
  including manager blocks 128/384 and nonzero offsets. Device operators have
  not been executed.
- `git diff --check` and all-file `bash format.sh ci` pass, including Python,
  Markdown and shell checks. The first full run found missing shellcheck;
  installing it enabled that check. Ruff then fixed an import order in the
  new NPU test; the corrected files were re-added before the final full run.

Portable CPU commands, from the repository root:

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_hybrid_state_page_layout.py -q
python -m pytest --confcutdir=tests/ut/attention \
  tests/ut/attention/test_flashmla_contract.py \
  tests/ut/attention/test_flashmla_metadata_lifecycle.py -q
python tests/ut/worker/v2/hybrid_zeroer_cpu_smoke.py \
  /path/to/paired-vllm/vllm/v1/worker/utils.py
bash format.sh ci
```

The complete UT/ST suite, build and NPU performance checks cannot run in this
CPU environment. The updated ordinary Kimi UT retains mocks for its device
gather/scatter while preserving its gate/operator assertions.

## NPU validation run sheet

These are prepared tests and acceptance criteria for an explicitly authorized
validation environment. They do not transfer publisher ownership or authorize
service restarts, installation, deployment, automation changes or remote load.

Before running, record the actual VA/vLLM SHAs, container and hardware IDs,
torch/torch_npu/CANN versions, loaded FLA package path/version and binary/OPP
identity, external FlashMLA wheel identity, existing flag values, eager/graph
mode and full launch arguments. Confirm that the live allocation, binding,
COW and zeroer symbols resolve to the intended patches. Record real shape,
storage base, storage offset, dtype, first-axis stride, descriptor offset,
physical page size, manager block size and kernel split ratio for every cache
group; compare intervals for simultaneously valid distinct block IDs.

On an existing compatible NPU test environment, run:

```bash
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_hybrid_page_state_npu.py
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_state_index.py
python -m pytest -sv \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_kimi_kda_recurrent_ascendc_npu.py
python -m pytest -sv tests/ut/ops/test_kimi_kda.py
```

The new integration test uses the production allocator/reshape and real
conv prefill/decode, KDA prefill/recurrent, gather/scatter, COW and V2 zeroer.
It compares selected state/output to dense test references, checks a live MLA
guard page and unselected byte/page gaps, and exercises actual state dimensions
for TP8 with new 128-token geometry plus 384/128 and 768/128 manager/kernel
splits. Raw byte snapshots separately guard state updates, COW and zeroing. Its
dense references are test-only and are not a production whole-cache workaround.

After operator checks, compare **A / B / A / B** in the authorized lab:
A is PR 13 parent `3c3cca`, B is this candidate; both use paired vLLM `ced685`
and identical software, hardware, model, flags and request inputs. Keep the
old PR 10 incident/environment as separate evidence, because changing both
the paired vLLM and VA cannot isolate this patch's effect.

| Workload | Required evidence |
| --- | --- |
| Low concurrency baseline | Request results, raw logits finite, state/MLA bytes finite, latency and memory baseline |
| Four nodes, per-node TP8, total DP4, eager, max sequences 64; C60/max tokens 8, at least 10 repeated rounds | Same prompt/request corpus and seeds for all four runs; per-request/rank/step raw logits and token IDs, anomaly counts, live MLA/state intervals, first nonfinite stage |
| Long/short mixed inputs and repeated block reuse | Pure prefill, decode and mixed steps; state resets, chunked prefill, selected state return, block COW/zero, unaffected live pages |
| Performance and memory | Peak allocated/reserved memory including per-batch COW gathers, allocation failure, steady-state latency distributions and throughput; account for first-use JIT separately |
| Graph follow-up when eager passes | PAD/empty rows, speculative metadata when enabled, captured buffer addresses, replay consistency, stream fences and FIA zero-history/LSE behavior |

Pass requires finite raw model logits, valid live-state and MLA payloads, no
cross-block byte corruption, and successful repeated reuse in the final B run.
HTTP 200/SSE completion or masking nonfinite outputs is insufficient. Save raw
measurements and failure logs before deriving summary rates; do not report a
runtime fix until this evidence exists.
