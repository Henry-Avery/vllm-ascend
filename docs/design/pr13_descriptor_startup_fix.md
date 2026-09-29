# PR 13 query-head and descriptor startup correction

Parent: `b4af8712a4d08e424ca0729acc11e9fe667c44fc`.
Paired vLLM: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
Plan-1 reference: `de31c53dc5b94ff246b17aa198404a082162c2f9`.

## Reported startup failure

The [publisher's PR 13 report](https://github.com/Henry-Avery/vllm-ascend/pull/13#issuecomment-5886709324)
records A5 TP8/DP1, five dummy Kimi-K3 layers, eager V2, and both
`VLLM_USE_V2_MODEL_RUNNER=1` and `VLLM_ASCEND_ENABLE_FLASH_MLA=1`.
The configured FlashMLA log precedes an allocator failure on all eight
workers, before health checks. Layout is **LBNHC**, with MLA page size
884,736 bytes, block stride 10,616,832, layer stride 570,474,233,856,
and 53,733 manager blocks. These are supplied runtime observations;
the patched candidate has not been executed on that machine.

There are two distinct source defects:

1. `AscendMLAAttentionSpec.num_heads` stored per-rank query heads, overriding
   upstream `AttentionSpec.num_heads`, which means KV head slots. V2 passes
   12 query heads while MLA page sizing still uses one latent KV head.
   The paired planner consequently constructs strides twelve times the
   page-budget geometry. Its real source exactly reproduces both reported
   strides. The query-head override entered the integration history through
   `d726b7d95` (`Restore BBND MLA cache protocol`). Plan-1 keeps the upstream
   KV-head meaning.
2. The older allocator only accepted contiguous per-layer descriptors.
   That guard originated at `fd815467c` (`main2main vllm 0828 (#14872)`)
   and persisted across both dual-source integrations. Legal block-outer
   descriptors also fail it, independently of the query-head error.
   Plan-1 has a descriptor allocation route before that guard; the previous
   port and review omitted it. Prior allocator tests only constructed
   layer-major descriptors and missed this limitation.

Neither failure is explained by an unset FlashMLA flag or a conflict during
the additional fixed-main merge. Simply accepting twelvefold strides would
retain contradictory planner/page-budget metadata and possible allocation
overruns. The old PR 10 state-major overlap is a separate incident; this
change retains PR 13's page-major KDA states.

## Correction and ownership contract

Query heads are now `num_query_heads`. The inherited `num_heads` remains
the upstream KV-head property. Ascend spec construction, merge/replace,
and V1/V2 BBND backend selection use the appropriate field. For the same
53,733-block LBNHC input, the planner now produces block stride 884,736
and layer stride 47,539,519,488. No flag defaults or planner budgets change.

The hybrid allocator retains descriptor `offset`, `layer_stride`, and
`block_stride` in byte-row views for Mamba and single-raw MLA. Dense
per-layer descriptors retain their existing flat representation. Bounds,
page capacity, and dtype alignment remain checked. Other strided hybrid
attention formats still fail explicitly rather than receiving an incorrect
flat slice.

Mamba convolution and recurrent-state payloads keep their original manager
stride and offsets. MLA manager pages split into 128-token kernel pages:

- Layer-outer LBHNC/LBNHC requires dense manager pages per layer. Kernel
  stride is page size divided by the manager/kernel ratio.
- Block-outer BLHNC/BLNHC organizes an attention-owned manager region as
  `[kernel subpage, layer, payload]`. Kernel stride is descriptor block stride
  divided by the ratio; the within-manager layer offset divides by that
  ratio too. An allocation-base storage offset is never divided.
- Invalid divisibility, non-affine layer regions, and unsupported split
  layouts fail. There is no cache-pool normalization or byte movement.

Cache groups may alias the same free physical manager ID. Simultaneously
owned MLA and KDA manager IDs must differ. Writer components and the
external reader share the resulting fused MLA storage/view. COW snapshots
all selected kernel subpages before writing; the paired zeroer builds its
segments from the final cache stride. Both therefore advance manager IDs
by the original descriptor block stride.

## Local validation and limits

`tests/ut/worker/v2/test_hybrid_descriptor_layout.py` executes the real
Ascend spec class and V2 spec constructor, then the unmodified paired
planner, allocator entry point, reshape, COW, and zeroer metadata bodies.
The fixture JSON locks upstream source excerpts to `ced685`, with complete
source-file SHA-256 values. External layer/config objects, Mamba spec
metadata, NPU writer/attention launches, and zero device stores are explicit
CPU substitutes. This does not claim numerical NPU validation.

Coverage includes BLHNC/LBHNC/LBNHC, manager/kernel ratios 1/3/6,
FlashMLA flag off/on with identical descriptors, both backend layouts,
supported and unsupported query-head counts, two layers, nonzero allocation
storage offset, real KDA payload shapes/dtypes, shared backing, source reads
after KDA updates, COW/zero payload-only writes, and protected byte gaps.
The exact reported geometry is checked without a physical allocation of
hundreds of GiB; a meta allocation exercises the production bounds at that
size. Real spec merge and malformed geometry have regression coverage.

Commands used:

```bash
python -m pytest --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_hybrid_descriptor_layout.py \
  tests/ut/worker/v2/test_hybrid_state_page_layout.py -q
python -m pytest --confcutdir=tests/ut/attention \
  tests/ut/attention/test_flashmla_contract.py \
  tests/ut/attention/test_flashmla_metadata_lifecycle.py -q
bash format.sh ci
```

The cache suites pass 59 tests; the existing FlashMLA suites pass 44.
The exported `b4af8712` source reproduces the reported twelvefold geometry
and its original allocator error using the same real spec/planner fixture.
All-file formatting is required before publication.

Full paired-runtime UT/ST, build, actual NPU writer/FIA/external FlashMLA,
eager/graph serving, concurrency, memory, accuracy, and performance remain
unverified. The prepared NPU test uses the renamed query metadata; its
device execution remains pending. Acceptance starts with the publisher's
exact startup command and fixed paired SHA, then the existing hybrid-state
stress and numerical checks. Only PR 13 receives this append; PR 15 at
`d843e2e5d865926ba27c6d3222993b61db48380f` needs a separate follow-up.
