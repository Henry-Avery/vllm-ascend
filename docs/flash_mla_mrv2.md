# MRV2 target FlashMLA (experimental)

`VLLM_ASCEND_ENABLE_FLASH_MLA=1` enables the external `cann_ops_transformer`
two-stage FlashMLA path. It defaults to `0`. The worker environment must provide
both `flash_mla_with_kvcache_metadata` and `flash_mla_with_kvcache`, including
the metadata operator's Meta implementation. There is no silent fallback.

This initial integration requires MRV2, the hardware profile supporting the
token-fused MLA cache, unquantized FP16/BF16 attention, 512 latent plus 64
positional lanes, and 8/12/64/96 **actual local** query heads. DCP and PCP must
both be 1; speculative decoding (including DSpark) and DBO must be disabled.
Weight quantization is separate from attention/cache quantization.

## Scope and data contract

- Queries of up to 16 tokens use the decode path. Longer prefill and mixed
  batches' prefill suffix retain the existing mainline implementation.
- The original runner-owned cache is `[pages, 128, 1, 576]`, `PA_BBND`.
  Page/token strides may include gaps; nonzero storage offsets are retained.
  The consumer does not compact, allocate, zero or copy the KV cache. Cache
  writers, slot mappings, page numbering, COW and zeroing remain dependency-owned.
- The metadata operator consumes current device `seq_lens`, cumulative
  `query_start_loc` and actual query counts. Both calls use `max_seqlen_q=-1`,
  `max_seqlen_kv=-1`, `head_dim_v=512`, identical mask mode and `layout_q=TND`.
  The main call uses `layout_kv=PA_BBND`, `layout_out=NTD`, the layer's scale,
  and `return_softmax_lse=False`.
- Mask mode is 3 for causal attention with an int8 upper-triangular
  2048-by-2048 mask; mode 0 has no mask. The existing per-layer RoPE/NoPE
  preprocessing is preserved. No heads are replicated or padded for FlashMLA.
- NTD latent output feeds the existing V up-projection, gate and O projection.
  Invalid latent/gated padding rows are selected to zero, including NaNs.
  The fused decode prolog is disabled so the existing BBND writer and unfused
  projection weights remain available.

## Capture and replay

MRV2's `build_attn_metadata` prepares stable buffers before eager forward,
capture, and FULL replay. Buffers are owned by each attention-group builder
and keyed by batch capacity, token capacity, page-table width and causality.
The external operator's Meta output determines the exact schedule capacity;
an incompatible runtime output raises before copying it.

Device input preparation, schedule generation, forward/replay, and subsequent
reuse run in order on the execution stream. When vLLM switches between its
capture and execution streams, the new preparation stream waits for the old
one before overwriting any buffer. The consumer must use the preparation
stream. FULL capture records group-to-buffer identities; replay checks those
identities before launching the graph. This intentionally serializes metadata
and attention; it does not implement cross-stream overlap.

The legacy FIA graph updater skips only target entries carrying FlashMLA
metadata. Other backends and draft paths retain their updater behavior.
CPU length mirrors remain available to the existing prefill and FIA paths.

## Validation boundary

Run the CPU contract tests without importing the NPU test harness:

```sh
python tests/ut/attention/test_external_flash_mla.py
```

These tests use real CPU tensors and spy operators/streams, and execute the
actual forward and graph-binding methods in isolation. They cover layout,
strides, offsets, shared arguments, lengths, padding, metadata capacity,
buffer reuse, scope guards and mixed-prefill postprocessing. They do **not**
prove the external binary ABI, numerical accuracy, actual ACL Graph capture,
GE/compilation compatibility, cache-writer kernel correctness, or performance.

Before deployment, validate the installed package's schema/Meta, strided
BBND numerical output, RoPE and NoPE layers, eager versus repeated graph
replay, mixed prefill, TP, and COW/zeroing on the pinned candidate. Package-level
PA_Nz support and the full long-context matrix remain separate unverified
capabilities; this integration neither implements nor rejects those package
capabilities on behalf of the external operator.
