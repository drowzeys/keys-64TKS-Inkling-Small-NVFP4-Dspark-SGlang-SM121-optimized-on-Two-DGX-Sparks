# Roadmap

Committed follow-on campaigns (in order):

0. **STS + SPS calibration for cap-accept scheduling** (tooling shipped in `benchmarks/`): the
   confidence head is trained and present, but unusable until per-position temperatures are fitted
   and a cost table is profiled. The only remaining pure-config lever on accept.

1. **mxfp8 KV cache on the triton backend** (implemented, gates pending): ~1.94× KV pool. Note the
   port keeps upstream's kernels byte-untouched and puts all quantized paths in a twin-kernel module
   that is only imported when quantization is on.

2. **NVFP4 KV cache on the triton backend** — the capacity unlock toward true 1M in-flight tokens.
   The pool/storage side exists (fa4 uses it); the gap is `q/k/v_descale` handling + fp4
   block-scale dequant in the triton extend/decode/verify kernels. Donor code identified:
   upstream PR #32333 (DSV4 fp4 triton dequant) + a fleet-internal MLA nvfp4-KV triton mod
   (proves fp4-KV triton kernels run on this silicon). Acceptance gate: quality parity +
   long-context needle tests (quantized KV is not byte-lossless by definition).
   **Status: scoped + in progress** — full kernel-level plan in `KV-QUANT-TRITON-PLAN.md`
   (mxfp8 write path already exists for the triton page-1 layout; ~240 lines across 4 files
   for full mxfp8, ~250 more for nvfp4; capacity payoff 1.94×/3.5×).
2. **A4Q activation quantization** (jethac, Blackwell ISA — GB10 shares the RTX 5090 ISA):
   prefill/TTFT acceleration, decode-neutral. Ported after NVFP4-KV.
3. **DSpark SPS cost table** (`SGLANG_DSPARK_ENABLE_SPS_RECORD` → fit → `--speculative-dspark-sps-table-path`):
   confidence-scheduled variable verify windows for better batch throughput.
4. **Helion native autotune** (`HELION_AOT_AUTOTUNE=create`) to replace the seeded sm_100 configs.
