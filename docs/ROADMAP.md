# Roadmap

Committed follow-on campaigns (in order):

0. **STS + SPS calibration for cap-accept scheduling** (tooling shipped in `benchmarks/`): the
   confidence head is trained and present, but unusable until per-position temperatures are fitted
   and a cost table is profiled. The only remaining pure-config lever on accept.

1. ~~**mxfp8 KV cache**~~ — ✅ DONE (1.94× pool, superseded by fp4).
2. ~~**NVFP4 KV cache on the triton backend**~~ — ✅ **DONE: 3.12× pool (1.1M tokens), shipped in
   `patches/kv-quant/`.** See the README headline section. Remaining follow-ons: long-context needle
   tests at 500K–1M, concurrency benchmarks under fp4, and a TTFT check (the draft KV write falls back
   to a per-layer python loop under quantized dtypes — correctness-neutral, possible prefill cost).

3. **~~NVFP4 KV~~ (historical note)** — the capacity unlock toward true 1M in-flight tokens.
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
4. **DSpark cap-accept calibration — BLOCKED on this build.** The confidence (STS) recorder only runs
   inside the cap-accept planner, but the planner degenerates to verify-all until an SPS cost table
   exists (`sps_table=uninitialized ... zero scheduling gain`), and the SPS recorder writes through an
   info-dumper with no retrievable output path exposed. So cap-accept cannot be calibrated here and
   measures worse than static block-7. Tooling for both fits is in `benchmarks/` if a future build
   exposes the dump path.

5. **Draft finetune** — the only remaining lever that raises accept fundamentally (a 0.9B draft
   predicting a 276B target caps around accept 3.5). Everything else is scheduling.
4. **Helion native autotune** (`HELION_AOT_AUTOTUNE=create`) to replace the seeded sm_100 configs.
