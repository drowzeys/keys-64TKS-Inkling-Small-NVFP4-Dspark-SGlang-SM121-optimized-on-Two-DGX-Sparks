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
3. ~~**A4Q native-fp4 attention**~~ — ❌ **NOT APPLICABLE to Inkling-Small on SGLang.** Evaluated and
   rejected on four independent grounds:
   - **KV width 1024** (8 kv-heads × 128 head_dim). A4Q's gain scales with KV width and needs
     **≥4096** to amortize its quantization overhead — at 1024 it is expected to be a net loss.
   - **Implementation is a FlashInfer FA2 kernel**, but Inkling asserts `attention_backend in
     ("fa4","triton")` and fa4 is sm_100-only ⇒ triton is the only legal lane on GB10 (wall 1).
     There is no seam to attach it to.
   - **It is a vLLM integration**, not SGLang.
   - **Inkling is sliding-window (512) + short-conv**, so its attention prefill is closer to linear
     than quadratic; A4Q's advantage comes precisely from accelerating quadratic prefill.

   A4Q remains excellent for *dense-GQA, wide-KV* models on vLLM (measured elsewhere on this fleet:
   Nemotron-3-Omni TTFT −22% @60K scaling to −39% @256K). It is simply the wrong tool for this model.
4. ~~**DSpark cap-accept calibration**~~ — ✅ **RESOLVED: it works, and it still loses.** Both
   artifacts were produced (SPS table with `match_fraction=1.00`; STS from 19,871 samples, ECE
   0.03651→0.03453 with a joint coordinate-descent fitter that beats the shipped greedy one).
   Calibrated cap-accept reaches accept 3.39 ± 0.18 — statistically level with static — but at
   23.4 ± 1.3 tok/s vs 34.7 ± 1.5. Full mechanism in
   [DSPARK-CALIBRATION-FINDINGS.md](DSPARK-CALIBRATION-FINDINGS.md). **Keep static block-7.**
   *(superseded note, kept for context:)* The confidence (STS) recorder only runs
   inside the cap-accept planner, but the planner degenerates to verify-all until an SPS cost table
   exists (`sps_table=uninitialized ... zero scheduling gain`), and the SPS recorder writes through an
   info-dumper with no retrievable output path exposed. So cap-accept cannot be calibrated here and
   measures worse than static block-7. Tooling for both fits is in `benchmarks/` if a future build
   exposes the dump path.

5. **Draft finetune** — the only remaining lever that raises accept fundamentally (a 0.9B draft
   predicting a 276B target caps around accept 3.5). Everything else is scheduling.
4. **Helion native autotune** (`HELION_AOT_AUTOTUNE=create`) to replace the seeded sm_100 configs.
