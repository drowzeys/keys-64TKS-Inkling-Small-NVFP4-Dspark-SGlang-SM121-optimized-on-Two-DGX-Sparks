# Every wall we hit on GB10 (sm_121a), with fixes

Chronological debug ledger from the bring-up and optimization campaign (2026-07-31, ~30 boot cycles).

**Before you trust any performance comparison in here or anywhere else, read
[MEASUREMENT-PROTOCOL.md](MEASUREMENT-PROTOCOL.md) — wall 18 is the one that cost the most time.**
If your boot dies, find your symptom here.

| # | Symptom | Root cause | Fix (baked unless noted) |
|---|---------|-----------|--------------------------|
| 1 | `NCCL error: invalid usage` at comm init, zero NCCL log output | Container can't see RDMA devices; `NCCL_NET=IB` then hard-fails (`NET/IB : No device found`) | `--device /dev/infiniband --cap-add IPC_LOCK` in docker run (launch script) |
| 2 | ~15 GB vanish at NCCL init | Bundled NCCL 2.28.9 alloc behavior on GB10 unified memory | pip `nvidia-nccl-cu13` ≥2.30 in the bake |
| 3 | `ValueError: Invalid forward mode: EXTEND for CUDA Graph replay` at boot | This build has a separate PREFILL graph phase; triton backend can't replay EXTEND | `--disable-prefill-cuda-graph` (launch script) |
| 4 | `triton OutOfResources: shared memory Required: 110592, limit: 101376` on first request | Inkling grouped-GEMM tuned for B200 (228 KB smem); GB10 has 99 KB | `num_stages` 3/4→2 in `inkling_moe.py` |
| 5 | `Helion kernel not tuned yet ... silu_and_mul_interleaved_sm_121.json` | No sm_121 Helion configs shipped | seed from sm_100 copies (bake); proper `HELION_AOT_AUTOTUNE=create` retune is a perf TODO |
| 6 | `Error occurred when running GEMM ... sm100f` in `trtllm_fp4_block_scale_moe` | FlashInfer TRT-LLM routed FP4 MoE = sm_100-only cubins | `--moe-runner-backend marlin` |
| 7 | `'PackedTopKOutput' object has no attribute 'topk_weights'` | Inkling gate emits packed top-k that only the TRT-LLM runner consumes | `emit_packed_topk = False` in `inkling_common/moe.py` |
| 8 | Fluent but WRONG output (temp 0), e.g. hallucinated facts, with cutlass MoE | `flashinfer_cutlass` FP4 MoE silently mis-computes on sm_121 | marlin only. Do not trust cutlass numerics on this arch without a lossless check |
| 9 | Spec-on: near-zero accept (~1.0) with triton draft backend | Draft attention backend inherits `speculative_num_draft_tokens = gamma+1`, proposer emits gamma rows → strided OOB KV reads ([sgl-project/sglang#30555](https://github.com/sgl-project/sglang/issues/30555)) | `-1` on draft workers in `triton_backend.py`. NOTE: do NOT also pin the draft ServerArgs (the issue's suggested fix) — current builds already subtract 1 in graph sizing and you'll double-correct (symptom: `shape '[N, 7, -1]' is invalid` at graph capture) |
| 10 | Spec-on: output degenerates into repeating/prompt-replay whenever accept > 1; spec-off clean | **Conv-state commit corruption** (novel, upstream-worthy): DSpark passes `commit_lens` EXCLUDING the bonus token, but the sconv window commit uses it as the last-accepted step index → commits the window one token short, state regresses cumulatively. Only the NON-symm-mem save path is affected — every B200 recipe runs `--enable-torch-symm-mem` (its fused kernel takes another path), so upstream never sees it. GB10 cannot run symm-mem cross-node. | torch-native commit in `inkling.py` gated by `INKLING_TORCH_CONV_COMMIT=1` with `INKLING_COMMIT_STEP_BIAS=1` (window at `step+1`, clamped — the clamp exactly handles full-accept). Lossless verified byte-exact vs spec-off |
| 11 | `Capture cuda graph failed: shape '[N, 7, -1]' invalid` | Draft decode-graph tier vs sampler gamma mismatch (see #9 note) | fixed by the same triton_backend `-1` (and NOT pinning ServerArgs) |
| 12 | Page-size 128: verify-path corruption even with all fixes | fa4-oriented 128-token pages mis-drive the triton verify path | `--page-size 1` (champion). Page-128 + triton remains unfixed upstream territory |
| 13b | `TypeError: TritonAttnBackend.forward_extend() got an unexpected keyword argument 'q_descale'` on first request with `--kv-cache-dtype mxfp8` (or nvfp4) | KV-quant descale plumbing exists only for the fa4/flashinfer backends; the triton backend (the ONLY lane for Inkling on sm_121) has no `q_descale` support | No KV quantization on GB10 until upstream adds descale args to the triton backend. bf16 KV only — this closes both the mxfp8 (official long-ctx tier) and nvfp4-KV routes on this hardware |
| 13 (diagnosed) | — | Root-cause class for wall 13 found during KV-quant scoping: the triton kernels silently cast `q.to(K_Buffer.dtype)` / `p.to(v.dtype)` — with a quantized KV dtype the QUERY gets crushed to fp8 too, producing the `!!!` garbage. See docs/KV-QUANT-TRITON-PLAN.md | Fix ships with the mxfp8-triton port (roadmap #1, in progress) |
| 13 | Output becomes `!!!!!...` garbage with `--kv-cache-dtype fp8_e4m3` (accept looks HIGH — garbage drafts vs garbage targets accept trivially) | FP8 KV silently corrupts on this engine/arch (matches sglang #19603-class behavior on hybrids) | bf16 KV only. FP8 KV was the only path to 1M-token pools on 2 nodes — therefore 1M is out of reach; max self-consistent context ≈ 384K |

| 14 | `RuntimeError: Tensor match failed ... causal_conv1d.cuh:183` at first request | `SGLANG_RAGGED_VERIFY_MODE=compact` reshapes verify metadata; Inkling's sconv JIT kernel has a fixed shape contract | leave the env UNSET. **Correction: only `compact` crashes.** `cap-accept` runs fine — see wall 17 for its real caveat |
| 15 | Accept drops to ~3.3 (from 7.5) with `SGLANG_RAGGED_VERIFY_MODE=static` | explicit static mode degrades DSpark verify vs the unset default | leave the env UNSET; also beware: forgetting `--page-size 1` reproduces a ~2.2-accept collapse silently (launcher now defaults PAGE=1) |

| 16 | Native MTP / EAGLE 8-1-9 (the official cookbook recipe) will not run | Two independent blockers on 2× GB10: **(a) memory** — the MTP heads are a separate 4.2 GB `mtp.safetensors` (160 tensors) that loads on top of the 156 GB target, leaving a fixed −6 GB deficit for the 9-token draft window *regardless* of `--max-running-requests` or context; **(b) code** — at reduced width (k=3) it boots past memory then dies `KeyError: 0` in `layers_mapping` (the EAGLE path isn't wired for Inkling's hybrid-SWA layer map in this build) | Use DSpark. Its 1.7 GB external draft exists precisely for memory-constrained deployments. Native MTP needs more memory *and* an upstream fix |
| 17 | `SGLANG_RAGGED_VERIFY_MODE=cap-accept` (confidence-scheduled adaptive width) measures *worse* than static block-7 | The planner logs `sps_table=uninitialized` and the confidence head is uncalibrated, so it budgets proposal width blind | Needs BOTH calibration artifacts: an **SPS cost table** (`--speculative-dspark-sps-table-path`) and an **STS confidence calibration** (`--speculative-dspark-confidence-sts-path`, collected via `SGLANG_DSPARK_STS_COLLECT_PATH`). Tooling: `benchmarks/sps_calibrate.py`, `benchmarks/sts_fit.py` |
| 18 | Benchmarks disagree wildly between runs / sessions / reboots — same config, same image, same weights | **Not a bug in the stack.** Target forward is nondeterministic at temp 0 (triton split-KV / marlin reduction order on sm_121a), and accept depends on the *style* of continuation each run lands on | Use `docs/MEASUREMENT-PROTOCOL.md` + `benchmarks/accept_probe.py` (32 samples, mean±se). This single issue burned more time in this campaign than every real bug combined |

| 19 | `ValueError: MXFP8/FP4 KV cache requires K and V scale tensors` on the FIRST request (boot succeeds) | Inkling has **three** KV-pool writers; quantizing in the attention backend misses DSpark's hidden-state injector (`dspark_kv_inject.py` → `models/dspark.py:658`) which writes KV straight from the model file | Quantize **inside the pool** (`patches/kv-quant/srt/mem_cache/kv_quant_pools.py`), which covers all writers. Also note `_fused_kv_write_bundle` probes `get_key_buffer()` — with the stock FP4 pool that probe alone dequantizes the entire pool |
| 20 | fp4 KV pool comes out ~1.78× smaller than expected | `_element_size(float4_e2m1fn_x2)` is **1**, so fp4 bytes/token get over-counted unless the pool configurator special-cases it | fixed in `patches/kv-quant/srt/model_executor/pool_configurator.py` |
| 21 | `--kv-cache-dtype nvfp4` fails or is ignored on the triton lane | In this image `nvfp4` = flashinfer/trtllm recipe (e4m3 block scales + checkpoint global scales) | use **`fp4_mx_block16`** (e2m1 + uint8 biased-exponent scale/16). Identical capacity |
| 22 | Half of every attention head reads as garbage under fp4 (~3.4e38) | `_decode_softmax_reducev_fwd` derives `Lv` from `v_buffer.shape[-1]`, which is D/2 for packed fp4 | zero-element shape proxy; caught by the GPU test suite before it ever reached a serve |

## Constraints discovered (not bugs — architecture facts)

- **Inkling attention asserts `fa4|triton`** (`inkling_common/attn.py`). fa4 = sm_100 only ⇒ triton is
  the ONLY lane on GB10. flashinfer target attention is rejected by the model.
- **Radix cache cannot be disabled** for Inkling (`assert not disable_radix_cache`), and it **requires**
  `--mamba-radix-cache-strategy extra_buffer` — which makes `--enable-linear-replayssm` unusable
  (it requires `no_buffer`): mutually exclusive.
- **Cross-node TP on vLLM's NVIDIA-optimized Inkling path requires MNNVL fabric** (Lamport
  reduce-scatter+conv op); `LAMPORT_RS_SCONV=0` disables it (vLLM path, not used here).
- Temp-0 outputs differ slightly between page sizes (accumulation order). Compare lossless
  references only within the same page size.

## Upstream-worthy findings

1. **#30555 done right** — the fix belongs in the attention backend's draft-worker width, not the
   draft ServerArgs (which now double-corrects graph capture). It fixes OOB draft KV reads (which pin accept near 1.0) and unblocks decode graphs, via the single
   `-1` in `triton_backend.py`. (Campaign-era accept figures quoted around this fix were single-run
   measurements — see wall 18.)
2. **DSpark conv-commit off-by-one** (wall #10) — reproducible on any hybrid-conv model running
   DSpark without symm-mem. Repro: temp-0 prompt, watch output degrade into prompt-replay
   exactly when accept_len > 1; disable commit (`INKLING_NOOP_CONV_COMMIT=1`) and the replay
   disappears (stale-state artifacts remain); apply +1-biased commit and output is byte-exact
   lossless vs spec-off.
