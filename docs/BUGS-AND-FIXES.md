# Every wall we hit on GB10 (sm_121a), with fixes

Chronological debug ledger from the original bring-up (2026-07-31, ~14 boot cycles).
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
   draft ServerArgs (which now double-corrects graph capture). Our accept went 2.5 → 7.31 and
   decode graphs unblocked with the single `-1` in `triton_backend.py`.
2. **DSpark conv-commit off-by-one** (wall #10) — reproducible on any hybrid-conv model running
   DSpark without symm-mem. Repro: temp-0 prompt, watch output degrade into prompt-replay
   exactly when accept_len > 1; disable commit (`INKLING_NOOP_CONV_COMMIT=1`) and the replay
   disappears (stale-state artifacts remain); apply +1-biased commit and output is byte-exact
   lossless vs spec-off.
