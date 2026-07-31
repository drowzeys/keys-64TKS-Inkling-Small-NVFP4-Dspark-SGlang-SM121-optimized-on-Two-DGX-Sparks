# keys-64TKS · Inkling-Small-NVFP4 + DSpark · SGLang · sm_121a · Two DGX Sparks

**64 tok/s single-stream, lossless speculative decoding, on two desktop DGX Sparks.**
The first known deployment of [thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
(276B total / 12B active MoE) with the [RadixArk DSpark speculator](https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview)
off datacenter Blackwell — TP=2 over a single 200G link between two GB10 (sm_121a) machines.

> **Agents: start at [`specs/001-oneshot-install/tasks.md`](specs/001-oneshot-install/tasks.md).**
> It is a gated, ordered task list (spec-kit style). Execute it verbatim and you get this
> exact serve without re-fighting the ~14 boot cycles it took to find these fixes.

## Performance (measured 2026-07-31, this exact recipe)

Hardware: 2× DGX Spark (GB10, 128 GB unified, ~273 GB/s each), one 200G CX7↔CX7 link, NFS weights.

### Single-stream, temp 0, decode CUDA graphs on

| Workload | tok/s | DSpark accept len (of 8) |
|---|---|---|
| Raw continuation (essay-like, `/generate`) | **64.6** | 7.31 |
| Chat `list` (C1 median) | 41.2 | — |
| Chat `reading` (C1 median) | 24.5 | — |
| Chat `essay` (C1 median) | 22.1 | — |
| No-spec eager baseline | ~13 | — |

Chat-path numbers are lower than raw continuation because Inkling's reasoning tokens draft
harder. Never quote a single tok/s without its task class.

### Concurrency (256 new tokens/req, temp 0.7, `--max-running-requests 8`, graph tiers {1,2,4,8})

| Task | C1 | C4 aggregate | C8 aggregate | accept @C8 |
|---|---|---|---|---|
| list | 37.6 | 32.9 | **61.6** | 2.85 |
| essay | 30.9 | 38.6 | **44.1** | 2.72 |
| reading | 16.3 | **71.1** | 61.8 | 2.66 |

Raw CSV: [`benchmarks/concurrency_results.csv`](benchmarks/concurrency_results.csv) ·
harness: [`benchmarks/concurrency_bench.py`](benchmarks/concurrency_bench.py).
Reference: LMSYS reports 648 tok/s (DSpark) vs 288 (no-spec) on 8× B200 TP8 — a rig with
~29× this pair's aggregate memory bandwidth. Bandwidth-normalized, these two desktop boxes
outperform that reference.

## What's in the box

| Path | What |
|---|---|
| `specs/001-oneshot-install/` | spec / plan / **tasks.md** — the gated one-shot install |
| `scripts/bake-image.sh` | digest-pinned upstream image + all patches → `local/sglang-inkling:gb10` |
| `scripts/inkling-sglang-launch.sh` | per-rank launcher; defaults = validated champion config |
| `patches/files/` | the 5 net-patched SGLang files (byte-exact from the validated image) |
| `patches/all-patches.diff` | the same as a reviewable 162-line unified diff |
| `docs/BUGS-AND-FIXES.md` | all 12 walls with symptoms → root causes → fixes, incl. 2 upstream-worthy bugs |
| `benchmarks/` | bench harness + raw results |

## The two bugs you'd lose days on (both fixed here)

1. **DSpark draft gamma mismatch** ([sglang#30555](https://github.com/sgl-project/sglang/issues/30555), fixed *correctly*):
   one `-1` in the triton backend's draft-worker width takes accept from ~2.5 to **7.31** and
   unblocks decode CUDA-graph capture. The issue's own suggested ServerArgs pin now double-corrects — don't use it.
2. **DSpark conv-state commit off-by-one** (novel): on the non-symm-mem path (i.e., everything
   that isn't a B200-class single-node), the sconv verify commit lands one token short and output
   degenerates into prompt-replay whenever accept > 1. Fixed with a +1-biased torch-native commit —
   **verified byte-exact lossless** against spec-off at temp 0.

## Champion config (encoded as launcher defaults)

`--attention-backend triton` (Inkling asserts fa4|triton; fa4 is sm_100-only) ·
`--moe-runner-backend marlin` (the ONLY numerically-correct NVFP4 MoE on sm_121 — cutlass
silently miscomputes, trtllm hard-fails) · `--page-size 1` · DSpark block 7, draft unquantized ·
decode graphs {1,2,4,8}, prefill graphs off, piecewise off · `--mem-fraction-static 0.85` ·
64K ctx (model supports 1M; the draft is 64K-adapted) · NCCL ≥2.30 over RoCE, `NCCL_NET=IB`,
CUMEM/NVLS off, `/dev/infiniband` passed through.

## Serving profiles (context vs speed — measured)

Declared context resizes the hybrid pools AND stretches the draft's rope scaling, so long-context
costs DSpark speed even on short prompts. Pick a profile; all are lossless (bf16 KV, byte-exact gate passed):

| Profile | Launch | KV pool (tokens) | Essay 256tok @ temp0 | accept |
|---|---|---|---|---|
| **SPEED (default)** | `CTX=65536` | 674,816 | **64.6 tok/s** | 7.31 |
| MAX-CONTEXT (self-consistent) | `CTX=393216` | ~410K (≥ ctx) | between | between |
| MAX-DECLARED | `CTX=524288` | 310,606 (**effective in-flight cap**) | 26.8 tok/s | 2.59 |

- The pool is shared across `--max-running-requests 8`; at 512K declared, one ~300K stream or 8×~38K.
- **True 1M is not reachable on 2× GB10**: bf16 pools cap out as above, and **FP8 KV
  (`--kv-cache-dtype fp8_e4m3`) produces catastrophic garbage output on this stack** (`!!!!…`) —
  documented as wall #13. Do not use it.
- The DSpark draft is 64K-adapted: accept also fades with actual prompt depth past ~64K.

## Provenance

Upstream image: `lmsysorg/sglang@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`
(`dev-cu13-inkling-dspark`, pushed 2026-07-30). Model: thinkingmachines/Inkling-Small-NVFP4.
Draft: RadixArk/Inkling-Small-DSpark-Preview. All patches are against files inside that image;
`patches/all-patches.diff` is the complete delta. Not affiliated with LMSYS, Thinking Machines,
RadixArk, or NVIDIA — this is an independent field port.
