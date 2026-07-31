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

**V2 champion (current defaults): 68.2 tok/s @ accept 7.53, lossless, 427K-token pool at 512K declared context**
(`CTX=524288 · MEMFRAC=0.87 · MAXREQ=16 · graph tiers 1–16 · page 1 · draft-ctx cap`). v1 numbers below for history.

| Workload | tok/s | DSpark accept len (of 8) |
|---|---|---|
| Raw continuation (essay-like, `/generate`) | **64.6** | 7.31 |
| Chat `list` (C1 median) | 41.2 | — |
| Chat `reading` (C1 median) | 24.5 | — |
| Chat `essay` (C1 median) | 22.1 | — |
| No-spec eager baseline | ~13 | — |

Chat-path numbers are lower than raw continuation because Inkling's reasoning tokens draft
harder. Never quote a single tok/s without its task class.

### Concurrency (256 new tokens/req, temp 0.7)

V2 (`MAXREQ=16`, tiers 1–16, 0.87): | v1 (`MAXREQ=8`, tiers {1,2,4,8}, 0.85):

| Task | C1 | C4 agg | C8 agg (v2) | C8 agg (v1) |
|---|---|---|---|---|
| list | 26.5 | 44.9 | **64.8** | 61.6 |
| essay | 23.1 | 36.2 | **51.7** | 44.1 |
| reading | 23.0 | 54.7 | **67.8** | 61.8 |

**Accept is temperature-dependent**: ~7.5 at temp 0 vs ~2.3-2.8 at temp 0.7 (sampling diversity
fights the draft) — that's why C1 rows here are lower than the temp-0 single-stream numbers.

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
| `docs/BUGS-AND-FIXES.md` | all 13 walls with symptoms → root causes → fixes, incl. 2 upstream-worthy bugs |
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
| SPEED | `CTX=65536` | 674,816 | **64.6 tok/s** | 7.31 |
| **LONG-CTX (draft-cap fix, default-ready)** | `CTX=524288` | 310,606 (in-flight cap) | **65.2 tok/s** | 7.31 |

> **Update**: the draft-context cap (baked patch #6, `INKLING_DRAFT_CTX_CAP=65536`) removed the
> long-context speed penalty entirely — 512K declared now runs at full speed on prompts within
> the draft's 64K adaptation. The earlier 26.8 tok/s row is preserved below for history:
>
> | pre-fix MAX-DECLARED | `CTX=524288` | 310,606 | 26.8 tok/s | 2.59 |

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
