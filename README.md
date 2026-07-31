# keys-64TKS · Inkling-Small-NVFP4 + DSpark · SGLang · sm_121a · Two DGX Sparks

**64 tok/s single-stream, lossless speculative decoding, on two desktop DGX Sparks.**
The first known deployment of [thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
(276B total / 12B active MoE) with the [RadixArk DSpark speculator](https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview)
off datacenter Blackwell — TP=2 over a single 200G link between two GB10 (sm_121a) machines.

> **Agents: start at [`specs/001-oneshot-install/tasks.md`](specs/001-oneshot-install/tasks.md).**
> It is a gated, ordered task list (spec-kit style). Execute it verbatim and you get this
> exact serve without re-fighting the ~14 boot cycles it took to find these fixes.

## Performance — measured with the 32-sample protocol

> **Read [`docs/MEASUREMENT-PROTOCOL.md`](docs/MEASUREMENT-PROTOCOL.md) first.** The target forward
> is nondeterministic at temp 0 on this stack, and DSpark acceptance swings with the *style* of the
> continuation a run happens to land on. Single-run numbers here are meaningless. Earlier revisions
> of this README quoted 64.6 / 68.2 tok/s from single probes — those were the top of a distribution
> whose mean is ~30. They have been replaced with mean ± standard error over 32 samples.

Hardware: 2× DGX Spark (GB10, 128 GB unified, ~273 GB/s each), one 200G CX7↔CX7 link, NFS weights.
Config: DSpark block-7, decode graphs, marlin MoE, page 1, 64K ctx, mem-fraction 0.85.

| Config | accept (of 8) | tok/s | n |
|---|---|---|---|
| DSpark, bf16 attention reduce | 3.09 ± 0.18 | 29.7 ± 2.1 | 32 |
| **DSpark + `--triton-attention-reduce-in-fp32`** | **3.44 ± 0.17** | **34.3 ± 1.7** | 32 |
| no speculation (eager reference) | — | ~13 | — |

fp32 reduction is ~+11% accept / +15% tok/s — a ~1.4σ effect, so *probably* real but not
conclusively separated from noise. It costs nothing measurable, so it ships as a default.

Peak single runs reach ~50–60 tok/s; the floor is ~10–15. Both are the same config. Plan capacity
against the mean, not the peak.

**Task class matters**: raw mid-prose continuation accepts best; chat-template traffic with
reasoning tokens accepts noticeably worse. Never quote tok/s without saying which you measured.

### Concurrency (256 tok/req, temp 0.7, MAXREQ 16, graph tiers 1–16)

| Task | C1 | C4 agg | C8 agg |
|---|---|---|---|
| list | 26.5 | 44.9 | 64.8 |
| essay | 23.1 | 36.2 | 51.7 |
| reading | 23.0 | 54.7 | 67.8 |

Aggregate throughput scales well to C8; per-stream rate falls as expected. These predate the
32-sample protocol — treat as indicative, re-measure before relying on them.

## What's in the box

| Path | What |
|---|---|
| `specs/001-oneshot-install/` | spec / plan / **tasks.md** — the gated one-shot install |
| `scripts/bake-image.sh` | digest-pinned upstream image + all patches → `local/sglang-inkling:gb10` |
| `scripts/inkling-sglang-launch.sh` | per-rank launcher; defaults = validated champion config |
| `patches/files/` | the 6 net-patched SGLang files (byte-exact from the validated image) |
| `patches/all-patches.diff` | the same as a reviewable 162-line unified diff |
| `docs/BUGS-AND-FIXES.md` | all 18 walls with symptoms → root causes → fixes, incl. 2 upstream-worthy bugs |
| `docs/MEASUREMENT-PROTOCOL.md` | **read before benchmarking** — this stack is nondeterministic at temp 0 |
| `benchmarks/` | bench harness + raw results |

## The two bugs you'd lose days on (both fixed here)

1. **DSpark draft gamma mismatch** ([sglang#30555](https://github.com/sgl-project/sglang/issues/30555), fixed *correctly*):
   one `-1` in the triton backend's draft-worker width fixes out-of-bounds draft KV reads (which
   otherwise pin accept near 1.0) **and** unblocks decode CUDA-graph capture. The issue's own
   suggested ServerArgs pin double-corrects on current builds — don't use it.
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

| Profile | Launch | KV pool (tokens) | note |
|---|---|---|---|
| **SPEED (default)** | `CTX=65536` | 674,816 | the measured champion; see the performance table above |
| LONG-CTX | `CTX=524288` | 310,606 (in-flight cap) | with the draft-context cap, speed is comparable to 64K on prompts inside the draft's 64K adaptation |

Earlier revisions listed per-profile tok/s from single runs (64.6 / 65.2 / 26.8). Those were
single draws from a wide distribution and have been removed; re-measure any profile you care
about with `benchmarks/accept_probe.py`.

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
