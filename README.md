# keys-1M-context · Inkling-Small-NVFP4 + DSpark + **NVFP4 KV** · SGLang · sm_121a · Two DGX Sparks

**A full 1M-token context on two desktop DGX Sparks — with the first NVFP4 KV cache for
Inkling-Small NVFP4 + DSpark.**

Serve [thinkingmachines/Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
(276B total / 12B active MoE) with the [RadixArk DSpark speculator](https://huggingface.co/RadixArk/Inkling-Small-DSpark-Preview)
across **two desktop DGX Sparks** (GB10 / sm_121a), at a **full 1,048,576-token context**.

This is a field port. SGLang's triton backend — the only attention lane this model can use on
consumer Blackwell — shipped with **no KV quantization at all**, and none of this had been run on
GB10 before. Everything needed is here: patched files, bake script, launcher, benchmarks, and every
wall we hit with its fix.

---

## Quick start — the champion stack

**Prereqs**: 2× DGX Spark / GB10 (128 GB unified each, DGX OS, CUDA 13, docker + nvidia runtime); a
direct 200G CX7↔CX7 link with IPs on both ends; `ls /dev/infiniband` non-empty on both nodes; ~165 GB
of storage for weights, reachable at the **same path** on both nodes (NFS or local copies).

```bash
# 0) on the HEAD node (rank 0), with SSH access to the worker
git clone https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-SGlang-SM121-optimized-on-Two-DGX-Sparks.git
cd keys-1M-CTX-*

# 1) weights — once, wherever the shared storage lives
python3 -m venv ~/hfdl-venv && ~/hfdl-venv/bin/pip install -q huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfdl-venv/bin/hf download \
  thinkingmachines/Inkling-Small-NVFP4  --local-dir <STORE>/inkling/inkling-small-nvfp4
HF_HUB_ENABLE_HF_TRANSFER=1 ~/hfdl-venv/bin/hf download \
  RadixArk/Inkling-Small-DSpark-Preview --local-dir <STORE>/inkling/dspark-draft

# 2) build the image — ON EACH NODE (digest-pinned upstream + every patch)
KVQUANT=1 ./scripts/bake-image.sh

# 3) launch — WORKER FIRST, then head. Defaults are the champion config.
#    worker:
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 1
#    head:
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 0
```

`IF` is the NIC carrying the inter-node link, `HCA` its RDMA device (`ibv_devices`). Boot takes
~8 min (156 GB weight load + first-run JIT); follow it with `docker logs -f inkling-sglang`.
OpenAI-compatible endpoint on port **30000**.

### Verify (30 seconds)

```bash
curl -s localhost:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-small","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

Expect byte-for-byte: ` Paris. The capital of Germany is Berlin. The capital of`

That is the bf16 reference output — matching it proves both the quantized-KV path and the speculator
are numerically clean. Then confirm the pool exceeds your context:

```
grep -aoE 'context_len=[0-9]+|max_total_num_tokens=[0-9]+' ~/inkling-serve.log | tail -2
→ context_len=1048576    max_total_num_tokens=1082627
```

---

## The exact recipe (what the launcher actually runs)

If you prefer to run it by hand, or need to adapt it, this is the champion invocation verbatim.
`scripts/nvfp4-kv-boot.sh` is exactly this with the site values as env vars.

```bash
docker run --name inkling-sglang --rm --gpus all --network host --ipc host \
  --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v <STORE>/inkling:/models:ro \
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  -e INKLING_TORCH_CONV_COMMIT=1 -e INKLING_COMMIT_STEP_BIAS=1 \
  -e NCCL_IB_HCA=<rdma-dev> -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=<link-nic> -e GLOO_SOCKET_IFNAME=<link-nic> -e TP_SOCKET_IFNAME=<link-nic> \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint python3 local/sglang-inkling:gb10-kvquant \
  -m sglang.launch_server \
    --model-path /models/inkling-small-nvfp4 --trust-remote-code \
    --served-model-name inkling-small \
    --host 0.0.0.0 --port 30000 \
    --tp-size 2 --nnodes 2 --node-rank <0|1> --dist-init-addr <rank0-link-ip>:25000 \
    --context-length 1048576 \
    --quantization modelopt_fp4 \
    --kv-cache-dtype fp4_mx_block16 \
    --attention-backend triton --triton-attention-reduce-in-fp32 \
    --page-size 1 \
    --fp4-gemm-backend flashinfer_trtllm \
    --moe-runner-backend marlin \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
    --max-running-requests 16 \
    --chunked-prefill-size 8192 \
    --reasoning-parser inkling --tool-call-parser inkling \
    --skip-server-warmup --disable-flashinfer-autotune \
    --stream-interval 32 \
    --speculative-algorithm DSPARK \
    --speculative-draft-model-path /models/dspark-draft \
    --speculative-draft-model-quantization unquant \
    --speculative-dspark-block-size 7 \
    --cuda-graph-bs 1 2 3 4 5 6 7 8 10 12 14 16 \
    --disable-piecewise-cuda-graph --disable-prefill-cuda-graph
```

**Every non-obvious flag, and why it is not optional:**

| Flag | Why |
|---|---|
| `--kv-cache-dtype fp4_mx_block16` | the triton-compatible fp4 recipe. `nvfp4` selects the flashinfer/trtllm packing this lane cannot read; `fp8_e4m3` produces garbage (wall 13) |
| `--attention-backend triton` | Inkling asserts `fa4\|triton`, and fa4 is sm_100-only ⇒ triton is the only legal lane on GB10 |
| `--triton-attention-reduce-in-fp32` | bf16 accumulation across KV splits perturbs logits → fewer draft matches. ~+11% accept |
| `--moe-runner-backend marlin` | the only numerically-correct NVFP4 MoE runner on sm_121: cutlass **silently miscomputes**, trtllm hard-fails on sm_100-only cubins |
| `--page-size 1` | page-128 (the fa4 layout) corrupts the triton verify path |
| `--disable-prefill-cuda-graph` | the triton backend cannot replay `EXTEND` mode |
| `--disable-piecewise-cuda-graph` | the sm_121 piecewise compiler hard-fails |
| `--cuda-graph-bs 1 2 … 16` | an explicit list; `--cuda-graph-max-bs` does **not** filter and the default list OOMs the pool |
| `--speculative-dspark-block-size 7` | measured optimum; 15 (the checkpoint's native block) is worse here |
| `INKLING_TORCH_CONV_COMMIT=1` + `INKLING_COMMIT_STEP_BIAS=1` | the conv-state commit fix. Without them output degenerates into prompt-replay whenever accept > 1 |
| `--device /dev/infiniband --cap-add IPC_LOCK` | without RDMA passthrough NCCL fails with a bare `invalid usage` |
| `--mem-fraction-static 0.85` | 0.87 boots fine but buys nothing measurable |

**Order matters**: start `--node-rank 1` (worker) first, then `--node-rank 0` (head). The head is the
rendezvous point at `--dist-init-addr`. If you script this across nodes, put the environment in a
**file on each node** rather than passing it through nested SSH — multi-flag `EXTRA_ARGS` gets split
by the second shell and the worker silently never launches (you'll see `1/2 clients joined`).

---

## What you get

| | **1M profile** (default) | 64K profile |
|---|---|---|
| launch | `./scripts/nvfp4-kv-boot.sh <rank>` | `CTX=65536 ./scripts/nvfp4-kv-boot.sh <rank>` |
| context | **1,048,576** | 65,536 |
| KV pool | **1,082,627 tokens** | 1,104,683 tokens |
| decode | ~33 tok/s | ~33 tok/s |
| accept (of 8) | ~3.5 | ~3.5 |

All figures are `n=32` means (see [measurement protocol](docs/MEASUREMENT-PROTOCOL.md) —
**single-run numbers on this stack are meaningless**; they vary 3–4× on identical config).

- **Speculation is lossless** — DSpark output is byte-exact vs non-speculative decoding at temp 0.
- **fp4 KV is quality-neutral** — byte-exact vs bf16 KV on the reference probe; needle-in-a-haystack
  retrieval verified at **21K, 64K and 113K** token depths.
- **Without quantized KV the pool caps near 354K tokens** — fp4 is what makes 1M reachable at all.
- For scale: no speculation + no quantization ≈ 13 tok/s at 64K. This is ~2.6× that, at 16× the context.

### ⚑ Expect accept ≈ 3.4 — that's the published spec, not a problem

RadixArk's card reports `acc_len` **mean 3.348** across 9 datasets at temp 0 / block 7 — exactly this
config (range 2.70 Arena-Hard → 4.79 GSM8K). **If you measure ~3.4 the speculator is working and
there is nothing left to tune.** This repo's history contains a long hunt for a "7.31" that was
simply a lucky draw from a nondeterministic distribution. The only lever beyond ~3.4 is finetuning
the draft ([plan](docs/DRAFT-FINETUNE-PLAN.md); realistic ceiling ~4.2–4.6).

---

## Why patches are needed

`scripts/bake-image.sh` bakes them all. Full symptom → cause → fix table for **22 walls** lives in
[`docs/BUGS-AND-FIXES.md`](docs/BUGS-AND-FIXES.md). The load-bearing ones:

| Area | Fix |
|---|---|
| **KV quantization** | The triton backend had none. `patches/kv-quant/` adds it: quantize **inside the pool** (Inkling has *three* KV writers — DSpark's hidden-state injector writes KV directly), an fp4 branch for the hybrid-SWA pool upstream never wrote, e2m1 nibble decode + block-16 scales in cloned kernels, correct fp4 byte accounting. Upstream's `decode_attention.py`/`extend_attention.py` stay **byte-untouched**. |
| **DSpark draft OOB** | [sglang#30555](https://github.com/sgl-project/sglang/issues/30555) fixed *correctly*: one `-1` on the draft worker's width in `triton_backend.py`. (The issue's own suggested ServerArgs pin double-corrects on current builds — don't use it.) Fixes OOB draft-KV reads **and** unblocks decode CUDA graphs. |
| **Conv-state commit off-by-one** *(novel)* | DSpark's `commit_lens` excludes the bonus token, but the sconv commit used it as a last-step index — state regressed and output degenerated into prompt-replay whenever accept > 1. Hits every non-symm-mem deployment, i.e. everything that isn't a B200-class single node. |
| **GB10 kernel limits** | MoE grouped-GEMM `num_stages` 3/4→2 (99 KB smem vs B200's 228 KB); Helion sm_121 configs seeded; `emit_packed_topk=False`; **marlin is the only numerically-correct NVFP4 MoE runner** here (cutlass silently miscomputes, trtllm hard-fails). |
| **Long-context speed** | The draft's context is pinned to its 64K adaptation, so declaring a huge context no longer craters acceptance. |

---

## Repo map

| Path | What |
|---|---|
| `scripts/nvfp4-kv-boot.sh` | **the champion launcher** (1M context, fp4 KV) |
| `scripts/bake-image.sh` | builds `local/sglang-inkling:gb10[-kvquant]` from a digest-pinned upstream |
| `scripts/inkling-sglang-launch.sh` | underlying launcher; every knob is an env var |
| `patches/kv-quant/` | the KV-quantization implementation (6 files) |
| `patches/files/` + `patches/all-patches.diff` | base GB10 patches, byte-exact and as a reviewable diff |
| `docs/MEASUREMENT-PROTOCOL.md` | **read before benchmarking anything** |
| `docs/BUGS-AND-FIXES.md` | 22 walls: symptom → root cause → fix |
| `docs/KV-QUANT-IMPLEMENTATION-NOTES.md` | how the fp4 KV path works internally |
| `docs/DRAFT-FINETUNE-PLAN.md` | the remaining accept lever (+ A4Q applicability appendix) |
| `docs/ROADMAP.md` | done / blocked / why |
| `benchmarks/accept_probe.py` | the 32-sample harness to use for every comparison |
| `benchmarks/tests_verify_nvfp4.py` | 20 bitwise-exactness tests for the fp4 kernels |
| `specs/001-oneshot-install/` | the same install as a **gated** task list for agents |

**Agents**: start at [`specs/001-oneshot-install/tasks.md`](specs/001-oneshot-install/tasks.md).

---

## Knobs

All are env vars on the launchers: `CTX` · `MEMFRAC` (0.85 default; 0.87 works, buys nothing
measurable) · `MAXREQ` · `BLOCK` (7 is optimal; 15 is worse here) · `KVD` (`fp4_mx_block16` default —
**not** `nvfp4`, which selects the flashinfer/trtllm recipe the triton lane cannot consume) ·
`GRAPH_BS` · `IMAGE` · `EXTRA_ARGS`.

Leave `SGLANG_RAGGED_VERIFY_MODE` **unset**: `compact` crashes Inkling's sconv JIT, and `cap-accept`
needs calibration artifacts (see [ROADMAP](docs/ROADMAP.md)).

## Provenance

Upstream image `lmsysorg/sglang@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`
(`dev-cu13-inkling-dspark`, 2026-07-30); all patches are against files inside it. Not affiliated with
LMSYS, Thinking Machines, RadixArk, or NVIDIA — an independent field port.
