# Tasks: One-shot install — Inkling-Small-NVFP4 + DSpark on 2× DGX Spark (SGLang, sm_121a)

Agent contract: execute in order; every task has a **GATE** — do not proceed past a failed gate.
All commands assume this repo is cloned on the head node (rank 0) with SSH access to the worker.

## T0 — Preflight (both nodes)

- 2× NVIDIA DGX Spark / GB10 (128 GB unified each), DGX OS with CUDA 13, docker + nvidia runtime.
- A direct 200G link between the CX7 ports of the two nodes (or a switch path), with IPs on both ends
  (this recipe's reference: rank0 `10.100.20.1`, rank1 `10.100.20.2`, MTU 9000).
- `ls /dev/infiniband` shows `uverbs*` on both nodes; `ibv_devices` lists the RDMA device for your link NIC.
- ~35 GB free disk per node (image), and one node/NAS with ~165 GB for weights, NFS-exportable.

**GATE T0**: `ping <peer-ip>` works both ways; `ibv_devices` non-empty on both nodes.

## T1 — Weights (once, on the storage node)

```bash
python3 -m venv ~/hfdl-venv && ~/hfdl-venv/bin/pip install -q 'huggingface_hub[cli]' hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
~/hfdl-venv/bin/hf download RadixArk/Inkling-Small-DSpark-Preview --local-dir <STORE>/inkling/dspark-draft
~/hfdl-venv/bin/hf download thinkingmachines/Inkling-Small-NVFP4  --local-dir <STORE>/inkling/inkling-small-nvfp4
```

Export `<STORE>/inkling` read-only over NFS and mount it at the SAME path on both serving nodes
(reference: `/mnt/models-7552/inkling`). Local disk works too if you have 165 GB per node.

**GATE T1**: both nodes see `inkling-small-nvfp4/config.json` and `dspark-draft/model.safetensors`
at the mount path; total ≈161 GB.

## T2 — Bake the patched image (both nodes)

```bash
KVQUANT=1 ./scripts/bake-image.sh   # digest-pinned upstream + all patches
                                    # -> local/sglang-inkling:gb10  AND  :gb10-kvquant (fp4 KV)
```

What it bakes (full deltas in `patches/all-patches.diff`, mechanisms in `docs/BUGS-AND-FIXES.md`):
1. NCCL → ≥2.30 (bundled 2.28.9 burns ~15 GB at init on GB10).
2. `inkling_moe.py`: grouped-GEMM `num_stages` → 2 (sm_121 has 99 KB smem; B200 tuning wants 110+ KB).
3. Helion kernel configs: seed `*_sm_121.json` from `*_sm_100.json` (loader refuses to run without them).
4. `inkling_common/moe.py`: `emit_packed_topk = False` (packed top-k is consumable only by the
   sm_100-only TRT-LLM routed MoE; marlin/cutlass runners need standard top-k).
5. `inkling.py`: torch-native conv-state commit with **+1 step bias** (fixes DSpark verify-commit
   corruption on the non-symm-mem path — the bug that garbles all hybrid output when accept > 1).
6. `sconv.py`: save per-step conv windows BEFORE the in-place conv (mirrors the fused path's x_scratch).
7. `triton_backend.py`: draft workers use gamma (= dspark_block_size), not the target's gamma+1
   (upstream sgl-project/sglang#30555; fixes both draft-KV OOB reads AND decode-graph capture).

**GATE T2**: script prints `nccl 23xxx`, `BAKED local/sglang-inkling:gb10` **and**
`BAKED local/sglang-inkling:gb10-kvquant` on BOTH nodes.

## T3 — Launch (rank 1 first, then rank 0)

```bash
# CHAMPION: 1M context + NVFP4 KV.  worker FIRST, then head.
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 1
MASTER_IP=<rank0-link-ip> IF=<link-nic> HCA=<rdma-dev> MODELS=<mount>/inkling ./scripts/nvfp4-kv-boot.sh 0
```

(For a bf16-KV serve instead, use `./scripts/inkling-sglang-launch.sh <rank>` — same flags, ~354K pool.)

Defaults encode the measured champion: marlin MoE · triton attention **+ fp32 reduction** ·
page-size 1 · DSpark block 7 · decode graphs · mem-fraction 0.85 · 64K ctx · conv-commit fix ·
draft-context cap.
Boot takes ~6-8 min (156 GB NFS weight load). Watch: `docker logs -f inkling-sglang`.

**GATE T3**: log shows `Initialized DSpark draft runner ... gamma=7`, `The server is fired up and
ready to roll!`, and `max_total_num_tokens` **greater than** `context_len` (≈1,082,627 vs 1,048,576). If the scheduler dies on the first request,
see the wall table in `docs/BUGS-AND-FIXES.md` — every failure we hit is listed with its fix.

## T4 — Verify losslessness (MANDATORY before trusting any numbers)

```bash
curl -s http://<head>:30000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-small","prompt":"The capital of France is","max_tokens":12,"temperature":0}'
```

Expected text, byte-exact: ` Paris. The capital of Germany is Berlin. The capital of`
(this equals the spec-OFF output at temp 0, page-size 1 — DSpark verified lossless).

**GATE T4**: exact match. A fluent-but-different answer at temp 0 means a kernel/commit
regression — do not proceed; diff your image against this repo's patches.

## T5 — Benchmark (optional but recommended)

```bash
python3 benchmarks/accept_probe.py "my-install" --reps 8   # 32 samples, mean +/- se
python3 benchmarks/concurrency_bench.py                    # C1/C4/C8 aggregate throughput
```

**Read [`docs/MEASUREMENT-PROTOCOL.md`](../../docs/MEASUREMENT-PROTOCOL.md) before comparing
anything.** This stack is nondeterministic at temp 0; single-run numbers vary 3-4x on identical
config. Expect accept ~3.4 ± 0.2 and ~34 ± 2 tok/s mean on the champion. If you see a single run
at 60+ or at 12, that is the same distribution, not a finding.

## T6 — Point your client at it

OpenAI-compatible: `http://<head>:30000/v1` · model `inkling-small` · reasoning + tool-call
parsers active.

Context profiles (see README table for measured numbers): `CTX=65536` = max speed (default);
`CTX=393216` = max self-consistent context (pool >= ctx); `CTX=524288` = max declared
(pool 310K is the real in-flight cap). NEVER use `--kv-cache-dtype fp8_e4m3` (wall #13:
catastrophic silent corruption). Declared context stretches the draft's rope scaling —
long-context profiles are slower even on short prompts. Re-run the T4 lossless gate after
any CTX change.
