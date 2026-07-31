#!/bin/bash
# Inkling-Small-NVFP4 + RadixArk DSpark draft — TP=2 across two DGX Sparks (GB10, sm_121a)
#
# Usage:  ./inkling-sglang-launch.sh <rank 0|1>     (start rank 1 on the worker FIRST, then rank 0 on the head)
#
# DEFAULTS = the validated champion config (lossless, 64.6 tok/s raw single-stream,
# accept_len 7.31): marlin MoE, triton attention, page-size 1, DSpark block 7,
# decode CUDA graphs bs {1,2,4,8}, mem-fraction 0.85, 64K ctx, conv-commit fix ON.
#
# Site knobs (env): MASTER_IP IF HCA GID MODELS IMAGE SGLANG_PORT
# Tuning knobs (env): ATTN MOE FP4GEMM MEMFRAC CTX SPEC GRAPHS BLOCK MAXREQ PAGE EXTRA_ARGS
set -euo pipefail
RANK=${1:?rank 0|1}

# ---- site config (edit or override) ----
MASTER_IP=${MASTER_IP:-10.100.20.1}     # rank0's IP on the 200G link
IF=${IF:-enp1s0f0np0}                   # NIC carrying the inter-node link (both nodes)
HCA=${HCA:-rocep1s0f0}                  # RDMA device for that NIC (ibv_devices)
GID=${GID:-3}                           # RoCEv2 IPv4 GID index (show_gids)
MODELS=${MODELS:-/mnt/models-7552/inkling}  # dir with inkling-small-nvfp4/ + dspark-draft/
IMAGE=${IMAGE:-local/sglang-inkling:gb10}   # baked by scripts/bake-image.sh

# ---- champion defaults ----
PORT=${SGLANG_PORT:-30000}
ATTN=${ATTN:-triton}                    # Inkling asserts fa4|triton; fa4 is sm_100-only -> triton
MOE=${MOE:-marlin}                      # ONLY numerically-correct NVFP4 MoE runner on sm_121
FP4GEMM=${FP4GEMM:-flashinfer_trtllm}   # dense FP4 GEMMs are fine on sm_121
MEMFRAC=${MEMFRAC:-0.85}
CTX=${CTX:-65536}                       # draft is 64K-adapted; model itself goes to 1M
SPEC=${SPEC:-1}
GRAPHS=${GRAPHS:-1}
PAGE=${PAGE:-1}                         # page 128 corrupts the triton verify path
export INKLING_TORCH_CONV_COMMIT=${INKLING_TORCH_CONV_COMMIT:-1}   # conv-commit fix (see docs/BUGS-AND-FIXES.md)
export INKLING_COMMIT_STEP_BIAS=${INKLING_COMMIT_STEP_BIAS:-1}

EXTRA=()
if [ "$SPEC" = 1 ]; then
  EXTRA+=(--speculative-algorithm DSPARK
          --speculative-draft-model-path /models/dspark-draft
          --speculative-draft-model-quantization unquant
          --speculative-dspark-block-size "${BLOCK:-7}")
fi
if [ "$GRAPHS" = 1 ]; then
  EXTRA+=(--cuda-graph-bs 1 2 4 8 --disable-piecewise-cuda-graph --disable-prefill-cuda-graph)
else
  EXTRA+=(--disable-cuda-graph --disable-prefill-cuda-graph)
fi

docker rm -f inkling-sglang 2>/dev/null || true
exec docker run --name inkling-sglang --rm --gpus all --network host --ipc host \
 --shm-size 16g --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 \
 -v "$MODELS":/models:ro \
 -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
 -e NCCL_IB_HCA="$HCA" -e NCCL_IB_GID_INDEX="$GID" \
 -e NCCL_SOCKET_IFNAME="$IF" -e GLOO_SOCKET_IFNAME="$IF" -e TP_SOCKET_IFNAME="$IF" \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_NET_PLUGIN=none \
 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 \
 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
 -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
 -e INKLING_TORCH_CONV_COMMIT -e INKLING_COMMIT_STEP_BIAS \
 -e INKLING_NOOP_CONV_COMMIT="${INKLING_NOOP_CONV_COMMIT:-0}" \
 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 --entrypoint python3 "$IMAGE" \
 -m sglang.launch_server \
  --model-path /models/inkling-small-nvfp4 --trust-remote-code \
  --served-model-name inkling-small \
  --host 0.0.0.0 --port "$PORT" \
  --tp-size 2 --nnodes 2 --node-rank "$RANK" --dist-init-addr "$MASTER_IP:25000" \
  --context-length "$CTX" \
  --quantization modelopt_fp4 \
  --attention-backend "$ATTN" --page-size "$PAGE" \
  --fp4-gemm-backend "$FP4GEMM" --moe-runner-backend "$MOE" \
  --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static "$MEMFRAC" --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
  --max-running-requests "${MAXREQ:-8}" \
  --chunked-prefill-size 8192 \
  --reasoning-parser inkling --tool-call-parser inkling \
  --skip-server-warmup --disable-flashinfer-autotune \
  --stream-interval 32 \
  "${EXTRA[@]}" ${EXTRA_ARGS:-}
