#!/bin/bash
# Bake local/sglang-inkling:gb10 from the upstream LMSYS Inkling+DSpark dev image.
# Run ON EACH serving node (needs docker + internet). Idempotent.
set -euo pipefail

# Pin to the exact digest this recipe was validated against (2026-07-30 push).
UPSTREAM="${UPSTREAM:-lmsysorg/sglang@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7}"
TAG="${TAG:-local/sglang-inkling:gb10}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SGL=/sgl-workspace/sglang/python/sglang

echo "== pulling $UPSTREAM"
docker pull "$UPSTREAM"

docker rm -f inkling-bake 2>/dev/null || true
docker create --name inkling-bake --entrypoint bash "$UPSTREAM" -c "
  pip install -q --no-deps -U nvidia-nccl-cu13 &&
  cd $SGL/srt/layers/moe/moe_runner/triton_utils/configs/ &&
  for f in *_sm_100.json; do cp \"\$f\" \"\${f%_sm_100.json}_sm_121.json\"; done &&
  python3 -c 'import ctypes;l=ctypes.CDLL(\"/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2\");v=ctypes.c_int();l.ncclGetVersion(ctypes.byref(v));assert v.value>=23000,v.value;print(\"nccl\",v.value)'
"

# Overlay the 5 net-patched files (see patches/all-patches.diff for the deltas).
docker cp "$REPO_DIR/patches/files/inkling.py.sglang_srt_models"                    inkling-bake:$SGL/srt/models/inkling.py
docker cp "$REPO_DIR/patches/files/moe.py.sglang_srt_models_inkling_common"         inkling-bake:$SGL/srt/models/inkling_common/moe.py
docker cp "$REPO_DIR/patches/files/sconv.py.sglang_srt_models_inkling_common"       inkling-bake:$SGL/srt/models/inkling_common/sconv.py
docker cp "$REPO_DIR/patches/files/triton_backend.py.sglang_srt_layers_attention"   inkling-bake:$SGL/srt/layers/attention/triton_backend.py
docker cp "$REPO_DIR/patches/files/inkling_moe.py.kernels_ops_moe"                  inkling-bake:$SGL/kernels/ops/moe/inkling_moe.py
docker cp "$REPO_DIR/patches/files/draft_worker_common.py.sglang_srt_speculative"    inkling-bake:$SGL/srt/speculative/draft_worker_common.py

echo "== running NCCL upgrade + helion seed inside container"
docker start -a inkling-bake

docker commit inkling-bake "$TAG" >/dev/null
docker rm inkling-bake >/dev/null
echo "== BAKED $TAG"

# Optional: KV-quant build (NVFP4 / mxfp8 KV cache — 3.12x KV pool). KVQUANT=1 ./bake-image.sh
if [ "${KVQUANT:-0}" = "1" ]; then
  echo "== baking KV-quant overlay -> local/sglang-inkling:gb10-kvquant"
  docker rm -f kvq 2>/dev/null || true
  docker create --name kvq --entrypoint true "$TAG" >/dev/null
  docker cp "$REPO_DIR/patches/kv-quant/srt/models/inkling_common/attn.py"        kvq:$SGL/srt/models/inkling_common/attn.py
  docker cp "$REPO_DIR/patches/kv-quant/srt/layers/attention/triton_backend.py"   kvq:$SGL/srt/layers/attention/triton_backend.py
  docker cp "$REPO_DIR/patches/kv-quant/srt/mem_cache/kv_quant_pools.py"          kvq:$SGL/srt/mem_cache/kv_quant_pools.py
  docker cp "$REPO_DIR/patches/kv-quant/srt/mem_cache/kv_cache_configurator.py"   kvq:$SGL/srt/mem_cache/kv_cache_configurator.py
  docker cp "$REPO_DIR/patches/kv-quant/srt/model_executor/pool_configurator.py"  kvq:$SGL/srt/model_executor/pool_configurator.py
  docker cp "$REPO_DIR/patches/kv-quant/kernels/ops/attention/kv_quant_attention.py" kvq:$SGL/kernels/ops/attention/kv_quant_attention.py
  docker commit kvq local/sglang-inkling:gb10-kvquant >/dev/null
  docker rm kvq >/dev/null
  echo "== BAKED local/sglang-inkling:gb10-kvquant"
fi
