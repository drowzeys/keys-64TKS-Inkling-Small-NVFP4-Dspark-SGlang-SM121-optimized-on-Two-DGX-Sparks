"""Block-scaled quantized KV pools for the SGLang TRITON lane (GB10 sm_121a).

Two subclasses that make the in-image quantized pools usable by
``TritonAttnBackend`` + ``kv_quant_attention``:

``MHATokenToKVPoolMXFP8Triton``
    Quantizes at store for EVERY writer. Upstream
    ``MHATokenToKVPoolMXFP8.set_kv_buffer`` (memory_pool.py:3384-3402) demands
    a caller-supplied fp8 payload + UE8M0 scale tensors unless
    ``page_size == 128`` (the fa4 fused quant-store kernel). At the triton
    lane's page_size=1 that made every writer which does NOT go through the
    attention backend fail: the DSpark hidden-state injector
    (``srt/models/dspark.py:658 write_target_hidden_kv``) hands the pool bf16
    K/V with ``layer.k_scale``/``v_scale`` = None and raised
    ``ValueError: MXFP8 KV cache requires K and V scale tensors.`` on the very
    first request of the mxfp8 boot test.

``MHATokenToKVPoolFP4Triton``
    ``fp4_mx_block16`` storage (packed e2m1 + one UE8M0 scale per 16 elements)
    with a RAW reader, so attention dequantizes per tile in-kernel instead of
    materializing the whole pool.

Both are only ever instantiated when ``--kv-cache-dtype`` selects a quantized
recipe, so they are structurally inert under ``auto``/bf16.
"""

from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
)

FP4_SCALE_BLOCK_SIZE = 16


def _committed_locs(loc_2d: torch.Tensor, commit_lens: torch.Tensor) -> torch.Tensor:
    """Flatten ``loc_2d`` with uncommitted rows redirected to the pad slot 0.

    The upstream tiled ``_set_kv_buffer_prefix_valid_impl`` writes only the
    committed rows, but it writes RAW bytes and knows nothing about scale
    buffers. Quantized pools have to go through ``set_kv_buffer`` instead, so
    the row selection is expressed as an index remap rather than a
    ``nonzero()`` gather: shape-static and free of a device sync, hence safe
    under CUDA-graph capture. Slot 0 is the pool's documented dummy pad slot
    ("used for writing dummy outputs from padded tokens"), never read back.
    """
    col = torch.arange(loc_2d.shape[1], device=loc_2d.device).view(1, -1)
    valid = col < commit_lens.to(torch.int64).view(-1, 1)
    return torch.where(valid, loc_2d, torch.zeros_like(loc_2d)).reshape(-1)


class _QuantizedPrefixValidMixin:
    """``set_kv_buffer_prefix_valid`` that routes through ``set_kv_buffer``."""

    def set_kv_buffer_prefix_valid(
        self,
        layer: RadixAttention,
        loc_2d: torch.Tensor,
        commit_lens: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
    ):
        if loc_2d.ndim != 2:
            raise ValueError(f"loc_2d must be rank-2, got shape={tuple(loc_2d.shape)}.")
        if commit_lens.ndim != 1 or commit_lens.shape[0] != loc_2d.shape[0]:
            raise ValueError(
                "commit_lens must match loc_2d batch size: "
                f"{tuple(commit_lens.shape)=} {tuple(loc_2d.shape)=}."
            )
        num_rows = int(loc_2d.numel())
        if cache_k.shape[0] != num_rows or cache_v.shape[0] != num_rows:
            raise ValueError(
                "dense KV rows must match loc_2d size: "
                f"{tuple(cache_k.shape)=} {tuple(cache_v.shape)=} "
                f"{tuple(loc_2d.shape)=}."
            )
        if loc_2d.device != self.k_buffer[0].device:
            loc_2d = loc_2d.to(device=self.k_buffer[0].device, non_blocking=True)
        if commit_lens.device != loc_2d.device:
            commit_lens = commit_lens.to(device=loc_2d.device, non_blocking=True)
        if loc_2d.dtype != torch.int64:
            loc_2d = loc_2d.to(torch.int64)

        self.set_kv_buffer(
            layer,
            _committed_locs(loc_2d, commit_lens),
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=layer_id_override,
        )


class MHATokenToKVPoolMXFP8Triton(_QuantizedPrefixValidMixin, MHATokenToKVPoolMXFP8):
    """MXFP8 pool that quantizes bf16 K/V inside ``set_kv_buffer``."""

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if cache_k.dtype != self.store_dtype and (k_scale is None or v_scale is None):
            # Per-tensor float scales are meaningless for a block-scaled pool;
            # a caller that has real per-token scale TENSORS must pass both.
            if isinstance(k_scale, torch.Tensor) or isinstance(v_scale, torch.Tensor):
                raise ValueError(
                    "MXFP8 KV cache: got one scale TENSOR and one None; pass "
                    "both per-token UE8M0 scale tensors or neither."
                )
            if isinstance(k_scale, (int, float)) and float(k_scale) != 1.0:
                raise NotImplementedError(
                    "MXFP8 KV cache does not support a per-tensor k_scale "
                    f"({k_scale}); block scales replace it."
                )
            if isinstance(v_scale, (int, float)) and float(v_scale) != 1.0:
                raise NotImplementedError(
                    "MXFP8 KV cache does not support a per-tensor v_scale "
                    f"({v_scale}); block scales replace it."
                )
            from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8

            k_mx = to_mxfp8(cache_k)
            v_mx = to_mxfp8(cache_v)
            cache_k = k_mx.data
            cache_v = v_mx.data
            k_scale = k_mx.scale.view(torch.float8_e8m0fnu)
            v_scale = v_mx.scale.view(torch.float8_e8m0fnu)

        return super().set_kv_buffer(
            layer,
            loc_info,
            cache_k,
            cache_v,
            k_scale,
            v_scale,
            layer_id_override=layer_id_override,
            dcp_kv_mask=dcp_kv_mask,
        )


class MHATokenToKVPoolFP4Triton(_QuantizedPrefixValidMixin, MHATokenToKVPoolFP4):
    """fp4_mx_block16 pool read per-tile by the triton kernels.

    Differences from ``MHATokenToKVPoolFP4``:

    * ``_get_key_buffer``/``_get_value_buffer`` return the RAW packed uint8
      buffer. The parent runs ``FP4MXBlock16KVQuantizeUtil.batched_dequantize``
      over the ENTIRE per-layer buffer on every access
      (memory_pool.py:2992/3010) -- at a 1M-token pool that materializes a full
      bf16 copy and defeats the whole point. It also disarms
      ``srt/models/dspark.py:_fused_kv_write_bundle``, which probes
      ``pool.get_key_buffer(layer_id)`` and bails out on non-bf16 buffers: with
      the parent's reader that probe ALONE would allocate the dequantized pool.
    * ``v_head_dim`` is honoured (the parent sizes V payload and V scales with
      ``head_dim``).
    * ``get_kv_scale_buffer(layer_id)`` mirrors the MXFP8 pool's accessor so
      ``SWAKVPool.get_kv_scale_buffer`` (swa_memory_pool.py:168-174) and the
      backend's per-tile scale fetch work unchanged.
    * ``_init_data_ptrs_and_strides()`` is called (the parent's
      ``_create_buffers`` skips it, so ``move_kv_cache`` would AttributeError).
      The base ``_slot_move_pointer_buffers`` already includes the scale
      buffers, so scale rows travel with their payload -- required by the
      mamba ``extra_buffer`` radix strategy.
    * CPU offload / disagg raise instead of silently dropping scales.

    Storage cost per element: 0.5 B payload + 1/16 B scale = 0.5625 B, i.e.
    3.5556x more tokens than bf16's 2 B for the same budget.
    """

    SCALE_BLOCK_SIZE = FP4_SCALE_BLOCK_SIZE

    def _create_buffers(self):
        from contextlib import nullcontext

        m = self.size + self.page_size
        n = self.head_num
        k = self.head_dim
        v = self.v_head_dim
        sb = self.SCALE_BLOCK_SIZE

        for name, d in (("head_dim", k), ("v_head_dim", v)):
            if d % sb != 0:
                raise ValueError(
                    f"fp4_mx_block16 KV cache requires {name} divisible by {sb}, "
                    f"got {d}."
                )
        if self.use_hnd:
            # Buffers are NHD; the inherited HND move_kv_cache branch would
            # relocate the wrong bytes (and the packed row width is head_dim//2).
            raise ValueError(
                "fp4_mx_block16 KV cache does not support SGLANG_USE_HND_KVCACHE."
            )

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.store_dtype = torch.uint8
                # Packed e2m1: two head_dim entries per byte. Zero-init keeps
                # unwritten slots at value 0 with scale 2^-127.
                self.k_buffer = [
                    torch.zeros(
                        (m, n, k // 2), dtype=self.store_dtype, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (m, n, v // 2), dtype=self.store_dtype, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                # Flat (m, H * D // 16); batched_quantize blocks a (T, H, D)
                # tensor as view(T, H*D//16, 16), so the scale for (h, d) is at
                # column h * (D // 16) + d // 16 -- exactly a (m, H, D // 16)
                # reshape, which is what the kernels index.
                self.k_scale_buffer = [
                    torch.zeros(
                        (m, (n * k) // sb), dtype=self.store_dtype, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scale_buffer = [
                    torch.zeros(
                        (m, (n * v) // sb), dtype=self.store_dtype, device=self.device
                    )
                    for _ in range(self.layer_num)
                ]
                self.dq_k_buffer = None
                self.dq_v_buffer = None

        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self._init_data_ptrs_and_strides()

    # -- raw reads: dequant happens per tile inside the triton kernels -------
    def _get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_scale_buffer(self, layer_id: int):
        idx = layer_id - self.start_layer
        return self.k_scale_buffer[idx], self.v_scale_buffer[idx]

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        layer_id_override: Optional[int] = None,
        dcp_kv_mask: Optional[torch.Tensor] = None,
    ):
        if dcp_kv_mask is not None:
            raise NotImplementedError(
                "fp4_mx_block16 KV cache does not support DCP KV masks."
            )
        # The parent divides cache_k IN PLACE by a per-tensor scale
        # (memory_pool.py:3035-3038), which would corrupt the caller's
        # K_Extend/V_Extend. Block scales replace per-tensor scales entirely.
        for name, s in (("k_scale", k_scale), ("v_scale", v_scale)):
            if isinstance(s, torch.Tensor):
                raise NotImplementedError(
                    f"fp4_mx_block16 KV cache does not consume a {name} tensor; "
                    "block scales are computed at store time."
                )
            if isinstance(s, (int, float)) and float(s) != 1.0:
                raise NotImplementedError(
                    f"fp4_mx_block16 KV cache does not support a per-tensor "
                    f"{name} ({s}); block scales replace it."
                )
        return super().set_kv_buffer(
            layer,
            loc_info,
            cache_k,
            cache_v,
            None,
            None,
            layer_id_override=layer_id_override,
        )

    # -- paths that would move payload without its scales --------------------
    def get_cpu_copy(self, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported for fp4_mx_block16 KV cache."
        )

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        raise NotImplementedError(
            "CPU offloading is unsupported for fp4_mx_block16 KV cache."
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "KV transfer / disaggregation is unsupported for fp4_mx_block16 KV "
            "cache (scale buffers are not exposed)."
        )
