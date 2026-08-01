"""Block-scaled quantized KV-cache attention for the SGLang TRITON backend
(GB10 sm_121a). Supports two storage recipes with ONE set of kernels:

  * ``mxfp8``            -- fp8-e4m3 payload, one UE8M0 scale / 32 elements.
  * ``fp4_mx_block16``   -- packed e2m1 payload (two nibbles per byte), one
    UE8M0 scale / 16 elements. This is what ``--kv-cache-dtype nvfp4`` /
    ``fp4_mx_block16`` resolves to for the Inkling hybrid-SWA pools.

Restructured for UNCONDITIONAL bf16-path inertness: the hot kernels in
``decode_attention.py`` / ``extend_attention.py`` are BYTE-IDENTICAL to
upstream (same triton cache keys -- which hash function line numbers -- same
compiled binaries, same launch signatures). Everything quantized lives here:

- ``_prepare_kv_sf_args``: validates/unpacks the pool's FLAT page_size=1
  scale buffers into ``(m, H_kv, head_dim // SF_BLOCK)`` uint8 views (the
  byte IS the biased exponent; dequant multiplier = 2^(sf - 127)).
- ``_dequant_kv_block``: payload -> f32. mxfp8 is a plain cast; nvfp4 selects
  the e2m1 nibble by head_dim parity and decodes it arithmetically.
- ``_fwd_kernel_stage1_kv_quant`` / ``_fwd_grouped_kernel_stage1_kv_quant`` /
  ``_fwd_kernel_kv_quant``: clones of the upstream kernels whose POOL K/V load
  sites add blockwise dequant. Q stays bf16 (never cast to the storage
  dtype); the extend clone's stage-2 chunk (K_Extend/V_Extend) stays bf16
  untouched; custom-mask / SWA / SKIP_TILE logic is byte-identical to
  upstream (DSpark verify masking undisturbed).
- ``decode_attention_fwd_kv_quant`` / ``extend_attention_fwd_kv_quant``: wrappers
  mirroring the upstream wrappers' prep, launching the clones. Called by
  TritonAttnBackend ONLY when the KV pool is block-scaled quantized; this
  module is never even imported otherwise.

page_size=1 layout only; MLA/DPE layouts rejected. See STAGE12_NOTES.md and
STAGE3_NVFP4_NOTES.md.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.decode_attention import (
    _MIN_BLOCK_KV,
    _decode_softmax_reducev_fwd,
    _extract_kv_strides,
    tanh,
)
from sglang.kernels.ops.attention.extend_attention import (
    _get_block_sizes_for_extend_attention,
)
from sglang.kernels.ops.attention.score_mod import unpack_aux_tensors
from sglang.srt.utils import is_hip

_is_hip = is_hip()


SF_BLOCK_MXFP8 = 32
SF_BLOCK_FP4 = 16


def _prepare_kv_sf_args(
    k_sf_buffer, v_sf_buffer, page_size, kv_fp4=False, head_dim=None, v_head_dim=None
):
    """Validate + unpack the block scale buffers for kernel launch.

    Returns ``(k_sf, v_sf, ksf_bs, ksf_h, vsf_bs, vsf_h, use_kv_quant)``.
    With both buffers None (the bf16 / per-tensor-fp8 path) this returns the
    inert defaults, keeping the launch byte-identical to the pre-quant code.

    Layouts (page_size=1 only, both scale bytes are biased exponents, the
    dequant multiplier being 2^(sf - 127)):

    * mxfp8 -- ``MHATokenToKVPoolMXFP8``'s flat ``(m, H_kv, head_dim // 32)``
      ``torch.float8_e8m0fnu``; triton has no e8m0 type so it is
      reinterpreted as uint8.
    * nvfp4 -- ``MHATokenToKVPoolFP4``'s flat ``(m, H_kv * head_dim // 16)``
      ``uint8``, reshaped here to ``(m, H_kv, head_dim // 16)``. That reshape
      is EXACT: ``FP4MXBlock16KVQuantizeUtil.batched_quantize`` blocks a
      ``(T, H, D)`` tensor as ``view(T, H*D//16, 16)``, so the scale for
      ``(h, d)`` sits at column ``h * (D // 16) + d // 16`` and ``head_dim``
      is a multiple of 16.
    """
    if k_sf_buffer is None and v_sf_buffer is None:
        return None, None, 0, 0, 0, 0, False
    if k_sf_buffer is None or v_sf_buffer is None:
        raise ValueError(
            "quantized KV requires BOTH k_sf_buffer and v_sf_buffer (got one None)."
        )
    if page_size != 1:
        raise NotImplementedError(
            "block-scaled quantized KV supports only the flat page_size=1 scale "
            f"layout; got page_size={page_size}."
        )
    if kv_fp4:
        sf_block = SF_BLOCK_FP4
        if head_dim is None:
            raise ValueError("nvfp4 scale prep requires head_dim.")
        if v_head_dim is None:
            v_head_dim = head_dim
        for _name, _d in (("head_dim", head_dim), ("v_head_dim", v_head_dim)):
            if _d % sf_block != 0:
                raise NotImplementedError(
                    f"nvfp4 KV requires {_name} % {sf_block} == 0; got {_d}."
                )
        # (m, H_kv * D // 16) uint8 -> (m, H_kv, D // 16); .view() is safe
        # because the pool allocates these contiguous.
        def _as3d(t, name, hd):
            if t.dtype != torch.uint8:
                t = t.view(torch.uint8)
            if t.ndim == 3:
                return t
            if t.ndim != 2:
                raise NotImplementedError(
                    f"nvfp4 {name} scale buffer must be 2-D (m, H*D//16) or 3-D "
                    f"(m, H, D//16); got ndim {t.ndim}."
                )
            per_head = hd // sf_block
            if t.shape[1] % per_head != 0:
                raise NotImplementedError(
                    f"nvfp4 {name} scale row {t.shape[1]} is not a multiple of "
                    f"head_dim//{sf_block}={per_head}."
                )
            return t.view(t.shape[0], t.shape[1] // per_head, per_head)

        k_sf = _as3d(k_sf_buffer, "K", head_dim)
        v_sf = _as3d(v_sf_buffer, "V", v_head_dim)
    else:
        if k_sf_buffer.ndim != 3 or v_sf_buffer.ndim != 3:
            raise NotImplementedError(
                "mxfp8 KV scale buffers must be the flat (m, H_kv, head_dim//32) "
                "layout; the fa4-interleaved page_size=128 layout is unsupported "
                f"(got ndim {k_sf_buffer.ndim}/{v_sf_buffer.ndim})."
            )
        k_sf = k_sf_buffer.view(torch.uint8)
        v_sf = v_sf_buffer.view(torch.uint8)
    return (
        k_sf,
        v_sf,
        k_sf.stride(0),
        k_sf.stride(1),
        v_sf.stride(0),
        v_sf.stride(1),
        True,
    )


@triton.jit
def _dequant_kv_block(payload, sf_u8, parity, KV_FP4: tl.constexpr):
    """Block-scaled dequant of one KV tile to fp32.

    ``sf_u8`` is the biased-exponent scale byte (multiplier 2^(sf - 127)),
    loaded with ``other=127`` on masked lanes so they contribute 0 * 1.0.

    For ``KV_FP4`` the payload byte holds two e2m1 values; ``parity`` (the
    head_dim index mod 2) picks the low nibble for even indices. e2m1 decode
    (sign | 2-bit exponent | 1-bit mantissa), magnitudes
    {0, .5, 1, 1.5, 2, 3, 4, 6}: for m >= 2 the value is
    (1 + .5*(m & 1)) * 2^((m >> 1) - 1); m < 2 is the subnormal 0.5 * m.
    """
    scale = tl.exp2(sf_u8.to(tl.float32) - 127.0)
    if KV_FP4:
        b = payload.to(tl.int32)
        nib = tl.where(parity == 0, b & 0x0F, (b >> 4) & 0x0F)
        m = nib & 0x07
        mant = 1.0 + 0.5 * (m & 1).to(tl.float32)
        val = mant * tl.exp2(((m // 2) - 1).to(tl.float32))
        val = tl.where(m < 2, 0.5 * m.to(tl.float32), val)
        val = tl.where((nib & 0x08) != 0, -val, val)
    else:
        val = payload.to(tl.float32)
    return val * scale


@triton.jit
def _fwd_kernel_stage1_kv_quant(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1). For
    # PAGE_SIZE == 1 the address math degenerates and these are unused
    # (Triton specializes the dead branch away at compile time).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
    # mxfp8 KV: flat per-slot UE8M0 scale buffers (uint8 views), one scale per
    # 32-element block along head_dim. Defaults keep the bf16 path unchanged
    # (USE_KV_QUANT=False specializes every new branch away at compile time).
    # page_size=1 layout only (asserted in the python wrapper).
    K_SF=None,
    V_SF=None,
    stride_ksf_bs=0,
    stride_ksf_h=0,
    stride_vsf_bs=0,
    stride_vsf_h=0,
    USE_KV_QUANT: tl.constexpr = False,
    # nvfp4 (fp4_mx_block16): payload rows are PACKED uint8 (two e2m1
    # nibbles per byte, even head_dim index in the low nibble) and
    # SF_BLOCK is 16. For mxfp8 the payload is fp8-e4m3 and SF_BLOCK 32.
    KV_FP4: tl.constexpr = False,
    SF_BLOCK: tl.constexpr = 32,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    # Payload column index into the KV buffers. nvfp4 packs two head_dim
    # entries per stored byte, so the column is offs_d // 2 and the nibble is
    # selected by the parity (even -> low nibble, matching
    # FP4MXBlock16KVQuantizeUtil.batched_quantize's
    # ``packed = (v[..., 1::2] << 4) + v[..., 0::2]``). For mxfp8/bf16 the
    # column IS offs_d and the parity operand is unused.
    if KV_FP4:
        kcol = offs_d // 2
        vcol = offs_dv // 2
        kpar = offs_d % 2
        vpar = offs_dv % 2
    else:
        kcol = offs_d
        vcol = offs_dv
        kpar = offs_d
        vpar = offs_dv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Page-aware KV address math. At PAGE_SIZE==1 (legacy
            # / non-shared / shared-at-ps=1), Triton specializes the
            # else-branch away and the SASS is byte-identical to today.
            if PAGE_SIZE == 1:
                offs_buf_k = (
                    kv_loc[:, None] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + kcol[None, :]
                )
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[:, None] * stride_buf_kpage
                    + tok_in_p[:, None] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + kcol[None, :]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            if USE_KV_QUANT:
                # Blockwise dequant: payload is fp8-e4m3, scale index is
                # offs_d // 32 (flat page_size=1 layout, same slot/head
                # indexing as the payload). other=127 -> scale 2^0 = 1.0 for
                # masked lanes (payload there is already 0).
                offs_ksf = (
                    kv_loc[:, None] * stride_ksf_bs
                    + cur_kv_head * stride_ksf_h
                    + (offs_d[None, :] // SF_BLOCK)
                )
                k_sf = tl.load(
                    K_SF + offs_ksf,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                    other=127,
                )
                k = _dequant_kv_block(
                    k, k_sf, kpar[None, :], KV_FP4
                ).to(q.dtype)
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    cur_batch_seq_len - 1,
                    offs_n,
                    cur_batch,
                    cur_head,
                    offs_n < split_kv_end,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + vcol[None, :]
                )
            else:
                offs_buf_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + vcol[None, :]
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )
            if USE_KV_QUANT:
                offs_vsf = (
                    kv_loc[:, None] * stride_vsf_bs
                    + cur_kv_head * stride_vsf_h
                    + (offs_dv[None, :] // SF_BLOCK)
                )
                v_sf = tl.load(
                    V_SF + offs_vsf,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=127,
                )
                v = _dequant_kv_block(
                    v, v_sf, vpar[None, :], KV_FP4
                ).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_grouped_kernel_stage1_kv_quant(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
    # mxfp8 KV: flat per-slot UE8M0 scale buffers (uint8 views), one scale per
    # 32-element block along head_dim. Defaults keep the bf16 path unchanged.
    # page_size=1 layout only; incompatible with HAS_MLA / BLOCK_DPE > 0
    # (asserted in the python wrapper).
    K_SF=None,
    V_SF=None,
    stride_ksf_bs=0,
    stride_ksf_h=0,
    stride_vsf_bs=0,
    stride_vsf_h=0,
    USE_KV_QUANT: tl.constexpr = False,
    # nvfp4 (fp4_mx_block16): payload rows are PACKED uint8 (two e2m1
    # nibbles per byte, even head_dim index in the low nibble) and
    # SF_BLOCK is 16. For mxfp8 the payload is fp8-e4m3 and SF_BLOCK 32.
    KV_FP4: tl.constexpr = False,
    SF_BLOCK: tl.constexpr = 32,
):
    # int64 to avoid overflow of flat offsets into Mid_O when
    # batch * num_head * max_kv_splits * head_dim exceeds 2**31.
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    # Payload column index into the KV buffers. nvfp4 packs two head_dim
    # entries per stored byte, so the column is offs_d // 2 and the nibble is
    # selected by the parity (even -> low nibble, matching
    # FP4MXBlock16KVQuantizeUtil.batched_quantize's
    # ``packed = (v[..., 1::2] << 4) + v[..., 0::2]``). For mxfp8/bf16 the
    # column IS offs_d and the parity operand is unused.
    if KV_FP4:
        kcol = offs_d // 2
        vcol = offs_dv // 2
        kpar = offs_d % 2
        vpar = offs_dv % 2
    else:
        kcol = offs_d
        vcol = offs_dv
        kpar = offs_d
        vpar = offs_dv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    # Hoist loop-invariant base offsets
    base_offs_k = cur_kv_head * stride_buf_kh + kcol[:, None]
    if BLOCK_DPE > 0:
        base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
    if not HAS_MLA:
        base_offs_v = cur_kv_head * stride_buf_vh + vcol[None, :]

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if USE_KV_QUANT:
            # K_Buffer holds raw fp8-e4m3 payload here; casting Q to it would
            # destroy Q (the KV-dtype == compute-dtype assumption does not
            # hold). Q stays in its own (bf16) dtype; K is dequantized to
            # q.dtype below before the dot.
            q_k = q
        else:
            q_k = q.to(K_Buffer.dtype.element_ty)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            # Page-aware KV address math (see _fwd_kernel_stage1).
            if PAGE_SIZE == 1:
                offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + base_offs_k
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            if USE_KV_QUANT:
                # Blockwise dequant (k is transposed [D, N] here): scale index
                # along D is offs_d // 32; slot/head indexing matches the
                # payload. other=127 -> scale 1.0 for masked lanes.
                offs_ksf = (
                    kv_loc[None, :] * stride_ksf_bs
                    + cur_kv_head * stride_ksf_h
                    + (offs_d[:, None] // SF_BLOCK)
                )
                k_sf = tl.load(
                    K_SF + offs_ksf,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                    other=127,
                )
                k = _dequant_kv_block(
                    k, k_sf, kpar[:, None], KV_FP4
                ).to(q.dtype)
            qk = tl.dot(q_k, k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                else:
                    offs_buf_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + base_offs_kpe
                    )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale_withk

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    cur_batch_seq_len - 1,
                    offs_n[None, :],
                    cur_batch,
                    cur_head[:, None],
                    mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )
            if HAS_MLA:
                v = tl.trans(k)
            else:
                if PAGE_SIZE == 1:
                    offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
                else:
                    offs_buf_v = (
                        page_id[:, None] * stride_buf_vpage
                        + tok_in_p[:, None] * stride_buf_vtok
                        + base_offs_v
                    )
                v = tl.load(
                    V_Buffer + offs_buf_v,
                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                    other=0.0,
                )
                if USE_KV_QUANT:
                    offs_vsf = (
                        kv_loc[:, None] * stride_vsf_bs
                        + cur_kv_head * stride_vsf_h
                        + (offs_dv[None, :] // SF_BLOCK)
                    )
                    v_sf = tl.load(
                        V_SF + offs_vsf,
                        mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                        other=127,
                    )
                    v = _dequant_kv_block(
                        v, v_sf, vpar[None, :], KV_FP4
                    ).to(q.dtype)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _fwd_kernel_kv_quant(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    LSE_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_indptr,
    sink_ptr,
    window_kv_offset_ptr,
    sm_scale,
    k_scale,
    v_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    # Page-aware strides (used when PAGE_SIZE > 1).
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: tl.constexpr,
    STORE_LSE: tl.constexpr,
    SKIP_PREFIX: tl.constexpr,
    SKIP_EXTEND: tl.constexpr,
    STORE_TRANSPOSE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    PAGE_SIZE: tl.constexpr = 1,
    SCORE_MOD: tl.constexpr = None,
    Aux0=None,
    aux0_stride_t=0,
    aux0_stride_h=0,
    aux0_len=0,
    # mxfp8 KV: flat per-slot UE8M0 scale buffers (uint8 views) for the PREFIX
    # K_Buffer/V_Buffer pool reads only — stage 2 consumes the bf16
    # K_Extend/V_Extend chunk and needs no dequant. Defaults keep the bf16
    # path unchanged (USE_KV_QUANT=False specializes the new branches away).
    # page_size=1 layout only (asserted in the python wrapper).
    K_SF=None,
    V_SF=None,
    stride_ksf_bs=0,
    stride_ksf_h=0,
    stride_vsf_bs=0,
    stride_vsf_h=0,
    USE_KV_QUANT: tl.constexpr = False,
    # nvfp4 (fp4_mx_block16): payload rows are PACKED uint8 (two e2m1
    # nibbles per byte, even head_dim index in the low nibble) and
    # SF_BLOCK is 16. For mxfp8 the payload is fp8-e4m3 and SF_BLOCK 32.
    KV_FP4: tl.constexpr = False,
    SF_BLOCK: tl.constexpr = 32,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq)
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq)
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend

    # Grid axis 2 spans the batch-max extend length; all stores are masked by mask_m.
    if cur_block_m * BLOCK_M >= cur_seq_len_extend:
        return

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_indptr + cur_seq)

    # For SWA, we should only load the mask in the sliding window
    window_kv_offset = 0
    if USE_CUSTOM_MASK and SLIDING_WINDOW_SIZE > 0:
        window_kv_offset = tl.load(window_kv_offset_ptr + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    # Payload column index into the KV buffers. nvfp4 packs two head_dim
    # entries per stored byte, so the column is offs_d // 2 and the nibble is
    # selected by the parity (even -> low nibble, matching
    # FP4MXBlock16KVQuantizeUtil.batched_quantize's
    # ``packed = (v[..., 1::2] << 4) + v[..., 0::2]``). For mxfp8/bf16 the
    # column IS offs_d and the parity operand is unused.
    if KV_FP4:
        kcol = offs_d // 2
        vcol = offs_dv // 2
        kpar = offs_d % 2
        vpar = offs_dv % 2
    else:
        kcol = offs_d
        vcol = offs_dv
        kpar = offs_d
        vpar = offs_dv

    if xai_temperature_len > 0:
        offs_qidx = cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        xai_temperature_reg = tl.where(
            offs_qidx > xai_temperature_len,
            tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale,
            1.0,
        )

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    prefix_end = 0 if SKIP_PREFIX else cur_seq_len_prefix
    for start_n in range(0, prefix_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            final_mask &= custom_mask
        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            # q_id = prefix_len + cur_m, kv_id = cur_n
            window_mask = (
                cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m[:, None]
            ) <= (start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE)
            final_mask &= window_mask

        SKIP_TILE = False
        if (USE_CUSTOM_MASK and not SKIP_PREFIX_CUSTOM_MASK) or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            offs_kv_loc = tl.load(
                kv_indices + cur_seq_kv_start_idx + start_n + offs_n,
                mask=mask_n,
                other=0,
            )

            # Page-aware KV address math. At PAGE_SIZE==1
            # (legacy / non-shared / shared-at-ps=1), Triton specializes
            # the else-branch away — byte-identical SASS to today.
            if PAGE_SIZE == 1:
                # load k in transposed way
                offs_buf_k = (
                    offs_kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + kcol[:, None]
                )
            else:
                page_id = offs_kv_loc // PAGE_SIZE
                tok_in_p = offs_kv_loc % PAGE_SIZE
                offs_buf_k = (
                    page_id[None, :] * stride_buf_kpage
                    + tok_in_p[None, :] * stride_buf_ktok
                    + cur_kv_head * stride_buf_kh
                    + kcol[:, None]
                )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(mask_n[None, :]) & (mask_d[:, None]),
                other=0.0,
            )
            if USE_KV_QUANT:
                # Blockwise dequant (k is transposed [D, N] here): scale index
                # along D is offs_d // 32, slot/head indexing matches the
                # payload (flat page_size=1 layout). other=127 -> scale 1.0
                # for masked lanes (payload there is already 0). After this,
                # k.dtype == q.dtype so the q.to(k.dtype) below is a no-op —
                # Q is NEVER cast to the fp8 storage dtype.
                offs_ksf = (
                    offs_kv_loc[None, :] * stride_ksf_bs
                    + cur_kv_head * stride_ksf_h
                    + (offs_d[:, None] // SF_BLOCK)
                )
                k_sf = tl.load(
                    K_SF + offs_ksf,
                    mask=(mask_n[None, :]) & (mask_d[:, None]),
                    other=127,
                )
                k = _dequant_kv_block(
                    k, k_sf, kpar[:, None], KV_FP4
                ).to(q.dtype)
            qk = tl.dot(q.to(k.dtype), k)
            if BLOCK_DPE > 0:
                if PAGE_SIZE == 1:
                    offs_kpe = (
                        offs_kv_loc[None, :] * stride_buf_kbs
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                else:
                    offs_kpe = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + cur_kv_head * stride_buf_kh
                        + offs_dpe[:, None]
                    )
                kpe = tl.load(
                    K_Buffer + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe.to(kpe.dtype), kpe)
            qk *= sm_scale * k_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    start_n + offs_n[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            if PAGE_SIZE == 1:
                offs_buf_v = (
                    offs_kv_loc[:, None] * stride_buf_vbs
                    + cur_kv_head * stride_buf_vh
                    + vcol[None, :]
                )
            else:
                offs_buf_v = (
                    page_id[:, None] * stride_buf_vpage
                    + tok_in_p[:, None] * stride_buf_vtok
                    + cur_kv_head * stride_buf_vh
                    + vcol[None, :]
                )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=mask_n[:, None] & mask_dv[None, :],
                other=0.0,
            )
            if USE_KV_QUANT:
                # Blockwise dequant to q.dtype BEFORE p = p.to(v.dtype): p
                # must not be silently cast to the fp8 storage dtype.
                offs_vsf = (
                    offs_kv_loc[:, None] * stride_vsf_bs
                    + cur_kv_head * stride_vsf_h
                    + (offs_dv[None, :] // SF_BLOCK)
                )
                v_sf = tl.load(
                    V_SF + offs_vsf,
                    mask=mask_n[:, None] & mask_dv[None, :],
                    other=127,
                )
                v = _dequant_kv_block(
                    v, v_sf, vpar[None, :], KV_FP4
                ).to(q.dtype)
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale

            e_max = n_e_max

    # stage 2: compute the triangle part

    cur_block_m_end = (
        cur_seq_len_extend
        if not IS_CAUSAL
        else tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    )
    extend_end = 0 if SKIP_EXTEND else cur_block_m_end
    for start_n in range(0, extend_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        final_mask = mask_m[:, None] & mask_n[None, :]
        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None])
                * (cur_seq_len + window_kv_offset)
                + window_kv_offset
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            final_mask &= custom_mask
        elif IS_CAUSAL:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_causual
        else:
            mask_non_causal = mask_m[:, None] & mask_n[None, :]
            final_mask &= mask_non_causal

        if SLIDING_WINDOW_SIZE > 0:
            # Add mask where q_id <= kv_id + sliding_window_size
            window_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) <= (
                start_n + offs_n[None, :] + SLIDING_WINDOW_SIZE
            )
            final_mask &= window_mask

        SKIP_TILE = False
        if USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            SKIP_TILE = tl.max(tl.max(final_mask.to(tl.int32), axis=1), axis=0) == 0

        if not SKIP_TILE:
            # load k in transposed way
            offs_k = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)
            if BLOCK_DPE > 0:
                offs_kpe = (
                    (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                    + cur_kv_head * stride_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Extend + offs_kpe,
                    mask=mask_n[None, :],
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            if SCORE_MOD is not None:
                qk = SCORE_MOD(
                    qk,
                    (cur_seq_len_prefix + cur_block_m * BLOCK_M + offs_m)[:, None],
                    cur_seq_len_prefix + start_n + offs_n[None, :],
                    (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m)[
                        :, None
                    ],
                    cur_head,
                    final_mask,
                    Aux0,
                    aux0_stride_t,
                    aux0_stride_h,
                    aux0_len,
                )

            qk = tl.where(final_mask, qk, float("-inf"))

            row_max = tl.max(qk, 1)
            row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
            n_e_max = tl.maximum(row_max_fixed, e_max)

            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            deno = deno * re_scale + tl.sum(p, 1)

            offs_v = (
                (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
                + cur_kv_head * stride_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
            )
            p = p.to(v.dtype)
            acc = acc * re_scale[:, None] + tl.dot(p, v)

            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        deno += tl.exp(cur_sink - e_max)

    if STORE_LSE:
        offs_lse = (
            cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m
        ) * stride_lse_bs + cur_head * stride_lse_h
        lse = tl.log(deno) + e_max
        tl.store(LSE_Extend + offs_lse, lse, mask=mask_m)

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    if STORE_TRANSPOSE:
        tl.store(
            O_Extend + offs_o.T,
            (acc / deno[:, None]).T,
            mask=(mask_m[:, None] & mask_dv[None, :]).T,
        )
    else:
        tl.store(
            O_Extend + offs_o,
            acc / deno[:, None],
            mask=mask_m[:, None] & mask_dv[None, :],
        )


def decode_attention_fwd_kv_quant(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    attn_lse,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    k_scale,
    v_scale,
    k_sf_buffer,
    v_sf_buffer,
    logit_cap=0.0,
    sinks=None,
    xai_temperature_len=-1,
    has_mla=False,
    use_pdl=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    kv_fp4: bool = False,
):
    """Block-scaled-KV twin of ``decode_attention_fwd``: identical contract
    plus the REQUIRED scale buffers. ``kv_fp4=False`` -> k_buffer/v_buffer
    hold fp8-e4m3 payload with block-32 scales; ``kv_fp4=True`` -> they hold
    PACKED e2m1 (row width head_dim // 2) with block-16 scales."""
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    sf_block = SF_BLOCK_FP4 if kv_fp4 else SF_BLOCK_MXFP8
    # The pool row is head_dim for mxfp8 and head_dim // 2 for packed nvfp4;
    # Lk/Lv must be the LOGICAL head dims (they drive offs_d and the masks).
    Lk = k_buffer.shape[-1] * (2 if kv_fp4 else 1)
    Lv = v_buffer.shape[-1] * (2 if kv_fp4 else 1)

    k_sf, v_sf, ksf_bs, ksf_h, vsf_bs, vsf_h, _use = _prepare_kv_sf_args(
        k_sf_buffer, v_sf_buffer, page_size, kv_fp4=kv_fp4, head_dim=Lk, v_head_dim=Lv
    )
    if not _use:
        raise ValueError(
            "decode_attention_fwd_kv_quant requires k_sf_buffer and v_sf_buffer; "
            "use decode_attention_fwd for the bf16 / per-tensor-scale path."
        )

    if has_mla or Lk in (576, 288):
        raise NotImplementedError(
            f"quantized KV decode does not support MLA layouts "
            f"(has_mla={has_mla}, Lk={Lk})."
        )
    if q.shape[-1] != Lk:
        raise ValueError(
            f"quantized KV decode: q head_dim {q.shape[-1]} != pool head_dim {Lk} "
            f"(kv_fp4={kv_fp4})."
        )

    kv_head_num = v_buffer.shape[-2]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // kv_head_num

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )
    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )
    MAX_KV_SPLITS = max_kv_splits
    BLOCK_DV = triton.next_power_of_2(Lv)

    if kv_group_num == 1:
        # MHA variant (mirrors _decode_att_m_fwd's prep).
        BLOCK = 64
        if _is_hip:
            BLOCK = 8
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        grid = (batch, head_num, MAX_KV_SPLITS)
        num_warps = 4
        _fwd_kernel_stage1_kv_quant[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale * k_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_slot_stride,
            k_head_stride,
            v_slot_stride,
            v_head_stride,
            k_page_stride,
            k_tok_stride,
            v_page_stride,
            v_tok_stride,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            kv_group_num=kv_group_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            logit_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
            num_warps=num_warps,
            num_stages=2,
            Lk=Lk,
            Lv=Lv,
            PAGE_SIZE=page_size,
            SCORE_MOD=score_mod,
            Aux0=aux0,
            aux0_stride_t=aux0_stride_t,
            aux0_stride_h=aux0_stride_h,
            aux0_len=aux0_len,
            K_SF=k_sf,
            V_SF=v_sf,
            stride_ksf_bs=ksf_bs,
            stride_ksf_h=ksf_h,
            stride_vsf_bs=vsf_bs,
            stride_vsf_h=vsf_h,
            USE_KV_QUANT=True,
            KV_FP4=kv_fp4,
            SF_BLOCK=sf_block,
        )
    else:
        # GQA variant (mirrors _decode_grouped_att_m_fwd's prep).
        BLOCK = 32
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
        BLOCK_H = 16
        grid = (
            batch,
            triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
            MAX_KV_SPLITS,
        )
        extra_kargs = {}
        num_stages = 2
        if _is_hip:
            extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
            num_stages = 1
        _fwd_grouped_kernel_stage1_kv_quant[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale * k_scale,
            kv_indptr,
            kv_indices,
            attn_logits,
            attn_lse,
            num_kv_splits,
            q.stride(0),
            q.stride(1),
            k_slot_stride,
            k_head_stride,
            v_slot_stride,
            v_head_stride,
            k_page_stride,
            k_tok_stride,
            v_page_stride,
            v_tok_stride,
            attn_logits.stride(0),
            attn_logits.stride(1),
            attn_logits.stride(2),
            kv_group_num=kv_group_num,
            q_head_num=head_num,
            BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_DPE=BLOCK_DPE,
            BLOCK_DV=BLOCK_DV,
            BLOCK_N=BLOCK,
            BLOCK_H=BLOCK_H,
            MIN_BLOCK_KV=_MIN_BLOCK_KV,
            logit_cap=logit_cap,
            xai_temperature_len=xai_temperature_len,
            num_warps=4,
            num_stages=num_stages,
            Lk=Lk,
            Lv=Lv,
            HAS_MLA=False,
            USE_PDL=use_pdl,
            PAGE_SIZE=page_size,
            SCORE_MOD=score_mod,
            Aux0=aux0,
            aux0_stride_t=aux0_stride_t,
            aux0_stride_h=aux0_stride_h,
            aux0_len=aux0_len,
            K_SF=k_sf,
            V_SF=v_sf,
            stride_ksf_bs=ksf_bs,
            stride_ksf_h=ksf_h,
            stride_vsf_bs=vsf_bs,
            stride_vsf_h=vsf_h,
            USE_KV_QUANT=True,
            KV_FP4=kv_fp4,
            SF_BLOCK=sf_block,
            **extra_kargs,
        )

    # Stage 2 (softmax-reduce over fp32 partials) is quantization-agnostic:
    # reuse the UNMODIFIED upstream kernel/wrapper. It reads ONLY
    # ``v_buffer.shape[-1]`` (to derive Lv / BLOCK_DV), and the packed nvfp4
    # pool row is head_dim // 2 -- handing it the raw buffer would reduce and
    # store only half the head. Pass a zero-element shape proxy instead.
    v_shape_proxy = v_buffer if not kv_fp4 else v_buffer.new_empty((0, Lv))
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_scale,
        v_shape_proxy,
        kv_indptr,
        num_kv_splits,
        max_kv_splits,
        sinks,
        use_pdl=use_pdl,
    )


def extend_attention_fwd_kv_quant(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    is_causal,
    mask_indptr,
    max_len_extend,
    k_scale,
    v_scale,
    k_sf_buffer,
    v_sf_buffer,
    sm_scale=None,
    logit_cap=0.0,
    skip_prefix_custom_mask=True,
    sliding_window_size=-1,
    sinks=None,
    window_kv_offsets=None,
    xai_temperature_len=-1,
    lse_extend=None,
    skip_prefix=False,
    skip_extend=False,
    page_size: int = 1,
    score_mod=None,
    aux_tensors=None,
    kv_fp4: bool = False,
):
    """Block-scaled-KV twin of ``extend_attention_fwd``: identical contract
    plus the REQUIRED scale buffers for the PREFIX pool reads. k_extend/
    v_extend (the current chunk) stay bf16 -- stage 2 of the clone is
    untouched. ``kv_fp4`` selects the packed-e2m1/block-16 recipe instead of
    mxfp8's fp8-e4m3/block-32."""
    Lq, Lk, Lv = (
        q_extend.shape[-1],
        k_extend.shape[-1],
        v_extend.shape[-1],
    )
    sf_block = SF_BLOCK_FP4 if kv_fp4 else SF_BLOCK_MXFP8

    BLOCK_DMODEL, BLOCK_DPE, BLOCK_DV, BLOCK_M, BLOCK_N, num_warps = (
        _get_block_sizes_for_extend_attention(Lq, Lv)
    )
    if BLOCK_DPE > 0:
        raise NotImplementedError(
            f"quantized KV extend does not support MLA head layouts (Lq={Lq})."
        )
    if kv_fp4 and k_buffer.shape[-1] * 2 != Lk:
        raise ValueError(
            "nvfp4 KV extend: packed pool row "
            f"{k_buffer.shape[-1]} does not match head_dim {Lk} // 2."
        )

    k_sf, v_sf, ksf_bs, ksf_h, vsf_bs, vsf_h, _use = _prepare_kv_sf_args(
        k_sf_buffer, v_sf_buffer, page_size, kv_fp4=kv_fp4, head_dim=Lk, v_head_dim=Lv
    )
    if not _use:
        raise ValueError(
            "extend_attention_fwd_kv_quant requires k_sf_buffer and v_sf_buffer; "
            "use extend_attention_fwd for the bf16 / per-tensor-scale path."
        )

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None
    SKIP_PREFIX_CUSTOM_MASK = skip_prefix_custom_mask

    HAS_SINK = sinks is not None
    STORE_LSE = lse_extend is not None
    stride_lse_bs = lse_extend.stride(0) if STORE_LSE else 0
    stride_lse_h = lse_extend.stride(1) if STORE_LSE else 0

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    num_stages = 1

    extra_kargs = {}
    if _is_hip:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    k_slot_stride, k_head_stride, k_page_stride, k_tok_stride = _extract_kv_strides(
        k_buffer, page_size
    )
    v_slot_stride, v_head_stride, v_page_stride, v_tok_stride = _extract_kv_strides(
        v_buffer, page_size
    )

    aux0, aux0_stride_t, aux0_stride_h, aux0_len = unpack_aux_tensors(
        score_mod, aux_tensors
    )

    _fwd_kernel_kv_quant[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        lse_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_indptr,
        sinks,
        window_kv_offsets,
        sm_scale,
        k_scale,
        v_scale,
        kv_group_num,
        q_extend.stride(0),
        q_extend.stride(1),
        k_extend.stride(0),
        k_extend.stride(1),
        v_extend.stride(0),
        v_extend.stride(1),
        o_extend.stride(0),
        o_extend.stride(1),
        stride_lse_bs,
        stride_lse_h,
        k_slot_stride,
        k_head_stride,
        v_slot_stride,
        v_head_stride,
        k_page_stride,
        k_tok_stride,
        v_page_stride,
        v_tok_stride,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        IS_CAUSAL=is_causal,
        SKIP_PREFIX_CUSTOM_MASK=SKIP_PREFIX_CUSTOM_MASK,
        STORE_LSE=STORE_LSE,
        SKIP_PREFIX=skip_prefix,
        SKIP_EXTEND=skip_extend,
        HAS_SINK=HAS_SINK,
        STORE_TRANSPOSE=_is_hip,
        PAGE_SIZE=page_size,
        SCORE_MOD=score_mod,
        Aux0=aux0,
        aux0_stride_t=aux0_stride_t,
        aux0_stride_h=aux0_stride_h,
        aux0_len=aux0_len,
        K_SF=k_sf,
        V_SF=v_sf,
        stride_ksf_bs=ksf_bs,
        stride_ksf_h=ksf_h,
        stride_vsf_bs=vsf_bs,
        stride_vsf_h=vsf_h,
        USE_KV_QUANT=True,
        KV_FP4=kv_fp4,
        SF_BLOCK=sf_block,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )
