# Quantized KV cache (mxfp8 → nvfp4) for the SGLang TRITON attention backend on GB10 (sm_121a)

Campaign scope: fix BUGS-AND-FIXES #13b (`TypeError: TritonAttnBackend.forward_extend() got an
unexpected keyword argument 'q_descale'`) by teaching the triton backend + kernels to read
block-scaled quantized KV. mxfp8 first (fp8 payload + per-32-elem UE8M0 scales), then nvfp4
(packed e2m1 pairs + per-16-elem scales).

All paths below are the read-only source mirror:
- `SRC = ~/nvfp4-kv-triton/image-src` (extracted from the working docker image)
- `ATTN = ~/inkling-ablit/patches/attn.py` (local mirror of the image's
  `sglang/srt/models/inkling_common/attn.py` — the file that PASSES the crashing kwargs)

---

## 1. WHERE the descale kwargs originate, their shapes, and pool state

### 1.1 Call chain (crash anatomy)

1. **`InklingAttention.forward`** — `sglang/srt/models/inkling_common/attn.py` (mirror:
   `ATTN:872-948`). When `server_args.kv_cache_dtype == "mxfp8"` it builds
   `extra_attn_kwargs`:
   - `ATTN:883-890` — Q is quantized with `to_mxfp8(q.view(T, num_tp_heads, head_dim))`
     (from `sglang.kernels.ops.quantization.mxfp8_quant`); `extra_attn_kwargs["q_descale"] =
     q_mxfp.scale.view(torch.float8_e8m0fnu)`. If the fused attn prologue ran
     (`head_dim == 128`, fa4-only gate at `ATTN:748-756`), `prologue_q_descale` is used instead.
   - `ATTN:891-904` — when the prologue did NOT store and
     `SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE` is off, K and V are quantized too and
     `k_descale` / `v_descale` are attached.
   - `ATTN:938-948` — the **triton lane** calls
     `self.attn(q, k, v, forward_batch, save_kv_cache=…, score_mod=triton_relative_bias_score_mod,
     aux_tensors=[rel_logits], **extra_attn_kwargs)`.
2. **`RadixAttention.forward`** (in-image, not in extract) forwards `**kwargs` unchanged to
   the attention backend.
3. **`AttentionBackend.forward`** — `SRC/srt/layers/attention/base_attn_backend.py:188-231`:
   dispatches on forward mode and forwards `**kwargs` verbatim to `forward_decode` /
   `forward_extend`.
4. **`TritonAttnBackend.forward_extend`** — `SRC/srt/layers/attention/triton_backend.py:1229-1240`
   — signature accepts only `(…, save_kv_cache=True, sinks=None, score_mod=None,
   aux_tensors=None)`. `q_descale` is unexpected → **TypeError**. `forward_decode`
   (`triton_backend.py:1690-1701`) has the identical gap; extend crashes first because the
   first request prefills.

BUGS-AND-FIXES #13b's diagnosis is confirmed: the descale plumbing exists only in the fa4 /
flashinfer backends (fa4 consumes them as `sfq`/`sfk`/`sfv` per the "FA4 downloads contract"
comment at `ATTN:879-882`).

### 1.2 Exact shapes/dtypes of the kwargs (mxfp8)

Let `T` = tokens in the batch, `H_q = num_tp_heads`, `H_kv = num_tp_kv_heads`, `D = head_dim`.

| kwarg | producer | shape | dtype |
|---|---|---|---|
| `q_descale` | `to_mxfp8` at `ATTN:888-890` (or fused prologue) | `(T, H_q, D//32)` | `float8_e8m0fnu` (view) |
| `k_descale` | `to_mxfp8` at `ATTN:895-897` | `(T, H_kv, D//32)` | `float8_e8m0fnu` |
| `v_descale` | `to_mxfp8` at `ATTN:898-900` | `(T, H_kv, D//32)` | `float8_e8m0fnu` |

With these set, `q`, `k`, `v` arriving at the backend are **already fp8 e4m3 payloads**
(`q = q_mxfp.data.view(T, -1)`, `ATTN:889/901-902`) — not bf16. This matters for Stage 1
design (see §4.0: we gate the pre-quant to fa4 instead of dequantizing Q in-kernel).

For **nvfp4** the model file has NO analogous branch — `ATTN:873` tests only `== "mxfp8"`.
The nvfp4 lane relies purely on the pool's own quantize-on-store (see §1.4), so no descale
kwargs would even be produced; the crash for nvfp4 comes from the same kwargs only if the fa4
descale plumbing in `RadixAttention`/quant-method paths adds them (Fp8KVCacheMethod →
`layer.k_scale`). Triton must still accept-and-route the kwargs for mxfp8, and needs
completely new pool wiring for nvfp4.

### 1.3 mxfp8 pool: storage layout and write path — WRITE PATH IS DONE (page_size=1)

`MHATokenToKVPoolMXFP8` — `SRC/srt/mem_cache/memory_pool.py:3232-3517`:

- Payload: `k_buffer[layer]` = `(m, H_kv, D)`, `v_buffer[layer]` = `(m, H_kv, Dv)`,
  dtype `float8_e4m3fn` (`memory_pool.py:3274-3282`), `m = size + page_size`.
- Scales: one UE8M0 per 32-elem block (`MXFP8_SCALE_BLOCK_SIZE = 32`, line 3239).
  - `page_size == 128` → **interleaved** FA4 `BlockScaledBasicChunk` layout
    `(num_pages, H, 32, page//32, sf_dim)` (`3292-3310`), written by `store_sf_interleaved`.
  - **any other page size (our champion page_size=1, per bug #12) → flat**
    `(m, H_kv, D//32)` `float8_e8m0fnu` (`3311-3325`). This is the layout the triton kernels
    will read — a plain per-slot gather, same indexing as the payload.
- Write: `set_kv_buffer(layer, loc, cache_k, cache_v, k_scale, v_scale)` (`3362-3417`).
  With scale tensors provided it scatters payload + flat scales (`_write_scales`,
  `3419-3435`). **Without** scales it requires the fused-quant path which asserts
  `mxfp8_sf_interleaved` i.e. page_size==128 (`3384-3401`) — so at page_size=1 the caller
  MUST hand fp8 K/V + e8m0 scale tensors (exactly what `ATTN:895-904` produces).
- Read: `_get_key_buffer`/`_get_value_buffer` return the RAW fp8 buffer (`3352-3356`) —
  no dequant view; `get_kv_scale_buffer(layer_id)` returns the scale tensors (`3358-3360`).
- Guards: `head_dim % 32 == 0` and `v_head_dim % 32 == 0` enforced (`3253-3262`);
  HND layout rejected (`3267-3272`); CPU offload / disagg / prefix-valid commit raise
  (`3486-3504`); `move_kv_cache` moves scales with payload (`3460-3484`) — the mamba
  `extra_buffer` radix strategy (mandatory for Inkling) depends on this and it is done.

**Pool selection for Inkling (hybrid SWA)** — `SRC/srt/mem_cache/kv_cache_configurator.py`:
`_build_token_to_kv_pool` takes the `is_hybrid_swa` branch (`847-852`) →
`_build_hybrid_swa_kv_pool` (`1184-1247`), which routes BOTH sub-pools (full + SWA ring)
through `swa_pool_class = MHATokenToKVPoolMXFP8` when `kv_cache_dtype == "mxfp8"`
(`1203-1207`), including the Inkling MTP draft's banded-depth SWA ring (`1210-1231`).
`SWAKVPool` (`SRC/srt/mem_cache/swa_memory_pool.py:19-75`) instantiates
`token_to_kv_pool_class` for both `swa_kv_pool` and `full_kv_pool`.

**Conclusion (mxfp8): the pool write path, storage layout, slot-move and eviction handling
are complete for the triton page_size=1 layout. What is missing is 100% read-side: (a) the
backend does not accept/route the kwargs, (b) `TritonAttnBackend._set_kv_buffer`
(`triton_backend.py:1196-1227`) passes `layer.k_scale` (a per-tensor scalar, None for
Inkling) instead of the per-token descale tensors — the MXFP8 pool would raise
`ValueError("MXFP8 KV cache requires K and V scale tensors.")` at `memory_pool.py:3389`
even before any kernel runs, and (c) the kernels dot raw fp8 against Q with no block scales.**

### 1.4 nvfp4 pool: NOT implemented for the triton/Inkling layout — say it explicitly

Two nvfp4 storage mechanisms exist in the image, **neither reachable for Inkling**:

1. `MHATokenToKVPoolFP4` — `memory_pool.py:2924-3076`. Packed payload `(m, H, D//2)` uint8
   (two e2m1 per byte), scales `(m, (H*D)//16)` uint8, block 16
   (`scale_block_size = 16`, line 2938). Write path done
   (`FP4MXBlock16KVQuantizeUtil.batched_quantize` in `set_kv_buffer`, `3044-3049`).
   BUT its read path (`_get_key_buffer`, `2980-2996`) calls
   `FP4MXBlock16KVQuantizeUtil.batched_dequantize` on the **entire pool buffer** per layer
   per access — materializing a full bf16 copy; at 1M-token pools this is
   capacity-defeating and unusable. It is only constructed by `_build_mha_fp4_kv_pool`
   (`kv_cache_configurator.py:1327-1343`), i.e. plain-MHA models — never for hybrid-SWA.
2. `quant_method`-based `MHATokenToKVPool` (`fp4_kv_cache_quant_method`, imported at
   `kv_cache_configurator.py:24-25`; buffers via `quant_method.create_buffers`,
   `memory_pool.py:1900-1922`; store via `quantize_and_store`, `2407-2438`; reads via
   `get_raw_kv_buffer` + FlashInfer fp8 **dequant workspaces**
   `_prepare_dequant_extend_workspace` / `_prepare_dequant_decode_workspace`,
   `2557-2644` — a python-loop, flashinfer-oriented design). Wired only in the non-SWA
   branches: plain MHA (`864-876`), MLA (`838-841`), and hybrid-LINEAR
   (`_build_hybrid_linear_kv_pool:1296-1305`).

**`_build_hybrid_swa_kv_pool` (`1184-1247`) has NO fp4 branch at all.** With
`--kv-cache-dtype nvfp4`, `configure_kv_cache_dtype`
(`SRC/srt/mem_cache/kv_cache_dtype.py:62-74`) resolves to `torch.float4_e2m1fn_x2`, and the
SWA sub-pools would be plain `MHATokenToKVPool` with `store_dtype = float4_e2m1fn_x2`
(`KVCache.__init__` remaps only fp8 dtypes to uint8, `memory_pool.py:1604-1609`) — no scale
buffers, and `set_kv_buffer`'s `cache_k.to(self.dtype)` cast (`2304-2310`) is not a
quantization. **This is why the fa4 path "works" upstream and triton has nothing to inherit:
fa4 targets are plain-MHA/MLA models whose pools go through branches 1/2 above; the
Inkling hybrid-SWA layout was never given fp4 storage.** Stage 3 must add it (§4.3).

---

## 2. WHICH kernels load K/V, and the exact dequant insertion sites

The triton backend binds three kernels (`triton_backend.py:126-148`): `decode_attention_fwd`,
`extend_attention_fwd`, `extend_attention_fwd_unified` (+ `verify_splitkv_fwd`, gfx95-only —
dead on GB10, `triton_backend.py:181-185`).

### 2a. Decode — `SRC/kernels/ops/attention/decode_attention.py`

Two stage-1 variants; Inkling is GQA (`kv_group_num > 1`) so the **grouped** kernel is the
hot one, but both should be patched (dispatch at `1001/1023` picks by `kv_group_num`).

| Site | file:line | what |
|---|---|---|
| `_fwd_kernel_stage1` K load | `decode_attention.py:201-205` (addr math 186-200) | `k = tl.load(K_Buffer + offs_buf_k, …)` → dequant here |
| `_fwd_kernel_stage1` V load | `decode_attention.py:244-248` (addr 231-243) | `v = tl.load(V_Buffer + offs_buf_v, …)` → dequant here |
| `_fwd_grouped_kernel_stage1` Q cast | `decode_attention.py:487` | `q_k = q.to(K_Buffer.dtype.element_ty)` — **the KV-dtype==compute-dtype assumption**; must NOT fire for mxfp8 (would dot raw fp8 with no block scales) |
| grouped K load | `decode_attention.py:510-514` (addr 500-509) | dequant before `tl.dot(q_k, k)` at 515 |
| grouped KPE load | `decode_attention.py:525-530` | MLA-only (`BLOCK_DPE>0` needs Lk∈{576,288}); not hit for Inkling D=128/SWA — guard, don't implement |
| grouped V load | `decode_attention.py:567-571` (addr 559-566) | dequant before `tl.dot(p.to(v.dtype), v)` at 577 |
| `p.to(v.dtype)` | `decode_attention.py:577` | after dequant v is fp32/bf16 — cast p explicitly to bf16 |
| per-tensor scale folding | `decode_attention.py:1012,1034` (`sm_scale * k_scale`) and stage-2 `* v_scale` at `803` | mxfp8/nvfp4 pass 1.0 here; block scales replace them |

Scale operand: block scale index along D is `offs_d // 32` (mxfp8). Flat scale gather offset:
`kv_loc * stride_sf_bs + cur_kv_head * stride_sf_h + (offs_d // 32)` — identical page math to
the payload (PAGE_SIZE==1 branch only; see §5 risk 2).

### 2b. Extend / verify — `SRC/kernels/ops/attention/extend_attention.py`

`_fwd_kernel` is BOTH the prefill kernel and the **DSpark verify kernel** (target_verify
runs `forward_extend` with `custom_mask`/`mask_indptr`; metadata built at
`triton_backend.py:804-855`, call at `1387-1412`). The kernel has two stages:

- **Stage 1 (prefix, reads the KV POOL — this is where dequant goes):**
  - K load: `extend_attention.py:429-433` (addr 413-428), consumed by
    `qk = tl.dot(q.to(k.dtype), k)` at **434** — same compute-dtype trap as decode.
  - KPE load: `449-454` — MLA-only, guard.
  - V load: `502-506` (addr 489-501), consumed at `508`; `p = p.to(v.dtype)` at 507.
  - Custom-mask / SWA / SKIP_TILE logic (`377-401`, `550-559`) is orthogonal: loads sit
    inside `if not SKIP_TILE:` and dequant simply joins them. **Patching stage 1 covers
    normal prefill AND DSpark verify AND draft-extend in one shot.**
- **Stage 2 (current chunk, reads `K_Extend`/`V_Extend` — the tensors passed as k/v):**
  loads at `568-570` and `625-627`. With the Stage-1 design below (bf16 k/v into the
  kernel, quantize only at store) these need NO change. If we instead accepted
  pre-quantized fp8 k/v + descale kwargs, both sites would need per-token dequant too —
  avoid that.

Per-tensor scale hooks already exist end-to-end (`k_scale`/`v_scale` args,
`extend_attention.py:678-679`, applied at `455` and `508`) — the new block-scale path
supersedes them (pass 1.0).

`_fwd_kernel_unified` (`853-1158`; K load `1057-1062`, V load `1132-1136`) is used only for
`--enable-deterministic-inference` (`triton_backend.py:1315-1326`) — out of campaign scope;
add a loud NotImplementedError guard.

Block-size note: `_get_block_sizes_for_extend_attention` already has the sm12x 99KB-smem
tier (`extend_attention.py:79-86`: Lq≤128 → BLOCK_M,BLOCK_N = 64,128).

### 2c. Backend plumbing sites (`triton_backend.py`)

- `forward_extend` signature `1229-1240`, store block `1261-1290`, kernel call `1387-1412`.
- `forward_decode` signature `1690-1701`, store block `1714-1739`, kernel call `1809-1831`.
- `_set_kv_buffer` `1196-1227` (routes scales into the pool).
- k/v==None re-read path `1250-1257` (`k = k_buffer[cache_loc]`) — reads raw fp8 without
  scales; must raise for quantized pools (see §5 risk 9).
- Buffer fetches `self.token_to_kv_pool.get_key_buffer/get_value_buffer` at
  `1365-1366, 1392-1393, 1450-1451, 1665-1666, 1784-1785, 1811-1812` — add a parallel
  `get_kv_scale_buffer(layer_id)` fetch (exists at `memory_pool.py:3358-3360`; needs a
  passthrough on `SWAKVPool` that routes layer_id → sub-pool like get_key_buffer does).

---

## 3. WHAT pr32333.diff provides

**Nothing reusable for KV cache.** `~/nvfp4-kv-triton/upstream/pr32333.diff` (77 lines, one
file: `srt/layers/quantization/fp8.py`) is entirely about **MoE routed-expert WEIGHTS**
(MXFP4-packed experts vs `--moe-runner-backend`): it reorganizes `get_quant_method` and adds
a fail-early `ValueError` when triton MoE runners meet packed-K//2 FP4 experts, plus guidance
to set `SGLANG_DSV4_FP4_DEQUANT=1`. There is no fp4 unpack helper, no scale application, no
attention/KV code. Its only transferable value is the *pattern*: fail early with an
actionable message instead of a cryptic downstream crash — worth copying as a boot-time
guard (reject `--kv-cache-dtype nvfp4` + triton until Stage 3 lands). The ROADMAP's real
donor ("fleet-internal MLA nvfp4-KV triton mod") is not in this extract; the e2m1 dequant
sequence must be written fresh (it is small — §4.3).

---

## 4. Staged implementation plan

### Stage 0 (prep, ~30 lines): accept kwargs + fail loudly
- `triton_backend.py:1229/1690`: add `q_descale=None, k_descale=None, v_descale=None` to
  both signatures. If any is not None and the pool is not a supported quantized pool,
  raise a clear NotImplementedError (pr32333-style message).
- This alone converts the TypeError into an actionable gate and unblocks incremental work.

### Stage 1 — mxfp8, decode only (est. ~150 new lines total, ~70 in triton)

**Design choice:** keep Q/K/V **bf16 through the kernels**; quantize only at the store;
dequant only pool reads. This (a) avoids Q-dequant entirely (no `q_descale` consumer needed),
(b) leaves extend stage-2 untouched for Stage 2, (c) is strictly more accurate than fa4's
fp8×fp8 QK dot.

1. **`inkling_common/attn.py`** (~10 lines): gate the mxfp8 pre-quant block
   (`ATTN:873-907`) on `fa4` — the triton lane keeps bf16 q/k/v and sends
   NO descale kwargs. (The fused prologue is already fa4-only, `ATTN:748-756`.)
2. **`triton_backend.py forward_decode`** (~40 lines):
   - Detect `isinstance(pool_for_layer, MHATokenToKVPoolMXFP8)` (via SWAKVPool routing).
   - Store: quantize k/v with `to_mxfp8` (same kernel the model uses), call
     `_set_kv_buffer(..., k_mxfp.data, v_mxfp.data, k_scale=k_sf, v_scale=v_sf)` — pool
     flat write path at `memory_pool.py:3414-3417/3433-3435` finishes the job.
     Assert `not pool.mxfp8_sf_interleaved` (page_size=1).
   - Read: fetch `k_sf_buf, v_sf_buf = pool.get_kv_scale_buffer(layer.layer_id)` and pass
     to `decode_attention_fwd` with `k_scale=1.0, v_scale=1.0`.
3. **`decode_attention.py`** (~70 lines): add `K_SF, V_SF` pointers + 2 strides each +
   `USE_KV_MXFP8: tl.constexpr = False` to `_fwd_kernel_stage1` and
   `_fwd_grouped_kernel_stage1`; at the K/V load sites (201/244, 510/567):
   ```
   sf = tl.load(K_SF + kv_loc[...]*stride_ksf_bs + cur_kv_head*stride_ksf_h + (offs_d//32)[...])
   k = (k.to(tl.float32) * tl.exp2(sf.to(tl.float32) - 127.0)).to(Q dtype)
   ```
   (e8m0 loaded as uint8 view; scale = 2^(e-127)). Skip the `q.to(K_Buffer.dtype)` cast at
   487 when USE_KV_MXFP8. `_decode_att_m_fwd`/`_decode_grouped_att_m_fwd` wrappers thread
   the tensors (view scales as uint8 before launch — triton has no e8m0 type).
4. **`SWAKVPool`** (~10 lines): `get_kv_scale_buffer(layer_id)` passthrough.

**Test gate 1:** (a) kernel unit test: random KV → `to_mxfp8` → pool → decode kernel vs
bf16-KV reference, rel-err within mxfp8 quantization noise (~2^-4 rms); (b) spec-off e2e
decode on the champion config, output sane (no bug-#13-style `!!!` garbage), logprob drift
vs bf16-KV bounded; (c) CUDA-graph capture+replay clean (scale-buffer pointers are
allocation-stable; nothing else changes). Decode tok/s ≥ bf16 parity expected (KV bytes
halve; GB10 is bandwidth-bound per the 273 GB/s ceiling).

### Stage 2 — mxfp8 extend + DSpark verify (est. ~90 new lines)

1. **`extend_attention.py _fwd_kernel`** (~50 lines): same `K_SF/V_SF/USE_KV_MXFP8`
   treatment at the two **prefix** load sites (429-433, 502-506); fix the `q.to(k.dtype)`
   dot at 434 and `p.to(v.dtype)` at 507 under the flag. Stage-2 chunk loads (568, 625)
   untouched (bf16 by design). Thread args through `extend_attention_fwd` (664-812).
2. **`triton_backend.py forward_extend`** (~40 lines): mirror the decode plumbing — store
   block at 1261-1290 quantizes-at-store (note: current code clones k/v for the
   scalar-scale path at 1286-1289; the mxfp8 branch replaces that), kernel call at
   1387-1412 gains the scale buffers; pass `k_descale=1.0, v_descale=1.0` (per-tensor slots).
3. Because verify IS `_fwd_kernel` stage 1 + custom mask, DSpark verify and
   DRAFT_EXTEND_V2 come for free; the multi-step draft decode
   (`TritonMultiStepDraftBackend`, 1835-2008) reuses `forward_decode` → already covered.

**Test gate 2:** (a) prefill needle @64K vs bf16 baseline; (b) DSpark spec-on: accept_len
within noise of 7.31 champion (accept uses the same quantized target KV — a real
regression detector); (c) `--disable-prefill-cuda-graph` boot path unchanged (bug #3);
(d) 384K long-run stability. This gate decides whether mxfp8 becomes the shipping
long-context tier.

### Stage 3 — nvfp4 (fp4_mx_block16) (est. ~250 new lines)

Pool storage for the Inkling layout **does not exist** (§1.4) — wiring first:

1. **`kv_cache_configurator.py _build_hybrid_swa_kv_pool`** (~15 lines): add an fp4 branch
   mirroring mxfp8's (`1203-1207`) that routes `token_to_kv_pool_class=MHATokenToKVPoolFP4`
   (write path already correct: packed `(m,H,D//2)` uint8 + `(m,(H*D)//16)` uint8 scales,
   `memory_pool.py:2925-2972/3016-3075`). Do NOT use its whole-pool-dequant read path.
2. **`triton_backend.py`** (~30 lines): for FP4 pools fetch raw packed buffers + scale
   buffers directly (`k_buffer[idx]`, `k_scale_buffer[idx]` — bypass
   `get_key_buffer`'s batched_dequantize), pass a `USE_KV_NVFP4` flag.
3. **Kernels** (~150 lines): the payload byte at packed offset `offs_d // 2` holds two
   e2m1 nibbles; per D-column: load byte, select nibble by `offs_d & 1`, decode e2m1 via
   bit math (sign `<<`… or an 8-entry `tl.where` LUT: {0,.5,1,1.5,2,3,4,6}), multiply by
   block scale at `(cur_kv_head*D + offs_d) // 16` from the flat scale row. Address math
   changes because the packed K stride differs from D — compute `offs_buf_k` on `D//2`
   granularity and expand. Confirm the scale byte encoding against
   `FP4MXBlock16KVQuantizeUtil` (`srt/layers/quantization/kvfp4_tensor.py`, in-image, not
   in extract): class name says MX-block16 — verify e8m0-vs-e4m3 before writing the decode.
4. Same store-side: `set_kv_buffer` already quantizes bf16 → packed+scales (`3034-3049`).
5. Boot guard from Stage 0 flips to allow nvfp4 + triton.

**Test gate 3:** unit dequant-parity vs `batched_dequantize`; e2e quality-parity harness +
long-context needle (ROADMAP acceptance gate); capacity check: ~0.56 B/elem vs bf16's 2
(≈3.5×) — the true-1M-in-flight unlock; decode perf ≥ mxfp8 (fewer bytes, more ALU).

---

## 5. RISKS

1. **SWA window KV buffers.** Sliding-window layers read via `window_kv_indices` already
   translated full→swa (`update_sliding_window_buffer`, `triton_backend.py:2011-2066`);
   scale rows share slot indexing with payload inside the SWA sub-pool, so the SAME
   translated indices index the scale buffer — correct by construction, but the
   `SWAKVPool.get_kv_scale_buffer` passthrough must route layer_id→sub-pool exactly like
   `get_key_buffer`, and the SWA sub-pool may have a DIFFERENT head_dim
   (`swa_head_dim != 128`, `ATTN`-side; `inkling.py:646`) → per-layer `sf_dim`; also
   MXFP8 pool asserts `head_dim % 32 == 0` (`memory_pool.py:3253-3262`) — verify
   `swa_head_dim` and `v_head_dim` divisibility from the model config before committing to
   mxfp8 (nvfp4 needs %16, plus %2 for packing).
2. **page_size=1 vs 128 scale layout.** All kernel work targets the FLAT scale layout
   (`page_size != 128`). page 128 stores scales interleaved in the FA4 atom layout
   (`memory_pool.py:3292-3310`) — unreadable by simple gather. Bug #12 already bans
   page 128 + triton; still, assert `not pool.mxfp8_sf_interleaved` at backend init so a
   future page-size experiment fails at boot, not with garbage.
3. **DSpark verify custom-mask kernel.** Verify correctness is accept-rate-critical: the
   draft and target must see identically-dequantized KV. Since verify shares `_fwd_kernel`
   stage 1 with prefill, one dequant implementation serves both — but the
   `SKIP_TILE` early-outs (`extend_attention.py:399-401, 557-559`) mean scale loads must
   stay INSIDE the `if not SKIP_TILE:` bodies or masked-out tiles read unwritten scale rows
   (slot 0 padding is zero-initialized e8m0 = 2^-127, harmless but keep loads masked with
   `other=127` to yield scale 1.0). Also bug #10's conv-commit interplay: quantized KV does
   not touch conv state, but re-run the byte-exact spec-off vs spec-on check anyway.
4. **sm_121a smem (99KB / 101376B).** Extend already tunes for sm12x
   (`extend_attention.py:79-86`). Dequant adds only the sf tiles
   (BLOCK_N × sf_dim bytes ≈ 128×4 = 512B) but converting K to bf16 pre-dot raises
   REGISTER pressure and may push `num_stages` pipelining over smem if triton stages the
   bf16 tile (num_stages=1 in extend — safe; decode uses num_stages=2 at
   `decode_attention.py:371,664` — watch `OutOfResources` à la bug #4; fallback: drop
   decode to num_stages=1 or BLOCK_N 64→32 under USE_KV_MXFP8).
5. **Code that assumes KV dtype == compute dtype** (the silent-corruption class):
   - `q_k = q.to(K_Buffer.dtype.element_ty)` `decode_attention.py:487`; `qk = tl.dot(q.to(k.dtype), k)` `extend_attention.py:434/1063`; `p.to(v.dtype)` `decode_attention.py:577`, `extend_attention.py:507/628/1137` — all must be bypassed under the quantized flags (this exact pattern is how fp8_e4m3 KV produced bug #13's `!!!` garbage: fp8 dot with only per-tensor scales on a hybrid).
   - `forward_extend` k/v==None re-read `triton_backend.py:1250-1257` returns raw fp8/packed bytes as if compute dtype — raise for quantized pools.
   - `_kv_buffer_shapes`/copy utilities size rows by `store_dtype.itemsize`; MXFP8 overrides its own ptr/stride tables (`3327-3344`) but note `data_ptrs` there EXCLUDES scale buffers, unlike the base `_slot_move_pointer_buffers` (`1958-1968`) — any future use of `copy_all_layer_kv_cache_func` on the MXFP8 pool would move payload without scales (today `enable_kv_cache_copy` warms only via `_init_kv_copy_and_warmup` on the base class; MXFP8 `move_kv_cache` override is correct).
   - FP4 `get_key_buffer` whole-pool dequant (`2980-3014`) must never be hit on the hot path (bypass in backend).
6. **CUDA graphs.** Scale buffers are allocated once (stable addresses) and all new kernel
   args are pointers/constexprs → capture-safe. The quantize-at-store `to_mxfp8` call must
   be graph-recordable (it is a triton kernel; no `.item()`), and runs inside the captured
   region exactly where `set_kv_buffer` runs today. Verify with decode-graph replay + the
   bug-#3 `--disable-prefill-cuda-graph` posture unchanged.
7. **Draft worker divergence.** The DSpark draft shares `kv_cache_dtype`; its banded SWA
   ring (`kv_cache_configurator.py:1210-1231`) gets the MXFP8 class too — good — but the
   draft's accept is brutally sensitive to KV noise (memory: draft-weights/ring lessons).
   If accept collapses at gate 2, first A/B: quantize target KV only (draft pool forced
   bf16 via a kv_cache_dtype override for draft workers, mirroring the DFLASH fa4
   override at `kv_cache_dtype.py:80-92`).
8. **`q_descale` on the triton lane.** After the attn.py gating, a non-None `q_descale`
   reaching triton means someone re-enabled Q pre-quant → assert None with a pointed
   message rather than silently attending with double-scaled Q.
9. **DCP / deterministic / MLA subpaths** (`_forward_extend_dcp:1415`,
   `_forward_extend_unified:1548`, KPE/MLA branches): not needed on this rig
   (dcp_size=1, deterministic off, no MLA) — guard each with NotImplementedError under
   quantized KV instead of plumbing them.

---

## Appendix: file inventory

| Role | Path |
|---|---|
| Crash site / backend to modify | `SRC/srt/layers/attention/triton_backend.py` (forward_extend 1229, forward_decode 1690, _set_kv_buffer 1196) |
| kwargs pass-through | `SRC/srt/layers/attention/base_attn_backend.py:188-231` |
| kwargs producer (in-image; local mirror) | `sglang/srt/models/inkling_common/attn.py` ≈ `~/inkling-ablit/patches/attn.py:872-948` |
| Decode kernels | `SRC/kernels/ops/attention/decode_attention.py` (K 201/510, V 244/567, q-cast 487) |
| Extend/verify kernel | `SRC/kernels/ops/attention/extend_attention.py` (prefix K 429, V 502, dot 434; wrapper 664; sm12x blocks 79-86) |
| MXFP8 pool (write path done) | `SRC/srt/mem_cache/memory_pool.py:3232-3517` |
| FP4 pool (write done, read unusable) | `SRC/srt/mem_cache/memory_pool.py:2924-3076` |
| fp4 quant_method plumbing (non-SWA only) | `SRC/srt/mem_cache/kv_cache_configurator.py:211-223, 863-876, 1296-1305` |
| Pool selection to extend for Stage 3 | `SRC/srt/mem_cache/kv_cache_configurator.py:1184-1247` (`_build_hybrid_swa_kv_pool`) |
| dtype resolution | `SRC/srt/mem_cache/kv_cache_dtype.py:53-74` |
| Non-donor | `~/nvfp4-kv-triton/upstream/pr32333.diff` (MoE weights guard only) |
