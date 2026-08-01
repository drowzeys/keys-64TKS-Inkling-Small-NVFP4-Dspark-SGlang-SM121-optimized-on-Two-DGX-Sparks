# Stage 3 — nvfp4 (`fp4_mx_block16`) KV cache for the SGLang TRITON backend (GB10 / sm_121a)

**Rev 1 (2026-08-01).** Extends the Stage-1/2 mxfp8 twin-kernel port to packed
e2m1 KV, fixes the mxfp8-active runtime crash (shared plumbing), and adds the
pool/sizing wiring that the Inkling hybrid-SWA layout never had.

---

## 0. Which "nvfp4" this is — verified against the image, not assumed

The image has **two** fp4 KV recipes
(`srt/layers/quantization/fp4_kv_cache_quant_method.py:777-797`):

| `--kv-cache-dtype` | recipe | block scale | needs | backends declared |
|---|---|---|---|---|
| `nvfp4` | `NVFP4KVCacheMethod` | fp8-**e4m3** + per-layer fp32 global scale from the checkpoint | `flashinfer.nvfp4_kv_quantize/dequantize` | prefill `flashinfer`, decode `trtllm_mha` |
| `fp4_mx_block16` | `FP4MXBlock16KVCacheMethod` | **uint8 biased exponent** (`ceil(log2(amax/6)) + 127`), block 16, no global scale | pure torch | **`triton`**, torch_native, flex_attention, trtllm_mha, (+fa4 prefill) |

So the brief's "block-16 fp8 scales" describes the `nvfp4` recipe; the recipe
that is (a) declared for triton, (b) self-contained (no checkpoint global
scales, no flashinfer), and (c) backed by a complete write path in
`MHATokenToKVPoolFP4` is **`fp4_mx_block16`**. That is what this patch
implements, and `--kv-cache-dtype nvfp4` is accepted as an alias for it on the
triton lane (`kv_cache_dtype.py:62-74` already resolves BOTH strings to
`torch.float4_e2m1fn_x2`). Both give identical capacity (0.5625 B/elem);
`fp4_mx_block16`'s power-of-two scale also cannot underflow, which e4m3 block
scales can below ~1.2e-2 block amax.

Encoding verified from source, then verified numerically (§4):
`FP4MXBlock16KVQuantizeUtil.batched_quantize` (`kvfp4_tensor.py:65-104`)
blocks a `(T, H, D)` tensor as `view(T, H*D//16, 16)` and packs
`packed = (v[..., 1::2] << 4) + v[..., 0::2]` — **even head_dim index in the
LOW nibble**, scale for `(h, d)` at column `h*(D//16) + d//16`, dequant
multiplier `2^(sf - 127)`, magnitudes `{0,.5,1,1.5,2,3,4,6}` with sign in bit 3.

---

## 1. mxfp8-active root cause and fix (the shared plumbing)

**The `mxfp8` boot does reach ready** — `~/mxfp8-active.log` shows
`The server is fired up and ready to roll!` at 00:39:33 (7m42s after launch;
~4.7 min of that is weight load + graph capture, so a short readiness timeout
would report it as "never ready"). It dies on the **first request**:

```
File ".../srt/speculative/dspark_components/dspark_kv_inject.py:71  inject_target_hidden
File ".../srt/models/dspark.py:658                                  write_target_hidden_kv
    pool.set_kv_buffer(attn.attn, cache_loc, k, v, attn.attn.k_scale, attn.attn.v_scale)
File ".../srt/mem_cache/memory_pool.py:3389                          set_kv_buffer
ValueError: MXFP8 KV cache requires K and V scale tensors.
```

**Root cause: there are THREE writers into the KV pool, and Stage 1/2 only
covered one.** `TritonAttnBackend.forward_extend`/`forward_decode` quantized at
store, but the DSpark hidden-state injector writes the draft's KV **directly
from the model file**, bypassing the attention backend entirely, with
`layer.k_scale = layer.v_scale = None`. `MHATokenToKVPoolMXFP8.set_kv_buffer`
(`memory_pool.py:3384-3389`) only tolerates missing scales when
`mxfp8_sf_interleaved` (i.e. `page_size == 128`, the fa4 fused quant-store
kernel) — at the triton lane's page_size=1 it raises. Two sibling writers were
broken the same way and had not been reached yet:

* `dspark.py:648 pool.set_kv_buffer_prefix_valid(...)` — the verify-commit
  path (`dspark_kv_inject.py:105-156 inject_ragged`). `MHATokenToKVPoolMXFP8`
  raises `NotImplementedError` there (`memory_pool.py:3500-3504`).
* `dspark.py:599 _fused_kv_write_bundle(pool)` — a fused norm+rope+write
  kernel that writes bf16 straight into `k_buf.data_ptr()`. It self-disables on
  non-bf16 buffers (`dspark.py:529-530`), so mxfp8/fp4 silently lose this fast
  path (a perf cost, not a correctness bug). **But it probes
  `pool.get_key_buffer(layer_id)` first (`dspark.py:525-526`)** — see §2.

**Fix: make the POOL quantize, so every writer is covered by construction.**
New `srt/mem_cache/kv_quant_pools.py`:

* `MHATokenToKVPoolMXFP8Triton.set_kv_buffer` (`kv_quant_pools.py:106-153`) —
  when handed bf16 K/V with no scale tensors, runs `to_mxfp8` and passes
  payload + UE8M0 scales to `super()`. Per-tensor float scales != 1.0 are
  rejected (block scales replace them); one-tensor-one-None is rejected.
* `_QuantizedPrefixValidMixin.set_kv_buffer_prefix_valid`
  (`kv_quant_pools.py:57-100`) — reuses upstream's validation, then routes
  through `set_kv_buffer` so payload AND scales are written. Row selection is
  an index remap (uncommitted rows -> pad slot 0), **not** a `nonzero()`
  gather: shape-static, no device sync, CUDA-graph safe.

The backend's own `_quantize_kv_mxfp8` is deleted; both store sites now just
hand bf16 to the pool (§3). One quantization path, three writers, both recipes.

---

## 2. Pool wiring (the part that did not exist for nvfp4)

`_build_hybrid_swa_kv_pool` had **no fp4 branch at all**
(`kv_cache_configurator.py:1203-1207` upstream), so `--kv-cache-dtype nvfp4`
produced a plain `MHATokenToKVPool` with `store_dtype = float4_e2m1fn_x2`,
whose `cache_k.to(self.dtype)` is a cast, not a quantization, and which has no
scale buffers. `MHATokenToKVPoolFP4` exists but is only constructed by
`_build_mha_fp4_kv_pool` (`:1327-1343`) — **dead code, no callers anywhere in
`srt/`** (verified by grep).

### `srt/mem_cache/kv_quant_pools.py` (NEW, 317 lines)

`MHATokenToKVPoolFP4Triton(MHATokenToKVPoolFP4)`:

* **`:253-257` `_get_key_buffer`/`_get_value_buffer` return the RAW packed
  uint8 buffer** — this is the "bypass the whole-pool dequant reader"
  requirement. The parent runs
  `FP4MXBlock16KVQuantizeUtil.batched_dequantize` over the ENTIRE per-layer
  buffer on every access (`memory_pool.py:2992/3010`). Bypassing it inside
  `_get_*_buffer` (rather than in the backend) is what also **disarms
  `dspark.py:_fused_kv_write_bundle`**: that function probes
  `pool.get_key_buffer(...)` and bails out on non-bf16 buffers, so with the
  parent's reader the probe ALONE would materialize a multi-GB bf16 copy of
  the draft pool at init. With the raw reader it sees `uint8` and returns
  `None` for free.
* **`:187-251` `_create_buffers`** — `k_buffer (m, H, D//2)`,
  `v_buffer (m, H, Dv//2)`, `k_scale_buffer (m, H*D//16)`,
  `v_scale_buffer (m, H*Dv//16)`, all uint8, zero-initialised. Two fixes vs
  the parent: **`v_head_dim` is honoured** (the parent sizes V payload and V
  scales with `head_dim`), and **`_init_data_ptrs_and_strides()` is called**
  (the parent skips it, so `move_kv_cache` would `AttributeError` — and that
  runs for real: the mamba `extra_buffer` radix strategy relocates KV rows
  during serving). The base `_slot_move_pointer_buffers`
  (`memory_pool.py:1958-1968`) already includes the scale buffers, so scale
  rows travel with their payload.
* **`:259-261` `get_kv_scale_buffer(layer_id)`** — mirrors the MXFP8
  accessor, so `SWAKVPool.get_kv_scale_buffer` (`swa_memory_pool.py:168-174`)
  and the backend's fetch work unchanged. Window (SWA) indices index scales
  exactly like the payload, inside the same sub-pool.
* **`:263-301` `set_kv_buffer`** — rejects `dcp_kv_mask` and any per-tensor
  k/v scale, then delegates. The parent's `cache_k.div_(k_scale)`
  (`memory_pool.py:3035-3038`) is an **in-place mutation of the caller's
  tensor**; for Inkling `layer.k_scale is None` so it never fired, but a
  future non-None scale would have corrupted `K_Extend` mid-kernel.
* **`:303-316`** CPU offload / disagg raise instead of moving payload without
  scales.

### `srt/mem_cache/kv_cache_configurator.py`

* **`:44-47`** import the two subclasses.
* **`:215-262` `_triton_quant_pool_class()`** (NEW) — returns
  `MHATokenToKVPoolMXFP8Triton` / `MHATokenToKVPoolFP4Triton` **only** when
  all three of `attention_backend` / `prefill_attention_backend` /
  `decode_attention_backend` resolve to `triton` AND the dtype string is
  quantized; `None` otherwise. Boot-time `NotImplementedError` for fp4 +
  MLA, fp4 + `page_size != 1`, fp4 + post-capture KV backing.
* **`:1256-1262` `_build_hybrid_swa_kv_pool`** — the new fp4 branch (both the
  full and SWA sub-pools, and the Inkling MTP draft's banded ring, since
  `SWAKVPool` instantiates one class for both).
* **`:1405-1412` `_build_mha_kv_pool`** — the **DSpark draft's** pool comes
  through here (log: one bare `KV Cache is allocated` after the SWAKVPool
  block, no second `SWAKVPool mem usage`). It must get the SAME storage class
  as the target or the backend would see two layouts; upstream would instead
  have given it a `quant_method`-based pool with a DIFFERENT scale shape
  (`(m, H, D//16)` 3-D and flashinfer dequant workspaces), so `quant_method`
  is forced to `None` on this lane.

### `srt/model_executor/pool_configurator.py`

* **`:371-384` `HybridSWAPoolConfigurator.__init__`** — new fp4 branch beside
  the mxfp8 one. `torch._utils._element_size(torch.float4_e2m1fn_x2)` is **1**
  (measured), so without this the budget counts 1 B/elem instead of
  0.5 + 1/16 = 0.5625 and **over-counts fp4 by 1.78x** — capacity would land
  at roughly the mxfp8 multiplier instead of 3.556x.

---

## 3. Backend (`srt/layers/attention/triton_backend.py`, 2320 lines, src 2066)

* **`:31-35`** import `MHATokenToKVPoolFP4` alongside `MHATokenToKVPoolMXFP8`.
* **`:284-374`** detection generalised: `use_kv_mxfp8`, `use_kv_nvfp4`,
  `use_kv_quant = mxfp8 or nvfp4`; mixed sub-pool classes, mixed recipes,
  `mxfp8_sf_interleaved` (fa4 page-128), `page_size != 1`, MLA, DCP and
  deterministic mode all raise at boot. **`:320-329`** additionally requires
  the fp4 pool to expose `get_kv_scale_buffer`, i.e. rejects a stock
  `MHATokenToKVPoolFP4` whose reader would dequantize the whole pool.
* **`:376-386`** boot log now prints both flags and the sub-pool class names:
  `TritonAttnBackend init: use_kv_mxfp8=… use_kv_nvfp4=… page_size=…
  is_draft_worker=… pools=[…]`.
* **`:1388-1393`** k/v == None pool re-read raises under any quantized pool.
* **`:1411-1423` (extend) / `:1927-1945` (decode)** store: **the backend no
  longer quantizes** — it hands bf16 k/v to `_set_kv_buffer` and the pool does
  the work (§1). `_quantize_kv_mxfp8` removed.
* **`:1515`** gfx95 `verify_splitkv` fast path gated on `not use_kv_quant`.
* **`:1545-1580` / `:2029-2060`** the calls: fetch
  `get_kv_scale_buffer(layer_id)` and call
  `extend_attention_fwd_kv_quant` / `decode_attention_fwd_kv_quant` with
  `kv_fp4=self.use_kv_nvfp4`; otherwise fall through to the **byte-identical**
  upstream call.
* `srt/models/inkling_common/attn.py` is **unchanged from Rev 3** — its
  pre-quant block tests `== "mxfp8"` only and is `and fa4` gated, so the nvfp4
  lane produces no descale kwargs at all.

### Inertness (re-checked)

Under `--kv-cache-dtype auto`: `_triton_quant_pool_class()` returns `None`
(dtype string is `auto`), no quantized pool is constructed, `use_kv_mxfp8 =
use_kv_nvfp4 = use_kv_quant = False`, `kv_quant_attention` and
`kv_quant_pools`' pool bodies are never executed, and both kernel calls are
the verbatim upstream ones. `decode_attention.py` / `extend_attention.py` are
**still not shipped** — the champion's bytes (and therefore triton cache keys,
which hash source line numbers) are untouched.

---

## 4. Kernels (`kernels/ops/attention/kv_quant_attention.py`, 1613 lines)

Renamed from `mxfp8_kv_attention.py` (which is deleted from the patched tree);
one set of clones now serves both recipes via constexprs, instead of a second
1400-line copy.

* **`:53-54`** `SF_BLOCK_MXFP8 = 32`, `SF_BLOCK_FP4 = 16`.
* **`:57-136` `_prepare_kv_sf_args(..., kv_fp4, head_dim, v_head_dim)`** —
  reshapes the fp4 pool's flat `(m, H*D//16)` uint8 scale row to
  `(m, H, D//16)`. That reshape is **exact**, not an approximation: see §0.
  page_size != 1, ndim != 2/3, one-None and non-multiple rows raise.
* **`:144-168` `_dequant_kv_block(payload, sf_u8, parity, KV_FP4)`** (NEW,
  `@triton.jit`) — `scale = exp2(sf - 127)`; mxfp8 is a plain f32 cast, nvfp4
  selects the nibble by `parity` and decodes e2m1 arithmetically:
  `m = nib & 7`; `val = (1 + .5*(m&1)) * 2^((m>>1)-1)` for `m >= 2`,
  `0.5*m` for the subnormals `m < 2`; sign from bit 3. No LUT, no shared
  memory, ~6 ALU ops per element.
* **Per-kernel `kcol`/`vcol`/`kpar`/`vpar`** (`:248-262`, `:511-525`,
  `:840-854`): under `KV_FP4` the payload column is `offs_d // 2` and the
  parity is `offs_d % 2`; otherwise both degrade to `offs_d`, so the mxfp8/
  bf16 address math is textually unchanged. The payload *mask* is unchanged
  (`offs_d < Lk`), so padded lanes never touch memory.
* Six pool-load sites (K and V in `_fwd_kernel_stage1_kv_quant`,
  `_fwd_grouped_kernel_stage1_kv_quant`, `_fwd_kernel_kv_quant`) now call the
  helper with `SF_BLOCK` instead of a hard-coded 32. Masked lanes load
  `other=0` payload and `other=127` scale -> 0 * 1.0.
* **Extend stage 2 (`K_Extend`/`V_Extend`) and ALL custom-mask / SWA /
  SKIP_TILE logic remain byte-identical to upstream** — DSpark verify masking
  undisturbed (hard requirement).
* **`:1215-1441` `decode_attention_fwd_kv_quant`** — `Lk`/`Lv` are now derived
  as `k_buffer.shape[-1] * 2` under `kv_fp4` (the packed row is `D//2`), and
  the stage-2 reduce is passed a **zero-element shape proxy**
  (`:1424-1429`): `_decode_softmax_reducev_fwd` reads only
  `v_buffer.shape[-1]` to derive `Lv`/`BLOCK_DV`, so handing it the packed
  buffer would have reduced and stored **only half of every head**. *This was
  a real bug caught by the GPU test* (§5) — the first nvfp4 decode run
  produced `3.4e38`.
* **`:1443-1613` `extend_attention_fwd_kv_quant`** — `Lk` comes from the bf16
  `k_extend` (already logical); asserts `k_buffer.shape[-1]*2 == Lk`.
* No new shared-memory tiles; the scale reads are small extra **global**
  loads, and fp4 halves the payload traffic. sm_121a's 99 KB budget is
  untouched (still watch for `OutOfResources` on first JIT).

---

## 5. GPU numerical validation (real hardware, head node spark-13b3, idle GB10)

`~/nvfp4-kv-triton/tests_verify_nvfp4.py`, torch 2.11.0+cu130 / triton 3.6.0 /
sm_121a — the same versions as the image. It stubs the `sglang` namespace and
loads the image's **pristine** `decode_attention.py` / `extend_attention.py`
next to the patched module, so every comparison is against the real upstream
kernel fed **host-dequantized bf16 KV**. All 20 checks pass:

| check | result |
|---|---|
| e2m1 dequant vs `FP4MXBlock16KVQuantizeUtil.batched_dequantize`, 512x128 over 28 exponent decades, all 16 codes exercised | **bitwise exact** |
| mxfp8 dequant unchanged by the refactor | **bitwise exact** |
| nvfp4 extend, GQA-4 D=128, DSpark verify custom-mask | **bitwise equal** |
| nvfp4 extend, GQA-4 D=128, causal prefill | **bitwise equal** |
| nvfp4 decode, grouped GQA-4 D=128 | **bitwise equal** |
| mxfp8 extend (both masks) / decode grouped — regression check | equal within bf16 ULP (1.2e-4 / 1.5e-5) |
| nvfp4 shape sweep: (HQ,HKV,D) = (32,4,128) (32,8,64) (8,1,80) (64,4,128) | max 2.0e-3 vs a 1.6e-2 bf16-ULP bound; three of four **exactly 0** |
| nvfp4 relative RMS quantization error, 4096x4x128 unit-normal | 0.113 (e2m1 block-16 expectation) |
| guards: page_size=128 rejected, one-None rejected, both-None inert, flat->3-D scale reshape | pass |

`D=80` covers the padded case (`BLOCK_DMODEL = 128 > Lk = 80`) — masked lanes
address only `offs_d // 2 < 40`, i.e. inside the packed row.

**Documented non-gate — MHA decode (`kv_group_num == 1`).** At unit-scale KV
with D=128 (|qk| ~ 70) the *upstream* MHA stage-1 kernel accumulates
`tl.sum(q[None,:] * k, 1)` **elementwise in bf16**, so it is bf16-unstable and
*any* reassociation moves the result by O(0.5) — including a provably lossless
dequant. Isolated: with KV drawn exactly from the e2m1 grid (round-trip
verified lossless) and |qk| brought into range, the clone matches the pristine
kernel to **2.4e-4**. Unreachable for this deployment (target 16q/4kv per rank,
draft 4kv -> always the grouped kernel). Not a regression: the same property
was recorded in STAGE12_NOTES §4.

---

## 6. Expected pool multiplier (Inkling-Small, TP=2, CTX=65536, MEMFRAC=0.85)

Geometry from `/mnt/models-7552/inkling/inkling-small-nvfp4/config.json` +
the boot logs: `num_key_value_heads=8` (4/rank at TP=2),
`head_dim = v_head_dim = swa_head_dim = 128`, 42 layers = **7 full-attention +
35 SWA**, `swa_full_tokens_ratio = 0.1`, `page_size = 1`.

Per token, per layer, K+V:

| dtype | payload | scales | total | cell_size = t*(7 + 0.1*35) |
|---|---|---|---|---|
| bf16 | 4*(128+128)*2 = 2048 B | — | 2048 B | 21,504 B |
| mxfp8 | 1024 B | 1024/32 = 32 B | 1056 B | 11,088 B |
| **fp4_mx_block16** | **512 B** | **1024/16 = 64 B** | **576 B** | **6,048 B** |

* **mxfp8 vs bf16: 1.939x** — measured **1.938x** (bf16 7.09 GiB / 354,077 =
  21,507 B/tok; mxfp8 5.76 GiB / 557,413 = 11,097 B/tok). The sizing math is
  already correct.
* **nvfp4 vs bf16: 3.5556x** bytes/token; **1.834x** vs mxfp8.

### Why the mxfp8 boot showed only 1.574x more TOKENS

The bf16 champion at the identical config (`~/inkling-serve.log`, kv=auto) is
**354,077 tokens**, not 674,816 — that reference belongs to a different config.
557,413 / 354,077 = 1.574x. The per-token cost fell the full 1.94x; the token
count did not, because that boot simply had **~1.9 GB less memory to divide**:
target weights `85.03` vs `84.63 GB`, draft weights `1.19` vs `0.79 GB`,
`avail mem` after load `24.43` vs `25.50 GB`. Total pool bytes actually
allocated: bf16 12.97 GB vs mxfp8 10.86 GB. **Nothing is double-allocated and
no bf16 staging buffer exists** — `MHATokenToKVPoolMXFP8` allocates only fp8
payload + e8m0 scales (`memory_pool.py:3274-3325`), which the logged
`K size: 1.92 GB` at 557,413 tokens confirms exactly (3,699 B/tok = 3,584
payload + 115 scale over 7 layers).

**Expected fp4 result:** at the bf16 boot's budget, `max_total_num_tokens` ≈
`354,077 * 3.556` ≈ **1.24-1.26 M** (SWA ring 10% of that), with the same
±1-2 GB boot-to-boot variance moving it by roughly ∓5%. That clears 1M
in-flight context on the 2-node pair with margin.

---

## 7. Change inventory

```
patched/kernels/ops/attention/kv_quant_attention.py   1613 lines (NEW; replaces
                                                      mxfp8_kv_attention.py,
                                                      which is DELETED)
patched/srt/mem_cache/kv_quant_pools.py                317 lines (NEW)
patched/srt/mem_cache/kv_cache_configurator.py        +76 lines vs image
patched/srt/model_executor/pool_configurator.py       +13 lines vs image
patched/srt/layers/attention/triton_backend.py        2320 lines (src 2066)
patched/srt/models/inkling_common/attn.py              991 lines (UNCHANGED
                                                      from Rev 3)
(decode_attention.py / extend_attention.py: STILL NOT SHIPPED — champion bytes,
 mtimes and triton cache keys preserved.)
```

New file `kv_quant_pools.py` lands at
`/sgl-workspace/sglang/python/sglang/srt/mem_cache/kv_quant_pools.py`;
`kv_quant_attention.py` at
`/sgl-workspace/sglang/python/sglang/kernels/ops/attention/kv_quant_attention.py`.

## 8. Re-test protocol

1. **Bake**, then confirm in-image `md5sum decode_attention.py
   extend_attention.py` == champion (`a9f2d70d…`, `94fc254f…`).
2. **kv=auto inertness gate** (unchanged bar): grep
   `TritonAttnBackend init: use_kv_mxfp8=False use_kv_nvfp4=False page_size=1`
   on target AND draft; accept must match champion 3.44±0.17 / 3.70±0.23.
3. **kv=mxfp8**: must now serve. Watch the first request — it is the one that
   previously raised at `dspark.py:658`.
4. **kv=fp4_mx_block16** (or `nvfp4`): expect
   `use_kv_nvfp4=True pools=['MHATokenToKVPoolFP4Triton', …]` on both workers,
   `max_total_num_tokens` ≈ 1.24 M, and `KV Cache is allocated. dtype:
   torch.uint8`. Allow ~8 min to ready (weight load 4.7 min + JIT).
