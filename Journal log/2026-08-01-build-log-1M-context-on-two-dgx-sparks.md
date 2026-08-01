# We ran a 276B model at 1M context on two desktop DGX Sparks — here's every wall we hit

Two GB10 machines. 128 GB unified memory each, ~273 GB/s of bandwidth — roughly 1/29th the aggregate bandwidth of the 8×B200 rig the official recipes assume.

The goal: serve **Inkling-Small-NVFP4** (276B total / 12B active MoE, from Thinking Machines) with **RadixArk's DSpark speculator**, on SGLang, at the longest context we could reach.

Final result: **1,048,576-token context, 1,082,627-token KV pool, byte-exact quality, ~33 tok/s.** Along the way we shipped the first NVFP4 KV cache for SGLang's triton backend, found two upstream-worthy bugs, and lost most of a day to a measurement trap that had nothing to do with the model.

Everything below is the honest log — including the parts where we were wrong.

---

## What worked immediately

Almost nothing. That's the point of writing this down.

LMSYS had pushed an `inkling-dspark` dev image the same day, so the engine existed. Weights downloaded fine. Then the first boot failed, and so did the next thirteen.

---

## Wall 1 — NCCL: `invalid usage`, no other output

Two nodes, 200G RoCE link, and NCCL refused to initialize with a four-word error.

**Symptom:** `RuntimeError: NCCL error: invalid usage`, zero diagnostic output.
**Cause:** `NCCL_DEBUG=INFO` revealed `NET/IB : No device found` — the container couldn't see the RDMA devices at all.
**Fix:** `--device /dev/infiniband --cap-add IPC_LOCK`. Verified with a two-rank allreduce before touching the real serve.

**Lesson:** when a distributed init fails silently, reproduce it in the smallest possible program first. A 12-line allreduce found in minutes what a 156 GB model boot would have obscured for hours.

---

## Wall 2 — 15 GB vanishing at startup

The bundled NCCL 2.28.9 consumed ~15 GiB at init on GB10's unified memory. NCCL 2.30 uses ~3.5. On a machine where the model needs 156 GB of 250, that's not a rounding error.

**Fix:** `pip install -U nvidia-nccl-cu13` baked into the image.

---

## Wall 3 — kernels written for a bigger GPU

```
triton OutOfResources: shared memory
Required: 110592, Hardware limit: 101376
```

Inkling's MoE grouped-GEMM was tuned for B200's 228 KB of shared memory. GB10 has 99 KB.

**Fix:** pipeline stages 3/4 → 2. Ten characters. Two hours to find.

Then the next request died on missing Helion autotune configs for `sm_121` — the file simply didn't exist. We seeded them from the `sm_100` copies.

---

## Wall 4 — the MoE kernel that lies

FlashInfer's TRT-LLM routed FP4 MoE ships `sm_100`-only cubins: hard failure, easy to diagnose.

So we switched to the cutlass runner. It ran. It produced *fluent, confident, wrong* text at temperature 0.

**This is the most dangerous class of bug on new hardware.** A crash tells you something is broken. Silently incorrect numerics tell you nothing, and you will happily benchmark garbage.

**Fix:** `--moe-runner-backend marlin` — the only numerically-correct NVFP4 MoE runner we found on sm_121. We now gate every config change on a byte-exact reference probe before recording any number.

---

## Wall 5 — the speculator that drafted nothing (upstream bug)

DSpark ran, but acceptance sat near 1.0 — no speedup at all.

**Cause:** the draft worker inherits the *target's* `speculative_num_draft_tokens` (gamma+1) while the proposer emits gamma rows. The attention backend strides by 8 over 7-row tensors — out-of-bounds KV reads, garbage hidden states. This is [sglang#30555](https://github.com/sgl-project/sglang/issues/30555).

**Subtlety:** the issue's own suggested fix (pin the draft's ServerArgs) *double-corrects* on current builds, because they already subtract 1 in graph sizing. You get a capture-time shape error instead.

**Fix:** subtract 1 in the triton backend's draft-worker width, and nowhere else. That single `-1` fixed both the OOB reads and CUDA-graph capture.

---

## Wall 6 — the off-by-one that only bites outside the datacenter (novel)

With speculation on, output degenerated into prompt-replay whenever acceptance exceeded 1. Spec off: perfect prose. The target was provably fine — byte-exact on reference probes.

**Cause:** DSpark passes `commit_lens` *excluding* the bonus token, but the short-conv state commit used it directly as a last-accepted-step index. Every verify round committed the convolution window one token short, and the state regressed cumulatively.

**Why nobody upstream sees it:** every B200-class recipe runs `--enable-torch-symm-mem`, whose fused kernel takes a different path. GB10 can't run symm-mem cross-node. So this bug is invisible on the hardware the developers use, and fatal on everything else.

**Fix:** commit the window at `step+1`, clamped. Verified byte-exact against non-speculative decoding.

---

## The day we lost to a phantom

Here's the part worth reading even if you never touch this stack.

We measured accept 7.31 and 64.6 tok/s. Published it. Then hours later, the same config measured 2.4. Same image hash. Same weights (sha256-verified against upstream). Same flags, diffed field by field.

We chased it: rebooted both nodes, bisected the kernel patches, blamed the GPU driver (there *was* NVRM OOM spam in dmesg), suspected a competing session, and burned a second agent's entire context proving the kernels were PTX-instruction-identical.

**The truth:** the target forward pass is nondeterministic at temp 0 on this stack — triton split-KV reduction order and marlin MoE reductions, neither covered by `--enable-deterministic-inference`. Greedy decoding flips on near-tie tokens and the output diverges. And crucially, **acceptance depends on what the model happens to generate**: repetitive text drafts trivially (accept 4–5), novel prose doesn't (accept 1.5–2.5).

So a single-prompt, 10-run benchmark has a noise band wider than most of the effects we were measuring. Our "7.31 champion" was a lucky draw. There was never a regression.

Then the closing blow: **RadixArk's own model card publishes `acc_len` mean 3.348** across 9 datasets at temp 0, block 7 — our exact config. Our 3.44 ± 0.17 *was the published number all along.* We had spent a day hunting a value the draft never produces.

**What we do now:** every comparison uses 4 distinct prose seeds × 8 reps = 32 samples, reported as mean ± standard error. If the error bars overlap, it's not a finding. That harness ships in the repo, and it's arguably the most valuable file in it.

---

## The actual win: NVFP4 KV cache

bf16 KV capped the pool near 354K tokens. `fp8_e4m3` KV produced pure garbage (`!!!!!...`). So 1M context looked impossible on two nodes.

The blocker was structural: **SGLang's triton backend had no KV quantization at all.** No descale plumbing, no dequant in the attention kernels. And triton is the *only* legal attention lane for this model on GB10 (it asserts `fa4|triton`, and fa4 is sm_100-only).

So we built it. What it took:

**Quantize inside the pool, not the backend.** Inkling has *three* KV writers — and DSpark's hidden-state injector writes KV straight from the model file, bypassing the attention backend entirely. Backend-side quantization boots fine and dies on the first request with `requires K and V scale tensors`.

**An fp4 branch for the hybrid-SWA pool**, which upstream never wrote — plus bypassing the stock FP4 pool's reader, which dequantizes the *entire pool* per access.

**e2m1 nibble decode with block-16 scales**, in *cloned* kernels — upstream's `decode_attention.py` and `extend_attention.py` are left byte-untouched so triton cache keys and compiled binaries match the known-good build exactly. All quantized code lives in a module that is only imported when quantization is on.

**Correct fp4 byte accounting** — `_element_size(float4_e2m1fn_x2)` is 1, so fp4 gets over-counted 1.78× unless the pool configurator special-cases it.

One bug caught by the test suite before it ever reached a serve: the decode reduction derives its head dimension from `v_buffer.shape[-1]`, which is D/2 for packed fp4 — it was reducing half of every attention head.

**And a naming trap:** `--kv-cache-dtype nvfp4` selects the flashinfer/trtllm recipe that the triton lane cannot consume. The triton-compatible packing is `fp4_mx_block16`. Identical capacity, completely different plumbing.

---

## Results

| | bf16 KV | **NVFP4 KV** |
|---|---|---|
| KV pool @ 64K ctx | 354,077 | **1,104,683** (3.12×) |
| accept (n=32) | 3.44 ± 0.17 | 3.54 ± 0.16 |
| tok/s (n=32) | 34.3 ± 1.7 | 32.9 ± 1.5 |
| quality | reference | **byte-exact match** |

Statistically identical speed. 3.12× the capacity. Which makes this possible:

```
context_len          = 1,048,576
max_total_num_tokens = 1,082,627   ← pool exceeds context
```

Needle-in-a-haystack retrieval verified at 21K, 64K, and 113K token depths. Speculative decoding is byte-exact against non-speculative decoding at temp 0.

---

## Things that did NOT work (and shouldn't be retried)

**Native MTP (the official EAGLE 8-1-9 recipe).** Inkling ships 8 trained MTP heads — a separate 4.2 GB file. On 2×GB10 that's a fixed −6 GB deficit regardless of batch size or context, *and* at reduced width it dies with `KeyError: 0` in `layers_mapping` because the EAGLE path isn't wired for this model's hybrid-SWA layer map. Two independent blockers.

**Confidence-scheduled speculation (`cap-accept`).** Real feature, runs clean, and measures *worse* than a fixed block size — because it needs a cost table to budget proposal width, and building that table requires a recorder that only runs once the table exists. Circular, in this build.

**A4Q native-fp4 attention.** Excellent on other models here (−22% to −39% TTFT). Not applicable: it needs KV width ≥4096 to amortize, and this model has 1024 (8 kv-heads × 128). It's also a FlashInfer kernel, and this model can't use FlashInfer attention. Rejected before writing code — the cheapest good decision of the project.

**`--enable-deterministic-inference`.** Doesn't cover the kernels that are actually nondeterministic here. Costs speed, fixes nothing.

---

## The five lessons

1. **Silently wrong beats loudly broken, in the bad way.** Gate every config on a byte-exact reference probe before you record a number.
2. **Measure with error bars or don't measure.** On nondeterministic stacks, single runs will invent findings that don't exist — and you'll chase them for a day.
3. **Check the spec sheet before optimizing.** The number we hunted for a day was published on the model card the whole time.
4. **Read the code before believing the issue tracker.** The upstream fix for #30555 now double-corrects; applying it verbatim creates a new bug.
5. **Bugs hide where the developers' hardware differs from yours.** The conv-commit off-by-one is invisible on B200 and fatal everywhere else. If you're off the beaten path, expect to find things nobody has seen.

---

Everything — patched files, the digest-pinned bake script, the exact launch invocation, the 32-sample benchmark harness, 20 bitwise kernel tests, and all 22 walls with symptoms and fixes — is public:

**github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-SGlang-SM121-optimized-on-Two-DGX-Sparks**

Two desktop machines. A 276B model. A million tokens of context. No datacenter required.
