# DSpark Draft Finetune Campaign — Inkling-Small-NVFP4

**Written:** 2026-07-31/08-01 (session on .4 / spark-13b3)
**Goal:** raise speculative acceptance above the stock draft's **3.44 ± 0.17** (of 8),
tok/s **34.3 ± 1.7**, by finetuning `RadixArk/Inkling-Small-DSpark-Preview` onto our
target's output distribution.

**VERDICT: CONDITIONAL GO.** The trainer is feasible and mostly already written. The
campaign is gated on **one production-serve restart** (to enable aux-hidden capture) and
**a 2-node window** for the capture engine. Honest expected gain is **+15-30% accept
(3.44 → 4.0-4.5)**, not +45%. See §5.

---

## 0. THE SINGLE MOST IMPORTANT FINDING — READ THIS FIRST

The draft's own model card publishes its acceptance:

| Dataset | acc_len | | Dataset | acc_len |
|:--|--:|---|:--|--:|
| GSM8K | 4.787 | | LiveCodeBench | 2.929 |
| MATH500 | 4.143 | | AIME25 | 2.894 |
| MBPP | 3.439 | | Alpaca | 2.782 |
| HumanEval | 3.349 | | Arena-Hard-v2 | 2.698 |
| MT-Bench | 3.114 | | **Mean** | **3.348** |

*(temp 0, 128 prompts/task, block size 7, thinking effort 0.99 — i.e. our exact serving config.)*

**Our measured 3.44 ± 0.17 is the published number (3.348), within one standard error.**

Consequences, all load-bearing:

1. **The draft is not broken and is not mis-served.** Everything upstream of this campaign
   (PROGRESS.md's collapse hunt, the reboot, the determinism work, cap-accept/SPS) was
   chasing a number the draft was never going to produce. Stop looking for a config lever.
2. **There is no "restore the champion" upside.** The documented 64.6 tok/s @ accept 7.31
   is not a steady state of this draft; PROGRESS.md already reclassified it as peak-of-variance.
3. **The only remaining lever is the draft weights themselves** — exactly this campaign.
4. **The realistic ceiling is set by the task spread, not the mean.** The authors' own
   per-task range is 2.70 (Arena-Hard) → 4.79 (GSM8K), a 1.8× spread on a *fixed* draft.
   Acceptance is dominated by how predictable the target's output is, and a general
   finetune cannot move all nine tasks to the top of that range. Reaching a *general* 5+
   would mean beating the authors' full-scale training run. Reaching ~4.5 **on our narrow
   serving distribution** is a credible, defensible target.

---

## 1. TRAINER

### 1a. SpecForge — feasibility

RadixArk's card is accurate: this draft was built with SpecForge, and **DSpark is merged on
`main`**, not a PR: `specforge/modeling/draft/dspark.py` (`DSparkDraftModel`,
`VanillaMarkovHead`, `GatedMarkovHead`, `RNNMarkovHead`, `AcceptRatePredictor`) on top of
`specforge/modeling/draft/dflash.py` (`DFlashDraftModel`). Upstream even ships
`examples/configs/inkling-dspark-disaggregated.yaml` + `configs/inkling-dspark.json`
— though for a **66-layer** Inkling variant (`target_layer_ids [5,17,35,47,59]`), not our
42-layer Small (`[5,11,23,29,35]`). `target_layer_ids` is a plain config field, so that is
an edit, not a port.

**(a) DSpark/dflash architectures: YES, first-class.**

**(b) aarch64 / GB10: LIKELY OK, untested upstream.** SpecForge is *pure Python* — PyPI ships
`specforge-0.1.0.tar.gz` with no wheel and no compiled extensions of its own, so there is no
arm64 wheel to be missing. `flash-attn` is an optional extra (`[fa]`) only; the shipped
configs use `attention_backend: flex_attention`, which is pure PyTorch and is the correct
choice for us anyway (the local `dflash.py` header already documents that the FA4 flex
backend fails on DSpark's captured-tensor mask_mod). Docs mention CUDA / ROCm / Ascend and
**never mention aarch64, Blackwell, or sm_120/121** — assume untested.

**The real install risk is the hard pins, and they conflict with our stack:**

| SpecForge pin | Our stack | verdict |
|---|---|---|
| `torch==2.11.0` | `2.11.0+cu130` on .4 native **and** in `local/sglang-inkling:gb10` | ✅ match |
| `transformers==5.8.1` | `5.12.1` everywhere | ⚠️ needs `--no-deps` or a pin override |
| `sglang==0.5.14` | `0.0.0.dev1+gb7252cc6b` (our custom GB10 build) | ❌ **hard conflict** |
| `mooncake-transfer-engine` (disaggregated only) | no aarch64 build verified | ❌ unverified |

**(c) online vs offline:** three topologies, and the choice is forced for us.

- **`disaggregated` (= the "online, live SGLang engine" mode the card describes).** Trainer
  never runs the target; a **patched** SGLang server publishes captured features over
  **Mooncake** (`scripts/apply_sglang_spec_capture_patch.sh` applies
  `patches/sglang/v0.5.14/spec-capture.patch` into site-packages, adding
  `--enable-spec-capture`, `--spec-capture-aux-layer-ids`, `--spec-capture-method`, and a
  `spec_capture` request field). **RULED OUT HERE:** the patch is version-locked to the
  SGLang v0.5.14 source tree and our image is a custom GB10 dev build — the patch will not
  apply, and floating SGLang forward would forfeit every GB10 fix baked into our image.
  Plus an unverified aarch64 Mooncake.
- **`local_colocated` / offline.** `scripts/prepare_hidden_states.py` builds an **in-process**
  SGLang runner (`specforge/offline_capture/sglang_backend/model_runner.py` subclasses
  `sglang.srt.model_executor.model_runner.ModelRunner` — no HTTP, no Mooncake) and returns
  `torch.cat([h.unsqueeze(0) for h in aux_states])`, i.e. a stacked `[num_layers, seq, hidden]`.
  Training then reads `data.hidden_states_path`. **This is the only SpecForge path that
  could work here** — but it still subclasses *SGLang's* ModelRunner, so it inherits the
  same version conflict, just less acutely (it needs our fork's ModelRunner API, not a patch file).
- Single-GPU training is supported: `specforge/distributed.py` is raw `torch.distributed` +
  `init_device_mesh` with `world_size==1` short-circuits — **no FSDP, no DeepSpeed, no DDP**.

### 1b. DECISION: standalone PyTorch trainer (SpecForge as reference, not dependency)

**We already have one, written and debugged on this fleet:** `~/dspark_finetune.py` (569 lines,
this head) — *"faithful training-forward mirror of the fork's inference path
(qwen3_dflash/qwen3_dspark)"*, built for the GLM-5.2 DSpark speculator. It implements the
full DSpark stack from scratch in plain PyTorch: RMSNorm, YaRN-less RoPE, the 5 Qwen3-style
layers, **dual-source non-causal attention over `[window ; block]`**, `fc`+`hidden_norm`
context projection, `markov_w1/markov_w2`, the confidence head, fp32 master weights, AdamW,
gradient accumulation, multi-node data-parallel via `torchrun`, and a `NAME_MAP` that saves
back into the original checkpoint layout.

Porting it to Inkling is a **parameter-swap, not a rewrite**:

| | GLM-5.2 DSpark | Inkling DSpark | action |
|---|---|---|---|
| hidden | 6144 | 4096 | config |
| aux width | 30720 (5×6144) | **20480 (5×4096)** — matches `fc.weight [4096, 20480]` ✓ | config |
| n_heads / kv | 64 / 64 | 64 / 16 (GQA) | `enable_gqa=True` already in the SDPA call ✓ |
| head_dim | 64 | 64 | ✓ |
| FFN | 12288 | 8192 | config |
| vocab | 154880 | **201024** (padded; 200058 tokenizer entries) | config |
| markov_rank | 256 | 256 ✓ | — |
| block_size | 7 | 15 (train) / 7 (serve) | config |
| checkpoint layout | speculators (vLLM) | **SpecForge** (`markov_head.markov_w1.weight`, `confidence_head.proj.{weight,bias}`) | `NAME_MAP` — already the exact names it maps to ✓ |
| `embed_tokens` / `lm_head` | **in the draft ckpt** | **NOT in the draft ckpt** | ⚠️ see below |

**The one structural difference.** Verified by reading the draft's safetensors header: it has
**62 tensors and no `embed_tokens`, no `lm_head`** (the card confirms: *"target embedding and
unembedding weights are not included"*). SGLang feeds the draft the **target's** embedding
(`noise_embedding = target.model.embed_tokens(block_ids)`) and unembedding
(`draft_logits = target.lm_head(...)`). So the trainer must load them **frozen** from the target.

> **DONE THIS SESSION.** `model.llm.embed.weight` and `model.llm.unembed.weight` are plain
> **BF16 `[201024, 4096]`** in the NVFP4 checkpoint (no dequant needed), extracted by byte-range
> `dd` (golden weights read-only, never opened for write) to
> `~/inkling-campaign/work/target_embed_unembed.safetensors` (3.29 GB, keys
> `embed_tokens.weight` / `lm_head.weight`, finiteness- and absmax-checked).

**Param count check:** 5 × 142.6M (attn+MLP) + 83.9M (`fc`) + 2 × 51.5M (markov) ≈ **0.90B**,
matching the card and the 1.80 GB bf16 file. ✓

### 1c. Two prep bugs the GLM campaign already paid for — do not rediscover

From `~/GLM52_1M_RECIPE.md` §9, both read directly off `spec_generate`, and both apply
verbatim to the local `dflash.py` (`[:, -block_size + 1 :, :]` is right there in `spec_generate`):

1. **The draft predicts `block_size - 1` tokens, not `block_size`.** At block 15 that is **14**.
2. **Block slot 0 is the real token at position *p*, not a MASK.** Only slots `1..K` are masked.

Getting either wrong silently trains against a shifted target and produces a draft that looks
fine in the loss curve and dies at the gate.

---

## 2. DATA

### 2a. What the architecture actually consumes

From `dflash.py` / `dspark.py` (read on disk, not inferred), one training row needs:

| tensor | shape | source |
|---|---|---|
| `target_hidden` | `[T, 20480]` — layers **[5,11,23,29,35]** concatenated | target forward |
| `input_ids` | `[T]` | tokens |
| `positions` | `[T]` absolute | tokens |

`fc` projects `[T, 20480] → [T, 4096]`, `hidden_norm` normalizes, and every draft layer takes
its K/V from that stream while the query block attends non-causally over `[window ; block]`.
**There is no draft-only forward** — token-only distillation is architecturally impossible.

### 2b. `--enable-return-hidden-states` — CHECKED, AND IT DOES NOT WORK FOR US

The flag **exists** in our image (`python -m sglang.launch_server --help` → *"Enable returning
hidden states with responses"*). Three independent reasons it is useless here:

1. **Last layer only.** `batch_result_processor.py:805` — *"hidden_states is `[bs * stride,
   hidden_dim]`, one row per emitted token"*. One layer. We need five. SGLang issue #8069
   ("extract hidden states from intermediate layers") is **closed with no implementation**.
2. **Explicitly incompatible with our speculator.** `srt/speculative/dflash_utils.py:840-841`:
   `if enable_overlap and req.return_hidden_states: return "DFLASH speculative decoding does
   not support return_hidden_states yet."`
3. **The production serve was not launched with it** (verified in `docker inspect` — the flag
   is absent), and enabling it requires a restart we are not permitted to do unilaterally.

### 2c. THE CAPTURE POINT — a ~20-line patch, and it is already computing exactly what we need

SGLang *already* extracts our five layers every prefill to feed the draft. In
`srt/speculative/dspark_components/dspark_worker_v2.py`:

```python
# ~L406  batch_output = self.target_worker.forward_batch_generation(
#            batch, capture_hidden_mode=CaptureHiddenMode.FULL)
# ~L414  if logits_output.hidden_states is None: raise RuntimeError(
#            "DSpark requires target aux hidden capture for prefill, but got None. ...")
# ~L437  self._kv_injector.inject_target_hidden(
#            target_hidden=logits_output.hidden_states,
#            cache_loc=batch.out_cache_loc, positions=positions)
# ~L442  logits_output.hidden_states = None      # <-- INSERT DUMP IMMEDIATELY ABOVE
```

At that line `logits_output.hidden_states` is precisely `[sum(extend_lens), 20480]` — the packed
five-layer aux stream — alongside `positions`, `batch.extend_lens`, `batch.prefix_lens` and the
request ids. An env-gated `INKLING_AUX_CAPTURE_DIR` hook writing
`{aux [T,20480] bf16, input_ids [T], positions [T], req_id}` shards is all the plumbing needed.
This is **far simpler than the SpecForge Mooncake path and works with our own SGLang build**.

Note it captures on the **prefill/extend** path — which is exactly the teacher-forcing regime
the trainer wants, and is 1-2 orders of magnitude faster than harvesting during decode.

### 2d. Pipeline — decoupled into a restart-free stage and a restart-gated stage

**Stage 1 (NO restart, NO extra GPU, can run today): harvest target self-regenerations.**

`~/inkling-campaign/tools/gen_corpus.py` (written this session). Sends rate-limited,
strictly-sequential requests to the live serve at **temp 1.0 / top-p 0.95** (RadixArk's stage-1
sampling), mixing prose continuation (`/generate`), chat (`/v1/chat/completions`, so the chat
template + `inkling` reasoning parser apply) and code — shaped to our serving distribution.
Writes JSONL `{id, kind, prompt, text, completion_tokens}`. Has a `--min-tps` circuit breaker
that aborts the harvest if observed throughput drops, so production traffic always wins.

> **Why self-regenerations and not a scraped corpus.** At temperature 0, speculative decoding
> accepts iff the draft token equals the **target's argmax**. The draft must be fit to the
> target's own output distribution, not to human text. This is why RadixArk used *"444,264
> Inkling-Small self-regenerations"* rather than raw `open-perfectblend`. Prefilling
> scraped text would train the draft on the wrong distribution — it would look like it was
> learning and gain nothing at the gate. **This is the highest-leverage design decision in
> the whole plan.**

> 🔴 **SMOKE-TEST FINDING (2026-08-01) — THE HARVEST RECIPE AS PUBLISHED DOES NOT WORK HERE.**
> The first 12-row smoke run produced 3 rows before the serve was restarted out from under it,
> and **all 3 rows are degenerate word-salad**, across prose *and* code:
>
> ```
> kind=prose words=277 uniq_ratio=0.41     kind=code words=293 uniq_ratio=0.58
> "...owns US local noon became time via rail conciseI think theyPrior to railway standard,
>  local noon common via \n\n producePrior standard local noon railwayBefore cleaned:..."
> ```
>
> Healthy English runs uniq ≈ 0.65-0.75 with ~25-30% function words; these have visibly
> dropped function words. Two independent signals point the same way: throughput during the
> harvest was **10.2 tok/s**, versus 34-50 tok/s measured at temp 0 minutes earlier.
>
> **Hypothesis (untested — the serve went down before a sweep was possible):** the sampled
> (temp > 0) path on this stack degrades under speculative decoding. At temp 0 spec-decode
> accepts on exact argmax match; at temp 1.0 it must use rejection sampling, and a bug or
> distribution mismatch there would produce *both* the accept collapse (→ 10 tok/s) *and* the
> corrupted token stream. PROGRESS.md never benchmarked anything but temp 0, so this path is
> unexercised on this stack.
>
> **REQUIRED BEFORE ANY BULK HARVEST — a temperature sweep** (temp ∈ {0, 0.3, 0.7, 1.0} ×
> `/generate` vs `/v1/chat/completions`), scoring coherence and tok/s at each point, to find
> the highest temperature that still yields clean text. If **no** temp > 0 is clean, fall back
> to **temp 0 with a large, diverse prompt bank** for corpus diversity instead of sampling
> diversity — slower to diversify but it is the regime we actually serve in, so it is arguably
> the *better* training distribution anyway.
>
> A coherence gate (`--min-uniq 0.60 --min-stop 0.22`, rejecting and counting degenerate rows)
> has been added to `gen_corpus.py` so this can never silently poison the corpus.
>
> 🔴 **Throughput correction.** At the observed 10.2 tok/s, 2 M tokens single-stream is
> **~55 h**, not 12 h. Mitigations, in preference order: (a) fix/avoid the sampling path per
> the sweep — temp 0 runs at 34-50 tok/s, i.e. **~15 h** for 2 M; (b) raise harvest concurrency
> to 4 (serve is `--max-running-requests 16`), costing production some capacity; (c) cut the
> primary tier to 1 M tokens. The smoke run's `--min-tps 8.0` circuit breaker was **1 tok/s
> away from firing** — raise it once the clean regime is known.

**Stage 2 (ONE restart, 2-node window): replay-prefill the corpus to harvest aux.**

Relaunch the serve from `~/inkling-sglang-launch.sh` with the capture hook enabled and
`INKLING_AUX_CAPTURE_DIR=/capture` bind-mounted to `.2:~/finetune-inkling-dspark/aux`, then
push the stage-1 corpus back through as prefill-only requests (`max_new_tokens: 1`). Prefill
runs at `--chunked-prefill-size 8192`, so this is throughput-bound, not decode-bound.

**Stage 3: prep → training rows.** Port `~/dflash-tools/dflash_capture_prep.py` (on .2):
sample anchors, build `[window ; block]` pairs, hold out topic-disjoint rows.

> ⚠️ **RINGFIX — the known failure mode. `feedback_dspark_ring_contamination`: "DSpark ring
> ingests rejected rows".** The GLM campaign dropped **47% of raw rows** as rejected/duplicate
> because *"the capture hook includes rejected positions by design"*. Our capture is on the
> **prefill/extend** path, where every token is a real accepted target token — which structurally
> avoids the ring bug. **The prep step must still assert this**: every captured span must be
> contiguous in `positions`, and any capture from a verify/decode step must be discarded, not
> merely deduplicated. **Valid-rows-only. Assert it; do not assume it.**

### 2e. Volume and storage — recalibrated against real prior art

Aux is **5 × 4096 × 2 B = 40 KB/token** (bf16). Sizing anchored on the GLM DFlash finetune,
which is the only measured precedent on this fleet: `~/dflash-ft-data` on .2 is **6.7 GB** for
**220,175 raw rows → 113,873 training pairs**, and produced accept@1 **0.2515 → 0.3774 (1.50×)**
in **4000 steps**.

| tier | tokens | aux size (bf16) | stage-1 harvest @ temp 0 (34-50 tok/s) | @ temp 1.0 (10 tok/s, degenerate) | stage-2 replay-prefill | fits .2 (841 GB free) |
|---|---|---|---|---|---|---|
| smoke | 0.2 M | 8 GB | ~1.5 h | ~5.5 h | ~5 min | ✅ |
| **primary** | **2 M** | **80 GB** | **~15 h** | ~55 h | **~40 min** | ✅ |
| stretch | 10 M | 400 GB | ~3 d | — | ~3 h | ✅ (tight) |

Stage-1 is the slow half — it is generation-bound at ~35-50 tok/s single-stream, deliberately
rate-limited. It is also **fully background and disturbance-free**, so it can run for days
while everything else proceeds. fp8 aux storage halves the footprint if the stretch tier is
wanted; not needed for the primary tier.

**2 M tokens ≈ 10× the GLM campaign's data** on a draft of similar size. That is the right
primary target; more data was explicitly listed in the GLM recipe as an open question for
why that finetune plateaued (§10 Q3), so the stretch tier is the natural follow-on if the
primary run plateaus rather than diverges.

---

## 3. TRAINING

### 3a. Node: `.4` / spark-13b3 — **verified free this session**

| | |
|---|---|
| GPU | GB10, **zero compute apps**, no containers |
| memory | 121 GB total, **108 GB available** |
| torch | **2.11.0+cu130 native** (matches SpecForge's pin exactly), transformers 5.12.1, safetensors 0.7.0, datasets 5.0.0 |
| disk | **68 GB free — the constraint.** Aux data lives on `.2` (841 GB) and is read over the 200G fabric, or staged per-shard. |

No container, no new wheels, no arm64 wheel hunt. This is why the standalone trainer beats
SpecForge here: **the environment already exists.**

### 3b. Memory budget (0.9B draft, one GB10)

| item | bytes |
|---|---|
| fp32 master weights (0.90 B) | 3.6 GB |
| bf16 compute copy | 1.8 GB |
| grads fp32 | 3.6 GB |
| AdamW `m`, `v` fp32 | 7.2 GB |
| frozen `embed_tokens` + `lm_head` bf16 (no optimizer state) | 3.3 GB |
| activations, window 1024, B=8, K=14, grad-ckpt off | ~4-8 GB |
| logits `[8, 14, 201024]` fp32 + CE workspace | ~0.3 GB |
| **total** | **≈ 24-28 GB** |

Against 108 GB available: **~4× headroom.** Training is not the bottleneck; nothing needs to be
sharded, offloaded, or quantized. Per `feedback_gmu_085_universal`, if a GPU-memory fraction is
ever set for this job it is **0.85**, and if it does not fit we surface it rather than raise it —
but at 25/108 GB that will not arise.

### 3c. Config

Anchored on RadixArk's own recipe (from the card) but scaled to one GPU and a warm start:

| knob | value | rationale |
|---|---|---|
| init | **warm start from the stock draft** | we are adapting, not pretraining |
| `block_size` | **15** (train) / 7 (serve) | the card trains at 15, serves at 7; `config.json` says 15 |
| K (predicted slots) | **14** = `block_size - 1` | §1c bug 1 |
| window | 1024 → 2048 | GLM used 1024; longer window costs attention quadratically |
| batch | 8 anchors × accum 4 = 32 | fits trivially; raise if step time is launch-bound |
| optimizer | AdamW, `weight_decay=0`, grad-clip 1.0 | card + GLM trainer |
| **LR** | **peak 5e-5, cosine to 5e-6, 200-step warmup** | ⚠️ **10× below the card's 6e-4.** The card's 6e-4 is a *from-scratch* stage-1 LR over 444 K sequences; the GLM trainer defaulted to **1e-5** for a warm-start adapt and noted *"bf16 AdamW at lr 1e-5 rounds updates to zero"* — hence fp32 master weights, which the trainer already does. 5e-5 splits the difference for a 10× larger dataset than GLM's. |
| steps | 4000, eval every 250 | GLM plateaued from step 2000; 4000 is the measured plateau point |
| precision | bf16 compute, **fp32 master + fp32 Adam states** | non-negotiable, see LR note |
| est. wall-clock | **2-6 h** for 4000 steps on one GB10 | dominated by the `[·, 14, 201024]` lm_head matmul |

### 3d. Loss — and how the two DSpark-specific heads are trained

The card specifies `0.1 CE + 0.9 L1 distillation + 1.0 confidence BCE`, 512 sampled anchors per
sequence, within-block decay γ = 28/3. Mapping each term onto the modules read from `dspark.py`:

- **CE (backbone + `fc` + `lm_head` path).** Cross-entropy of the draft's logits at slots `1..K`
  against the target's true next tokens. Weight per slot decays within the block by γ = 28/3
  (later slots matter less — they are reached less often).
- **L1 distillation.** L1 between the draft's final hidden `h` (post-`norm`) and the target's
  hidden at the same positions. This is the *dominant* term (0.9) in the card's recipe and is
  what actually aligns the draft into the target's representation space; plain CE alone
  underperforms. Our aux capture gives the target hidden for free — layer 35 of the packed
  stream is the natural regression goal.
- **Markov head (`markov_head.markov_w1: Embedding[201024, 256]`, `markov_w2: Linear[256 → 201024]`).**
  Not a separate loss. Per `VanillaMarkov.apply_block_logits`, it is a low-rank **bigram bias
  added to the base logits**, conditioned on the *teacher-forced previous token*, and it trains
  through the CE gradient. Two traps: (i) it must be conditioned on the **true** previous token
  during training, not the draft's own sample; (ii) `markov_w1`/`markov_w2` are 51.5 M params
  each — the GLM trainer deliberately kept `markov_w1` in bf16 without fp32 master to save
  optimizer state, and that choice should be re-examined here, not copied blindly.
- **Confidence head (`confidence_head.proj: Linear[4352 → 1]`, i.e. `hidden 4096 ⊕ markov_rank 256`).**
  Per `AcceptRatePredictor` and the card, a **BCE against the empirical per-position accept
  indicator** — label 1 if the draft's argmax at that slot equals the target's true token,
  else 0. The label is computable offline from the same captured rows, no extra data needed.
  ⚠️ The confidence head **does not affect acceptance under our current serving config**: the
  live serve runs `SGLANG_RAGGED_VERIFY_MODE=static`, which uses a fixed gamma and no confidence
  scheduling. It only pays off with `cap-accept` **plus a profiled SPS table** — which
  PROGRESS.md already measured as *worse* than static-7 without the table. So train it
  (it is nearly free, and it feeds the `confidence_head_with_markov` input path), but **do not
  count it in the accept forecast.**
- **Weighting.** Start at the card's `0.1 / 0.9 / 1.0`. The GLM trainer used `CE + 0.1·BCE`
  with no L1 term at all — and plateaued. **Adding the L1 distillation term is the single
  most likely reason our run beats GLM's +14%.**

### 3e. Fidelity gate BEFORE training — do not skip this

`~/GLM52_1M_RECIPE.md` §9: *"Fidelity gate PASSED before training (the step two earlier
campaigns skipped and died from)"*. Port `~/dflash-tools/dflash_fidelity_gate.py`. Requirement:
running the **stock** draft through our offline trainer forward on held-out captured rows must
reproduce the **live** serve's acceptance (3.44) to within noise. If offline and online disagree,
every subsequent training number is meaningless. **No training run starts until this passes.**

---

## 4. EVAL

**Gate:** `~/inkling-campaign/accept_probe3.py` — 4 topic-disjoint seeds × 8 reps = **32 samples**,
mean ± stderr. Mandatory: this stack is nondeterministic at temp 0 even with spec decoding off
(PROGRESS.md proved the nondeterminism lives in the **target forward** — triton split-KV /
marlin reduction order on sm_121), so single-run numbers are noise. Observed range on a fixed
config is 1.7-7.0 accept.

| | accept | tok/s |
|---|---|---|
| **baseline (n=32)** | **3.44 ± 0.17** | **34.3 ± 1.7** |
| baseline re-verified this session (n=12) | 3.68 ± 0.37 | 35.2 ± 3.4 |
| **minimum publishable win** | **≥ 3.95** (+15%, ~3σ) | ~38 |
| target | 4.2-4.5 | 40-45 |
| stretch | 5.0+ | 50+ |

- tok/s ≈ **8.5 × accept** on this stack (base decode ~8.5 tok/s), so accept is the whole story.
- **Both arms must be measured back-to-back in the same serve session**, stock vs tuned, or
  cross-session drift will swamp the effect.
- Report **per-seed** as well as pooled — the authors' 2.70-4.79 task spread means a win
  concentrated in one seed is a distribution-narrowing artifact, not a draft improvement.
- Secondary: offline per-slot `accept@1..@7` on the held-out rows (the GLM gate's output
  format), which is cheap, deterministic, and diagnoses *where* in the block the draft improved.
- ⚠️ Never bench with `SGLANG_SIMULATE_ACC_LEN` set.

---

## 5. GO / NO-GO

### VERDICT: CONDITIONAL GO

**Feasible and already de-risked:**
- ✅ Trainer exists (`~/dspark_finetune.py`), and the port is a parameter swap.
- ✅ Training node `.4` verified free — GPU idle, 108 GB available, torch 2.11+cu130 native.
- ✅ Memory budget 25/108 GB — 4× headroom.
- ✅ Frozen embed/unembed extracted and validated (done this session).
- ✅ Capture point located to the exact line; ~20-line patch, our own build, no Mooncake.
- ✅ Corpus harvest needs no restart and no extra GPU.
- ✅ Both known prep bugs and the ring-contamination failure mode are pre-solved.
- ✅ Statistically-powered gate already written and re-verified against the live serve.

**Blockers, both resource not technical:**
1. 🔴 **One production-serve restart** to enable the capture hook. Needs owner approval.
   Mitigation: stage 1 runs first and needs no restart, so the restart is only required
   ~12 h into the campaign.
2. 🔴 **A 2-node window for the capture engine.** The target is 160 GB NVFP4 and cannot fit
   on one 128 GB GB10 — capture needs TP=2. **All four nodes are currently GPU-saturated:**
   `.1` + `5482` = our production serve (~103 GB/GPU); `.2` + `.3` = **another session's**
   TP=2 serve of `inkling-small-nvfp4-ablit-v6` (~102 GB/GPU, ~114 GB host RAM each).
   Mitigation: fold capture into the production serve itself (§2c) — zero extra GPU, one restart.
   ⚠️ Do **not** run jobs on `.2` while it is serving: 6 GB available RAM, and per
   `feedback_never_run_jobs_on_serving_head` an OOM there triggers `panic_on_oom` → reboot.
3. 🔴 **Another session is churning `.1`.** The production serve was restarted **twice in
   15 minutes** (03:20 and 03:33), unannounced, killing both a smoke run and the harvest.
   Each restart costs ~80 s of weight load + helion compile. **Stage-1 harvest cannot run
   reliably until node ownership is settled** — a multi-hour background harvest against a
   serve someone else is cycling will produce a truncated, biased corpus.
4. 🟡 **The sampled-decode path may be broken on this stack** (§2d smoke finding). If no
   temperature > 0 produces clean text, the corpus must be built at temp 0, which changes
   the diversity strategy (prompt-bank breadth instead of sampling entropy).

### Expected gain — honest

| outcome | accept | tok/s | probability |
|---|---|---|---|
| no usable gain (< +10%) | ≤ 3.8 | ≤ 32 | **~25%** |
| modest win | 3.9-4.1 | 33-35 | ~25% |
| **target** | **4.2-4.6** | **36-39** | **~35%** |
| stretch | ≥ 5.0 | ≥ 42 | **~15%** |

Reasoning, not vibes:
- The only measured finetune on this fleet (GLM DFlash) got **accept@1 ×1.50 but accept_len
  only +14%** — per-slot gains compound poorly into block length.
- We start from a **fully-trained** draft at its **published** accuracy, not from a mismatched
  one. GLM's draft was trained for an FP8 target and adapted to Int4-Int8 — a domain gap that
  gave it easy headroom we **do not have**. This is the main reason to discount GLM's ×1.5.
- Offsetting that: 10× GLM's data, the L1 distillation term GLM omitted, and specialization
  to a genuinely narrower distribution than the authors' 9-task mix.
- Getting to 5.0 means a **general** +45% over the authors' own full-scale run. Only credible
  if our serving distribution is far narrower than we think — hence 15%, not 40%.

### Time and risk

| phase | wall-clock | risk |
|---|---|---|
| 1. Trainer port + fidelity gate (`.4`) | 1-1.5 d | low — trainer exists |
| 2. Stage-1 corpus harvest (background, no restart) | 12 h, overlaps phase 1 | **none** |
| 3. Capture patch + smoke (needs restart + approval) | 0.5 d | med — restart coordination |
| 4. Stage-2 replay-prefill (2 M tokens → 80 GB) | ~1 h | low |
| 5. Prep + ringfix assertions | 0.5 d | low |
| 6. Training, 4000 steps | 2-6 h | low |
| 7. Gate + package + A/B | 0.5 d | med — needs a second restart to serve the tuned draft |
| **total** | **≈ 3-5 days elapsed**, ~2 restarts | |

**Cheapest kill points, in order:** (a) fidelity gate fails → stop, the offline forward does not
mirror the serve; (b) 0.2 M-token smoke run shows no train-loss movement → stop, do not scale to 2 M.

### Cheaper alternative worth scoping first (~1 h, no training)

The target checkpoint ships **its own 8-layer MTP module** — `model.mtp.layers.0..7` in
`model.safetensors.index.json` / `mtp.safetensors`, which nothing in our serving config uses.
On this fleet MTP has repeatedly beaten a finetuned DFlash/DSpark draft outright
(`project_dflash_prose_finetune_result`, and `GLM52_1M_RECIPE.md` §9: *"the finetune genuinely
worked (1.5×) but it does NOT beat MTP"*). **If SGLang's Inkling path can serve those MTP
layers, that is a zero-training accept lever and should be checked before spending 3-5 days
on a finetune.** Cost to check: grep the Inkling model/spec code for an MTP path and try one
launch — but that needs a restart, so bundle it with the capture-hook restart.

---

## 6. WHAT WAS STARTED THIS SESSION (step 1)

| artifact | location | state |
|---|---|---|
| this plan | `~/inkling-campaign/DRAFT_FINETUNE_PLAN.md` | ✅ |
| stage-1 corpus harvester | `~/inkling-campaign/tools/gen_corpus.py` | ✅ written; smoke blocked on serve reboot (see below) |
| frozen target embed + unembed | `~/inkling-campaign/work/target_embed_unembed.safetensors` (3.29 GB) | ✅ extracted, shape/finiteness verified |
| campaign workspace | `.2:~/finetune-inkling-dspark/{corpus,aux,ckpt,tools,logs}` | ✅ created |
| baseline re-verification | `accept_probe3.py`, n=12 → 3.68 ± 0.37 accept / 35.2 ± 3.4 tok/s | ✅ consistent with 3.44 ± 0.17 |
| draft architecture audit | `dspark.py` / `dflash.py` / safetensors header (62 tensors, no embed/lm_head) | ✅ |
| capture point | `dspark_worker_v2.py` ~L437-442 | ✅ located, patch not written |

**Smoke test — RAN, and it earned its keep.** Two attempts:

- *Attempt 1:* `Connection refused` × 12 — another session restarted `.1` at 03:20.
- *Attempt 2 (after the serve came back):* **3 rows / 1152 tokens harvested**, then
  `Connection refused` × 9 — the same session restarted `.1` again at 03:33.

The 3 rows that did land are the valuable part: they proved the end-to-end path works
(request → parse → coherence scoring → JSONL) **and** exposed the degenerate-sampling blocker
in §2d before it could cost 55 hours of harvesting. The `--min-tps` breaker and the error
handling both behaved correctly. Raw smoke output kept at
`~/inkling-campaign/work/corpus-smoke.jsonl` (3 rows, `uniq` 0.41-0.58 — all would now be
rejected by the new gate).

Re-run once `.1` is stable and the temperature sweep has picked a clean regime:

```bash
cd ~/inkling-campaign && python3 tools/gen_corpus.py \
  --out ~/inkling-campaign/work/corpus-smoke.jsonl \
  --n 12 --max-new 384 --sleep 1.5 --temp <from sweep>
```

**Not started, by design:** no training run, no serve restart, no capture patch applied, no
golden weight touched, no long-running harvest launched (blocked on `.1` ownership).

---

## 7. NEXT ACTIONS (in order)

1. **[needs owner]** Confirm who owns `.1` and `.2`/`.3`, stop the unannounced `.1` restarts,
   and decide whether the production serve may be restarted once with the capture hook.
2. **[1 h, no restart] Temperature sweep** — temp ∈ {0, 0.3, 0.7, 1.0} × `/generate` vs
   `/v1/chat/completions`, scoring `coherence()` and tok/s, to find the cleanest harvest
   regime. **This gates everything in stage 1.** Do not start a bulk harvest before it.
3. Re-run the stage-1 smoke at the chosen temp, then launch the full background harvest at
   `--sleep 3.0` toward 2 M tokens.
4. Port `~/dspark_finetune.py` → `inkling_dspark_finetune.py` on `.4` (parameter swap per §1b,
   plus loading the frozen embed/unembed).
5. Port `~/dflash-tools/dflash_fidelity_gate.py` and **pass the fidelity gate** (§3e).
6. Write the ~20-line capture hook (§2c); bundle its restart with the MTP scoping check (§5).
7. Only then: capture → prep (ringfix assertions) → 0.2 M smoke train → 2 M full run → gate.

---

## APPENDIX: A4Q (native fp4 attention) — evaluated, not applicable to this stack

Added 2026-08-01 for completeness, since A4Q is a standing fleet capability
(`reference_jethac_a4q_blackwell_isa`) and the question will recur.

**What A4Q is:** jethac's native `mma.sync mxf4nvf4.block_scale` QKᵀ kernel, replacing the
~9-instruction fp4→fp16 unpack chain. Measured on this fleet: Nemotron-3-Omni TTFT −22% @60K
scaling to −39% @256K; Qwen3.6-27B −11% @60K → −23% @256K. Decode always neutral
(bandwidth-bound), quality/retrieval always identical.

**Why it does not apply to Inkling-Small + DSpark on SGLang:**

| Requirement | A4Q needs | Inkling-Small has | Verdict |
|---|---|---|---|
| KV width (`kv_heads × head_dim`) | **≥ 4096** to amortize quantization overhead | 8 × 128 = **1024** | ✗ 4× under — expected net *loss* |
| Attention backend | FlashInfer FA2 (jethac's fork) | `attn.py` asserts `fa4\|triton`; fa4 is sm_100-only ⇒ **triton mandatory** on GB10 | ✗ no seam to attach |
| Serving engine | vLLM integration (`aeon-vllm-a4q:port`, patched `flashinfer.py`) | SGLang | ✗ different plumbing |
| Prefill shape | gain ∝ *quadratic* attention prefill | `sliding_window_size: 512` + short-conv ⇒ ~linear prefill | ✗ little to accelerate |

The draft is no better a candidate: 16 kv-heads × 64 head_dim = **1024**, same threshold failure.

**When to revisit:** a future target/draft on this stack with dense-GQA and KV width ≥4096, served
on vLLM, or if A4Q's kernel is ever ported to a triton attention backend. Neither is on our path.

**Conclusion:** excluded from the finetune campaign. It would cost days and is projected to make
prefill *slower* here. A4Q remains the right tool for the fleet's dense-GQA wide-KV models on vLLM.
