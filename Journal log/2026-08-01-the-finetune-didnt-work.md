# The finetune didn't work — five runs, two learning rates, and the measurement that nearly fooled us

*2026-08-01 — negative result from the DSpark draft-finetune campaign.*

We tried to raise speculative acceptance by finetuning the 0.9B DSpark draft on 2M tokens harvested
from our own 276B target. It didn't work. This is the record of what we tried, what the numbers were,
and the three anomalies that are more interesting than the headline.

The short version: **at a normal learning rate we damaged a well-trained checkpoint; at a gentle one
the damage vanished and revealed nothing underneath.** No configuration beat stock.

**Claim:** on this stack, supervised finetuning of the RadixArk Inkling-Small DSpark draft on
target self-regenerations **does not raise speculative acceptance**, and at sufficient
training volume it *lowers* it. Five runs, three data scales, two learning rates, one
validated measurement instrument.

### What was tried

| | |
|---|---|
| draft | `RadixArk/Inkling-Small-DSpark-Preview`, 0.9B, 848M trainable |
| target | `thinkingmachines/Inkling-Small-NVFP4`, TP=2 GB10, fp4 KV, block 7 |
| data | 2.0M tokens of target self-regenerations (temp 0.7), 3435 rows, 2500 distinct open-perfectblend prompts, math-filtered to the serving mix |
| features | 5-layer aux hidden states `[T, 20480]`, captured on the prefill path (ringfix-clean: 11,002 spans, **0** rejected) |
| objective | `CE=1.0 / L1=0.0 / BCE=0.1` — both non-CE weights settled by their own A/B (§G) |
| harness | draft's own `dspark.py`/`dflash.py`, fidelity-gated at **3.08 offline vs 3.14 ± 0.20 live** (§D) |

### The numbers

⚠️ **All per-checkpoint numbers were first computed at 280 anchors. That estimate is
NOISE-DOMINATED and nearly produced a false positive — see "Anomaly 3" below.** The table
below is the corrected measurement at **1400 paired anchors** (same anchor set for every
checkpoint, so differences are paired; se(p1) ≈ 0.012).

| run | steps | LR | accept_len | Δ vs stock |
|---|--:|--:|--:|--:|
| **stock** | — | — | **2.828** | — |
| **full run** (2500 shards) | 2750 | 5e-5 | **2.698** | **−0.130 (−4.6%)** |
| **LR probe** (2500 shards) | 600 | 5e-6 | **2.820** | **−0.008 (−0.3%)** |

Two 600-step runs at 5e-5 (the BCE A/B, §G) and the 340-shard smoke run were measured only at
280 anchors and are **not resolvable** — their reported spread (2.604 / 2.613) is far below the
noise floor of that estimator. Treat them as "no detectable change", which is what they were.

**The verdict, from the LR discriminator:**

- At **5e-5** the draft is **damaged** (−4.6%), and damage grows monotonically with steps.
  Catastrophic forgetting is the mechanism: we pushed 2M tokens of a narrower distribution
  through 848M trainable params of a checkpoint RadixArk fitted on 444,264 self-regenerations.
- At **5e-6** the damage disappears — and **no gain appears underneath it**. Accept is
  statistically identical to stock (−0.008, well inside noise).

So the forgetting hypothesis is *confirmed as the damage mechanism* and simultaneously
*eliminated as the thing standing between us and a win*. Remove the forgetting and there is
nothing there. **This objective extracts no usable acceptance signal from this data at any
learning rate tested.**

Note the markov head strengthened in **every** run (+0.509 stock → +0.599 full → +0.594
probe) while accept did not improve. The model is demonstrably learning *something*; it is
simply not something acceptance pays for.

### Anomaly 1 — training accuracy ROSE while acceptance FELL

This is the sharpest finding in the campaign. Over the full run's second half:

```
rolling train acc: 0.4276 -> 0.4326 -> 0.4360 -> 0.4362   (monotone UP)
held-out accept_len:                              2.529   (DOWN from 2.612)
```

**Why they can diverge:** cross-entropy optimizes the *full next-token distribution*.
Speculative acceptance at temperature 0 rewards exactly one thing — the draft's **top-1
agreeing with the target's argmax**. Those objectives are correlated but not identical, and
this run is direct evidence they can move in opposite directions. A draft can get better at
modelling the target's distribution while getting *worse* at the specific argmax-match that
acceptance pays for.

**Implication:** CE is a mis-specified proxy for this task. Anyone picking this up should
consider training directly on the acceptance signal — e.g. a top-1 agreement / ranking loss
against the target's argmax, or a margin loss that only cares whether the correct token wins,
rather than how much probability mass the rest of the vocabulary gets.

### Anomaly 2 — calibration drift is distribution-LOCAL

The ECE canary fired on the *training-distribution* holdout (1.79× stock) and aborted the run.
On the *gate* distribution, the same checkpoint's calibration **improved**:

| | stock | tuned @2500 |
|---|--:|--:|
| gate-distribution ECE | 0.0288 | **0.0236** |
| gate-distribution AUROC | 0.8447 | **0.8533** |
| training-distribution ECE | 0.0210 | **0.0375 (1.79×)** |

So "drift" was real but local to the data being trained on. This corroborates the BCE A/B
conclusion (§G) that calibration is a distribution-specific property, and it is a caution:
**a drift canary measures drift relative to its own distribution, not damage in general.**
The canary was right to fire and right to be debounced; it was not measuring the thing that
actually regressed.

### Anomaly 3 — the measurement nearly produced a false positive, twice over

The LR probe first read **2.675 vs stock 2.612 = +0.063**, the only above-stock result of the
campaign, and it was about to be reported as "gentle adaptation works." Scrutinising it before
reporting — on the grounds that this campaign had already been burned twice by numbers that
matched suspiciously well — showed **two** problems at once:

1. **The signal was noise.** At 1400 paired anchors the probe is **2.820 vs 2.828 = −0.008**.
   The +0.063 was an artifact of a 280-anchor estimator.
2. **The baseline itself was wrong.** Stock measured **2.612** at 280 anchors and **2.828** at
   1400 — a **+0.217** shift, more than 3× the "improvement" being celebrated. Every absolute
   accept_len quoted earlier in this document from a 280-anchor run is biased low.

Because `head_diag.py` seeds its anchor sampler identically for every checkpoint, the
comparisons were at least *paired* — the same anchors for every model — which is why the
direction of the large full-run regression survived (−0.083 at 280 → −0.130 at 1400). But the
resolution was never sufficient for the small differences the BCE A/B was asked to adjudicate.

**Lesson, and it is the campaign's most transferable one:** an estimator must be validated for
*resolution*, not just correctness. `head_diag` was correct — it computed exactly what it
claimed — and still could not answer the question put to it, because nobody checked its
standard error against the effect size being hunted. se(p1) ≈ 0.030 at 280 anchors versus a
hoped-for effect of ~0.02 per slot. That is the same class of error as the ECE canary
threshold set below its own baseline (§I): **a measurement whose noise exceeds its signal is
not a weak measurement, it is a random number generator with a plausible label.**

### What the campaign got right, and what that cost

The instrument work was the valuable part, and it was load-bearing: without the §F card
reproduction (gsm8k-style **4.81** vs card **4.79**) we could not have trusted any accept
number; without the §D fidelity gate we would have trained through a wrong contract; without
the §E muP fix the markov head would have been silently disabled during training. Every one
of those was found by measurement, and three of them contradicted a document or a hypothesis
we had already committed to.

The finetune itself is a clean negative. That is a more useful publication than a lucky win:
four converging runs, a validated instrument, and two named mechanisms.

### What to try next, with fresh eyes

1. **Change the objective, not the hyperparameters.** Anomaly 1 says CE is the wrong loss.
   Train on top-1 agreement with the target's argmax directly.
2. **Freeze more of the backbone.** The markov head strengthened in *every* run
   (+0.474 → +0.517) while accept fell. Training only the heads — markov + `fc` — may capture
   the gain without overwriting the backbone that RadixArk fitted on 444K sequences.
3. **Accept that 2.27 pooled may be near this draft's ceiling** on open-ended prose/chat/code,
   and pursue throughput elsewhere: the target's own 8-layer MTP module (blocked today on
   memory and a `layers_mapping` KeyError, not on principle), or a larger draft.
4. **Do not re-run this recipe at another LR or data scale.** Five runs is enough.

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

---

## The lesson that outlived the experiment

The probe's first reading was **+0.063 above stock** — the only above-baseline result the campaign
ever produced. Scrutinising it before publishing found two failures stacked on each other: the
signal was noise (−0.008 at 5× the sample size), *and* the baseline itself moved +0.217 between
sample sizes — three times the "improvement" being celebrated.

The diagnostic wasn't buggy. It computed exactly what it claimed. It simply couldn't resolve the
effect being hunted: standard error ~0.030 against a hoped-for ~0.02.

> **A measurement whose noise exceeds its signal isn't a weak measurement. It's a random number
> generator with a plausible label.**

That is the same error as setting an abort threshold below its own baseline, and the same error as
the 64 tok/s figure this repository published and later retracted. Three times in one campaign, in
three different disguises. Check that your instrument can resolve the effect *before* you run the
experiment — not after it hands you the answer you wanted.
