# Three things we got wrong about measuring Inkling-Small

*2026-08-01 — from the data-collection phase of the DSpark draft-finetune campaign.*

We set out to harvest a training corpus from our own serve. Before writing a line of training code
we ran a sampling sweep, and it invalidated three assumptions we'd been operating on — one of which
was a hypothesis *we* had published hours earlier as a probable bug.

These are useful independently of whether the finetune succeeds. All three generalize beyond this
model.

Three results from Phase A that are useful independently of whether the finetune succeeds.


### 1. `/generate` is an echo trap on Inkling-Small. Use the chat endpoint for data collection.

Feeding untemplated raw text to `/generate` makes this target regurgitate the prompt verbatim,
at **every** temperature from 0 to 1.0 (trigram-repeat 0.5-0.8):

```
"Before Gutenberg, books were copied by hand, and ideas travelled at the pace of a walking
 scribe. Before Gutenberg, books were copied by hand, and ideas travelled at the pace of a
 walking scribe. Before Gutenberg, exte..."
```

Inkling-Small is a chat/reasoning model with a chat template and an `inkling` reasoning parser;
raw continuation is off-distribution and it collapses into an echo loop. Temperature does not
rescue it — at 1.0 the model abandons the continuation and starts narrating *"The user is
asking for an analysis of..."*. **Any corpus built from `/generate` on this model is garbage.**
`/generate` remains appropriate for the *accept gate*, where a fixed repetitive continuation is
a stable, low-variance probe.

### 2. Stop-word ratio is the wrong degeneracy metric for a reasoning model. Use n-gram repetition.

Our first coherence gate required a function-word floor. It rejected **100% of chat rows** — and
the rows were fine. This target's reasoning traces are telegraphic scratchpad prose that
legitimately drops articles:

```
"We need to explain mechanism: B-tree is balanced tree, sorted keys, leaf nodes linked
 (usually). Range query: find lower bound via descent, then scan leaves..."
```

That scores stop-word ratio 0.16-0.22 versus ~0.30 for ordinary prose, and it is the single most
valuable training text the model produces. A **trigram-repeat ratio** separates the real failure
cleanly: validated on live rows, healthy text sits at rep **0.02-0.09** and degenerate text at
**0.18+**, so a 0.15 threshold has margin on both sides. Generalizes to any reasoning-model
corpus pipeline: *gate on repetition, never on function-word density.*

### 3. Speculative acceptance decays monotonically with sampling temperature (−42% from 0 → 1.0).

Same prompts, same draft, `/generate`, block size 7:

| temp | 0.00 | 0.01 | 0.30 | 0.70 | 1.00 |
|---|--:|--:|--:|--:|--:|
| accept_len | 4.03 | 4.07 | 3.66 | 3.10 | **2.33** |

Correct and expected — at temp 0 spec-decode accepts on exact argmax match, while at temp > 0 it
must pass a rejection-sampling test that fails more often. Two practical consequences:

- **Benchmark numbers are meaningless without a stated temperature.** A tok/s figure at temp 1.0
  is ~40% below the same config at temp 0 purely through acceptance, with no config difference.
  (Companion to `feedback_spec_k_is_task_dependent`: never quote tok/s without task class *or*
  temperature.)
- **Harvest temperature is a real trade-off**, not a free diversity knob: it buys corpus variety
  while moving the data away from the temp-0 regime we serve and gate in, and it costs harvest
  throughput roughly in proportion to the accept loss.

**Also worth recording:** the temp-0.01 cell is a cheap, general discriminator for "is the
sampled path broken or is my prompt wrong". At temp 0.01 sampling is effectively a delta at the
argmax, so it must reproduce greedy output. It did (accept 4.07 vs 4.03), which refuted the
sampled-decode-bug hypothesis in one measurement.

---

---

## Why we ran the sweep at all

The honest version: we didn't plan to. A smoke test harvested three rows, they looked like garbage,
and the fast conclusion was "we've found a sampled-decode bug on this stack — publishable." The
sweep was meant to characterize the bug before a 50-hour harvest ran on top of it.

Instead the sweep killed the hypothesis in one cell, exposed a metric of our own that was rejecting
good data, and quantified a temperature effect we'd have otherwise attributed to the wrong cause.
The cost was about forty minutes.

That's the pattern worth taking away — not any single finding. On unfamiliar hardware running an
unfamiliar model, the cheap characterization run before the expensive run keeps paying for itself.
See also [MEASUREMENT-PROTOCOL.md](../docs/MEASUREMENT-PROTOCOL.md), which exists because we learned
the same lesson the expensive way.
