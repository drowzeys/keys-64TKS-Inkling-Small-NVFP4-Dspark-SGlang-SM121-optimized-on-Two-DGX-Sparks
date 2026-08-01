# Measurement protocol — read this before quoting any number

**The target forward pass on this stack is nondeterministic at temperature 0.** Same prompt,
same seed, same config → different output text run to run. Proven spec-independent: it persists
with speculation OFF and is *not* fixed by `--enable-deterministic-inference` (the culprit is
kernel-level: triton split-KV attention reduction order and/or marlin MoE reduction on sm_121a,
neither covered by the deterministic path).

## Why that wrecks naive benchmarking

DSpark acceptance depends heavily on *which* continuation a run lands on:

| Continuation style | typical accept |
|---|---|
| repetitive / list-like / templated | 4.0 – 5.7 |
| novel coherent prose | 1.5 – 2.5 |

So a single 10-run probe on one prompt has a noise band wider than most config effects. During
this campaign that trap produced, on **identical** configs and images: 7.31, 2.44, 4.51, 3.09 —
and sent two separate agent sessions chasing a "regression" that never existed. An early headline
of "64.6 tok/s" was one lucky draw from a distribution whose mean was ~30.

## Two traps that produced published-but-wrong numbers here

**1. The echo trap inflates acceptance ~60%.** Feeding untemplated text to `/generate` makes this
model regurgitate the prompt, and repetitive output drafts trivially. A raw-continuation probe
measured 3.44 accept / 34.3 tok/s where a chat-templated, serving-representative gate measures
2.27 / 23.9. Always benchmark through the model's own chat template.

**2. Acceptance is task-dependent by more than 2×.** On this exact serve: GSM8K-style 4.81,
code 2.46, chat 2.20, open-ended prose 2.15. A single pooled number hides that. Report per class,
and say which class you measured — a "faster" config may simply have been measured on easier text.

## The protocol

Use [`benchmarks/accept_probe.py`](../benchmarks/accept_probe.py):

- **4 distinct mid-prose seeds × 8 reps = 32 samples** (mid-sentence continuations, never bare
  instructions — bare instructions through `/generate` fall into repetition traps that inflate accept)
- 2 warm-up calls first (cold first-request always reads low)
- reports **mean ± standard error** and range, per-seed and overall

```bash
python3 benchmarks/accept_probe.py "my-config-label" --reps 8
```

**Rules**
1. Never quote a single-run number. Ever.
2. Compare configs only via non-overlapping error bars. A +0.3 accept difference with ±0.18 se
   on each side is *suggestive*, not proven.
3. Re-measure after every reboot — boot-to-boot means shift.
4. Always state the task class. Chat-template traffic accepts differently from raw continuation.
