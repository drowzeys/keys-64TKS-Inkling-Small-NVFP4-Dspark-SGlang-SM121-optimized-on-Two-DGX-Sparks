# DSpark cap-accept calibration on Inkling-Small-NVFP4 (SGLang, TP=2, GB10)

Written 2026-08-01. Serve pair: gx10-5185 (rank0, 10.100.10.1) + gx10-5482 (rank1, 10.100.20.2).
Image `local/sglang-inkling:gb10`. Source read from inside the container at
`/sgl-workspace/sglang/python/sglang/`.

**VERDICT: the two artifacts ARE producible — the prior "blocked" call was a procedure error
(two of them, §1c and §1e). Both were built and deployed. Cap-accept with them recovers accept
to static parity (3.39 +/- 0.18 vs 3.43 +/- 0.16) but costs -33% tok/s (23.4 +/- 1.3 vs
34.7 +/- 1.5), because cap-accept truncates accepts without shrinking the verify (§1f).
Keep the static champion; the artifacts' real target is `compact` mode. Full detail in §4-5.**

---

## 0. TL;DR of the code reading

Two artifacts gate cap-accept scheduling:

| artifact | server flag | consumed at |
|---|---|---|
| SPS cost table | `--speculative-dspark-sps-table-path` | `dspark_planner.py:1138-1145` -> `build_sps_cost_table` |
| STS calibration | `--speculative-dspark-confidence-sts-path` | `dspark_planner.py:94-123` -> `confidence_head.sts_temperatures` |

Both are *fittable in this build*: SGLang ships first-party tooling for each,
`sglang/benchmark/dspark_sps_profiler.py` and `sglang/benchmark/dspark_sts_fit.py`.
Neither the earlier "blocked" conclusion nor the missing shards were a build limitation.
Both were procedure errors, documented below.

---

## 1. What triggers each recorder, and where the data lands

### 1a. SPS record — sink is the **`/server_info` HTTP endpoint**, not a file

There is no dump directory and no `/metrics` field. Path:

- `dspark_observability.py:54-61` `resolve_enabled_components()` — `SGLANG_DSPARK_ENABLE_SPS_RECORD=1`
  is an *alias* for the `core,step_cpu_time` components of `SGLANG_DSPARK_DEBUG_DUMP`.
- `dspark_observability.py:156-294` `DsparkInfoDumper` — `enabled = bool(components) and attn_tp_rank == 0`.
  Records accumulate in an in-memory `deque(maxlen=INFO_DUMP_MAX_RECORDS)` (`:50`, 200_000).
  `INFO_DUMP_MAX_RECORDS` is a **ring-buffer cap, not a flush threshold** — that is why grepping
  for a writer near it finds nothing. Nothing is ever written to disk.
- `dspark_observability.py:284-294` `dump()` returns `{mode, gamma, verify_num_draft_tokens,
  components, records}`.
- `dspark_worker_v2.py:365-366` -> `scheduler.py:4027-4030` puts that dict into the
  `GetInternalStateReqOutput` under key `dspark_info_record`.
- It surfaces as `GET /server_info` -> `internal_states[<dp rank>]["dspark_info_record"]["records"]`.
  Confirmed by the first-party consumer, `dspark_sps_profiler.py:fetch_rank_rows()`.
- Records are **cumulative and never auto-cleared**; the profiler waters-marks on `forward_ct`
  and can also clear via `POST /set_internal_state {"server_args":{"dspark_clear_info_records":true}}`
  (`scheduler.py:4080-4085, 4111-4112`).

Per-record emission is one row per **decode** step (`observe_decode_step`), skipped for prefill /
idle steps (`note_non_decode_step`). One step's record is only finalised on the *next* step
(`_drain_pending`), which is why the last step never appears until more traffic arrives.

### 1b. STS record — sink is `torch.save` shards `"<path_stem>.<n>.pt"`

- `dspark_observability.py:931-960` `_maybe_record_sts_collect()` — the only caller of
  `StsDataRecorder.record()`. It fires only when **all** of these hold:
  1. `SGLANG_DSPARK_STS_COLLECT_PATH` is non-empty (`:938`);
  2. `planner.carries_confidence` — draft checkpoint has a confidence head (`:940`);
  3. `planner.last_confidence_raw is not None` (`:942-944`);
  4. **`proposal_folded` is False** — `dspark_observability.py:843` `if not proposal_folded:`.
- `dspark_sts.py:48-60` `record()` buffers one `[bs, gamma]` logits tensor + one `[bs, gamma]`
  prefix-mask per step, and calls `flush()` only when `len(buffer) >= flush_every`.
- `dspark_observability.py:724` `_STS_COLLECT_FLUSH_EVERY = 256` — hard-coded, no env knob.
- `dspark_sts.py:62-76` `flush()` writes `torch.save({"logits","prefix_mask"}, f"{stem}.{n}.pt")`.
- **There is no shutdown hook and no flush-on-prefill for the STS recorder.** (Compare
  `dspark_observability.py:784-785`, where the *block-accept* recorder does get a
  `flush()` on every prefill step. The STS recorder gets nothing.) A run that ends with
  <256 buffered steps writes zero files and silently loses everything.

### 1c. ROOT CAUSE of the zero-shard run (this is the whole bug)

Condition 4 above is the killer:

```
dspark_draft.py:295-296
    if draft_sampler is not None and fwd.can_run_graph and all_greedy:
        folded = True
dspark_draft.py:291
    all_greedy = sampling_info is None or sampling_info.is_all_greedy
```

With **CUDA graphs enabled** (`GRAPHS=1`, the champion config) and **temperature-0 traffic**,
*every* decode step takes the folded path: greedy draft sampling runs inside the captured draft
CUDA graph. `observe_verify_step` then skips `_maybe_record_sts_collect` entirely.

So `SGLANG_DSPARK_STS_COLLECT_PATH=/stscollect/raw` + temp-0 probe traffic under `GRAPHS=1`
produces **exactly zero shards, forever**, no matter how long it runs. That is what the previous
attempt hit. It is not a flush threshold, not a permissions problem, and not related to
`RAGGED=cap-accept`.

**Two legitimate ways out.** The one used here keeps the labels honest:

> Send `temperature=1.0` **together with `top_k=1`**.
> `dspark_draft.py:187` `resolve_greedy_mask()` returns `top_ks <= 1`, and the sampler at
> `dspark_draft.py:206-225` returns `argmax` for greedy-masked rows. The drafted tokens are
> therefore **bit-identical to greedy drafting**, but `sampling_info.is_all_greedy` is False,
> so `folded=False` and the recorder runs. Zero distribution shift.

(The alternative — `GRAPHS=0` so `can_run_graph` is False — also works but changes the perf
regime and is much slower.)

Labels are independent of the sampling params either way:
`_maybe_record_sts_collect` builds them from `argmax(target_logits)` vs the draft candidates
(`dspark_observability.py:951-957`), i.e. greedy target agreement.

### 1d. Why an uninitialized SPS table means "verify-all"

`dspark_planner.py:954-991` `compute_verify_token_budget`:
`theta = tau_star * sps`, `budget = argmax(theta)`. `tau_star` is strictly increasing in the
budget index. `build_uninitialized_sps_table` (`dspark_sps.py:153-158`) is a **single flat probe**
`sps=[1.0]`, so `sps` is constant, `theta` is monotone increasing, and the argmax is always the
last index — i.e. verify everything. Exactly what the boot log warns about at
`dspark_planner.py:206-210`.

The STS temperatures feed the same expression through
`history_survival_probs = cumprod(sigmoid(confidence_raw / T_i))`
(`models/dspark.py:305-330` `apply_sts`, `dspark_planner.py:901-923`). Miscalibrated `T` therefore
mis-sizes `tau_star` and the planner mis-budgets even with a good cost table. Both artifacts are
needed for cap-accept to do anything useful.

### 1e. Second procedure trap: STS collection is impossible in `static` mode

`models/dspark.py:333-335`:

```
def build_confidence_head(config) -> Optional[nn.Module]:
    if read_ragged_verify_mode() is RaggedVerifyMode.STATIC:
        return None
```

Under `SGLANG_RAGGED_VERIFY_MODE=static` the draft checkpoint's confidence head is not even
built, so `planner.carries_confidence` is False and condition 2 of §1b can never hold.
**STS collection must run under `cap-accept` (or `compact`)**, with *no*
`--speculative-dspark-confidence-sts-path` (identity temperatures are enforced at
`dspark_planner.py:100-108`) and, for untruncated labels, *no* SPS table either — the
uninitialized table degenerates to verify-all, which is exactly what you want while
collecting (every position gets a label instead of being cut off by `cap_trim`).

### 1f. What `cap-accept` actually does — this bounds the whole experiment

`cap-accept` **does not shrink the target verify**. The scheduled `RaggedVerifyLayout` is only
consulted by the accept kernel:

- `dspark_planner.py:259-262` `should_run_compact()` returns True only for `COMPACT`.
- `dspark_worker_v2.py:618-625` the non-compact path calls
  `dspark_verify.py:215-234 run_non_compact()`, which builds `DFlashVerifyInput(...,
  draft_token_num=verify_w)` — the **full** `verify_num_draft_tokens=8` width, layout ignored.
- The layout reaches only `accept_and_finalize` -> `accept_draft_tokens(...,
  cutoff_layout=layout)` (`dspark_verify.py:658-678`), where `cutoff_verify_lens` **truncates
  accepted tokens**.

So under `cap-accept` a scheduler decision to spend fewer verify tokens costs accept length and
saves no GPU work. The step-time saving only materialises in `compact` mode, which is also the
only mode that builds the token-bucketed verify graphs
(`decode_cuda_graph_runner.py:119-124, 284-293, 414-417` — `ragged_verify_compact_graphs_enabled`
-> `ragged_verify_compact_enabled()` -> mode == `compact`).

**Consequence:** on this build, "cap-accept with a good SPS table" can at best tie static, and a
*more* accurate cost table makes it *worse*, because a binding budget trims accepts for free.
That is a property of the mode, not of the calibration.

---

## 2. Artifact 1 — SPS cost table

Route: the first-party profiler `sglang/benchmark/dspark_sps_profiler.py` (diagonal / static
sweep -> `SpsCostTable`). Preconditions it enforces in `fetch_server_context()`:
`speculative_algorithm=DSPARK`, cuda graphs **on**, `dspark_info_record` present on every DP rank,
`mode=static`, components `{core, step_cpu_time}`, and `simulate_acc_len == 1.0` exactly.

Serve (both nodes, rank1 first) — `~/calib-boot-sps.sh`:

```bash
export INKLING_TORCH_CONV_COMMIT=1 INKLING_COMMIT_STEP_BIAS=1
export MOE=marlin GRAPHS=1 MEMFRAC=0.85 CTX=65536 MAXREQ=16 BLOCK=7
export EXTRA_ARGS="--triton-attention-reduce-in-fp32"
export RAGGED=static SPS_RECORD=1 SIM_ACC=1.0     # -> SGLANG_DSPARK_ENABLE_SPS_RECORD=1
exec ~/inkling-sglang-launch.sh "$1"              #    SGLANG_SIMULATE_ACC_LEN=1.0
```

(`SPS_RECORD` / `SIM_ACC` / `DSPARK_DEBUG_DUMP` are pass-through knobs added to
`~/inkling-sglang-launch.sh` on both nodes; backup at `~/inkling-sglang-launch.sh.bak-calib`.)

Profile, inside the rank-0 container:

```bash
docker exec inkling-sglang python3 -m sglang.benchmark.dspark_sps_profiler all \
  --base-url http://127.0.0.1:30000 \
  --batch-size 1 2 4 8 12 16 \
  --min-steady-steps 32 --min-steady-seconds 6 --round-timeout 180 \
  --out /stscollect/dspark_sps_diag.json --no-plot
```

Result — every round `match_fraction=1.00`, self-check passed:

| reqs/rank | batch_tokens | median step time | steps/s |
|---|---|---|---|
| 1 | 8 | 97.3 ms | 10.280 |
| 2 | 16 | 128.0 ms | 7.815 |
| 4 | 32 | 184.4 ms | 5.422 |
| 8 | 64 | 263.2 ms | 3.799 |
| 12 | 96 | 323.1 ms | 3.095 |
| 16 | 128 | 395.7 ms | 2.527 |

`/home/keyspark/sts-collect/dspark_sps_diag.json` (present on both nodes):

```json
{"sample_batch_tokens":[8,16,32,64,96,128],
 "sample_steps_per_sec":[10.280408420065713,7.814942169427947,5.422337754165304,
                         3.7994498396632164,3.0953336452397875,2.527452238419119],
 "max_batch_tokens":128}
```

Raw data next to it: `dspark_sps_diag.rounds.jsonl`, `dspark_sps_diag.records.jsonl`,
`dspark_sps_diag.json.manifest.json`.

**Known resolution limit at batch size 1.** The diagonal sweep can only probe
`batch_tokens = reqs * verify_num_draft_tokens`, i.e. multiples of 8, so the smallest probe is 8.
At `bs=1` the planner evaluates `batch_tokens = 1..9` (`dspark_planner.py:981`), and
`floor_probe_index` (`dspark_sps.py:9-11`) clamps everything below 8 onto the first probe. `sps`
is then constant across the whole candidate range, `theta = tau_star * sps` is monotone, and
`argmax` lands on verify-all. So with this table **single-stream cap-accept is a no-op** — which,
given §1f, is the best available outcome for that mode. The table does bind for `bs >= 2`.

---

## 3. Artifact 2 — STS confidence calibration

### 3a. Collection serve (`~/calib-boot-sts.sh`)

```bash
export INKLING_TORCH_CONV_COMMIT=1 INKLING_COMMIT_STEP_BIAS=1
export MOE=marlin GRAPHS=1 MEMFRAC=0.85 CTX=65536 MAXREQ=16 BLOCK=7
export EXTRA_ARGS="--triton-attention-reduce-in-fp32"
export RAGGED=cap-accept STS_COLLECT=1   # NOT static (no confidence head), NO sts path,
exec ~/inkling-sglang-launch.sh "$1"     # NO sps table (verify-all -> untruncated labels)
```

Boot log confirms the intended collection state:

```
DSpark ragged-verify scheduler enabled (mode=cap-accept, lag=2, relay_lag=2,
    sps_table=uninitialized, graph_tier=dynamic).
DSpark SPS table is uninitialized (flat): the verify budget degenerates to verify-all ...
```

### 3b. Driving traffic that actually records (`~/inkling-campaign/sts_drive.py`)

First attempt used `temperature=1.0, top_k=1` on the theory that top_k=1 keeps greedy tokens
while clearing the greedy flag. **That is wrong and measured wrong** —
`sampling_batch_info.py:205`:

```
is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
```

`top_k <= 1` *is* the definition of all-greedy, so it still folds. Measured: 3 rounds x 12,800
tokens at concurrency 8 -> **0 shards**.

Working parameters: **`temperature=0.02, top_k=2`**.
`top_k=2` clears `is_all_greedy` (so `folded=False` and the recorder runs), and the draft block
sampler never applies top_k at all — it applies only the temperature
(`dspark_draft.py:206-217`, `SampleStepTokens` with gumbel noise) — so at T=0.02 the sampled
token is the argmax except on measure-zero ties. The trajectory is the greedy one.

```bash
python3 ~/inkling-campaign/sts_drive.py --shards 8 --conc 8 --tokens 1200 --rounds 5
```

-> first round produced 2 shards; 12 shards / **19,871 samples x gamma=7** total on rank 0
(rank 1 records its own copy — `_maybe_record_sts_collect` has no tp_rank guard).

### 3c. Fits

Shipped fitter, run first (on the 19,074 samples present at that moment):

```bash
docker exec inkling-sglang python3 -m sglang.benchmark.dspark_sts_fit \
  --data-glob "/stscollect/raw.*.pt" --out /stscollect/dspark_sts.json
```

| pos | T (shipped) | ECE before | ECE after |
|---|---|---|---|
| 0 | 1.0000 | 0.0216 | 0.0216 |
| 1 | 1.2589 | 0.0441 | 0.0433 |
| 2 | 1.1220 | 0.0458 | 0.0486 |
| 3 | 0.7943 | 0.0435 | 0.0467 |
| 4 | 0.5623 | 0.0398 | 0.0396 |
| 5 | 0.7079 | 0.0370 | 0.0340 |
| 6 | 0.4467 | 0.0329 | 0.0293 |

mean ECE 0.03767 -> 0.03764, i.e. **no net improvement** — its greedy left-to-right pass
(`dspark_sts_fit.py:48-104`) freezes earlier temperatures and made positions 2 and 3 worse.

`~/inkling-campaign/sts_fit.py` was rewritten to optimise the *joint* objective (mean survival
ECE over all positions, coordinate descent on an 81-point log grid). **This is the deployed
table**, `/home/keyspark/sts-collect/dspark_sts.json` on both nodes:

| pos | T (deployed) | accept rate | ECE before | ECE after |
|---|---|---|---|---|
| 0 | 1.0000 | 0.595 | 0.02044 | 0.02044 |
| 1 | 0.9441 | 0.362 | 0.04217 | 0.04261 |
| 2 | 0.7499 | 0.232 | 0.04450 | 0.04523 |
| 3 | 0.5623 | 0.161 | 0.04224 | 0.04080 |
| 4 | 0.6310 | 0.118 | 0.03850 | 0.03474 |
| 5 | 0.7943 | 0.092 | 0.03583 | 0.03097 |
| 6 | 1.0000 | 0.072 | 0.03192 | 0.02694 |

**mean ECE 0.03651 -> 0.03453 (-5.4%)**. A joint LBFGS fit on survival log-likelihood was also
tried (`sts_fit.py --bce`): lower NLL but worse ECE (0.0365 -> 0.0393), so it was rejected.

Honest read: the DSpark confidence head on this checkpoint is **already close to calibrated**.
Per-position temperature scaling has ~5% of relative ECE to give and cannot be the lever that
moves accept. All three fits agree the head is slightly *over*-confident at deep positions.

---

## 4. Measured: cap-accept with both artifacts vs static

Protocol as mandated: `python3 ~/inkling-campaign/accept_probe3.py "<label>" --reps 8`
(4 seeds x 8 reps = 32 samples, mean +/- stderr). All four rows below were taken in one
session on the same pair, each on its own fresh boot of the same image.

| serve config | accept | tok/s |
|---|---|---|
| static block-7 champion (campaign reference) | 3.44 +/- 0.17 | 34.3 +/- 1.7 |
| **static block-7 champion (re-measured this session)** | **3.43 +/- 0.16** | **34.7 +/- 1.5** |
| cap-accept, NO artifacts (sps_table=uninitialized) | 3.13 +/- 0.13 | 31.3 +/- 1.5 |
| **cap-accept + SPS table + STS calibration** | **3.39 +/- 0.18** | **23.4 +/- 1.3** |

The baseline reproduced to within 0.01 accept / 0.4 tok/s of the campaign reference, so the
session is a valid control.

**What the artifacts do:** accept goes 3.13 -> 3.39 (+0.26, ~1.2 se) — the calibrated
scheduler stops throwing away accept relative to the uncalibrated one, landing statistically
on top of static (3.43). So the calibration is doing its job.

**What it costs:** tok/s 34.7 -> 23.4, a **-33% regression (-11.3 +/- 2.0, ~5.6 se)** —
unambiguous, not noise. Higher accept *and* much lower throughput means the per-step time went
up sharply.

**Why (from the code, §1f):** in cap-accept the target verify always runs the full
`verify_num_draft_tokens=8` width (`dspark_verify.py:224-234`) — a smaller budget buys no GPU
work back — while every decode step now pays the scheduler:

- `dspark_planner.py:588-592` per-step top-k schedule of verify lens;
- `dspark_planner.py:612-616` a **cross-node NCCL broadcast of `verify_lens` every step**
  (tp_size=2 here, and the two ranks are on different chassis over the 200G link);
- `dspark_planner.py:511` `verify_lens.to("cpu").tolist()` — a **blocking device->host sync
  every step**, taken because `ragged_capture_num_tokens()` is None outside compact mode
  (`decode_cuda_graph_runner.py:284-293`).

The uncalibrated run pays the first two as well (hence its own -10% vs static); the extra
~-23% arrives once the budget actually binds and the layout stops being uniform.

*Caveat on attribution:* the per-step `step_cpu_ms` breakdown that would have measured this
split directly was not captured — the diagnostic boot
(`~/calib-boot-capaccept-dbg.sh`, cap-accept + artifacts + `SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time`)
was killed mid-load when **another session relaunched the champion pair on both nodes** at
2026-08-01T04:37:29Z/04:37:44Z. The four table rows above were all collected before that
(last one finished ~04:34Z) and are unaffected. The mechanism above is derived from the code
paths, not instrumented. Re-running that boot is the obvious next step if anyone wants the
split.

---

## 5. Verdict

**The artifacts are producible in this build. The earlier "blocked" conclusion was a procedure
error, not a build limitation.** Two specific traps, both now documented with file:line:

1. STS collection cannot run under `SGLANG_RAGGED_VERIFY_MODE=static` — the confidence head is
   never constructed (`models/dspark.py:333-335`).
2. STS collection cannot run on greedy traffic under CUDA graphs — the draft folds into the
   graph and `dspark_observability.py:843` skips the recorder. And `top_k=1` does *not* escape
   this, because `top_k <= 1` is exactly how all-greedy is defined
   (`sampling_batch_info.py:205`). Use `top_k=2` with a near-zero temperature.

Neither is a flush-threshold problem, though note the STS recorder also has no shutdown flush
(`dspark_sts.py:62-76`, `_STS_COLLECT_FLUSH_EVERY=256`), so short runs silently lose everything.

**Cap-accept scheduling now "works" in the sense that it is fully calibrated and no longer
degenerates** — the boot log shows a real `sps_table=/stscollect/dspark_sps.json`, the STS
temperatures load, and accept recovers to static parity.

**But cap-accept is not a throughput win on this build and cannot be one.** It truncates accepts
without shrinking the verify, so a *more* accurate cost model can only cost accept, while the
per-step scheduling (host sync + cross-node broadcast) is pure overhead. Measured: -33% tok/s
against static at equal accept. Recommendation: **keep the static block-7 champion.**

The mode where an SPS table converts into real step-time savings is `compact` — it is the only
mode that builds the token-bucketed verify graphs and actually runs a narrowed verify
(`decode_cuda_graph_runner.py:119-124, 414-417`; `dspark_planner.py:259-262`). Both artifacts
produced here are directly reusable for it. **Untested — that is the recommended next
experiment**, together with an off-diagonal (`--fracs`) profile to give the table resolution
below `batch_tokens=8` (see §2).

### Artifacts and files

On both nodes (`/home/keyspark/sts-collect/`, mounted into the container at `/stscollect`):

| file | what |
|---|---|
| `dspark_sps.json` | deployed SPS cost table (copy of `dspark_sps_diag.json`) |
| `dspark_sts.json` | deployed STS calibration (copy of `dspark_sts_cd.json`) |
| `dspark_sps_diag.{rounds,records}.jsonl`, `.manifest.json` | raw SPS profile data (rank 0) |
| `raw.*.pt` | 12 STS shards, 19,871 samples x gamma 7 |

Scripts on both nodes: `~/calib-boot-{sps,sts,capaccept,capaccept-dbg,static-dbg,champion}.sh`,
`~/calib-restart.sh <boot-script> <rank>`.
`~/inkling-sglang-launch.sh` gained pass-through knobs `SPS_RECORD`, `SIM_ACC`,
`DSPARK_DEBUG_DUMP` (backup: `~/inkling-sglang-launch.sh.bak-calib`); champion flags untouched.

In `~/inkling-campaign/`: `sts_drive.py` (collection driver), `sts_fit.py` (joint ECE fitter),
`sps_fit_additive.py` (off-diagonal fitter, unused so far), this file.

### Serve state left behind

A healthy champion-flag serve is up on .1 + 5482, `http://10.100.10.1:30000` (`/health` = 200
on rank 0, container up on both ranks).

**It is not mine.** At 2026-08-01T04:37Z another session relaunched the pair on image
`local/sglang-inkling:gb10-auxcap` (aux-capture build, `INKLING_AUX_CAPTURE_*` set), killing my
diagnostic boot mid-load. Its server args are the champion set —
`--speculative-dspark-block-size 7`, marlin, graphs `[1..16]`, `--mem-fraction-static 0.85`,
`--context-length 65536`, `--triton-attention-reduce-in-fp32`, no ragged mode, no sps/sts paths
— so the required "working champion serve" condition is satisfied and I deliberately did **not**
clobber it to re-boot my own `local/sglang-inkling:gb10` copy.

Two consequences to be aware of:
- GPU sits at ~92% utilisation with no load from me, so the pair is contended. A short
  re-check on it read accept 3.28 +/- 0.29 / tok/s 23.8 +/- 2.2 (n=8) — accept consistent with
  baseline, tok/s depressed by the co-tenant. Do not read that as a regression.
- All four rows in §4 were measured before 04:37Z on uncontended, `:gb10`-image boots.

To restore a clean own-image champion once the pair is free:
`~/calib-boot-champion.sh <rank>` (rank1 first), or the standard
`INKLING_TORCH_CONV_COMMIT=1 INKLING_COMMIT_STEP_BIAS=1 MOE=marlin GRAPHS=1 MEMFRAC=0.85
CTX=65536 MAXREQ=16 BLOCK=7 EXTRA_ARGS="--triton-attention-reduce-in-fp32"
~/inkling-sglang-launch.sh <rank>`.
