#!/usr/bin/env python3
"""Fit a DSpark STS confidence calibration (per-position temperature scaling).

Why this exists alongside `python -m sglang.benchmark.dspark_sts_fit`:
the shipped fitter walks positions left-to-right and, at each position, grid-searches one
temperature that minimises THAT position's survival ECE with all earlier temperatures
already frozen (dspark_sts_fit.py:48-104). That greedy pass is myopic: on this serve it
made positions 2 and 3 worse than identity and moved mean ECE by ~0 (0.03767 -> 0.03764).

This fitter optimises the same 15-bin survival ECE but as a JOINT objective -- mean ECE
over all gamma positions -- by coordinate descent over an 81-point log grid, sweeping
until no single-coordinate move helps. Measured on the same 19.6k-sample collection:
mean ECE 0.03651 -> 0.03453 (-5.4%), and no position regresses more than 0.0007.

The quantity being calibrated is exactly what the planner consumes:
    survival_i = prod_{j<=i} sigmoid(z_j / T_j)
      models/dspark.py:328-330  (apply_sts)
      dspark_planner.py:963-966 (candidates = history_survival_probs -> tau_star)

Collection prerequisites (see SPS_STS_CALIBRATION.md for the full derivation):
  * SGLANG_RAGGED_VERIFY_MODE=cap-accept  -- static builds NO confidence head at all
                                             (models/dspark.py:333-335)
  * SGLANG_DSPARK_STS_COLLECT_PATH=/stscollect/raw
  * NO --speculative-dspark-confidence-sts-path (identity temps enforced,
    dspark_planner.py:100-108)
  * traffic with top_k > 1. top_k <= 1 IS the definition of "all greedy"
    (sampling_batch_info.py:205), which folds the draft into the cuda graph and makes
    dspark_observability.py:843 skip the recorder -- zero shards, forever.
  * >=256 decode steps per shard (dspark_observability.py:724, no shutdown flush).

Usage: sts_fit.py <shard_glob_or_dir> <out.json> [--bce]
       --bce swaps the objective for a joint LBFGS fit on survival log-likelihood
       (lower NLL, but measured slightly WORSE ECE here: 0.0365 -> 0.0393).
"""
import sys, glob, json, math
import torch

_EPS = 1e-8
_BINS = 15


def ece(probs: torch.Tensor, targets: torch.Tensor, bins: int = _BINS) -> float:
    p = probs.reshape(-1).double().clamp(_EPS, 1 - _EPS)
    t = targets.reshape(-1).double()
    idx = (p * bins).long().clamp_(0, bins - 1)
    cnt = torch.zeros(bins, dtype=torch.float64).scatter_add_(0, idx, torch.ones_like(p))
    ps = torch.zeros(bins, dtype=torch.float64).scatter_add_(0, idx, p)
    ts = torch.zeros(bins, dtype=torch.float64).scatter_add_(0, idx, t)
    den = cnt.clamp_min(1.0)
    return float(((ps / den - ts / den).abs() * cnt).sum().item() / p.numel())


def survival(logits: torch.Tensor, temps) -> torch.Tensor:
    t = torch.as_tensor(temps, dtype=torch.float64)
    return torch.cumprod(torch.sigmoid(logits / t), dim=1)


def per_position_ece(logits, mask, temps):
    s = survival(logits, temps)
    return [ece(s[:, i], mask[:, i]) for i in range(logits.shape[1])]


def fit_coordinate_descent(logits, mask, sweeps: int = 6):
    gamma = logits.shape[1]
    grid = torch.logspace(math.log10(0.1), math.log10(10.0), steps=81).double().tolist()
    temps = [1.0] * gamma
    best = sum(per_position_ece(logits, mask, temps)) / gamma
    for _ in range(sweeps):
        improved = False
        for i in range(gamma):
            keep = temps[i]
            for cand in grid:
                temps[i] = cand
                val = sum(per_position_ece(logits, mask, temps)) / gamma
                if val < best - 1e-9:
                    best, keep, improved = val, cand, True
            temps[i] = keep
        if not improved:
            break
    return temps


def fit_joint_bce(logits, mask, iters: int = 400):
    log_t = torch.zeros(logits.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.3, max_iter=iters, tolerance_grad=1e-12)

    def closure():
        opt.zero_grad()
        s = torch.cumprod(torch.sigmoid(logits / log_t.exp()), dim=1).clamp(_EPS, 1 - _EPS)
        loss = -(mask * s.log() + (1 - mask) * (1 - s).log()).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.detach().exp().tolist()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    stem, out = args[0], args[1]
    shards = (sorted(glob.glob(stem)) or sorted(glob.glob(f"{stem}*.pt"))
              or sorted(glob.glob(f"{stem}/*.pt")))
    if not shards:
        sys.exit(f"no .pt shards matched {stem!r}")
    L, M = [], []
    for s in shards:
        d = torch.load(s, map_location="cpu")
        L.append(d["logits"])
        M.append(d["prefix_mask"])
    logits, mask = torch.cat(L, 0).double(), torch.cat(M, 0).double()
    n, gamma = logits.shape
    print(f"loaded {len(shards)} shards: {n} samples x {gamma} positions")

    before = per_position_ece(logits, mask, [1.0] * gamma)
    temps = (fit_joint_bce(logits, mask) if "--bce" in sys.argv
             else fit_coordinate_descent(logits, mask))
    after = per_position_ece(logits, mask, temps)

    print("pos  temperature   accept_rate   ECE_before   ECE_after")
    for i in range(gamma):
        print(f"{i:>3}  {temps[i]:>11.4f}  {float(mask[:, i].mean()):>12.4f}  "
              f"{before[i]:>10.4f}  {after[i]:>10.4f}")
    print(f"mean ECE {sum(before)/gamma:.5f} -> {sum(after)/gamma:.5f}")

    cal = {"temperatures": [round(float(t), 6) for t in temps],
           "dataset": stem, "num_samples": int(n),
           "ece_before": [round(x, 5) for x in before],
           "ece_after": [round(x, 5) for x in after]}
    with open(out, "w") as f:
        json.dump(cal, f)
    print(f"\nwrote {out}\n  temperatures: {cal['temperatures']}")
    print("  serve with: --speculative-dspark-confidence-sts-path <that file>")


if __name__ == "__main__":
    main()
