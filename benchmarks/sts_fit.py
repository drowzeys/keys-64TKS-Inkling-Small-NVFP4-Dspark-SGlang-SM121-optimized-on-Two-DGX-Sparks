#!/usr/bin/env python3
"""Fit a DSpark STS confidence calibration (per-position temperature scaling).

Why: DSpark's confidence head emits raw logits per draft position. The cap-accept
planner turns those into accept probabilities. Uncalibrated, the planner mis-budgets
proposal width (measured: cap-accept LOSES to static block-7). Temperature scaling per
position makes sigmoid(logit / T_i) match the empirical accept rate, which is what the
planner actually needs.

Pipeline:
  1. serve with SGLANG_DSPARK_STS_COLLECT_PATH=/models-rw/sts/raw  (writes raw.<n>.pt shards)
     — plain static serving, NO --speculative-dspark-confidence-sts-path (identity temps required)
  2. drive diverse traffic (sps_calibrate.py --drive)
  3. python3 sts_fit.py /path/to/raw '<out.json>'
  4. serve cap-accept + --speculative-dspark-confidence-sts-path <out.json>

Usage: sts_fit.py <shard_stem_or_dir> <out.json>
"""
import sys, glob, json, math
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float, float]:
    """1-D temperature scaling by BCE minimisation. Returns (T, ece_before, ece_after)."""
    logits = logits.double()
    labels = labels.double()

    def ece(probs, labels, bins=15):
        e, n = 0.0, labels.numel()
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            m = (probs > lo) & (probs <= hi)
            if m.any():
                e += (m.double().sum() / n) * abs(probs[m].mean() - labels[m].mean())
        return float(e)

    before = ece(torch.sigmoid(logits), labels)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    lossfn = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossfn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp().item())
    after = ece(torch.sigmoid(logits / T), labels)
    return T, before, after


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    stem, out = sys.argv[1], sys.argv[2]
    shards = sorted(glob.glob(f"{stem}*.pt")) or sorted(glob.glob(f"{stem}/*.pt"))
    if not shards:
        sys.exit(f"no .pt shards found at {stem}")
    L, M = [], []
    for s in shards:
        d = torch.load(s, map_location="cpu")
        L.append(d["logits"])
        M.append(d["prefix_mask"])
    logits = torch.cat(L, 0)
    mask = torch.cat(M, 0)
    print(f"loaded {len(shards)} shards: {tuple(logits.shape)} samples x positions")

    temps, eb, ea = [], [], []
    for pos in range(logits.shape[1]):
        col_l, col_m = logits[:, pos], mask[:, pos]
        if col_m.numel() < 50 or col_m.min() == col_m.max():
            temps.append(1.0); eb.append(0.0); ea.append(0.0)
            print(f"  pos {pos}: degenerate -> T=1.0")
            continue
        T, b, a = fit_temperature(col_l, col_m)
        temps.append(round(T, 5)); eb.append(round(b, 5)); ea.append(round(a, 5))
        print(f"  pos {pos}: T={T:.4f}  ECE {b:.4f} -> {a:.4f}  accept_rate={col_m.mean():.3f}")

    cal = {"temperatures": temps, "dataset": "inkling-campaign coherent-prose mix",
           "num_samples": int(logits.shape[0]), "ece_before": eb, "ece_after": ea}
    json.dump(cal, open(out, "w"))
    print(f"\nwrote {out}\n  temperatures: {temps}")
    print("  serve with: --speculative-dspark-confidence-sts-path <that file> + SGLANG_RAGGED_VERIFY_MODE=cap-accept")


if __name__ == "__main__":
    main()
