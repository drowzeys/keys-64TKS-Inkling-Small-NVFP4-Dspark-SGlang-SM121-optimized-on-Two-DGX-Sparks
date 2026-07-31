#!/usr/bin/env python3
"""Statistically-powered accept/tok-s probe.

Why: on this stack the target forward is nondeterministic at temp 0, and accept depends
heavily on WHICH continuation a run lands on (repetitive text drafts easily -> high accept;
novel prose -> low). A single 10-run probe on one seed cannot separate a config effect from
content noise. This runs multiple seeds x repeats and reports mean +/- stderr so two configs
can actually be compared.

Usage: accept_probe3.py [label] [--reps N] [--tokens N]
"""
import json, sys, time, statistics, urllib.request

URL = "http://10.100.10.1:30000"
SEEDS = [
    ("press", "The invention of the printing press in the fifteenth century transformed European society in ways that its creators could scarcely have imagined. Before Gutenberg, books were copied by hand, a slow and expensive process, and ideas travelled at the pace of a walking scribe"),
    ("litho", "Modern semiconductor manufacturing depends on photolithography, a process in which light is projected through a patterned mask onto a silicon wafer coated with photoresist. As feature sizes shrank below the wavelength of the light itself, engineers turned to"),
    ("river", "The Mississippi River drains thirty-one states and two Canadian provinces, carrying sediment from the continental interior toward the Gulf. Long before engineers built levees along its banks, the river routinely changed course, and the land it built"),
    ("raft",  "In distributed systems, consensus protocols such as Raft and Paxos exist to answer a deceptively simple question: how can a group of machines agree on a single value when any of them may fail at any moment, and messages between them"),
]

def gen(seed, max_new):
    body = {"text": seed, "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    req = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.perf_counter() - t0
    m = d["meta_info"]
    ct, st = m.get("completion_tokens"), m.get("spec_verify_ct")
    return (ct / st if st else None), (ct / dt if ct else None)

def main():
    label = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "config"
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 8
    toks = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 160
    for name, s in SEEDS[:2]:
        gen(s, 32)  # warm
    acc, tps, per_seed = [], [], {}
    for name, s in SEEDS:
        sa, st_ = [], []
        for _ in range(reps):
            a, t = gen(s, toks)
            if a: sa.append(a); acc.append(a)
            if t: st_.append(t); tps.append(t)
        per_seed[name] = (round(statistics.mean(sa), 2), round(statistics.mean(st_), 1))
        print(f"  seed {name:6s}: accept {per_seed[name][0]:5.2f}  tok/s {per_seed[name][1]:5.1f}  (n={len(sa)})", flush=True)
    n = len(acc)
    se_a = statistics.stdev(acc) / (n ** 0.5) if n > 1 else 0
    se_t = statistics.stdev(tps) / (n ** 0.5) if n > 1 else 0
    print(f"\n=== {label} (n={n}) ===")
    print(f"accept mean {statistics.mean(acc):.2f} +/- {se_a:.2f} (se)   range {min(acc):.2f}-{max(acc):.2f}")
    print(f"tok/s  mean {statistics.mean(tps):.1f} +/- {se_t:.1f} (se)   range {min(tps):.1f}-{max(tps):.1f}")

if __name__ == "__main__":
    main()
