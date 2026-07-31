#!/usr/bin/env python3
"""SPS cost-table calibration for DSpark cap-accept scheduling.

Stage 1 (record): serve with SGLANG_DSPARK_ENABLE_SPS_RECORD=1, run this with --drive
                  to push diverse traffic (varied prompt lens / batch sizes).
Stage 2 (fit):    --fit <dump.jsonl> <out.json>  -> groups by verify_len, median steps/sec,
                  emits the probes JSON that --speculative-dspark-sps-table-path consumes.
"""
import json, sys, time, statistics, urllib.request, concurrent.futures as cf

URL = "http://10.100.10.1:30000"
SEEDS = [
    "The invention of the printing press in the fifteenth century transformed European society in ways that its creators could scarcely have imagined. Before Gutenberg, books were copied by hand, and ideas travelled at the pace of a walking scribe",
    "Modern semiconductor manufacturing depends on photolithography, a process in which light is projected through a patterned mask onto a silicon wafer coated with photoresist. As feature sizes shrank below the wavelength of the light itself",
    "The Mississippi River drains thirty-one states and two Canadian provinces, carrying sediment from the continental interior toward the Gulf. Long before engineers built levees along its banks, the river routinely changed course",
    "In distributed systems, consensus protocols such as Raft and Paxos exist to answer a deceptively simple question: how can a group of machines agree on a single value when any of them may fail",
]

def gen(seed, max_new):
    body = {"text": seed, "sampling_params": {"temperature": 0, "max_new_tokens": max_new}}
    req = urllib.request.Request(URL + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

def drive():
    """Diverse traffic: varied lengths x concurrency so the recorder sees many verify widths."""
    print("driving traffic for SPS record ...")
    for conc in (1, 2, 4, 8, 16):
        for max_new in (64, 160, 320):
            with cf.ThreadPoolExecutor(conc) as ex:
                list(ex.map(lambda i: gen(SEEDS[i % len(SEEDS)] + f" (v{i})", max_new), range(conc)))
            print(f"  conc={conc} max_new={max_new} done", flush=True)
    print("done — collect the recorder dump from the serve container/host dump dir")

def fit(dump_path, out_path):
    """dump: jsonl with per-step records carrying verify_len + target_verify_gpu_time (seconds)."""
    by_len = {}
    with open(dump_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            vl = rec.get("verify_len") or rec.get("num_verify_tokens") or rec.get("batch_tokens")
            t = rec.get("target_verify_gpu_time") or rec.get("step_gpu_time") or rec.get("step_time")
            if not vl or not t or t <= 0:
                continue
            by_len.setdefault(int(vl), []).append(1.0 / float(t))  # steps per second
    probes = [[vl, round(statistics.median(sps), 4)] for vl, sps in sorted(by_len.items()) if len(sps) >= 3]
    if not probes:
        sys.exit("no usable records — check the dump's field names against dspark_sps.py")
    json.dump({"probes": probes}, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}: {len(probes)} probes, verify_len {probes[0][0]}..{probes[-1][0]}")

if __name__ == "__main__":
    if "--drive" in sys.argv:
        drive()
    elif "--fit" in sys.argv:
        i = sys.argv.index("--fit")
        fit(sys.argv[i + 1], sys.argv[i + 2])
    else:
        print(__doc__)
