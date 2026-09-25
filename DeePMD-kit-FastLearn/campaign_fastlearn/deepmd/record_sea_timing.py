"""Consolidate the chained arms' per-block timings into one table.

Each chained arm already writes `timing.json` (per block: seconds, s/batch, return
code) next to its `segments.json`. This flattens those into the campaign's
`analysis/data/` so the SR-vs-LES cost is one readable table instead of eight
JSON blobs per arm, and so the LR premium is a column rather than mental arithmetic.

Re-run it as blocks land; it rewrites the whole file each time, so a partial chain
simply shows up with `complete=0` and the blocks it has.

Usage:
    python record_sea_timing.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(CAMPAIGN, "analysis", "data", "timing_sea.tsv")
ARMS = ["deepmd-cace-sched", "deepmd-les-cace-sched"]
COLS = ["arm", "replicate", "host", "device", "blocks_done", "blocks_total",
        "total_steps", "total_seconds", "s_per_batch", "lr_premium_vs_sr"]


def load(arm, rep):
    d = os.path.join(RUNS, f"{arm}_{rep}")
    p = os.path.join(d, "timing.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        t = json.load(fh)
    segs = os.path.join(d, "segments.json")
    n_total = len(json.load(open(segs))["segments"]) if os.path.exists(segs) else 8
    blocks = t.get("blocks") or []
    done = [b for b in blocks if b.get("ok")]
    steps = sum(b["num_steps"] for b in done)
    secs = sum(b["seconds"] for b in done)
    return {
        "arm": arm, "replicate": rep, "host": t.get("host", ""),
        "device": t.get("device", ""), "blocks_done": len(done),
        "blocks_total": n_total, "total_steps": steps, "total_seconds": secs,
        "s_per_batch": (secs / steps) if steps else 0.0,
    }


def main():
    rows = [r for r in (load(a, rep) for a in ARMS for rep in ("sA", "sB")) if r]
    sr = {r["replicate"]: r for r in rows if r["arm"] == "deepmd-cace-sched"}
    for r in rows:
        base = sr.get(r["replicate"])
        r["lr_premium_vs_sr"] = (r["s_per_batch"] / base["s_per_batch"]
                                 if base and base["s_per_batch"] else float("nan"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\t".join(COLS) + "\n")
        for r in sorted(rows, key=lambda x: (x["arm"], x["replicate"])):
            fh.write("\t".join([
                r["arm"], r["replicate"], r["host"], r["device"],
                str(r["blocks_done"]), str(r["blocks_total"]),
                str(r["total_steps"]), f"{r['total_seconds']:.2f}",
                f"{r['s_per_batch']:.6f}",
                "" if r["lr_premium_vs_sr"] != r["lr_premium_vs_sr"]
                else f"{r['lr_premium_vs_sr']:.4f}",
            ]) + "\n")
    print(f"wrote {OUT}: {len(rows)} arms")
    for r in sorted(rows, key=lambda x: (x["arm"], x["replicate"])):
        print(f"  {r['arm']:<22} {r['replicate']}  {r['blocks_done']}/{r['blocks_total']} blocks"
              f"  {r['total_seconds']/60:8.2f} min  {r['s_per_batch']*1000:7.3f} ms/batch"
              f"  LR premium {r['lr_premium_vs_sr']:.3f}x")


if __name__ == "__main__":
    main()
