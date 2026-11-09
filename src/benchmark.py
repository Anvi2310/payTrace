"""Latency analysis (a required DN-FN-01 deliverable).

Scores N unseen transactions one at a time, end to end (feature computation + both anomaly
models + LightGBM + calibration + decision + reason codes when flagged), and reports
p50 / p95 / p99 / max latency, per-stage breakdown and single-core throughput.

  python -m src.benchmark --n 30000
"""
import argparse
import json
import platform
import time

import numpy as np

from src.config import REPORTS_DIR
from src.scorer import FraudScorer
from src.stream import TXN_FIELDS, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    a = ap.parse_args()
    scorer = FraudScorer()
    hist, test = load_split(scorer)
    scorer.warmup(hist[TXN_FIELDS].to_dict("records"))
    recs = test[TXN_FIELDS].head(a.n).to_dict("records")
    for r in recs[:500]:                                   # JIT / cache warm-up
        scorer.score(dict(r), explain=False)
    recs = recs[500:]

    lat, stages, flagged = [], {k: [] for k in ("features", "anomaly_models", "lightgbm", "explain")}, []
    t0 = time.perf_counter()
    for r in recs:
        res = scorer.score(r, timings=True)
        lat.append(res["latency_ms"])
        flagged.append(res["decision"] != "ALLOW")
        for k, v in res["timings_ms"].items():
            stages[k].append(v)
    wall = time.perf_counter() - t0
    lat = np.array(lat); flagged = np.array(flagged)
    pct = lambda v, q: float(np.percentile(v, q))
    out = {
        "n_transactions": len(lat), "machine": f"{platform.processor() or platform.machine()}, "
                                              f"Python {platform.python_version()}, single thread",
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99),
                       "max": float(lat.max()), "mean": float(lat.mean())},
        "latency_ms_flagged_with_explanations": {"p50": pct(lat[flagged], 50), "p99": pct(lat[flagged], 99)}
        if flagged.any() else None,
        "stage_mean_ms": {k: float(np.mean(v)) for k, v in stages.items()},
        "throughput_txn_per_sec_single_core": len(lat) / wall,
        "histogram": {"edges_ms": np.linspace(0, pct(lat, 99.5), 41).tolist()},
    }
    out["histogram"]["counts"] = np.histogram(lat, out["histogram"]["edges_ms"])[0].tolist()
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "latency.json").write_text(json.dumps(out, indent=1))
    l = out["latency_ms"]
    print(f"{out['n_transactions']:,} txns | p50 {l['p50']:.3f} ms | p95 {l['p95']:.3f} ms | "
          f"p99 {l['p99']:.3f} ms | max {l['max']:.2f} ms | "
          f"{out['throughput_txn_per_sec_single_core']:,.0f} txn/s single core")
    print("stage means (ms):", {k: round(v, 4) for k, v in out["stage_mean_ms"].items()})


if __name__ == "__main__":
    main()
