"""Latency and throughput benchmark for the 2.0 scorer (features + 5 anomaly detectors + LightGBM + fusion + reasons).

    python -m src.benchmark_v2 --n 20000
"""
import argparse
import json
import platform
import time

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR
from src.scorer_v2 import FraudScorerV2
from src.stream_v2 import TXN_FIELDS
from src.train_v2 import dataset_path, slices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dataset", default="v2")
    a = ap.parse_args()
    df = pd.read_csv(dataset_path(a.dataset), parse_dates=["timestamp"])
    s = slices(df)
    sc = FraudScorerV2()
    hist = df[~s["test"]]
    sc.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist.is_fraud.to_numpy())
    recs = df[s["test"]][TXN_FIELDS].astype({"timestamp": str}).head(a.n + 500).to_dict("records")
    for r in recs[:500]:
        sc.score(r, explain=False)
    lat, flagged, st = [], [], {k: [] for k in ("features", "anomaly", "model_fusion", "explain")}
    t0 = time.perf_counter()
    for r in recs[500:]:
        o = sc.score(r)
        lat.append(o["latency_ms"]); flagged.append(o["decision"] != "ALLOW")
        for k, v in o["timings_ms"].items():
            st[k].append(v)
    wall = time.perf_counter() - t0
    lat = np.array(lat); flagged = np.array(flagged)
    out = {"n_transactions": int(len(lat)), "machine": f"{platform.machine()}, Python {platform.python_version()}, single thread",
           "latency_ms": {q: float(np.percentile(lat, p)) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))} | {"max": float(lat.max())},
           "latency_ms_flagged": {"p50": float(np.percentile(lat[flagged], 50)), "p99": float(np.percentile(lat[flagged], 99))},
           "stage_mean_ms": {k: float(np.mean(v)) for k, v in st.items()},
           "throughput_txn_per_sec_single_core": len(lat) / wall}
    edges = np.linspace(0, np.percentile(lat, 99.5), 41)
    out["histogram"] = {"edges_ms": edges.tolist(), "counts": np.histogram(lat, edges)[0].tolist()}
    (REPORTS_DIR / "v2").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v2" / "latency.json").write_text(json.dumps(out, indent=1))
    L = out["latency_ms"]
    print(f"{len(lat):,} txns | p50 {L['p50']:.3f} p95 {L['p95']:.3f} p99 {L['p99']:.3f} ms | "
          f"{out['throughput_txn_per_sec_single_core']:,.0f} txn/s | stages {({k: round(v, 3) for k, v in out['stage_mean_ms'].items()})}")


if __name__ == "__main__":
    main()
