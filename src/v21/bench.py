"""Phase J: latency and throughput of the app's live decision path (Sparkov model), on this machine.

    PAYTRACE_DB=/tmp/bench.db python -m src.v21.bench      # writes reports/v21/bench.json (never the real DB)

Measures, over N payments drawn from the dataset's unseen test period (real customers and merchants):
  scorer      features + 4 detectors + fusion + SHAP reasons (scorer.score)
  rings       online fraud-ring signals (RingMonitor.check + add)
  decision    the whole customer-payment path: limits, catch-up, scoring, rings, policy, explanation template,
              database writes (ENGINE.submit_payment, template explanations, no LLM)
p50 / p95 / p99 in milliseconds, and throughput as payments per second on one thread.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from src.config import REPORTS_DIR


def pct(a):
    a = np.asarray(a)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
            "mean": float(a.mean())}


def main(n: int = 600):
    assert "bench" in os.environ.get("PAYTRACE_DB", ""), "set PAYTRACE_DB to a scratch file"
    from webapp.engine import AppEngine
    eng = AppEngine()
    eng.start(background=False) if "background" in AppEngine.start.__code__.co_varnames else eng.start()
    while not eng.ready:
        time.sleep(0.5)
    rows = eng.replay.sample(n, random_state=3).to_dict("records")
    # 1. full decision path
    t_dec = []
    t0 = time.perf_counter()
    for r in rows:
        a = time.perf_counter()
        eng.submit_payment(user_id=r["user_id"], payee=r["merchant_id"], amount=float(r["amount"]), category=r["category"],
                           channel=r["channel"], device_id=f"BENCH_{r['user_id']}", city=r["city"], source="bench",
                           actor="bench", role="admin", explain_with_llm=False)
        t_dec.append((time.perf_counter() - a) * 1e3)
    wall = time.perf_counter() - t0
    # 2. components alone
    t_sc, t_rg = [], []
    for r in rows:
        txn = {"txn_id": "B", "timestamp": eng.now().strftime("%Y-%m-%d %H:%M:%S"), "user_id": r["user_id"], "amount": float(r["amount"]),
               "merchant_id": r["merchant_id"], "category": r["category"], "channel": r["channel"], "device_id": "UNKNOWN", "city": r["city"]}
        a = time.perf_counter(); eng.scorer.score(txn); t_sc.append((time.perf_counter() - a) * 1e3)
        a = time.perf_counter(); s = eng.rings.check(txn); eng.rings.add(txn, s); t_rg.append((time.perf_counter() - a) * 1e3)
    out = {"payments": n, "dataset": eng.dataset, "cpu_count": os.cpu_count(),
           "decision_path_ms": pct(t_dec), "decision_throughput_per_sec": n / wall,
           "scorer_ms": pct(t_sc), "ring_monitor_ms": pct(t_rg),
           "note": "single thread, includes SQLite writes; measured in the build container (2 vCPU), not a production server"}
    (REPORTS_DIR / "v21").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "v21" / "bench.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
