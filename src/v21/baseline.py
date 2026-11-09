"""Phase A: reproduce the 2.0 baseline from the SAVED model artifacts (no retraining).

    python -m src.v21.baseline      # writes reports/v21/baseline.json

Re-scores each saved model on its dataset's untouched test period (last 25% of the timeline) and compares the
numbers with reports/real/results.json and models/ieee_native/meta.json.
"""
from __future__ import annotations

import gc
import json
import time

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import MODELS_DIR, REPORTS_DIR
from src.train_v2 import evaluate, load, score_frame, slices

OUT = REPORTS_DIR / "v21"


def sparkov():
    df, X = load(str(__import__("src.config", fromlist=["DATA_DIR"]).DATA_DIR / "transactions_sparkov.csv"))
    s = slices(df)
    comps = joblib.load(MODELS_DIR / "real_sparkov" / "components.joblib")
    t0 = time.time()
    S = score_frame(comps, X)
    from src.train_v2 import FUSION_INPUTS
    S["fused"] = comps.fusion.predict_proba(comps.fusion_scaler.transform(S[FUSION_INPUTS]))[:, 1]
    secs = time.time() - t0
    y = df.is_fraud.to_numpy()
    te = s["test"]
    ev = evaluate(df, S, comps, te)
    out = {"pr_auc_fused": float(average_precision_score(y[te], S["fused"][te])),
           "roc_auc_fused": float(roc_auc_score(y[te], S["fused"][te])),
           "pr_auc_supervised": float(average_precision_score(y[te], S["sup_prob"][te])),
           "test_rows": int(te.sum()), "test_frauds": int(y[te].sum()), "batch_score_seconds": round(secs, 1),
           "decisions": ev}
    return out


def ieee_native():
    import lightgbm as lgb
    import pandas as pd
    from src.train_ieee_native import prepare, read_raw
    from src.train_v2 import SLICES
    meta = json.loads((MODELS_DIR / "ieee_native" / "meta.json").read_text())
    df = read_raw()
    frac = (df.TransactionDT - df.TransactionDT.min()) / (df.TransactionDT.max() - df.TransactionDT.min())
    a, b = SLICES["test"]
    te = ((frac >= a) & (frac < b)).to_numpy()
    df = df[te].reset_index(drop=True); gc.collect()
    X, _ = prepare(df, meta["categories"])
    y = df.isFraud.to_numpy()
    p = lgb.Booster(model_file=str(MODELS_DIR / "ieee_native" / "model.txt")).predict(X[meta["columns"]])
    t = meta["thresholds"]
    rev, blk = p >= t["review"], p >= t["block"]
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "review_recall": float(rev[y == 1].mean()), "review_fpr": float(rev[y == 0].mean()),
            "block_recall": float(blk[y == 1].mean()), "block_precision": float(y[blk].mean()) if blk.any() else None,
            "test_rows": int(len(y)), "test_frauds": int(y.sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((REPORTS_DIR / "real" / "results.json").read_text())["experiments"]
    native_meta = json.loads((MODELS_DIR / "ieee_native" / "meta.json").read_text())["metrics"]
    res = {"sparkov_app_model": sparkov()}
    gc.collect()
    res["ieee_native"] = ieee_native()
    res["recorded"] = {"single_sparkov": recorded["single_sparkov"], "ieee_native": native_meta,
                       "table": {k: v.get("pr_auc") for k, v in recorded.items()}}
    (OUT / "baseline.json").write_text(json.dumps(res, indent=1, default=float))
    sp, ie = res["sparkov_app_model"], res["ieee_native"]
    print(f"Sparkov app model: PR-AUC {sp['pr_auc_fused']:.4f} (recorded {recorded['single_sparkov']['pr_auc']:.4f})")
    print(f"IEEE native:       PR-AUC {ie['pr_auc']:.4f} (recorded {native_meta['pr_auc']:.4f})")


if __name__ == "__main__":
    main()
