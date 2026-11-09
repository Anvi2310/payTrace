"""Train and save the IEEE-CIS "native" model used by the admin Real-data replay.

IEEE-CIS (Kaggle, Vesta) is real card-not-present payment data with ~430 anonymised columns. A web form cannot supply
those columns, so this model is not used for typed-in payments; instead the admin can replay the dataset's UNSEEN test
period through it and watch real fraud being caught or missed against the true labels.

Splits follow the rest of payTrace (by time): train 0-50%, early stopping 50-60%, threshold choice 67-75%,
replay/test 75-100%. Nothing from the test period is used for training or thresholds.

    python -m src.train_ieee_native        # ~5 min; writes models/ieee_native/ and data/ieee_replay.parquet
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.adapters import INR_PER_USD
from src.config import DATA_DIR, MODELS_DIR
from src.evaluation import at_threshold, cost_threshold, precision_threshold
from src.train_v2 import LGB_PARAMS, SLICES

OUT = MODELS_DIR / "ieee_native"
REPLAY = DATA_DIR / "ieee_replay.parquet"
RAW = DATA_DIR / "raw" / "ieee"
TEXT = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)],
        "DeviceType", "DeviceInfo", *[f"id_{i}" for i in range(12, 39)]}
SHOW = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "DeviceInfo"]


def read_raw() -> pd.DataFrame:
    def dtypes(path):
        return {c: ("object" if c in TEXT else "float32") for c in pd.read_csv(path, nrows=0).columns
                if c not in ("TransactionID", "isFraud", "TransactionDT")}
    tr = pd.read_csv(RAW / "train_transaction.csv", dtype=dtypes(RAW / "train_transaction.csv"))
    idn = pd.read_csv(RAW / "train_identity.csv", dtype=dtypes(RAW / "train_identity.csv"))
    return tr.merge(idn, on="TransactionID", how="left").sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def prepare(df: pd.DataFrame, categories: dict[str, list] | None = None):
    """Model matrix: numeric columns as float32, text columns as pandas categoricals with a FIXED category list so
    training and replay encode identically."""
    feats = [c for c in df.columns if c not in ("TransactionID", "isFraud", "TransactionDT")]
    X = df[feats].copy()
    categories = categories or {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in feats if c in TEXT}
    for c, cats in categories.items():
        X[c] = pd.Categorical(X[c].astype("object").where(X[c].notna(), None), categories=cats)
    return X, categories


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = read_raw()
    y = df["isFraud"].to_numpy()
    frac = (df.TransactionDT - df.TransactionDT.min()) / (df.TransactionDT.max() - df.TransactionDT.min())
    s = {k: ((frac >= a) & (frac < b)).to_numpy() for k, (a, b) in SLICES.items()}
    te = s["test"]
    rep = df[te].copy()                                          # replay rows, kept aside as raw values
    amt = (df.TransactionAmt * INR_PER_USD).to_numpy()
    dt = df.TransactionDT.to_numpy()
    X, cats = prepare(df)
    del df
    import gc; gc.collect()
    print(f"{len(X):,} rows x {X.shape[1]} columns, read in {time.time() - t0:.0f}s", flush=True)
    m = lgb.train({**LGB_PARAMS, "num_threads": 2}, lgb.Dataset(X[s["train"]], y[s["train"]]), 4000,
                  valid_sets=[lgb.Dataset(X[s["es"]], y[s["es"]])], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(X, num_iteration=m.best_iteration)
    thr = s["thr"]
    t_review = float(cost_threshold(y[thr], p[thr], amt[thr])[0])
    t_block = float(precision_threshold(y[thr], p[thr], 0.95, floor=t_review))
    days = (dt[te].max() - dt[te].min()) / 86400
    review = at_threshold(y[te], p[te] >= t_review, amt[te], days)
    block = at_threshold(y[te], p[te] >= t_block, amt[te], days)
    metrics = {"pr_auc": float(average_precision_score(y[te], p[te])), "roc_auc": float(roc_auc_score(y[te], p[te])),
               "review": review, "block": block, "test_rows": int(te.sum()), "test_frauds": int(y[te].sum())}
    m.save_model(str(OUT / "model.txt"), num_iteration=m.best_iteration)
    meta = {"columns": list(X.columns), "categories": cats, "thresholds": {"review": t_review, "block": t_block},
            "best_iteration": int(m.best_iteration), "metrics": metrics, "inr_per_usd": INR_PER_USD,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "Real Vesta card-not-present payments (Kaggle IEEE-CIS). Columns are anonymised by the data owner."}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    rep["txn_id"] = "IE" + rep.TransactionID.astype(int).astype(str)
    rep["timestamp"] = (pd.Timestamp("2017-12-01") + pd.to_timedelta(rep.TransactionDT, unit="s")).astype(str)
    for c in TEXT & set(rep.columns):
        rep[c] = rep[c].astype("object").where(rep[c].notna(), None)
    rep.drop(columns=["TransactionDT"]).to_parquet(REPLAY, index=False)
    print(f"test PR-AUC {metrics['pr_auc']:.3f} ROC-AUC {metrics['roc_auc']:.3f} | review recall {review['recall']:.3f} "
          f"precision {review['precision']:.3f} FPR {review['fpr']:.4f} | block recall {block['recall']:.3f} precision "
          f"{block['precision']:.3f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
