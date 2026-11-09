"""Real/public-data experiments: does training on public datasets beat the synthetic-trained model, and does
combining datasets help?

Datasets (see src/adapters.py): sparkov (third-party simulated card data, CC0), ieee (real Vesta card-not-present
data), v2 (payTrace's own synthetic data). Every dataset keeps its OWN temporal split (train 0-50%, early stopping
50-60%, calibration 60-67%, thresholds 67-75%, test 75-100% of its own timeline), so no model ever sees its test
period, and pooled models are tested separately on each dataset's test period.

    python -m src.experiments_real            # writes reports/real/results.json and prints a summary

Experiments
  baseline_synthetic   the currently deployed v2 (synthetic-trained) system, applied as-is to each dataset
  single_<d>           full system trained on one dataset
  pooled_sp_ie         full system trained on sparkov + ieee
  pooled_all           full system trained on sparkov + ieee + v2
  transfer_<a>_to_<b>  trained on a, tested on b without retraining (thresholds from a)
  ieee_native          LightGBM on IEEE-CIS's own ~430 anonymised columns (+ our behavioural features): the
                       ceiling for that dataset. The web app cannot supply those columns, so it is a reference only.
"""
from __future__ import annotations

import gc
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.train_v2 import (FEATS, FUSION_INPUTS, LGB_PARAMS, V2_DIR, decisions, evaluate, fit_system, load,
                          save, score_frame, slices)

OUT = REPORTS_DIR / "real"
PATHS = {"sparkov": DATA_DIR / "transactions_sparkov.csv", "ieee": DATA_DIR / "transactions_ieee.csv", "v2": "v2"}


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(msg + "\n")


def summary(res):
    r, b = res["review_tier"], res["block_tier"]
    return {"pr_auc": round(res["ranking"]["pr_auc"], 4), "roc_auc": round(res["ranking"].get("roc_auc", float("nan")), 4),
            "review_recall": round(r["recall"], 4), "review_precision": round(r["precision"], 4),
            "review_fpr": round(r["fpr"], 5), "block_recall": round(b["recall"], 4), "block_precision": round(b["precision"], 4),
            "fraud_value_caught_share": round(1 - r["loss_inr"] / max(res["no_model_loss_inr"], 1), 4)}


def apply_system(comps, X):
    """Score a dataset with an already-fitted system (no refitting): used for baseline and transfer tests."""
    S = score_frame(comps, X)
    if comps.fusion is not None:
        S["fused"] = comps.fusion.predict_proba(comps.fusion_scaler.transform(S[FUSION_INPUTS]))[:, 1]
    return S


def pooled(names, data):
    dfs, Xs, masks = [], [], {}
    for n in names:
        df, X, s = data[n]
        dfs.append(df.assign(source=n)); Xs.append(X)
    df = pd.concat(dfs, ignore_index=True)
    X = pd.concat(Xs, ignore_index=True)
    s = {k: np.concatenate([data[n][2][k] for n in names]) for k in data[names[0]][2]}
    src = df.source.to_numpy()
    return df, X, s, {n: s["test"] & (src == n) for n in names}


def ieee_native(df_canon, X_canon, s):
    """LightGBM on IEEE-CIS's raw columns (+ our behavioural features) with the same temporal slices."""
    text = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)],
            "DeviceType", "DeviceInfo", *[f"id_{i}" for i in range(12, 39)]}
    f32 = lambda path: {c: ("object" if c in text else "float32")
                        for c in pd.read_csv(path, nrows=0).columns if c not in ("TransactionID", "isFraud")}
    raw = pd.read_csv(DATA_DIR / "raw/ieee/train_transaction.csv", dtype=f32(DATA_DIR / "raw/ieee/train_transaction.csv"))
    ident = pd.read_csv(DATA_DIR / "raw/ieee/train_identity.csv", dtype=f32(DATA_DIR / "raw/ieee/train_identity.csv"))
    raw = raw.merge(ident, on="TransactionID", how="left")
    raw["txn_id"] = "IE" + raw.TransactionID.astype(str)
    raw = raw.set_index("txn_id").loc[df_canon.txn_id]          # same row order as the canonical frame
    y = raw.pop("isFraud").to_numpy()
    raw = raw.drop(columns=["TransactionID", "TransactionDT"])
    for c in raw.columns:
        if raw[c].dtype == object:
            raw[c] = raw[c].astype("category")
        else:
            raw[c] = raw[c].astype("float32")
    out = {}
    for name, feats in [("raw_columns", raw), ("raw_plus_behaviour", pd.concat([raw.reset_index(drop=True),
                                                                             X_canon[FEATS].reset_index(drop=True)], axis=1))]:
        feats = feats.reset_index(drop=True)
        p = {**LGB_PARAMS, "num_threads": 2}
        m = lgb.train(p, lgb.Dataset(feats[s["train"]], y[s["train"]]), 3000,
                      valid_sets=[lgb.Dataset(feats[s["es"]], y[s["es"]])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pr = m.predict(feats[s["test"]], num_iteration=m.best_iteration)
        out[name] = {"pr_auc": round(float(average_precision_score(y[s["test"]], pr)), 4),
                     "roc_auc": round(float(roc_auc_score(y[s["test"]], pr)), 4), "n_features": feats.shape[1],
                     "best_iteration": int(m.best_iteration)}
        log("ieee_native", name, out[name])
        del m; gc.collect()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "log.txt").write_text("")
    results = {"note": "test = last 25% of each dataset's own timeline; amounts converted to INR at a fixed rate",
               "datasets": {}, "experiments": {}}
    data = {}
    for n, p in PATHS.items():
        t = time.time()
        df, X = load(str(p))
        s = slices(df)
        data[n] = (df, X, s)
        te = s["test"]
        results["datasets"][n] = {"rows": len(df), "customers": int(df.user_id.nunique()), "fraud_rate": round(float(df.is_fraud.mean()), 5),
                                  "test_rows": int(te.sum()), "test_frauds": int(df.is_fraud[te].sum()),
                                  "from": str(df.timestamp.min()), "to": str(df.timestamp.max())}
        log("loaded", n, results["datasets"][n], f"{time.time() - t:.0f}s")

    E = results["experiments"]
    base = joblib.load(V2_DIR / "components.joblib")
    for n in ("sparkov", "ieee"):
        df, X, s = data[n]
        E[f"baseline_synthetic_on_{n}"] = summary(evaluate(df, apply_system(base, X), base, s["test"]))
        log("baseline_synthetic_on", n, E[f"baseline_synthetic_on_{n}"])

    fitted = {}
    for n in ("sparkov", "ieee"):
        t = time.time()
        df, X, s = data[n]
        comps, S = fit_system(df, X, s)
        fitted[n] = comps
        E[f"single_{n}"] = summary(evaluate(df, S, comps, s["test"]))
        E[f"single_{n}"]["train_minutes"] = round((time.time() - t) / 60, 1)
        log("single", n, E[f"single_{n}"])
        save(comps, {"dataset": n, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}, out=MODELS_DIR / f"real_{n}")

    for a, b in (("sparkov", "ieee"), ("ieee", "sparkov")):
        df, X, s = data[b]
        E[f"transfer_{a}_to_{b}"] = summary(evaluate(df, apply_system(fitted[a], X), fitted[a], s["test"]))
        log("transfer", a, "->", b, E[f"transfer_{a}_to_{b}"])

    for name, names in (("pooled_sp_ie", ["sparkov", "ieee"]), ("pooled_all", ["sparkov", "ieee", "v2"])):
        t = time.time()
        df, X, s, tests = pooled(names, data)
        comps, S = fit_system(df, X, s)
        E[name] = {n: summary(evaluate(df, S, comps, m)) for n, m in tests.items()}
        E[name]["train_minutes"] = round((time.time() - t) / 60, 1)
        log("pooled", name, E[name])
        save(comps, {"dataset": "+".join(names), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")},
             out=MODELS_DIR / f"real_{name}")
        del df, X, S, comps; gc.collect()
        (OUT / "results.json").write_text(json.dumps(results, indent=1))

    df, X, s = data["ieee"]
    del data, fitted, base; gc.collect()          # free memory: the native run loads ~430 raw columns
    E["ieee_native"] = ieee_native(df, X, s)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    log("done")


def finish():
    """Re-score every saved model with the current decision rules (no retraining) and run the IEEE native ceiling.
    Used after the strict-tripwire fix: decisions changed, the trained models did not."""
    results = json.loads((OUT / "results.json").read_text())
    E = results["experiments"]
    data = {n: load(str(PATHS[n])) for n in ("sparkov", "ieee")}
    data = {n: (df, X, slices(df)) for n, (df, X) in data.items()}
    models = {"baseline_synthetic": V2_DIR, "single_sparkov": MODELS_DIR / "real_sparkov",
              "single_ieee": MODELS_DIR / "real_ieee", "pooled_sp_ie": MODELS_DIR / "real_pooled_sp_ie"}
    for name, d in models.items():
        comps = joblib.load(d / "components.joblib")
        for n, (df, X, s) in data.items():
            key = {"baseline_synthetic": f"baseline_synthetic_on_{n}", "pooled_sp_ie": "pooled_sp_ie"}.get(name)
            if name.startswith("single_"):
                key = name if name.endswith(n) else f"transfer_{name[7:]}_to_{n}"
            res = summary(evaluate(df, apply_system(comps, X), comps, s["test"]))
            if name == "pooled_sp_ie":
                E["pooled_sp_ie"][n] = res
            else:
                res["train_minutes"] = E.get(key, {}).get("train_minutes")
                E[key] = {k: v for k, v in res.items() if v is not None}
            log("rescored", name, "on", n, res)
        del comps; gc.collect()
    E.pop("pooled_all", None)
    results["skipped"] = {"pooled_all": "ran out of memory (8 GB workspace) on 2.7M rows; not retried because pooling two "
                                        "datasets already lowered Sparkov PR-AUC from 0.94 to 0.66 and it would put "
                                        "synthetic data back into training"}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    df, X, s = data["ieee"]
    del data; gc.collect()
    E["ieee_native"] = ieee_native(df, X, s)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    log("done")


if __name__ == "__main__":
    import sys
    finish() if "--finish" in sys.argv else main()
