"""payTrace 2.0 training pipeline (library + CLI).

Temporal slices (fractions of the dataset's time span, so it works for any dataset):
    train        0%  - 50%   fit supervised model, Isolation Forests, autoencoder
    early-stop  50%  - 60%   LightGBM/XGBoost/CatBoost early stopping only
    calibration 60%  - 67%   isotonic calibration + risk-fusion weights
    threshold   67%  - 75%   choose REVIEW / BLOCK thresholds and tripwire alert levels
    test        75%  - 100%  untouched; every reported number comes from here
For the 60-day synthetic data these are days 0-30 / 30-36 / 36-40 / 40-45 / 45-60.

Components
    supervised   LightGBM (default) | XGBoost | CatBoost on v2 features + Isolation Forest score
    iforest      Isolation Forest on all features (unsupervised)
    autoencoder  MLP autoencoder reconstruction error (unsupervised)
    graph_if     Isolation Forest on payee/device/money-flow graph features (novelty tripwire)
    fusion       logistic regression over component scores -> one calibrated, explainable risk score

Usage:  python -m src.train_v2 [--dataset v2|v1|path.csv] [--model lgbm|xgb|cat]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR
from src.evaluation import (at_threshold, calibration_bins, cost_threshold, pr_curve_points, precision_threshold,
                            ranking, roc_curve_points)
from src.features_v2 import V2_FEATURES, build_feature_table_v2

FEATS = FEATURE_COLUMNS + V2_FEATURES
# Specialised novelty detectors, one per entity type, so one entity's signal is not diluted by another's.
TRIPWIRES = {
    "payee": ["m_age_hours", "m_users_1h", "m_users_24h", "u_new_merchant", "m_new_payer_share_24h",
              "m_round_share_24h", "m_night_share_24h"],
    "device": ["d_n_users", "d_new_accounts_24h", "d_cnt_1h", "u_new_device", "u_n_devices", "u_age_hours"],
    "flow": ["u_passthrough", "u_in_cnt_1h", "u_in_amt_1h", "log_amount", "is_p2p", "u_cnt_1h"],
}
GRAPH_COLS = TRIPWIRES["payee"]          # kept for backwards compatibility
FUSION_INPUTS = ["sup_logit", "if_score", "ae_score", "payee_score", "device_score", "flow_score",
                 "m_conf_fraud_rate", "d_conf_fraud_rate"]
SLICES = {"train": (0.0, 0.50), "es": (0.50, 0.60), "cal": (0.60, 0.67), "thr": (0.67, 0.75), "test": (0.75, 1.0001)}
TRIPWIRE_RATE = 0.001            # each of the 4 tripwires may alert on at most 0.1% of traffic (label-free)
V2_DIR = MODELS_DIR / "v2"
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=5.0, verbose=-1, n_jobs=4)


def dataset_path(name: str) -> Path:
    return {"v1": DATA_DIR / "transactions.csv", "v2": DATA_DIR / "transactions_v2.csv"}.get(name, Path(name))


def load(name: str):
    path = dataset_path(name)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    cache = DATA_DIR / f"features_v2_{path.stem}.parquet"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        X = pd.read_parquet(cache)
        if len(X) == len(df):
            return df, X
    t = time.time()
    X, _ = build_feature_table_v2(df)
    print(f"built {len(X):,} x {X.shape[1]} features in {time.time() - t:.1f}s")
    X.to_parquet(cache)
    return df, X


def slices(df) -> dict[str, np.ndarray]:
    t = (df.timestamp - df.timestamp.min()).dt.total_seconds().to_numpy()
    frac = t / max(t.max(), 1)
    return {k: (frac >= a) & (frac < b) for k, (a, b) in SLICES.items()}


def trip_matrix(X, name):
    """All tripwire inputs are non-negative; log1p evens out heavy tails before random splits."""
    return np.log1p(np.clip(X[TRIPWIRES[name]].to_numpy(float), 0, None))


def ae_transform(X):
    """Signed log1p squashes heavy tails (amounts, counts) before scaling."""
    v = X[FEATS].to_numpy(float)
    return np.sign(v) * np.log1p(np.abs(v))


# ---------------- component fitting ----------------
@dataclass
class Components:
    iforest: IsolationForest
    trip_if: dict
    ae_scaler: StandardScaler
    ae: MLPRegressor
    supervised: object
    kind: str
    calibrator: IsotonicRegression
    fusion: LogisticRegression | None = None
    fusion_scaler: StandardScaler | None = None
    thresholds: dict = field(default_factory=dict)
    best_iteration: int | None = None
    train_seconds: float = 0.0


def fit_unsupervised(X, train_mask, seed=7):
    samp = X[train_mask].sample(min(60000, int(train_mask.sum())), random_state=seed)
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=4).fit(samp[FEATS].to_numpy())
    giso = {k: IsolationForest(n_estimators=100, max_samples=256, random_state=seed, n_jobs=4).fit(trip_matrix(samp, k))
            for k in TRIPWIRES}
    sc = StandardScaler().fit(ae_transform(samp))
    ae = MLPRegressor(hidden_layer_sizes=(24, 8, 24), max_iter=60, random_state=seed, early_stopping=True)
    ae.fit(sc.transform(ae_transform(samp)), sc.transform(ae_transform(samp)))
    return iso, giso, sc, ae


def unsupervised_scores(X, iso, giso, sc, ae):
    z = sc.transform(ae_transform(X))
    out = {"if_score": -iso.score_samples(X[FEATS].to_numpy()),
           "ae_score": np.log1p(((ae.predict(z) - z) ** 2).mean(1))}
    for k, m in giso.items():
        out[f"{k}_score"] = -m.score_samples(trip_matrix(X, k))
    return pd.DataFrame(out, index=X.index)


def fit_supervised(kind, Xtr, ytr, Xes, yes, params=None, rounds=None):
    t = time.time()
    if kind == "lgbm":
        p = {**LGB_PARAMS, **(params or {})}
        if rounds:
            m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=rounds); best = rounds
        else:
            m = lgb.train(p, lgb.Dataset(Xtr, ytr), 2000, valid_sets=[lgb.Dataset(Xes, yes)],
                          feval=lambda pr, d: ("ap", average_precision_score(d.get_label(), pr), True),
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            best = m.best_iteration
    elif kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=rounds or 2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                              colsample_bytree=0.8, scale_pos_weight=5.0, eval_metric="aucpr", tree_method="hist",
                              early_stopping_rounds=None if rounds else 100, n_jobs=4)
        m.fit(Xtr, ytr, **({} if rounds else {"eval_set": [(Xes, yes)], "verbose": False}))
        best = rounds or int(m.best_iteration)
    elif kind == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=rounds or 2000, learning_rate=0.08, depth=6, scale_pos_weight=5.0,
                               eval_metric="PRAUC", verbose=False, thread_count=4, random_seed=7,
                               od_type=None if rounds else "Iter", od_wait=None if rounds else 100)
        m.fit(Xtr, ytr, **({} if rounds else {"eval_set": (Xes, yes), "use_best_model": True}))
        best = rounds or int(m.get_best_iteration())
    else:
        raise ValueError(kind)
    return m, best, time.time() - t


def predict_raw(kind, model, X):
    return model.predict(X) if kind == "lgbm" else model.predict_proba(X)[:, 1]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def fit_system(df, X, s, kind="lgbm", feats=None, exclude=None, extra_train=None, rounds=None, params=None,
               unsup=None, use_fusion=True):
    """Fit every component on the proper slices. `exclude` drops rows (e.g. a held-out fraud pattern) from all
    fitting slices; `extra_train` adds rows (e.g. analyst-labelled feedback) to the supervised training set."""
    feats = feats or FEATS
    y = df.is_fraud.to_numpy()
    keep = ~exclude if exclude is not None else np.ones(len(df), bool)
    iso, giso, sc, ae = unsup or fit_unsupervised(X, s["train"])
    U = unsupervised_scores(X, iso, giso, sc, ae)
    XX = pd.concat([X[feats], U[["if_score"]]], axis=1)
    tr = s["train"] & keep
    if extra_train is not None:
        tr = tr | extra_train
    es = s["es"] & keep
    model, best, secs = fit_supervised(kind, XX[tr], y[tr], XX[es], y[es], params=params, rounds=rounds)
    raw = predict_raw(kind, model, XX)
    cal = s["cal"] & keep
    calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[cal], y[cal])
    comps = Components(iso, giso, sc, ae, model, kind, calib, best_iteration=best, train_seconds=secs)
    S = score_frame(comps, X, XX, U, raw)
    if use_fusion:
        fz = StandardScaler().fit(S.loc[cal, FUSION_INPUTS])
        comps.fusion = LogisticRegression(C=1.0, max_iter=1000).fit(fz.transform(S.loc[cal, FUSION_INPUTS]), y[cal])
        comps.fusion_scaler = fz
        S["fused"] = comps.fusion.predict_proba(fz.transform(S[FUSION_INPUTS]))[:, 1]
    thr = s["thr"] & keep
    risk = S["fused"] if use_fusion else S["sup_prob"]
    t_review, grid, costs = cost_threshold(y[thr], risk[thr].to_numpy(), df.amount.to_numpy()[thr])
    t_block = precision_threshold(y[thr], risk[thr].to_numpy(), 0.95, floor=t_review)
    t_sup = cost_threshold(y[thr], S.loc[thr, "sup_prob"].to_numpy(), df.amount.to_numpy()[thr])[0]
    comps.thresholds = {
        "review": t_review, "block": t_block, "sup_review": t_sup,
        "cost_grid": grid[::10].tolist(), "cost_curve": costs[::10].tolist(),
    }
    # label-free tripwires: the score that only TRIPWIRE_RATE of threshold-slice traffic exceeds
    for k in list(TRIPWIRES) + ["ae"]:
        comps.thresholds[f"{k}_trip"] = float(np.quantile(S.loc[s["thr"], f"{k}_score"], 1 - TRIPWIRE_RATE))
    return comps, S


def score_frame(comps, X, XX=None, U=None, raw=None):
    U = U if U is not None else unsupervised_scores(X, comps.iforest, comps.trip_if, comps.ae_scaler, comps.ae)
    if XX is None:
        XX = pd.concat([X[[c for c in FEATS if c in X]], U[["if_score"]]], axis=1)
    raw = raw if raw is not None else predict_raw(comps.kind, comps.supervised, XX)
    S = U.copy()
    S["sup_raw"] = raw
    S["sup_prob"] = comps.calibrator.predict(raw)
    S["sup_logit"] = logit(S["sup_prob"].to_numpy())
    S["m_conf_fraud_rate"] = X["m_conf_fraud_rate"].to_numpy()
    S["d_conf_fraud_rate"] = X["d_conf_fraud_rate"].to_numpy()
    return S


def decisions(comps, S, use_fusion=True):
    risk = S["fused"] if use_fusion and "fused" in S else S["sup_prob"]
    t = comps.thresholds
    trip = np.zeros(len(S), bool)
    for k in list(TRIPWIRES) + ["ae"]:
        trip |= (S[f"{k}_score"] > t[f"{k}_trip"]).to_numpy()   # strict ">" so a constant score never fires
    trip = pd.Series(trip, index=S.index)
    block = risk >= t["block"]
    # REVIEW if the fused score OR the supervised model alone OR any novelty tripwire says so: the fusion layer is
    # fitted on known fraud and can down-weight anomaly evidence, so it must not veto the other signals.
    review = (risk >= t["review"]) | (S["sup_prob"] >= t.get("sup_review", 2.0)) | trip
    return np.where(block, "BLOCK", np.where(review, "REVIEW", "ALLOW")), risk.to_numpy(), trip.to_numpy()


def fusion_weights(comps):
    if comps.fusion is None:
        return {}
    return {k: float(v) for k, v in zip(FUSION_INPUTS, comps.fusion.coef_[0])}


def evaluate(df, S, comps, mask, use_fusion=True):
    y = df.is_fraud.to_numpy()[mask]; amt = df.amount.to_numpy()[mask]
    dec, risk, trip = decisions(comps, S, use_fusion)
    dec, risk = dec[mask], risk[mask]
    days = (df.timestamp[mask].max() - df.timestamp[mask].min()).total_seconds() / 86400
    return {"ranking": ranking(y, risk), "review_tier": at_threshold(y, dec != "ALLOW", amt, days),
            "block_tier": at_threshold(y, dec == "BLOCK", amt, days),
            "no_model_loss_inr": float(amt[y == 1].sum())}


def save(comps, meta, out=V2_DIR):
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(comps, out / "components.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--model", default="lgbm", choices=["lgbm", "xgb", "cat"])
    a = ap.parse_args()
    df, X = load(a.dataset)
    s = slices(df)
    comps, S = fit_system(df, X, s, kind=a.model)
    res = evaluate(df, S, comps, s["test"])
    y = df.is_fraud.to_numpy(); te = s["test"]
    dec, risk, trip = decisions(comps, S)
    ft = df.fraud_type.to_numpy()
    by_pattern = {k: float((dec[te & (ft == k)] != "ALLOW").mean()) for k in sorted(set(ft[te])) if k != "LEGIT"}
    report = {
        "dataset": a.dataset, "model": a.model, "slices": {k: int(v.sum()) for k, v in s.items()},
        "thresholds": {k: v for k, v in comps.thresholds.items() if not k.startswith("cost")},
        "fusion_weights": fusion_weights(comps), "test": res, "recall_by_pattern": by_pattern,
        "pr_curve": pr_curve_points(y[te], risk[te]), "roc_curve": roc_curve_points(y[te], risk[te]),
        "calibration": {"supervised": calibration_bins(y[te], S["sup_prob"].to_numpy()[te]),
                        "fused": calibration_bins(y[te], risk[te])},
        "cost_curve": {"threshold": comps.thresholds["cost_grid"], "cost_inr": comps.thresholds["cost_curve"]},
        "test_start": str(df.timestamp[te].min()), "best_iteration": comps.best_iteration,
    }
    out = REPORTS_DIR / "v2"; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=1))
    save(comps, {"dataset": a.dataset, "model": a.model, "test_start": report["test_start"],
                 "thresholds": report["thresholds"], "fusion_inputs": FUSION_INPUTS, "features": FEATS,
                 "tripwires": TRIPWIRES, "fusion_weights": report["fusion_weights"],
                 "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    r = res["review_tier"]
    print(f"[{a.dataset}/{a.model}] test PR-AUC {res['ranking']['pr_auc']:.3f} | review P {r['precision']:.3f} "
          f"R {r['recall']:.3f} FPR {r['fpr']:.4f} | block P {res['block_tier']['precision']:.3f} "
          f"R {res['block_tier']['recall']:.3f} | loss {r['loss_inr']:,.0f} vs no-model {res['no_model_loss_inr']:,.0f}")
    print("recall by pattern:", {k: round(v, 3) for k, v in by_pattern.items()})
    print("fusion weights:", {k: round(v, 2) for k, v in report["fusion_weights"].items()})


if __name__ == "__main__":
    # run through the importable module so pickled classes are saved as src.train_v2.*, not __main__.*
    from src.train_v2 import main as _main
    _main()
