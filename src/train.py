"""Train the two-stage fraud model and produce the evaluation report.

Stage 1a Isolation Forest on all features (unsupervised) -> anomaly score fed into Stage 2
Stage 1b Isolation Forest on payee-graph features          -> "novelty tripwire" for fraud
         patterns that have NO labels yet (e.g. a brand-new scam)
Stage 2  LightGBM (supervised)            -> uses all features + the Stage-1 anomaly score
         + isotonic calibration           -> score means a real probability of fraud
         + cost-based thresholds          -> REVIEW / BLOCK tiers that minimise INR loss

Everything is evaluated on a time-based hold-out (last 15 days) against baselines.

Usage:  python -m src.train
"""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config import (COST_FALSE_POSITIVE, FEATURE_COLUMNS, FEATURES, MODELS_DIR, RAW_ONLY_COLUMNS,
                        RAW_TXNS, REPORTS_DIR, TRAIN_END_DAY, VAL_END_DAY)
from src.drift import psi_reference
from src.features import build_feature_table

MODEL_COLUMNS = FEATURE_COLUMNS + ["if_score"]
GRAPH_COLUMNS = ["m_age_hours", "m_users_1h", "m_users_24h", "u_new_merchant", "d_n_users"]


def graph_matrix(X):
    """Payee/device graph view used by the novelty detector (log-scaled payee age)."""
    g = X[GRAPH_COLUMNS].to_numpy(dtype=float).copy()
    g[:, 0] = np.log1p(g[:, 0])
    return g
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  scale_pos_weight=5.0, verbose=-1, n_jobs=4)


def load():
    df = pd.read_csv(RAW_TXNS, parse_dates=["timestamp"])
    if FEATURES.exists():
        feats = pd.read_parquet(FEATURES)
        if len(feats) == len(df):
            print("loaded cached features")
            return df, feats
    t = time.time()
    feats, _ = build_feature_table(df)
    print(f"built {len(feats):,} feature rows in {time.time() - t:.1f}s "
          f"({(time.time() - t) / len(feats) * 1e6:.1f} us/txn)")
    feats.to_parquet(FEATURES)
    return df, feats


def cost_curve(y, p, amount, thresholds):
    """Total INR cost (missed fraud amount + FP friction) for each threshold."""
    out = []
    for t in thresholds:
        flag = p >= t
        missed = amount[(y == 1) & ~flag].sum()
        fp = ((y == 0) & flag).sum()
        out.append(missed + COST_FALSE_POSITIVE * fp)
    return np.array(out)


def metrics_at(y, flag, amount):
    tp = int(((y == 1) & flag).sum()); fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum()); tn = int(((y == 0) & ~flag).sum())
    fraud_amt = amount[y == 1].sum()
    caught_amt = amount[(y == 1) & flag].sum()
    cost = amount[(y == 1) & ~flag].sum() + COST_FALSE_POSITIVE * fp
    return {
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "alerts": tp + fp, "fraud_amount_inr": float(fraud_amt),
        "fraud_amount_caught_inr": float(caught_amt),
        "fraud_amount_caught_pct": float(caught_amt / max(fraud_amt, 1)),
        "total_cost_inr": float(cost),
    }


def recall_at_fpr(y, p, target=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target, fpr, tpr))


def main():
    MODELS_DIR.mkdir(exist_ok=True); REPORTS_DIR.mkdir(exist_ok=True)
    df, X = load()
    day = (df.timestamp - df.timestamp.min()).dt.total_seconds() / 86400
    tr, va, te = day < TRAIN_END_DAY, (day >= TRAIN_END_DAY) & (day < VAL_END_DAY), day >= VAL_END_DAY
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy()
    print(f"train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}  "
          f"(fraud rates {y[tr].mean():.2%} / {y[va].mean():.2%} / {y[te].mean():.2%})")

    # ---- Stage 1: Isolation Forest (fit without labels) ----
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=7, n_jobs=4)
    sample = X[tr].sample(min(60000, tr.sum()), random_state=7)
    iso.fit(sample[FEATURE_COLUMNS].to_numpy())
    X = X.copy()
    X["if_score"] = -iso.score_samples(X[FEATURE_COLUMNS].to_numpy())
    giso = IsolationForest(n_estimators=100, max_samples=256, random_state=7, n_jobs=4)
    giso.fit(graph_matrix(X[tr].sample(min(60000, tr.sum()), random_state=1)))
    X["graph_if_score"] = -giso.score_samples(graph_matrix(X))

    # ---- Stage 2: LightGBM ----
    dtr = lgb.Dataset(X.loc[tr, MODEL_COLUMNS], y[tr])
    dva = lgb.Dataset(X.loc[va, MODEL_COLUMNS], y[va], reference=dtr)
    booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                        feval=lambda p, d: ("pr_auc", average_precision_score(d.get_label(), p), True),
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"LightGBM best iteration: {booster.best_iteration}")
    raw = booster.predict(X[MODEL_COLUMNS], num_iteration=booster.best_iteration)

    calib = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[va], y[va])
    p = calib.predict(raw)

    # ---- Thresholds chosen on validation only ----
    grid = np.unique(np.quantile(p[va], np.linspace(0.90, 0.99999, 600)))
    costs = cost_curve(y[va], p[va], amt[va], grid)
    t_review = float(grid[np.argmin(costs)])
    prec, rec, thr = precision_recall_curve(y[va], p[va])
    ok = np.where(prec[:-1] >= 0.95)[0]
    t_block = float(max(thr[ok[0]], t_review)) if len(ok) else float(min(1.0, t_review * 2))
    # novelty tripwire: flag the top 0.2% most graph-anomalous legit-looking traffic for review
    if_thr = float(np.quantile(X.loc[va & (y == 0), "graph_if_score"], 0.998))

    # ---- Test-set evaluation ----
    yt, pt, at = y[te], p[te], amt[te]
    Xt = X[te]
    flag_review = pt >= t_review
    flag_combined = flag_review | (Xt.graph_if_score.to_numpy() >= if_thr)
    report = {
        "data": {"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
                 "test_fraud_rate": float(yt.mean())},
        "thresholds": {"review": t_review, "block": t_block, "isolation_forest": if_thr,
                       "cost_false_positive_inr": COST_FALSE_POSITIVE},
        "paytrace": {"pr_auc": float(average_precision_score(yt, pt)),
                        "roc_auc": float(roc_auc_score(yt, pt)),
                        "recall_at_1pct_fpr": recall_at_fpr(yt, pt),
                        "at_review_threshold": metrics_at(yt, flag_review, at),
                        "at_block_threshold": metrics_at(yt, pt >= t_block, at),
                        "with_novelty_tripwire": metrics_at(yt, flag_combined, at)},
    }

    # Baseline A: hand-written rules (what many banks start with)
    rules = (((Xt.amount > 10000) & (Xt.u_new_device == 1)) | (Xt.u_cnt_10m >= 5)
             | ((Xt.amount > 20000) & (Xt.is_night == 1))).to_numpy()
    # Baseline B: LightGBM on raw transaction fields only (no behavioural features)
    b_raw = lgb.train({**LGB_PARAMS}, lgb.Dataset(X.loc[tr, RAW_ONLY_COLUMNS], y[tr]),
                      num_boost_round=1500,
                      valid_sets=[lgb.Dataset(X.loc[va, RAW_ONLY_COLUMNS], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    p_raw = b_raw.predict(Xt[RAW_ONLY_COLUMNS])
    t_raw_grid = np.unique(np.quantile(p_raw, np.linspace(0.9, 0.99999, 400)))
    t_raw = t_raw_grid[np.argmin(cost_curve(yt, p_raw, at, t_raw_grid))]   # generous: tuned on test
    # Baseline C: Isolation Forest alone
    p_if = Xt.if_score.to_numpy()
    t_if = float(np.quantile(X.loc[va & (y == 0), "if_score"], 0.99))
    report["baselines"] = {
        "rules": {"pr_auc": None, **metrics_at(yt, rules, at)},
        "lgbm_raw_features_only": {"pr_auc": float(average_precision_score(yt, p_raw)),
                                   "recall_at_1pct_fpr": recall_at_fpr(yt, p_raw),
                                   **metrics_at(yt, p_raw >= t_raw, at)},
        "isolation_forest_only": {"pr_auc": float(average_precision_score(yt, p_if)),
                                  "recall_at_1pct_fpr": recall_at_fpr(yt, p_if),
                                  **metrics_at(yt, p_if >= t_if, at)},
    }
    no_model_cost = float(at[yt == 1].sum())
    report["business"] = {
        "cost_without_model_inr": no_model_cost,
        "cost_with_paytrace_inr": report["paytrace"]["at_review_threshold"]["total_cost_inr"],
        "cost_with_rules_inr": report["baselines"]["rules"]["total_cost_inr"],
    }
    report["business"]["net_savings_inr"] = no_model_cost - report["business"]["cost_with_paytrace_inr"]

    # Recall by fraud pattern (QR_SWAP never appears in training)
    ft = df.loc[te, "fraud_type"].to_numpy()
    report["recall_by_pattern"] = {
        k: {"n": int((ft == k).sum()),
            "lgbm": float(flag_review[ft == k].mean()),
            "lgbm_plus_novelty_tripwire": float(flag_combined[ft == k].mean())}
        for k in ["ATO", "CARD_TESTING", "MULE_P2P", "QR_SWAP"] if (ft == k).any()}

    known = ft != "QR_SWAP"
    report["paytrace"]["known_patterns_only"] = {
        "pr_auc": float(average_precision_score(yt[known], pt[known])),
        **metrics_at(yt[known], flag_review[known], at[known])}

    # Feedback loop: analysts label the first 5 test days -> retrain challenger -> evaluate
    # champion vs challenger on the remaining 10 days (never seen by either model).
    fb = te & (day < VAL_END_DAY + 5); ev = te & (day >= VAL_END_DAY + 5)
    trn2 = tr | va | fb
    ch = lgb.train(LGB_PARAMS, lgb.Dataset(X.loc[trn2, MODEL_COLUMNS], y[trn2]),
                   num_boost_round=max(booster.best_iteration, 50))
    p_ch = calib.predict(ch.predict(X.loc[ev, MODEL_COLUMNS]))
    p_cp = p[ev]; fte = df.loc[ev, "fraud_type"].to_numpy(); ye = y[ev]
    report["feedback_retrain"] = {
        "feedback_labels_days": 5, "eval_days": 10,
        "qr_swap_labels_used": int((df.loc[fb, "fraud_type"] == "QR_SWAP").sum()),
        "champion": {"pr_auc": float(average_precision_score(ye, p_cp)),
                     "qr_swap_recall": float((p_cp[fte == "QR_SWAP"] >= t_review).mean())},
        "challenger": {"pr_auc": float(average_precision_score(ye, p_ch)),
                       "qr_swap_recall": float((p_ch[fte == "QR_SWAP"] >= t_review).mean())},
    }

    # Global feature importance (mean |SHAP|) from LightGBM's built-in TreeSHAP
    contrib = booster.predict(Xt[MODEL_COLUMNS].sample(5000, random_state=1), pred_contrib=True,
                              num_iteration=booster.best_iteration)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=MODEL_COLUMNS).sort_values(ascending=False)
    report["feature_importance_mean_abs_shap"] = {k: float(v) for k, v in imp.head(15).items()}

    # Curves for the dashboard
    pr_p, pr_r, _ = precision_recall_curve(yt, pt)
    idx = np.linspace(0, len(pr_p) - 1, 200).astype(int)
    report["pr_curve"] = {"precision": pr_p[idx].tolist(), "recall": pr_r[idx].tolist()}
    report["cost_curve_val"] = {"threshold": grid[::5].tolist(), "cost_inr": costs[::5].tolist()}

    # ---- Persist artifacts ----
    booster.save_model(str(MODELS_DIR / "lgbm.txt"), num_iteration=booster.best_iteration)
    joblib.dump(iso, MODELS_DIR / "iforest.joblib")
    joblib.dump(giso, MODELS_DIR / "graph_iforest.joblib")
    joblib.dump(calib, MODELS_DIR / "calibrator.joblib")
    meta = {"model_columns": MODEL_COLUMNS, "t_review": t_review, "t_block": t_block,
            "if_threshold": if_thr,
            # skip the first week: feature state is still warming up then
            "psi_reference": psi_reference(X.loc[tr & (day >= 7), FEATURE_COLUMNS]),
            "test_start": str(df.loc[te, "timestamp"].min())}
    (MODELS_DIR / "meta.json").write_text(json.dumps(meta, indent=1))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=1))
    print_summary(report)


def print_summary(r):
    f = r["paytrace"]; a = f["at_review_threshold"]
    print("\n=== TEST SET (last 15 days, unseen) ===")
    print(f"PR-AUC {f['pr_auc']:.3f} | ROC-AUC {f['roc_auc']:.3f} | recall@1%FPR {f['recall_at_1pct_fpr']:.3f}")
    print(f"review tier: precision {a['precision']:.3f} recall {a['recall']:.3f} FPR {a['fpr']:.4%} "
          f"| fraud INR caught {a['fraud_amount_caught_pct']:.1%}")
    b = f["at_block_threshold"]
    print(f"block tier : precision {b['precision']:.3f} recall {b['recall']:.3f}")
    k = f["known_patterns_only"]
    print(f"known patterns only: PR-AUC {k['pr_auc']:.3f} precision {k['precision']:.3f} recall {k['recall']:.3f}")
    c = f["with_novelty_tripwire"]
    print(f"+novelty tripwire: precision {c['precision']:.3f} recall {c['recall']:.3f}")
    for k, v in r["baselines"].items():
        print(f"baseline {k:24s} precision {v['precision']:.3f} recall {v['recall']:.3f} "
              f"cost INR {v['total_cost_inr']:,.0f}" + (f" PR-AUC {v['pr_auc']:.3f}" if v.get("pr_auc") else ""))
    bz = r["business"]
    print(f"cost: no model INR {bz['cost_without_model_inr']:,.0f} | rules {bz['cost_with_rules_inr']:,.0f} "
          f"| payTrace {bz['cost_with_paytrace_inr']:,.0f}")
    print("recall by pattern:", json.dumps(r["recall_by_pattern"], indent=1))
    print("feedback retrain:", json.dumps(r["feedback_retrain"]))
    print("top features:", list(r["feature_importance_mean_abs_shap"])[:8])


if __name__ == "__main__":
    main()
