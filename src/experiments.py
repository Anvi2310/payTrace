"""payTrace 2.0 experiment suite. Every number in EXPERIMENTS.md is produced here.

  python -m src.experiments                 # all experiments (~10-15 min on a laptop)
  python -m src.experiments --only compare  # one experiment

Experiments
  compare      v1 system (retrained on proper slices) vs LightGBM / XGBoost / CatBoost (+/- fusion), both datasets
  anomaly      unsupervised detectors alone: Isolation Forest, autoencoder, graph Isolation Forest
  ablation     LightGBM with each feature group removed
  hpo          small random search for LightGBM (selected on the early-stopping slice)
  calibration  reliability bins + Brier score, raw vs isotonic vs fused
  thresholds   threshold sensitivity: recall, precision, FPR, loss, friction, workload
  segments     unseen vs seen customers; fairness across cities, account age, channel
  robustness   noisy amounts and missing device ids in the test period
  lopo         leave-one-pattern-out unseen-fraud test + analyst-feedback adaptation (1/3/5 days)
  confirm      compare + lopo once on a fresh dataset (seed 11) never used while designing the system
"""
from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

from src.config import FEATURE_COLUMNS, REPORTS_DIR
from src.evaluation import (at_threshold, calibration_bins, cost_threshold, pr_curve_points, precision_threshold,
                            ranking, roc_curve_points)
from src.features_v2 import GROUPS, build_feature_table_v2
from src.train_v2 import (FEATS, LGB_PARAMS, decisions, evaluate, fit_supervised, fit_system, fit_unsupervised,
                          fusion_weights, load, slices)

OUT = REPORTS_DIR / "experiments"
V1_GRAPH = ["m_age_hours", "m_users_1h", "m_users_24h", "u_new_merchant", "d_n_users"]


def days_of(df, m):
    return (df.timestamp[m].max() - df.timestamp[m].min()).total_seconds() / 86400


def summarise(df, mask, flags, risk):
    y = df.is_fraud.to_numpy()[mask]; amt = df.amount.to_numpy()[mask]
    return {**(ranking(y, risk[mask]) if risk is not None else {}),
            **at_threshold(y, flags[mask], amt, days_of(df, mask))}


# ---------------- v1 baseline, retrained on the same rigorous slices ----------------
def v1_system(df, X, s):
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy()
    samp = X[s["train"]].sample(min(60000, int(s["train"].sum())), random_state=7)
    iso = IsolationForest(n_estimators=100, max_samples=256, random_state=7, n_jobs=4).fit(samp[FEATURE_COLUMNS].to_numpy())
    g = lambda d: np.column_stack([np.log1p(d[V1_GRAPH[0]]), d[V1_GRAPH[1:]].to_numpy(float)])
    giso = IsolationForest(n_estimators=100, max_samples=256, random_state=7, n_jobs=4).fit(g(samp))
    XX = X[FEATURE_COLUMNS].copy(); XX["if_score"] = -iso.score_samples(X[FEATURE_COLUMNS].to_numpy())
    m, best, secs = fit_supervised("lgbm", XX[s["train"]], y[s["train"]], XX[s["es"]], y[s["es"]])
    raw = m.predict(XX)
    p = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw[s["cal"]], y[s["cal"]]).predict(raw)
    tr = s["thr"]
    t_rev = cost_threshold(y[tr], p[tr], amt[tr])[0]
    t_blk = precision_threshold(y[tr], p[tr], 0.95, floor=t_rev)
    gs = -giso.score_samples(g(X))
    trip = gs >= np.quantile(gs[s["thr"]], 0.998)
    return {"risk": p, "review": (p >= t_rev) | trip, "block": p >= t_blk, "train_seconds": secs, "best_iteration": best}


# ---------------- experiments ----------------
def exp_compare(datasets=("v2", "v1")):
    out = {}
    for name in datasets:
        df, X = load(name); s = slices(df); te = s["test"]
        res = {}
        b = v1_system(df, X, s)
        res["v1_system (27 features, LightGBM)"] = {**summarise(df, te, b["review"], b["risk"]),
                                                    "block_precision": summarise(df, te, b["block"], None)["precision"],
                                                    "block_recall": summarise(df, te, b["block"], None)["recall"],
                                                    "train_seconds": b["train_seconds"]}
        curves = {"v1_system": pr_curve_points(df.is_fraud[te], b["risk"][te])}
        rocs = {"v1_system": roc_curve_points(df.is_fraud[te], b["risk"][te])}
        unsup = fit_unsupervised(X, s["train"])
        for kind in ["lgbm", "xgb", "cat"]:
            for fuse in ([False, True] if kind == "lgbm" else [False]):
                comps, S = fit_system(df, X, s, kind=kind, unsup=unsup, use_fusion=fuse)
                dec, risk, _ = decisions(comps, S, fuse)
                label = f"v2 {kind}" + (" + fusion" if fuse else "")
                res[label] = {**summarise(df, te, dec != "ALLOW", risk),
                              "block_precision": summarise(df, te, dec == "BLOCK", None)["precision"],
                              "block_recall": summarise(df, te, dec == "BLOCK", None)["recall"],
                              "train_seconds": comps.train_seconds, "best_iteration": comps.best_iteration}
                curves[label] = pr_curve_points(df.is_fraud[te], risk[te])
                rocs[label] = roc_curve_points(df.is_fraud[te], risk[te])
                if fuse:
                    res[label]["fusion_weights"] = fusion_weights(comps)
                print(f"  [{name}] {label}: PR-AUC {res[label]['pr_auc']:.3f} R {res[label]['recall']:.3f} "
                      f"P {res[label]['precision']:.3f} loss {res[label]['loss_inr']:,.0f}")
        out[name] = {"results": res, "pr_curves": curves, "roc_curves": rocs,
                     "no_model_loss_inr": float(df.amount[te & (df.is_fraud == 1)].sum())}
    return out


def exp_anomaly(df, X, s):
    from src.train_v2 import unsupervised_scores
    te = s["test"]; y = df.is_fraud.to_numpy()
    U = unsupervised_scores(X, *fit_unsupervised(X, s["train"]))
    res = {}
    for col, name in [("if_score", "Isolation Forest (all features)"), ("ae_score", "Autoencoder reconstruction error"),
                      ("payee_score", "Payee-graph Isolation Forest"), ("device_score", "Device-graph Isolation Forest"),
                      ("flow_score", "Money-flow Isolation Forest")]:
        v = U[col].to_numpy()
        flag = v >= np.quantile(v[s["thr"]], 0.999)
        res[name] = {**ranking(y[te], v[te]), "recall_at_0.1pct_alert_budget": float(flag[te & (y == 1)].mean()),
                     "precision_at_budget": float(y[te & flag].mean()) if (te & flag).any() else 0.0}
        print(f"  {name}: PR-AUC {res[name]['pr_auc']:.3f}")
    return res


def exp_ablation(df, X, s):
    y = df.is_fraud.to_numpy(); te = s["test"]; amt = df.amount.to_numpy()
    unsup = fit_unsupervised(X, s["train"])
    res = {}
    for drop in [None] + list(GROUPS) + ["anomaly_score"]:
        feats = [f for f in FEATS if drop is None or drop == "anomaly_score" or f not in GROUPS[drop]]
        comps, S = fit_system(df, X, s, feats=feats, unsup=unsup, use_fusion=False)
        if drop == "anomaly_score":
            # retrain without the Isolation Forest score as an input
            m, best, _ = fit_supervised("lgbm", X.loc[s["train"], FEATS], y[s["train"]], X.loc[s["es"], FEATS], y[s["es"]])
            raw = m.predict(X[FEATS]); risk = IsotonicRegression(out_of_bounds="clip").fit(raw[s["cal"]], y[s["cal"]]).predict(raw)
        else:
            risk = S["sup_prob"].to_numpy()
        t = cost_threshold(y[s["thr"]], risk[s["thr"]], amt[s["thr"]])[0]
        label = "all features" if drop is None else f"without {drop}"
        res[label] = {**ranking(y[te], risk[te]), **at_threshold(y[te], risk[te] >= t, amt[te])}
        print(f"  {label}: PR-AUC {res[label]['pr_auc']:.3f} recall@1%FPR {res[label]['recall_at_1pct_fpr']:.3f}")
    return res


def exp_hpo(df, X, s, n=12, seed=3):
    from sklearn.metrics import average_precision_score
    from src.train_v2 import unsupervised_scores
    y = df.is_fraud.to_numpy(); rng = np.random.default_rng(seed)
    U = unsupervised_scores(X, *fit_unsupervised(X, s["train"]))
    XX = pd.concat([X[FEATS], U[["if_score"]]], axis=1)
    trials = []
    for i in range(n):
        p = {"num_leaves": int(rng.choice([15, 31, 63, 127])), "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
             "min_child_samples": int(rng.choice([20, 40, 100])), "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
             "scale_pos_weight": float(rng.choice([1, 5, 20])), "reg_lambda": float(rng.choice([0, 1, 5]))}
        m, best, _ = fit_supervised("lgbm", XX[s["train"]], y[s["train"]], XX[s["es"]], y[s["es"]], params=p)
        es_ap = average_precision_score(y[s["es"]], m.predict(XX[s["es"]]))
        trials.append({"params": p, "es_pr_auc": float(es_ap), "best_iteration": best, "model": m})
    default = max(trials[:0] or [None], key=lambda t: 0) if False else None
    best = max(trials, key=lambda t: t["es_pr_auc"])
    d, _, _ = fit_supervised("lgbm", XX[s["train"]], y[s["train"]], XX[s["es"]], y[s["es"]])
    te = s["test"]
    res = {"trials": [{k: v for k, v in t.items() if k != "model"} for t in trials],
           "best_params": best["params"],
           "test_pr_auc_default": float(average_precision_score(y[te], d.predict(XX[te]))),
           "test_pr_auc_tuned": float(average_precision_score(y[te], best["model"].predict(XX[te])))}
    print(f"  default test PR-AUC {res['test_pr_auc_default']:.3f} vs tuned {res['test_pr_auc_tuned']:.3f}")
    return res


def exp_calibration_thresholds_segments(df, X, s):
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy(); te = s["test"]
    comps, S = fit_system(df, X, s)
    dec, risk, trip = decisions(comps, S)
    from sklearn.metrics import brier_score_loss
    raw = S["sup_raw"].to_numpy()
    cal = {"brier_raw": float(brier_score_loss(y[te], np.clip(raw[te], 0, 1))),
           "brier_isotonic": float(brier_score_loss(y[te], S["sup_prob"].to_numpy()[te])),
           "brier_fused": float(brier_score_loss(y[te], risk[te])),
           "bins_raw": calibration_bins(y[te], np.clip(raw[te], 0, 1)),
           "bins_isotonic": calibration_bins(y[te], S["sup_prob"].to_numpy()[te]),
           "bins_fused": calibration_bins(y[te], risk[te])}
    days = days_of(df, te)
    grid = sorted(set(np.round(np.quantile(risk[te], [0.9, 0.95, 0.97, 0.98, 0.985, 0.99, 0.993, 0.996, 0.998]), 5).tolist()
                      + [round(comps.thresholds["review"], 5), round(comps.thresholds["block"], 5)]))
    sens = [{"threshold": t, **at_threshold(y[te], risk[te] >= t, amt[te], days),
             "is_review": abs(t - comps.thresholds["review"]) < 1e-5, "is_block": abs(t - comps.thresholds["block"]) < 1e-5}
            for t in grid]
    # segments
    seen = set(df.user_id[s["train"]])
    unseen = ~df.user_id.isin(seen).to_numpy()
    flag = dec != "ALLOW"
    seg = {"customers": {}}
    for name, m in [("seen in training", te & ~unseen), ("never seen in training", te & unseen)]:
        seg["customers"][name] = {"n": int(m.sum()), "fraud": int(y[m].sum()), **ranking(y[m], risk[m]),
                                  **at_threshold(y[m], flag[m], amt[m])}
    age = X["u_age_hours"].to_numpy()
    groups = {"city": df.city.to_numpy(), "channel": df.channel.to_numpy(),
              "account_age": np.where(age < 24 * 7, "< 1 week", np.where(age < 24 * 30, "1-4 weeks", "> 4 weeks"))}
    for gname, gv in groups.items():
        seg[gname] = {}
        for val in sorted(set(gv[te])):
            m = te & (gv == val)
            if m.sum() < 500:
                continue
            r = at_threshold(y[m], flag[m], amt[m])
            seg[gname][str(val)] = {"n": int(m.sum()), "fpr": r["fpr"], "recall": r["recall"], "precision": r["precision"]}
    print(f"  Brier raw {cal['brier_raw']:.4f} isotonic {cal['brier_isotonic']:.4f} fused {cal['brier_fused']:.4f}")
    print("  unseen customers:", {k: round(v.get("recall", 0), 3) for k, v in seg["customers"].items()})
    return {"calibration": cal, "thresholds": sens, "segments": seg, "fusion_weights": fusion_weights(comps)}


def exp_robustness(df, X, s):
    y = df.is_fraud.to_numpy(); te = s["test"]; amt = df.amount.to_numpy()
    comps, S = fit_system(df, X, s)
    base = evaluate(df, S, comps, te)
    res = {"clean": {"pr_auc": base["ranking"]["pr_auc"], "recall": base["review_tier"]["recall"], "fpr": base["review_tier"]["fpr"]}}
    rng = np.random.default_rng(0)
    from src.train_v2 import score_frame
    for name in ["amount noise ±10%", "30% device ids missing"]:
        d2 = df.copy()
        if name.startswith("amount"):
            d2.loc[te, "amount"] = d2.loc[te, "amount"] * rng.uniform(0.9, 1.1, te.sum())
        else:
            hit = te & (rng.random(len(d2)) < 0.3)
            d2.loc[hit, "device_id"] = "UNKNOWN"
        X2, _ = build_feature_table_v2(d2)
        S2 = score_frame(comps, X2)
        from src.train_v2 import FUSION_INPUTS
        S2["fused"] = comps.fusion.predict_proba(comps.fusion_scaler.transform(S2[FUSION_INPUTS]))[:, 1]
        r = evaluate(d2, S2, comps, te)
        res[name] = {"pr_auc": r["ranking"]["pr_auc"], "recall": r["review_tier"]["recall"], "fpr": r["review_tier"]["fpr"]}
        print(f"  {name}: {res[name]}")
    return res


def exp_lopo(df, X, s, feedback_days=(1, 3, 5)):
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy(); ft = df.fraud_type.to_numpy()
    te = s["test"]
    t0 = df.timestamp[te].min()
    dday = ((df.timestamp - t0).dt.total_seconds() / 86400).to_numpy()
    eval_mask = te & (dday >= max(feedback_days))            # separate future period, after every feedback window
    unsup = fit_unsupervised(X, s["train"])
    patterns = [p for p in sorted(set(ft)) if p != "LEGIT"]
    res = {}
    for p in patterns:
        hold = ft == p
        comps, S = fit_system(df, X, s, exclude=hold, unsup=unsup)
        dec, risk, trip = decisions(comps, S)
        sup_flag = S["sup_prob"].to_numpy() >= cost_threshold(y[s["thr"] & ~hold], S["sup_prob"].to_numpy()[s["thr"] & ~hold],
                                                              amt[s["thr"] & ~hold])[0]
        fused_flag = risk >= comps.thresholds["review"]
        hybrid = dec != "ALLOW"
        mp = te & hold
        r = {"n_test": int(mp.sum()),
             "supervised_only": float(sup_flag[mp].mean()),
             "fused_score": float(fused_flag[mp].mean()),
             **{f"{k}_tripwire": float((S[f"{k}_score"].to_numpy() >= comps.thresholds[f"{k}_trip"])[mp].mean())
                for k in ["payee", "device", "flow", "ae"]},
             "hybrid_system": float(hybrid[mp].mean()),
             "hybrid_overall_precision": float(y[te & hybrid].mean()), "hybrid_overall_fpr": float(hybrid[te & (y == 0)].mean())}
        # feedback: analysts label ONLY alerted test transactions in the first k days; challenger retrained
        r["adaptation"] = {"champion_eval_recall": float(hybrid[eval_mask & hold].mean()),
                           "champion_eval_pr_auc": ranking(y[eval_mask], risk[eval_mask])["pr_auc"]}
        for k in feedback_days:
            fbm = te & (dday < k) & hybrid
            ch, S2 = fit_system(df, X, s, exclude=hold, extra_train=fbm, unsup=unsup, rounds=comps.best_iteration)
            d2, r2, _ = decisions(ch, S2)
            r["adaptation"][f"after_{k}d_feedback"] = {
                "labels_used": int(fbm.sum()), "fraud_labels_of_pattern": int((fbm & hold).sum()),
                "eval_recall": float((d2 != "ALLOW")[eval_mask & hold].mean()),
                "eval_pr_auc": ranking(y[eval_mask], r2[eval_mask])["pr_auc"]}
        res[p] = r
        a = r["adaptation"]
        print(f"  {p:20s} sup {r['supervised_only']:.2f} fused {r['fused_score']:.2f} payee {r['payee_tripwire']:.2f} "
              f"device {r['device_tripwire']:.2f} flow {r['flow_tripwire']:.2f} ae {r['ae_tripwire']:.2f} "
              f"hybrid {r['hybrid_system']:.2f} | eval recall champ "
              f"{a['champion_eval_recall']:.2f} -> " + " / ".join(f"{a[f'after_{k}d_feedback']['eval_recall']:.2f}" for k in feedback_days))
    return res


def exp_confirm(path="data/transactions_v2_seed11.csv"):
    """One-time confirmation on a freshly generated dataset (different seed) that was never used for design."""
    out = {"compare": exp_compare((path,))[path]}
    d2, X2 = load(path)
    out["lopo"] = exp_lopo(d2, X2, slices(d2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--dataset", default="v2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    df, X = load(a.dataset); s = slices(df)
    runs = {"compare": lambda: exp_compare(), "anomaly": lambda: exp_anomaly(df, X, s),
            "ablation": lambda: exp_ablation(df, X, s), "hpo": lambda: exp_hpo(df, X, s),
            "calibration": lambda: exp_calibration_thresholds_segments(df, X, s),
            "robustness": lambda: exp_robustness(df, X, s), "lopo": lambda: exp_lopo(df, X, s),
            "confirm": lambda: exp_confirm()}
    for name, fn in runs.items():
        if a.only and name != a.only:
            continue
        print(f"== {name} ==")
        t = time.time()
        results[name] = fn()
        results[name + "_seconds"] = round(time.time() - t, 1)
        path.write_text(json.dumps(results, indent=1, default=float))
    results["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(results, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
