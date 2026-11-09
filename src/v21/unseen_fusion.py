"""Phases D and E: unseen-fraud evaluation (leave one pattern out) with graph-assisted detection, and risk-fusion
ablation. Writes reports/v21/unseen.json and reports/v21/fusion_<dataset>.json.

    python -m src.v21.unseen_fusion unseen            # synthetic UPI-style data (v2), 7 held-out patterns
    python -m src.v21.unseen_fusion fusion v2         # fusion ablation on v2
    python -m src.v21.unseen_fusion fusion sparkov    # fusion ablation on Sparkov (app model's data)

Leave-one-pattern-out: the pattern is removed from EVERY fitting slice (train, early stopping, calibration,
thresholds) and appears only in the test period (last 25%). Detectors compared on that pattern:
  LightGBM alone       calibrated supervised probability, flagged at its own cost-minimising threshold
  Isolation Forest     general anomaly score    } each flagged at the score that only 1% of genuine
  Autoencoder          reconstruction error     } threshold-slice payments exceed (label-free, same FPR budget)
  Hybrid (2.0)         the deployed decision rule: fused risk OR supervised OR any tripwire
  Graph-assisted (2.1) hybrid OR an online fraud-ring signal (src/fraud_rings.py RingMonitor, past-only)
Metrics: recall on the unseen pattern, recall at 1% FPR per score, PR-AUC of each score on genuine + that pattern,
overall false-positive rate, detection delay (hours from the pattern's first test payment to its first alert), and
loss (missed fraud of the pattern + Rs 100 per false alarm, test period).
"""
from __future__ import annotations

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluation import COST_FP, at_threshold, cost_threshold, precision_threshold, recall_at_fpr
from src.fraud_rings import RingMonitor
from src.train_v2 import FUSION_INPUTS, decisions, fit_system, fit_unsupervised, load, score_frame, slices

OUT = REPORTS_DIR / "v21"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ring_flags(df) -> np.ndarray:
    rm, out = RingMonitor(), []
    for r in df[["timestamp", "user_id", "merchant_id", "device_id", "amount"]].astype({"timestamp": str}).to_dict("records"):
        s = rm.check(r); out.append(bool(s["ring_rules"])); rm.add(r, s)
    return np.array(out)


def unseen():
    df, X = load("v2")
    s = slices(df)
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy(); ft = df.fraud_type.to_numpy()
    te, thr = s["test"], s["thr"]
    ts = df.timestamp.to_numpy()
    ring = ring_flags(df)
    unsup = fit_unsupervised(X, s["train"])
    res = {"ring_rule_fpr_test": float(ring[te & (y == 0)].mean()), "patterns": {}}
    for p in [p for p in sorted(set(ft)) if p != "LEGIT"]:
        t0 = time.time()
        hold = ft == p
        comps, S = fit_system(df, X, s, exclude=hold, unsup=unsup)
        dec, risk, _ = decisions(comps, S)
        gen_thr = thr & (y == 0)
        scores = {"lightgbm": S["sup_prob"].to_numpy(), "isolation_forest": S["if_score"].to_numpy(),
                  "autoencoder": S["ae_score"].to_numpy(), "fused_risk": np.asarray(risk)}
        sup_t = cost_threshold(y[thr & ~hold], scores["lightgbm"][thr & ~hold], amt[thr & ~hold])[0]
        flags = {"lightgbm": scores["lightgbm"] >= sup_t,
                 "isolation_forest": scores["isolation_forest"] >= np.quantile(scores["isolation_forest"][gen_thr], 0.99),
                 "autoencoder": scores["autoencoder"] >= np.quantile(scores["autoencoder"][gen_thr], 0.99),
                 "hybrid_2_0": dec != "ALLOW"}
        flags["graph_assisted_2_1"] = flags["hybrid_2_0"] | ring
        mp = te & hold
        sub = te & ((y == 0) | hold)                        # genuine + the unseen pattern
        first = ts[mp].min()
        r = {"n_test": int(mp.sum()), "detectors": {}}
        for k, f in flags.items():
            alerts = ts[mp & f]
            r["detectors"][k] = {
                "recall": float(f[mp].mean()), "fpr": float(f[te & (y == 0)].mean()),
                "detection_delay_hours": float((alerts.min() - first) / np.timedelta64(1, "h")) if len(alerts) else None,
                "loss_inr": float(amt[mp & ~f].sum() + COST_FP * (f & te & (y == 0)).sum()),
                "value_caught_pct": float(amt[mp & f].sum() / max(amt[mp].sum(), 1))}
        for k, sc in scores.items():
            r["detectors"].setdefault(k, {}).update({
                "pr_auc_on_pattern": float(average_precision_score(hold[sub], sc[sub])),
                "recall_at_1pct_fpr": recall_at_fpr(hold[sub], sc[sub], 0.01)})
        res["patterns"][p] = r
        d = r["detectors"]
        log(f"{p:20s} n={r['n_test']:4d} lgb {d['lightgbm']['recall']:.2f} IF {d['isolation_forest']['recall']:.2f} "
            f"AE {d['autoencoder']['recall']:.2f} hybrid {d['hybrid_2_0']['recall']:.2f} graph {d['graph_assisted_2_1']['recall']:.2f} "
            f"({time.time() - t0:.0f}s)")
    ks = ["lightgbm", "isolation_forest", "autoencoder", "hybrid_2_0", "graph_assisted_2_1"]
    res["mean_recall"] = {k: float(np.mean([v["detectors"][k]["recall"] for v in res["patterns"].values()])) for k in ks}
    res["mean_fpr"] = {k: float(np.mean([v["detectors"][k]["fpr"] for v in res["patterns"].values()])) for k in ks}
    (OUT / "unseen.json").write_text(json.dumps(res, indent=1, default=float))
    log("mean recall", res["mean_recall"])


# ------------------------------------------------------------------------------------------------ fusion ablation
VARIANTS = {
    "lightgbm_only": ["sup_logit"],
    "lightgbm_plus_iforest": ["sup_logit", "if_score"],
    "lightgbm_plus_autoencoder": ["sup_logit", "ae_score"],
    "all_anomaly_detectors": ["sup_logit", "if_score", "ae_score", "payee_score", "device_score", "flow_score"],
    "full_fusion_with_graph": FUSION_INPUTS,
}


def ece(y, p, n=10):
    e = np.unique(np.quantile(p, np.linspace(0, 1, n + 1)))
    idx = np.clip(np.digitize(p, e[1:-1]), 0, len(e) - 2)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(len(e) - 1) if (idx == b).any()))


def fusion(dataset):
    path = {"v2": "v2", "sparkov": str(DATA_DIR / "transactions_sparkov.csv")}[dataset]
    model_dir = {"v2": MODELS_DIR / "v2", "sparkov": MODELS_DIR / "real_sparkov"}[dataset]
    df, X = load(path)
    s = slices(df)
    comps = joblib.load(model_dir / "components.joblib")
    S = score_frame(comps, X)
    del X
    y = df.is_fraud.to_numpy(); amt = df.amount.to_numpy()
    cal, thr, te = s["cal"], s["thr"], s["test"]
    days = (df.timestamp[te].max() - df.timestamp[te].min()).total_seconds() / 86400
    out = {"dataset": dataset, "variants": {}}
    risks = {}
    for name, cols in VARIANTS.items():
        sc = StandardScaler().fit(S.loc[cal, cols])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(S.loc[cal, cols]), y[cal])
        p = lr.predict_proba(sc.transform(S[cols]))[:, 1]
        risks[name] = p
        t_rev = cost_threshold(y[thr], p[thr], amt[thr])[0]
        t_blk = precision_threshold(y[thr], p[thr], 0.95, floor=t_rev)
        rv = at_threshold(y[te], p[te] >= t_rev, amt[te], days)
        bk = at_threshold(y[te], p[te] >= t_blk, amt[te], days)
        out["variants"][name] = {
            "inputs": cols, "weights": {c: round(float(w), 3) for c, w in zip(cols, lr.coef_[0])},
            "pr_auc": float(average_precision_score(y[te], p[te])), "recall_at_1pct_fpr": recall_at_fpr(y[te], p[te], 0.01),
            "brier": float(brier_score_loss(y[te], p[te])), "ece": ece(y[te], p[te]),
            "review": {k: rv[k] for k in ("precision", "recall", "f1", "fpr", "loss_inr", "fraud_value_caught_pct", "alerts_per_day")},
            "block": {k: bk[k] for k in ("precision", "recall")}}
        log(f"{dataset} {name:28s} PR-AUC {out['variants'][name]['pr_auc']:.4f} R@1%FPR {out['variants'][name]['recall_at_1pct_fpr']:.3f} "
            f"review rec {rv['recall']:.3f} fpr {rv['fpr']:.4f} loss {rv['loss_inr']:,.0f}")
    # the deployed decision rule (fused risk OR supervised OR tripwire), with and without the 2.1 ring signal
    S["fused"] = risks["full_fusion_with_graph"]
    dec, _, trip = decisions(comps, S)
    hyb = dec != "ALLOW"
    out["deployed_rule"] = {"review_or_block": at_threshold(y[te], hyb[te], amt[te], days),
                            "tripwire_only_alerts_share": float((trip & te & (S["fused"].to_numpy() < comps.thresholds["review"])).sum() / max((hyb & te).sum(), 1))}
    if dataset == "v2":
        ring = ring_flags(df)
        out["deployed_rule_plus_rings"] = at_threshold(y[te], (hyb | ring)[te], amt[te], days)
    # threshold sensitivity of the full fusion (test period)
    p = risks["full_fusion_with_graph"]
    grid = np.quantile(p[thr], [0.9, 0.95, 0.98, 0.99, 0.995, 0.999])
    out["threshold_sensitivity"] = [{"threshold": float(t), **{k: v for k, v in at_threshold(y[te], p[te] >= t, amt[te], days).items()
                                                                if k in ("precision", "recall", "fpr", "loss_inr", "alerts_per_day")}} for t in grid]
    # detector correlation on test payments (Spearman)
    cols = ["sup_logit", "if_score", "ae_score", "payee_score", "device_score", "flow_score"]
    samp = S.loc[te, cols].sample(min(50000, int(te.sum())), random_state=1)
    rho = spearmanr(samp.to_numpy()).correlation
    out["detector_correlation_spearman"] = {a: {b: round(float(rho[i, j]), 3) for j, b in enumerate(cols)} for i, a in enumerate(cols)}
    # calibration of the deployed components
    out["calibration_deployed"] = {"supervised_isotonic": {"brier": float(brier_score_loss(y[te], S["sup_prob"][te])), "ece": ece(y[te], S["sup_prob"].to_numpy()[te])}}
    (OUT / f"fusion_{dataset}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "unseen":
        unseen()
    else:
        fusion(sys.argv[2])
