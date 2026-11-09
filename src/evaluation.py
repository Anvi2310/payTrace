"""Metrics for imbalanced fraud detection. All functions are pure (no side effects)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

COST_FP = 100.0          # INR per false alarm (analyst review + customer friction)


def recall_at_fpr(y, s, fpr_target=0.01):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def precision_at_recall(y, s, recall_target=0.8):
    p, r, _ = precision_recall_curve(y, s)
    ok = r >= recall_target
    return float(p[ok].max()) if ok.any() else 0.0


def at_threshold(y, flag, amount, days: float | None = None):
    y = np.asarray(y).astype(bool); flag = np.asarray(flag).astype(bool); amount = np.asarray(amount, float)
    tp = int((y & flag).sum()); fp = int((~y & flag).sum()); fn = int((y & ~flag).sum()); tn = int((~y & ~flag).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    fraud_amt = amount[y].sum(); caught = amount[y & flag].sum()
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "fpr": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "fraud_value_caught_pct": float(caught / max(fraud_amt, 1)),
           "loss_inr": float(amount[y & ~flag].sum() + COST_FP * fp),
           "genuine_flagged_per_10k": 1e4 * fp / max(fp + tn, 1)}
    if days:
        out["alerts_per_day"] = (tp + fp) / days
    return out


def ranking(y, s):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return {"pr_auc": None, "roc_auc": None}
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "recall_at_0.5pct_fpr": recall_at_fpr(y, s, 0.005), "recall_at_1pct_fpr": recall_at_fpr(y, s, 0.01),
            "precision_at_80pct_recall": precision_at_recall(y, s, 0.8)}


def cost_threshold(y, s, amount, grid=None):
    """Threshold minimising missed-fraud INR + COST_FP per false alarm."""
    y = np.asarray(y).astype(bool); s = np.asarray(s); amount = np.asarray(amount, float)
    if grid is None:
        grid = np.unique(np.quantile(s, np.linspace(0.85, 0.99995, 500)))
    costs = [amount[y & (s < t)].sum() + COST_FP * (~y & (s >= t)).sum() for t in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), np.array(grid), np.array(costs)


def precision_threshold(y, s, target=0.95, floor=None):
    p, r, t = precision_recall_curve(y, s)
    ok = np.where(p[:-1] >= target)[0]
    th = float(t[ok[0]]) if len(ok) else float(np.max(s))
    return max(th, floor) if floor is not None else th


def calibration_bins(y, p, n=10):
    y = np.asarray(y); p = np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out.append({"mean_pred": float(p[m].mean()), "frac_fraud": float(y[m].mean()), "n": int(m.sum())})
    return out


def pr_curve_points(y, s, n=150):
    p, r, _ = precision_recall_curve(y, s)
    i = np.linspace(0, len(p) - 1, n).astype(int)
    return {"precision": p[i].round(4).tolist(), "recall": r[i].round(4).tolist()}


def roc_curve_points(y, s, n=150):
    f, t, _ = roc_curve(y, s)
    i = np.linspace(0, len(f) - 1, n).astype(int)
    return {"fpr": f[i].round(5).tolist(), "tpr": t[i].round(4).tolist()}
