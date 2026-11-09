"""Champion / challenger retraining from validated analyst feedback, with a model registry.

    python -m src.retrain            # train a challenger from accepted feedback and compare it with the champion
    python -m src.retrain --promote  # also promote it if it wins

Only feedback with status 'accepted' is used (see src/feedback.py). Analyst labels override replay labels for those
rows. The comparison window is every streamed payment AFTER the newest feedback label, so neither model has seen
any of it. Promotion requires: challenger PR-AUC >= champion PR-AUC and false-positive rate no more than 0.2
points worse.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.evaluation import at_threshold, ranking
from src.feedback import accepted_labels
from src.store import audit, connect
from src.train_v2 import V2_DIR, decisions, fit_system, load, save, slices


def register(con, version, role, kind, trained_on, metrics, notes=""):
    con.execute("INSERT OR REPLACE INTO model_registry VALUES (?,?,?,?,?,?,?)",
                (version, time.time(), kind, role, trained_on, json.dumps(metrics), notes))
    con.commit()


def ensure_champion(con):
    if con.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0] == 0:
        meta = json.loads((V2_DIR / "meta.json").read_text())
        register(con, meta.get("trained_at", "v2-initial"), "champion", meta.get("model", "lgbm"), meta.get("dataset", "v2"),
                 {}, "initial model trained by src.train_v2")


def run(promote=False, actor="cli", role="admin", dataset="v2"):
    con = connect()
    ensure_champion(con)
    labels = accepted_labels(con)
    if len(labels) < 20:
        return {"status": "skipped", "reason": f"only {len(labels)} accepted labels (need at least 20)"}
    df, X = load(dataset)
    s = slices(df)
    idx = df.index[df.txn_id.isin(labels)]
    fb_mask = np.zeros(len(df), bool); fb_mask[idx] = True
    d2 = df.copy()
    d2.loc[idx, "is_fraud"] = d2.loc[idx, "txn_id"].map(labels).astype(int)
    newest = df.loc[idx, "timestamp"].max()
    window = s["test"] & (df.timestamp > newest).to_numpy() & ~fb_mask
    if window.sum() < 1000:
        return {"status": "skipped", "reason": "fewer than 1,000 payments after the newest label to compare on"}

    champion = joblib.load(V2_DIR / "components.joblib")
    from src.train_v2 import score_frame, FUSION_INPUTS
    S0 = score_frame(champion, X)
    S0["fused"] = champion.fusion.predict_proba(champion.fusion_scaler.transform(S0[FUSION_INPUTS]))[:, 1]
    challenger, S1 = fit_system(d2, X, s, extra_train=fb_mask, rounds=champion.best_iteration)
    y = df.is_fraud.to_numpy()[window]; amt = df.amount.to_numpy()[window]
    res = {}
    for name, comps, S in (("champion", champion, S0), ("challenger", challenger, S1)):
        dec, risk, _ = decisions(comps, S)
        res[name] = {**ranking(y, risk[window]), **at_threshold(y, dec[window] != "ALLOW", amt)}
    wins = (res["challenger"]["pr_auc"] >= res["champion"]["pr_auc"]
            and res["challenger"]["fpr"] <= res["champion"]["fpr"] + 0.002)
    version = time.strftime("%Y-%m-%d %H:%M:%S")
    out_dir = MODELS_DIR / f"v2_challenger_{version.replace(' ', '_').replace(':', '')}"
    meta = json.loads((V2_DIR / "meta.json").read_text())
    meta.update(trained_at=version, version=version, thresholds={k: v for k, v in challenger.thresholds.items()
                                                                 if not k.startswith("cost")}, feedback_labels=len(labels))
    save(challenger, meta, out_dir)
    summary = {"labels_used": len(labels), "comparison_window_payments": int(window.sum()),
               "champion": {k: res["champion"][k] for k in ("pr_auc", "recall", "precision", "fpr")},
               "challenger": {k: res["challenger"][k] for k in ("pr_auc", "recall", "precision", "fpr")},
               "challenger_wins": bool(wins), "note": "comparison uses replay labels after the newest feedback label"}
    register(con, version, "challenger", challenger.kind, f"{dataset} + {len(labels)} feedback labels", summary, str(out_dir))
    audit(con, actor, role, "retrain", version, summary)
    if promote and wins:
        con.execute("UPDATE model_registry SET role='retired' WHERE role='champion'")
        con.execute("UPDATE model_registry SET role='champion' WHERE version=?", (version,))
        con.commit()
        backup = MODELS_DIR / f"v2_backup_{int(time.time())}"
        shutil.copytree(V2_DIR, backup)
        for f in ("components.joblib", "meta.json"):
            shutil.copy(out_dir / f, V2_DIR / f)
        audit(con, actor, role, "promote", version, {"backup": str(backup)})
        summary["promoted"] = True
    return {"status": "done", "version": version, **summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.promote), indent=1, default=float))
