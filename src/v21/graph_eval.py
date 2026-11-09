"""Phase C evaluation: fraud-ring intelligence on the synthetic UPI-style data (v2), the only dataset with devices
and person-to-person transfers (Sparkov has neither; IEEE-CIS has no payee ids).

    python -m src.v21.graph_eval      # writes reports/v21/graph_eval.json

1. Transaction level (online RingMonitor, past-only): for each payment in the TEST period (last 25%), which ring
   rules fire; recall per fraud pattern and false-positive rate on genuine payments. The monitor's thresholds are
   fixed constants (src/fraud_rings.py) set before this evaluation, not tuned on it. They were written knowing the
   pattern descriptions in src/generate_data_v2.py, so treat synthetic-data results as optimistic.
2. Entity level (offline analyse, per 7-day window of the test period, model decisions from the saved v2 system):
   are the entities in the reported rings really fraud entities? precision = share of ring entities touched by at
   least one fraud payment; recall = share of fraud-ring entities (shared devices, collector payees, mule accounts)
   that appear in some reported ring.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, REPORTS_DIR
from src.fraud_rings import RingMonitor, analyse
from src.graph_intel import node_of_payee
from src.train_v2 import FUSION_INPUTS, decisions, load, score_frame, slices

OUT = REPORTS_DIR / "v21" / "graph_eval.json"
RULES = ["fan_in", "shared_device", "device_new_accounts", "pass_through", "mule_chain", "cycle"]


def transaction_level(df, te):
    rm, rules, t0 = RingMonitor(), [], time.time()
    for r in df[["timestamp", "user_id", "merchant_id", "device_id", "amount"]].astype({"timestamp": str}).to_dict("records"):
        s = rm.check(r); rules.append(s["ring_rules"]); rm.add(r, s)
    ms = (time.time() - t0) / len(df) * 1e3
    d = df[te].assign(rules=np.array(rules, dtype=object)[te])
    out = {"ms_per_txn": ms, "rules": {}}
    pats = sorted(p for p in d.fraud_type.unique() if p != "LEGIT")
    for rule in RULES + ["any"]:
        f = d.rules.apply(bool) if rule == "any" else d.rules.apply(lambda x, r=rule: r in x)
        out["rules"][rule] = {"fpr": float(f[d.is_fraud == 0].mean()),
                              "recall": {p: float(f[d.fraud_type == p].mean()) for p in pats},
                              "recall_all_fraud": float(f[d.is_fraud == 1].mean())}
    return out, np.array([bool(x) for x in rules])


def fraud_entities(d):
    """Entities that fraud actually runs through (not the victims): shared ring devices, collector payees, mules."""
    f = d[d.is_fraud == 1]
    ents = set()
    ents |= {f"D:{x}" for x in f[f.fraud_type == "DEVICE_RING"].device_id}
    ents |= {node_of_payee(x) for x in f[f.fraud_type.isin(["QR_SWAP", "MERCHANT_COLLUSION", "MULE_P2P", "RAPID_MOVEMENT", "DEVICE_RING"])].merchant_id}
    ents |= {f"U:{x}" for x in f[f.fraud_type.isin(["RAPID_MOVEMENT", "DEVICE_RING"])].user_id}
    return ents


def entity_level(df, S, comps, te):
    dec, risk, _ = decisions(comps, S)
    ev = df.assign(decision=dec, probability=risk, ts=df.timestamp.astype(str))[te]
    start, end = pd.Timestamp(ev.ts.min()), pd.Timestamp(ev.ts.max())
    windows, tp_all, ring_all, truth_all = [], set(), set(), set()
    t = start
    while t < end:
        w = ev[(pd.to_datetime(ev.ts) >= t) & (pd.to_datetime(ev.ts) < t + pd.Timedelta(days=7))]
        t += pd.Timedelta(days=7)
        if w.empty:
            continue
        t0 = time.time()
        rep = analyse(w[["txn_id", "ts", "user_id", "device_id", "merchant_id", "amount", "probability", "decision"]], top=1000)
        secs = time.time() - t0
        touched = set()
        for r in w[w.is_fraud == 1].itertuples():
            touched |= {f"U:{r.user_id}", node_of_payee(r.merchant_id), f"D:{r.device_id}"}
        ring_nodes = set().union(*[set(r["nodes"]) for r in rep["rings"]]) if rep["rings"] else set()
        truth = fraud_entities(w)
        windows.append({"from": str(w.ts.min())[:10], "payments": len(w), "rings": len(rep["rings"]),
                        "ring_entities": len(ring_nodes), "precision": len(ring_nodes & touched) / max(len(ring_nodes), 1),
                        "fraud_entity_recall": len(truth & ring_nodes) / max(len(truth), 1), "analyse_seconds": round(secs, 1),
                        "cycles": rep["stats"]["cycles"], "chains": rep["stats"]["chains"],
                        "top_ring": {k: rep["rings"][0][k] for k in ("size", "score", "evidence")} if rep["rings"] else None})
        tp_all |= ring_nodes & touched; ring_all |= ring_nodes; truth_all |= truth
    return {"windows": windows, "precision": len(tp_all) / max(len(ring_all), 1),
            "fraud_entity_recall": len(truth_all & ring_all) / max(len(truth_all), 1),
            "ring_entities": len(ring_all), "fraud_entities": len(truth_all)}


def main():
    df, X = load("v2")
    s = slices(df)
    te = s["test"]
    tl, _ = transaction_level(df, te)
    comps = joblib.load(MODELS_DIR / "v2" / "components.joblib")
    S = score_frame(comps, X)
    S["fused"] = comps.fusion.predict_proba(comps.fusion_scaler.transform(S[FUSION_INPUTS]))[:, 1]
    el = entity_level(df, S, comps, te)
    res = {"dataset": "synthetic UPI-style v2 (test period = last 25%)", "transaction_level": tl, "entity_level": el}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"any_rule": tl["rules"]["any"], "entity": {k: el[k] for k in ("precision", "fraud_entity_recall", "ring_entities", "fraud_entities")}}, indent=1))


if __name__ == "__main__":
    main()
