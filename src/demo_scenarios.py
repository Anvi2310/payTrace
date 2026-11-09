"""Scripted demo scenarios, scored by the real 2.0 engine after it has been warmed on the dataset's history.

    python -m src.demo_scenarios            # print decisions and reasons for each scenario
    python -m src.demo_scenarios --to-db    # also write them to the event store (they appear in the dashboard,
                                            # tagged fraud_type=DEMO_*, so they are never mistaken for replay data)

Scenarios are hand-written transactions for demonstration. They are clearly labelled as such everywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

import pandas as pd

from src.scorer_v2 import FraudScorerV2
from src.stream_v2 import TXN_FIELDS
from src.train_v2 import dataset_path, slices


def scenarios(start: datetime, users: list[str]) -> dict[str, list[dict]]:
    t = lambda m: (start + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S")
    v1, v2, v3 = users[:3]
    base = dict(category="grocery", channel="UPI", city="Chennai")
    sc = {}
    sc["Normal grocery payment"] = [dict(base, txn_id="DEMO-N1", timestamp=t(0), user_id=v1, amount=420.0,
                                         merchant_id="M00010", device_id=f"D{v1[1:]}_0")]
    sc["Card testing burst"] = [dict(txn_id=f"DEMO-CT{i}", timestamp=t(10 + i * 0.5), user_id=v2, amount=3.0 + i,
                                     merchant_id=f"M0050{i % 9}", category="gaming", channel="CARD", device_id="D_DEMO_ATTACK",
                                     city="Delhi") for i in range(8)] + \
        [dict(txn_id="DEMO-CT9", timestamp=t(15), user_id=v2, amount=18999.0, merchant_id="M00400", category="electronics",
              channel="CARD", device_id="D_DEMO_ATTACK", city="Delhi")]
    sc["Device ring (new accounts, one phone)"] = [dict(txn_id=f"DEMO-DR{i}", timestamp=t(30 + i), user_id=f"U_DEMO_RING_{i}",
                                                        amount=2500.0, merchant_id="P2P_DEMO_CASHOUT", category="P2P",
                                                        channel="UPI", device_id="D_DEMO_RING", city="Pune") for i in range(9)]
    sc["Rapid money movement (mule chain)"] = [
        dict(base, txn_id="DEMO-RM1", timestamp=t(60), user_id=v3, amount=45000.0, merchant_id=f"P2P_{v1}", category="P2P",
             device_id=f"D{v3[1:]}_0"),
        dict(base, txn_id="DEMO-RM2", timestamp=t(66), user_id=v1, amount=43500.0, merchant_id=f"P2P_{v2}", category="P2P",
             device_id=f"D{v1[1:]}_0"),
        dict(base, txn_id="DEMO-RM3", timestamp=t(71), user_id=v2, amount=42000.0, merchant_id="M01400", category="crypto",
             channel="NETBANKING", device_id=f"D{v2[1:]}_0")]
    sc["QR swap (new merchant, many first-time payers)"] = [
        dict(txn_id=f"DEMO-QR{i}", timestamp=t(90 + i * 6), user_id=users[10 + i], amount=1450.0 + 37 * i,
             merchant_id="M_DEMO_QR", category="utilities", channel="UPI", device_id=f"D{users[10 + i][1:]}_0", city="Chennai")
        for i in range(12)]
    return sc


def run(to_db=False, dataset="v2"):
    df = pd.read_csv(dataset_path(dataset), parse_dates=["timestamp"])
    s = slices(df)
    hist = df[~s["test"]]
    sc = FraudScorerV2()
    sc.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist.is_fraud.to_numpy())
    users = hist.user_id.value_counts().index[:40].tolist()
    start = hist.timestamp.max().to_pydatetime() + timedelta(hours=1)
    results = []
    con = None
    if to_db:
        from src.store import connect
        con = connect()
    for name, txns in scenarios(start, users).items():
        print(f"\n== {name} ==")
        for x in txns:
            r = sc.score(x)
            results.append({"scenario": name, **{k: x[k] for k in ("txn_id", "amount", "user_id", "merchant_id")},
                            "decision": r["decision"], "risk": round(r["risk"], 4)})
            print(f"  {x['txn_id']:<10} ₹{x['amount']:>9,.0f}  {r['decision']:<6} risk {r['risk']:6.1%}  "
                  + "; ".join(z["reason"] for z in r["reasons"][:3]))
            if con is not None:
                feats = json.dumps({k: round(float(v), 4) for k, v in r["features"].items()}) if r["decision"] != "ALLOW" else None
                con.execute(f"INSERT OR REPLACE INTO events VALUES ({','.join('?' * 27)})",
                            (None, x["txn_id"], x["timestamp"], x["user_id"], x["amount"], x["merchant_id"], x["category"],
                             x["channel"], x["device_id"], x["city"], r["risk"], r["supervised_prob"], r["decision"],
                             ",".join(r["tripwires"]), r["if_score"], r["ae_score"], r["graph_score"], json.dumps(r["reasons"]),
                             json.dumps(r["risk_breakdown"]), feats, r["latency_ms"], r["latency_ms"], 0, -1,
                             "DEMO_" + name.split(" (")[0].upper().replace(" ", "_"), "demo", time.time()))
    if con is not None:
        con.commit()
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-db", action="store_true")
    run(ap.parse_args().to_db)
