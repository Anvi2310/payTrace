import pandas as pd

from src.graph_intel import build_graph, entity_findings, money_chains, suspicious_clusters
from src.investigation import SYSTEM, template_summary


def events():
    rows = []
    for i in range(8):                                    # 8 accounts on one device paying one payee
        rows.append(dict(txn_id=f"R{i}", ts=f"2026-09-01 10:{i:02d}:00", user_id=f"U{i}", device_id="DRING",
                         merchant_id="P2P_UX", amount=1000, probability=0.9, decision="BLOCK"))
    rows += [dict(txn_id="C1", ts="2026-09-01 11:00:00", user_id="V", device_id="DV", merchant_id="P2P_UA", amount=9000, probability=0.5, decision="REVIEW"),
             dict(txn_id="C2", ts="2026-09-01 11:05:00", user_id="UA", device_id="DA", merchant_id="P2P_UB", amount=8800, probability=0.4, decision="REVIEW"),
             dict(txn_id="C3", ts="2026-09-01 11:09:00", user_id="UB", device_id="DB", merchant_id="P2P_UC", amount=8600, probability=0.4, decision="REVIEW")]
    return pd.DataFrame(rows)


def test_graph_findings_are_counted():
    G = build_graph(events())
    assert any("8 different accounts" in f for f in entity_findings(G, "D:DRING"))
    assert any("Received payments from 8" in f for f in entity_findings(G, "U:UX"))


def test_money_chain_found():
    ch = money_chains(events())
    assert ch and [h["txn_id"] for h in ch[0]] == ["C1", "C2", "C3"]


def test_clusters():
    assert suspicious_clusters(build_graph(events()), min_flagged=3)


def test_copilot_rules_and_template_never_claims_guilt():
    assert "Never approve, reject or block" in SYSTEM and "guilty" in SYSTEM
    pack = {"transaction": {"txn_id": "T", "ts": "2026", "user_id": "U", "amount": 10.0, "merchant_id": "M",
                            "category": "c", "channel": "UPI", "device_id": "D", "city": "x"},
            "model": {"decision": "REVIEW", "risk": 0.4, "supervised_probability": 0.3, "tripwires": [], "reason_codes": [],
                      "risk_breakdown": [], "model_version": "v"},
            "customer_history": {"payments_last_30d": 0, "median_amount": None, "this_amount_vs_median": None,
                                 "devices_seen": 0, "device_seen_before": False, "payee_paid_before": False},
            "network_findings": {}, "connected_flagged_entities": [], "money_chains": [], "similar_cases": [],
            "suggested_next_steps": ["x"]}
    s = template_summary(pack)
    assert "not a finding of guilt" in s and "fraudster" not in s.lower()
