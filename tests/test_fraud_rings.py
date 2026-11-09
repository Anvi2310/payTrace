"""Fraud-ring intelligence: online ring signals (past-only), ring decisions, offline ring analysis, cycles."""
import pandas as pd

from src.fraud_rings import RingMonitor, analyse, ring_decision, temporal_cycles, build_temporal_graph


def t(m):                                   # minutes after a fixed start
    return (pd.Timestamp("2026-09-01 10:00:00") + pd.Timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S")


def tx(m, u, payee, amt=500.0, dev=None):
    return {"timestamp": t(m), "user_id": u, "merchant_id": payee, "amount": amt, "device_id": dev or f"D_{u}"}


def feed(rm, rows):
    out = []
    for r in rows:
        s = rm.check(r); out.append(s); rm.add(r, s)
    return out


def test_fan_in_needs_many_new_senders_to_a_new_payee():
    rm = RingMonitor()
    sig = feed(rm, [tx(i, f"V{i}", "QR1") for i in range(6)])
    assert not sig[3]["fan_in"] and sig[4]["fan_in"] and sig[5]["fan_in"]      # fires at the 5th distinct sender
    assert sig[4]["payee_senders_24h"] == 5 and sig[4]["payee_new_sender_share_24h"] == 1.0
    # an established shop paid by its regulars does not look like a collector
    rm2 = RingMonitor()
    feed(rm2, [tx(-5000 + i, f"C{i % 5}", "SHOP") for i in range(50)])
    later = feed(rm2, [tx(i, f"C{i}", "SHOP") for i in range(5)])
    assert not any(s["fan_in"] for s in later)


def test_shared_device_and_new_accounts():
    rm = RingMonitor()
    sig = feed(rm, [tx(i, f"A{i}", "P1", dev="FARM") for i in range(3)])
    assert sig[2]["shared_device"] and sig[2]["device_accounts_7d"] == 3 and sig[2]["device_new_accounts"]
    assert not feed(rm, [tx(10, "Z", "P1", dev="UNKNOWN")])[0]["shared_device"]   # missing device ids never link accounts


def test_pass_through_chain_and_cycle():
    rm = RingMonitor()
    s = feed(rm, [tx(0, "V", "P2P_A", 10000), tx(2, "A", "P2P_B", 9500), tx(4, "B", "P2P_C", 9000)])
    assert s[1]["pass_through"] and s[2]["pass_through"] and s[2]["chain_depth"] >= 2 and "mule_chain" in s[2]["ring_rules"]
    loop = feed(RingMonitor(), [tx(0, "A", "P2P_B"), tx(1, "B", "P2P_C"), tx(2, "C", "P2P_A")])
    assert not loop[1]["cycle"] and loop[2]["cycle"]


def test_signals_use_only_the_past():
    rm = RingMonitor()
    rows = [tx(i, f"V{i}", "QR9") for i in range(5)]
    first = rm.check(rows[0])
    assert first["payee_senders_24h"] == 1                   # later payments can't influence an earlier decision


def test_ring_decision_only_raises_and_confirmed_link_blocks():
    assert ring_decision({"ring_rules": []}, "BLOCK")[0] == "BLOCK"
    assert ring_decision({"ring_rules": ["fan_in"]}, "ALLOW")[0] == "REVIEW"
    assert ring_decision({"ring_rules": ["fan_in"]}, "BLOCK")[0] == "BLOCK"
    rm = RingMonitor(); rm.confirm_fraud(payee="BAD")
    s = rm.check(tx(0, "U", "BAD"))
    assert s["confirmed_link"] and ring_decision(s, "ALLOW")[0] == "BLOCK"


def test_offline_analysis_finds_ring_and_marks_confirmed():
    rows = []
    for i in range(6):
        rows.append({"txn_id": f"T{i}", "ts": t(i), "user_id": f"M{i}", "device_id": "FARM" if i < 3 else "FARM2",
                     "merchant_id": "P2P_COL", "amount": 5000.0, "probability": 0.4, "decision": "REVIEW", "confirmed": int(i == 0)})
    for i in range(30):                                      # unrelated normal traffic
        rows.append({"txn_id": f"N{i}", "ts": t(100 + i * 60), "user_id": f"N{i % 10}", "device_id": f"DN{i % 10}",
                     "merchant_id": f"SHOP{i % 3}", "amount": 300.0, "probability": 0.01, "decision": "ALLOW", "confirmed": 0})
    rep = analyse(pd.DataFrame(rows))
    assert rep["rings"], rep["stats"]
    r = rep["rings"][0]
    assert "U:COL" in r["nodes"] and "D:FARM" in r["nodes"] and r["status"] == "confirmed fraud links"
    assert not any(n.startswith("U:N") for n in r["nodes"])  # normal customers are not pulled into the ring
    assert rep["top_pagerank"] and "cycles" in rep


def test_temporal_cycles_respect_time_order():
    def ev(rows):
        return pd.DataFrame([{"txn_id": f"X{i}", "ts": t(m), "user_id": a, "device_id": "UNKNOWN", "merchant_id": f"P2P_{b}",
                              "amount": 100.0, "probability": 0, "decision": "ALLOW"} for i, (m, a, b) in enumerate(rows)])
    ok = temporal_cycles(build_temporal_graph(ev([(0, "A", "B"), (5, "B", "C"), (9, "C", "A")])))
    assert ok and set(ok[0]["accounts"]) == {"U:A", "U:B", "U:C"}
    # the loop closes only if money can move in time order: C->A happened BEFORE A->B and B->C
    late = temporal_cycles(build_temporal_graph(ev([(10, "A", "B"), (20, "B", "C"), (0, "C", "A")])), window=15 * 60)
    assert not late


def test_page_hinkley_detects_an_upward_shift_only():
    import numpy as np
    from src.drift import PageHinkley
    rng = np.random.default_rng(0)
    ph = PageHinkley(delta=0.005, threshold=15.0)
    calm = [ph.update(float(x)) for x in rng.random(3000) < 0.02]
    assert not any(calm)
    shifted = [ph.update(float(x)) for x in rng.random(2000) < 0.10]
    assert any(shifted)


def test_drift_report_flags_a_shifted_window():
    import numpy as np
    from src.drift import drift_report
    rng = np.random.default_rng(1)
    ts = pd.date_range("2026-09-01", periods=4000, freq="min").strftime("%Y-%m-%d %H:%M:%S")
    amt = np.r_[rng.lognormal(6, 0.5, 2560), rng.lognormal(8, 0.5, 1440)]          # last 24 h: much larger amounts
    ev = pd.DataFrame({"ts": ts, "amount": amt, "decision": "ALLOW", "risk": rng.random(4000) * 0.1,
                       "merchant_id": [f"M{i % 50}" for i in range(4000)], "device_id": "UNKNOWN", "user_id": "U"})
    rep = drift_report(ev)
    assert rep["psi"]["amount"] > 0.25 and rep["status"] == "significant shift"


def test_copilot_routes_questions_and_separates_facts_from_hypotheses():
    from src.copilot_qa import answer, route, to_markdown
    assert route("Why was this flagged?") == "why" and route("show me the money chain") == "chain"
    assert route("which ones should I investigate next") == "next" and route("anything similar in the past?") == "similar"
    pack = {"transaction": {"txn_id": "T", "ts": "2026-09-01 10:00:00", "user_id": "U1", "merchant_id": "P2P_U9", "device_id": "D", "amount": 5000},
            "model": {"decision": "REVIEW", "risk": 0.4, "supervised_probability": 0.2, "tripwires": ["payee-graph"],
                      "reason_codes": [{"feature": "policy:ring_fan_in", "reason": "Fraud-ring signal: collector", "value": "REVIEW"}],
                      "risk_breakdown": [{"component": "Supervised model", "contribution": 1.2}]},
            "customer_history": {"payments_last_30d": 12, "median_amount": 500, "this_amount_vs_median": 10.0,
                                 "device_seen_before": False, "payee_paid_before": False, "devices_seen": 1},
            "network_findings": {}, "money_chains": [], "similar_cases": [], "connected_flagged_entities": []}
    a = answer("behaviour", {"pack": pack, "rings": [], "timeline": []})
    assert any("10.0x" in v for v in a["verified"]) and a["hypotheses"]
    md = to_markdown(a)
    assert "Verified from the data" in md and "Hypotheses (not established)" in md
    assert answer("next", {"pack": pack, "open_cases": []})["verified"][0].startswith("No other open cases")


def test_takeover_rule_raises_to_review_only():
    from webapp.policy import apply_rules
    base = dict(amount=900, payee="M1", customer_known=True, payee_known=True, past_payments=50, median_amount=800,
                max_amount=5000, new_device=True, model_decision="ALLOW")
    assert apply_rules(**base)["decision"] == "ALLOW"
    r = apply_rules(**base, takeover=True)
    assert r["decision"] == "REVIEW" and r["rules"][0]["rule"] == "takeover_pattern"
    assert apply_rules(**{**base, "model_decision": "BLOCK"}, takeover=True)["decision"] == "BLOCK"


def test_a_single_customers_burst_is_not_a_ring():
    rows = [{"txn_id": f"T{i}", "ts": t(i), "user_id": "SOLO", "device_id": "D1", "merchant_id": f"SHOP{i}",
             "amount": 900.0, "probability": 0.5, "decision": "REVIEW"} for i in range(6)]
    assert analyse(pd.DataFrame(rows))["rings"] == []
