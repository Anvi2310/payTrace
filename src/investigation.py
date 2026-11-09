"""Evidence-grounded investigation copilot.

`evidence_pack()` gathers ONLY facts the application can verify: the transaction, the model's decision and its
reasons, the customer's history, connected entities in the payment graph, money chains and similar past cases.
`summarise()` turns that pack into an investigation summary, with an optional LLM that may only rephrase the pack.

Hard rules (also given to the LLM):
  * the copilot never approves, rejects or blocks a payment; decisions come from the scoring engine and analysts
  * it never states that a person or account is guilty; it reports risk indicators and their counts
  * every figure in a summary comes from the evidence pack
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src import settings
from src.graph_intel import build_graph, ego, entity_findings, money_chains, node_of_payee

LABEL_DELAY = timedelta(days=3)          # outcomes of past cases become known after this (chargeback delay)
_HISTORY = {}


def dataset_history() -> pd.DataFrame:
    """Payments before the streamed period (what the bank already knew), loaded once."""
    if "df" not in _HISTORY:
        from src.train_v2 import dataset_path, slices
        df = pd.read_csv(dataset_path(settings.DATASET), parse_dates=["timestamp"],
                         usecols=["timestamp", "user_id", "amount", "merchant_id", "device_id"])
        _HISTORY["df"] = df[~slices(df)["test"]]
    return _HISTORY["df"]


def _events(con, until_ts: str, days: int = 7) -> pd.DataFrame:
    since = (datetime.fromisoformat(until_ts[:19]) - timedelta(days=days)).isoformat(sep=" ")
    return pd.read_sql("SELECT txn_id, ts, user_id, device_id, merchant_id, amount, risk AS probability, decision, "
                       "is_fraud, fraud_type FROM events WHERE ts>=? AND ts<=?", con, params=(since, until_ts))


def evidence_pack(con, txn_id: str, history: pd.DataFrame | None = None) -> dict:
    row = con.execute("SELECT * FROM events WHERE txn_id=?", (txn_id,)).fetchone()
    if row is None:
        raise KeyError(txn_id)
    e = dict(row)
    ts = e["ts"]
    reasons = json.loads(e["reasons"] or "[]")
    breakdown = json.loads(e["breakdown"] or "[]")

    # customer history (dataset history before the stream + streamed events), strictly before this payment
    win = _events(con, ts, days=30)
    hist = win[(win.user_id == e["user_id"]) & (win.ts < ts)]
    if history is None:
        history = dataset_history()
    since30 = pd.Timestamp(ts[:19]) - pd.Timedelta(days=30)
    history = history[history.timestamp >= since30]
    if len(history):
        h0 = history[(history.user_id == e["user_id"]) & (history.timestamp.astype(str) < ts)]
        hist = pd.concat([h0.rename(columns={"timestamp": "ts"})[["ts", "amount", "merchant_id", "device_id"]],
                          hist[["ts", "amount", "merchant_id", "device_id"]]])
    customer = {"payments_last_30d": int(len(hist)),
                "median_amount": float(hist.amount.median()) if len(hist) else None,
                "this_amount_vs_median": float(e["amount"] / hist.amount.median()) if len(hist) and hist.amount.median() > 0 else None,
                "devices_seen": int(hist.device_id.nunique()) if len(hist) else 0,
                "device_seen_before": bool((hist.device_id == e["device_id"]).any()) if len(hist) else False,
                "payee_paid_before": bool((hist.merchant_id == e["merchant_id"]).any()) if len(hist) else False}

    # network evidence from the last 7 days of scored events, up to this payment
    ev7 = _events(con, ts, days=7)
    G = build_graph(ev7)
    nodes = [f"U:{e['user_id']}", f"D:{e['device_id']}", node_of_payee(e["merchant_id"])]
    network = {n: entity_findings(G, n) for n in nodes if n in G}
    H = ego(G, nodes, radius=2, max_nodes=40)
    connected_flagged = sorted({n for n in H.nodes if n not in nodes and H.nodes[n]["n_flagged"] > 0})
    chains = [c for c in money_chains(ev7) if any(h["txn_id"] == txn_id for h in c)]

    # similar past cases: nearest flagged events by feature vector, only those old enough to have a known outcome
    similar = []
    f0 = json.loads(e["features"]) if e["features"] else None
    cutoff = (datetime.fromisoformat(ts[:19]) - LABEL_DELAY).isoformat(sep=" ")
    if f0:
        past = pd.read_sql("SELECT txn_id, ts, amount, decision, risk, features, is_fraud, fraud_type FROM events "
                           "WHERE features IS NOT NULL AND ts<=? ORDER BY ts DESC LIMIT 4000", con, params=(cutoff,))
        fb = dict(con.execute("SELECT txn_id, label FROM feedback WHERE status='accepted'").fetchall())
        if len(past):
            keys = sorted(f0)
            M = np.array([[json.loads(f).get(k, 0.0) for k in keys] for f in past.features])
            v = np.array([f0[k] for k in keys])
            sd = M.std(0)
            sd = np.where(sd < 1e-6, 1.0, sd)            # constant features must not dominate the distance
            d = np.sqrt((np.clip((M - v) / sd, -10, 10) ** 2).mean(1))
            for i in np.argsort(d)[:5]:
                r = past.iloc[i]
                outcome = ("fraud" if fb[r.txn_id] else "genuine") + " (analyst)" if r.txn_id in fb else \
                          ("fraud" if r.is_fraud == 1 else "genuine") + " (replay label)" if r.is_fraud in (0, 1) else \
                          "unknown (no label yet)"
                similar.append({"txn_id": r.txn_id, "ts": r.ts, "amount": float(r.amount), "decision": r.decision,
                                "risk": float(r.risk), "outcome": outcome, "distance": float(d[i])})

    steps = next_steps(e, customer, network, chains)
    return {"transaction": {k: e[k] for k in ("txn_id", "ts", "user_id", "amount", "merchant_id", "category", "channel",
                                              "device_id", "city")},
            "model": {"decision": e["decision"], "risk": float(e["risk"]), "supervised_probability": float(e["supervised_prob"]),
                      "tripwires": [t for t in (e["tripwires"] or "").split(",") if t], "reason_codes": reasons,
                      "risk_breakdown": breakdown, "model_version": e["model_version"]},
            "customer_history": customer, "network_findings": network, "connected_flagged_entities": connected_flagged[:15],
            "money_chains": chains, "similar_cases": similar, "suggested_next_steps": steps,
            "replay_ground_truth": {"is_fraud": e["is_fraud"], "fraud_type": e["fraud_type"],
                                    "note": "synthetic replay label, shown for evaluation only"}}


def next_steps(e, customer, network, chains) -> list[str]:
    steps = ["Contact the customer through their registered channel (not contact details found in this transaction)."]
    txt = " ".join(sum(network.values(), []))
    if "different accounts" in txt and "Device" in txt:
        steps.append(f"Review the other accounts linked to device {e['device_id']} for signs of takeover or a coordinated ring.")
    if "Received payments from" in txt:
        steps.append(f"Review payee {e['merchant_id']}: consider a beneficiary hold pending verification.")
    if chains:
        steps.append("Trace the onward transfers in the money chain and alert the receiving institutions if they are external.")
    if not customer.get("device_seen_before") and customer.get("payments_last_30d", 0) > 0:
        steps.append("Verify the new device with step-up authentication before further payments.")
    if customer.get("payments_last_30d", 0) == 0:
        steps.append("New customer: re-check KYC documents.")
    return steps


SYSTEM = ("You are a payments fraud investigation assistant. Use ONLY the JSON evidence provided. Do not invent facts, "
          "names, amounts or dates. Never say a customer or account is guilty or a fraudster; describe risk indicators. "
          "Never approve, reject or block a payment; you only summarise evidence for a human analyst. Write: 1) a two-"
          "sentence summary, 2) the key evidence as short bullet points, 3) recommended next investigative steps.")


def template_summary(pack: dict) -> str:
    t, m, c = pack["transaction"], pack["model"], pack["customer_history"]
    lines = [f"**{m['decision']}** recommendation, risk {m['risk']:.0%}: ₹{t['amount']:,.0f} from {t['user_id']} to "
             f"{t['merchant_id']} ({t['category']}, {t['channel']}) at {t['ts']}."]
    ev = [r["reason"] for r in m["reason_codes"]]
    if c["this_amount_vs_median"]:
        ev.append(f"Amount is {c['this_amount_vs_median']:.1f}× the customer's 30-day median")
    if not c["device_seen_before"] and c["payments_last_30d"]:
        ev.append("Device not seen for this customer in the last 30 days")
    for node, fs in pack["network_findings"].items():
        ev += [f"{node}: {f}" for f in fs]
    if pack["money_chains"]:
        ev.append(f"Part of a money chain with {len(pack['money_chains'][0])} hops within an hour")
    if pack["similar_cases"]:
        known = [s for s in pack["similar_cases"] if s["outcome"].startswith("fraud")]
        ev.append(f"{len(known)} of the {len(pack['similar_cases'])} most similar past alerts were confirmed fraud")
    lines.append("**Evidence**\n" + "\n".join(f"- {x}" for x in ev))
    lines.append("**Suggested next steps**\n" + "\n".join(f"- {s}" for s in pack["suggested_next_steps"]))
    lines.append("_Risk indicators only; this is not a finding of guilt. The payment decision rests with the payment "
                 "system and the analyst._")
    return "\n\n".join(lines)


def summarise(pack: dict, use_llm: bool = True, timeout: int = 20) -> tuple[str, str]:
    """Returns (summary, source): source is 'template' or the LLM provider that wrote it from the evidence pack."""
    from src import llm
    evidence = json.dumps({k: v for k, v in pack.items() if k != "replay_ground_truth"}, default=str)
    if use_llm and llm.config()["provider"]:
        try:
            text, provider = llm.complete(llm.ANALYST_SYSTEM, evidence, max_tokens=1400)
            return text, provider
        except llm.LLMError as e:
            return template_summary(pack) + f"\n\n_(LLM unavailable: {str(e)[:120]}; template summary shown.)_", "template"
    return template_summary(pack), "template"


def case_report(pack: dict, summary: str, analyst: str) -> str:
    t, m = pack["transaction"], pack["model"]
    rows = "\n".join(f"| {r['reason']} | {r.get('value', '')} | "
                     f"{r['impact'] if r.get('impact') is not None else ('policy rule' if str(r.get('feature', '')).startswith('policy:') else 'tripwire')} |"
                     for r in m["reason_codes"])
    bd = "\n".join(f"| {b['component']} | {b['contribution']:+.2f} |" for b in m["risk_breakdown"])
    sim = "\n".join(f"| {s['txn_id']} | {s['ts']} | ₹{s['amount']:,.0f} | {s['decision']} | {s['outcome']} |"
                    for s in pack["similar_cases"])
    return f"""# payTrace case report: {t['txn_id']}

Generated {datetime.now():%Y-%m-%d %H:%M} by {analyst}. Model version {m['model_version']}.
Risk indicators only; not a finding of guilt. Payment authorisation is outside payTrace.

## Transaction
| Field | Value |
|---|---|
""" + "\n".join(f"| {k} | {v} |" for k, v in t.items()) + f"""

## Model recommendation
**{m['decision']}**, fused risk {m['risk']:.1%}, supervised probability {m['supervised_probability']:.1%}.
Novelty tripwires: {', '.join(m['tripwires']) or 'none'}.

| Reason code | Value | SHAP impact |
|---|---|---|
{rows}

| Risk component | Contribution (log-odds) |
|---|---|
{bd}

## Summary
{summary}

## Similar past cases (outcomes known)
| Transaction | Time | Amount | Decision | Outcome |
|---|---|---|---|---|
{sim or '| none | | | | |'}

## Network findings
""" + "\n".join(f"- **{n}**: " + "; ".join(fs) for n, fs in pack["network_findings"].items() if fs) + "\n"
