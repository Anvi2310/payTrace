"""Investigation copilot: answers an analyst's question about one case from application data only.

Each answer is assembled deterministically from stored records (the case's scored event, the customer's history,
the payment network, fraud rings, the evidence timeline and similar past cases) and split into
  verified  facts read from the data (counts, amounts, decisions, labels), each traceable to a record
  hypotheses  interpretations that would explain the facts; they are NOT established and are labelled as such
An optional LLM may only rewrite these two lists into prose; it receives nothing else, cannot add records, and never
makes or changes a payment decision. If the LLM is off, slow or fails, the deterministic answer is returned.
"""
from __future__ import annotations

QUESTIONS = {
    "why": "Why was this transaction flagged?",
    "behaviour": "What changed in the customer's behaviour?",
    "beneficiary": "Is this beneficiary connected to other suspicious accounts?",
    "chain": "Show the transaction chain.",
    "evidence": "What evidence supports the fraud hypothesis?",
    "similar": "Are there similar historical cases?",
    "next": "Which transactions should be investigated next?",
}
KEYWORDS = {"why": ["why", "flag", "reason"], "behaviour": ["behav", "changed", "usual", "different"],
            "beneficiary": ["beneficiar", "payee", "connected", "linked", "account"], "chain": ["chain", "path", "flow", "hop"],
            "evidence": ["evidence", "support", "hypothes", "proof"], "similar": ["similar", "historical", "past case"],
            "next": ["next", "investigate", "priorit"]}


def route(question: str) -> str:
    q = (question or "").lower()
    if q in QUESTIONS:
        return q
    best = max(KEYWORDS, key=lambda k: sum(w in q for w in KEYWORDS[k]))
    return best if any(w in q for w in KEYWORDS[best]) else "why"


def _inr(x):
    return f"Rs {float(x):,.0f}"


def answer(intent: str, ctx: dict) -> dict:
    """ctx: pack (evidence_pack), rings, timeline, open_cases (other open cases), network (ego graph json)."""
    pack, rings, tl = ctx["pack"], ctx.get("rings") or [], ctx.get("timeline") or []
    t, m, c = pack["transaction"], pack["model"], pack["customer_history"]
    payee_node = ("U:" + t["merchant_id"][4:]) if str(t["merchant_id"]).startswith("P2P_") else "P:" + t["merchant_id"]
    V, Hy = [], []
    if intent == "why":
        V.append(f"Decision {m['decision']} at fused risk {m['risk']:.1%} (supervised model {m['supervised_probability']:.1%}).")
        for r in m["reason_codes"][:6]:
            V.append(f"Reason: {r['reason']}" + (f" (value {r['value']})" if r.get("value") not in (None, "") and not str(r.get('feature', '')).startswith("policy") else ""))
        if m["tripwires"]:
            V.append("Novelty detectors that fired: " + ", ".join(m["tripwires"]) + ".")
        top = [b for b in m.get("risk_breakdown", []) if b["contribution"] > 0][:3]
        if top:
            V.append("Largest risk contributions: " + ", ".join(f"{b['component']} (+{b['contribution']:.2f})" for b in top) + ".")
    elif intent == "behaviour":
        V.append(f"{c['payments_last_30d']} payments in the 30 days before; median {_inr(c['median_amount'] or 0)}.")
        if c.get("this_amount_vs_median"):
            V.append(f"This payment is {c['this_amount_vs_median']:.1f}x the customer's median amount.")
        V.append(f"Device seen before: {'yes' if c['device_seen_before'] else 'no'}; payee paid before: {'yes' if c['payee_paid_before'] else 'no'}; "
                 f"devices used: {c['devices_seen']}.")
        same_cust = [x for x in tl if x["kind"] == "same customer"]
        if same_cust:
            V.append(f"{len(same_cust)} other payment(s) by this customer in the 48 h before, totalling {_inr(sum(x['amount'] for x in same_cust))}.")
        if not c["device_seen_before"] or (c.get("this_amount_vs_median") or 0) >= 5:
            Hy.append("The change is consistent with someone else operating the account (account takeover), or with a genuine one-off purchase; contacting the customer through a known channel separates the two.")
    elif intent == "beneficiary":
        nf = pack.get("network_findings", {})
        for n, fs in nf.items():
            if fs and n == payee_node:
                V.extend(f"{n}: {f}" for f in fs)
        same_payee = [x for x in tl if x["kind"] == "same payee"]
        if same_payee:
            flagged = [x for x in same_payee if x["decision"] in ("REVIEW", "BLOCK")]
            V.append(f"{len(same_payee)} other payment(s) to this payee in the 48 h before ({len(flagged)} flagged), from "
                     f"{len({x['text'].split(' → ')[0] for x in same_payee})} different account(s).")
        pr = [r for r in rings if payee_node in r.get("hubs", []) or payee_node in str(r.get("evidence"))]
        for r in rings:
            V.append(f"Ring {r['id']} ({r['status']}): {r['size']} entities, {r['flagged_payments']} flagged payments; "
                     f"confirmed entities: {', '.join(r.get('confirmed_entities') or []) or 'none'}.")
        conn = pack.get("connected_flagged_entities") or []
        if conn:
            V.append(f"{len(conn)} connected entities with flagged activity in the last 7 days.")
        if not V:
            V.append("No other suspicious accounts are linked to this beneficiary in the last 7 days of scored payments.")
        elif rings and not any(r.get("confirmed_entities") for r in rings):
            Hy.append("The beneficiary may be a collection (mule) account; no entity in its ring is confirmed as fraud yet, so this is unproven.")
    elif intent == "chain":
        chains = pack.get("money_chains") or []
        if chains:
            for ch in chains[:3]:
                V.append("Chain: " + " → ".join([ch[0]["from"]] + [h["to"] for h in ch]) + " ("
                         + ", ".join(f"{_inr(h['amount'])} at {h['ts'][11:16]}" for h in ch) + ")")
        else:
            V.append("No person-to-person chain (money received and sent on within 60 minutes) involves this case's accounts.")
        for x in [x for x in tl if x["kind"] != "analyst label"][:8]:
            V.append(f"{x['ts']} · {x['kind']} · {x['text']} · {_inr(x['amount'])} · {x['decision']}")
    elif intent == "evidence":
        V.append(f"Model: {m['decision']} at {m['risk']:.1%}; supervised {m['supervised_probability']:.1%}.")
        pol = [r["reason"] for r in m["reason_codes"] if str(r.get("feature", "")).startswith("policy")]
        V.extend(pol[:4])
        sim = pack.get("similar_cases") or []
        fr = [s for s in sim if str(s["outcome"]).startswith("fraud")]
        if sim:
            V.append(f"{len(fr)} of the {len(sim)} most similar past alerts with known outcome were fraud.")
        for r in rings:
            V.append(f"Member of ring {r['id']} ({r['status']}).")
        labels = [x for x in tl if x["kind"] == "analyst label"]
        V.extend(x["text"] for x in labels)
        Hy.append("Supporting the fraud hypothesis: " + ("the signals above. " if V else "") +
                  "Against it: anything in the customer's own history that makes the payment ordinary (the 'behaviour' answer), "
                  "and the absence of confirmed fraud links. No label has been set until an analyst decides.")
    elif intent == "similar":
        sim = pack.get("similar_cases") or []
        if not sim:
            V.append("No comparable past alerts with a known outcome yet.")
        for s in sim:
            V.append(f"{s['txn_id']} · {s['ts'][:16]} · {_inr(s['amount'])} · {s['decision']} · outcome {s['outcome']}")
    elif intent == "next":
        oc = ctx.get("open_cases") or []
        if not oc:
            V.append("No other open cases share this case's customer, payee, device or ring.")
        for o in oc[:8]:
            V.append(f"{o['txn_id']} · {o['why']} · {_inr(o['amount'])} · {o['decision']} at {o['risk']:.0%}")
        if oc:
            Hy.append("Ordered by link strength (same ring, payee or device first) and then risk; cases in a ring with confirmed fraud come first.")
    return {"intent": intent, "question": QUESTIONS[intent], "verified": V, "hypotheses": Hy}


def to_markdown(a: dict) -> str:
    out = [f"**{a['question']}**", "", "**Verified from the data**"] + [f"- {v}" for v in a["verified"]]
    if a["hypotheses"]:
        out += ["", "**Hypotheses (not established)**"] + [f"- {h}" for h in a["hypotheses"]]
    return "\n".join(out)


SYSTEM = ("You help a fraud analyst. Rewrite the VERIFIED facts and HYPOTHESES below as a short answer to the question. "
          "Rules: use only these facts; do not add numbers, accounts or events; keep the two sections separate with the "
          "headings 'Verified from the data' and 'Hypotheses (not established)'; never recommend allowing or blocking "
          "the payment, the analyst decides. Markdown, under 160 words.")


def answer_text(a: dict, use_llm: bool = False) -> tuple[str, str]:
    base = to_markdown(a)
    if not use_llm:
        return base, "template"
    try:
        from src.llm import complete
        txt, src = complete(SYSTEM, base, max_tokens=500, timeout=12)
        if txt and "Verified" in txt:
            return txt, src
    except Exception:
        pass
    return base, "template"
