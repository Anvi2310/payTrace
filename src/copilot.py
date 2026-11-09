"""Analyst copilot: turns a scored alert into a short investigation note.

- Always available: a deterministic template note (zero latency, used in the stream).
- Optional: an LLM-written note when GEMINI_API_KEY is set.
  The LLM only rephrases facts we pass in (amount, reason codes, feature values); it never
  makes the decision, so a model hallucination cannot block or allow a payment.
"""
import json
import os

import requests

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def template_note(txn, result):
    f = result["features"]
    bits = [f"{result['decision']} · risk {result['probability']:.0%} · ₹{float(txn['amount']):,.0f} "
            f"to {txn['merchant_id']} ({txn['category']}, {txn['channel']})"]
    facts = []
    if f["u_new_device"]:
        facts.append("first time on this device")
    if f["u_new_city"]:
        facts.append(f"new location ({txn['city']})")
    if f["u_amt_ratio_mean"] >= 3:
        facts.append(f"{f['u_amt_ratio_mean']:.0f}× the user's typical amount")
    if f["u_cnt_10m"] >= 3:
        facts.append(f"{int(f['u_cnt_10m']) + 1} transactions in 10 min")
    if f["u_small_cnt_10m"] >= 3:
        facts.append(f"{int(f['u_small_cnt_10m'])} tiny test payments just before")
    if f["d_n_users"] >= 3:
        facts.append(f"device shared by {int(f['d_n_users'])} accounts")
    if f["m_users_24h"] >= 5 and f["m_age_hours"] < 72:
        facts.append(f"payee is {f['m_age_hours']:.0f}h old and paid by {int(f['m_users_24h'])} users today")
    elif f["u_new_merchant"]:
        facts.append("first payment to this payee")
    if facts:
        bits.append("Signals: " + "; ".join(facts) + ".")
    action = {"BLOCK": "Hold payment and trigger step-up verification (OTP / call-back).",
              "REVIEW": "Queue for analyst review; allow with friction if customer confirms.",
              "ALLOW": "No action."}[result["decision"]]
    bits.append("Suggested action: " + action)
    return " ".join(bits)


def _prompt(txn, result):
    facts = {"transaction": {k: txn[k] for k in ("amount", "merchant_id", "category", "channel", "city")},
             "decision": result["decision"], "fraud_probability": round(result["probability"], 3),
             "top_reasons": [r["reason"] for r in result["reasons"]],
             "key_features": {k: round(float(v), 2) for k, v in result["features"].items()
                              if k.startswith(("u_", "d_", "m_"))}}
    return ("You are a payments fraud analyst assistant. Using ONLY the facts below, write a "
            "3-sentence alert note: what happened, why it looks risky, and the recommended next "
            "step. Do not invent facts.\n\n" + json.dumps(facts, default=str))


def llm_note(txn, result, timeout=15):
    try:
        if os.getenv("GEMINI_API_KEY"):
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
                   f":generateContent?key={os.environ['GEMINI_API_KEY']}")
            r = requests.post(url, json={"contents": [{"parts": [{"text": _prompt(txn, result)}]}]},
                              timeout=timeout)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:                       # never let the copilot break the pipeline
        return template_note(txn, result) + f"  [LLM unavailable: {type(e).__name__}]"
    return template_note(txn, result)
