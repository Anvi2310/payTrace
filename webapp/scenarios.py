"""Scenario lab: SIMULATED fraud scenarios played through the live engine, for demonstrations.

Each scenario is a short script of payments built from the app's real customers and merchants (plus clearly named
simulated mule, QR and device-farm entities). Every payment goes through exactly the same path as a customer payment
(ENGINE.submit_payment: limits -> features -> models -> fusion -> policy -> fraud-ring signals -> decision), so the
decisions are the system's own: nothing here sets or expects an outcome. Payments are stored with source='scenario'
and the payee/device names start with SIM_ where they are invented, so they are never mistaken for real activity.

Time is compressed: steps run about a second apart in real time, which the engine sees as seconds apart. Bursts and
chains are therefore naturally fast; the 'normal' scenario uses different customers so it is not a burst.
"""
from __future__ import annotations

import random
import threading
import time

SCENARIOS = {
    "normal": ("Normal customer payments", "Three customers pay merchants they use regularly, for typical amounts."),
    "account_takeover": ("Account takeover", "A customer's account is used from a new device in a new city: a burst of "
                         "purchases at merchants they have never used, with rising amounts."),
    "card_testing": ("Card testing", "A stolen card is checked with tiny online payments to different merchants, then "
                     "used for a large purchase."),
    "upi_mule": ("UPI mule account", "Several unrelated customers send money to the same newly active account, which "
                 "forwards it on within minutes."),
    "qr_fraud": ("QR payment fraud", "A fake QR code is pasted over a shop's code: many customers pay a brand-new payee "
                 "within minutes."),
    "suspicious_beneficiary": ("Suspicious beneficiary", "A customer sends a large amount to a beneficiary the bank "
                               "has no record of."),
    "shared_device": ("Shared-device attack", "One device (a device farm) is used to operate several different customer "
                      "accounts that all pay the same payee."),
    "rapid_transfers": ("Rapid money transfers", "Money hops through a chain of accounts within minutes, each passing on "
                        "most of what it received."),
    "fraud_ring": ("Coordinated fraud ring", "Apparently unrelated customers, sharing two devices, pay two collector "
                   "accounts; the collectors move money between each other in a loop."),
    "unseen_pattern": ("Previously unseen pattern", "A pattern absent from the app model's training data (Sparkov card "
                       "payments contain no person-to-person transfers at all): small round-amount transfers at night "
                       "circulate through a loop of accounts. Only the anomaly detectors and graph signals can see it."),
}


def _pick_customers(engine, n, rng):
    prof = engine.profiles
    pool = prof[prof.payments >= 200].index.tolist() or prof.index.tolist()
    return rng.sample(pool, n)


def _usual(engine, uid):
    p = engine.profile(uid)
    payees = p["saved_payees"] or [engine.examples["merchant"]]
    return p, payees


def _new_merchants(engine, uid, rng, n, online=False):
    seen = set(engine.user_history(uid).merchant_id.unique())
    cats = [c for c in engine.categories if c.endswith("_net")] if online else engine.categories
    cand = [m for m, c in engine.merchant_category.items() if m not in seen and (c in cats)]
    return rng.sample(cand, min(n, len(cand)))


def build(name: str, engine, seed: int) -> list[dict]:
    """The payments of one scenario (inputs only, no outcomes)."""
    rng = random.Random(seed)
    tag = f"{seed % 10000:04d}"
    steps = []

    def pay(uid, payee, amount, device=None, city=None, channel=None, note=""):
        p = engine.profile(uid)
        steps.append({"user_id": uid, "payee": payee, "amount": round(float(amount), 2), "category": "",
                      "channel": channel or "", "device_id": device or (p["usual_device"] or f"SIM_DEV_HOME_{uid}"),
                      "city": city or p["usual_city"] or "UNKNOWN", "note": note})

    if name == "normal":
        for uid in _pick_customers(engine, 3, rng):
            p, payees = _usual(engine, uid)
            pay(uid, payees[0], max(50, round((p["usual_amount"] or 500) * rng.uniform(0.6, 1.2))), note="usual merchant, usual amount")
    elif name == "account_takeover":
        uid = _pick_customers(engine, 1, rng)[0]
        p, _ = _usual(engine, uid)
        base = p["usual_amount"] or 1000
        far = next((c for c in engine.cities if c != p["usual_city"]), "Lagos")
        _, usual = _usual(engine, uid)
        pay(uid, usual[0], base, note="the real customer pays as usual from their own phone")
        for i, m in enumerate(_new_merchants(engine, uid, rng, 4, online=True)):
            pay(uid, m, base * (1.5 + 2.5 * i), device=f"SIM_DEV_ATTACKER_{tag}", city=far, note=f"attacker purchase {i + 1}")
    elif name == "card_testing":
        uid = _pick_customers(engine, 1, rng)[0]
        p, _ = _usual(engine, uid)
        ms = _new_merchants(engine, uid, rng, 6, online=True)
        for i, m in enumerate(ms[:5]):
            pay(uid, m, rng.choice([10, 19, 25, 49, 1]), device=f"SIM_DEV_BOT_{tag}", note=f"test charge {i + 1}")
        pay(uid, ms[5], max(30000, (p["largest_amount"] or 20000) * 1.2), device=f"SIM_DEV_BOT_{tag}", note="cash-out purchase")
    elif name == "upi_mule":
        victims = _pick_customers(engine, 6, rng)
        mule, onward = victims[0], victims[1]
        for v in victims[2:]:
            pay(v, f"P2P_{mule}", rng.choice([9500, 12000, 15000, 18000]), note=f"victim pays mule {mule}")
        pay(mule, f"P2P_{onward}", 48000, device=f"SIM_DEV_MULE_{tag}", note="mule forwards the money")
    elif name == "qr_fraud":
        qr = f"SIM_QR_{tag}"
        for v in _pick_customers(engine, 7, rng):
            p, _ = _usual(engine, v)
            pay(v, qr, round(rng.uniform(150, 900)), note="customer scans the swapped QR code")
    elif name == "suspicious_beneficiary":
        uid = _pick_customers(engine, 1, rng)[0]
        pay(uid, f"SIM_BENEFICIARY_{tag}", 85000, note="large transfer to an unknown beneficiary")
    elif name == "shared_device":
        farm = f"SIM_DEV_FARM_{tag}"
        target = f"SIM_PAYEE_{tag}"
        for v in _pick_customers(engine, 4, rng):
            pay(v, target, rng.choice([4999, 7500, 9999]), device=farm, note="account operated from the device farm")
    elif name == "rapid_transfers":
        chain = _pick_customers(engine, 5, rng)
        amt = 60000
        for a, b in zip(chain, chain[1:]):
            pay(a, f"P2P_{b}", amt, device=f"SIM_DEV_{a}_{tag}", note=f"hop {a} -> {b}")
            amt = round(amt * rng.uniform(0.9, 0.97))
    elif name == "fraud_ring":
        members = _pick_customers(engine, 8, rng)
        c1, c2, payers = members[0], members[1], members[2:]
        devs = [f"SIM_DEV_RING_A_{tag}", f"SIM_DEV_RING_B_{tag}"]
        for i, v in enumerate(payers):
            pay(v, f"P2P_{c1 if i % 2 == 0 else c2}", rng.choice([6000, 8000, 11000]), device=devs[i % 2],
                note=f"ring member pays collector {'A' if i % 2 == 0 else 'B'}")
        pay(c1, f"P2P_{c2}", 20000, device=devs[0], note="collector A -> collector B")
        pay(c2, f"P2P_{c1}", 19000, device=devs[1], note="collector B -> collector A (loop)")
    elif name == "unseen_pattern":
        loop = _pick_customers(engine, 4, rng)
        for a, b in zip(loop, loop[1:] + loop[:1]):
            pay(a, f"P2P_{b}", 5000, device=f"SIM_DEV_{a}_{tag}", note=f"round-amount loop transfer {a} -> {b}")
    else:
        raise KeyError(name)
    return steps


class ScenarioRunner:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.thread = None
        self.state = {"running": False, "scenario": None, "steps": [], "error": None, "run": 0}

    def status(self) -> dict:
        with self.lock:
            return {**self.state, "steps": list(self.state["steps"]),
                    "catalog": [{"id": k, "title": t, "description": d} for k, (t, d) in SCENARIOS.items()]}

    def start(self, name: str, actor: str, delay: float = 1.0):
        if name not in SCENARIOS:
            raise KeyError(name)
        if self.thread and self.thread.is_alive():
            raise RuntimeError("a scenario is already running")
        with self.lock:
            self.state.update(running=True, scenario=name, steps=[], error=None, run=self.state["run"] + 1)
        seed = int(time.time())
        self.thread = threading.Thread(target=self._run, args=(name, actor, delay, seed), daemon=True)
        self.thread.start()

    def _run(self, name, actor, delay, seed):
        from src.scorer_v2 import InvalidTransaction
        try:
            plan = build(name, self.engine, seed)
            for i, st in enumerate(plan):
                note = st.pop("note")
                t0 = time.perf_counter()
                try:
                    res = self.engine.submit_payment(**st, source="scenario", actor=actor, role="admin", explain_with_llm=False)
                    out = {"step": i + 1, "note": note, "input": st, "simulated": True,
                           "txn_id": res["txn_id"], "ts": res["ts"], "decision": res["decision"],
                           "model_decision": res["model_decision"], "status": res["status"],
                           "scores": res["scores"], "features": res["features_view"], "ring": res["ring"],
                           "rules": res["policy"], "tripwires": res["tripwires"],
                           "reasons": [x["reason"] for x in res["reasons"][:6]], "explanation": res["plain_reasons"],
                           "message": res["message"], "case_created": res["decision"] != "ALLOW",
                           "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
                except InvalidTransaction as e:
                    out = {"step": i + 1, "note": note, "input": st, "simulated": True, "decision": "DECLINED",
                           "error": str(e), "case_created": False}
                with self.lock:
                    self.state["steps"].append(out)
                time.sleep(delay)
        except Exception as e:                                     # surfaced in the UI, never crashes the server
            self.state["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.state["running"] = False
