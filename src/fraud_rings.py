"""Temporal fraud-ring intelligence (payTrace 2.1). Extends src/graph_intel.py.

Two parts:

1. RingMonitor (online, per payment, past-only)
   Keeps a small time-windowed memory of the payment network and, for each payment BEFORE it is decided, reports
   ring signals that a single-transaction model cannot see:
     fan_in          the payee received money from many DIFFERENT senders in the last 24 h, mostly first-time payers,
                     and the payee is new (a collection account or fake QR merchant)
     shared_device   the device was used by several different accounts in the last 7 days (synthetic accounts, a
                     device farm, or account takeover across accounts)
     device_new_accounts  several accounts first seen in the last 24 h on this device
     pass_through    the sender received person-to-person money in the last hour and is now sending most of it on
     chain_depth     how many person-to-person hops the money has already made within the hour (mule chain)
     cycle           this transfer closes a loop: money can already reach the sender from the receiver (72 h)
     confirmed_link  the payee or device is linked to a payment an analyst CONFIRMED as fraud
   Ring signals are investigation candidates: they raise a decision to REVIEW. Only a confirmed-fraud link (or a ring
   signal on top of the model's own BLOCK) supports BLOCK. Nothing here is a verdict that something is fraud.

2. analyse(events) (offline, for analysts and admins)
   Builds a timestamped graph of customers, devices and payees and reports rings: Louvain communities, connected
   components, PageRank (where money concentrates), betweenness (brokers between groups), directed P2P cycles,
   fan-in collectors, shared devices and mule chains. Each ring lists the evidence, which entities are CONFIRMED
   fraud (analyst label) and which are only suspicious.
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd

from src.features import MISSING
from src.graph_intel import money_chains, node_of_payee

HOUR, DAY = 3600.0, 86400.0
# thresholds (documented in docs/PAYTRACE_2_IMPROVEMENTS.md; FPR measured on the synthetic UPI test period)
FAN_IN_SENDERS = 5            # distinct senders in 24 h
FAN_IN_NEW_SHARE = 0.8        # share of those that never paid this payee before
FAN_IN_PAYEE_AGE_H = 72.0     # payee first seen within this many hours
SHARED_DEVICE_ACCOUNTS = 3    # distinct accounts on one device in 7 days
DEVICE_NEW_ACCOUNTS = 2       # accounts first seen in the last 24 h on one device
PASS_THROUGH_SHARE = 0.5      # sends on at least this share of what arrived in the last hour
CYCLE_WINDOW = 72 * HOUR
CYCLE_MAX_LEN = 5
MAX_RING = 60                 # larger connected groups are split into Louvain communities


def _ts(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, datetime):
        return x.timestamp()
    return pd.Timestamp(str(x)[:19]).timestamp()


def _p2p_target(payee: str) -> str | None:
    p = str(payee)
    return p[4:] if p.startswith("P2P_") else None


class RingMonitor:
    """Online ring signals. Call check(txn) to read the signals for a payment, then add(txn) once it is processed.
    Both are O(window) per payment and keep only bounded recent history."""

    def __init__(self):
        self.payee_in = defaultdict(deque)          # payee -> (t, sender, was first payment to it)  24 h
        self.payee_payers = defaultdict(set)        # payee -> every sender ever
        self.payee_first = {}                       # payee -> first seen t
        self.dev_users = defaultdict(deque)         # device -> (t, user)              7 d
        self.user_first = {}                        # user  -> first seen t
        self.p2p_in = defaultdict(deque)            # receiver -> (t, amount, depth)   1 h
        self.p2p_out = defaultdict(deque)           # sender -> (t, receiver)          72 h (for cycles)
        self.conf_payees, self.conf_devices, self.conf_users = set(), set(), set()

    # ---------------- labels from analysts / chargebacks
    def confirm_fraud(self, payee=None, device=None, user=None):
        if payee:
            self.conf_payees.add(str(payee))
        if device and str(device) not in MISSING:
            self.conf_devices.add(str(device))
        if user:
            self.conf_users.add(str(user))

    # ---------------- reading
    @staticmethod
    def _trim(dq, t, width):
        while dq and dq[0][0] < t - width:
            dq.popleft()

    def check(self, txn: dict) -> dict:
        t = _ts(txn["timestamp"])
        u, payee, dev, amt = str(txn["user_id"]), str(txn["merchant_id"]), str(txn.get("device_id")), float(txn["amount"])
        sig = {}
        # fan-in to the payee (including this payment)
        dq = self.payee_in[payee]; self._trim(dq, t, DAY)
        first = {s: f for _, s, f in dq}
        first[u] = first.get(u, False) or u not in self.payee_payers.get(payee, set())
        senders = set(first)
        first_time = [s for s, f in first.items() if f]
        age_h = (t - self.payee_first.get(payee, t)) / HOUR
        sig["payee_senders_24h"] = len(senders)
        sig["payee_new_sender_share_24h"] = len(first_time) / max(len(senders), 1)
        sig["payee_age_hours"] = age_h
        sig["fan_in"] = (len(senders) >= FAN_IN_SENDERS and sig["payee_new_sender_share_24h"] >= FAN_IN_NEW_SHARE
                         and age_h <= FAN_IN_PAYEE_AGE_H)
        # device sharing
        if dev not in MISSING:
            dd = self.dev_users[dev]; self._trim(dd, t, 7 * DAY)
            users = {x for _, x in dd} | {u}
            new_acc = {x for x in users if t - self.user_first.get(x, t) <= DAY}
            sig["device_accounts_7d"] = len(users)
            sig["device_new_accounts_24h"] = len(new_acc)
            sig["shared_device"] = len(users) >= SHARED_DEVICE_ACCOUNTS
            sig["device_new_accounts"] = len(new_acc) >= DEVICE_NEW_ACCOUNTS
        else:
            sig.update(device_accounts_7d=0, device_new_accounts_24h=0, shared_device=False, device_new_accounts=False)
        # money in -> money out (mule behaviour) and chain depth
        di = self.p2p_in[u]; self._trim(di, t, HOUR)
        recv = sum(a for _, a, _ in di)
        depth = max((d for _, _, d in di), default=0)
        sig["received_p2p_1h"] = recv
        sig["chain_depth"] = depth
        sig["pass_through"] = recv > 0 and amt >= PASS_THROUGH_SHARE * recv and _p2p_target(payee) is not None
        # cycle: can money already flow from the receiver back to the sender?
        tgt = _p2p_target(payee)
        sig["cycle"] = bool(tgt) and self._reaches(tgt, u, t)
        # confirmed-fraud links
        sig["confirmed_link"] = payee in self.conf_payees or dev in self.conf_devices or (tgt in self.conf_users if tgt else False)
        sig["ring_rules"] = [k for k in ("fan_in", "shared_device", "device_new_accounts", "pass_through", "cycle",
                                         "confirmed_link") if sig.get(k)]
        if sig["chain_depth"] >= 2 and "pass_through" in sig["ring_rules"]:
            sig["ring_rules"].append("mule_chain")
        return sig

    def _reaches(self, src: str, dst: str, t: float) -> bool:
        seen, frontier = {src}, [src]
        for _ in range(CYCLE_MAX_LEN - 1):
            nxt = []
            for n in frontier:
                dq = self.p2p_out.get(n)
                if not dq:
                    continue
                self._trim(dq, t, CYCLE_WINDOW)
                for _, r in dq:
                    if r == dst:
                        return True
                    if r not in seen:
                        seen.add(r); nxt.append(r)
            frontier = nxt
            if not frontier:
                break
        return False

    # ---------------- writing
    def add(self, txn: dict, sig: dict | None = None):
        t = _ts(txn["timestamp"])
        u, payee, dev, amt = str(txn["user_id"]), str(txn["merchant_id"]), str(txn.get("device_id")), float(txn["amount"])
        self.user_first.setdefault(u, t)
        self.payee_first.setdefault(payee, t)
        self.payee_in[payee].append((t, u, u not in self.payee_payers[payee]))
        self.payee_payers[payee].add(u)
        if dev not in MISSING:
            self.dev_users[dev].append((t, u))
        tgt = _p2p_target(payee)
        if tgt:
            self.user_first.setdefault(tgt, t)
            depth = (sig or {}).get("chain_depth", 0) + 1 if (sig or {}).get("received_p2p_1h", 0) > 0 else 1
            self.p2p_in[tgt].append((t, amt, depth))
            self.p2p_out[u].append((t, tgt))


RING_TEXT = {
    "fan_in": "the payee is new and received money from many different accounts in 24 h, mostly first-time payers",
    "shared_device": "the device was used by several different accounts in the last 7 days",
    "device_new_accounts": "several brand-new accounts appeared on this device in the last 24 h",
    "pass_through": "money received in the last hour is being sent straight on",
    "mule_chain": "the money has already passed through other accounts within the hour (mule chain)",
    "cycle": "this transfer closes a loop: money can already flow back from the receiver to the sender",
    "confirmed_link": "the payee or device is linked to a payment an analyst confirmed as fraud",
}


def ring_decision(sig: dict, current: str) -> tuple[str, list[dict]]:
    """Raise (never lower) a decision from ring signals. Candidates -> REVIEW; confirmed links -> BLOCK."""
    order = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
    rules = []
    for r in sig.get("ring_rules", []):
        effect = "BLOCK" if r == "confirmed_link" else "REVIEW"
        rules.append({"rule": f"ring_{r}", "effect": effect, "reason": "Fraud-ring signal: " + RING_TEXT[r],
                      "plain": RING_TEXT[r]})
    out = current
    for x in rules:
        if order[x["effect"]] > order[out]:
            out = x["effect"]
    return out, rules


# =====================================================================================================  offline
def build_temporal_graph(events: pd.DataFrame, confirmed: set[str] | None = None) -> nx.MultiDiGraph:
    """Timestamped multigraph: customer -> payee 'paid' edges and customer -> device 'used' edges, one per payment,
    with time, amount, risk and decision. Node attribute `confirmed` marks analyst-confirmed fraud entities."""
    confirmed = confirmed or set()
    G = nx.MultiDiGraph()
    for r in events.itertuples(index=False):
        un, pn = f"U:{r.user_id}", node_of_payee(r.merchant_id)
        t = _ts(r.ts)
        flagged = getattr(r, "decision", "ALLOW") in ("REVIEW", "BLOCK")
        conf = bool(getattr(r, "confirmed", 0))
        for n, kind in ((un, "customer"), (pn, "customer" if pn.startswith("U:") else "payee")):
            if n not in G:
                G.add_node(n, kind=kind, n_txn=0, n_flagged=0, n_confirmed=0, max_risk=0.0, first=t, last=t)
            a = G.nodes[n]
            a["n_txn"] += 1; a["n_flagged"] += flagged; a["n_confirmed"] += conf
            a["max_risk"] = max(a["max_risk"], float(getattr(r, "probability", 0) or 0)); a["last"] = max(a["last"], t)
        G.add_edge(un, pn, rel="paid", t=t, amount=float(r.amount), flagged=flagged, confirmed=conf, txn_id=r.txn_id)
        if str(r.device_id) not in MISSING:
            dn = f"D:{r.device_id}"
            if dn not in G:
                G.add_node(dn, kind="device", n_txn=0, n_flagged=0, n_confirmed=0, max_risk=0.0, first=t, last=t)
            G.nodes[dn]["n_txn"] += 1; G.nodes[dn]["n_flagged"] += flagged; G.nodes[dn]["n_confirmed"] += conf
            G.add_edge(un, dn, rel="used", t=t, amount=0.0, flagged=flagged, confirmed=conf, txn_id=r.txn_id)
    for n in confirmed:
        if n in G:
            G.nodes[n]["n_confirmed"] = max(1, G.nodes[n]["n_confirmed"])
    return G


def _simple(G: nx.MultiDiGraph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for a, b, d in G.edges(data=True):
        if H.has_edge(a, b):
            e = H[a][b]; e["count"] += 1; e["amount"] += d["amount"]; e["flagged"] += d["flagged"]; e["confirmed"] += d["confirmed"]
        else:
            H.add_edge(a, b, rel=d["rel"], count=1, amount=d["amount"], flagged=int(d["flagged"]), confirmed=int(d["confirmed"]))
    return H


def money_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    D = nx.DiGraph()
    for a, b, d in G.edges(data=True):
        if d["rel"] != "paid":
            continue
        if D.has_edge(a, b):
            D[a][b]["amount"] += d["amount"]; D[a][b]["count"] += 1
        else:
            D.add_edge(a, b, amount=d["amount"], count=1, t=d["t"])
    return D


def temporal_cycles(G: nx.MultiDiGraph, window: float = CYCLE_WINDOW, max_len: int = CYCLE_MAX_LEN, limit: int = 50):
    """Directed customer-to-customer cycles whose transfers happen in time order within `window`."""
    P = nx.DiGraph()
    times = defaultdict(list)
    for a, b, d in G.edges(data=True):
        if d["rel"] == "paid" and b.startswith("U:"):
            P.add_edge(a, b); times[(a, b)].append(d["t"])
    out = []
    for cyc in nx.simple_cycles(P, length_bound=max_len):
        if len(cyc) < 2:
            continue
        # rotate so the earliest transfer starts the loop, then require increasing times within the window
        hops = list(zip(cyc, cyc[1:] + cyc[:1]))
        best = None
        for start in range(len(hops)):
            seq = hops[start:] + hops[:start]
            t0 = min(times[seq[0]]); t, ok = t0, True
            for h in seq[1:]:
                nxt = [x for x in times[h] if x >= t]
                if not nxt:
                    ok = False; break
                t = min(nxt)
            if ok and t - t0 <= window:
                best = (seq, t0, t); break
        if best:
            seq, t0, t1 = best
            out.append({"accounts": [a for a, _ in seq], "hours": round((t1 - t0) / HOUR, 2)})
        if len(out) >= limit:
            break
    return out


def analyse(events: pd.DataFrame, confirmed_txns: set[str] | None = None, top: int = 12, seed: int = 7) -> dict:
    """Ring report for a window of scored events (columns txn_id, ts, user_id, device_id, merchant_id, amount,
    probability, decision; optional `confirmed` 0/1 from analyst labels)."""
    if events.empty:
        return {"rings": [], "stats": {"nodes": 0, "edges": 0}, "top_pagerank": [], "top_betweenness": [], "cycles": [], "chains": []}
    ev = events.copy()
    if confirmed_txns is not None and "confirmed" not in ev:
        ev["confirmed"] = ev.txn_id.isin(confirmed_txns).astype(int)
    G = build_temporal_graph(ev)
    H = _simple(G)
    D = money_graph(G)
    pr = nx.pagerank(D, weight="amount") if D.number_of_edges() else {}
    k = min(200, H.number_of_nodes())
    bc = nx.betweenness_centrality(H, k=k, seed=seed) if k > 2 else {}
    comps = list(nx.connected_components(H))
    cycles = temporal_cycles(G)
    chains = money_chains(ev, window_min=60, min_hops=2)
    in_cycle = {a for c in cycles for a in c["accounts"]}
    in_chain = {f"U:{h['from']}" for c in chains for h in c} | {f"U:{h['to']}" for c in chains for h in c}

    def node_flags(n):
        a = H.nodes[n]; f = []
        nb_users = [x for x in H.neighbors(n) if x.startswith("U:") and x != n]
        if a["kind"] == "device" and len(nb_users) >= SHARED_DEVICE_ACCOUNTS:
            f.append(f"device shared by {len(nb_users)} accounts")
        if a["kind"] == "payee" or (a["kind"] == "customer" and any(H[n][x]["rel"] == "paid" for x in nb_users)):
            payers = [x for x in nb_users if H[n][x]["rel"] == "paid"]
            # a collector: many payers in a short burst (a popular shop also has many payers, but spread over time)
            if len(payers) >= FAN_IN_SENDERS and a["last"] - a["first"] <= 2 * DAY:
                f.append(f"collected from {len(payers)} accounts within {max(1, round((a['last'] - a['first']) / HOUR))} h")
        if n in in_cycle:
            f.append("in a circular transfer")
        if n in in_chain:
            f.append("in a rapid mule chain")
        return f

    structural = {n: node_flags(n) for n in H.nodes}
    structural = {n: f for n, f in structural.items() if f}
    # suspicious subgraph: structural or confirmed entities, plus entities where flagged payments concentrate,
    # joined by the flagged/confirmed payments between them. Communities are found in THIS subgraph, so a normal
    # shop that a victim also paid does not pull its whole customer base into a ring.
    hot = set(structural) | {n for n, a in H.nodes(data=True)
                             if a["n_confirmed"] > 0 or (a["n_flagged"] >= 2 and a["n_flagged"] / max(a["n_txn"], 1) >= 0.3)}
    R = nx.Graph()
    for a, b, d in H.edges(data=True):
        if (a in hot and b in hot) or ((a in hot or b in hot) and (d["flagged"] or d["confirmed"])) \
                or ((a in structural or b in structural) and d["rel"] == "used"):
            R.add_edge(a, b, **d)
    for n in hot:
        R.add_node(n)
    # a ring = a connected part of the suspicious subgraph; Louvain only splits parts too large to be one ring
    comm = []
    for comp in nx.connected_components(R):
        if len(comp) <= MAX_RING:
            comm.append(comp)
        else:
            comm.extend(nx.community.louvain_communities(R.subgraph(comp), seed=seed, weight="count"))
    rings = []
    for c in comm:
        if len(c) < 3:
            continue
        sub = H.subgraph(c)
        flagged_edges = sum(d["flagged"] for _, _, d in sub.edges(data=True))
        confirmed_nodes = [n for n in c if H.nodes[n]["n_confirmed"] > 0]
        struct_c = {n: structural[n] for n in c if n in structural}
        if not confirmed_nodes and flagged_edges < 3 and not struct_c:
            continue
        kinds = defaultdict(int)
        for n in c:
            kinds[H.nodes[n]["kind"]] += 1
        if kinds["customer"] < 2:                         # one customer's own burst is not coordinated activity
            continue
        score = (3.0 * len(confirmed_nodes) + 1.0 * flagged_edges + 2.0 * len(struct_c)) / np.sqrt(len(c))
        hubs = sorted(c, key=lambda n: -(pr.get(n, 0) + bc.get(n, 0)))[:3]
        rings.append({
            "id": "", "score": round(float(score), 2), "size": len(c),
            "customers": kinds["customer"], "devices": kinds["device"], "payees": kinds["payee"],
            "flagged_payments": int(flagged_edges), "confirmed_entities": sorted(confirmed_nodes),
            "status": "confirmed fraud links" if confirmed_nodes else "suspicious (unconfirmed)",
            "evidence": [f"{n}: {', '.join(f)}" for n, f in list(struct_c.items())[:8]]
                        + ([f"{flagged_edges} payments in this group were flagged by the model"] if flagged_edges else []),
            "hubs": hubs, "nodes": sorted(c),
            "amount": round(float(sum(d["amount"] for _, _, d in sub.edges(data=True))), 2)})
    rings.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rings):
        r["id"] = f"R{i + 1}"
    risky_comps = [c for c in comps if any(H.nodes[n]["n_confirmed"] for n in c)]
    return {
        "stats": {"nodes": H.number_of_nodes(), "edges": H.number_of_edges(), "payments": int(len(ev)),
                  "components": len(comps), "components_with_confirmed_fraud": len(risky_comps),
                  "communities": len(comm), "suspicious_entities": len(hot), "rings": len(rings), "cycles": len(cycles), "chains": len(chains)},
        "rings": rings[:top],
        "top_pagerank": [{"node": n, "pagerank": round(v, 5), "kind": H.nodes[n]["kind"]}
                         for n, v in sorted(pr.items(), key=lambda x: -x[1])[:10]],
        "top_betweenness": [{"node": n, "betweenness": round(v, 5), "kind": H.nodes[n]["kind"]}
                            for n, v in sorted(bc.items(), key=lambda x: -x[1])[:10] if v > 0],
        "cycles": cycles[:20], "chains": chains[:20],
    }


def node_rings(report: dict, node: str) -> list[dict]:
    return [r for r in report.get("rings", []) if node in r["nodes"]]
