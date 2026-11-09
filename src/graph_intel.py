"""Graph fraud intelligence: customers, devices and payees as one network, scored with real model outputs.

Nodes   U:<customer>   D:<device>   P:<payee/merchant>   (a P2P payee "P2P_U00012" is the customer U:U00012)
Edges   customer-device (used), customer->payee (paid; amount, count, max risk, flagged count)

Used by the dashboard's Fraud Network Explorer and by the investigation copilot. Every statement this module
produces is computed from transactions and model decisions; nothing is inferred beyond the counts it reports.
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from src.features import MISSING


def node_of_payee(merchant_id: str) -> str:
    m = str(merchant_id)
    return f"U:{m[4:]}" if m.startswith("P2P_") and len(m) > 4 else f"P:{m}"


def build_graph(events: pd.DataFrame) -> nx.Graph:
    """events: columns user_id, device_id, merchant_id, amount, probability (risk), decision, ts."""
    G = nx.Graph()
    flagged = events["decision"].isin(["REVIEW", "BLOCK"]) if "decision" in events else pd.Series(False, index=events.index)
    risk = events["probability"] if "probability" in events else pd.Series(0.0, index=events.index)
    for (u, d, m, a, r, f) in zip(events.user_id, events.device_id, events.merchant_id, events.amount, risk, flagged):
        un, pn = f"U:{u}", node_of_payee(m)
        for n, kind in ((un, "customer"), (pn, "customer" if pn.startswith("U:") else "payee")):
            if n not in G:
                G.add_node(n, kind=kind, n_txn=0, n_flagged=0, max_risk=0.0, amount=0.0)
        G.nodes[un]["n_txn"] += 1; G.nodes[un]["n_flagged"] += int(f); G.nodes[un]["max_risk"] = max(G.nodes[un]["max_risk"], r)
        G.nodes[pn]["amount"] += a; G.nodes[pn]["n_flagged"] += int(f); G.nodes[pn]["max_risk"] = max(G.nodes[pn]["max_risk"], r)
        e = G.get_edge_data(un, pn)
        if e is None:
            G.add_edge(un, pn, rel="paid", amount=0.0, count=0, flagged=0, max_risk=0.0)
            e = G.get_edge_data(un, pn)
        e["amount"] += a; e["count"] += 1; e["flagged"] += int(f); e["max_risk"] = max(e["max_risk"], r)
        if str(d) not in MISSING:
            dn = f"D:{d}"
            if dn not in G:
                G.add_node(dn, kind="device", n_txn=0, n_flagged=0, max_risk=0.0, amount=0.0)
            G.nodes[dn]["n_txn"] += 1; G.nodes[dn]["n_flagged"] += int(f); G.nodes[dn]["max_risk"] = max(G.nodes[dn]["max_risk"], r)
            if not G.has_edge(un, dn):
                G.add_edge(un, dn, rel="used", count=0, flagged=0, max_risk=0.0, amount=0.0)
            ed = G.get_edge_data(un, dn); ed["count"] += 1; ed["flagged"] += int(f); ed["max_risk"] = max(ed["max_risk"], r)
    return G


def entity_findings(G: nx.Graph, node: str) -> list[str]:
    """Evidence statements about one node, each backed by counts in the graph."""
    if node not in G:
        return []
    a, out = G.nodes[node], []
    nbrs = list(G.neighbors(node))
    if a["kind"] == "device":
        users = [n for n in nbrs if n.startswith("U:")]
        if len(users) >= 3:
            out.append(f"Device used by {len(users)} different accounts")
    else:
        payers = [n for n in nbrs if n.startswith("U:") and G.edges[node, n].get("rel") == "paid"]
        if len(payers) >= 5:
            fl = sum(1 for n in payers if G.edges[node, n]["flagged"] > 0)
            out.append(f"Received payments from {len(payers)} different accounts ({fl} of them flagged)")
        devs = [n for n in nbrs if n.startswith("D:")]
        if len(devs) >= 3:
            out.append(f"Account used {len(devs)} different devices")
    if a["n_flagged"]:
        out.append(f"{a['n_flagged']} flagged transaction(s) touch this entity; highest risk {a['max_risk']:.0%}")
    return out


def ego(G: nx.Graph, nodes: list[str], radius: int = 2, max_nodes: int = 60) -> nx.Graph:
    """Neighbourhood around the given nodes, keeping the riskiest neighbours when it is too large."""
    keep = set(n for n in nodes if n in G)
    frontier = set(keep)
    for _ in range(radius):
        nxt = set()
        for n in frontier:
            nb = sorted(G.neighbors(n), key=lambda x: (-G.nodes[x]["n_flagged"], -G.nodes[x]["max_risk"]))
            nxt.update(nb[:15])
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if len(keep) >= max_nodes:
            break
    if len(keep) > max_nodes:
        ranked = sorted(keep - set(nodes), key=lambda x: (-G.nodes[x]["n_flagged"], -G.nodes[x]["max_risk"]))
        keep = set(n for n in nodes if n in G) | set(ranked[: max_nodes - len(nodes)])
    return G.subgraph(keep).copy()


def suspicious_clusters(G: nx.Graph, min_flagged: int = 3, top: int = 10, seed: int = 7) -> list[dict]:
    """Communities (Louvain) in the subgraph of flagged activity, ranked by flagged volume."""
    hot = [n for n, a in G.nodes(data=True) if a["n_flagged"] > 0]
    H = G.subgraph(hot)
    if H.number_of_edges() == 0:
        return []
    comms = nx.community.louvain_communities(H, seed=seed, weight="flagged")
    out = []
    for c in comms:
        flagged = sum(G.nodes[n]["n_flagged"] for n in c if n.startswith("U:"))
        if flagged < min_flagged or len(c) < 3:
            continue
        kinds = defaultdict(int)
        for n in c:
            kinds[G.nodes[n]["kind"]] += 1
        hubs = sorted(c, key=lambda n: -H.degree(n))[:3]
        out.append({"size": len(c), "flagged_txns": int(flagged), "customers": kinds["customer"],
                    "devices": kinds["device"], "payees": kinds["payee"], "hubs": hubs,
                    "max_risk": float(max(G.nodes[n]["max_risk"] for n in c)), "nodes": sorted(c)})
    return sorted(out, key=lambda x: -x["flagged_txns"])[:top]


def high_risk_entities(G: nx.Graph, kind: str, top: int = 15) -> pd.DataFrame:
    rows = []
    for n, a in G.nodes(data=True):
        if a["kind"] != kind:
            continue
        deg_u = sum(1 for x in G.neighbors(n) if x.startswith("U:"))
        rows.append({"entity": n, "linked_accounts": deg_u, "flagged_txns": a["n_flagged"], "max_risk": a["max_risk"],
                     "amount": a["amount"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df.flagged_txns > 0].sort_values(["flagged_txns", "linked_accounts"], ascending=False).head(top)


def money_chains(events: pd.DataFrame, window_min: float = 60.0, min_hops: int = 2) -> list[list[dict]]:
    """Money received by a customer and sent on (P2P) within `window_min` minutes: victim -> A -> B -> ..."""
    p2p = events[events.merchant_id.astype(str).str.startswith("P2P_")].copy()
    if p2p.empty:
        return []
    p2p["t"] = pd.to_datetime(p2p["ts"], format="mixed")
    p2p["recv"] = p2p.merchant_id.str[4:]
    by_sender = {u: g.sort_values("t") for u, g in p2p.groupby("user_id")}
    chains = []
    for _, first in p2p.sort_values("t").iterrows():
        chain = [first]
        cur = first
        while True:
            nxt = by_sender.get(cur["recv"])
            if nxt is None:
                break
            cand = nxt[(nxt.t > cur.t) & (nxt.t <= cur.t + pd.Timedelta(minutes=window_min))
                       & (nxt.amount <= cur.amount * 1.05) & (nxt.amount >= cur.amount * 0.5)]
            if cand.empty or len(chain) > 6:
                break
            cur = cand.iloc[0]
            chain.append(cur)
        if len(chain) >= min_hops:
            chains.append([{"txn_id": r["txn_id"], "from": r["user_id"], "to": r["recv"], "amount": float(r["amount"]),
                            "ts": str(r["ts"]), "decision": r.get("decision", "")} for r in chain])
    seen, uniq = set(), []
    for c in sorted(chains, key=len, reverse=True):
        key = c[-1]["txn_id"]
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq


def layout(H: nx.Graph, seed: int = 7) -> dict:
    return nx.spring_layout(H, seed=seed, k=1.2 / np.sqrt(max(H.number_of_nodes(), 1)), iterations=80)
