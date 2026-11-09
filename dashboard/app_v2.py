"""payTrace 2.0 — fraud intelligence platform (Streamlit).

    streamlit run dashboard/app_v2.py
Every number on every page is read from the event store (paytrace.db) or from reports produced by the code.
Nothing is hard-coded. Live precision/recall use replay labels from the dataset and are labelled as such.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import auth  # noqa: E402
from src.config import REPORTS_DIR  # noqa: E402
from src.store import DB, audit, connect  # noqa: E402

# ---------------- look ----------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"
CRIT, WARN, GOOD = "#d03b3b", "#fab219", "#0ca30c"
INK, INK2, GRID = "#f0efec", "#a9a89f", "rgba(195,194,183,0.13)"
DEC_COLOR = {"BLOCK": CRIT, "REVIEW": WARN, "ALLOW": "#5b5a55"}

st.set_page_config(page_title="payTrace 2.0", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {font-family:'Inter',sans-serif;}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1500px}
h1,h2,h3{letter-spacing:-0.01em}
.fs-head{display:flex;align-items:baseline;gap:14px;margin:0 0 4px 0}
.fs-head h1{font-size:1.7rem;font-weight:700;margin:0}
.fs-sub{color:#a9a89f;font-size:.88rem;margin-bottom:18px}
.kpi{background:linear-gradient(180deg,#232322,#1f1f1e);border:1px solid #2f2f2d;border-radius:12px;padding:14px 16px;
     animation:fade .4s ease-out both}
.kpi .l{color:#a9a89f;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em}
.kpi .v{color:#f0efec;font-size:1.55rem;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.kpi .d{color:#a9a89f;font-size:.78rem;margin-top:2px}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:.74rem;font-weight:600;letter-spacing:.02em}
.BLOCK{background:#d03b3b2b;color:#ff8d8d}.REVIEW{background:#fab2192b;color:#fab219}.ALLOW{background:#5b5a5533;color:#cfcec8}
.sec{font-size:.78rem;color:#a9a89f;text-transform:uppercase;letter-spacing:.08em;margin:18px 0 6px}
.note{color:#a9a89f;font-size:.8rem}
.ev{background:#202020;border:1px solid #2f2f2d;border-radius:10px;padding:12px 14px;margin-bottom:8px}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
code{font-family:'JetBrains Mono',monospace}
</style>""", unsafe_allow_html=True)


def kpi(col, label, value, detail=""):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div>"
                 f"<div class='d'>{detail}</div></div>", unsafe_allow_html=True)


def header(title, sub):
    st.markdown(f"<div class='fs-head'><h1>{title}</h1></div><div class='fs-sub'>{sub}</div>", unsafe_allow_html=True)


def fig_base(fig, h=300, title=None, legend=True):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=48 if title else 10, b=8), hovermode="closest",
                      title=dict(text=title, font=dict(size=13, color=INK2)) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK2, size=11.5),
                      showlegend=legend, legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False); fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


@st.cache_data(ttl=30)
def report(path):
    p = REPORTS_DIR / path
    return json.loads(p.read_text()) if p.exists() else None


def q(sql, params=()):
    if not DB.exists():
        return pd.DataFrame()
    con = connect(readonly=True)
    try:
        return pd.read_sql(sql, con, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()


def status():
    df = q("SELECT k, v FROM status")
    return dict(zip(df.k, df.v)) if len(df) else {}


# ---------------- auth ----------------
def sidebar_auth():
    with st.sidebar:
        st.markdown("### 🛡️ payTrace 2.0")
        st.caption("Fraud intelligence platform · DataNova 2026 · DN-FN-01")
        if auth.demo_mode():
            st.warning("Demo mode: no users configured, role chosen freely.", icon="⚠️")
            role = st.selectbox("Role", ["viewer", "analyst", "admin"], index=1)
            st.session_state.user, st.session_state.role = f"demo-{role}", role
        elif "role" not in st.session_state:
            with st.form("login"):
                u = st.text_input("User"); p = st.text_input("Password", type="password")
                if st.form_submit_button("Sign in"):
                    r = auth.login(u, p)
                    if r:
                        st.session_state.user, st.session_state.role = u, r
                        con = connect(); audit(con, u, r, "login"); con.close(); st.rerun()
                    else:
                        st.error("Invalid credentials")
            st.stop()
        else:
            st.caption(f"Signed in as **{st.session_state.user}** ({st.session_state.role})")
        stt = status()
        if stt:
            st.caption(f"Stream: **{stt.get('state', '-')}** · model {stt.get('model_version', '-')}")
        st.divider()
        st.caption("Data: replay of the unseen test period of the synthetic v2 dataset. Not live bank traffic.")


# ================= pages =================
def page_overview():
    header("Executive overview", "Recommendations from the scoring engine over the streamed period. Payment authorisation stays with the payment system.")
    ev = q("SELECT ts, amount, decision, is_fraud, risk, latency_ms FROM events")
    if ev.empty:
        st.info("No scored transactions yet. An admin can start the stream on the Live monitor page."); return
    n_all = len(ev)
    ev = ev[ev.is_fraud >= 0]          # demo scenarios have no ground truth; keep them out of label-based metrics
    y = ev.is_fraud == 1; fl = ev.decision != "ALLOW"
    tp, fp = int((fl & y).sum()), int((fl & ~y).sum())
    caught = ev.amount[fl & y].sum(); missed = ev.amount[~fl & y].sum(); review_val = ev.amount[ev.decision == "REVIEW"].sum()
    c = st.columns(6)
    kpi(c[0], "Transactions processed", f"{n_all:,}", f"{(ev.decision == 'BLOCK').sum():,} block · {(ev.decision == 'REVIEW').sum():,} review")
    kpi(c[1], "Fraud detected", f"{tp:,}", f"of {int(y.sum()):,} fraud (replay labels) · recall {tp / max(y.sum(), 1):.0%}")
    kpi(c[2], "Fraud value intercepted", f"₹{caught / 1e5:,.1f}L", f"{caught / max(caught + missed, 1):.0%} of fraud value")
    kpi(c[3], "Financial exposure", f"₹{missed / 1e5:,.1f}L", f"missed fraud value · ₹{review_val / 1e5:,.1f}L in review")
    kpi(c[4], "False-positive rate", f"{fp / max((~y).sum(), 1):.2%}", f"{fp:,} genuine payments flagged")
    m = q("SELECT * FROM metrics ORDER BY wall_time DESC LIMIT 1")
    health = "Healthy" if (not m.empty and m.failures[0] == 0 and m.p99_ms[0] < 50) else ("Degraded" if not m.empty else "-")
    kpi(c[5], "Model health", health, f"p99 {m.p99_ms[0]:.2f} ms · {int(m.failures[0])} failures" if not m.empty else "")
    ev["hour"] = pd.to_datetime(ev.ts, format="mixed").dt.floor("6h")
    g = ev.groupby(["hour", "decision"]).size().unstack(fill_value=0)
    l, r = st.columns([3, 2])
    f = go.Figure()
    for d in ["REVIEW", "BLOCK"]:
        if d in g:
            f.add_bar(x=g.index, y=g[d], name=d.title(), marker_color=DEC_COLOR[d])
    f.update_layout(barmode="stack", bargap=0.12)
    l.plotly_chart(fig_base(f, 300, "Alerts per 6 hours"), width="stretch")
    cum = ev.assign(c=np.where(fl & y, ev.amount, 0), m_=np.where(~fl & y, ev.amount, 0)).groupby("hour")[["c", "m_"]].sum().cumsum() / 1e5
    f2 = go.Figure([go.Scatter(x=cum.index, y=cum.c, name="Intercepted", line=dict(color=BLUE, width=2)),
                    go.Scatter(x=cum.index, y=cum.m_, name="Missed", line=dict(color=ORANGE, width=2))])
    f2.update_yaxes(title="₹ lakh (cumulative)")
    r.plotly_chart(fig_base(f2, 300, "Fraud value, cumulative"), width="stretch")
    rep = report("v2/metrics.json")
    if rep:
        t = rep["test"]
        st.markdown("<div class='sec'>Held-out evaluation of the deployed model</div>", unsafe_allow_html=True)
        c = st.columns(5)
        kpi(c[0], "PR-AUC (test)", f"{t['ranking']['pr_auc']:.3f}", "synthetic v2, days 45–60")
        kpi(c[1], "Recall (review tier)", f"{t['review_tier']['recall']:.1%}")
        kpi(c[2], "Precision (review tier)", f"{t['review_tier']['precision']:.1%}")
        kpi(c[3], "Block precision", f"{t['block_tier']['precision']:.1%}")
        kpi(c[4], "Loss vs no model", f"−{1 - t['review_tier']['loss_inr'] / t['no_model_loss_inr']:.0%}",
            f"₹{t['review_tier']['loss_inr'] / 1e5:,.1f}L vs ₹{t['no_model_loss_inr'] / 1e5:,.1f}L")


def page_live():
    header("Live transaction monitor", "Every payment scored individually by the 2.0 engine. Refreshes every 2 seconds.")
    if auth.can(st.session_state.role, "stream_control"):
        with st.expander("Stream control (admin)", expanded=status().get("state") in (None, "", "done")):
            c = st.columns([2, 1, 1, 1])
            rate = c[0].slider("Replay speed (txns/sec)", 50, 3000, 300, 50)
            dup = c[1].number_input("Inject duplicates %", 0.0, 10.0, 0.0, 0.5)
            jit = c[2].number_input("Out-of-order jitter (s)", 0.0, 120.0, 0.0, 5.0)
            if c[3].button("▶ Start", type="primary", width="stretch"):
                p = st.session_state.get("proc")
                if p and p.poll() is None:
                    p.terminate()
                st.session_state.proc = subprocess.Popen([sys.executable, "-m", "src.stream_v2", "--rate", str(rate),
                                                          "--dup-rate", str(dup / 100), "--jitter", str(jit)], cwd=ROOT,
                                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                con = connect(); audit(con, st.session_state.user, st.session_state.role, "stream_start", "", {"rate": rate}); con.close()
                st.toast("Loading models and warming feature state (~10 s)")
            if c[3].button("■ Stop", width="stretch") and st.session_state.get("proc"):
                st.session_state.proc.terminate()
    live_fragment()


@st.fragment(run_every="2s")
def live_fragment():
    stt = status()
    ev = q("SELECT seq, txn_id, ts, user_id, amount, merchant_id, category, channel, risk, decision, tripwires, reasons, "
           "latency_ms, e2e_ms, is_fraud, fraud_type FROM events ORDER BY seq DESC LIMIT 400")
    tot = q("SELECT COUNT(*) n, SUM(decision!='ALLOW') a, SUM(decision='BLOCK') b FROM events")
    if ev.empty:
        st.info(f"Stream state: {stt.get('state', 'not started')}"); return
    total = int(stt.get("total", 0) or 0); n = int(tot.n[0])
    st.progress(min(n / max(total, 1), 1.0), text=f"{stt.get('state', '')} · {n:,} / {total:,} payments scored")
    m = q("SELECT * FROM metrics ORDER BY wall_time DESC LIMIT 1")
    c = st.columns(6)
    kpi(c[0], "Scored", f"{n:,}"); kpi(c[1], "Alerts", f"{int(tot.a[0]):,}", f"{int(tot.b[0]):,} block")
    if not m.empty:
        kpi(c[2], "Throughput", f"{m.throughput[0]:,.0f}/s"); kpi(c[3], "Scoring p50 / p99", f"{m.p50_ms[0]:.2f} / {m.p99_ms[0]:.2f} ms")
        kpi(c[4], "Duplicates dropped", f"{int(m.duplicates[0]):,}", f"{int(m.late[0]):,} late"); kpi(c[5], "Failures", f"{int(m.failures[0])}", "dead-letter")
    l, r = st.columns([3, 2])
    view = ev.head(60)
    tbl = pd.DataFrame({"Time": pd.to_datetime(view.ts, format="mixed").dt.strftime("%d %b %H:%M:%S"),
                        "Decision": view.decision, "Risk": view.risk * 100, "Amount ₹": view.amount, "Customer": view.user_id,
                        "Payee": view.merchant_id, "Category": view.category, "Latency ms": view.latency_ms.round(3)})
    l.markdown("<div class='sec'>Incoming payments</div>", unsafe_allow_html=True)
    l.dataframe(tbl, hide_index=True, height=420, width="stretch",
                column_config={"Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                               "Amount ₹": st.column_config.NumberColumn(format="₹%.0f")})
    al = ev[ev.decision != "ALLOW"].head(12)
    r.markdown("<div class='sec'>Alert feed</div>", unsafe_allow_html=True)
    for _, a in al.iterrows():
        rs = " · ".join(x["reason"] for x in json.loads(a.reasons)[:3])
        r.markdown(f"<div class='ev'><span class='pill {a.decision}'>{a.decision}</span> &nbsp;<b>₹{a.amount:,.0f}</b> "
                   f"{a.user_id} → {a.merchant_id} · risk {a.risk:.0%}<br><span class='note'>{rs}</span></div>", unsafe_allow_html=True)


def page_network():
    from src.graph_intel import build_graph, ego, entity_findings, high_risk_entities, layout, money_chains, node_of_payee, suspicious_clusters
    header("Fraud network explorer", "Customers, devices and payees from scored payments; colour shows model risk.")
    ev = q("SELECT txn_id, ts, user_id, device_id, merchant_id, amount, risk AS probability, decision FROM events")
    if ev.empty:
        st.info("No scored transactions yet."); return
    days = st.slider("Window (days, most recent)", 1, 15, 3)
    end = pd.to_datetime(ev.ts, format="mixed").max()
    w = ev[pd.to_datetime(ev.ts, format="mixed") >= end - pd.Timedelta(days=days)]
    G = build_graph(w)
    tab1, tab2, tab3 = st.tabs(["Suspicious clusters", "Entity explorer", "Money chains"])
    with tab1:
        cl = suspicious_clusters(G)
        st.caption(f"{G.number_of_nodes():,} entities, {G.number_of_edges():,} links in the window. Clusters = Louvain communities of flagged activity.")
        if cl:
            df = pd.DataFrame([{k: v for k, v in c.items() if k != "nodes"} for c in cl])
            df["hubs"] = df.hubs.map(lambda h: ", ".join(h))
            st.dataframe(df, hide_index=True, width="stretch")
            pick = st.selectbox("Show cluster", range(len(cl)), format_func=lambda i: f"Cluster {i + 1}: {cl[i]['flagged_txns']} flagged, hubs {', '.join(cl[i]['hubs'][:2])}")
            draw_network(G.subgraph(cl[pick]["nodes"][:80]).copy(), [], layout)
        c1, c2 = st.columns(2)
        c1.markdown("<div class='sec'>High-risk devices</div>", unsafe_allow_html=True); c1.dataframe(high_risk_entities(G, "device"), hide_index=True, width="stretch")
        c2.markdown("<div class='sec'>High-risk payees</div>", unsafe_allow_html=True); c2.dataframe(high_risk_entities(G, "payee"), hide_index=True, width="stretch")
    with tab2:
        alerts = w[w.decision != "ALLOW"].sort_values("probability", ascending=False)
        if alerts.empty:
            st.info("No alerts in this window."); return
        tid = st.selectbox("Flagged transaction", alerts.txn_id.head(300),
                           format_func=lambda t: f"{t} · ₹{alerts.set_index('txn_id').amount[t]:,.0f} · risk {alerts.set_index('txn_id').probability[t]:.0%}")
        row = alerts.set_index("txn_id").loc[tid]
        seeds = [f"U:{row.user_id}", f"D:{row.device_id}", node_of_payee(row.merchant_id)]
        H = ego(G, seeds, radius=2, max_nodes=60)
        l, r = st.columns([3, 2])
        with l:
            draw_network(H, seeds, layout)
        with r:
            st.markdown("<div class='sec'>Why these entities are suspicious</div>", unsafe_allow_html=True)
            for n in seeds + sorted([x for x in H.nodes if x not in seeds and H.nodes[x]["n_flagged"]], key=lambda x: -H.nodes[x]["n_flagged"])[:6]:
                f = entity_findings(G, n)
                if f:
                    st.markdown(f"<div class='ev'><b>{n}</b><br><span class='note'>" + "<br>".join(f) + "</span></div>", unsafe_allow_html=True)
    with tab3:
        ch = money_chains(w)
        st.caption("Money received by a customer and forwarded (P2P) within 60 minutes.")
        if not ch:
            st.info("No chains in this window.")
        for c in ch[:15]:
            st.markdown("<div class='ev'>" + " → ".join([c[0]["from"]] + [h["to"] for h in c]) +
                        f"<br><span class='note'>{len(c)} hops · ₹{c[0]['amount']:,.0f} → ₹{c[-1]['amount']:,.0f} · "
                        + ", ".join(f"<span class='pill {h['decision']}'>{h['decision']}</span>" for h in c) + "</span></div>", unsafe_allow_html=True)


def draw_network(H, seeds, layout_fn):
    if H.number_of_nodes() == 0:
        return
    pos = layout_fn(H)
    ex, ey, fx, fy = [], [], [], []
    for a, b, d in H.edges(data=True):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        (fx if d.get("flagged") else ex).extend([x0, x1, None]); (fy if d.get("flagged") else ey).extend([y0, y1, None])
    fig = go.Figure([go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(169,168,159,0.25)", width=1), hoverinfo="skip", name="link"),
                     go.Scatter(x=fx, y=fy, mode="lines", line=dict(color="rgba(208,59,59,0.55)", width=1.6), hoverinfo="skip", name="flagged link")])
    sym = {"customer": "circle", "device": "square", "payee": "diamond"}
    for kind in ("customer", "device", "payee"):
        ns = [n for n in H.nodes if H.nodes[n]["kind"] == kind]
        if not ns:
            continue
        fig.add_scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", name=kind.title(),
                        text=[n.split(":", 1)[1][:12] if (n in seeds or H.nodes[n]["n_flagged"] > 1) else "" for n in ns],
                        textposition="top center", textfont=dict(size=9, color=INK2),
                        marker=dict(symbol=sym[kind], size=[18 if n in seeds else 10 + min(H.nodes[n]["n_flagged"], 8) for n in ns],
                                    color=[H.nodes[n]["max_risk"] for n in ns], colorscale=[[0, "#3a3a38"], [0.3, WARN], [1, CRIT]],
                                    cmin=0, cmax=1, line=dict(color=[INK if n in seeds else "#1a1a19" for n in ns], width=2)),
                        hovertext=[f"{n}<br>{H.nodes[n]['n_flagged']} flagged · max risk {H.nodes[n]['max_risk']:.0%}" for n in ns], hoverinfo="text")
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    st.plotly_chart(fig_base(fig, 520, None), width="stretch")
    st.caption("○ customer · □ device · ◇ payee · colour = highest model risk · red links carry flagged payments · outlined = selected transaction")


def page_investigation():
    from src.investigation import case_report, evidence_pack, summarise
    header("Investigation centre", "Evidence-grounded case analysis. The copilot only summarises evidence; it never decides.")
    queue_ = q("SELECT txn_id, ts, user_id, merchant_id, amount, risk, decision FROM events WHERE decision!='ALLOW' ORDER BY risk DESC LIMIT 500")
    if queue_.empty:
        st.info("No alerts yet."); return
    fb = q("SELECT txn_id, status FROM feedback")
    labelled = set(fb.txn_id) if len(fb) else set()
    pick = st.selectbox("Case queue (highest risk first)", queue_.txn_id,
                        format_func=lambda t: f"{t} · {queue_.set_index('txn_id').decision[t]} · ₹{queue_.set_index('txn_id').amount[t]:,.0f} · risk {queue_.set_index('txn_id').risk[t]:.0%}" + (" · labelled" if t in labelled else ""))
    con = connect()
    pack = evidence_pack(con, pick)
    t, m = pack["transaction"], pack["model"]
    st.markdown(f"<span class='pill {m['decision']}'>{m['decision']}</span> &nbsp;<b>₹{t['amount']:,.0f}</b> · {t['user_id']} → "
                f"{t['merchant_id']} · {t['category']} · {t['channel']} · {t['ts'][:19]}", unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], "Fused risk", f"{m['risk']:.1%}"); kpi(c[1], "Supervised probability", f"{m['supervised_probability']:.1%}")
    kpi(c[2], "Novelty tripwires", ", ".join(m["tripwires"]) or "none")
    ch = pack["customer_history"]
    kpi(c[3], "Customer, last 30 days", f"{ch['payments_last_30d']} payments",
        f"median ₹{ch['median_amount']:,.0f}" if ch["median_amount"] else "no history")
    l, r = st.columns(2)
    if m["reason_codes"]:
        rc = [x for x in m["reason_codes"] if x["impact"] is not None]
        f = go.Figure(go.Bar(x=[x["impact"] for x in rc][::-1], y=[x["reason"] for x in rc][::-1], orientation="h", marker_color=BLUE,
                             hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        l.plotly_chart(fig_base(f, 240, "SHAP reason codes (supervised model, log-odds)", legend=False), width="stretch")
    if m["risk_breakdown"]:
        b = m["risk_breakdown"]
        f = go.Figure(go.Bar(x=[x["contribution"] for x in b][::-1], y=[x["component"] for x in b][::-1], orientation="h",
                             marker_color=[CRIT if x["contribution"] > 0 else AQUA for x in b][::-1], hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
        r.plotly_chart(fig_base(f, 240, "Risk fusion: contribution of each component", legend=False), width="stretch")
    l, r = st.columns(2)
    with l:
        st.markdown("<div class='sec'>Network findings</div>", unsafe_allow_html=True)
        for n, fs in pack["network_findings"].items():
            if fs:
                st.markdown(f"<div class='ev'><b>{n}</b><br><span class='note'>" + "<br>".join(fs) + "</span></div>", unsafe_allow_html=True)
        if pack["money_chains"]:
            st.markdown("<div class='ev'><b>Money chain</b><br><span class='note'>" + " → ".join([pack["money_chains"][0][0]["from"]] + [h["to"] for h in pack["money_chains"][0]]) + "</span></div>", unsafe_allow_html=True)
    with r:
        st.markdown("<div class='sec'>Similar past cases (outcome known)</div>", unsafe_allow_html=True)
        if pack["similar_cases"]:
            st.dataframe(pd.DataFrame(pack["similar_cases"])[["txn_id", "amount", "decision", "outcome", "distance"]],
                         hide_index=True, width="stretch")
    st.markdown("<div class='sec'>Copilot summary</div>", unsafe_allow_html=True)
    if auth.can(st.session_state.role, "copilot"):
        key = f"sum_{pick}"
        if st.button("Generate investigation summary"):
            st.session_state[key] = summarise(pack)
            audit(con, st.session_state.user, st.session_state.role, "copilot_summary", pick, {"source": st.session_state[key][1]})
        if key in st.session_state:
            s, src = st.session_state[key]
            st.markdown(f"<div class='ev'>{'' if src == 'template' else f'<span class=note>Written by {src} from the evidence pack only.</span><br>'}</div>", unsafe_allow_html=True)
            st.markdown(s)
            st.download_button("⬇ Download case report (Markdown)", case_report(pack, s, st.session_state.user),
                               file_name=f"case_{pick}.md", mime="text/markdown")
    else:
        st.caption("Analyst role required.")
    st.markdown("<div class='sec'>Analyst decision</div>", unsafe_allow_html=True)
    if auth.can(st.session_state.role, "label"):
        from src.feedback import FeedbackRejected, submit
        with st.form(f"fb_{pick}"):
            lab = st.radio("Outcome", ["Confirmed fraud", "Genuine payment"], horizontal=True)
            note = st.text_area("Note (required when you disagree with a BLOCK)")
            if st.form_submit_button("Submit label"):
                try:
                    stt = submit(con, pick, st.session_state.user, st.session_state.role, 1 if lab.startswith("Confirmed") else 0, note)
                    st.success(f"Label recorded · status: {stt}")
                except FeedbackRejected as e:
                    st.error(str(e))
        gt = pack["replay_ground_truth"]
        st.caption(f"Replay ground truth (evaluation only): {'fraud · ' + gt['fraud_type'] if gt['is_fraud'] == 1 else 'genuine'}")
    con.close()


def page_models():
    header("Model intelligence", "Every chart comes from reports/experiments/results.json and reports/v2/metrics.json, produced by the code.")
    rep, ex = report("v2/metrics.json"), report("experiments/results.json")
    if not rep or not ex:
        st.warning("Run `python -m src.train_v2` and `python -m src.experiments` first."); return
    tabs = st.tabs(["Comparison", "Curves", "Unseen fraud", "Ablation", "Calibration & thresholds", "Fairness & robustness", "Drift"])
    with tabs[0]:
        ds = st.radio("Dataset", [k for k in ("v2", "v1") if k in ex.get("compare", {})] + (["fresh seed-11 (confirmation)"] if "confirm" in ex else []), horizontal=True)
        res = ex["confirm"]["compare"]["results"] if ds.startswith("fresh") else ex["compare"][ds]["results"]
        df = pd.DataFrame([{"Model": k, "PR-AUC": v["pr_auc"], "ROC-AUC": v["roc_auc"], "Recall@1%FPR": v["recall_at_1pct_fpr"],
                            "Precision@80%R": v["precision_at_80pct_recall"], "Precision": v["precision"], "Recall": v["recall"],
                            "F1": v["f1"], "FPR": v["fpr"], "Loss ₹": v["loss_inr"], "Train s": v.get("train_seconds")} for k, v in res.items()])
        st.dataframe(df, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["PR-AUC", "ROC-AUC", "Recall@1%FPR", "Precision@80%R", "Precision", "Recall", "F1"]} | {"FPR": st.column_config.NumberColumn(format="%.4f"), "Loss ₹": st.column_config.NumberColumn(format="₹%.0f"), "Train s": st.column_config.NumberColumn(format="%.1f")})
        st.caption("All models trained on the same slices; thresholds chosen on the threshold slice; metrics on the untouched test slice. v1 system = original 27 features, retrained on the same slices.")
        from src.scorer_v2 import COMPONENT_NAMES
        fw = {COMPONENT_NAMES.get(k, k): v for k, v in rep.get("fusion_weights", {}).items()}
        if fw:
            f = go.Figure(go.Bar(x=list(fw.values()), y=list(fw), orientation="h", marker_color=[BLUE if v > 0 else AQUA for v in fw.values()]))
            st.plotly_chart(fig_base(f, 260, "Risk fusion weights (standardised logistic coefficients)", legend=False), width="stretch")
    with tabs[1]:
        cmp = ex["compare"]["v2"]
        l, r = st.columns(2)
        f = go.Figure(); f2 = go.Figure()
        for i, (k, cpts) in enumerate(cmp["pr_curves"].items()):
            col = [INK2, BLUE, ORANGE, AQUA, VIOLET, MAGENTA][i % 6]
            f.add_scatter(x=cpts["recall"], y=cpts["precision"], name=k, line=dict(color=col, width=2))
            rc = cmp["roc_curves"][k]; f2.add_scatter(x=rc["fpr"], y=rc["tpr"], name=k, line=dict(color=col, width=2))
        f.update_xaxes(title="Recall"); f.update_yaxes(title="Precision", range=[0, 1.02])
        f2.update_xaxes(title="False-positive rate", type="log", range=[-4, 0]); f2.update_yaxes(title="True-positive rate")
        l.plotly_chart(fig_base(f, 380, "Precision–recall (test, v2 data)"), width="stretch")
        r.plotly_chart(fig_base(f2, 380, "ROC (log FPR axis)"), width="stretch")
        t = rep["test"]["review_tier"]
        cm = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        f3 = go.Figure(go.Heatmap(z=np.log10(cm + 1), x=["Allowed", "Flagged"], y=["Genuine", "Fraud"], text=cm, texttemplate="%{text:,}",
                                  colorscale=[[0, "#232322"], [1, BLUE]], showscale=False))
        st.plotly_chart(fig_base(f3, 260, "Confusion matrix, deployed system (review tier, test)", legend=False), width="stretch")
    with tabs[2]:
        lp = ex.get("lopo", {}); pre = ex.get("lopo_before_merchant_profile", {})
        if lp:
            pats = list(lp)
            f = go.Figure([go.Bar(x=pats, y=[lp[p]["supervised_only"] for p in pats], name="Supervised only", marker_color=INK2),
                           go.Bar(x=pats, y=[lp[p]["hybrid_system"] for p in pats], name="Hybrid (fusion + tripwires)", marker_color=BLUE),
                           go.Bar(x=pats, y=[lp[p]["adaptation"]["after_5d_feedback"]["eval_recall"] for p in pats], name="After 5 days of analyst labels", marker_color=AQUA)])
            f.update_layout(barmode="group", bargap=0.25); f.update_yaxes(tickformat=".0%", range=[0, 1.05])
            st.plotly_chart(fig_base(f, 360, "Recall on a fraud pattern completely removed from training"), width="stretch")
            ad = pd.DataFrame([{"Pattern": p, "Champion": lp[p]["adaptation"]["champion_eval_recall"],
                                **{f"+{k} day labels": lp[p]["adaptation"][f"after_{k}d_feedback"]["eval_recall"] for k in (1, 3, 5)},
                                "Labels used (5 d)": lp[p]["adaptation"]["after_5d_feedback"]["labels_used"]} for p in pats])
            st.dataframe(ad, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ad.columns if c not in ("Pattern", "Labels used (5 d)")})
            st.caption("Feedback: analysts label only transactions the system alerted on in the first k test days; the challenger is evaluated on the following days, which no model saw.")
            if pre:
                st.caption("Before the merchant-behaviour profile and per-entity tripwires, mean hybrid recall was "
                           f"{np.mean([v['hybrid_system'] for v in pre.values()]):.0%}; now {np.mean([v['hybrid_system'] for v in lp.values()]):.0%}.")
    with tabs[3]:
        ab = ex.get("ablation", {})
        if ab:
            names = list(ab); vals = [ab[k]["pr_auc"] for k in names]
            f = go.Figure(go.Bar(x=vals[::-1], y=names[::-1], orientation="h", marker_color=[BLUE if n == "all features" else INK2 for n in names][::-1],
                                 text=[f"{v:.3f}" for v in vals][::-1], textposition="outside"))
            f.update_xaxes(range=[min(vals) - 0.05, 1])
            st.plotly_chart(fig_base(f, 380, "Test PR-AUC when one feature group is removed", legend=False), width="stretch")
        an = ex.get("anomaly", {})
        if an:
            st.dataframe(pd.DataFrame([{"Detector": k, "PR-AUC alone": v["pr_auc"], "Recall at 0.1% alert budget": v.get("recall_at_0.1pct_alert_budget")} for k, v in an.items()]), hide_index=True, width="stretch")
    with tabs[4]:
        cal = ex.get("calibration", {})
        if cal:
            c = cal["calibration"]; l, r = st.columns(2)
            f = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=INK2, dash="dot", width=1), name="perfect")])
            for k, col in (("bins_raw", ORANGE), ("bins_isotonic", BLUE), ("bins_fused", AQUA)):
                f.add_scatter(x=[b["mean_pred"] for b in c[k]], y=[b["frac_fraud"] for b in c[k]], mode="lines+markers", name=k[5:], line=dict(color=col, width=2), marker=dict(size=8))
            f.update_xaxes(title="Predicted probability", type="log"); f.update_yaxes(title="Observed fraud rate", type="log")
            l.plotly_chart(fig_base(f, 360, f"Reliability (Brier raw {c['brier_raw']:.4f} · isotonic {c['brier_isotonic']:.4f} · fused {c['brier_fused']:.4f})"), width="stretch")
            th = pd.DataFrame(cal["thresholds"])
            f2 = go.Figure([go.Scatter(x=th.threshold, y=th.recall, name="Recall", line=dict(color=BLUE, width=2)),
                            go.Scatter(x=th.threshold, y=th.precision, name="Precision", line=dict(color=ORANGE, width=2))])
            f2.update_xaxes(title="Fused-risk threshold", type="log")
            r.plotly_chart(fig_base(f2, 360, "Threshold sensitivity"), width="stretch")
            th["role"] = np.where(th.is_review, "REVIEW", np.where(th.is_block, "BLOCK", ""))
            st.dataframe(th[["threshold", "role", "recall", "precision", "fpr", "loss_inr", "genuine_flagged_per_10k", "alerts_per_day"]],
                         hide_index=True, width="stretch", column_config={"loss_inr": st.column_config.NumberColumn("Loss ₹", format="₹%.0f"),
                                                                          "fpr": st.column_config.NumberColumn(format="%.4f")})
    with tabs[5]:
        cal = ex.get("calibration", {})
        if cal:
            seg = cal["segments"]
            st.dataframe(pd.DataFrame([{"Customers": k, **{x: v.get(x) for x in ("n", "fraud", "pr_auc", "recall", "precision", "fpr")}} for k, v in seg["customers"].items()]), hide_index=True, width="stretch")
            cols = st.columns(3)
            for c, g in zip(cols, ("city", "account_age", "channel")):
                d = pd.DataFrame([{"group": k, **v} for k, v in seg[g].items()])
                f = go.Figure(go.Bar(x=d.group, y=d.fpr, marker_color=BLUE, text=[f"{v:.2%}" for v in d.fpr], textposition="outside"))
                f.update_yaxes(tickformat=".1%")
                c.plotly_chart(fig_base(f, 300, f"False-positive rate by {g.replace('_', ' ')}", legend=False), width="stretch")
        rb = ex.get("robustness", {})
        if rb:
            st.dataframe(pd.DataFrame([{"Condition": k, **v} for k, v in rb.items()]), hide_index=True, width="stretch")
            st.caption("Missing device ids originally collapsed PR-AUC to 0.15 (one 'UNKNOWN' device looked shared by thousands of accounts); fixed by treating missing values as missing.")
    with tabs[6]:
        drift_panel()


def drift_panel():
    from src.drift import psi
    from src.train_v2 import load, slices
    ev = q("SELECT txn_id, ts FROM events ORDER BY seq DESC LIMIT 6000")
    if ev.empty:
        st.info("Drift needs streamed events."); return
    df, X = load("v2")
    s = slices(df)
    cols = ["amount", "hour", "u_cnt_1h", "u_amt_ratio_mean", "u_new_merchant", "m_users_1h", "d_n_users", "u_passthrough", "m_new_payer_share_24h"]
    ref = X.loc[s["train"], cols]
    from src.drift import psi_reference
    import src.drift as dmod
    old = dmod.MONITORED
    dmod.MONITORED = [c for c in cols]
    ref_stats = psi_reference(ref.assign(m_age_hours=999))
    dmod.MONITORED = old
    cur = X.loc[df.txn_id.isin(ev.txn_id), cols]
    vals = psi({k: v for k, v in ref_stats.items() if k in cols}, {c: cur[c].to_numpy() for c in cols})
    f = go.Figure(go.Bar(x=list(vals.values()), y=list(vals), orientation="h", marker_color=[CRIT if v > 0.25 else WARN if v > 0.1 else BLUE for v in vals.values()]))
    for x in (0.1, 0.25):
        f.add_vline(x=x, line=dict(color=INK2, width=1, dash="dot"))
    st.plotly_chart(fig_base(f, 360, f"PSI: last {len(cur):,} streamed payments vs training period", legend=False), width="stretch")
    st.caption("< 0.1 stable · 0.1–0.25 moderate · > 0.25 investigate / retrain. Computed live from the feature store.")


def page_infra():
    header("Performance & infrastructure", "Measured by the stream processor (metrics table) and the offline benchmark.")
    m = q("SELECT * FROM metrics ORDER BY wall_time")
    lat = report("v2/latency.json")
    if lat:
        L = lat["latency_ms"]; c = st.columns(5)
        kpi(c[0], "Benchmark p50", f"{L['p50']:.3f} ms"); kpi(c[1], "p95", f"{L['p95']:.3f} ms"); kpi(c[2], "p99", f"{L['p99']:.3f} ms")
        kpi(c[3], "Throughput (1 core)", f"{lat['throughput_txn_per_sec_single_core']:,.0f}/s"); kpi(c[4], "Machine", lat["machine"].split(",")[0])
    if m.empty:
        st.info("No stream metrics yet."); return
    m["t"] = pd.to_datetime(m.wall_time, unit="s")
    l, r = st.columns(2)
    f = go.Figure([go.Scatter(x=m.t, y=m.p50_ms, name="p50", line=dict(color=BLUE, width=2)),
                   go.Scatter(x=m.t, y=m.p95_ms, name="p95", line=dict(color=AQUA, width=2)),
                   go.Scatter(x=m.t, y=m.p99_ms, name="p99", line=dict(color=ORANGE, width=2))])
    f.update_yaxes(title="ms")
    l.plotly_chart(fig_base(f, 300, "Scoring latency (rolling 5,000 payments)"), width="stretch")
    f2 = go.Figure(go.Scatter(x=m.t, y=m.throughput, line=dict(color=BLUE, width=2), name="txn/s"))
    r.plotly_chart(fig_base(f2, 300, "Throughput (payments per second)", legend=False), width="stretch")
    l, r = st.columns(2)
    f3 = go.Figure(go.Scatter(x=m.t, y=m.cpu_pct, line=dict(color=VIOLET, width=2)))
    l.plotly_chart(fig_base(f3, 260, "Processor CPU %", legend=False), width="stretch")
    f4 = go.Figure(go.Scatter(x=m.t, y=m.rss_mb, line=dict(color=MAGENTA, width=2)))
    r.plotly_chart(fig_base(f4, 260, "Processor memory (MB)", legend=False), width="stretch")
    last = m.iloc[-1]
    c = st.columns(5)
    kpi(c[0], "Processed", f"{int(last.processed):,}"); kpi(c[1], "Duplicates dropped", f"{int(last.duplicates):,}")
    kpi(c[2], "Late events", f"{int(last.late):,}"); kpi(c[3], "Failures (dead-letter)", f"{int(last.failures)}")
    kpi(c[4], "End-to-end p50", f"{last.e2e_p50_ms:,.1f} ms", "includes queue wait")
    dl = q("SELECT * FROM deadletter ORDER BY wall_time DESC LIMIT 50")
    if len(dl):
        st.dataframe(dl, hide_index=True, width="stretch")


def page_learning():
    header("Model feedback & learning", "Validated analyst labels, champion/challenger retraining and model versions.")
    fb = q("SELECT datetime(wall_time,'unixepoch') AS time, txn_id, analyst, label, status, model_decision, model_risk, note FROM feedback ORDER BY wall_time DESC")
    c = st.columns(4)
    kpi(c[0], "Labels", f"{len(fb):,}")
    for col, sname in zip(c[1:], ("accepted", "pending_second_review", "conflict")):
        kpi(col, sname.replace("_", " ").title(), f"{int((fb.status == sname).sum()) if len(fb) else 0:,}")
    if len(fb):
        st.dataframe(fb, hide_index=True, width="stretch", height=240)
    reg = q("SELECT version, datetime(created,'unixepoch') AS created, kind, role, trained_on, metrics, notes FROM model_registry ORDER BY created DESC")
    st.markdown("<div class='sec'>Model registry</div>", unsafe_allow_html=True)
    if len(reg):
        st.dataframe(reg.drop(columns=["metrics"]), hide_index=True, width="stretch")
        ch = reg[reg.role == "challenger"]
        if len(ch):
            mt = json.loads(ch.metrics.iloc[0])
            d = pd.DataFrame([{"model": k, **mt[k]} for k in ("champion", "challenger")])
            st.dataframe(d, hide_index=True, width="stretch"); st.caption(mt.get("note", ""))
    else:
        st.caption("Registry is created on the first retrain.")
    if auth.can(st.session_state.role, "retrain"):
        c = st.columns(2)
        if c[0].button("Train challenger from accepted labels"):
            from src.retrain import run
            with st.spinner("Training challenger…"):
                st.json(run(False, st.session_state.user, st.session_state.role))
        if c[1].button("Train and promote if better"):
            from src.retrain import run
            with st.spinner("Training challenger…"):
                st.json(run(True, st.session_state.user, st.session_state.role))
    al = q("SELECT datetime(wall_time,'unixepoch') AS time, actor, role, action, target, detail FROM audit_log ORDER BY wall_time DESC LIMIT 200")
    st.markdown("<div class='sec'>Audit log</div>", unsafe_allow_html=True)
    st.dataframe(al, hide_index=True, width="stretch", height=260)


# ---------------- manual test lab ----------------
PRESETS = {
    "Custom": {},
    "Everyday payment (usual device, usual amount)": dict(amount_x=1.0, device="usual", category="grocery", channel="UPI", hour=13, repeat=1, gap=60),
    "Large transfer to a new payee from a new phone at 3 am": dict(amount_x=15.0, device="new", category="P2P", channel="UPI", hour=3, repeat=1, gap=60, payee="P2P_U09999"),
    "Card-testing burst (10 tiny payments, 20 s apart)": dict(amount=9.0, device="new", category="gaming", channel="CARD", hour=2, repeat=10, gap=20, vary_payee=True),
    "Rapid spending spree (6 electronics purchases)": dict(amount_x=6.0, device="new", category="electronics", channel="CARD", hour=23, repeat=6, gap=90, vary_payee=True),
}


@st.cache_resource(show_spinner="Loading models and customer history (about 20-30 s, once)…")
def lab_engine():
    from src.scorer_v2 import FraudScorerV2
    from src.stream_v2 import TXN_FIELDS
    from src.train_v2 import dataset_path, slices
    s = FraudScorerV2()
    df = pd.read_csv(dataset_path("v2"), parse_dates=["timestamp"])
    hist = df[~slices(df)["test"]]
    s.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist.is_fraud.to_numpy())
    prof = (hist[hist.is_fraud == 0].groupby("user_id")
            .agg(payments=("txn_id", "size"), median_amount=("amount", "median"),
                 device=("device_id", lambda x: x.mode().iat[0]), city=("city", lambda x: x.mode().iat[0]),
                 payee=("merchant_id", lambda x: x.mode().iat[0]))
            .sort_values("payments", ascending=False).head(300))
    return s, prof, hist.timestamp.max().to_pydatetime()


def page_lab():
    from src.config import CATEGORIES, CHANNELS
    from src.generate_data import CITIES
    header("Test lab", "Type your own payments and see how the 2.0 engine scores them, with the same models and "
           "customer history as the live stream. Test inputs are yours, not dataset transactions.")
    scorer, prof, hist_end = lab_engine()
    if "lab_clock" not in st.session_state:
        st.session_state.lab_clock = pd.Timestamp(hist_end) + pd.Timedelta(hours=1)
        st.session_state.lab_log = []
    top = st.columns([3, 1])
    preset_name = top[0].selectbox("Start from a preset", list(PRESETS))
    if top[1].button("Reset lab", help="Reload models and history; forgets every test payment"):
        st.cache_resource.clear(); st.session_state.pop("lab_clock", None); st.rerun()
    P = PRESETS[preset_name]

    who = st.radio("Customer", ["Existing customer (has history)", "Brand-new customer"], horizontal=True)
    if who.startswith("Existing"):
        uid = st.selectbox("Customer id", prof.index, format_func=lambda u: f"{u} · {prof.payments[u]} past payments · "
                           f"usual ₹{prof.median_amount[u]:,.0f} · {prof.city[u]}")
        p = prof.loc[uid]
        st.caption(f"Usual device **{p.device}**, usual city **{p.city}**, most-paid payee **{p.payee}**.")
    else:
        uid = st.text_input("New customer id", "U_TEST_NEW")
        p = pd.Series({"median_amount": 1000.0, "device": "D_TEST_NEW", "city": "Chennai", "payee": "M00010"})

    with st.form("lab"):
        c = st.columns(4)
        default_amt = P.get("amount", round(float(p.median_amount) * P.get("amount_x", 1.0), 2))
        amount = c[0].number_input("Amount (₹)", min_value=1.0, max_value=1e7, value=float(default_amt), step=100.0)
        payee = c[1].text_input("Payee / merchant id", P.get("payee", str(p.payee)),
                                help="P2P_U<id> pays another customer; M<id> is a merchant. A new id = a payee never seen before.")
        cat = c[2].selectbox("Category", CATEGORIES, index=CATEGORIES.index(P.get("category", "grocery")))
        ch = c[3].selectbox("Channel", CHANNELS, index=CHANNELS.index(P.get("channel", "UPI")))
        c = st.columns(4)
        dev_mode = c[0].selectbox("Device", ["usual", "new", "custom"], index=["usual", "new", "custom"].index(P.get("device", "usual")))
        dev_custom = c[1].text_input("Custom device id (if custom)", "D_SHARED_TEST")
        city = c[2].selectbox("City", CITIES, index=CITIES.index(p.city) if p.city in CITIES else 0)
        hour = c[3].slider("Hour of day", 0, 23, int(P.get("hour", 14)))
        c = st.columns(4)
        repeat = c[0].number_input("Send how many times", 1, 50, int(P.get("repeat", 1)), help="Simulates bursts")
        gap = c[1].number_input("Seconds between repeats", 1, 3600, int(P.get("gap", 60)))
        vary = c[2].checkbox("Different payee each repeat", P.get("vary_payee", False))
        save = c[3].checkbox("Also save to event store", False, help="Saved tests appear in the Investigation centre, tagged MANUAL")
        go_ = st.form_submit_button("Score payment(s)", type="primary")

    if go_:
        day = st.session_state.lab_clock.normalize() + pd.Timedelta(hours=hour)
        t = max(day, st.session_state.lab_clock)                     # the engine needs time to move forward
        device = {"usual": str(p.device), "new": f"D_TEST_{np.random.randint(1e9)}", "custom": dev_custom}[dev_mode]
        con = connect() if save else None
        for i in range(int(repeat)):
            txn = {"txn_id": f"MANUAL-{int(time.time() * 1000)}-{i}", "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                   "user_id": uid, "amount": amount, "merchant_id": f"{payee}_{i}" if vary else payee, "category": cat,
                   "channel": ch, "device_id": device, "city": city}
            r = scorer.score(txn)
            st.session_state.lab_log.insert(0, {"txn": txn, "r": r})
            if con is not None:
                feats = json.dumps({k: round(float(v), 4) for k, v in r["features"].items()}) if r["decision"] != "ALLOW" else None
                con.execute(f"INSERT OR REPLACE INTO events VALUES ({','.join('?' * 27)})",
                            (None, txn["txn_id"], txn["timestamp"], uid, amount, txn["merchant_id"], cat, ch, device, city,
                             r["risk"], r["supervised_prob"], r["decision"], ",".join(r["tripwires"]), r["if_score"], r["ae_score"],
                             r["graph_score"], json.dumps(r["reasons"]), json.dumps(r["risk_breakdown"]), feats, r["latency_ms"],
                             r["latency_ms"], 0, -1, "MANUAL", "lab", time.time()))
            t += pd.Timedelta(seconds=int(gap))
        if con is not None:
            con.commit(); audit(con, st.session_state.user, st.session_state.role, "lab_test", uid, {"count": int(repeat)}); con.close()
        st.session_state.lab_clock = t

    if st.session_state.lab_log:
        last = st.session_state.lab_log[0]; r, txn = last["r"], last["txn"]
        st.markdown(f"<div class='sec'>Latest result</div><span class='pill {r['decision']}'>{r['decision']}</span> &nbsp;"
                    f"<b>₹{txn['amount']:,.0f}</b> {txn['user_id']} → {txn['merchant_id']} · {txn['timestamp']}", unsafe_allow_html=True)
        c = st.columns(4)
        kpi(c[0], "Fused risk", f"{r['risk']:.1%}"); kpi(c[1], "Supervised probability", f"{r['supervised_prob']:.1%}")
        kpi(c[2], "Novelty tripwires", ", ".join(r["tripwires"]) or "none"); kpi(c[3], "Scoring time", f"{r['latency_ms']:.2f} ms")
        if r["decision"] == "ALLOW":
            st.caption("ALLOW: risk below the REVIEW threshold "
                       f"({scorer.t['review']:.3f}) and no tripwire fired, so no reasons are generated.")
        l, rr = st.columns(2)
        if r["reasons"]:
            l.dataframe(pd.DataFrame(r["reasons"]), hide_index=True, width="stretch")
        if r["risk_breakdown"]:
            b = r["risk_breakdown"]
            f = go.Figure(go.Bar(x=[x["contribution"] for x in b][::-1], y=[x["component"] for x in b][::-1], orientation="h",
                                 marker_color=[CRIT if x["contribution"] > 0 else AQUA for x in b][::-1]))
            rr.plotly_chart(fig_base(f, 260, "What pushed the risk up or down", legend=False), width="stretch")
        f = r["features"]
        keyf = {k: round(float(f[k]), 3) for k in ("u_txn_count_total", "u_amt_ratio_mean", "u_new_device", "u_new_city",
                                                     "u_new_merchant", "u_cnt_10m", "u_small_cnt_10m", "d_n_users",
                                                     "m_users_24h", "m_age_hours", "u_passthrough")}
        with st.expander("Key behavioural features the engine computed"):
            st.json(keyf)
        st.markdown("<div class='sec'>Your test history (newest first)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"time": x["txn"]["timestamp"], "customer": x["txn"]["user_id"], "amount": x["txn"]["amount"],
                                    "payee": x["txn"]["merchant_id"], "device": x["txn"]["device_id"], "decision": x["r"]["decision"],
                                    "risk": round(x["r"]["risk"], 4), "top reason": (x["r"]["reasons"] or [{"reason": ""}])[0]["reason"]}
                                   for x in st.session_state.lab_log]), hide_index=True, width="stretch")
        st.caption("Each test payment updates the engine's memory (velocity, devices, payees), exactly as live traffic would. "
                   "Use Reset lab to start clean.")


sidebar_auth()
nav = st.navigation([st.Page(page_overview, title="Executive overview", icon="📊", default=True),
                     st.Page(page_live, title="Live monitor", icon="🔴", url_path="live"),
                     st.Page(page_network, title="Fraud network", icon="🕸️", url_path="network"),
                     st.Page(page_investigation, title="Investigation centre", icon="🔎", url_path="investigate"),
                     st.Page(page_models, title="Model intelligence", icon="🧠", url_path="models"),
                     st.Page(page_infra, title="Performance", icon="⚡", url_path="performance"),
                     st.Page(page_learning, title="Feedback & learning", icon="🔁", url_path="learning"),
                     st.Page(page_lab, title="Test lab", icon="🧪", url_path="lab")])
nav.run()
