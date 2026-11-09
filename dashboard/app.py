"""payTrace live dashboard.

  streamlit run dashboard/app.py
Then press "Start live replay" in the sidebar (or run `python -m src.stream` separately).
"""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import REPORTS_DIR, STREAM_DB  # noqa: E402

# Palette (dark-mode steps of the validated reference palette)
BLUE, ORANGE, AQUA, VIOLET = "#3987e5", "#d95926", "#199e70", "#9085e9"
CRITICAL, WARNING, GOOD = "#d03b3b", "#fab219", "#0ca30c"
INK2, GRID = "#c3c2b7", "rgba(195,194,183,0.15)"

st.set_page_config(page_title="payTrace", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;padding-bottom:1rem}
div[data-testid="stMetricValue"]{font-size:1.55rem}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:.78rem;font-weight:600}
.BLOCK{background:#d03b3b33;color:#ff8a8a} .REVIEW{background:#fab21933;color:#fab219}
</style>""", unsafe_allow_html=True)


def fig_base(fig, h=260, title=None):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=64 if title else 8, b=8),
                      title=dict(text=title, font=dict(size=14, color=INK2)) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK2, size=12), hovermode="x unified",
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    return fig


@st.cache_data
def load_json(name):
    p = REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def read_db():
    if not STREAM_DB.exists():
        return None, None, {}
    con = sqlite3.connect(f"file:{STREAM_DB}?mode=ro", uri=True, timeout=5)
    try:
        ev = pd.read_sql("SELECT * FROM events", con)
        dr = pd.read_sql("SELECT * FROM drift", con)
        stt = dict(con.execute("SELECT k, v FROM status").fetchall())
    except Exception:
        return None, None, {}
    finally:
        con.close()
    return ev, dr, stt


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🛡️ payTrace")
    st.caption("Real-time payment fraud detection · DataNova 2026 · DN-FN-01")
    rate = st.slider("Replay speed (txns/sec)", 50, 2000, 300, 50)
    if st.button("▶ Start live replay", width="stretch", type="primary"):
        if "proc" in st.session_state and st.session_state.proc.poll() is None:
            st.session_state.proc.terminate()
        st.session_state.proc = subprocess.Popen(
            [sys.executable, "-m", "src.stream", "--rate", str(rate)], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        st.toast("Warming feature state, stream starts in ~5s")
    if st.button("■ Stop", width="stretch"):
        if "proc" in st.session_state:
            st.session_state.proc.terminate()
    st.divider()
    st.caption("Test period = last 15 days of data, never seen in training. "
               "Includes a fraud pattern (QR-swap fake merchants) absent from training.")

st.markdown("## Live payment fraud monitor")
tab_live, tab_model, tab_lat, tab_drift = st.tabs(
    ["🔴 Live stream", "📊 Model evaluation", "⚡ Latency", "🧭 Drift & novelty"])


# ---------------- live tab ----------------
@st.fragment(run_every="2s")
def live():
    ev, _, stt = read_db()
    if ev is None or ev.empty:
        st.info("No stream yet. Press **Start live replay** in the sidebar." if not stt
                else "Warming up feature state from history…")
        return
    total = int(stt.get("total", len(ev)))
    st.progress(min(len(ev) / max(total, 1), 1.0),
                text=f"{stt.get('state', '')} · {len(ev):,} / {total:,} transactions scored")
    flagged = ev.decision != "ALLOW"
    y = ev.is_fraud == 1
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    blocked_fraud = ev.loc[(ev.decision == "BLOCK") & y, "amount"].sum()
    caught_fraud = ev.loc[flagged & y, "amount"].sum()
    fraud_total = ev.loc[y, "amount"].sum()
    c = st.columns(6)
    c[0].metric("Transactions", f"{len(ev):,}")
    c[1].metric("Alerts", f"{int(flagged.sum()):,}",
                f"{int((ev.decision == 'BLOCK').sum())} blocked", delta_color="off")
    c[2].metric("Precision (live)", f"{tp / max(tp + fp, 1):.0%}")
    c[3].metric("Recall (live)", f"{tp / max(int(y.sum()), 1):.0%}")
    c[4].metric("Latency p50 / p99", f"{ev.latency_ms.quantile(.5):.2f} / {ev.latency_ms.quantile(.99):.2f} ms")
    c[5].metric("Fraud ₹ intercepted", f"₹{caught_fraud / 1e5:,.1f}L",
                f"{caught_fraud / max(fraud_total, 1):.0%} of fraud value", delta_color="off")

    ev["hour"] = pd.to_datetime(ev.ts, format="mixed").dt.floor("h")
    g = ev.groupby("hour").agg(n=("txn_id", "size"),
                               block=("decision", lambda s: (s == "BLOCK").sum()),
                               review=("decision", lambda s: (s == "REVIEW").sum()))
    l, r = st.columns(2)
    f1 = go.Figure(go.Scatter(x=g.index, y=g.n, mode="lines", line=dict(color=BLUE, width=2),
                              name="Transactions", hovertemplate="%{y:,} txns<extra></extra>"))
    l.plotly_chart(fig_base(f1, 230, "Transactions per hour"), width="stretch")
    f2 = go.Figure([go.Bar(x=g.index, y=g.block, name="Blocked", marker_color=CRITICAL),
                    go.Bar(x=g.index, y=g.review, name="Review", marker_color=WARNING)])
    f2.update_layout(barmode="stack", bargap=0.15)
    r.plotly_chart(fig_base(f2, 230, "Alerts per hour"), width="stretch")

    st.markdown("#### Alert feed")
    al = ev[flagged].sort_values("seq", ascending=False).head(60)
    feed = pd.DataFrame({
        "Time": pd.to_datetime(al.ts, format="mixed").dt.strftime("%d %b %H:%M:%S"),
        "Decision": al.decision, "Risk": al.probability * 100, "Amount ₹": al.amount,
        "User": al.user_id, "Payee": al.merchant_id, "Category": al.category,
        "Top reasons": al.reasons.map(lambda s: " · ".join(x["reason"] for x in json.loads(s)[:3])),
        "Ground truth": al.fraud_type,
    })
    st.dataframe(feed, hide_index=True, width="stretch", height=300,
                 column_config={"Risk": st.column_config.ProgressColumn(format="%.0f%%", min_value=0,
                                                                        max_value=100),
                                "Amount ₹": st.column_config.NumberColumn(format="₹%.0f")})
    if len(al):
        top = al.iloc[0]
        st.markdown(f"**Latest alert** <span class='pill {top.decision}'>{top.decision}</span> "
                    f"&nbsp;`{top.txn_id}`", unsafe_allow_html=True)
        st.caption("🤖 Analyst copilot: " + top.note)


with tab_live:
    live()

# ---------------- model tab ----------------
with tab_model:
    m = load_json("metrics.json")
    if not m:
        st.warning("Run `python -m src.train` first.")
    else:
        fs, kn, bz = m["paytrace"], m["paytrace"]["known_patterns_only"], m["business"]
        c = st.columns(5)
        c[0].metric("PR-AUC (known patterns)", f"{kn['pr_auc']:.3f}")
        c[1].metric("Recall, known patterns", f"{kn['recall']:.1%}")
        c[2].metric("Block-tier precision", f"{fs['at_block_threshold']['precision']:.1%}")
        c[3].metric("False-positive rate", f"{fs['at_review_threshold']['fpr']:.2%}")
        c[4].metric("Loss reduction vs no model",
                    f"{1 - bz['cost_with_paytrace_inr'] / bz['cost_without_model_inr']:.0%}")
        l, r = st.columns(2)
        costs = {"No model": bz["cost_without_model_inr"], "Rules": bz["cost_with_rules_inr"],
                 "Isolation Forest only": m["baselines"]["isolation_forest_only"]["total_cost_inr"],
                 "LightGBM raw fields": m["baselines"]["lgbm_raw_features_only"]["total_cost_inr"],
                 "payTrace": bz["cost_with_paytrace_inr"]}
        fc = go.Figure(go.Bar(x=list(costs.values()), y=list(costs), orientation="h",
                              marker_color=[GRID.replace("0.15", "0.55")] * 4 + [BLUE],
                              text=[f"₹{v / 1e5:,.1f}L" for v in costs.values()], textposition="outside",
                              hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        fc.update_layout(hovermode="closest")
        fc.update_xaxes(showticklabels=False, range=[0, max(costs.values()) * 1.18])
        l.plotly_chart(fig_base(fc, 300, "Total loss on test period (missed fraud ₹ + ₹100 per false alarm)"),
                       width="stretch")
        rp = m["recall_by_pattern"]
        names = {"ATO": "Account takeover", "CARD_TESTING": "Card testing",
                 "MULE_P2P": "Mule P2P scam", "QR_SWAP": "QR-swap (novel)"}
        fr = go.Figure([go.Bar(x=[names[k] for k in rp], y=[v["lgbm"] for v in rp.values()],
                               name="Supervised model", marker_color=BLUE),
                        go.Bar(x=[names[k] for k in rp], y=[v["lgbm_plus_novelty_tripwire"] for v in rp.values()],
                               name="+ novelty tripwire", marker_color=ORANGE)])
        fr.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fr.update_yaxes(tickformat=".0%", range=[0, 1.05])
        r.plotly_chart(fig_base(fr, 300, "Recall by fraud pattern"), width="stretch")

        l, r = st.columns(2)
        pr = m["pr_curve"]
        fp_ = go.Figure(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines",
                                   line=dict(color=BLUE, width=2), name="payTrace",
                                   hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}<extra></extra>"))
        fp_.update_layout(hovermode="closest")
        fp_.update_xaxes(title="Recall", range=[0, 1]); fp_.update_yaxes(title="Precision", range=[0, 1.02])
        l.plotly_chart(fig_base(fp_, 300, "Precision–recall curve (test, all patterns)"), width="stretch")
        imp = m["feature_importance_mean_abs_shap"]
        fi = go.Figure(go.Bar(x=list(imp.values())[::-1], y=list(imp)[::-1], orientation="h",
                              marker_color=BLUE, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fi.update_layout(hovermode="closest")
        r.plotly_chart(fig_base(fi, 300, "Top features (mean |SHAP|)"), width="stretch")

        fb = m["feedback_retrain"]
        st.markdown("#### Feedback loop: closing the gap on a new scam")
        c = st.columns(3)
        c[0].metric("QR-swap recall, original model", f"{fb['champion']['qr_swap_recall']:.0%}")
        c[1].metric("After retrain on 5 days of analyst labels", f"{fb['challenger']['qr_swap_recall']:.0%}",
                    f"{fb['qr_swap_labels_used']} new labels used", delta_color="off")
        c[2].metric("PR-AUC on following 10 days", f"{fb['challenger']['pr_auc']:.3f}",
                    f"{fb['challenger']['pr_auc'] - fb['champion']['pr_auc']:+.3f} vs original")
        rows = [{"Approach": "payTrace (review tier)", **{k: fs["at_review_threshold"][k] for k in ("precision", "recall", "fpr", "alerts")}},
                {"Approach": "payTrace (block tier)", **{k: fs["at_block_threshold"][k] for k in ("precision", "recall", "fpr", "alerts")}},
                {"Approach": "payTrace + novelty tripwire", **{k: fs["with_novelty_tripwire"][k] for k in ("precision", "recall", "fpr", "alerts")}}]
        rows += [{"Approach": f"Baseline: {k.replace('_', ' ')}", **{x: v[x] for x in ("precision", "recall", "fpr", "alerts")}}
                 for k, v in m["baselines"].items()]
        with st.expander("Full comparison table"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"precision": st.column_config.NumberColumn(format="%.3f"),
                                        "recall": st.column_config.NumberColumn(format="%.3f"),
                                        "fpr": st.column_config.NumberColumn(format="%.4f")})

# ---------------- latency tab ----------------
with tab_lat:
    lat = load_json("latency.json")
    if not lat:
        st.warning("Run `python -m src.benchmark` first.")
    else:
        L = lat["latency_ms"]
        c = st.columns(5)
        c[0].metric("p50", f"{L['p50']:.3f} ms"); c[1].metric("p95", f"{L['p95']:.3f} ms")
        c[2].metric("p99", f"{L['p99']:.3f} ms"); c[3].metric("max", f"{L['max']:.2f} ms")
        c[4].metric("Throughput (1 core)", f"{lat['throughput_txn_per_sec_single_core']:,.0f} txn/s")
        l, r = st.columns([3, 2])
        e = lat["histogram"]["edges_ms"]
        fh = go.Figure(go.Bar(x=[(a + b) / 2 for a, b in zip(e, e[1:])], y=lat["histogram"]["counts"],
                              marker_color=BLUE, hovertemplate="%{x:.3f} ms: %{y:,}<extra></extra>"))
        fh.update_layout(bargap=0.08, hovermode="closest")
        fh.update_xaxes(title="Latency (ms)")
        l.plotly_chart(fig_base(fh, 300, f"End-to-end latency, {lat['n_transactions']:,} transactions"),
                       width="stretch")
        s = lat["stage_mean_ms"]
        fs_ = go.Figure(go.Bar(x=list(s.values()), y=list(s), orientation="h", marker_color=BLUE,
                               text=[f"{v * 1000:.0f} µs" for v in s.values()], textposition="outside",
                               hovertemplate="%{y}: %{x:.4f} ms<extra></extra>"))
        fs_.update_layout(hovermode="closest")
        fs_.update_xaxes(showticklabels=False, range=[0, max(s.values()) * 1.25])
        r.plotly_chart(fig_base(fs_, 300, "Mean time per stage"), width="stretch")
        st.caption(f"Measured on {lat['machine']}. Includes feature computation, both Isolation "
                   "Forests, LightGBM, calibration, decision and SHAP reason codes for flagged payments.")

# ---------------- drift tab ----------------
with tab_drift:
    ev, dr, _ = read_db()
    if dr is None or dr.empty:
        st.info("Drift is computed every 2,000 streamed transactions once a full 24h window exists.")
    else:
        l, r = st.columns(2)
        last = dr[dr.seq == dr.seq.max()].sort_values("psi")
        fd = go.Figure(go.Bar(x=last.psi, y=last.feature, orientation="h", marker_color=BLUE,
                              hovertemplate="%{y}: PSI %{x:.3f}<extra></extra>"))
        for x, lab in ((0.1, "moderate"), (0.25, "significant")):
            fd.add_vline(x=x, line=dict(color=INK2, width=1, dash="dot"),
                         annotation_text=lab, annotation_font_color=INK2)
        fd.update_layout(hovermode="closest")
        l.plotly_chart(fig_base(fd, 320, "Feature drift, last 24h vs training (PSI)"), width="stretch")
        ev["day"] = pd.to_datetime(ev.ts, format="mixed").dt.floor("D")
        nv = ev.groupby("day").novelty.sum()
        fn = go.Figure(go.Scatter(x=nv.index, y=nv.values, mode="lines+markers",
                                  line=dict(color=ORANGE, width=2), marker=dict(size=8),
                                  hovertemplate="%{x|%d %b}: %{y} novelty alerts<extra></extra>"))
        fn.update_yaxes(rangemode="tozero")
        r.plotly_chart(fig_base(fn, 320, "Novelty-tripwire alerts per day (early warning for new scams)"),
                       width="stretch")
        st.caption("PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 retrain. PSI barely moves for rare new "
                   "fraud, so the novelty-alert rate is the faster signal: a rise triggers analyst labelling "
                   "and a challenger retrain (see Model evaluation → Feedback loop).")
