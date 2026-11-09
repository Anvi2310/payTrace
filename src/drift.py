"""Population Stability Index (PSI) drift monitoring over a rolling 24h window.

PSI < 0.1  stable | 0.1-0.25 moderate shift | > 0.25 significant shift (retrain / investigate)

Only stationary signals are monitored: raw payee age grows naturally as the data ages, so we
track the share of payments going to payees first seen <24h ago instead.
PSI is insensitive to rare new fraud patterns, so the dashboard also tracks the novelty-
tripwire alert rate, which is the faster signal for a brand-new scam.
"""
import numpy as np
import pandas as pd

MONITORED = ["amount", "hour", "u_cnt_1h", "u_amt_ratio_mean", "u_new_merchant",
             "m_new_payee_24h", "m_users_1h", "d_n_users", "is_p2p"]


def drift_values(f):
    """Monitored values from one feature dict."""
    out = {c: f[c] for c in MONITORED if c in f}
    out["m_new_payee_24h"] = 1.0 if f["m_age_hours"] < 24 else 0.0
    return out


def psi_reference(df, bins=10):
    df = df.copy()
    df["m_new_payee_24h"] = (df["m_age_hours"] < 24).astype(float)
    ref = {}
    for col in MONITORED:
        v = df[col].to_numpy(dtype=float)
        edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:                      # near-constant / binary feature
            edges = np.array([-np.inf, 0.5, np.inf])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        counts = np.histogram(v, edges)[0]
        ref[col] = {"edges": [float(e) for e in edges], "pct": (counts / counts.sum()).tolist()}
    return ref


def psi(reference, values_by_col):
    out = {}
    for col, r in reference.items():
        v = np.asarray(values_by_col.get(col, []), dtype=float)
        if len(v) < 50:
            continue
        edges = np.array(r["edges"])
        cur = np.histogram(v, edges)[0] / len(v)
        ref = np.array(r["pct"])
        cur, ref = np.clip(cur, 1e-4, None), np.clip(ref, 1e-4, None)
        out[col] = float(np.sum((cur - ref) * np.log(cur / ref)))
    return out


# ------------------------------------------------------------------------------------------- 2.1: streaming drift
class PageHinkley:
    """Page-Hinkley test for an upward shift in the mean of a stream (e.g. alert rate, risk score, amount).
    delta: tolerated change; threshold (lambda): alarm level. Alarms reset the statistic so later shifts are found."""

    def __init__(self, delta: float = 0.005, threshold: float = 5.0, burn_in: int = 200):
        self.delta, self.threshold, self.burn_in = delta, threshold, burn_in
        self.reset()

    def reset(self):
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n > self.burn_in and self.cum - self.min_cum > self.threshold:
            self.reset()
            return True
        return False


def _psi_arrays(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float | None:
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 100 or len(cur) < 50:
        return None
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        e = np.array([-np.inf, 0.5, np.inf])
    else:
        e[0], e[-1] = -np.inf, np.inf
    a = np.clip(np.histogram(ref, e)[0] / len(ref), 1e-4, None)
    b = np.clip(np.histogram(cur, e)[0] / len(cur), 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def level(v):
    return None if v is None else "stable" if v < 0.1 else "moderate shift" if v < 0.25 else "significant shift"


def drift_report(events: pd.DataFrame, training: pd.DataFrame | None = None, current_hours: float = 24.0) -> dict:
    """events: scored events (ts, amount, decision, risk, merchant_id, device_id, user_id), time-ordered.
    Reference for amounts and hour of day = the training data (if given), else the older part of the window;
    reference for model outputs (risk, alert rate) = events before the current window.
    Also: share of payments to payees / from devices first seen in the window (new beneficiary / device behaviour),
    and Page-Hinkley alarms on the alert-rate and risk streams."""
    if events.empty:
        return {"status": "no scored events yet"}
    ev = events.sort_values("ts").reset_index(drop=True)
    t = pd.to_datetime(ev.ts.str[:19])
    cut = t.max() - pd.Timedelta(hours=current_hours)
    cur, ref = ev[t > cut], ev[t <= cut]
    if len(ref) < 100:                                   # not enough history: compare halves
        half = len(ev) // 2
        ref, cur = ev.iloc[:half], ev.iloc[half:]
    hour = lambda d: pd.to_datetime(d.ts.str[:19]).dt.hour.to_numpy(float)
    out = {"window": {"reference": [str(ref.ts.min())[:16], str(ref.ts.max())[:16], int(len(ref))],
                      "current": [str(cur.ts.min())[:16], str(cur.ts.max())[:16], int(len(cur))]}, "psi": {}}
    tr = training if training is not None and len(training) else None
    out["psi"]["amount (vs training data)" if tr is not None else "amount"] = _psi_arrays(
        np.log1p((tr if tr is not None else ref).amount.to_numpy(float)), np.log1p(cur.amount.to_numpy(float)))
    out["psi"]["hour of day (vs training data)" if tr is not None else "hour of day"] = _psi_arrays(
        (pd.to_datetime(tr.timestamp).dt.hour.to_numpy(float) if tr is not None else hour(ref)), hour(cur))
    out["psi"]["risk score"] = _psi_arrays(ref.risk.to_numpy(float), cur.risk.to_numpy(float))
    first_payee = ev.groupby("merchant_id").ts.transform("min")
    first_dev = ev.groupby("device_id").ts.transform("min")
    newp = (ev.ts == first_payee).to_numpy(float)
    newd = ((ev.ts == first_dev) & ~ev.device_id.isin(["UNKNOWN", "", "nan"])).to_numpy(float)
    m_cur = (t > cut).to_numpy() if len(ev[t <= cut]) >= 100 else np.arange(len(ev)) >= len(ev) // 2
    out["rates"] = {
        "alert_rate": {"reference": float((ref.decision != "ALLOW").mean()), "current": float((cur.decision != "ALLOW").mean())},
        "block_rate": {"reference": float((ref.decision == "BLOCK").mean()), "current": float((cur.decision == "BLOCK").mean())},
        "new_payee_share": {"reference": float(newp[~m_cur].mean()), "current": float(newp[m_cur].mean())},
        "new_device_share": {"reference": float(newd[~m_cur].mean()), "current": float(newd[m_cur].mean())},
        "mean_amount": {"reference": float(ref.amount.mean()), "current": float(cur.amount.mean())}}
    alarms = {}
    for name, series, ph in (("alert rate", (ev.decision != "ALLOW").to_numpy(float), PageHinkley(0.005, 15.0)),
                             ("risk score", ev.risk.to_numpy(float), PageHinkley(0.01, 10.0))):
        hits = [i for i, x in enumerate(series) if ph.update(float(x))]
        alarms[name] = [{"at": str(ev.ts.iloc[i])[:16], "index": int(i)} for i in hits[-5:]]
    out["page_hinkley_alarms"] = alarms
    out["levels"] = {k: level(v) for k, v in out["psi"].items()}
    worst = max((v for v in out["psi"].values() if v is not None), default=0.0)
    out["status"] = level(worst) if worst else "not enough data"
    return out
