"""Feature engine v2 = v1's 27 features + money-flow, cold-start and confirmed-fraud graph signals.

New feature groups
------------------
money flow      u_in_amt_1h, u_in_cnt_1h, u_passthrough   (money received as a P2P payee, then sent on)
cold start      u_age_hours, d_new_accounts_24h            (new accounts, many new accounts on one device)
amount pattern  is_round_amount
graph risk      m_conf_fraud_rate, d_conf_fraud_rate, u_conf_fraud_payees
                (share of a payee's / device's past payments CONFIRMED as fraud)

Leakage control: confirmed-fraud labels only become visible LABEL_DELAY_DAYS after the payment happened
(like a chargeback or analyst confirmation). In training replay the label of payment t is pushed into a
queue and released once the clock passes t + delay; the current payment's own label is never visible.
In live use, labels come from analyst feedback / chargebacks through `add_label`.
"""
from __future__ import annotations

import heapq
from collections import defaultdict, deque

from src.features import DAY, HOUR, MISSING, FeatureEngine

LABEL_DELAY_DAYS = 3.0
V2_FEATURES = ["u_in_amt_1h", "u_in_cnt_1h", "u_passthrough", "u_age_hours", "d_new_accounts_24h",
               "is_round_amount", "m_conf_fraud_rate", "d_conf_fraud_rate", "u_conf_fraud_payees",
               "m_new_payer_share_24h", "m_round_share_24h", "m_night_share_24h"]
GROUPS = {
    "transaction": ["amount", "log_amount", "hour", "is_night", "dow", "channel_code", "category_code", "is_p2p", "is_round_amount"],
    "velocity": ["u_secs_since_last", "u_cnt_10m", "u_cnt_1h", "u_cnt_24h", "u_amt_1h", "u_amt_24h", "u_small_cnt_10m"],
    "behaviour": ["u_txn_count_total", "u_amt_zscore", "u_amt_ratio_mean", "u_new_device", "u_new_city", "u_new_merchant",
                  "u_n_devices", "u_age_hours"],
    "device_graph": ["d_n_users", "d_cnt_1h", "d_new_accounts_24h"],
    "payee_graph": ["m_age_hours", "m_users_1h", "m_users_24h"],
    "money_flow": ["u_in_amt_1h", "u_in_cnt_1h", "u_passthrough"],
    "confirmed_fraud_graph": ["m_conf_fraud_rate", "d_conf_fraud_rate", "u_conf_fraud_payees"],
    "merchant_profile": ["m_new_payer_share_24h", "m_round_share_24h", "m_night_share_24h"],
}


def _zero_pair():
    return [0, 0]                                # module-level so engine state can be pickled


class FeatureEngineV2(FeatureEngine):
    def __init__(self, label_delay_days: float = LABEL_DELAY_DAYS):
        super().__init__()
        self.delay = label_delay_days * DAY
        self.incoming = defaultdict(deque)          # payee user -> (ts, amount) received in last hour
        self.first_seen = {}
        self.dev_accounts = defaultdict(deque)      # device -> (ts, user, user_first_seen) in 24h
        self.pending = []                           # heap of (release_ts, payee, device, user, label)
        self.m_lab = defaultdict(_zero_pair)    # payee  -> [confirmed fraud, labelled total]
        self.d_lab = defaultdict(_zero_pair)
        self.u_fraud_payees = defaultdict(set)      # user -> payees later confirmed as fraud receivers
        self.fraud_payees = set()
        self.m_window = defaultdict(deque)          # payee -> (ts, first_time_payer, round, night) in 24h
        self.m_payers = defaultdict(set)            # payee -> customers who have ever paid it
        self.vocab = {}                             # unseen categories/channels (public datasets) -> stable codes

    # labels -------------------------------------------------------------
    def add_label(self, payee: str, device: str, user: str, label: int) -> None:
        m = self.m_lab[payee]
        d = self.d_lab[device] if str(device) not in MISSING else [0, 0]
        m[1] += 1; d[1] += 1
        if label:
            m[0] += 1; d[0] += 1
            self.fraud_payees.add(payee)

    def _release(self, now: float) -> None:
        while self.pending and self.pending[0][0] <= now:
            _, payee, device, user, label = heapq.heappop(self.pending)
            self.add_label(payee, device, user, label)

    # features -----------------------------------------------------------
    def process(self, txn, update: bool = True, label: int | None = None):
        now = self._ts(txn["timestamp"])
        self._release(now)
        user, dev, merch = txn["user_id"], txn["device_id"], txn["merchant_id"]
        amt = float(txn["amount"])
        f = super().process(txn, update=update)
        for key, field in (("category_code", "category"), ("channel_code", "channel")):
            if f[key] == -1:
                f[key] = self.vocab.setdefault((field, str(txn[field])), 100 + len(self.vocab))

        inc = self.incoming[f"P2P_{user}"]
        while inc and now - inc[0][0] > HOUR:
            inc.popleft()
        in_amt = sum(a for _, a in inc)
        first = self.first_seen.get(user)
        dev_ok = str(dev) not in MISSING
        dq = self.dev_accounts[dev] if dev_ok else deque()
        while dq and now - dq[0][0] > DAY:
            dq.popleft()
        new_accts = {u for _, u, fs in dq if now - fs < 3 * DAY}
        if first is None or now - first < 3 * DAY:
            new_accts.add(user)
        m, d = self.m_lab.get(merch, (0, 0)), (self.d_lab.get(dev, (0, 0)) if dev_ok else (0, 0))
        mw = self.m_window[merch]
        while mw and now - mw[0][0] > DAY:
            mw.popleft()
        first_payer = user not in self.m_payers[merch]
        is_round = amt >= 500 and abs(amt - round(amt, -2)) < 1e-6
        is_night = f["is_night"] == 1
        n_w = len(mw) + 1                                   # include this payment; shrink toward 0 with +5
        shares = [(sum(r[i] for r in mw) + v) / (n_w + 5) for i, v in ((1, first_payer), (2, is_round), (3, is_night))]

        f.update({
            "u_in_amt_1h": in_amt,
            "u_in_cnt_1h": len(inc),
            "u_passthrough": min(amt, in_amt) / max(amt, in_amt) if in_amt > 0 else 0.0,
            "u_age_hours": min((now - first) / HOUR, 24 * 365) if first is not None else 0.0,
            "d_new_accounts_24h": len(new_accts),
            "is_round_amount": 1 if amt >= 500 and abs(amt - round(amt, -2)) < 1e-6 else 0,
            "m_conf_fraud_rate": (m[0] + 0.0) / (m[1] + 5.0),      # shrunk toward 0 for small counts
            "d_conf_fraud_rate": (d[0] + 0.0) / (d[1] + 5.0),
            "u_conf_fraud_payees": len(self.u_fraud_payees[user] & self.fraud_payees),
            "m_new_payer_share_24h": shares[0], "m_round_share_24h": shares[1], "m_night_share_24h": shares[2],
        })

        if update:
            if str(merch).startswith("P2P_"):
                self.incoming[merch].append((now, amt))
            mw.append((now, first_payer, is_round, is_night))
            self.m_payers[merch].add(user)
            if first is None:
                self.first_seen[user] = now
            if dev_ok:
                dq.append((now, user, self.first_seen[user]))
            if label is not None:
                heapq.heappush(self.pending, (now + self.delay, merch, dev, user, int(label)))
                self.u_fraud_payees[user].add(merch)
        return f


def build_feature_table_v2(df, label_delay_days: float = LABEL_DELAY_DAYS):
    import pandas as pd
    from src.config import FEATURE_COLUMNS
    eng = FeatureEngineV2(label_delay_days)
    import numpy as np
    cols = FEATURE_COLUMNS + V2_FEATURES
    out = np.empty((len(df), len(cols)), dtype=np.float64)      # filled in place: memory stays flat on millions of rows
    use = ["timestamp", "amount", "user_id", "device_id", "merchant_id", "city", "category", "channel", "is_fraud"]
    chunk = 100_000
    for start in range(0, len(df), chunk):
        recs = df[use].iloc[start:start + chunk].to_dict("records")
        for i, r in enumerate(recs, start):
            f = eng.process(r, label=r["is_fraud"])
            out[i] = [f[c] for c in cols]
        del recs
    return pd.DataFrame(out, columns=cols, index=df.index), eng

