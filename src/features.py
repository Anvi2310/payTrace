"""Online (streaming) feature engine.

The SAME class is used for offline training (replaying history in time order) and for
real-time serving. That removes train/serve skew, the most common reason fraud models
look great in a notebook and fail in production.

For every transaction we first compute features from the state *before* the transaction,
then update the state. In production the state would live in Redis (keys per user /
device / payee with TTLs); here it is in-process dictionaries with identical semantics.
"""
import math
from collections import Counter, defaultdict, deque
from functools import partial
from datetime import datetime

from src.config import CATEGORIES, CHANNELS, FEATURE_COLUMNS

DAY = 86400.0
HOUR = 3600.0
SMALL_AMOUNT = 100.0
# Values that mean "not recorded". They must never be treated as a real shared entity, otherwise thousands of
# accounts appear to share one device (found by robustness testing: PR-AUC fell from 0.94 to 0.15).
MISSING = {"", "UNKNOWN", "unknown", "nan", "NaN", "None", "NA"}
_CH = {c: i for i, c in enumerate(CHANNELS)}
_CAT = {c: i for i, c in enumerate(CATEGORIES)}


class _UserState:
    __slots__ = ("events", "n", "mean", "m2", "last_ts", "devices", "cities", "merchants", "total")

    def __init__(self):
        self.events = deque()          # (ts, amount) within 24h
        self.n = 0                     # Welford stats on log(amount)
        self.mean = 0.0
        self.m2 = 0.0
        self.last_ts = None
        self.devices = set()
        self.cities = set()
        self.merchants = set()
        self.total = 0


class _WindowDistinct:
    """Distinct keys seen within a sliding time window."""
    __slots__ = ("window", "q", "cnt")

    def __init__(self, window):
        self.window = window
        self.q = deque()
        self.cnt = Counter()

    def prune(self, now):
        q, cnt, w = self.q, self.cnt, self.window
        while q and now - q[0][0] > w:
            _, k = q.popleft()
            cnt[k] -= 1
            if cnt[k] == 0:
                del cnt[k]

    def add(self, now, key):
        self.q.append((now, key))
        self.cnt[key] += 1

    def distinct(self, now, extra=None):
        self.prune(now)
        n = len(self.cnt)
        if extra is not None and extra not in self.cnt:
            n += 1
        return n


class FeatureEngine:
    def __init__(self):
        self.users = defaultdict(_UserState)
        self.dev_users = defaultdict(set)
        self.dev_events = defaultdict(deque)
        self.merch_first = {}
        self.merch_1h = defaultdict(partial(_WindowDistinct, HOUR))   # partial, not lambda: the state must be picklable
        self.merch_24h = defaultdict(partial(_WindowDistinct, DAY))

    @staticmethod
    def _ts(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        return datetime.fromisoformat(str(value)).timestamp()

    def process(self, txn, update=True):
        """Return the feature dict for `txn` (a dict) and update state."""
        now = self._ts(txn["timestamp"])
        amt = float(txn["amount"])
        user, dev, merch = txn["user_id"], txn["device_id"], txn["merchant_id"]
        city, cat, ch = txn["city"], txn["category"], txn["channel"]
        dt = datetime.fromtimestamp(now)
        log_amt = math.log1p(amt)

        u = self.users[user]
        ev = u.events
        while ev and now - ev[0][0] > DAY:
            ev.popleft()
        c10 = c1h = 0
        a1h = a24 = 0.0
        small10 = 0
        for t, a in ev:
            age = now - t
            a24 += a
            if age <= HOUR:
                c1h += 1
                a1h += a
                if age <= 600:
                    c10 += 1
                    if a < SMALL_AMOUNT:
                        small10 += 1
        std = math.sqrt(u.m2 / (u.n - 1)) if u.n > 2 else 0.0
        z = (log_amt - u.mean) / max(std, 0.25) if u.n >= 3 else 0.0
        ratio = amt / math.expm1(u.mean) if u.n >= 3 and u.mean > 0 else 1.0

        dev_ok = str(dev) not in MISSING
        city_ok = str(city) not in MISSING
        dq = self.dev_events[dev] if dev_ok else deque()
        while dq and now - dq[0] > HOUR:
            dq.popleft()
        d_users = self.dev_users[dev] if dev_ok else set()
        first = self.merch_first.get(merch)

        f = {
            "amount": amt,
            "log_amount": log_amt,
            "hour": dt.hour + dt.minute / 60.0,
            "is_night": 1 if dt.hour < 5 else 0,
            "dow": dt.weekday(),
            "channel_code": _CH.get(ch, -1),
            "category_code": _CAT.get(cat, -1),
            "is_p2p": 1 if cat == "P2P" else 0,
            "u_txn_count_total": u.total,
            "u_secs_since_last": min(now - u.last_ts, 30 * DAY) if u.last_ts is not None else 30 * DAY,
            "u_cnt_10m": c10,
            "u_cnt_1h": c1h,
            "u_cnt_24h": len(ev),
            "u_amt_1h": a1h,
            "u_amt_24h": a24,
            "u_small_cnt_10m": small10,
            "u_amt_zscore": z,
            "u_amt_ratio_mean": min(ratio, 1000.0),
            "u_new_device": 1 if (dev_ok and u.total > 0 and dev not in u.devices) else 0,
            "u_new_city": 1 if (city_ok and u.total > 0 and city not in u.cities) else 0,
            "u_new_merchant": 1 if (u.total > 0 and merch not in u.merchants) else 0,
            "u_n_devices": max(len(u.devices | ({dev} if dev_ok else set())), 1),
            "d_n_users": len(d_users | {user}),
            "d_cnt_1h": len(dq),
            "m_age_hours": min((now - first) / HOUR, 24 * 365) if first is not None else 0.0,
            "m_users_1h": self.merch_1h[merch].distinct(now, extra=user),
            "m_users_24h": self.merch_24h[merch].distinct(now, extra=user),
        }

        if update:
            ev.append((now, amt))
            u.n += 1
            d = log_amt - u.mean
            u.mean += d / u.n
            u.m2 += d * (log_amt - u.mean)
            u.last_ts = now
            if dev_ok:
                u.devices.add(dev)
                dq.append(now)
                d_users.add(user)
            if city_ok:
                u.cities.add(city)
            u.merchants.add(merch)
            u.total += 1
            if first is None:
                self.merch_first[merch] = now
            self.merch_1h[merch].add(now, user)
            self.merch_24h[merch].add(now, user)
        return f

    def vector(self, f):
        return [f[c] for c in FEATURE_COLUMNS]


def build_feature_table(df):
    """Replay a time-ordered transactions DataFrame through the engine."""
    import pandas as pd
    eng = FeatureEngine()
    recs = df[["timestamp", "amount", "user_id", "device_id", "merchant_id",
               "city", "category", "channel"]].to_dict("records")
    feats = [eng.process(r) for r in recs]
    out = pd.DataFrame(feats, columns=FEATURE_COLUMNS)
    return out, eng
