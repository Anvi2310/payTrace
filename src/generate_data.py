"""Synthetic payment-transaction generator.

Used to develop and demo the pipeline before (or alongside) the organizers' dataset.
It models realistic user behaviour plus four fraud patterns:

  1. ATO          - account takeover: new device + new city, burst of high-value spends
  2. CARD_TESTING - many tiny online payments in minutes, then a cash-out
  3. MULE_P2P     - social-engineering scam: victim (own device!) sends a large P2P transfer
                    to a mule account that receives from many victims
  4. QR_SWAP      - NOVEL pattern that appears only in the last 15 days (never in training):
                    a freshly created fake merchant collects payments from many users quickly.
                    Used to show drift detection + unsupervised anomaly detection.

Usage:  python -m src.generate_data --users 5000 --days 60
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import RAW_TXNS, DATA_DIR

CITIES = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Kolkata",
          "Pune", "Coimbatore", "Madurai", "Kochi"]
# category: (median amount INR, lognormal sigma, online?)
CATS = {
    "grocery": (450, 0.6, False), "food_delivery": (380, 0.5, True), "fuel": (1500, 0.4, False),
    "electronics": (7000, 0.9, True), "travel": (4500, 0.8, True), "utilities": (1200, 0.5, True),
    "fashion": (1800, 0.7, True), "P2P": (1500, 1.0, False), "gaming": (400, 0.8, True),
    "gift_cards": (1000, 0.7, True), "crypto": (5000, 1.0, True),
}
CAT_NAMES = list(CATS)
LEGIT_CAT_W = np.array([0.24, 0.17, 0.08, 0.03, 0.04, 0.10, 0.08, 0.20, 0.03, 0.02, 0.01])
START = datetime(2026, 8, 1)


def build_world(rng, n_users, n_merchants):
    merchants = pd.DataFrame({
        "merchant_id": [f"M{i:05d}" for i in range(n_merchants)],
        "category": rng.choice([c for c in CAT_NAMES if c != "P2P"], n_merchants,
                               p=_renorm(LEGIT_CAT_W, exclude=CAT_NAMES.index("P2P"))),
    })
    by_cat = {c: merchants.loc[merchants.category == c, "merchant_id"].to_numpy()
              for c in CAT_NAMES if c != "P2P"}
    users = []
    for u in range(n_users):
        n_dev = 1 if rng.random() < 0.8 else 2
        users.append({
            "user_id": f"U{u:05d}",
            "city": rng.choice(CITIES, p=[.18, .16, .14, .13, .1, .07, .07, .06, .05, .04]),
            "rate": rng.gamma(2.0, 0.5),                       # txns per day (mean 1.0)
            "amt_mult": float(np.exp(rng.normal(0, 0.45))),    # spending level
            "peak_hour": float(np.clip(rng.normal(15, 3.5), 7, 23)),
            "devices": [f"D{u:05d}_{k}" for k in range(n_dev)],
            "cat_w": rng.dirichlet(LEGIT_CAT_W * 40),
            "fav_merchants": {c: rng.choice(m, size=min(len(m), rng.integers(2, 6)), replace=False)
                              for c, m in by_cat.items() if len(m)},
            "contacts": [f"P2P_U{x:05d}" for x in rng.integers(0, n_users, rng.integers(3, 12))],
        })
    return users, by_cat


def _renorm(w, exclude):
    w = np.delete(w, exclude)
    return w / w.sum()


def _channel(rng, cat):
    if cat == "P2P":
        return "UPI"
    return rng.choice(["UPI", "CARD", "NETBANKING"], p=[0.55, 0.38, 0.07])


def gen_legit(rng, users, by_cat, days):
    rows = []
    for u in users:
        n = rng.poisson(u["rate"] * days)
        if n == 0:
            continue
        day = rng.uniform(0, days, n).astype(int)
        hour = np.mod(rng.normal(u["peak_hour"], 3.0, n), 24.0)   # wrap, never pile up at midnight
        night = rng.random(n) < 0.03                    # night owls / late orders
        hour[night] = rng.uniform(0, 5, night.sum())
        cats = rng.choice(CAT_NAMES, n, p=u["cat_w"])
        for i in range(n):
            cat = cats[i]
            med, sig, _ = CATS[cat]
            amt = med * u["amt_mult"] * np.exp(rng.normal(0, sig))
            if rng.random() < 0.01:                       # occasional big purchase
                amt *= rng.uniform(4, 12)
            if cat == "P2P":
                payee = (rng.choice(u["contacts"]) if rng.random() < 0.9
                         else f"P2P_U{rng.integers(0, len(users)):05d}")
            else:
                favs = u["fav_merchants"][cat]
                payee = rng.choice(favs) if rng.random() < 0.75 else rng.choice(by_cat[cat])
            r = rng.random()
            if r < 0.015:      # new phone / borrowed phone
                device = f"D_NEW_{rng.integers(0, 10**9)}"
            else:
                device = rng.choice(u["devices"])
            city = u["city"] if rng.random() > 0.025 else rng.choice(CITIES)  # travel
            ts = START + timedelta(days=int(day[i]), hours=float(hour[i]))
            rows.append((ts, u["user_id"], round(float(amt), 2), payee, cat, _channel(rng, cat),
                         device, city, 0, "LEGIT"))
    return rows


def gen_fraud(rng, users, by_cat, days):
    rows = []
    user_idx = np.arange(len(users))
    attacker_devices = [f"D_ATK_{k:03d}" for k in range(60)]      # reused across victims
    mule_accounts = [f"P2P_MULE{k:03d}" for k in range(25)]
    online_small = np.concatenate([by_cat["gaming"], by_cat["gift_cards"], by_cat["food_delivery"]])

    def ts_at(day, hour, minutes=0.0):
        return START + timedelta(days=float(day), hours=float(hour), minutes=float(minutes))

    # 1. Account takeover
    for _ in range(350):
        u = users[rng.choice(user_idx)]
        day = rng.uniform(1, days - 0.5)
        hour = rng.uniform(0, 24) if rng.random() < 0.6 else rng.uniform(0, 5)
        device = rng.choice(attacker_devices) if rng.random() < 0.7 else f"D_ATK_X{rng.integers(10**9)}"
        city = rng.choice([c for c in CITIES if c != u["city"]])
        stealthy = rng.random() < 0.2
        t = 0.0
        for _k in range(rng.integers(2, 8)):
            cat = rng.choice(["electronics", "gift_cards", "crypto", "P2P", "fashion"],
                             p=[.3, .25, .15, .2, .1])
            med = CATS[cat][0] * u["amt_mult"]
            amt = med * (rng.uniform(0.5, 1.5) if stealthy else rng.uniform(2, 8))
            payee = rng.choice(mule_accounts) if cat == "P2P" else rng.choice(by_cat[cat])
            rows.append((ts_at(day, hour, t), u["user_id"], round(float(amt), 2), payee, cat,
                         _channel(rng, cat), device, city, 1, "ATO"))
            t += rng.uniform(0.5, 8)

    # 2. Card testing
    for _ in range(110):
        u = users[rng.choice(user_idx)]
        day, hour = rng.uniform(1, days - 0.5), rng.uniform(0, 24)
        device = rng.choice(attacker_devices)
        t = 0.0
        for _k in range(rng.integers(5, 16)):
            m = rng.choice(online_small)
            rows.append((ts_at(day, hour, t), u["user_id"], round(float(rng.uniform(1, 99)), 2), m,
                         "gaming", "CARD", device, rng.choice(CITIES), 1, "CARD_TESTING"))
            t += rng.uniform(0.2, 2)
        for _k in range(rng.integers(0, 3)):
            cat = rng.choice(["electronics", "gift_cards"])
            rows.append((ts_at(day, hour, t), u["user_id"],
                         round(float(CATS[cat][0] * rng.uniform(2, 6)), 2), rng.choice(by_cat[cat]),
                         cat, "CARD", device, rng.choice(CITIES), 1, "CARD_TESTING"))
            t += rng.uniform(1, 5)

    # 3. Social-engineering P2P scam to mule accounts (victim's own device and city)
    for _ in range(650):
        u = users[rng.choice(user_idx)]
        day = rng.uniform(1, days - 0.5)
        hour = float(np.clip(rng.normal(u["peak_hour"], 3), 8, 22))
        amt = 1500 * u["amt_mult"] * rng.uniform(3, 15)
        rows.append((ts_at(day, hour), u["user_id"], round(float(amt), 2), rng.choice(mule_accounts),
                     "P2P", "UPI", rng.choice(u["devices"]), u["city"], 1, "MULE_P2P"))

    # 4. NOVEL: QR-swap fake merchants, only in the last 15 days
    for k in range(18):
        fake = f"M_NEW{k:03d}"
        day0 = rng.uniform(days - 15, days - 0.5)
        for _v in range(rng.integers(10, 25)):
            u = users[rng.choice(user_idx)]
            amt = CATS["utilities"][0] * u["amt_mult"] * rng.uniform(1.0, 3.0)
            rows.append((ts_at(day0, rng.uniform(9, 21)) + timedelta(hours=float(rng.uniform(0, 6))),
                         u["user_id"], round(float(amt), 2), fake, "utilities", "UPI",
                         rng.choice(u["devices"]), u["city"], 1, "QR_SWAP"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--merchants", type=int, default=1500)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    users, by_cat = build_world(rng, a.users, a.merchants)
    rows = gen_legit(rng, users, by_cat, a.days) + gen_fraud(rng, users, by_cat, a.days)
    df = pd.DataFrame(rows, columns=["timestamp", "user_id", "amount", "merchant_id", "category",
                                     "channel", "device_id", "city", "is_fraud", "fraud_type"])
    df = df[df.timestamp < START + timedelta(days=a.days)]
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RAW_TXNS, index=False)
    print(f"wrote {len(df):,} transactions -> {RAW_TXNS}")
    print(df.fraud_type.value_counts().to_string())
    print(f"fraud rate: {df.is_fraud.mean():.3%}")


if __name__ == "__main__":
    main()
