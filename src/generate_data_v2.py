"""Synthetic data v2: harder, more realistic, more fraud scenarios.

Fixes the v1 shortcuts found in AUDIT.md:
  * fraud now occurs in every merchant category (v1: never in fuel/food/grocery/travel)
  * most attacker devices are fresh per victim (v1: 60 reusable devices never touched by real users)
  * mule accounts are ordinary customers who also receive genuine payments (v1: mules received none)
  * customers join throughout the period, so the test period contains never-seen customers

Every fraud pattern is spread across the whole 60 days, so any one pattern can be held out of training
for the leave-one-pattern-out experiment. Patterns:

  ATO                account takeover: new device/city, burst of purchases (any category)
  CARD_TESTING       tiny payments then cash-out
  MULE_P2P           victim's own device sends a large P2P to a mule customer
  QR_SWAP            fake merchant paid by many customers within hours
  MERCHANT_COLLUSION a colluding merchant runs round-amount charges on compromised cards
  RAPID_MOVEMENT     stolen money hops mule -> mule -> mule within minutes
  DEVICE_RING        a ring of new synthetic accounts sharing a few devices, paying common payees

Usage:  python -m src.generate_data_v2            -> data/transactions_v2.csv
"""
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.generate_data import CATS, CAT_NAMES, CITIES, START, _channel, build_world

OUT = DATA_DIR / "transactions_v2.csv"
ALL_SPEND = [c for c in CAT_NAMES if c != "P2P"]


def ts_at(day, hour, minutes=0.0):
    return START + timedelta(days=float(day), hours=float(hour), minutes=float(minutes))


def gen_legit(rng, users, by_cat, days, start_day, mule_ids):
    rows = []
    for u in users:
        s = start_day[u["user_id"]]
        n = rng.poisson(u["rate"] * (days - s))
        if n == 0:
            continue
        day = rng.uniform(s, days, n)
        hour = np.mod(rng.normal(u["peak_hour"], 3.0, n), 24.0)
        night = rng.random(n) < 0.03
        hour[night] = rng.uniform(0, 5, night.sum())
        cats = rng.choice(CAT_NAMES, n, p=u["cat_w"])
        for i in range(n):
            cat = cats[i]
            med, sig, _ = CATS[cat]
            amt = med * u["amt_mult"] * np.exp(rng.normal(0, sig))
            if rng.random() < 0.01:
                amt *= rng.uniform(4, 12)
            if rng.random() < 0.04:                       # genuine round-amount payments exist too
                amt = float(np.round(amt, -2) or 100)
            if cat == "P2P":
                r = rng.random()
                if r < 0.85:
                    payee = rng.choice(u["contacts"])
                elif r < 0.90:                            # genuine payments to customers who are mules
                    payee = rng.choice(mule_ids)
                else:
                    payee = f"P2P_U{rng.integers(0, len(users)):05d}"
            else:
                favs = u["fav_merchants"][cat]
                payee = rng.choice(favs) if rng.random() < 0.75 else rng.choice(by_cat[cat])
            device = f"D_NEW_{rng.integers(0, 10**9)}" if rng.random() < 0.015 else rng.choice(u["devices"])
            city = u["city"] if rng.random() > 0.025 else rng.choice(CITIES)
            rows.append((ts_at(day[i], 0) + timedelta(hours=float(hour[i])), u["user_id"], round(float(amt), 2),
                         payee, cat, _channel(rng, cat), device, city, 0, "LEGIT"))
    return rows


def gen_fraud(rng, users, by_cat, days, start_day, mule_users):
    rows = []
    idx = np.arange(len(users))
    pool = [f"D_ATK_{k:03d}" for k in range(40)]
    mule_ids = [f"P2P_{m['user_id']}" for m in mule_users]

    def victim(day):
        for _ in range(50):
            u = users[rng.choice(idx)]
            if start_day[u["user_id"]] < day - 1:
                return u
        return users[0]

    def add(t, u, amt, payee, cat, ch, dev, city, kind):
        rows.append((t, u if isinstance(u, str) else u["user_id"], round(float(amt), 2), payee, cat, ch, dev, city, 1, kind))

    # ATO: any category, attacker device usually fresh
    for _ in range(330):
        day = rng.uniform(2, days - 0.5); u = victim(day)
        hour = rng.uniform(0, 24) if rng.random() < 0.7 else rng.uniform(0, 5)
        dev = rng.choice(pool) if rng.random() < 0.3 else f"D_ATK_X{rng.integers(10**9)}"
        city = rng.choice([c for c in CITIES if c != u["city"]]) if rng.random() < 0.7 else u["city"]
        t = 0.0
        for _k in range(rng.integers(1, 6)):
            cat = rng.choice(ALL_SPEND + ["P2P"])
            amt = CATS[cat][0] * u["amt_mult"] * rng.uniform(1.2, 5)
            payee = rng.choice(mule_ids) if cat == "P2P" else rng.choice(by_cat[cat])
            add(ts_at(day, hour, t), u, amt, payee, cat, _channel(rng, cat), dev, city, "ATO")
            t += rng.uniform(1, 20)

    # Card testing: tiny payments in several online categories
    online = np.concatenate([by_cat[c] for c in ("gaming", "gift_cards", "food_delivery", "fashion", "utilities")])
    for _ in range(100):
        day = rng.uniform(2, days - 0.5); u = victim(day); hour = rng.uniform(0, 24)
        dev = rng.choice(pool) if rng.random() < 0.4 else f"D_ATK_X{rng.integers(10**9)}"
        t = 0.0
        for _k in range(rng.integers(4, 12)):
            add(ts_at(day, hour, t), u, rng.uniform(1, 149), rng.choice(online), rng.choice(["gaming", "food_delivery", "fashion"]),
                "CARD", dev, rng.choice(CITIES), "CARD_TESTING")
            t += rng.uniform(0.3, 4)
        for _k in range(rng.integers(0, 3)):
            cat = rng.choice(ALL_SPEND)
            add(ts_at(day, hour, t), u, CATS[cat][0] * rng.uniform(1.5, 5), rng.choice(by_cat[cat]), cat, "CARD", dev,
                rng.choice(CITIES), "CARD_TESTING")
            t += rng.uniform(1, 6)

    # Mule P2P scam: victim's own device, payee is a mule customer
    for _ in range(600):
        day = rng.uniform(2, days - 0.5); u = victim(day)
        add(ts_at(day, float(np.clip(rng.normal(u["peak_hour"], 3), 8, 22))), u, 1500 * u["amt_mult"] * rng.uniform(2, 10),
            rng.choice(mule_ids), "P2P", "UPI", rng.choice(u["devices"]), u["city"], "MULE_P2P")

    # QR swap: fake merchant collects from many customers within hours (whole period)
    for k in range(40):
        fake = f"M{90000 + k}"; day0 = rng.uniform(2, days - 0.5); cat = rng.choice(["utilities", "food_delivery", "grocery"])
        for _v in range(rng.integers(6, 18)):
            u = victim(day0)
            add(ts_at(day0, rng.uniform(9, 21)) + timedelta(hours=float(rng.uniform(0, 6))), u,
                CATS[cat][0] * u["amt_mult"] * rng.uniform(1.0, 3.0), fake, cat, "UPI", rng.choice(u["devices"]), u["city"], "QR_SWAP")

    # Merchant collusion: an established-looking merchant charges compromised cards round amounts, at night
    for k in range(25):
        cat = rng.choice(["electronics", "fashion", "travel"]); m = rng.choice(by_cat[cat]); day0 = rng.uniform(2, days - 2)
        for _v in range(rng.integers(5, 14)):
            u = victim(day0)
            add(ts_at(day0 + rng.uniform(0, 1.5), rng.uniform(0, 6)), u, float(rng.choice([2000, 3000, 5000, 7500, 10000])),
                m, cat, "CARD", rng.choice(u["devices"]), u["city"], "MERCHANT_COLLUSION")

    # Rapid money movement: victim -> mule A -> mule B -> mule C within minutes (mules use their own devices)
    for _ in range(140):
        day = rng.uniform(2, days - 0.5); u = victim(day); hour = rng.uniform(8, 23)
        amt = 1500 * u["amt_mult"] * rng.uniform(3, 12)
        chain = rng.choice(len(mule_users), size=rng.integers(2, 4), replace=False)
        t = 0.0
        add(ts_at(day, hour, t), u, amt, f"P2P_{mule_users[chain[0]]['user_id']}", "P2P", "UPI", rng.choice(u["devices"]), u["city"], "RAPID_MOVEMENT")
        for a, b in zip(chain[:-1], chain[1:]):
            t += rng.uniform(2, 25); amt *= rng.uniform(0.9, 0.99)
            ma = mule_users[a]
            add(ts_at(day, hour, t), ma, amt, f"P2P_{mule_users[b]['user_id']}", "P2P", "UPI", rng.choice(ma["devices"]), ma["city"], "RAPID_MOVEMENT")
        last = mule_users[chain[-1]]
        t += rng.uniform(2, 30)
        add(ts_at(day, hour, t), last, amt * 0.95, rng.choice(by_cat["crypto"]), "crypto", "NETBANKING", rng.choice(last["devices"]), last["city"], "RAPID_MOVEMENT")

    # Device ring: new synthetic accounts sharing 2-3 devices, paying common payees
    for k in range(22):
        day0 = rng.uniform(2, days - 1.5); devs = [f"D_RING{k}_{j}" for j in range(rng.integers(2, 4))]
        payees = [f"P2P_RINGX{k}_{j}" for j in range(2)] + list(rng.choice(by_cat["gift_cards"], 2))
        for a in range(rng.integers(8, 18)):
            acct = f"U_RING{k:02d}_{a:02d}"; city = rng.choice(CITIES)
            for _t in range(rng.integers(1, 4)):
                p = rng.choice(payees); cat = "P2P" if str(p).startswith("P2P") else "gift_cards"
                add(ts_at(day0 + rng.uniform(0, 1.2), rng.uniform(8, 23)), acct, rng.uniform(300, 4000), p, cat,
                    "UPI" if cat == "P2P" else "CARD", rng.choice(devs), city, "DEVICE_RING")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    users, by_cat = build_world(rng, a.users, 1500)
    # 80% of customers exist from day 0; 20% join later (so the test period has unseen customers)
    start_day = {u["user_id"]: (0.0 if rng.random() < 0.8 else float(rng.uniform(5, a.days - 3))) for u in users}
    mule_users = [users[i] for i in rng.choice(len(users), 60, replace=False)]
    mule_ids = [f"P2P_{m['user_id']}" for m in mule_users]
    for m in mule_users:
        start_day[m["user_id"]] = 0.0
    rows = gen_legit(rng, users, by_cat, a.days, start_day, mule_ids) + gen_fraud(rng, users, by_cat, a.days, start_day, mule_users)
    df = pd.DataFrame(rows, columns=["timestamp", "user_id", "amount", "merchant_id", "category", "channel",
                                     "device_id", "city", "is_fraud", "fraud_type"])
    df = df[df.timestamp < START + timedelta(days=a.days)].sort_values("timestamp", kind="stable").reset_index(drop=True)
    df.insert(0, "txn_id", [f"V{i:07d}" for i in range(len(df))])
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} transactions -> {a.out}  fraud rate {df.is_fraud.mean():.3%}")
    print(df.fraud_type.value_counts().to_string())


if __name__ == "__main__":
    main()
