"""Dataset adapters: map public fraud datasets onto payTrace's transaction schema.

Canonical schema (one row per payment, sorted by time):
    txn_id, timestamp, user_id, amount, merchant_id, category, channel, device_id, city, is_fraud, fraud_type

Supported sources
-----------------
synthetic   data/transactions.csv or data/transactions_v2.csv (already canonical)
paysim      Kaggle "Synthetic Financial Datasets For Fraud Detection" (PaySim), file PS_*_log.csv.
            Licence: CC BY-SA 4.0. Simulated from real mobile-money logs (one African country).
sparkov     Kaggle "Credit Card Transactions Fraud Detection Dataset" (kartik2112/fraud-detection): fraudTrain.csv +
            fraudTest.csv. Licence: CC0. Simulated with the Sparkov generator (1,000 customers, 800 merchants).
ieee        Kaggle "IEEE-CIS Fraud Detection" (Vesta): train_transaction.csv + train_identity.csv.
            Licence: Kaggle competition rules (non-commercial research use). Real e-commerce data,
            heavily anonymised.

Download them yourself from Kaggle (login required) into data/raw/ and run:
    python -m src.adapters paysim data/raw/PS_20174392719_1491204439457_log.csv
    python -m src.adapters ieee   data/raw/ieee/train_transaction.csv --identity data/raw/ieee/train_identity.csv
    python -m src.adapters sparkov data/raw/sparkov/fraudTrain.csv data/raw/sparkov/fraudTest.csv

Known leakage traps handled here
--------------------------------
* PaySim balance columns (oldbalanceOrg/newbalanceOrig/oldbalanceDest/newbalanceDest) leak the label:
  fraud empties the origin account and destination balances are not updated for fraud. They are DROPPED.
  isFlaggedFraud is the simulator's own rule output and is also dropped.
* IEEE-CIS has no customer id. The proxy used by top competition solutions (card1 + addr1 + account-open day) is
  used; it is approximate. Only the labelled train file is usable (the competition's test labels are private).
  There is no merchant or payee id, so payee-graph features are weak on this dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = ["txn_id", "timestamp", "user_id", "amount", "merchant_id", "category", "channel",
             "device_id", "city", "is_fraud", "fraud_type"]
REQUIRED = ["timestamp", "user_id", "amount", "merchant_id"]
UNKNOWN = "UNKNOWN"


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a frame to the canonical schema: fill optional columns, fix types, drop bad rows, sort."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    out = df.copy()
    for col, default in [("category", "other"), ("channel", UNKNOWN), ("device_id", UNKNOWN),
                         ("city", UNKNOWN), ("is_fraud", 0), ("fraud_type", "")]:
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    bad = out["timestamp"].isna() | out["amount"].isna() | (out["amount"] < 0) | out["user_id"].isna()
    if bad.any():
        print(f"dropping {int(bad.sum())} invalid rows")
    out = out[~bad]
    out["is_fraud"] = out["is_fraud"].astype(int).clip(0, 1)
    out.loc[(out["fraud_type"] == "") & (out["is_fraud"] == 1), "fraud_type"] = "FRAUD"
    out.loc[out["fraud_type"] == "", "fraud_type"] = "LEGIT"
    for c in ["user_id", "merchant_id", "category", "channel", "device_id", "city"]:
        out[c] = out[c].astype(str)
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if "txn_id" not in out.columns:
        out["txn_id"] = [f"X{i:08d}" for i in range(len(out))]
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out[CANONICAL]


def from_paysim(path: str | Path, start: datetime = datetime(2026, 1, 1)) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rng = np.random.default_rng(0)
    # PaySim "step" = 1 simulated hour; spread events within the hour so ordering is not arbitrary
    ts = [start + timedelta(hours=int(s), seconds=int(x)) for s, x in zip(raw["step"], rng.integers(0, 3600, len(raw)))]
    df = pd.DataFrame({
        "txn_id": [f"PS{i:09d}" for i in range(len(raw))],
        "timestamp": ts,
        "user_id": raw["nameOrig"],
        "amount": raw["amount"],
        "merchant_id": raw["nameDest"],
        "category": raw["type"],                       # PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
        "channel": raw["type"],
        "device_id": UNKNOWN,                          # not in PaySim
        "city": UNKNOWN,
        "is_fraud": raw["isFraud"],
        "fraud_type": np.where(raw["isFraud"] == 1, "PAYSIM_" + raw["type"].astype(str), "LEGIT"),
    })
    return validate(df)                                # balance columns + isFlaggedFraud dropped on purpose


# Amounts are converted to rupees so every dataset shares one currency with the app. The rate is a fixed assumption
# (override with PAYTRACE_INR_PER_USD); the model's behavioural features are ratios, so the exact rate barely matters.
INR_PER_USD = float(__import__("os").getenv("PAYTRACE_INR_PER_USD", "88"))


def from_ieee(transactions: str | Path, identity: str | Path | None = None,
              start: datetime = datetime(2017, 12, 1)) -> pd.DataFrame:
    """IEEE-CIS (real Vesta card-not-present payments). No customer id is given; the standard proxy from the
    competition's top solutions is used: card1 + addr1 + the day the account was opened (day - D1)."""
    cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD", "card1", "card4",
            "addr1", "D1", "R_emaildomain"]
    raw = pd.read_csv(transactions, usecols=lambda c: c in cols)
    if identity:
        ident = pd.read_csv(identity, usecols=lambda c: c in ["TransactionID", "DeviceInfo", "DeviceType"])
        raw = raw.merge(ident, on="TransactionID", how="left")
    day = raw["TransactionDT"] // 86400
    d1 = raw["D1"] if "D1" in raw else pd.Series(np.nan, index=raw.index)      # optional in partial extracts
    opened = (day - d1).fillna(-1).astype(int).astype(str)
    uid = "IE_" + raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str) + "_" + opened
    dev = raw["DeviceInfo"] if "DeviceInfo" in raw else pd.Series(np.nan, index=raw.index)
    df = pd.DataFrame({
        "txn_id": "IE" + raw["TransactionID"].astype(str),
        "timestamp": pd.Timestamp(start) + pd.to_timedelta(raw["TransactionDT"], unit="s"),
        "user_id": uid,
        "amount": (raw["TransactionAmt"] * INR_PER_USD).round(2),
        # no merchant id exists; product code + recipient email domain is the closest public proxy (weak)
        "merchant_id": "IE_" + raw["ProductCD"].astype(str) + "_" + raw["R_emaildomain"].fillna("na").astype(str),
        "category": "ie_" + raw["ProductCD"].astype(str).str.lower(),
        "channel": "CARD_ONLINE",
        "device_id": ("IE_" + dev.astype(str)).where(dev.notna(), UNKNOWN),
        "city": "IE_" + raw["addr1"].fillna(-1).astype(int).astype(str),
        "is_fraud": raw["isFraud"],
        "fraud_type": np.where(raw["isFraud"] == 1, "IEEE_FRAUD", "LEGIT"),
    })
    return validate(df)


def from_sparkov(*paths: str | Path) -> pd.DataFrame:
    """Sparkov 'Credit Card Transactions Fraud Detection' (Kaggle kartik2112/fraud-detection, CC0): simulated card
    payments of 1,000 US customers at 800 merchants, Jan 2019 - Dec 2020. Personal fields (name, street, job, dob)
    are dropped. No device data. 'city' is the cardholder's home city, so it never changes per customer."""
    use = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "city", "state", "is_fraud", "trans_num"]
    raw = pd.concat([pd.read_csv(p, usecols=use) for p in paths], ignore_index=True)
    cust = {c: f"SP{i:04d}" for i, c in enumerate(sorted(raw["cc_num"].unique()))}
    df = pd.DataFrame({
        "txn_id": "SP" + raw["trans_num"].astype(str).str[:16],
        "timestamp": pd.to_datetime(raw["trans_date_trans_time"]),
        "user_id": raw["cc_num"].map(cust),
        "amount": (raw["amt"] * INR_PER_USD).round(2),
        "merchant_id": raw["merchant"].str.removeprefix("fraud_").str.strip(),   # every name has this prefix; not a label
        "category": raw["category"],
        "channel": np.where(raw["category"].str.endswith("_net"), "CARD_ONLINE", "CARD_POS"),
        "device_id": UNKNOWN,
        "city": raw["city"] + ", " + raw["state"],
        "is_fraud": raw["is_fraud"],
        "fraud_type": np.where(raw["is_fraud"] == 1, "SPARKOV_FRAUD", "LEGIT"),
    })
    return validate(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["paysim", "ieee", "sparkov", "csv"])
    ap.add_argument("path", nargs="+")
    ap.add_argument("--identity")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = {"paysim": lambda: from_paysim(a.path[0]), "ieee": lambda: from_ieee(a.path[0], a.identity),
          "sparkov": lambda: from_sparkov(*a.path), "csv": lambda: validate(pd.read_csv(a.path[0]))}[a.source]()
    out = Path(a.out or f"data/transactions_{a.source}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} rows -> {out}  fraud rate {df.is_fraud.mean():.3%}")


if __name__ == "__main__":
    main()
