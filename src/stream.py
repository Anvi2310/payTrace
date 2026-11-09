"""Stream simulator: replays the unseen test period through the real-time scorer, transaction
by transaction, and writes every decision to SQLite for the live dashboard.

  python -m src.stream                 # ~150 txns/sec, good for a live demo
  python -m src.stream --rate 0        # as fast as possible
  python -m src.stream --limit 20000

For a real message bus, see src/kafka_stream.py (same scorer, Kafka/Redpanda in and out).
"""
import argparse
import json
import sqlite3
import time
from collections import deque

import pandas as pd

from src.config import FEATURE_COLUMNS, RAW_TXNS, STREAM_DB
from src.copilot import template_note
from src.drift import MONITORED, drift_values, psi
from src.scorer import FraudScorer

TXN_FIELDS = ["txn_id", "timestamp", "user_id", "amount", "merchant_id", "category",
              "channel", "device_id", "city"]


def load_split(scorer):
    df = pd.read_csv(RAW_TXNS)
    start = scorer.meta["test_start"]
    hist = df[df.timestamp < start]
    test = df[df.timestamp >= start]
    return hist, test


def init_db(path=STREAM_DB):
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE events (
        seq INTEGER PRIMARY KEY, txn_id TEXT, ts TEXT, user_id TEXT, amount REAL,
        merchant_id TEXT, category TEXT, channel TEXT, city TEXT, probability REAL,
        decision TEXT, novelty INTEGER, latency_ms REAL, reasons TEXT, note TEXT,
        is_fraud INTEGER, fraud_type TEXT, wall_time REAL)""")
    con.execute("CREATE TABLE drift (seq INTEGER, ts TEXT, feature TEXT, psi REAL)")
    con.execute("CREATE TABLE status (k TEXT PRIMARY KEY, v TEXT)")
    con.commit()
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=150, help="txns/sec, 0 = unthrottled")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drift-every", type=int, default=2000)
    a = ap.parse_args()

    scorer = FraudScorer()
    hist, test = load_split(scorer)
    con = init_db()
    con.execute("INSERT OR REPLACE INTO status VALUES ('state','warming_up')"); con.commit()
    t = time.time()
    scorer.warmup(hist[TXN_FIELDS].to_dict("records"))
    print(f"warmed feature state with {len(hist):,} historical txns in {time.time() - t:.1f}s")
    if a.limit:
        test = test.head(a.limit)
    con.execute("INSERT OR REPLACE INTO status VALUES ('state','streaming')")
    con.execute("INSERT OR REPLACE INTO status VALUES ('total',?)", (str(len(test)),)); con.commit()

    window = deque()                      # (epoch seconds, monitored values) for the last 24h
    batch = []
    t_start = time.time()
    for i, row in enumerate(test.to_dict("records")):
        txn = {k: row[k] for k in TXN_FIELDS}
        res = scorer.score(txn)
        now = pd.Timestamp(txn["timestamp"]).timestamp()
        window.append((now, drift_values(res["features"])))
        while now - window[0][0] > 86400:
            window.popleft()
        note = template_note(txn, res) if res["decision"] != "ALLOW" else ""
        batch.append((i, txn["txn_id"], txn["timestamp"], txn["user_id"], txn["amount"],
                      txn["merchant_id"], txn["category"], txn["channel"], txn["city"],
                      res["probability"], res["decision"], int(res["novelty"]), res["latency_ms"],
                      json.dumps(res["reasons"]), note, int(row["is_fraud"]), row["fraud_type"],
                      time.time()))
        if (i + 1) % a.drift_every == 0 and now - window[0][0] > 0.95 * 86400:
            cols = {c: [w[1][c] for w in window] for c in MONITORED}
            for feat, v in psi(scorer.meta["psi_reference"], cols).items():
                con.execute("INSERT INTO drift VALUES (?,?,?,?)", (i, txn["timestamp"], feat, v))
        if len(batch) >= 25:
            con.executemany(f"INSERT INTO events VALUES ({','.join('?' * 18)})", batch)
            con.commit(); batch.clear()
        if a.rate:
            lag = (i + 1) / a.rate - (time.time() - t_start)
            if lag > 0:
                time.sleep(lag)
    if batch:
        con.executemany(f"INSERT INTO events VALUES ({','.join('?' * 18)})", batch)
    con.execute("INSERT OR REPLACE INTO status VALUES ('state','done')"); con.commit()
    print(f"streamed {len(test):,} txns in {time.time() - t_start:.1f}s -> {STREAM_DB}")


if __name__ == "__main__":
    main()
