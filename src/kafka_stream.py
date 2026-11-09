"""Kafka / Redpanda mode (production-style transport).

  docker compose up -d                       # starts Redpanda on localhost:9092
  pip install confluent-kafka
  python -m src.kafka_stream produce --rate 200
  python -m src.kafka_stream consume         # scores, writes stream.db, publishes alerts

Topics: `transactions` (in) -> scorer -> `fraud-alerts` (out, REVIEW/BLOCK only).
The scorer, feature engine and SQLite sink are the same ones used by src/stream.py, which
runs the identical pipeline in-process when no broker is available.
Note: this module needs a running broker; the in-process stream is the tested default.
"""
import argparse
import json
import os
import time

from src.config import STREAM_DB
from src.copilot import template_note
from src.scorer import FraudScorer
from src.stream import TXN_FIELDS, init_db, load_split

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")


def produce(rate):
    from confluent_kafka import Producer
    scorer = FraudScorer()                      # only used to locate the test window
    _, test = load_split(scorer)
    p = Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 5})
    t0 = time.time()
    for i, row in enumerate(test.to_dict("records")):
        p.produce("transactions", key=row["user_id"], value=json.dumps(row, default=str))
        if i % 1000 == 0:
            p.poll(0)
        if rate:
            lag = (i + 1) / rate - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
    p.flush()
    print(f"produced {len(test):,} transactions")


def consume():
    from confluent_kafka import Consumer, Producer
    scorer = FraudScorer()
    hist, _ = load_split(scorer)
    scorer.warmup(hist[TXN_FIELDS].to_dict("records"))
    con = init_db(STREAM_DB)
    c = Consumer({"bootstrap.servers": BOOTSTRAP, "group.id": "paytrace-scorer",
                  "auto.offset.reset": "earliest", "enable.auto.commit": True})
    c.subscribe(["transactions"])
    out = Producer({"bootstrap.servers": BOOTSTRAP})
    seq = 0
    print("scoring... Ctrl+C to stop")
    try:
        while True:
            msg = c.poll(0.5)
            if msg is None or msg.error():
                continue
            row = json.loads(msg.value())
            txn = {k: row[k] for k in TXN_FIELDS}
            res = scorer.score(txn)
            note = template_note(txn, res) if res["decision"] != "ALLOW" else ""
            con.execute(f"INSERT INTO events VALUES ({','.join('?' * 18)})",
                        (seq, txn["txn_id"], txn["timestamp"], txn["user_id"], txn["amount"],
                         txn["merchant_id"], txn["category"], txn["channel"], txn["city"],
                         res["probability"], res["decision"], int(res["novelty"]), res["latency_ms"],
                         json.dumps(res["reasons"]), note, int(row.get("is_fraud", -1)),
                         row.get("fraud_type", ""), time.time()))
            if res["decision"] != "ALLOW":
                out.produce("fraud-alerts", key=txn["user_id"],
                            value=json.dumps({**txn, "decision": res["decision"],
                                              "probability": res["probability"],
                                              "reasons": res["reasons"], "note": note}, default=str))
                out.poll(0)
            seq += 1
            if seq % 50 == 0:
                con.commit()
    except KeyboardInterrupt:
        pass
    finally:
        con.commit(); c.close(); out.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["produce", "consume"])
    ap.add_argument("--rate", type=float, default=200)
    a = ap.parse_args()
    produce(a.rate) if a.mode == "produce" else consume()
