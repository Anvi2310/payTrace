"""payTrace 2.0 streaming pipeline:  producer -> bus -> processor (features + models + decision) -> event store.

    python -m src.stream_v2                       # local bus, 300 txn/s replay of the unseen test period
    python -m src.stream_v2 --rate 0              # as fast as possible
    python -m src.stream_v2 --dup-rate 0.02 --jitter 20   # inject 2% duplicates and up to 20s out-of-order delivery
    python -m src.stream_v2 --bus kafka           # Kafka/Redpanda (needs a broker; see docker-compose.yml)

Guarantees implemented and tested (tests/test_stream.py):
  * each payment is processed individually, with per-customer / device / payee state
  * duplicates (same txn_id) are dropped -> idempotent, safe with at-least-once delivery
  * out-of-order events: by default every event is scored immediately (lowest latency) and events older than
    the newest event time already seen are marked late=1; with `--lateness N` events are buffered N seconds of
    event time and re-ordered before scoring (exact state, at the cost of N seconds of delay)
  * invalid events or scoring errors go to a dead-letter table; the stream keeps running
  * structured JSON logs (logs/stream.jsonl), throughput and latency metrics, resumable status

The data source is a REPLAY of the dataset's test period (synthetic or public), not live bank traffic.
Ground-truth labels are revealed to the feature engine 3 days (event time) after each payment to simulate
chargebacks; the dashboard's live precision/recall use those labels and are marked as replay metrics.
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import queue
import threading
import time
from collections import OrderedDict, deque
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import ROOT
from src.scorer_v2 import FraudScorerV2, InvalidTransaction
from src.store import connect, set_status
from src.train_v2 import dataset_path, slices

TXN_FIELDS = ["txn_id", "timestamp", "user_id", "amount", "merchant_id", "category", "channel", "device_id", "city"]
LOG_DIR = ROOT / "logs"
STOP = object()


def json_logger(name="paytrace.stream"):
    LOG_DIR.mkdir(exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.FileHandler(LOG_DIR / "stream.jsonl")
        h.setFormatter(logging.Formatter('{"t":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}'))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


class LocalBus:
    """In-process message bus with the same put/get contract as a Kafka topic consumer."""
    def __init__(self, maxsize=10000):
        self.q = queue.Queue(maxsize)

    def put(self, msg):
        self.q.put(msg)

    def get(self, timeout=1.0):
        return self.q.get(timeout=timeout)

    def depth(self):
        return self.q.qsize()


class Deduper:
    def __init__(self, capacity=500_000):
        self.seen, self.cap = OrderedDict(), capacity

    def is_dup(self, key) -> bool:
        if key in self.seen:
            return True
        self.seen[key] = None
        if len(self.seen) > self.cap:
            self.seen.popitem(last=False)
        return False


class Reorderer:
    """Buffers events and releases them in event-time order once they are `lateness` seconds behind the newest."""
    def __init__(self, lateness_s: float):
        self.lateness, self.heap, self.max_ts, self.last_out, self.n = lateness_s, [], -1e18, -1e18, 0

    def push(self, ts: float, msg):
        late = ts < self.last_out
        if late:                                     # behind the watermark: release immediately, flagged late
            return [(msg, True)]
        self.n += 1
        heapq.heappush(self.heap, (ts, self.n, msg))
        self.max_ts = max(self.max_ts, ts)
        return self._drain(self.max_ts - self.lateness)

    def _drain(self, upto):
        out = []
        while self.heap and self.heap[0][0] <= upto:
            ts, _, msg = heapq.heappop(self.heap)
            self.last_out = ts
            out.append((msg, False))
        return out

    def flush(self):
        return self._drain(float("inf"))


class Processor:
    def __init__(self, scorer: FraudScorerV2, con, model_version: str, lateness_s: float = 0.0, log=None):
        self.s, self.con, self.version = scorer, con, model_version
        self.dedupe, self.reorder = Deduper(), Reorderer(lateness_s)
        self.log = log or json_logger()
        self.n = self.dups = self.late = self.fail = 0
        self.lat, self.e2e = deque(maxlen=5000), deque(maxlen=5000)
        self.batch = []
        self.t_start = time.time()
        import psutil
        self.proc = psutil.Process(os.getpid())       # one handle, so cpu_percent measures since the last sample
        self.proc.cpu_percent(interval=None)

    def handle(self, msg: dict):
        tid = msg.get("txn_id")
        if tid is None or self.dedupe.is_dup(tid):
            self.dups += tid is not None
            return
        try:
            ts = datetime.fromisoformat(str(msg["timestamp"])).timestamp()
        except Exception as e:
            return self._dead(msg, f"bad timestamp: {e}")
        for m, late in self.reorder.push(ts, msg):
            self._score(m, late)

    def flush(self):
        for m, late in self.reorder.flush():
            self._score(m, late)
        self._write()

    def _dead(self, msg, err):
        self.fail += 1
        self.con.execute("INSERT INTO deadletter VALUES (?,?,?,?)", (time.time(), str(msg.get("txn_id")), err,
                                                                    json.dumps(msg, default=str)[:2000]))
        self.log.warning(json.dumps({"event": "deadletter", "txn_id": msg.get("txn_id"), "error": err}))

    def _score(self, msg, late):
        try:
            lab = msg.get("_label")
            r = self.s.score({k: msg.get(k) for k in TXN_FIELDS}, label_for_delay=None if lab is None else int(lab))
        except InvalidTransaction as e:
            return self._dead(msg, f"invalid: {e}")
        except Exception as e:                        # model failure must not stop the stream
            return self._dead(msg, f"scoring error: {type(e).__name__}: {e}")
        self.n += 1; self.late += int(late)
        e2e = (time.time() - msg.get("_ingest", time.time())) * 1000
        self.lat.append(r["latency_ms"]); self.e2e.append(e2e)
        feats = json.dumps({k: round(float(v), 4) for k, v in r["features"].items()}) if r["decision"] != "ALLOW" else None
        self.batch.append((None, msg["txn_id"], str(msg["timestamp"]), msg["user_id"], float(msg["amount"]), msg["merchant_id"],
                           msg.get("category"), msg.get("channel"), msg.get("device_id"), msg.get("city"), r["risk"],
                           r["supervised_prob"], r["decision"], ",".join(r["tripwires"]), r["if_score"], r["ae_score"],
                           r["graph_score"], json.dumps(r["reasons"]), json.dumps(r["risk_breakdown"]), feats,
                           r["latency_ms"], e2e, int(late), msg.get("_label", -1), msg.get("_type", ""), self.version, time.time()))
        if r["decision"] != "ALLOW":
            self.log.info(json.dumps({"event": "alert", "txn_id": msg["txn_id"], "decision": r["decision"],
                                      "risk": round(r["risk"], 4), "latency_ms": round(r["latency_ms"], 3)}))
        if len(self.batch) >= 50:
            self._write()

    def _write(self):
        if self.batch:
            self.con.executemany(f"INSERT OR IGNORE INTO events VALUES ({','.join('?' * 27)})", self.batch)
            self.con.commit()
            self.batch.clear()

    def metrics(self, depth=0):
        p = self.proc
        lat = np.array(self.lat) if self.lat else np.zeros(1); e2e = np.array(self.e2e) if self.e2e else np.zeros(1)
        row = (time.time(), self.n, self.dups, self.late, self.fail, self.n / max(time.time() - self.t_start, 1e-6),
               *np.percentile(lat, [50, 95, 99]).tolist(), *np.percentile(e2e, [50, 99]).tolist(),
               p.cpu_percent(interval=None), p.memory_info().rss / 2**20, depth)
        self.con.execute(f"INSERT INTO metrics VALUES ({','.join('?' * 14)})", row)
        self.con.commit()
        return row


def replay_events(dataset="v2", limit=0, dup_rate=0.0, jitter_s=0.0, seed=0):
    df = pd.read_csv(dataset_path(dataset), parse_dates=["timestamp"])
    s = slices(df)
    hist, test = df[~s["test"]], df[s["test"]]
    if limit:
        test = test.head(limit)
    rng = np.random.default_rng(seed)
    ev = test.copy()
    ev["timestamp"] = ev["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    recs = ev[TXN_FIELDS + ["is_fraud", "fraud_type"]].to_dict("records")
    order = np.arange(len(recs), dtype=float)
    if jitter_s:                                     # deliver some events late (out of order)
        t = (pd.to_datetime(ev["timestamp"]) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
        order = t + rng.uniform(0, jitter_s, len(t)) * (rng.random(len(t)) < 0.2)
    out = []
    for i in np.argsort(order, kind="stable"):
        r = dict(recs[i]); r["_label"] = int(r.pop("is_fraud")); r["_type"] = r.pop("fraud_type")
        out.append(r)
        if dup_rate and rng.random() < dup_rate:
            out.append(dict(r))                      # at-least-once delivery: same message twice
    return hist, out


def run(dataset="v2", rate=300.0, limit=0, dup_rate=0.0, jitter=0.0, lateness=0.0, bus_kind="local"):
    log = json_logger()
    con = connect()
    con.execute("DELETE FROM events"); con.execute("DELETE FROM metrics"); con.execute("DELETE FROM deadletter"); con.commit()
    set_status(con, state="loading", dataset=dataset)
    scorer = FraudScorerV2()
    version = scorer.meta.get("version", scorer.meta.get("trained_at", "v2"))
    hist, events = replay_events(dataset, limit, dup_rate, jitter)
    set_status(con, state="warming_up", total=len(events))
    t = time.time()
    scorer.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist["is_fraud"].to_numpy())
    log.info(json.dumps({"event": "warmup_done", "history": len(hist), "seconds": round(time.time() - t, 2)}))
    proc = Processor(scorer, con, version, lateness, log)
    set_status(con, state="streaming", model_version=version, started=time.time())

    if bus_kind == "kafka":
        return _run_kafka(proc, events, rate, con)

    bus = LocalBus()

    def producer():
        t0 = time.time()
        for i, e in enumerate(events):
            e["_ingest"] = time.time()
            bus.put(e)
            if rate:
                lag = (i + 1) / rate - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
        bus.put(STOP)

    th = threading.Thread(target=producer, daemon=True); th.start()
    last_metrics = time.time()
    while True:
        try:
            msg = bus.get(timeout=2.0)
        except queue.Empty:
            continue
        if msg is STOP:
            break
        proc.handle(msg)
        if time.time() - last_metrics > 1.0:
            proc._write(); proc.metrics(bus.depth()); last_metrics = time.time()
            set_status(con, processed=proc.n)
    proc.flush(); row = proc.metrics(0)
    set_status(con, state="done", processed=proc.n, duplicates=proc.dups, late=proc.late, failures=proc.fail)
    summary = {"processed": proc.n, "duplicates_dropped": proc.dups, "late": proc.late, "failures": proc.fail,
               "throughput_per_s": round(row[5], 1), "p50_ms": round(row[6], 3), "p99_ms": round(row[8], 3),
               "e2e_p50_ms": round(row[9], 3), "e2e_p99_ms": round(row[10], 3)}
    log.info(json.dumps({"event": "stream_done", **summary}))
    print(json.dumps(summary))
    return summary


def _run_kafka(proc, events, rate, con):            # pragma: no cover - needs a running broker
    from confluent_kafka import Consumer, Producer
    boot = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    prod = Producer({"bootstrap.servers": boot, "enable.idempotence": True})
    for e in events:
        e["_ingest"] = time.time()
        prod.produce("transactions", key=e["user_id"], value=json.dumps(e, default=str))
    prod.flush()
    c = Consumer({"bootstrap.servers": boot, "group.id": "paytrace-v2", "auto.offset.reset": "earliest",
                  "enable.auto.commit": False})
    c.subscribe(["transactions"])
    idle = 0
    while idle < 10:
        m = c.poll(1.0)
        if m is None or m.error():
            idle += 1; continue
        idle = 0
        proc.handle(json.loads(m.value())); proc._write()
        c.commit(m, asynchronous=False)              # commit AFTER the write: at-least-once + dedupe = effectively once
    proc.flush(); c.close()
    return proc.metrics(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--rate", type=float, default=300)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dup-rate", type=float, default=0.0)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--lateness", type=float, default=0.0,
                    help="seconds of event time to buffer for re-ordering (0 = score immediately, flag stragglers late)")
    ap.add_argument("--bus", default="local", choices=["local", "kafka"])
    a = ap.parse_args()
    run(a.dataset, a.rate, a.limit, a.dup_rate, a.jitter, a.lateness, a.bus)
