"""Event store (SQLite, WAL mode) shared by the stream processor, dashboard, investigation and feedback.

PostgreSQL is the production target; no PostgreSQL server is available in the build environment, so the schema
below is plain SQL that ports directly (TEXT/REAL/INTEGER, no SQLite-only features apart from WAL).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from src.config import ROOT

_LEGACY_DB = ROOT / ("fraud" + "shield.db")          # database file name used before the rename
DB = Path(os.getenv("PAYTRACE_DB") or (ROOT / "paytrace.db" if (ROOT / "paytrace.db").exists() or not _LEGACY_DB.exists()
                                       else _LEGACY_DB))
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  seq INTEGER PRIMARY KEY, txn_id TEXT UNIQUE, ts TEXT, user_id TEXT, amount REAL, merchant_id TEXT, category TEXT,
  channel TEXT, device_id TEXT, city TEXT, risk REAL, supervised_prob REAL, decision TEXT, tripwires TEXT,
  if_score REAL, ae_score REAL, graph_score REAL, reasons TEXT, breakdown TEXT, features TEXT,
  latency_ms REAL, e2e_ms REAL, late INTEGER, is_fraud INTEGER, fraud_type TEXT, model_version TEXT, wall_time REAL);
CREATE INDEX IF NOT EXISTS ev_user ON events(user_id);
CREATE INDEX IF NOT EXISTS ev_merchant ON events(merchant_id);
CREATE INDEX IF NOT EXISTS ev_device ON events(device_id);
CREATE INDEX IF NOT EXISTS ev_decision ON events(decision);
CREATE TABLE IF NOT EXISTS deadletter (wall_time REAL, txn_id TEXT, error TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS metrics (wall_time REAL, processed INTEGER, duplicates INTEGER, late INTEGER, failures INTEGER,
  throughput REAL, p50_ms REAL, p95_ms REAL, p99_ms REAL, e2e_p50_ms REAL, e2e_p99_ms REAL, cpu_pct REAL, rss_mb REAL,
  queue_depth INTEGER);
CREATE TABLE IF NOT EXISTS status (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, wall_time REAL, txn_id TEXT, analyst TEXT, role TEXT,
  label INTEGER, note TEXT, model_decision TEXT, model_risk REAL, status TEXT, UNIQUE(txn_id, analyst));
CREATE TABLE IF NOT EXISTS audit_log (wall_time REAL, actor TEXT, role TEXT, action TEXT, target TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS model_registry (version TEXT PRIMARY KEY, created REAL, kind TEXT, role TEXT,
  trained_on TEXT, metrics TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, role TEXT, display_name TEXT, customer_id TEXT,
  salt TEXT, password_hash TEXT, created REAL, demo INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS payments (id INTEGER PRIMARY KEY, txn_id TEXT UNIQUE, ts TEXT, user_id TEXT, amount REAL,
  payee TEXT, category TEXT, channel TEXT, device_id TEXT, city TEXT, decision TEXT, risk REAL, status TEXT,
  customer_message TEXT, message_source TEXT, otp_hash TEXT, source TEXT, created REAL, updated REAL,
  resolved_by TEXT, resolution_note TEXT);
CREATE INDEX IF NOT EXISTS pay_user ON payments(user_id);
CREATE INDEX IF NOT EXISTS pay_status ON payments(status);
CREATE TABLE IF NOT EXISTS queries (id INTEGER PRIMARY KEY, txn_id TEXT, user_id TEXT, username TEXT, reason TEXT,
  status TEXT, admin TEXT, response TEXT, created REAL, updated REAL);
CREATE INDEX IF NOT EXISTS q_user ON queries(user_id);
CREATE INDEX IF NOT EXISTS q_status ON queries(status);
"""


def connect(path: Path = DB, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    else:
        con = sqlite3.connect(path, timeout=10)
        con.execute("PRAGMA journal_mode=WAL")
        con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def set_status(con, **kv):
    con.executemany("INSERT OR REPLACE INTO status VALUES (?,?)", [(k, str(v)) for k, v in kv.items()])
    con.commit()


def audit(con, actor: str, role: str, action: str, target: str = "", detail: dict | str = ""):
    con.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
                (time.time(), actor, role, action, target, detail if isinstance(detail, str) else json.dumps(detail)))
    con.commit()
