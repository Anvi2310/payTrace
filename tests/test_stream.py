"""Streaming guarantees: idempotency, ordering, dead-letter handling."""
import sqlite3

import pytest

from conftest import needs_models, txn
from src.store import SCHEMA
from src.stream_v2 import Deduper, Processor, Reorderer


def test_deduper():
    d = Deduper(capacity=3)
    assert not d.is_dup("a") and d.is_dup("a") and not d.is_dup("b")


def test_reorderer_releases_in_event_time_order():
    r = Reorderer(lateness_s=10)
    out = []
    for ts in [100, 105, 102, 130, 101, 140]:
        out += [(m, late) for m, late in r.push(ts, ts)]
    out += r.flush()
    times = [m for m, late in out if not late]
    assert times == sorted(times)
    assert any(late for _, late in out)            # 101 arrived after 102..105 were released -> marked late


@pytest.fixture
def mem_con(tmp_path):
    con = sqlite3.connect(tmp_path / "t.db"); con.executescript(SCHEMA); con.row_factory = sqlite3.Row
    return con


@needs_models
def test_processor_dedupes_and_deadletters(mem_con):
    from src.scorer_v2 import FraudScorerV2
    p = Processor(FraudScorerV2(), mem_con, "test")
    good = txn(1, ts="2026-09-15 10:00:00")
    p.handle(dict(good)); p.handle(dict(good))                 # duplicate delivery
    p.handle({**txn(2, ts="2026-09-15 10:00:05"), "amount": "abc"})   # invalid amount
    p.handle({**txn(3), "timestamp": "yesterday"})            # invalid timestamp
    p.flush()
    assert mem_con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    assert p.dups == 1
    assert mem_con.execute("SELECT COUNT(*) FROM deadletter").fetchone()[0] == 2
