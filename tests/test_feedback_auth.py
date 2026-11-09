import sqlite3
import time

import pytest

from src import auth, settings
from src.feedback import FeedbackRejected, accepted_labels, submit
from src.store import SCHEMA


@pytest.fixture
def con(tmp_path):
    c = sqlite3.connect(tmp_path / "f.db"); c.executescript(SCHEMA); c.row_factory = sqlite3.Row
    for tid, dec, risk in [("B1", "BLOCK", 0.99), ("R1", "REVIEW", 0.3), ("A1", "ALLOW", 0.001)]:
        c.execute("INSERT INTO events (txn_id, decision, risk) VALUES (?,?,?)", (tid, dec, risk))
    c.commit()
    return c


def test_viewer_cannot_label(con):
    with pytest.raises(FeedbackRejected):
        submit(con, "R1", "v", "viewer", 1)
    assert con.execute("SELECT action FROM audit_log").fetchone()["action"] == "label_denied"


def test_contradicting_block_needs_note_and_second_review(con):
    with pytest.raises(FeedbackRejected):
        submit(con, "B1", "a1", "analyst", 0, "no")
    assert submit(con, "B1", "a1", "analyst", 0, "customer confirmed by phone call") == "pending_second_review"
    assert "B1" not in accepted_labels(con)
    assert submit(con, "B1", "a2", "analyst", 0, "verified with bank statement") == "accepted"
    assert accepted_labels(con)["B1"] == 0


def test_disagreement_becomes_conflict_and_is_excluded(con):
    submit(con, "R1", "a1", "analyst", 1)
    assert submit(con, "R1", "a2", "analyst", 0) == "conflict"
    assert "R1" not in accepted_labels(con)


def test_no_double_labelling_and_rate_limit(con, monkeypatch):
    submit(con, "R1", "a1", "analyst", 1)
    with pytest.raises(FeedbackRejected):
        submit(con, "R1", "a1", "analyst", 1)
    monkeypatch.setattr(settings, "FEEDBACK_RATE_LIMIT_PER_HOUR", 1)
    with pytest.raises(FeedbackRejected):
        submit(con, "A1", "a1", "analyst", 1, "fraud reported by the customer later")


def test_auth_roles_and_login(monkeypatch):
    monkeypatch.setattr(settings, "SALT", "s")
    monkeypatch.setattr(settings, "USERS", f"alice:analyst:{auth.hash_password('pw')}")
    assert not auth.demo_mode()
    assert auth.login("alice", "pw") == "analyst" and auth.login("alice", "bad") is None
    assert auth.can("analyst", "label") and not auth.can("analyst", "promote") and auth.can("admin", "promote")
