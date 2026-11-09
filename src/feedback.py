"""Analyst feedback with validation, and protection against erroneous or malicious labels.

Rules
  * only analysts and admins may label; every action is written to the audit log
  * one label per analyst per transaction (no silent edits)
  * a label that contradicts a high-confidence model output (a BLOCK labelled genuine, or a near-zero-risk payment
    labelled fraud) needs a written note AND a second analyst's agreement before it is used for training
  * two analysts who disagree put the transaction into 'conflict'; conflicting labels are never used for training
  * per-analyst rate limit (default 200 labels/hour) stops bulk poisoning
Only labels with status 'accepted' feed retraining.
"""
from __future__ import annotations

import time

from src import settings
from src.auth import can
from src.store import audit

LOW_RISK = 0.01


class FeedbackRejected(ValueError):
    pass


def submit(con, txn_id: str, analyst: str, role: str, label: int, note: str = "") -> str:
    if not can(role, "label"):
        audit(con, analyst, role, "label_denied", txn_id, "insufficient role")
        raise FeedbackRejected("only analysts or admins can label transactions")
    if label not in (0, 1):
        raise FeedbackRejected("label must be 0 (genuine) or 1 (fraud)")
    ev = con.execute("SELECT decision, risk FROM events WHERE txn_id=?", (txn_id,)).fetchone()
    if ev is None:
        raise FeedbackRejected("unknown transaction")
    recent = con.execute("SELECT COUNT(*) FROM feedback WHERE analyst=? AND wall_time>?",
                         (analyst, time.time() - 3600)).fetchone()[0]
    if recent >= settings.FEEDBACK_RATE_LIMIT_PER_HOUR:
        audit(con, analyst, role, "label_rate_limited", txn_id)
        raise FeedbackRejected("hourly labelling limit reached; contact an admin")
    if con.execute("SELECT 1 FROM feedback WHERE txn_id=? AND analyst=?", (txn_id, analyst)).fetchone():
        raise FeedbackRejected("you have already labelled this transaction")
    decision, risk = ev["decision"], float(ev["risk"])
    contradicts = (decision == "BLOCK" and label == 0) or (risk < LOW_RISK and label == 1)
    if contradicts and len(note.strip()) < 15:
        raise FeedbackRejected("this label contradicts a high-confidence model output: add a note of at least 15 characters")

    others = con.execute("SELECT id, label FROM feedback WHERE txn_id=?", (txn_id,)).fetchall()
    if any(o["label"] != label for o in others):
        status = "conflict"
        con.execute("UPDATE feedback SET status='conflict' WHERE txn_id=?", (txn_id,))
    elif others:
        status = "accepted"                           # second analyst agrees
        con.execute("UPDATE feedback SET status='accepted' WHERE txn_id=?", (txn_id,))
    else:
        status = "pending_second_review" if contradicts else "accepted"
    con.execute("INSERT INTO feedback (wall_time, txn_id, analyst, role, label, note, model_decision, model_risk, status) "
                "VALUES (?,?,?,?,?,?,?,?,?)", (time.time(), txn_id, analyst, role, label, note.strip()[:500], decision, risk, status))
    con.commit()
    audit(con, analyst, role, "label", txn_id, {"label": label, "status": status, "model_decision": decision})
    return status


def accepted_labels(con) -> dict[str, int]:
    rows = con.execute("SELECT txn_id, label FROM feedback WHERE status='accepted'").fetchall()
    return {r["txn_id"]: int(r["label"]) for r in rows}
