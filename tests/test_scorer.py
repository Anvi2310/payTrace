import pytest

from conftest import needs_models, txn
from src.scorer_v2 import InvalidTransaction, validate_txn


@pytest.mark.parametrize("bad", [{}, {**txn(), "amount": -5}, {**txn(), "amount": "x"}, {**txn(), "timestamp": "soon"},
                                 {**txn(), "user_id": ""}])
def test_validation_rejects(bad):
    with pytest.raises(InvalidTransaction):
        validate_txn(bad)


def test_validation_truncates_long_ids():
    assert len(validate_txn({**txn(), "user_id": "x" * 1000})["user_id"]) == 128


@needs_models
def test_decision_follows_thresholds():
    from src.scorer_v2 import FraudScorerV2
    s = FraudScorerV2()
    r = s.score(txn())
    t = s.t
    assert r["decision"] in ("ALLOW", "REVIEW", "BLOCK")
    if r["decision"] == "BLOCK":
        assert r["risk"] >= t["block"]
    if r["decision"] == "ALLOW":
        assert r["risk"] < t["review"] and not r["tripwires"]


@needs_models
def test_card_testing_burst_is_flagged():
    from src.scorer_v2 import FraudScorerV2
    s = FraudScorerV2()
    for i in range(30):                                   # normal history
        s.score(txn(i, ts=f"2026-08-{1 + i % 28:02d} 1{i % 9}:00:00", user="V", amount=400, payee="M9", device="DV"))
    last = None
    for i in range(12):                                   # tiny payments seconds apart from a new device
        last = s.score(txn(100 + i, ts=f"2026-09-01 03:00:{i * 4:02d}", user="V", amount=5 + i, payee=f"G{i}",
                           device="D_NEW_X", category="gaming", channel="CARD", city="Delhi"))
    assert last["decision"] != "ALLOW"
