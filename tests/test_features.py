"""Feature engines: no future leakage, missing-value handling, delayed labels."""
import pandas as pd

from conftest import txn
from src.features import FeatureEngine
from src.features_v2 import FeatureEngineV2, V2_FEATURES


def stream(n=60):
    out = []
    for i in range(n):
        out.append(txn(i, ts=f"2026-09-01 10:{i // 2:02d}:{(i % 2) * 30:02d}", user=f"U{i % 5}", amount=100 + i * 7,
                       payee=f"M{i % 7}", device=f"D{i % 4}"))
    return out


def test_features_ignore_the_future():
    events = stream()
    e_full, e_prefix = FeatureEngineV2(), FeatureEngineV2()
    full = [e_full.process(t, label=i % 9 == 0) for i, t in enumerate(events)]
    prefix = [e_prefix.process(t, label=i % 9 == 0) for i, t in enumerate(events[:30])]
    assert full[:30] == prefix                     # later events cannot change earlier features


def test_v2_has_all_features():
    f = FeatureEngineV2().process(txn())
    assert all(k in f for k in V2_FEATURES)


def test_missing_device_is_not_a_shared_device():
    e = FeatureEngine()
    for i in range(50):
        f = e.process(txn(i, user=f"U{i}", device="UNKNOWN"))
    assert f["d_n_users"] == 1 and f["u_new_device"] == 0


def test_real_shared_device_is_detected():
    e = FeatureEngine()
    for i in range(5):
        f = e.process(txn(i, user=f"U{i}", device="D_SHARED"))
    assert f["d_n_users"] == 5


def test_confirmed_fraud_label_only_after_delay():
    e = FeatureEngineV2(label_delay_days=3)
    e.process(txn(0, ts="2026-09-01 10:00:00", payee="MULE"), label=1)
    early = e.process(txn(1, ts="2026-09-02 10:00:00", user="U2", payee="MULE"), label=0)
    late = e.process(txn(2, ts="2026-09-05 10:00:01", user="U3", payee="MULE"), label=0)
    assert early["m_conf_fraud_rate"] == 0.0 and late["m_conf_fraud_rate"] > 0


def test_unknown_categories_get_stable_codes():
    e = FeatureEngineV2()
    a = e.process(txn(0, category="TRANSFER"))["category_code"]
    b = e.process(txn(1, category="TRANSFER"))["category_code"]
    assert a == b and a >= 100


def test_money_passthrough():
    e = FeatureEngineV2()
    e.process(txn(0, ts="2026-09-01 10:00:00", user="A", payee="P2P_B", amount=10000))
    f = e.process(txn(1, ts="2026-09-01 10:05:00", user="B", payee="P2P_C", amount=9500))
    assert f["u_in_amt_1h"] == 10000 and f["u_passthrough"] > 0.9
