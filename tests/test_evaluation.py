import numpy as np

from src.evaluation import at_threshold, cost_threshold, precision_threshold, ranking


def test_at_threshold_counts_and_loss():
    y = np.array([1, 1, 0, 0]); flag = np.array([1, 0, 1, 0]); amt = np.array([1000, 500, 10, 10])
    r = at_threshold(y, flag, amt)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["loss_inr"] == 500 + 100 and abs(r["fraud_value_caught_pct"] - 1000 / 1500) < 1e-9


def test_perfect_ranking():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.2, 0.8, 0.9])
    r = ranking(y, s)
    assert r["pr_auc"] == 1.0 and r["roc_auc"] == 1.0


def test_cost_threshold_prefers_catching_expensive_fraud():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    s = np.where(y == 1, rng.uniform(0.5, 1, 5000), rng.uniform(0, 0.6, 5000))
    t, _, _ = cost_threshold(y, s, np.where(y == 1, 50000, 100))
    assert at_threshold(y, s >= t, np.ones(5000))["recall"] > 0.9


def test_precision_threshold_floor():
    y = np.array([0, 1, 1]); s = np.array([0.1, 0.5, 0.9])
    assert precision_threshold(y, s, 0.95, floor=0.7) == 0.7
