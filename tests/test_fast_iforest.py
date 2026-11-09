import numpy as np
from sklearn.ensemble import IsolationForest

from src.fast_iforest import FastIsolationForest


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6))
    iso = IsolationForest(n_estimators=50, random_state=0).fit(X)
    f = FastIsolationForest(iso)
    assert np.allclose(-iso.score_samples(X[:300]), [f.anomaly_score(x) for x in X[:300]])
