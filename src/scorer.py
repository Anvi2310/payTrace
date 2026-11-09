"""Real-time transaction scorer: features -> Stage 1 anomaly scores -> Stage 2 LightGBM
-> calibrated probability -> ALLOW / REVIEW / BLOCK decision with reason codes."""
import json
import time

import joblib
import lightgbm as lgb
import numpy as np

from src.config import FEATURE_COLUMNS, MODELS_DIR, REASON_TEXT
from src.fast_iforest import FastIsolationForest
from src.features import FeatureEngine

GRAPH_IDX = [FEATURE_COLUMNS.index(c) for c in
             ["m_age_hours", "m_users_1h", "m_users_24h", "u_new_merchant", "d_n_users"]]


class FraudScorer:
    def __init__(self, models_dir=MODELS_DIR):
        meta = json.loads((models_dir / "meta.json").read_text())
        self.meta = meta
        self.columns = meta["model_columns"]
        self.t_review, self.t_block = meta["t_review"], meta["t_block"]
        self.if_threshold = meta["if_threshold"]
        self.booster = lgb.Booster(model_file=str(models_dir / "lgbm.txt"))
        calib = joblib.load(models_dir / "calibrator.joblib")
        # isotonic regression == piecewise-linear interpolation; np.interp is ~50x faster per call
        self._cx, self._cy = calib.X_thresholds_, calib.y_thresholds_
        self.iforest = FastIsolationForest(joblib.load(models_dir / "iforest.joblib"))
        self.graph_iforest = FastIsolationForest(joblib.load(models_dir / "graph_iforest.joblib"))
        self.engine = FeatureEngine()

    def warmup(self, records):
        """Load historical transactions into the feature state (like a warm Redis)."""
        for r in records:
            self.engine.process(r)

    def score(self, txn, explain=True, timings=False):
        t0 = time.perf_counter()
        f = self.engine.process(txn)
        x = np.array([f[c] for c in FEATURE_COLUMNS], dtype=float)
        t1 = time.perf_counter()
        if_score = self.iforest.anomaly_score(x)
        g = x[GRAPH_IDX].copy(); g[0] = np.log1p(g[0])
        graph_score = self.graph_iforest.anomaly_score(g)
        t2 = time.perf_counter()
        xm = np.append(x, if_score).reshape(1, -1)
        raw = float(self.booster.predict(xm, num_threads=1)[0])
        prob = float(np.interp(raw, self._cx, self._cy))
        t3 = time.perf_counter()

        novelty = graph_score >= self.if_threshold
        if prob >= self.t_block:
            decision = "BLOCK"
        elif prob >= self.t_review or novelty:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        reasons = []
        if explain and decision != "ALLOW":
            contrib = self.booster.predict(xm, pred_contrib=True, num_threads=1)[0][:-1]
            order = np.argsort(-contrib)
            seen = set()
            for i in order[:6]:
                if contrib[i] <= 0:
                    break
                text = REASON_TEXT.get(self.columns[i], self.columns[i])
                if text not in seen:
                    seen.add(text)
                    reasons.append({"feature": self.columns[i], "reason": text,
                                    "value": round(float(xm[0, i]), 3),
                                    "impact": round(float(contrib[i]), 3)})
                if len(reasons) == 3:
                    break
            if novelty:
                reasons.insert(0, {"feature": "graph_if_score",
                                   "reason": "Novel pattern: new payee receiving from many users",
                                   "value": round(graph_score, 3), "impact": None})
        t4 = time.perf_counter()

        out = {
            "txn_id": txn.get("txn_id"), "probability": prob, "decision": decision,
            "if_score": if_score, "graph_score": graph_score, "novelty": bool(novelty),
            "reasons": reasons, "latency_ms": (t4 - t0) * 1000, "features": f,
        }
        if timings:
            out["timings_ms"] = {"features": (t1 - t0) * 1000, "anomaly_models": (t2 - t1) * 1000,
                                 "lightgbm": (t3 - t2) * 1000, "explain": (t4 - t3) * 1000}
        return out
