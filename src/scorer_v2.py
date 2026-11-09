"""payTrace 2.0 real-time scorer.

One call per payment:  validate -> features (v2 engine) -> Isolation Forest, autoencoder, graph Isolation Forest
-> supervised model -> calibration -> risk fusion -> ALLOW / REVIEW / BLOCK + evidence.

The scorer only RECOMMENDS. Payment authorisation stays with the payment system (see README, "Decision boundary").
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime

import joblib
import numpy as np

from src.config import MODELS_DIR, REASON_TEXT
from src.fast_iforest import FastIsolationForest
from src.features_v2 import FeatureEngineV2
from src.train_v2 import FEATS, FUSION_INPUTS, TRIPWIRES

V2_DIR = MODELS_DIR / "v2"
REASONS = {**REASON_TEXT,
           "u_in_amt_1h": "Received money in the last hour", "u_in_cnt_1h": "Several incoming payments in the last hour",
           "u_passthrough": "Forwarding money just received (pass-through)", "u_age_hours": "Very new account",
           "d_new_accounts_24h": "Several new accounts on this device", "is_round_amount": "Round-number amount",
           "m_conf_fraud_rate": "Payee has confirmed fraud history", "d_conf_fraud_rate": "Device has confirmed fraud history",
           "u_conf_fraud_payees": "Customer previously paid a confirmed fraud payee",
           "m_new_payer_share_24h": "Payee paid mostly by first-time customers today",
           "m_round_share_24h": "Payee receiving many round-number amounts today",
           "m_night_share_24h": "Payee receiving many night-time payments today"}
COMPONENT_NAMES = {"sup_logit": "Supervised model", "if_score": "Isolation Forest anomaly", "ae_score": "Autoencoder anomaly",
                   "payee_score": "Payee-graph novelty", "device_score": "Device-graph novelty",
                   "flow_score": "Money-flow novelty", "m_conf_fraud_rate": "Payee fraud history",
                   "d_conf_fraud_rate": "Device fraud history"}
TRIP_NAMES = {"payee": "payee-graph", "device": "device-graph", "flow": "money-flow", "ae": "autoencoder"}
REQUIRED = ("txn_id", "timestamp", "user_id", "amount", "merchant_id")


class InvalidTransaction(ValueError):
    pass


def validate_txn(txn: dict) -> dict:
    if not isinstance(txn, dict):
        raise InvalidTransaction("transaction must be an object")
    missing = [k for k in REQUIRED if txn.get(k) in (None, "")]
    if missing:
        raise InvalidTransaction(f"missing fields: {missing}")
    try:
        amount = float(txn["amount"])
    except (TypeError, ValueError):
        raise InvalidTransaction("amount is not a number")
    if not (0 < amount < 1e9) or math.isnan(amount):
        raise InvalidTransaction("amount out of range")
    try:
        datetime.fromisoformat(str(txn["timestamp"]))
    except ValueError:
        raise InvalidTransaction("timestamp is not ISO-8601")
    clean = {k: str(txn.get(k, "UNKNOWN"))[:128] for k in ("txn_id", "user_id", "merchant_id", "category", "channel",
                                                           "device_id", "city")}
    clean.update(timestamp=str(txn["timestamp"]), amount=amount)
    return clean


class FraudScorerV2:
    def __init__(self, models_dir=V2_DIR):
        self.meta = json.loads((models_dir / "meta.json").read_text())
        c = joblib.load(models_dir / "components.joblib")
        self.c = c
        self.t = c.thresholds
        self.booster = c.supervised if c.kind == "lgbm" else None
        self.iforest = FastIsolationForest(c.iforest)
        self.trip_if = {k: FastIsolationForest(m) for k, m in c.trip_if.items()}
        self._tidx = {k: [FEATS.index(col) for col in cols] for k, cols in TRIPWIRES.items()}
        sc = c.ae_scaler
        self._ae_mu, self._ae_sd = sc.mean_, sc.scale_
        self._ae_W, self._ae_b = c.ae.coefs_, c.ae.intercepts_
        self._cx, self._cy = c.calibrator.X_thresholds_, c.calibrator.y_thresholds_
        self._fmu, self._fsd = c.fusion_scaler.mean_, c.fusion_scaler.scale_
        self._fw, self._fb = c.fusion.coef_[0], c.fusion.intercept_[0]
        self.columns = FEATS + ["if_score"]
        self.engine = FeatureEngineV2()

    # ---- helpers ----
    def _ae_score(self, x):
        z = (np.sign(x) * np.log1p(np.abs(x)) - self._ae_mu) / self._ae_sd
        h = z
        for i, (W, b) in enumerate(zip(self._ae_W, self._ae_b)):
            h = h @ W + b
            if i < len(self._ae_W) - 1:
                h = np.maximum(h, 0)
        return float(np.log1p(((h - z) ** 2).mean()))

    def _supervised(self, xm):
        if self.booster is not None:
            return float(self.booster.predict(xm, num_threads=1)[0])
        return float(self.c.supervised.predict_proba(xm)[0, 1])

    def warmup(self, records, labels=None):
        for i, r in enumerate(records):
            self.engine.process(r, label=None if labels is None else int(labels[i]))

    def add_label(self, txn: dict, label: int):
        """Confirmed outcome (chargeback / analyst decision) feeds the graph-risk features."""
        self.engine.add_label(txn["merchant_id"], txn["device_id"], txn["user_id"], int(label))

    # ---- main ----
    def score(self, txn: dict, explain: bool = True, label_for_delay: int | None = None) -> dict:
        t0 = time.perf_counter()
        txn = validate_txn(txn)
        f = self.engine.process(txn, label=label_for_delay)
        x = np.array([f[c] for c in FEATS], dtype=float)
        t1 = time.perf_counter()
        if_score = self.iforest.anomaly_score(x)
        tscore = {k: self.trip_if[k].anomaly_score(np.log1p(np.clip(x[idx], 0, None))) for k, idx in self._tidx.items()}
        ae_score = self._ae_score(x)
        tscore["ae"] = ae_score
        t2 = time.perf_counter()
        xm = np.append(x, if_score).reshape(1, -1)
        raw = self._supervised(xm)
        prob = float(np.interp(raw, self._cx, self._cy))
        comp = {"sup_logit": math.log(min(max(prob, 1e-5), 1 - 1e-5) / (1 - min(max(prob, 1e-5), 1 - 1e-5))),
                "if_score": if_score, "ae_score": ae_score, "payee_score": tscore["payee"],
                "device_score": tscore["device"], "flow_score": tscore["flow"],
                "m_conf_fraud_rate": f["m_conf_fraud_rate"], "d_conf_fraud_rate": f["d_conf_fraud_rate"]}
        zf = (np.array([comp[k] for k in FUSION_INPUTS]) - self._fmu) / self._fsd
        contrib = self._fw * zf
        risk = float(1 / (1 + math.exp(-(contrib.sum() + self._fb))))
        trips = [k for k in TRIP_NAMES if tscore[k] > self.t[f"{k}_trip"]]  # strict: a constant score (e.g. no device data) must never fire
        review = risk >= self.t["review"] or prob >= self.t.get("sup_review", 2.0) or bool(trips)
        decision = "BLOCK" if risk >= self.t["block"] else ("REVIEW" if review else "ALLOW")
        t3 = time.perf_counter()
        reasons, breakdown = [], []
        if explain and decision != "ALLOW":
            if self.booster is not None:
                sh = self.booster.predict(xm, pred_contrib=True, num_threads=1)[0][:-1]
                seen = set()
                for i in np.argsort(-sh)[:8]:
                    if sh[i] <= 0 or len(reasons) == 3:
                        break
                    text = REASONS.get(self.columns[i], self.columns[i])
                    if text not in seen:
                        seen.add(text)
                        reasons.append({"feature": self.columns[i], "reason": text, "value": round(float(xm[0, i]), 3),
                                        "impact": round(float(sh[i]), 3)})
            for k in trips:
                reasons.insert(0, {"feature": f"{k}_tripwire", "reason": f"Novel behaviour flagged by the {TRIP_NAMES[k]} detector",
                                   "value": round(tscore[k], 3), "impact": None})
            breakdown = sorted(({"component": COMPONENT_NAMES[k], "contribution": round(float(v), 3)}
                                for k, v in zip(FUSION_INPUTS, contrib)), key=lambda d: -d["contribution"])
            # the fusion intercept: fraud is rare, so every payment starts well below 50%. Shown so the bars add up:
            # risk = sigmoid(baseline + sum of contributions)
            breakdown.append({"component": "Starting point (fraud is rare)", "contribution": round(float(self._fb), 3)})
        t4 = time.perf_counter()
        return {"txn_id": txn["txn_id"], "risk": risk, "supervised_prob": prob, "decision": decision,
                "tripwires": [TRIP_NAMES[k] for k in trips], "if_score": if_score, "ae_score": ae_score,
                "graph_score": max(tscore["payee"], tscore["device"], tscore["flow"]), "tripwire_scores": tscore,
                "reasons": reasons, "risk_breakdown": breakdown, "features": f, "latency_ms": (t4 - t0) * 1000,
                "timings_ms": {"features": (t1 - t0) * 1e3, "anomaly": (t2 - t1) * 1e3, "model_fusion": (t3 - t2) * 1e3,
                               "explain": (t4 - t3) * 1e3}}
