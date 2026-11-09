"""Real-data replay: stream the UNSEEN test period of IEEE-CIS (real Vesta card-not-present payments) through the
IEEE-native model and keep a live score of what was caught and missed against the true labels.

The model was trained only on the earlier 50% of the timeline (thresholds on 67-75%); the replayed rows are the last
25%, never used for training. Columns are anonymised by the data owner, so reasons show column names such as C13 or
V258, which is all anyone outside Vesta can know about them.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from src.config import DATA_DIR, MODELS_DIR

# payTrace 2.1 model (raw + past-only engineered features) when it has been trained, else the 2.0 native model
if (MODELS_DIR / "ieee_v21" / "model.txt").exists() and (DATA_DIR / "ieee_replay_v21.parquet").exists():
    MODEL_DIR, DATA = MODELS_DIR / "ieee_v21", DATA_DIR / "ieee_replay_v21.parquet"
else:
    MODEL_DIR, DATA = MODELS_DIR / "ieee_native", DATA_DIR / "ieee_replay.parquet"
SHOW = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "DeviceInfo"]


def available() -> bool:
    return (MODEL_DIR / "model.txt").exists() and (MODEL_DIR / "meta.json").exists() and DATA.exists()


class Replay:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread, self.stop_evt = None, threading.Event()
        self.booster = self.meta = self.df = self.X = None
        self.reset()

    def reset(self):
        self.pos = 0
        self.stats = {"processed": 0, "frauds": 0, "caught": 0, "blocked": 0, "missed": 0, "false_alarms": 0,
                      "genuine": 0, "fraud_value_inr": 0.0, "caught_value_inr": 0.0}
        self.recent = deque(maxlen=40)
        self.state = {"running": False, "rate": 0, "loaded": self.df is not None, "error": None}

    def _ensure_loaded(self):
        if self.booster is not None:
            return
        import lightgbm as lgb
        self.meta = json.loads((MODEL_DIR / "meta.json").read_text())
        self.booster = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
        df = pd.read_parquet(DATA)
        cols = self.meta["columns"]
        X = df[cols].copy()
        for c, cats in self.meta["categories"].items():
            X[c] = pd.Categorical(X[c].astype("object"), categories=cats)
        for c in cols:
            if c not in self.meta["categories"]:
                X[c] = X[c].astype("float32")
        self.df, self.X = df.reset_index(drop=True), X.reset_index(drop=True)
        self.state["loaded"] = True

    def status(self) -> dict:
        s = dict(self.stats)
        total = len(self.df) if self.df is not None else (self.meta or {}).get("metrics", {}).get("test_rows", 0)
        s["recall"] = s["caught"] / s["frauds"] if s["frauds"] else None
        s["precision"] = s["caught"] / (s["caught"] + s["false_alarms"]) if (s["caught"] + s["false_alarms"]) else None
        s["false_alarm_rate"] = s["false_alarms"] / s["genuine"] if s["genuine"] else None
        s["value_caught_share"] = s["caught_value_inr"] / s["fraud_value_inr"] if s["fraud_value_inr"] else None
        meta = self.meta or (json.loads((MODEL_DIR / "meta.json").read_text()) if available() else {})
        return {"available": available(), **self.state, "total": total, "stats": s, "recent": list(self.recent),
                "offline_metrics": meta.get("metrics"), "thresholds": meta.get("thresholds"), "note": meta.get("note")}

    def start(self, rate: float = 300):
        if not available():
            raise RuntimeError("IEEE-CIS model not trained: run python -m src.train_ieee_native")
        if self.thread and self.thread.is_alive():
            return
        self.stop_evt.clear()
        self.state.update(running=True, rate=rate, error=None)
        self.thread = threading.Thread(target=self._run, args=(rate,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()

    def _run(self, rate):
        try:
            self._ensure_loaded()
            t_rev, t_blk = self.meta["thresholds"]["review"], self.meta["thresholds"]["block"]
            fx = self.meta.get("inr_per_usd", 88)
            batch = max(1, int(rate // 5))
            while self.pos < len(self.df) and not self.stop_evt.is_set():
                t0 = time.time()
                a, b = self.pos, min(self.pos + batch, len(self.df))
                p = self.booster.predict(self.X.iloc[a:b], num_threads=1)
                flagged = np.where(p >= t_rev)[0]
                contrib = self.booster.predict(self.X.iloc[a:b].iloc[flagged], pred_contrib=True, num_threads=1) if len(flagged) else None
                with self.lock:
                    for j, prob in enumerate(p):
                        row = self.df.iloc[a + j]
                        y = int(row["isFraud"]); amt = float(row["TransactionAmt"]) * fx
                        dec = "BLOCK" if prob >= t_blk else ("REVIEW" if prob >= t_rev else "ALLOW")
                        st = self.stats
                        st["processed"] += 1
                        if y:
                            st["frauds"] += 1; st["fraud_value_inr"] += amt
                            if dec != "ALLOW":
                                st["caught"] += 1; st["caught_value_inr"] += amt
                                st["blocked"] += dec == "BLOCK"
                            else:
                                st["missed"] += 1
                        else:
                            st["genuine"] += 1
                            st["false_alarms"] += dec != "ALLOW"
                        if dec != "ALLOW" or y:
                            reasons = []
                            if dec != "ALLOW":
                                k = int(np.where(flagged == j)[0][0])
                                c = contrib[k][:-1]
                                top = np.argsort(-c)[:3]
                                reasons = [f"{self.meta['columns'][i]} (+{c[i]:.2f})" for i in top if c[i] > 0]
                            self.recent.appendleft({
                                "txn_id": row["txn_id"], "time": row["timestamp"][:16], "amount_inr": round(amt),
                                "amount_usd": round(float(row["TransactionAmt"]), 2), "probability": round(float(prob), 4),
                                "decision": dec, "truth": "fraud" if y else "genuine",
                                "outcome": ("caught" if dec != "ALLOW" else "missed") if y else "false alarm",
                                "details": {k2: (None if pd.isna(row[k2]) else str(row[k2])) for k2 in SHOW if k2 in row},
                                "reasons": reasons})
                self.pos = b
                lag = (b - a) / rate - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
        except Exception as e:
            self.state["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.state["running"] = False


REPLAY = Replay()
