"""Application engine: one shared scorer, the bank clock, payment lifecycle, analyst actions, background traffic.

Payment lifecycle (status column of `payments`):
    ALLOW  -> completed
    REVIEW -> verification_required -> completed (customer confirmed with OTP) | cancelled (customer: "not me")
    BLOCK  -> held -> released (analyst: genuine) | confirmed_fraud (analyst: fraud)
An analyst can also confirm fraud on any flagged payment. The engine RECOMMENDS; these statuses simulate what the
bank's payment system would do with the recommendation.

Clock: the models were trained on a dataset timeline ending at the start of the replay period, so the bank clock
starts there and moves forward in real time (and jumps ahead when simulated background traffic is replayed).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import secrets
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.fraud_rings import RingMonitor, ring_decision

from src import llm
from webapp import policy
from webapp.captcha import CaptchaStore
from src.store import audit, connect, set_status
from src.config import DATA_DIR, MODELS_DIR
from src.features import MISSING
from src.train_v2 import V2_DIR, dataset_path, slices

# Which data + model the app runs on. "sparkov" = model trained on the Kaggle Sparkov card dataset (the default when
# it has been trained); "v2" = payTrace's original synthetic UPI/card data. Choose with PAYTRACE_APP_DATASET (PAYTRACE_DATASET is the separate Streamlit dashboard's setting).
DATASETS = {
    "sparkov": {"data": DATA_DIR / "transactions_sparkov", "models": MODELS_DIR / "real_sparkov",
                "label": "Sparkov card payments (Kaggle; simulated, 999 customers, 693 merchants)"},
    "v2": {"data": DATA_DIR / "transactions_v2", "models": V2_DIR,
           "label": "payTrace synthetic UPI / card data"},
}
CHANNEL_LABELS = {"CARD_POS": "Card, in store", "CARD_ONLINE": "Card, online", "UPI": "UPI", "CARD": "Card",
                  "NETBANKING": "Net banking"}


def default_dataset() -> str:
    env = os.getenv("PAYTRACE_APP_DATASET", "").strip().lower()
    if env in DATASETS:
        return env
    return "sparkov" if (DATASETS["sparkov"]["models"] / "components.joblib").exists() else "v2"


def read_dataset(name: str) -> pd.DataFrame:
    base = DATASETS[name]["data"] if name in DATASETS else dataset_path(name).with_suffix("")
    pq = base.with_suffix(".parquet")
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(base.with_suffix(".csv"), parse_dates=["timestamp"])

FIRST_NAMES = ["Aarav", "Priya", "Rahul", "Ananya", "Vikram", "Divya", "Arjun", "Kavya", "Rohan", "Meera", "Karthik",
               "Sneha", "Aditya", "Lakshmi", "Siddharth", "Pooja", "Nikhil", "Isha", "Varun", "Deepa", "Manoj", "Nisha"]
PLAIN = [  # (feature prefixes, customer-safe wording). Deliberately vague about exact rules.
    (("u_new_device", "device_tripwire"), "it was made from a device we haven't seen on your account"),
    (("u_new_city",), "it was made from a location you don't usually pay from"),
    (("u_amt_ratio_mean", "amount", "log_amount", "u_amt_zscore", "u_amt_1h", "u_amt_24h"), "the amount is much higher than you usually send"),
    (("u_new_merchant", "m_age_hours"), "it is going to a payee you haven't paid before"),
    (("u_cnt_10m", "u_cnt_1h", "u_cnt_24h", "u_secs_since_last", "d_cnt_1h", "u_small_cnt_10m"), "several payments were made in a short time"),
    (("is_night", "hour"), "it was made at an unusual time for you"),
    (("d_n_users", "d_new_accounts_24h", "u_n_devices"), "the device has been used with other accounts"),
    (("m_users_1h", "m_users_24h", "m_new_payer_share_24h", "m_conf_fraud_rate", "m_round_share_24h",
      "m_night_share_24h", "payee_tripwire", "u_conf_fraud_payees"), "the payee has recently received unusual payment activity"),
    (("u_passthrough", "u_in_amt_1h", "u_in_cnt_1h", "flow_tripwire"), "money is being sent on soon after it arrived"),
    (("if_score", "ae_tripwire", "u_age_hours", "u_txn_count_total"), "the overall pattern is unusual for your account"),
]


def plain_reasons(reasons: list[dict]) -> list[str]:
    pre = [r["plain"] for r in reasons if str(r.get("feature", "")).startswith("policy:") and r.get("plain")]
    rest = [r for r in reasons if not str(r.get("feature", "")).startswith("policy:")]
    return list(dict.fromkeys(pre + _plain_model(rest)))[:4]


def _plain_model(reasons: list[dict]) -> list[str]:
    out = []
    for r in reasons:
        f = r.get("feature", "")
        for keys, text in PLAIN:
            if f in keys and text not in out:
                out.append(text)
    generic = "the overall pattern is unusual for your account"
    if generic in out and len(out) > 1:                 # specific reasons first
        out.remove(generic); out.append(generic)
    return out[:3]


# behavioural features shown to analysts and in the scenario lab (all computed before the payment is decided)
FEATURE_VIEW = ["amount", "hour", "u_txn_count_total", "u_secs_since_last", "u_cnt_10m", "u_cnt_1h", "u_cnt_24h",
                "u_amt_24h", "u_amt_zscore", "u_amt_ratio_mean", "u_new_device", "u_new_city", "u_new_merchant",
                "u_age_hours", "d_n_users", "m_age_hours", "m_users_24h", "m_new_payer_share_24h", "u_in_amt_1h",
                "u_passthrough", "m_conf_fraud_rate", "d_conf_fraud_rate"]

# Only a moderate-risk (REVIEW) payment is the customer's choice. A high-risk (BLOCK) payment is blocked outright: no
# customer action can send it; only the bank's fraud team can release it after investigating.
CHOICES = {"verification_required": ["proceed", "cancel"]}


def risk_level(risk: float, decision: str) -> str:
    """Words for customers instead of a score."""
    if decision == "BLOCK" or risk >= 0.5:
        return "high"
    if decision == "REVIEW" or risk >= 0.1:
        return "elevated"
    return "low"


def otp_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


class AppEngine:
    def __init__(self, dataset: str | None = None):
        self.dataset = dataset or default_dataset()
        self.models_dir = DATASETS.get(self.dataset, {}).get("models", V2_DIR)
        self.captchas = CaptchaStore()
        self.fresh_devices = {}          # (customer, device) -> (first use, first payment flagged)
        self.lock = threading.RLock()
        self.ready, self.error, self.progress = False, None, "starting"
        self.scorer = None
        self.sim_thread, self.sim_stop = None, threading.Event()
        self.sim_state = {"running": False, "processed": 0, "total": 0, "rate": 0}

    # ---------------- startup ----------------
    def start(self, background: bool = True, history_limit: int = 0):
        t = threading.Thread(target=self._load, args=(history_limit,), daemon=True)
        t.start()
        if not background:
            t.join()

    def _load(self, history_limit: int):
        try:
            from src.scorer_v2 import FraudScorerV2
            from src.stream_v2 import TXN_FIELDS
            self.progress = "loading models"
            sc = FraudScorerV2(models_dir=self.models_dir)
            self.progress = "loading customer history"
            df = read_dataset(self.dataset)
            s = slices(df)
            hist = df[~s["test"]]
            if history_limit:
                hist = hist.tail(history_limit)
            # Rebuilding every customer's behavioural state from history takes minutes on large datasets, so the
            # rebuilt state is saved once next to the model and reloaded on later starts (it is invalidated when the
            # model, the data or the history window changes).
            key = {"rows": len(hist), "end": str(hist.timestamp.max()), "model": sc.meta.get("trained_at"),
                   "limit": history_limit}
            cache = self.models_dir / "engine_state.pkl"
            restored = False
            if cache.exists():
                try:
                    with open(cache, "rb") as f:
                        saved = pickle.load(f)
                    if saved.get("key") == key:
                        sc.engine, restored = saved["engine"], True
                except Exception:
                    restored = False
            if not restored:
                self.progress = f"rebuilding behavioural state from {len(hist):,} past payments (first start only)"
                sc.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist.is_fraud.to_numpy())
                try:
                    with open(cache, "wb") as f:
                        pickle.dump({"key": key, "engine": sc.engine}, f, protocol=pickle.HIGHEST_PROTOCOL)
                except OSError:
                    pass
            self.progress = "building customer profiles"
            self.history = hist[["txn_id", "timestamp", "user_id", "amount", "merchant_id", "category", "channel",
                                 "device_id", "city"]].copy()
            self.replay = df[s["test"]].sort_values("timestamp", kind="stable").reset_index(drop=True)
            self.replay_ts = self.replay.timestamp.to_numpy()
            self.replay_pos = 0          # next row of the bank's other traffic not yet seen by the engine
            clean = hist[hist.is_fraud == 0]
            g = clean.groupby("user_id")
            mode = lambda x: x.mode().iat[0]
            self.profiles = g.agg(payments=("txn_id", "size"), median_amount=("amount", "median"),
                                  max_amount=("amount", "max"), device=("device_id", mode), city=("city", mode))
            # registries for the policy layer: who exists, which payees exist, which devices each customer has used
            self.known_customers = set(df.user_id.unique())
            self.known_payees = set(df.merchant_id.unique()) | {f"P2P_{u}" for u in self.known_customers}
            self.user_devices = hist.groupby("user_id").device_id.agg(set).to_dict()
            self.top_payees = (clean.groupby(["user_id", "merchant_id"]).size()
                               .rename("n").reset_index().sort_values("n", ascending=False))
            # per-customer row indexes: a payment looks up one customer's history in microseconds instead of
            # scanning all past payments (~28 ms per scan on Sparkov)
            self._hist_idx = self.history.groupby("user_id").indices
            self._top_idx = self.top_payees.groupby("user_id", sort=False).indices
            # a merchant's category comes from the merchant (like a card network's merchant category code), not
            # from what the payer types; payments are scored with it
            self.merchant_category = hist.groupby("merchant_id").category.agg(mode).to_dict()
            self.categories = sorted(df.category.unique().tolist())
            per_cat = hist.groupby("category").channel.agg(n="nunique", top=mode)
            self.category_channel = per_cat.loc[per_cat["n"] == 1, "top"].to_dict()   # e.g. shopping_net -> online
            self.has_devices = bool((~hist.device_id.head(20000).isin(MISSING)).any())
            self.channels = df.channel.value_counts().index.tolist()
            self.cities = df.city.value_counts().index.tolist()[:400]
            top = clean.user_id.value_counts()
            self.examples = {"customer": top.index[0], "merchant": clean.merchant_id.value_counts().index[0]}
            self.hist_end = hist.timestamp.max().to_pydatetime()
            # fraud-ring memory only needs the recent network (7 days)
            self.progress = "building the recent payment network (fraud-ring monitor)"
            self.rings = RingMonitor()
            recent = hist[hist.timestamp >= hist.timestamp.max() - pd.Timedelta(days=7)]
            for rec in recent[["timestamp", "user_id", "merchant_id", "device_id", "amount"]].to_dict("records"):
                self.rings.add(rec)
            self.last_ts = self.hist_end
            self.clock_base = time.time()
            self.scorer = sc
            with self.lock:
                self.catch_up(self.now())
            self.ready, self.progress = True, "ready"
        except Exception as e:  # surfaced on /api/health
            self.error = f"{type(e).__name__}: {e}"
            self.progress = "failed"

    def options(self) -> dict:
        """Choices for payment forms, taken from the data the model was trained on."""
        if not self.ready:
            return {"categories": [], "channels": [], "cities": [], "channel_labels": CHANNEL_LABELS, "examples": {}}
        return {"categories": self.categories, "channels": self.channels, "cities": self.cities,
                "channel_labels": CHANNEL_LABELS, "examples": self.examples, "dataset": self.dataset,
                "dataset_label": DATASETS.get(self.dataset, {}).get("label", self.dataset),
                "has_devices": self.has_devices}

    def channel_for(self, category: str, chosen: str | None) -> str:
        """In some data a category implies the channel (Sparkov: *_net is always online). Otherwise the payer's choice."""
        if category in self.category_channel:
            return self.category_channel[category]
        return chosen if chosen in self.channels else self.channels[0]

    def category_for(self, payee: str, chosen: str | None) -> str:
        """Known merchant -> its own category. Unknown payee -> what the payer chose, or 'other' (unknown to the model)."""
        return self.merchant_category.get(payee) or (chosen if chosen in self.categories else "other")

    # ---------------- clock ----------------
    def now(self) -> datetime:
        """Bank clock. The calendar continues the dataset's timeline, but the time of day is the real local time:
        time of day matters to the models (in Sparkov, payments between 22:00 and 04:00 are ~18x more often fraud),
        so a test at 3 pm is scored as 3 pm. It is the first moment after the history ends that has today's time of
        day, plus one bank day per real day the app has been running."""
        real = datetime.now()
        start = datetime.fromtimestamp(self.clock_base)
        sim = datetime.combine(self.hist_end.date(), real.time().replace(microsecond=0))
        if sim <= self.hist_end:
            sim += timedelta(days=1)
        sim += timedelta(days=(real.date() - start.date()).days)
        return max(sim, self.last_ts + timedelta(seconds=1))

    def catch_up(self, until: datetime) -> int:
        """Feed the bank's OTHER traffic (the dataset's later payments, with their labels arriving 3 days late as
        usual) into the engine's memory up to `until`, without scoring it. Without this the bank would look silent
        between the end of the history and now, and to models trained on continuous traffic a silent bank is itself
        an anomaly (the autoencoder then flagged every payment). Call with the lock held."""
        from src.stream_v2 import TXN_FIELDS
        j = int(np.searchsorted(self.replay_ts, np.datetime64(until), side="right"))
        if j <= self.replay_pos:
            return 0
        rows = self.replay.iloc[self.replay_pos:j]
        eng = self.scorer.engine
        recs = rows[TXN_FIELDS].astype({"timestamp": str}).to_dict("records")
        for rec, y in zip(recs, rows.is_fraud.to_numpy()):
            eng.process(rec if self.has_devices else {**rec, "device_id": "UNKNOWN"}, update=True, label=int(y))
            self.rings.add(rec)
        n = j - self.replay_pos
        self.replay_pos = j
        self.last_ts = max(self.last_ts, rows.timestamp.iloc[-1].to_pydatetime())
        return n

    # ---------------- customers ----------------
    def profile(self, uid: str) -> dict:
        if uid in self.profiles.index:
            p = self.profiles.loc[uid]
            payees = self._user_top(uid).head(6).merchant_id.tolist()
            return {"customer_id": uid, "past_payments": int(p.payments), "usual_amount": float(p.median_amount),
                    "largest_amount": float(p.max_amount),
                    "usual_device": None if str(p.device) in MISSING else p.device, "usual_city": p.city,
                    "saved_payees": payees,
                    "saved_payee_categories": {m: self.merchant_category.get(m) for m in payees}}
        return {"customer_id": uid, "past_payments": 0, "usual_amount": None, "largest_amount": None, "usual_device": None, "usual_city": None,
                "saved_payees": []}

    def user_history(self, uid: str) -> pd.DataFrame:
        idx = getattr(self, "_hist_idx", {}).get(uid)
        return self.history.iloc[idx] if idx is not None else self.history.iloc[0:0]

    def _user_top(self, uid: str) -> pd.DataFrame:
        idx = getattr(self, "_top_idx", {}).get(uid)
        return self.top_payees.iloc[idx] if idx is not None else self.top_payees.iloc[0:0]

    def history_for(self, uid: str, n: int = 15) -> list[dict]:
        h = self.user_history(uid).tail(n).iloc[::-1]
        return [{"ts": str(r.timestamp)[:19], "amount": float(r.amount), "payee": r.merchant_id, "category": r.category,
                 "channel": r.channel, "device_id": r.device_id, "city": r.city} for r in h.itertuples()]

    # ---------------- payments ----------------
    def submit_payment(self, *, user_id: str, payee: str, amount: float, category: str, channel: str, device_id: str,
                       city: str, source: str, actor: str, role: str, explain_with_llm: bool = True) -> dict:
        if not self.ready:
            raise RuntimeError("engine is still loading")
        app_customers = self._app_customers()
        category = self.category_for(payee, category)            # a known merchant's own category is always used
        channel = self.channel_for(category, channel)
        policy.check_limits(float(amount), channel)
        with self.lock:
            ts = self.now()
            self.catch_up(ts)
            ts = max(ts, self.last_ts + timedelta(seconds=1))
            txn = {"txn_id": f"{source[:3].upper()}-{int(time.time() * 1000)}-{secrets.token_hex(2)}",
                   "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "user_id": user_id, "amount": float(amount),
                   "merchant_id": payee, "category": category, "channel": channel, "device_id": device_id, "city": city}
            # a model trained without device data must see "unknown device" (as in training); the real device is still
            # recorded and used by the policy rules
            ring = self.rings.check(txn)                          # past-only network signals, real device id
            r = self.scorer.score(txn if self.has_devices else {**txn, "device_id": "UNKNOWN"})
            self.rings.add(txn, ring)
            self.last_ts = ts
            seen_devices = self.user_devices.setdefault(user_id, set())
            new_device = device_id not in seen_devices
            known_devices = any(str(d) not in MISSING for d in seen_devices)
            # a device first used by this customer within the last hour whose first payment was flagged stays suspect
            fresh = self.fresh_devices.get((user_id, device_id))
            fresh_flagged = bool(fresh and fresh[1] and (ts - fresh[0]).total_seconds() <= 3600)
            seen_devices.add(device_id)
        prof = self.profile(user_id)
        conf_n, conf_max, conf_devices = self.confirmed_payee(user_id, payee, with_devices=True)
        # The customer already verified this payee FROM THIS DEVICE (or it was sent normally / released) and this
        # payment is no larger: novelty of the payee and the recent spend to it are already explained by the customer.
        verified_here = float(amount) <= conf_devices.get(device_id, 0.0)
        h = self.user_history(user_id)
        h = h[h.merchant_id == payee]
        hist_n, hist_max = len(h), float(h.amount.max()) if len(h) else 0.0
        familiar_max = max(conf_max, hist_max)
        # A familiar payee: one this customer has already paid (in their bank history, or confirmed in the app), for an
        # amount no more than twice the largest they've paid them. An unverified payee is never trusted above the
        # large-payment limit.
        trusted = (bool(conf_n or hist_n) and float(amount) <= 2 * familiar_max
                   and (payee in self.known_payees or float(amount) < policy.UNVERIFIED_BLOCK))
        # The anomaly tripwires flag a payee the bank has never seen (it has no history at all). If the model's own
        # fraud score is low and the ONLY concern is that tripwire, a payee this customer already confirmed is not
        # questioned again. The tripwires are still recorded for analysts.
        sup_review = self.scorer.t.get("sup_review", 2.0)
        # Strong evidence is never overridden: a ring signal, a confirmed-fraud link in the graph features, or a
        # tripwire far (>1.5x) above its threshold keeps the payment in REVIEW even for a familiar payee.
        trip_ratio = max((r["tripwire_scores"][k] / max(self.scorer.t[f"{k}_trip"], 1e-9)
                          for k in r.get("tripwire_scores", {}) if f"{k}_trip" in self.scorer.t), default=0.0)
        # For a payment the customer already verified (same payee, same device, not larger), the novelty tripwires are
        # not strong evidence: they react to the very payee and spend the customer just confirmed. Ring signals,
        # confirmed-fraud links and the supervised model still count.
        strong = (bool(ring["ring_rules"]) or (trip_ratio > 1.5 and not verified_here)
                  or r["features"].get("m_conf_fraud_rate", 0) > 0 or r["features"].get("d_conf_fraud_rate", 0) > 0)
        if (trusted and not strong and r["decision"] == "REVIEW" and r["risk"] < self.scorer.t["review"]
                and r.get("supervised_prob", 0) < sup_review):
            r["decision"] = "ALLOW"
            r["trusted_payee_override"] = True
        pol = policy.apply_rules(
            amount=float(amount), payee=payee, model_decision=r["decision"],
            customer_known=user_id in self.known_customers or user_id in app_customers,
            payee_known=payee in self.known_payees or (payee.startswith("P2P_") and payee[4:] in app_customers),
            past_payments=prof["past_payments"], median_amount=prof["usual_amount"], max_amount=prof["largest_amount"],
            new_device=new_device, confirmed_count=conf_n, confirmed_max=conf_max,
            takeover=bool((known_devices and new_device and str(device_id) not in MISSING
                           and r["features"].get("u_new_city", 0) >= 1 and r["features"].get("u_new_merchant", 0) >= 1
                           and prof["past_payments"] >= 5) or (fresh_flagged and not verified_here)))
        r["model_decision"] = r["decision"]
        if new_device and str(device_id) not in MISSING:
            self.fresh_devices[(user_id, device_id)] = (ts, pol["decision"] != "ALLOW")
        ring_dec, ring_rules = ring_decision(ring, pol["decision"])
        r["decision"] = ring_dec
        r["policy"] = pol["rules"] + ring_rules
        r["ring"] = ring
        r["reasons"] = [{"feature": f"policy:{x['rule']}", "reason": x["reason"] if x["rule"].startswith("ring_") else "Policy: " + x["reason"],
                         "value": x["effect"], "impact": None, "plain": x["plain"]}
                        for x in r["policy"]] + list(r["reasons"])
        decision = r["decision"]
        status = {"ALLOW": "completed", "REVIEW": "verification_required", "BLOCK": "held"}[decision]
        action = {"ALLOW": "SENT", "REVIEW": "VERIFY", "BLOCK": "HELD"}[decision]
        reasons_plain = plain_reasons(r["reasons"])
        tp = self._user_top(user_id)
        paid_before = int(tp.loc[tp.merchant_id == payee, "n"].sum()) + conf_n
        facts = {"action": action, "amount": float(amount), "payee": payee, "category": category.replace("_", " "),
                 "time_of_day": ts.strftime("%H:%M"), "usual_amount": prof["usual_amount"],
                 "largest_past_payment": prof["largest_amount"], "past_payments": prof["past_payments"],
                 "times_paid_this_payee_before": paid_before, "payee_known_to_bank": payee in self.known_payees,
                 "confirmed_by_you_before": conf_n, "largest_confirmed_to_payee": conf_max,
                 "new_device": new_device, "risk_level": risk_level(r["risk"], decision),
                 "plain_reasons": reasons_plain}
        if explain_with_llm and decision != "ALLOW":
            msg, msg_src = llm.customer_message(facts)
        else:
            msg, msg_src = llm.template_customer_message(facts), "template"
        con = connect()
        try:
            self._write_event(con, txn, r, source)
            con.execute("INSERT INTO payments (txn_id, ts, user_id, amount, payee, category, channel, device_id, city, "
                        "decision, risk, status, customer_message, message_source, otp_hash, source, created, updated) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (txn["txn_id"], txn["timestamp"], user_id, float(amount), payee, category, channel, device_id, city,
                         decision, r["risk"], status, msg, msg_src, None, source, time.time(), time.time()))
            con.commit()
            audit(con, actor, role, "payment_submitted", txn["txn_id"], {"decision": decision, "source": source})
        finally:
            con.close()
        return {"txn_id": txn["txn_id"], "ts": txn["timestamp"], "decision": decision, "status": status, "action": action,
                "category": category, "channel": channel,
                "message": msg, "message_source": msg_src, "plain_reasons": reasons_plain,
                "risk_level": facts["risk_level"], "checked": {"past_payments": prof["past_payments"],
                                                               "payee_known": facts["payee_known_to_bank"],
                                                               "paid_before": paid_before,
                                                               "confirmed_before": conf_n},
                "choices": CHOICES.get(status, []),
                "risk": r["risk"], "model_decision": r["model_decision"], "policy": r["policy"],
                "scores": {"supervised_prob": float(r.get("supervised_prob", 0)), "fused_risk": float(r["risk"]),
                           "isolation_forest": float(r.get("if_score", 0)), "autoencoder": float(r.get("ae_score", 0)),
                           "tripwires": {k: {"score": round(float(v), 4), "threshold": round(float(self.scorer.t.get(f"{k}_trip", 0)), 4)}
                                         for k, v in (r.get("tripwire_scores") or {}).items()}},
                "features_view": {k: round(float(r["features"][k]), 3) for k in FEATURE_VIEW if k in r["features"]},
                "ring": {k: (v if not isinstance(v, float) else round(v, 3)) for k, v in ring.items()},
                "familiar_payee_override": bool(r.get("trusted_payee_override")),
                "reasons": r["reasons"], "risk_breakdown": r["risk_breakdown"],
                "tripwires": r["tripwires"], "latency_ms": r["latency_ms"]}

    @staticmethod
    def _norm(payee: str) -> str:
        return "".join(str(payee).split()).lower()

    def confirmed_payee(self, user_id: str, payee: str, with_devices: bool = False):
        """How many payments this customer completed to this payee in the app (sent normally, sent after passing the
        security check, or released by the fraud team), and the largest of them. Spaces and case are ignored.
        with_devices: also the largest completed amount per device the customer used for it."""
        con = connect()
        try:
            rows = con.execute("SELECT payee, amount, device_id FROM payments WHERE user_id=? AND status IN "
                               "('completed','released','cleared')", (user_id,)).fetchall()
        finally:
            con.close()
        key = self._norm(payee)
        mine = [(float(a), d) for p, a, d in rows if self._norm(p) == key]
        out = (len(mine), max((a for a, _ in mine), default=0.0))
        if with_devices:
            dev = {}
            for a, d in mine:
                dev[d] = max(dev.get(d, 0.0), a)
            return (*out, dev)
        return out

    def _app_customers(self) -> set[str]:
        con = connect()
        try:
            return {row[0] for row in con.execute("SELECT customer_id FROM users WHERE customer_id IS NOT NULL")}
        finally:
            con.close()

    def _write_event(self, con, txn, r, source, label=-1, ftype=None):
        feats = json.dumps({k: round(float(v), 4) for k, v in r["features"].items()}) if r["decision"] != "ALLOW" else None
        con.execute(f"INSERT OR REPLACE INTO events VALUES ({','.join('?' * 27)})",
                    (None, txn["txn_id"], txn["timestamp"], txn["user_id"], float(txn["amount"]), txn["merchant_id"],
                     txn.get("category"), txn.get("channel"), txn.get("device_id"), txn.get("city"), r["risk"],
                     r["supervised_prob"], r["decision"], ",".join(r["tripwires"]), r["if_score"], r["ae_score"],
                     r["graph_score"], json.dumps(r["reasons"]), json.dumps(r["risk_breakdown"]), feats, r["latency_ms"],
                     r["latency_ms"], 0, label, ftype or source.upper(), self.scorer.meta.get("trained_at", "v2"), time.time()))

    def customer_respond(self, txn_id: str, user_id: str, choice: str, captcha: str | None) -> dict:
        """The customer's own decision on a flagged payment.
        verification_required: 'proceed' (after the security check) -> completed, or 'cancel' -> cancelled.
        held (blocked, high risk): no customer choice at all; only an analyst can release it."""
        con = connect()
        try:
            p = con.execute("SELECT * FROM payments WHERE txn_id=? AND user_id=?", (txn_id, user_id)).fetchone()
            if p is None:
                raise KeyError("payment not found")
            allowed = CHOICES.get(p["status"], [])
            if choice not in allowed:
                raise ValueError(f"payment is {p['status'].replace('_', ' ')}; choices now: {', '.join(allowed) or 'none'}")
            if choice == "proceed":
                self.captchas.check(txn_id, captcha)
                # the customer verified this payment on this device: it is no longer a suspect new device
                key = (user_id, p["device_id"])
                if key in self.fresh_devices:
                    self.fresh_devices[key] = (self.fresh_devices[key][0], False)
            status, note = {"proceed": ("completed", "customer chose to send it (passed the security check)"),
                            "cancel": ("cancelled", "customer cancelled the payment")}[choice]
            con.execute("UPDATE payments SET status=?, updated=?, resolution_note=? WHERE txn_id=?",
                        (status, time.time(), note, txn_id))
            con.commit()
            audit(con, user_id, "customer", f"payment_{choice}", txn_id)
            return {"txn_id": txn_id, "status": status}
        finally:
            con.close()

    def analyst_decide(self, txn_id: str, analyst: str, role: str, label: int, note: str) -> dict:
        from src.feedback import submit
        con = connect()
        try:
            fb_status = submit(con, txn_id, analyst, role, label, note)
            p = con.execute("SELECT status FROM payments WHERE txn_id=?", (txn_id,)).fetchone()
            new = None
            if p is not None:
                genuine_next = {"held": "released", "release_requested": "released", "review": "cleared",
                                "verification_required": "completed"}
                new = "confirmed_fraud" if label == 1 else genuine_next.get(p["status"])
                if label == 1 and fb_status == "accepted" and getattr(self, "rings", None) is not None:
                    row = con.execute("SELECT payee, device_id, user_id FROM payments WHERE txn_id=?", (txn_id,)).fetchone()
                    if row:
                        self.rings.confirm_fraud(payee=row["payee"], device=row["device_id"])
                if new and fb_status in ("accepted", "pending_second_review"):
                    con.execute("UPDATE payments SET status=?, updated=?, resolved_by=?, resolution_note=? WHERE txn_id=?",
                                (new, time.time(), analyst, note[:300], txn_id))
                    # a customer query on this payment is answered by the same decision
                    con.execute("UPDATE queries SET status=?, admin=?, response=?, updated=? WHERE txn_id=? AND status='open'",
                                ("rejected" if label == 1 else "approved", analyst,
                                 ("The fraud team reviewed the payment and confirmed it as fraud; it stays blocked."
                                  if label == 1 else "The fraud team reviewed the payment and released it."), time.time(), txn_id))
                    con.commit()
            return {"txn_id": txn_id, "label_status": fb_status, "payment_status": new or (p["status"] if p else None)}
        finally:
            con.close()

    # ---------------- simulated background traffic ----------------
    def start_simulator(self, rate: float = 200, actor: str = "admin"):
        if self.sim_thread and self.sim_thread.is_alive():
            return self.sim_state
        self.sim_stop.clear()
        self.sim_thread = threading.Thread(target=self._simulate, args=(rate,), daemon=True)
        self.sim_thread.start()
        return self.sim_state

    def stop_simulator(self):
        self.sim_stop.set()
        return self.sim_state

    def _simulate(self, rate):
        """Score the bank's later traffic (the same stream catch_up feeds, from the same position, so no payment is
        seen twice) faster than real time, so the case queue and network fill up."""
        from src.stream_v2 import TXN_FIELDS
        con = connect()
        self.sim_state.update(running=True, processed=0, total=len(self.replay) - self.replay_pos, rate=rate)
        set_status(con, simulator="running")
        t0 = time.time()
        i = 0
        try:
            while not self.sim_stop.is_set():
                with self.lock:
                    if self.replay_pos >= len(self.replay):
                        break
                    row = self.replay.iloc[self.replay_pos]
                    self.replay_pos += 1
                    rec = {k: row[k] for k in TXN_FIELDS + ["is_fraud", "fraud_type"]}
                    rec["timestamp"] = str(rec["timestamp"])
                    txn = {k: rec[k] for k in TXN_FIELDS}
                    r = self.scorer.score(txn if self.has_devices else {**txn, "device_id": "UNKNOWN"},
                                          label_for_delay=int(rec["is_fraud"]))
                    self.last_ts = max(self.last_ts, row.timestamp.to_pydatetime())
                self._write_event(con, txn, r, "simulator", label=int(rec["is_fraud"]), ftype=rec["fraud_type"])
                if r["decision"] != "ALLOW":
                    con.execute("INSERT OR IGNORE INTO payments (txn_id, ts, user_id, amount, payee, category, channel, "
                                "device_id, city, decision, risk, status, source, created, updated) VALUES "
                                "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (txn["txn_id"], txn["timestamp"], txn["user_id"], float(txn["amount"]), txn["merchant_id"],
                                 txn["category"], txn["channel"], txn["device_id"], txn["city"], r["decision"], r["risk"],
                                 "held" if r["decision"] == "BLOCK" else "review", "simulator", time.time(), time.time()))
                if i % 50 == 0:
                    con.commit()
                i += 1
                self.sim_state["processed"] = i
                if rate:
                    lag = i / rate - (time.time() - t0)
                    if lag > 0:
                        time.sleep(lag)
        finally:
            con.commit()
            set_status(con, simulator="stopped")
            con.close()
            self.sim_state["running"] = False
