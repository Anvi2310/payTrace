"""Phase B: IEEE-CIS optimisation. Every number in reports/v21/ieee_opt.json is produced here.

    python -m src.v21.ieee_opt build            # engineered matrix -> data/v21_ieee.parquet (~3 min)
    python -m src.v21.ieee_opt diagnose         # why did "+behaviour" score 0.394? stability (PSI) + early stopping
    python -m src.v21.ieee_opt run <name> ...   # one experiment (see EXPERIMENTS); results merged into the json
    python -m src.v21.ieee_opt final            # refit the winner and save models/ieee_v21/

Splits are the project's usual ones on IEEE-CIS's own timeline: train 0-50%, early stopping 50-60%, calibration
60-67%, thresholds 67-75%, TEST 75-100% (never used for fitting, early stopping, calibration or thresholds).

Engineered features are all PAST-ONLY (computed from rows strictly earlier than the scored row):
  day-normalised D columns   Dk_n = day - Dk   (Dk counts days since an event, so day - Dk is a fixed date per card)
  card-holder id             uid = card1 + addr1 + D1_n  (the same card holder across transactions)
  amount cents               the fractional part of the US$ amount (pricing pattern)
  windowed counts            transactions in the previous 1 / 7 days for card1, uid, addr1, P_emaildomain, DeviceInfo
                             (windows, not running totals: running totals grow with time and drift)
  card-holder behaviour      seconds since the card holder's previous payment, past mean amount, amount / past mean,
                             number of past payments
  delayed confirmed fraud    past frauds of the same card holder whose label is at least 7 days old (as chargebacks
                             arrive in practice). Reported with and without, because it uses label feedback.
"""
from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.adapters import INR_PER_USD
from src.config import DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.evaluation import at_threshold, cost_threshold, precision_threshold, recall_at_fpr
from src.train_ieee_native import TEXT, read_raw
from src.train_v2 import SLICES

OUT = REPORTS_DIR / "v21" / "ieee_opt.json"
MATRIX = DATA_DIR / "v21_ieee.parquet"
DAY = 86400
LABEL_DELAY_DAYS = 7
D_NORM = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15"]
WINDOW_KEYS = ["card1", "uid", "addr1", "P_emaildomain", "DeviceInfo"]
BASE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=5.0, verbose=-1,
                   num_threads=2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- features
def _window_count(code: np.ndarray, t: np.ndarray, width: float, valid: np.ndarray) -> np.ndarray:
    """For each row: rows with the same key in [t - width, t) (strictly earlier). Rows are in time order."""
    key = code.astype(np.float64) * 1e9 + t
    order = np.argsort(key, kind="stable")
    sk = key[order]
    lo = np.searchsorted(sk, key - width, side="left")
    hi = np.searchsorted(sk, key, side="left")              # excludes the row itself and same-second ties
    out = (hi - lo).astype(np.float32)
    out[~valid] = np.nan
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    t = df.TransactionDT.to_numpy(np.float64)
    day = np.floor(t / DAY)
    E = pd.DataFrame(index=df.index)
    for c in D_NORM:
        E[f"{c}_n"] = (day - df[c].to_numpy(np.float64)).astype(np.float32)
    uid = (df.card1.astype("Int64").astype(str) + "_" + df.addr1.astype("Int64").astype(str) + "_"
           + pd.Series(E["D1_n"].to_numpy()).astype("Int64").astype(str).to_numpy())
    keys = {"card1": df.card1.astype(str), "uid": uid, "addr1": df.addr1.astype(str),
            "P_emaildomain": df.P_emaildomain.astype(str), "DeviceInfo": df.DeviceInfo.astype(str)}
    amt = df.TransactionAmt.to_numpy(np.float64)
    E["amt_cents"] = (amt - np.floor(amt)).astype(np.float32)
    for k, v in keys.items():
        code = pd.factorize(v)[0]
        valid = ~v.isin(["nan", "<NA>", "None"]).to_numpy() & ~v.str.contains("<NA>").to_numpy()
        for w, name in ((DAY, "1d"), (7 * DAY, "7d")):
            E[f"{k}_cnt_{name}"] = _window_count(code, t, w, valid)
    # card-holder (uid) behaviour, past only
    code = pd.factorize(uid)[0]
    uid_ok = (df.card1.notna() & df.addr1.notna()).to_numpy() & ~np.isnan(E["D1_n"].to_numpy())
    g = pd.DataFrame({"u": code, "t": t, "a": amt, "y": df.isFraud.to_numpy()})
    gg = g.groupby("u", sort=False)
    n_past = gg.cumcount().to_numpy()
    csum = gg.a.cumsum().to_numpy() - amt
    E["uid_n_past"] = n_past.astype(np.float32)
    E["uid_secs_since_prev"] = gg.t.diff().to_numpy().astype(np.float32)
    mean_past = np.where(n_past > 0, csum / np.maximum(n_past, 1), np.nan)
    E["uid_amt_mean_past"] = mean_past.astype(np.float32)
    E["uid_amt_ratio"] = (amt / mean_past).astype(np.float32)
    # delayed confirmed fraud of this card holder: frauds whose label is >= LABEL_DELAY_DAYS old
    key = code.astype(np.float64) * 1e9 + t
    fk = np.sort(key[g.y.to_numpy() == 1])
    ak = np.sort(key)
    cut = key - LABEL_DELAY_DAYS * DAY
    start = code.astype(np.float64) * 1e9
    E["uid_conf_fraud_past"] = (np.searchsorted(fk, cut) - np.searchsorted(fk, start)).astype(np.float32)
    lab = np.searchsorted(ak, cut) - np.searchsorted(ak, start)
    E["uid_conf_fraud_rate_past"] = np.where(lab > 0, E["uid_conf_fraud_past"] / np.maximum(lab, 1), np.nan).astype(np.float32)
    # sensitivity: the same signal when labels only arrive 30 days later
    cut30 = key - 30 * DAY
    E["uid_conf_fraud_past_30d"] = (np.searchsorted(fk, cut30) - np.searchsorted(fk, start)).astype(np.float32)
    # a card holder id needs all three parts; otherwise its features are unknown (not one giant "missing" holder)
    for c in ["uid_n_past", "uid_secs_since_prev", "uid_amt_mean_past", "uid_amt_ratio", "uid_conf_fraud_past",
              "uid_conf_fraud_rate_past", "uid_conf_fraud_past_30d", "uid_cnt_1d", "uid_cnt_7d"]:
        E.loc[~uid_ok, c] = np.nan
    return E


ENGINEERED_GROUPS = {
    "d_normalised": [f"{c}_n" for c in D_NORM],
    "amount_cents": ["amt_cents"],
    "window_counts": [f"{k}_cnt_{w}" for k in WINDOW_KEYS for w in ("1d", "7d")],
    "card_holder": ["uid_n_past", "uid_secs_since_prev", "uid_amt_mean_past", "uid_amt_ratio"],
    "delayed_fraud_label": ["uid_conf_fraud_past", "uid_conf_fraud_rate_past"],
}
LABEL_30D = ["uid_conf_fraud_past_30d"]          # sensitivity only, never in the default sets


def build():
    t0 = time.time()
    df = read_raw()
    log(f"raw {df.shape}")
    E = engineer(df)
    beh = pd.read_parquet(DATA_DIR / "features_v2_transactions_ieee.parquet")
    beh.columns = [f"b_{c}" for c in beh.columns]
    assert len(beh) == len(df)
    out = pd.concat([df.reset_index(drop=True), E.reset_index(drop=True), beh.reset_index(drop=True)], axis=1)
    for c in TEXT & set(out.columns):
        out[c] = out[c].astype("object").where(out[c].notna(), None).astype(str)
    out.to_parquet(MATRIX, index=False)
    log(f"matrix {out.shape} written in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- data access
_CACHE: dict = {}


def load_matrix():
    if "df" not in _CACHE:
        df = pd.read_parquet(MATRIX)
        for c in TEXT & set(df.columns):
            v = df[c].replace({"None": None, "nan": None})
            df[c] = v.astype("category")
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        frac = (df.TransactionDT - df.TransactionDT.min()) / (df.TransactionDT.max() - df.TransactionDT.min())
        _CACHE["s"] = {k: ((frac >= a) & (frac < b)).to_numpy() for k, (a, b) in SLICES.items()}
        _CACHE["df"] = df
    return _CACHE["df"], _CACHE["s"]


def raw_columns(df):
    skip = {"TransactionID", "isFraud", "TransactionDT"}
    return [c for c in df.columns if c not in skip and not c.startswith("b_") and c not in sum(ENGINEERED_GROUPS.values(), []) + LABEL_30D]


def behaviour_columns(df):
    return [c for c in df.columns if c.startswith("b_")]


# ---------------------------------------------------------------- evaluation
def evaluate(y, p, amt_inr, s, secs_predict=None):
    """Thresholds chosen on the 67-75% slice; every metric on the 75-100% test slice."""
    thr, te = s["thr"], s["test"]
    t_rev = float(cost_threshold(y[thr], p[thr], amt_inr[thr])[0])
    t_blk = float(precision_threshold(y[thr], p[thr], 0.95, floor=t_rev))
    yt, pt, at = y[te], p[te], amt_inr[te]
    rev = at_threshold(yt, pt >= t_rev, at)
    blk = at_threshold(yt, pt >= t_blk, at)
    return {"pr_auc": float(average_precision_score(yt, pt)), "roc_auc": float(roc_auc_score(yt, pt)),
            "recall_at_1pct_fpr": recall_at_fpr(yt, pt, 0.01), "recall_at_5pct_fpr": recall_at_fpr(yt, pt, 0.05),
            "review": {k: rev[k] for k in ("precision", "recall", "f1", "fpr", "fraud_value_caught_pct", "loss_inr")},
            "block": {k: blk[k] for k in ("precision", "recall", "fpr")},
            "loss_no_model_inr": float(at[yt == 1].sum()),
            "thresholds": {"review": t_rev, "block": t_blk},
            **({"predict_ms_per_txn": secs_predict} if secs_predict is not None else {})}


def calibrate(y, p, s):
    cal, te = s["cal"], s["test"]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[cal], y[cal])
    q = iso.predict(p)
    def ece(yy, pp, n=10):
        e = np.quantile(pp, np.linspace(0, 1, n + 1)); idx = np.clip(np.digitize(pp, e[1:-1]), 0, n - 1)
        return float(sum(abs(yy[idx == b].mean() - pp[idx == b].mean()) * (idx == b).mean() for b in range(n) if (idx == b).any()))
    return {"brier_raw": float(brier_score_loss(y[te], np.clip(p[te], 0, 1))), "brier_isotonic": float(brier_score_loss(y[te], q[te])),
            "ece_raw": ece(y[te], np.clip(p[te], 0, 1)), "ece_isotonic": ece(y[te], q[te])}, iso


# ---------------------------------------------------------------- learners
def fit_lgb(X, y, s, params=None, rounds=5000, patience=200):
    import lightgbm as lgb
    p = {**BASE_PARAMS, **(params or {})}
    t0 = time.time()
    m = lgb.train(p, lgb.Dataset(X[s["train"]], y[s["train"]]), rounds,
                  valid_sets=[lgb.Dataset(X[s["es"]], y[s["es"]])],
                  callbacks=[lgb.early_stopping(patience, verbose=False)])
    fit = time.time() - t0
    t1 = time.time(); pr = m.predict(X, num_iteration=m.best_iteration); ms = (time.time() - t1) / len(X) * 1e3
    return m, pr, {"best_iteration": int(m.best_iteration), "fit_seconds": round(fit), "predict_ms": ms}


def fit_xgb(X, y, s, rounds=5000, patience=200):
    import xgboost as xgb
    t0 = time.time()
    dtr = xgb.DMatrix(X[s["train"]], y[s["train"]], enable_categorical=True)
    des = xgb.DMatrix(X[s["es"]], y[s["es"]], enable_categorical=True)
    p = dict(objective="binary:logistic", eta=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.6,
             min_child_weight=5, scale_pos_weight=5, tree_method="hist", nthread=2, eval_metric="aucpr",
             max_cat_to_onehot=1)
    m = xgb.train(p, dtr, rounds, evals=[(des, "es")], early_stopping_rounds=patience, verbose_eval=False)
    fit = time.time() - t0
    del dtr, des; gc.collect()
    t1 = time.time()
    pr = m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, m.best_iteration + 1))
    ms = (time.time() - t1) / len(X) * 1e3
    return m, pr, {"best_iteration": int(m.best_iteration), "fit_seconds": round(fit), "predict_ms": ms}


def fit_cat(X, y, s, rounds=3000, patience=200):
    from catboost import CatBoostClassifier, Pool
    cats = [c for c in X.columns if str(X[c].dtype) == "category"]
    Xc = X.copy()
    for c in cats:
        Xc[c] = Xc[c].astype(object).where(Xc[c].notna(), "NA").astype(str)
    t0 = time.time()
    m = CatBoostClassifier(iterations=rounds, learning_rate=0.08, depth=8, scale_pos_weight=5, thread_count=2,
                           eval_metric="PRAUC", od_type="Iter", od_wait=patience, verbose=False,
                           allow_writing_files=False, border_count=64)
    m.fit(Pool(Xc[s["train"]], y[s["train"]], cat_features=cats), eval_set=Pool(Xc[s["es"]], y[s["es"]], cat_features=cats),
          use_best_model=True)
    fit = time.time() - t0
    t1 = time.time(); pr = m.predict_proba(Pool(Xc, cat_features=cats))[:, 1]; ms = (time.time() - t1) / len(X) * 1e3
    return m, pr, {"best_iteration": int(m.get_best_iteration()), "fit_seconds": round(fit), "predict_ms": ms}


# ---------------------------------------------------------------- experiments
def feature_set(df, name):
    raw = raw_columns(df)
    eng = sum(ENGINEERED_GROUPS.values(), [])
    beh = behaviour_columns(df)
    stable_beh = [c for c in beh if c in _CACHE.get("stable_behaviour", [])]
    sets = {
        "raw": raw,
        "raw_plus_behaviour": raw + beh,
        "raw_plus_stable_behaviour": raw + stable_beh,
        "raw_plus_engineered": raw + eng,
        "raw_plus_engineered_no_label": raw + [c for c in eng if c not in ENGINEERED_GROUPS["delayed_fraud_label"]],
        "raw_plus_engineered_label30": raw + [c for c in eng if c not in ENGINEERED_GROUPS["delayed_fraud_label"]] + LABEL_30D,
    }
    for g, cols in ENGINEERED_GROUPS.items():
        sets[f"engineered_minus_{g}"] = raw + [c for c in eng if c not in cols]
    if "selected" in _CACHE:
        sets["selected"] = _CACHE["selected"]
    return sets[name]


def psi(a, b, bins=10):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 100 or len(b) < 100:
        return float("nan")
    e = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(e) < 3:
        return 0.0
    pa = np.histogram(np.clip(a, e[0], e[-1]), e)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, e[0], e[-1]), e)[0] / len(b) + 1e-4
    return float(((pb - pa) * np.log(pb / pa)).sum())


def diagnose(res):
    """Why '+behaviour' lost: stability of each behaviour feature between train and early-stopping slices, and
    the effect of those features on early stopping."""
    df, s = load_matrix()
    beh = behaviour_columns(df)
    stab = {}
    for c in beh:
        stab[c] = {"psi_train_vs_es": psi(df.loc[s["train"], c].to_numpy(float), df.loc[s["es"], c].to_numpy(float)),
                   "psi_train_vs_test": psi(df.loc[s["train"], c].to_numpy(float), df.loc[s["test"], c].to_numpy(float)),
                   "missing_or_constant": bool(df[c].nunique() <= 1)}
    unstable = sorted([c for c, v in stab.items() if v["psi_train_vs_es"] > 0.25], key=lambda c: -stab[c]["psi_train_vs_es"])
    constant = [c for c, v in stab.items() if v["missing_or_constant"]]
    res["diagnose"] = {"behaviour_stability": stab, "unstable_psi_gt_0.25": unstable, "constant": constant}
    _CACHE["stable_behaviour"] = [c for c in beh if c not in unstable and c not in constant]
    log("unstable:", unstable, "constant:", constant)
    return res


def missing_and_correlation(res):
    """Feature selection on the TRAIN slice only: drop columns that are >99% missing or constant, and from each
    group of V columns that share a missing-value pattern keep only columns not >0.98 correlated with a kept one."""
    df, s = load_matrix()
    tr = df[s["train"]]
    raw = raw_columns(df)
    miss = tr[raw].isna().mean()
    drop_missing = miss[miss > 0.99].index.tolist()
    num = [c for c in raw if c not in drop_missing and str(df[c].dtype) != "category"]
    const = [c for c in num if tr[c].nunique(dropna=True) <= 1]
    vcols = [c for c in num if c.startswith("V") and c not in const]
    samp = tr[vcols].sample(min(60000, len(tr)), random_state=1)
    patt = samp.isna().mean().round(3)
    drop_corr = []
    for _, cols in patt.groupby(patt).groups.items():
        cols = list(cols)
        if len(cols) < 2:
            continue
        cm = samp[cols].corr().abs().to_numpy()
        kept = []
        for i, c in enumerate(cols):
            if any(cm[i, j] > 0.98 for j in kept):
                drop_corr.append(c)
            else:
                kept.append(i)
    dropped = set(drop_missing) | set(const) | set(drop_corr)
    eng = sum(ENGINEERED_GROUPS.values(), [])
    _CACHE["selected"] = [c for c in raw if c not in dropped] + eng
    res["selection"] = {"raw_columns": len(raw), "dropped_mostly_missing": drop_missing, "dropped_constant": const,
                        "dropped_correlated_V": len(drop_corr), "kept_raw": len(raw) - len(dropped),
                        "selected_total": len(_CACHE["selected"]),
                        "missing_share_summary": {"median": float(miss.median()), "share_cols_over_50pct": float((miss > 0.5).mean())}}
    return res


def run(name, res, learner="lgb", fset=None, params=None):
    df, s = load_matrix()
    fset = fset or name
    cols = feature_set(df, fset)
    X = df[cols]
    y = df.isFraud.to_numpy()
    amt = df.TransactionAmt.to_numpy(np.float64) * INR_PER_USD
    log(f"{name}: {learner} on {len(cols)} features")
    fit = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat}[learner]
    m, p, info = fit(X, y, s, **({"params": params} if params and learner == "lgb" else {}))
    out = {"learner": learner, "feature_set": fset, "n_features": len(cols), **info, **evaluate(y, p, amt, s, info["predict_ms"])}
    out["calibration"], _ = calibrate(y, p, s)
    if learner == "lgb":
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        out["top_features_gain"] = {k: round(float(v), 1) for k, v in imp.head(25).items()}
        out["gain_share_engineered"] = float(imp[[c for c in cols if c in sum(ENGINEERED_GROUPS.values(), [])]].sum() / imp.sum())
        out["gain_share_behaviour"] = float(imp[[c for c in cols if c.startswith("b_")]].sum() / imp.sum())
    res.setdefault("experiments", {})[name] = out
    log(f"{name}: PR-AUC {out['pr_auc']:.4f} ROC {out['roc_auc']:.4f} it {info['best_iteration']} "
        f"recall {out['review']['recall']:.3f} fpr {out['review']['fpr']:.4f}")
    _CACHE["last"] = (m, p, cols)
    return res


def load_res():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_res(res):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))


EXPERIMENTS = {
    # name: (learner, feature set, params)
    "raw": ("lgb", "raw", None),
    "raw_plus_behaviour": ("lgb", "raw_plus_behaviour", None),
    "raw_plus_stable_behaviour": ("lgb", "raw_plus_stable_behaviour", None),
    "raw_plus_engineered": ("lgb", "raw_plus_engineered", None),
    "raw_plus_engineered_no_label": ("lgb", "raw_plus_engineered_no_label", None),
    "raw_plus_engineered_label30": ("lgb", "raw_plus_engineered_label30", None),
    **{f"ablate_{g}": ("lgb", f"engineered_minus_{g}", None) for g in ENGINEERED_GROUPS},
    "selected": ("lgb", "selected", None),
    "selected_spw1": ("lgb", "selected", {"scale_pos_weight": 1.0}),
    "selected_spw20": ("lgb", "selected", {"scale_pos_weight": 20.0}),
    "selected_hpo_a": ("lgb", "selected", {"num_leaves": 127, "min_child_samples": 100, "learning_rate": 0.03, "colsample_bytree": 0.5}),
    "selected_hpo_b": ("lgb", "selected", {"num_leaves": 63, "min_child_samples": 50, "colsample_bytree": 0.4, "reg_lambda": 5.0}),
    "selected_xgb": ("xgb", "selected", None),
    "selected_cat": ("cat", "selected", None),
}


def final(res, name):
    """Refit the chosen experiment, save it for the admin Real-data replay (models/ieee_v21/) together with the
    test-period rows (data/ieee_replay_v21.parquet). Thresholds come from the 67-75% slice as always."""
    import lightgbm as lgb  # noqa: F401  (LightGBM only: the replay explains alerts with pred_contrib)
    learner, fset, params = EXPERIMENTS[name]
    assert learner == "lgb", "the replay uses LightGBM contributions for reasons"
    res = run(name, res, learner, fset, params)
    m, p, cols = _CACHE["last"]
    df, s = load_matrix()
    out = MODELS_DIR / "ieee_v21"
    out.mkdir(parents=True, exist_ok=True)
    m.save_model(str(out / "model.txt"), num_iteration=m.best_iteration)
    e = res["experiments"][name]
    cats = {c: [str(x) for x in df[c].cat.categories] for c in cols if str(df[c].dtype) == "category"}
    meta = {"columns": cols, "categories": cats, "thresholds": e["thresholds"], "best_iteration": e["best_iteration"],
            "metrics": {"pr_auc": e["pr_auc"], "roc_auc": e["roc_auc"], "review": e["review"], "block": e["block"],
                        "test_rows": int(s["test"].sum()), "test_frauds": int(df.isFraud[s["test"]].sum())},
            "inr_per_usd": INR_PER_USD, "experiment": name, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "payTrace 2.1 IEEE-CIS model: raw Vesta columns + past-only engineered card-holder features (src/v21/ieee_opt.py)."}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    rep = df.loc[s["test"], list(dict.fromkeys(cols + ["TransactionID", "isFraud", "TransactionDT"]))].copy()
    rep["txn_id"] = "IE" + rep.TransactionID.astype(int).astype(str)
    rep["timestamp"] = (pd.Timestamp("2017-12-01") + pd.to_timedelta(rep.TransactionDT, unit="s")).astype(str)
    for c in cats:
        rep[c] = rep[c].astype("object")
    rep.drop(columns=["TransactionDT"]).to_parquet(DATA_DIR / "ieee_replay_v21.parquet", index=False)
    log(f"saved {out} and replay rows {len(rep):,}")
    return res


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        return build()
    res = load_res()
    if cmd == "diagnose":
        res = diagnose(res); save_res(res); return
    names = sys.argv[2:]
    if any(n.startswith("selected") for n in names) or cmd == "final":
        res = missing_and_correlation(res); save_res(res)
    if "raw_plus_stable_behaviour" in names:
        res = diagnose(res)
    if cmd == "final":
        res = final(res, names[0]); save_res(res); return
    if cmd == "run":
        for n in names:
            learner, fset, params = EXPERIMENTS[n]
            res = run(n, res, learner, fset, params)
            save_res(res)
            gc.collect()


if __name__ == "__main__":
    main()
