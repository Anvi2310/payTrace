# payTrace v1 — Technical Audit

Audited 6 Nov 2026 against the actual source code, data and artifacts (not the documentation).
Every finding marked **verified** was checked by running code against the repository.

## 1. Architecture as implemented

```
generate_data.py ──► transactions.csv ──► features.py (FeatureEngine, online, stateful)
                                              │
                         train.py ◄───────────┘  (replays history → 27 features)
                            │  IsolationForest(all) → if_score
                            │  IsolationForest(payee graph) → novelty tripwire
                            │  LightGBM(27 + if_score) → isotonic calibration
                            │  thresholds (review = min cost, block = 95% precision) on VALIDATION
                            ▼
                models/ + reports/metrics.json
                            │
       scorer.py (FastIsolationForest + LightGBM + reasons) ◄── stream.py (in-process replay) → stream.db (SQLite)
                            │                                ◄── kafka_stream.py (untested)
                            ▼
                 dashboard/app.py (Streamlit, reads stream.db + reports/*.json)
```

## 2. Implemented and verified working

| Feature | Status |
|---|---|
| Online feature engine, same code for training and serving | **verified**: streamed decisions match offline evaluation |
| No future information in features (state updated after computing) | **verified** by code review |
| Time-based train / validation / test split | **verified** (days 0–36 / 36–45 / 45–60) |
| Isolation Forest ×2, LightGBM, isotonic calibration | **verified** |
| Fast Isolation Forest equals sklearn | **verified** (max abs diff 4e-16) |
| SHAP reason codes (LightGBM `pred_contrib`) | **verified**: contributions are in log-odds of the raw model, ranking is correct |
| Cost-based REVIEW / BLOCK thresholds | **verified** |
| PSI drift on a 24h window | **verified** |
| Feedback retrain experiment | **verified valid**: 0 of 18 QR-swap merchants straddle the feedback/evaluation boundary |
| Latency benchmark | **verified** |
| Streamlit dashboard, 4 tabs | **verified** (headless test + screenshots + user's Mac) |
| Kafka/Redpanda mode | **NOT verified**: no broker available; code reviewed only |
| LLM copilot | **NOT verified**: no API key; template fallback verified |

## 3. Critical experimental weaknesses (these inflate v1's metrics)

1. **Merchant category leaks the label (verified).** The generator never produces fraud in fuel, food delivery,
   grocery or travel (0 of 161,158 payments), and 11% of gaming payments are fraud. `category_code` became the
   top SHAP feature. Real fraud occurs across categories.
2. **Attacker devices are perfectly clean signals (verified).** 74.4% of fraud uses a pool of 60 attacker devices
   that no genuine user ever touches, so "device shared across accounts" separates fraud almost trivially.
3. **Mule accounts are perfectly clean (verified).** Mule payees receive 0 genuine payments. Real mules are often
   real people with normal activity.
4. **No unseen-entity evaluation possible (verified).** Only 5 of 4,943 test users never appear in training, so
   generalisation to new customers is untested.
5. **Validation slice is reused three times**: early stopping, isotonic calibration and threshold selection all use
   days 36–45, which makes thresholds optimistic.
6. **One baseline is tuned on the test set (verified):** "LightGBM raw fields" picks its threshold on test data.
   This flatters the baseline rather than payTrace, but it is still test leakage.
7. **Novelty tripwire threshold uses labels** (99.8th percentile of *legit* validation traffic). Acceptable, but it
   should be stated; a label-free alternative is a fixed alert budget.
8. **Only one unseen pattern is tested** (QR-swap). A leave-one-pattern-out experiment over every pattern is needed.
9. **Live dashboard precision/recall use ground-truth labels instantly.** Fine for a replay, but must be labelled as
   such; real labels arrive days later.
10. Challenger in the feedback experiment reuses the champion's calibrator and thresholds.

## 4. Bugs and engineering gaps

| Area | Finding |
|---|---|
| Streaming | No de-duplication (a replayed Kafka message is scored twice and double-counts state). No out-of-order handling. `init_db` wipes `stream.db` on every run. Kafka consumer auto-commits before the write: a crash loses events (at-most-once). |
| Feature engine | Per-user device/city/merchant sets grow without bound (memory). Timestamps parsed as naive local time. |
| fast_iforest | Imports sklearn's private `_average_path_length`; could break on an sklearn upgrade. |
| Copilot | Gemini API key sent in the URL query string (ends up in proxy/server logs). |
| Dashboard | Any viewer can start/stop the replay subprocess; no authentication; no analyst feedback; no network view. |
| Tests | None. |
| Config | Thresholds, costs and paths hard-coded in `config.py`; no environment configuration. |

## 5. Security issues

1. **Dashboard exposed on the network (verified on the user's Mac):** Streamlit printed a Network URL and an
   External URL, so anyone who can reach the machine could open it and launch processes. Fix: bind to localhost.
2. API key in URL (above). 3. No input validation on incoming transactions. 4. No audit log of decisions or analyst
   actions. 5. No role separation between viewers and analysts.

## 6. Performance

p50 0.13 ms / p99 0.75 ms per payment on one core (container); 0.37 ms p50 on the user's MacBook Air. The two
Isolation Forests are the largest stage (~70 µs). No bottleneck that matters at hackathon scale; the Python feature
loop over a user's 24h deque is O(events in 24h) per payment.

## 7. Highest-impact improvements (implementation order)

1. **Honest data (v2 generator):** fraud across all categories, fresh attacker devices, mules with normal activity,
   new-customer cohorts, and new scenarios (merchant collusion, rapid money movement, coordinated device rings).
   Keep v1 as the baseline.
2. **Rigorous evaluation:** separate train / early-stopping / calibration / threshold / test slices; baselines tuned
   on validation only; F1, precision@recall, recall@FPR; unseen-customer evaluation; calibration curves;
   threshold sensitivity; fairness across customer groups.
3. **Leave-one-pattern-out unseen-fraud experiment** over every pattern, comparing supervised, anomaly and hybrid.
4. **Model comparison and ablations:** LightGBM vs XGBoost vs CatBoost; autoencoder vs Isolation Forest;
   feature-group ablations; graph features.
5. **Graph intelligence:** delayed-label graph risk features (no leakage), community detection, network explorer.
6. **Transparent risk fusion** of component scores with cost-sensitive thresholds.
7. **Streaming hardening:** idempotency, reordering, at-least-once with dedupe, structured logs, failure handling.
8. **Evidence-grounded investigation copilot**, analyst feedback with validation, audit log, case reports.
9. **Dashboard redesign** around the seven requested modules, all reading real data.
10. **Tests, security fixes, docs.**
