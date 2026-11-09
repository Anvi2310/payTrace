# payTrace 2.0 — Architecture

```
                 ┌──────────────────────────── offline (src/train_v2.py, src/experiments.py) ───────────────────────────┐
 dataset ──► adapters.validate ──► FeatureEngineV2 replay ──► slices: train │ early-stop │ calibration │ threshold │ test
 (synthetic v1/v2,                       (39 features)            │              │            │             │
  PaySim, IEEE-CIS)                                               ▼              ▼            ▼             ▼
                                            Isolation Forest, autoencoder,   LightGBM     isotonic +    REVIEW/BLOCK +
                                            payee/device/flow tripwires     (+ XGB/Cat)   fusion LR     tripwire levels
                                                                                                        └──► models/v2/
 ─────────────────────────────────────────────── online ──────────────────────────────────────────────────────────────────
 producer ──► bus (LocalBus | Kafka) ──► Processor ──► FraudScorerV2 ─────────────────────────► event store (SQLite/WAL)
              at-least-once              dedupe (txn_id)   validate → FeatureEngineV2 (state)      events · metrics · deadletter
                                         reorder/late flag → 4 tripwires + IF + LightGBM           feedback · audit_log · registry
                                         dead-letter        → calibration → risk fusion                   │
                                         JSON logs          → ALLOW / REVIEW / BLOCK + reasons             ▼
 api.py (POST /score) ─────────────────────────────────────► same scorer         dashboard/app_v2.py (7 modules)
                                                                                   graph_intel · investigation · feedback
                                                                                   retrain (champion/challenger)
```

## Components

| Module | Responsibility | Key decisions |
|---|---|---|
| `src/adapters.py` | Map synthetic / PaySim / IEEE-CIS onto one schema; validate and clean | PaySim balance columns dropped (label leakage); IEEE customer id is an approximation |
| `src/features.py` | v1 engine: 27 velocity, behaviour, device and payee features | Same code for training and serving; missing ids are never treated as entities |
| `src/features_v2.py` | +12 features: money flow, cold start, merchant behaviour profile, confirmed-fraud graph | Labels visible only after a 3-day delay; unknown categories get stable codes |
| `src/train_v2.py` | Five temporal slices; all components; fusion; thresholds | Nothing is fitted on the slice it is evaluated on |
| `src/scorer_v2.py` | Single-payment scoring in < 1 ms | Hand-rolled forward passes (Isolation Forest, autoencoder, fusion) match sklearn exactly; one LightGBM thread per call |
| `src/stream_v2.py` | Producer → bus → processor → store | Idempotent (dedupe), optional reorder buffer, dead-letter, metrics, JSON logs; Kafka commit after write |
| `src/store.py` | Event store schema | SQLite in WAL mode; plain SQL that ports to PostgreSQL |
| `src/api.py` | HTTP inference service | Starlette (already a dependency); optional bearer token; returns a recommendation only |
| `src/graph_intel.py` | Payment graph, findings, clusters, money chains | Every finding is a count from real events and model outputs |
| `src/investigation.py` | Evidence pack, similar cases, copilot, case reports | LLM sees only the evidence pack; never decides; never asserts guilt |
| `src/feedback.py` | Analyst labels | Role check, note + second analyst for contradicting labels, conflict exclusion, rate limit, audit |
| `src/retrain.py` | Champion / challenger | Compared on payments after the newest label; promoted only if PR-AUC ≥ and FPR ≤ +0.2 pt; backups kept |
| `src/auth.py`, `src/settings.py` | Roles and configuration | Secrets from the environment only; demo mode is explicit |

## Risk fusion and decisions

```
sup_logit, if_score, ae_score, payee_score, device_score, flow_score, m_conf_fraud_rate, d_conf_fraud_rate
        └────────── standardise ──► logistic regression (fitted on the calibration slice) ──► fused risk
BLOCK   fused risk ≥ t_block            (lowest threshold with ≥ 95% precision on the threshold slice)
REVIEW  fused risk ≥ t_review           (minimum cost: missed fraud ₹ + ₹100 per false alarm)
        OR supervised probability ≥ t_sup
        OR any novelty tripwire          (each alerts on ≤ 0.1% of traffic, set without labels)
ALLOW   otherwise
```
The fusion weights and each component's contribution are shown for every alert, so the score is never a black box.
The OR rule exists because the fusion layer is fitted on known fraud and learned to down-weight anomaly evidence;
without it, unseen fraud flagged by a tripwire or by the supervised model could be vetoed (see EXPERIMENTS.md §4).

## Decision boundary

payTrace produces **recommendations**. The payment switch (outside this project) owns authorisation and applies its
own policy: BLOCK is intended to trigger step-up verification or a hold, REVIEW an analyst queue. The LLM copilot has no
path to any decision; analyst labels only influence future models after validation and a champion/challenger test.

## What runs where in the hackathon build

| Production target | This build | Why |
|---|---|---|
| Kafka / Redpanda | In-process `LocalBus` (Kafka code path included) | No broker available in the build environment |
| PostgreSQL | SQLite (WAL) with portable schema | No database server available |
| Redis feature state | In-process dictionaries with identical semantics | Single process is enough for the demo |
| Live bank traffic | Replay of the dataset's unseen test period | No real data; replay labels are marked everywhere |
