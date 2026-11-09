# payTrace 2.1: what changed and why

payTrace 2.1 improves the existing 2.0 application in place (branch `paytrace-2.1`, baseline tag
`baseline-2.0`). Nothing was rebuilt: the same Starlette app, front end, models, roles, login and startup command
(`bash run_app.sh`). Every number below is produced by a script in `src/v21/` and stored in `reports/v21/`;
the full tables are in [reports/PAYTRACE_2_FINAL_EVALUATION.md](../reports/PAYTRACE_2_FINAL_EVALUATION.md).

## 1. Original functionality (kept)

Customer, analyst and admin roles; payments scored by LightGBM + Isolation Forest + autoencoder + tripwires with
isotonic calibration and logistic fusion; SHAP reasons; bank policy rules; customer choice with CAPTCHA on moderate
risk, hard block on high risk; analyst case queue and evidence-grounded AI summary (Gemini / Groq / template);
simulated traffic; IEEE-CIS real-data replay; REST scoring API. All 61 original tests still pass.

## 2. Modifications

| Area | Change | Files |
|---|---|---|
| Audit | Baseline reproduced from saved artifacts (Sparkov 0.9414, IEEE native 0.4609, exact) | `src/v21/baseline.py`, `docs/PAYTRACE_IMPROVEMENT_AUDIT.md` |
| Performance | Customer history indexed per customer (was a ~28 ms scan of 1.85M rows, several per payment) | `webapp/engine.py` |
| IEEE-CIS model | Diagnosis of the "+behaviour" drop; past-only card-holder features; feature selection; LightGBM / XGBoost / CatBoost; imbalance; calibration | `src/v21/ieee_opt.py` |
| Fraud rings | Online ring signals in every decision + offline ring analysis (Louvain, PageRank, betweenness, components, temporal cycles, mule chains) | `src/fraud_rings.py`, `webapp/engine.py`, `webapp/server.py` |
| Graph | Person-to-person payees now link to the customer for any id format (`P2P_<id>`), so app transfers form a network | `src/graph_intel.py` |
| Risk fusion | Ablation and calibration; familiar-payee shortcut can no longer override strong evidence | `src/v21/unseen_fusion.py`, `webapp/engine.py` |
| Unseen fraud | Leave-one-pattern-out with graph-assisted detection, detection delay, recall at fixed FPR | `src/v21/unseen_fusion.py` |
| Demo | Scenario lab: 10 simulated scenarios through the live engine, step-by-step view | `webapp/scenarios.py`, `webapp/static/app.js` |
| Investigation | Ring panel, 48 h evidence timeline, interactive graph explorer, confirmed vs suspicious styling | `webapp/server.py`, `webapp/static/app.js` |
| Copilot | 7 analyst questions + free text, answered from app data with verified facts and hypotheses separated; richer downloadable report | `src/copilot_qa.py`, `webapp/server.py` |
| Drift and learning | Page-Hinkley change detection next to PSI; drift and feedback-learning status in the admin Model page | `src/drift.py`, `webapp/server.py` |
| Device id in demos | "This browser" is scoped per account, because one presenter's laptop shared by all demo accounts is itself a ring signal; use "custom device" to show a real shared-device attack | `webapp/server.py` |

Not changed, by request: login flow, authentication (no OTP / TOTP / MFA), front-end framework, infrastructure.

## 3. New fraud detection capabilities

### 3.1 Fraud-ring signals in every decision (`RingMonitor`)

Before each payment is decided, a small time-windowed memory of the network (7 days of devices, 24 h of payees,
1 h of person-to-person money, 72 h of transfers for loops) reports:

| Signal | Meaning | Effect |
|---|---|---|
| `fan_in` | new payee (≤ 72 h) received from ≥ 5 different senders in 24 h, ≥ 80% of them first-time payers | REVIEW |
| `shared_device` | ≥ 3 different accounts on one device in 7 days | REVIEW |
| `device_new_accounts` | ≥ 2 brand-new accounts on one device in 24 h | REVIEW |
| `pass_through` | sender received P2P money in the last hour and sends ≥ 50% of it on | REVIEW |
| `mule_chain` | pass-through where the money already made ≥ 2 hops within the hour | REVIEW |
| `cycle` | the transfer closes a loop (money can already flow back to the sender within 72 h) | REVIEW |
| `confirmed_link` | payee or device linked to a payment an analyst confirmed as fraud | BLOCK |

Ring signals only raise a decision, and only a confirmed-fraud link can BLOCK. A ring signal means "investigate",
not "fraud". Cost: about 0.02 ms per payment.

### 3.2 Offline ring analysis (`analyse`)

A timestamped graph of customers, devices and payees from the stored scored events. Suspicious entities
(structural signals, confirmed fraud, or concentrated flagged activity) and the flagged links between them form a
suspicious subgraph. Rings are its connected parts; Louvain splits parts larger than 60 entities. Each ring reports
evidence, hubs (PageRank on the money graph plus betweenness), flagged payments, confirmed entities and status
(confirmed vs unconfirmed). Temporal cycles require transfers in time order within 72 h.

### 3.3 Analyst investigation

The case view adds the rings the case belongs to, a 48-hour evidence timeline (same customer, payee or device),
and an interactive graph explorer: click any entity to see its evidence and expand the graph around it. Solid red
marks analyst-confirmed fraud links; dashed amber marks links that are only flagged by the model. The copilot answers
"why flagged", "what changed", "is the beneficiary connected", "show the chain", "what evidence", "similar cases"
and "what next" from stored records, with facts and hypotheses kept apart. An LLM may only reword those facts.

### 3.4 Scenario lab (admin)

Ten SIMULATED scenarios: normal payments, account takeover, card testing, UPI mule, QR fraud, suspicious
beneficiary, shared-device attack, rapid transfers, coordinated fraud ring and a previously unseen pattern. Each
payment goes through `ENGINE.submit_payment`, the same path as a customer payment. For each step the lab shows the
input, behavioural features, model scores (supervised, fused, Isolation Forest, autoencoder, each tripwire against
its threshold), ring signals, rules, the final decision, the explanation, and whether a case was created. Outcomes
are never scripted. Invented entities are named `SIM_…` and payments are stored with source `scenario`.

### 3.5 Drift and feedback learning

PSI for amount and hour of day (against the training data) and for risk scores (against earlier traffic). Rates of
alerts, blocks, new payees and new devices. Page-Hinkley alarms on the alert-rate and risk streams. The feedback panel
shows label counts by quality status and whether a challenger is due (≥ 50 accepted labels including ≥ 10 fraud).
It also shows the promotion criteria of `src/retrain.py`; retraining is never automatic per label.

## 4. Model improvements (IEEE-CIS, real Vesta data, unseen last 25%)

| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Review recall / FPR | Loss |
|---|---:|---:|---:|---|---:|
| 2.0 native (431 raw columns) | 0.461 | 0.862 | 39.3% | 70.0% / 13.7% | ₹2.02 Cr |
| 2.0 "+behaviour" as recorded (stopped at iteration 9) | 0.394 | 0.852 | – | – | – |
| Same features, early-stopping patience 200 | 0.473 | 0.867 | 40.3% | 69.3% / 12.4% | ₹2.03 Cr |
| **2.1 raw + past-only engineered, no label feedback** | **0.494** | 0.886 | 41.8% | 75.3% / 13.6% | ₹1.58 Cr |
| 2.1 + card holder's confirmed fraud, labels 30 days late | 0.591 | 0.920 | 51.8% | 79.9% / 12.8% | ₹1.37 Cr |
| **2.1 + card holder's confirmed fraud, labels 7 days late (in the app)** | **0.669** | **0.945** | 58.6% | 86.7% / 14.2% | ₹0.98 Cr |

* **Diagnosis:** the 2.0 "+behaviour" drop was an early-stopping artifact (patience 100 → stop at iteration 9), not bad
  features. The payee-level behaviour features do drift (PSI > 0.25), but removing them did not help (0.469).
* **Engineered features (all past-only):** day-normalised D columns, a card-holder id (card1 + addr1 + D1 date), amount
  cents, 1- and 7-day windowed counts, the card holder's past behaviour, and the card holder's earlier frauds whose labels
  are at least 7 days old. Ablation: the delayed label carries most of the gain (0.669 → 0.494 without it); the D
  columns, cents and card-holder behaviour each add ~0.005; the windowed counts add nothing on IEEE-CIS.
* **Learners:** LightGBM 0.665, XGBoost 0.665, CatBoost 0.670 on the same 442 selected features. Class weighting 1
  (0.681) and two tuned settings (0.683 / 0.677) score slightly higher. The app's model was fixed in advance as the main
  experiment, not picked by test score.
* **Calibration:** isotonic calibration lowers Brier from 0.0185 to 0.0175 on the app's model.
* The admin **Real data** replay now uses this model (`models/ieee_v21/`). On the first 20,000 unseen real payments it
  caught 87.5% of fraud (2.0: 70%) at a 14.4% false-alarm rate.

## 5. Headline results (synthetic UPI-style data unless stated)

| Result | 2.0 | 2.1 |
|---|---|---|
| Mean recall on a fraud pattern removed from training (7 patterns) | 69% | **76%** |
| QR-swap fraud, unseen: recall / first alert after | 20% / 87.6 h | **70% / 0 h** |
| False-positive rate of the decision rule (unseen-pattern runs, mean) | 1.37% | 1.46% |
| Fraud entities covered by reported rings (test period) | n/a | 97% (68% of ring entities touched by fraud) |
| IEEE-CIS (real data) PR-AUC, label-free / with 7-day-old fraud labels | 0.461 | **0.494 / 0.669** |
| Account takeover in the scenario lab: first attacker payment | allowed | checked (takeover rule) |
| Full decision path latency p50 / p99 (Sparkov, incl. database writes, 2 vCPU) | – | 7.5 ms / 14.2 ms |

## 6. Limitations

* Synthetic UPI-style data is our own generator. The ring rules were written knowing its pattern descriptions, so ring
  results there are optimistic. The ring grouping was revised once after viewing test-period entity precision.
* The app's model is trained on Sparkov, which has no devices and no person-to-person transfers. Device and mule
  signals in the app come from the ring monitor and policy rules, not from a model trained on such fraud.
* On fraud patterns the model already knows, ring signals add about 0.1 points of false positives without recall gain.
* No GNN: there is no measured gain over the online graph features to justify one.
* Nothing here has been validated on live banking infrastructure.
