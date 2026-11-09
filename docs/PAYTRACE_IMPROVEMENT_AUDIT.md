# payTrace 2.0 → 2.1: technical audit

Audit date: 7 Nov 2026 · branch `paytrace-2.1` · baseline tag `baseline-2.0`.

## 1. Baseline reproduced from saved artifacts

`python -m src.v21.baseline` re-scores the **saved** models on each dataset's untouched test period (last 25% of the
timeline). No retraining. Output: `reports/v21/baseline.json`.

| Model | Recorded PR-AUC | Reproduced PR-AUC | Match |
|---|---:|---:|---|
| Sparkov app model (full hybrid, fused risk) | 0.9414 | 0.9414 | exact |
| IEEE-CIS native LightGBM (431 raw columns) | 0.4609 | 0.4609 | exact |

The other historical rows (synthetic model 0.115 / 0.067, pooled 0.665 / 0.129, transfer 0.018 / 0.038,
native + behaviour 0.394) are read from `reports/real/results.json`; they are kept as the historical baseline.

## 2. What is actually implemented (verified in code)

| Area | Status |
|---|---|
| Feature engine | `src/features.py` + `features_v2.py`: 39 online features (velocity, behaviour, device graph, payee graph, money flow, cold start, confirmed-fraud graph with 3-day label delay, merchant profile). Past-only by construction. |
| Models | LightGBM + Isolation Forest + MLP autoencoder + 3 tripwire Isolation Forests (payee, device, flow) + AE tripwire; isotonic calibration; logistic fusion of 8 inputs. |
| Graph | `src/graph_intel.py`: NetworkX graph of customers/devices/payees, Louvain clusters, 2-hop ego views, money chains. Used in the analyst case view (7-day ego graph) and the Streamlit dashboard. **No PageRank, betweenness, cycle detection, ring scoring, or graph signal in the decision.** |
| Unseen fraud | Leave-one-pattern-out exists for synthetic v2 data (`src/experiments.py lopo`): hybrid 69% mean recall vs 50% supervised. **No graph-assisted variant, no detection delay, no recall at fixed FPR.** |
| Drift | `src/drift.py` PSI over a 24 h window; shown only in the Streamlit dashboard. **Not in the web app; no Page-Hinkley/ADWIN.** |
| Feedback / retraining | `src/feedback.py` (label quality: second analyst for contradictions), `src/retrain.py` champion/challenger with promotion criteria. CLI only. |
| Demo scenarios | `src/demo_scenarios.py`, CLI only, synthetic v2 engine. **No scenario selector in the web app.** |
| Copilot | One evidence-grounded summary per case (Gemini/Groq/template). **No question answering.** Case report endpoint exists. |

## 3. Findings

### 3.1 Model performance and generalisation
1. **IEEE-CIS is the weak dataset.** Native LightGBM PR-AUC 0.461; our common 39 features only reach 0.121.
2. **"Behaviour features reduce IEEE performance" is not shown by the data.** The +behaviour run stopped at
   **iteration 9** (raw run: 1,874). Early stopping on the 50–60% slice cut it off almost immediately, so the 0.394 is
   an under-trained model. Likely cause: non-stationary features (running counts and account age grow with time, so
   their train and early-stopping distributions differ). To verify in Phase B with feature stability (PSI) and ablation.
3. The IEEE native model does not use the well-known IEEE signals: D-columns normalised by transaction day, a
   reconstructed card-holder id, per-card-holder aggregates and amount cents. All must be computed past-only.
4. Transfer between datasets fails (0.018 / 0.038): models must be dataset-specific. Already documented; kept.

### 3.2 Feature engineering
5. Missing from the online engine: rolling median and deviation of amount, time between transactions, burst
   indicator, payee centrality, transaction chain depth, money-in/money-out ratio over 24 h. (Velocity, account age,
   beneficiary history, device reuse, merchant reputation as confirmed-fraud rate already exist.)

### 3.3 Graph intelligence
6. No fraud-ring detector: fan-in to a beneficiary from many unrelated senders, shared devices across unrelated
   accounts, circular transfers, mule chains are visible only by eye in the ego graph.
7. Graph signals do not reach the decision except through per-transaction tripwires.
8. Sparkov (the app's dataset) has **no devices and no person-to-person transfers**, so device rings and mule chains
   can only be shown with UPI-style scenarios (synthetic v2 patterns), and must be labelled as simulated.

### 3.4 Unknown fraud
9. QR_SWAP (20%) and ATO (38%) are the weakest held-out patterns; graph structure (many first-time payers to one
   new payee; a burst on a new device) is the natural extra signal.

### 3.5 Risk fusion
10. No fusion ablation (LightGBM only vs +IF vs +AE vs all vs +graph vs +policy) and no calibration check on the real
    datasets.
11. **Familiar-payee override can suppress strong evidence.** The 2.0 override turns REVIEW into ALLOW whenever
    the model risk is low and the payee is familiar, even if a tripwire fires far above its threshold or the payee or
    device has confirmed fraud. Must be bounded.

### 3.6 Application, investigation and demo
12. No scenario selector for judges; no step-by-step view of features, scores, graph risk and decision.
13. Analyst view lacks an evidence timeline, ring panel, analyst notes on the case, interactive graph expansion and
    question answering.
14. Admin view lacks drift, unknown-fraud and model-comparison panels fed from backend data.

### 3.7 Performance and bugs
15. **Per-payment full scans:** `profile()`, `history_for()` and the familiar-payee lookup filter the 1.85M-row history
    with boolean masks: about 28 ms each, several per payment. Index by customer.
16. `confirmed_payee()` reads all of a customer's app payments per score; fine at demo scale.
17. Case network rebuilds a 7-day graph from SQL on every case open; acceptable at demo scale, cache if needed.

## 4. Prioritised modifications

| Priority | Change | Phase |
|---|---|---|
| P0 | Fix history scans (index by customer) | B |
| P0 | IEEE: explain the +behaviour drop; past-only card-holder features; LightGBM / XGBoost / CatBoost; imbalance; calibration | B |
| P0 | Fraud-ring module: temporal graph, PageRank, betweenness, components, cycles, fan-in, shared devices, mule chains; ring score | C |
| P0 | Analyst graph explorer and ring panel in the web app | C/G |
| P1 | LOPO with graph-assisted detection, detection delay, recall at fixed FPR | D |
| P1 | Fusion ablation and calibration on Sparkov and v2; bound the familiar-payee override | E |
| P1 | Admin scenario lab: 10 scenarios through the real engine, labelled simulated | F |
| P2 | Evidence timeline, notes, copilot Q&A, downloadable report | G/I |
| P2 | Drift (PSI + Page-Hinkley) and challenger workflow in the web app | H |
| P2 | Tests, latency/throughput benchmarks, documentation | J |

Out of scope by request: OTP, TOTP, MFA, identity verification, new infrastructure, front-end framework changes.
