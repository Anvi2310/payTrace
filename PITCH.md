# payTrace 2.0 — Presentation and demo material

## 3-minute demo script

1. **0:00 The problem (20 s).** "Fraud models only know yesterday's fraud. Scammers invent new tricks; the model is
   blind until enough labels pile up. payTrace closes that gap in real time, and explains itself."
2. **0:20 Live monitor (40 s).** Admin role → Live monitor → Start (1,200/s, 2% duplicates). Point at p50 latency,
   duplicates dropped and zero failures, and REVIEW/BLOCK alerts with reasons arriving.
3. **1:00 A new attack, live (40 s).** Run `python -m src.demo_scenarios --to-db`. Show the device ring: the first
   account passes, the second is blocked by the *device-graph tripwire*. The QR-swap merchant is flagged from its first
   payer and blocked by the fifth. "None of these needed a fraud label."
4. **1:40 Investigation (40 s).** Investigation centre → pick a device-ring case. Show SHAP reasons, the fusion breakdown,
   "Device used by 11 different accounts", and similar past cases with known outcomes. Generate the copilot summary and
   download the case report. Label it. Point out the second-analyst rule.
5. **2:20 The evidence (40 s).** Model intelligence → Unseen fraud: a pattern removed from training goes from 50% (supervised)
   to 69% (hybrid), and rises further after a few days of analyst labels. "And we confirmed it on fresh data we never
   looked at while building."

## Slide outline (10 slides)

1. **Title:** payTrace 2.0, real-time fraud intelligence for payments. DN-FN-01.
2. **Problem:** rare (1–2%), instant (< 1 ms budget), adversarial (new scams have no labels). Accuracy is meaningless.
3. **Audit first:** we found our own v1 shortcuts (category leaked the label; attacker devices and mules were too clean;
   missing values collapsed PR-AUC to 0.15). We rebuilt the data to be harder, and fixed the bug.
4. **Architecture:** diagram from ARCHITECTURE.md. Online features; 4 tripwires + LightGBM; calibrated fusion; decisions; stream; store.
5. **Rigour:** five temporal slices, baselines on the same slices, cost-based thresholds, fresh-seed confirmation.
6. **Results:** PR-AUC 0.831 → 0.946; losses ₹5.64L → ₹2.24L; latency 0.46 ms p50. CatBoost is competitive (honest).
7. **Unseen fraud:** leave-one-pattern-out chart; 50% → 69% mean recall; adaptation after 1/3/5 days of labels.
8. **Graph intelligence + copilot:** network explorer screenshot; evidence-grounded summary; never decides, never accuses.
9. **Responsible AI:** decision boundary, human in the loop, feedback validation, audit log, fairness by segment, secrets.
10. **Limits and next steps:** synthetic data; real-data run on the organizers' set / PaySim; Kafka + PostgreSQL + Redis
    in production; GNN only if real data shows multi-hop structure the features miss.

## Questions judges are likely to ask

* **"Isn't synthetic data easy?"** v1's data was too easy: we audited it, found three shortcuts and rebuilt it. All
  improvements were confirmed on a fresh seed that was never used during design.
* **"Why not a GNN?"** The ablation shows the graph signal is already captured online at < 1 ms. A GNN needs batch
  embeddings and showed no demonstrated gain here; we would test it on real data.
* **"Can the LLM block a payment?"** No. It receives an evidence pack and returns text. Decisions come from the scoring
  engine, and authorisation from the payment system.
* **"What stops someone poisoning the feedback?"** Roles, one label per analyst, a note plus a second analyst for
  contradicting labels, conflict exclusion, rate limits, an audit log, and promotion only if the challenger wins on
  unseen later data.
