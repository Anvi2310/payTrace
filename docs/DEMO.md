# payTrace 2.1 demo script (about 10 minutes)

Start: `bash run_app.sh`, open http://localhost:8000, sign in as **demo / demo1234** (admin). Everything in the
Scenario lab is simulated and labelled as such; decisions come from the live engine and are not scripted, so say
"let's see what it decides" rather than predicting it.

## 1. Normal payment (1 min)
Scenario lab → **Normal customer payments** → Run. Open a step's **Details**: behavioural features (amount vs the
customer's average, first payment to this payee or not), model scores far below their thresholds, no ring signal,
final decision. Optional: sign in as a customer (e.g. `sp0155`) and pay a usual merchant from *Send money*.

## 2. Account takeover (2 min)
Scenario lab → **Account takeover**. Point at: new device, new city, first payments to these merchants, payments
seconds apart, amount ratio growing step by step. Show which detectors fire and how the decision escalates over the
burst. Click **Open the case** on a flagged step: reasons, risk breakdown, evidence timeline (the burst is visible
as "same customer" rows), then ask the copilot *What changed in the customer's behaviour?*

## 3. Fraud ring (3 min)
Scenario lab → **Coordinated fraud ring** (or **QR payment fraud**). Watch the ring columns: payee senders in 24 h,
device accounts, and ring rules appearing as the ring forms. Open the last case:
* **Fraud rings this case belongs to**: evidence, hubs (PageRank + betweenness), confirmed vs unconfirmed.
* **Graph explorer**: click the collector and *Expand around this entity*; dashed amber links are model-flagged,
  solid red are analyst-confirmed.
* Copilot: *Is the beneficiary connected to other suspicious accounts?* and *What should I investigate next?*
* Mark one case **Confirm fraud**, then run the scenario again: the payee/device now carries a confirmed-fraud link,
  which is the only ring signal allowed to BLOCK.
Fraud rings tab: all rings in the last 7 days, PageRank, betweenness, loops and mule chains.

## 4. Previously unseen fraud (2 min)
Scenario lab → **Previously unseen pattern** (round-amount loop transfers; the app's Sparkov model never saw a
person-to-person transfer). Show that the supervised score alone says little, and point to what does react: the
anomaly detectors' scores against their thresholds and the ring signals (cycle, pass-through). Then the evidence:
Model tab → *Previously unseen fraud* table (pattern removed from training; hybrid 69% → 76% with ring signals,
QR swap 20% → 70%).

## 5. Evidence and honesty (1 min)
Model tab: IEEE-CIS (real Vesta data) experiments, fusion ablation, drift (PSI + Page-Hinkley) and the feedback
loop's promotion rules. Real data tab: replay of unseen real IEEE-CIS payments with live caught/missed counts.
Say clearly: synthetic and Sparkov data are simulated, IEEE-CIS is real but anonymised, nothing is validated on a
live bank.
