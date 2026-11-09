#!/usr/bin/env bash
# payTrace 2.0 one-command start: data -> features -> models -> benchmark -> dashboard
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -f data/transactions_v2.csv ] || $PY -m src.generate_data_v2
[ -f models/v2/components.joblib ] || $PY -m src.train_v2 --dataset v2
[ -f reports/v2/latency.json ] || $PY -m src.benchmark_v2 --n 10000
echo "Open http://localhost:8501  (Live monitor -> Start, as admin role)"
exec streamlit run dashboard/app_v2.py --server.address "${STREAMLIT_SERVER_ADDRESS:-localhost}"
