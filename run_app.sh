#!/usr/bin/env bash
# payTrace web app: customer banking app + analyst case management + admin console + integration API.
# First run builds the data and models (a few minutes); later runs start in ~30 s.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example (add a free GEMINI_API_KEY for AI explanations)."; }
[ -f data/transactions_v2.csv ] || $PY -m src.generate_data_v2
[ -f models/v2/components.joblib ] || $PY -m src.train_v2 --dataset v2
[ -f reports/v2/latency.json ] || $PY -m src.benchmark_v2 --n 10000
PORT=${PORT:-8000}
echo ""
echo "  payTrace is starting: open http://localhost:$PORT"
echo "  Demo sign-ins are listed on the login page (password demo1234)."
echo ""
exec $PY -m uvicorn webapp.server:app --host "${HOST:-127.0.0.1}" --port "$PORT"
