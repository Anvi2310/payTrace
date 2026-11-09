#!/usr/bin/env bash
# One command: data -> features -> models -> latency report -> dashboard
set -e
cd "$(dirname "$0")"
python -m src.generate_data          # skip if you use the organizers' dataset (see README)
python -m src.train
python -m src.benchmark
streamlit run dashboard/app.py       # then click "Start live replay" in the sidebar
