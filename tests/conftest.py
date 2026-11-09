import sys
from pathlib import Path

import os

import pytest

import tempfile
os.environ.setdefault("PAYTRACE_WARMUP", "0")
os.environ.setdefault("PAYTRACE_APP_DATASET", "v2")    # web-app tests use the small synthetic dataset (fast start)
os.environ.setdefault("PAYTRACE_DB", str(Path(tempfile.mkdtemp(prefix="fs_test_")) / "test.db"))  # never touch the real DB      # keep unit tests fast; warm start is tested separately

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HAS_MODELS = (ROOT / "models" / "v2" / "components.joblib").exists()
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="run python -m src.train_v2 first")


def txn(i=0, ts="2026-09-01 10:00:00", user="U1", amount=500.0, payee="M1", device="D1", city="Chennai",
        category="grocery", channel="UPI"):
    return {"txn_id": f"T{i}", "timestamp": ts, "user_id": user, "amount": amount, "merchant_id": payee,
            "category": category, "channel": channel, "device_id": device, "city": city}
