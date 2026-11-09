"""Model inference service (Starlette; already installed as a Streamlit dependency, so no new framework).

    uvicorn src.api:app --host 127.0.0.1 --port 8000

POST /score      one transaction JSON -> recommendation (ALLOW / REVIEW / BLOCK), risk, reasons
GET  /health     liveness + model version
The service returns a RECOMMENDATION. It does not authorise or block payments; the payment system decides.
Optional bearer-token protection: set PAYTRACE_API_TOKEN.
"""
from __future__ import annotations

import hmac
import os
import threading

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from src.scorer_v2 import FraudScorerV2, InvalidTransaction

_lock = threading.Lock()                 # the feature engine is stateful; score one payment at a time
_scorer: FraudScorerV2 | None = None


def scorer() -> FraudScorerV2:
    """Loads the models and, unless PAYTRACE_WARMUP=0, rebuilds behavioural state from the dataset history
    (in production this state lives in Redis and survives restarts). Without it every customer looks brand new."""
    global _scorer
    if _scorer is None:
        s = FraudScorerV2()
        if os.getenv("PAYTRACE_WARMUP", "1") == "1":
            import pandas as pd
            from src import settings
            from src.stream_v2 import TXN_FIELDS
            from src.train_v2 import dataset_path, slices
            path = dataset_path(settings.DATASET)
            if path.exists():
                df = pd.read_csv(path, parse_dates=["timestamp"])
                hist = df[~slices(df)["test"]]
                s.warmup(hist[TXN_FIELDS].astype({"timestamp": str}).to_dict("records"), labels=hist.is_fraud.to_numpy())
                s.meta["warm_state_until"] = str(hist.timestamp.max())
        _scorer = s
    return _scorer


def _authorised(request: Request) -> bool:
    token = os.getenv("PAYTRACE_API_TOKEN")
    if not token:
        return True
    got = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    return hmac.compare_digest(got, token)


async def health(request: Request):
    s = scorer()
    return JSONResponse({"status": "ok", "model_version": s.meta.get("version", s.meta.get("trained_at")),
                         "behavioural_state_until": s.meta.get("warm_state_until"),
                         "decision_boundary": "recommendation only; authorisation is external"})


async def score(request: Request):
    if not _authorised(request):
        return JSONResponse({"error": "unauthorised"}, status_code=401)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "body must be JSON"}, status_code=400)
    try:
        with _lock:
            r = scorer().score(body)
    except InvalidTransaction as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    return JSONResponse({"txn_id": r["txn_id"], "recommendation": r["decision"], "risk": round(r["risk"], 6),
                         "supervised_probability": round(r["supervised_prob"], 6), "tripwires": r["tripwires"],
                         "reasons": r["reasons"], "risk_breakdown": r["risk_breakdown"],
                         "latency_ms": round(r["latency_ms"], 3)})


app = Starlette(routes=[Route("/health", health), Route("/score", score, methods=["POST"])])
