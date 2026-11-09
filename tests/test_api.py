import pytest
from starlette.testclient import TestClient

from conftest import needs_models, txn


@needs_models
def test_health_and_score():
    from src.api import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/score", json=txn(1, ts="2026-09-15 10:00:00"))
    assert r.status_code == 200 and r.json()["recommendation"] in ("ALLOW", "REVIEW", "BLOCK")


@needs_models
def test_score_rejects_invalid_and_requires_token(monkeypatch):
    from src.api import app
    c = TestClient(app)
    assert c.post("/score", json={"amount": 5}).status_code == 422
    assert c.post("/score", content=b"not json").status_code == 400
    monkeypatch.setenv("PAYTRACE_API_TOKEN", "secret")
    assert c.post("/score", json=txn(2)).status_code == 401
    assert c.post("/score", json=txn(3, ts="2026-09-15 10:01:00"), headers={"Authorization": "Bearer secret"}).status_code == 200
