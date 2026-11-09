"""End-to-end tests of the web application: auth, roles, CSRF, the payment lifecycle, OTP, analyst decisions, LLM."""
import time

import pytest
from starlette.testclient import TestClient

from conftest import needs_models

H = {"x-fs-request": "1", "x-fs-device": "testbrowser0001"}


@pytest.fixture(scope="module")
def client():
    import os
    os.environ["PAYTRACE_HISTORY_LIMIT"] = "60000"
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER", "PAYTRACE_DEMO_PASSWORD"):
        os.environ.pop(k, None)
    from webapp.server import app
    with TestClient(app) as c:
        for _ in range(240):
            if c.get("/api/health").json()["ready"]:
                break
            time.sleep(0.5)
        assert c.get("/api/health").json()["ready"], "engine did not start"
        yield c


def login(c, user):
    c.cookies.clear()
    r = c.post("/api/login", json={"username": user, "password": "demo1234"}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()


def a_customer(c):
    return next(a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer")


@needs_models
def test_auth_csrf_and_roles(client):
    c = client
    c.cookies.clear()
    assert c.get("/api/me").status_code == 401
    assert c.post("/api/login", json={"username": "admin", "password": "demo1234"}).status_code == 403  # no CSRF header
    assert c.post("/api/login", json={"username": "admin", "password": "wrong"}, headers=H).status_code == 401
    login(c, a_customer(c))
    assert c.get("/api/cases").status_code == 403
    assert c.get("/api/admin/overview").status_code == 403
    login(c, "analyst1")
    assert c.get("/api/cases").status_code == 200
    assert c.get("/api/admin/overview").status_code == 403
    assert c.post("/api/customer/pay", json={}, headers=H).status_code == 403
    login(c, "admin")
    assert c.get("/api/admin/overview").status_code == 200


@needs_models
def test_payment_lifecycle_otp_and_privacy(client):
    c = client
    login(c, a_customer(c))
    prof = c.get("/api/me").json()["profile"]
    ok = c.post("/api/customer/pay", headers=H, json={"payee": prof["saved_payees"][0], "amount": max(50, round(prof["usual_amount"] or 300)),
                                                     "category": "grocery", "channel": "UPI", "device": "usual", "city": prof["usual_city"]}).json()
    assert ok["decision"] == "ALLOW" and ok["status"] == "completed"
    risky = c.post("/api/customer/pay", headers=H, json={"payee": "P2P_U09999", "amount": 48000, "category": "p2p_transfer",
                                                        "channel": "UPI", "device": "this_browser", "city": "Delhi"}).json()
    assert risky["decision"] in ("REVIEW", "BLOCK")
    assert "risk" not in risky and "reasons" not in risky          # customers never see model internals
    assert "What looked different" in risky["message"] and risky["risk_level"] in ("elevated", "high")
    from webapp.server import ENGINE
    tid = risky["txn_id"]
    if risky["status"] == "held":                                  # high risk: blocked outright, no customer choice
        assert risky["choices"] == []
        assert c.get(f"/api/customer/payments/{tid}/captcha").status_code == 409
        assert c.post(f"/api/customer/payments/{tid}/respond", headers=H, json={"choice": "proceed", "captcha": "x"}).status_code == 409
    else:
        assert risky["choices"] == ["proceed", "cancel"]
        img = c.get(f"/api/customer/payments/{tid}/captcha")
        assert img.status_code == 200 and img.headers["content-type"].startswith("image/svg") and "no-store" in img.headers["cache-control"]
        text = ENGINE.captchas.peek(tid)
        bad = c.post(f"/api/customer/payments/{tid}/respond", headers=H, json={"choice": "proceed", "captcha": text.swapcase()})
        assert bad.status_code == 403 and "tries left" in bad.json()["error"]       # case matters
        r = c.post(f"/api/customer/payments/{tid}/respond", headers=H, json={"choice": "proceed", "captcha": text})
        assert r.status_code == 200 and r.json()["status"] == "completed"
        assert c.post(f"/api/customer/payments/{tid}/respond", headers=H, json={"choice": "proceed", "captcha": text}).status_code == 409
    # cancelling needs no security check
    again = c.post("/api/customer/pay", headers=H, json={"payee": "P2P_U09998", "amount": 47000, "category": "p2p_transfer",
                                                        "channel": "UPI", "device": "this_browser", "city": "Delhi"}).json()
    if again["choices"]:
        r = c.post(f"/api/customer/payments/{again['txn_id']}/respond", headers=H, json={"choice": "cancel"})
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert c.post("/api/customer/pay", headers=H, json={"payee": "M1", "amount": "abc"}).status_code == 422
    pays = c.get("/api/customer/payments").json()["payments"]
    assert {p["txn_id"] for p in pays} >= {ok["txn_id"], risky["txn_id"]}


@needs_models
def test_customer_cannot_touch_other_customers_payment(client):
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    login(c, custs[1])
    r = c.post("/api/customer/pay", headers=H, json={"payee": "P2P_U08888", "amount": 52000, "device": "this_browser", "city": "Delhi"}).json()
    login(c, custs[2])
    assert c.post(f"/api/customer/payments/{r['txn_id']}/respond", headers=H, json={"choice": "cancel"}).status_code in (403, 404)
    assert c.get(f"/api/customer/payments/{r['txn_id']}/captcha").status_code in (403, 404)


@needs_models
def test_analyst_check_case_and_decision(client):
    c = client
    login(c, "analyst1")
    r = c.post("/api/check", headers=H, json={"customer_id": "U01022", "payee": "P2P_U07777", "amount": 65000, "device": "this_browser"}).json()
    assert r["decision"] in ("REVIEW", "BLOCK") and r["reasons"]
    d = c.get(f"/api/cases/{r['txn_id']}").json()
    assert d["pack"]["transaction"]["txn_id"] == r["txn_id"] and d["network"]["nodes"]
    s = c.post(f"/api/cases/{r['txn_id']}/summary", headers=H).json()
    assert s["source"] == "template" and r["txn_id"] in s["summary"] or s["summary"].strip()
    rep = c.get(f"/api/cases/{r['txn_id']}/report")
    assert rep.status_code == 200 and "payTrace" in rep.text
    dec = c.post(f"/api/cases/{r['txn_id']}/decision", headers=H, json={"label": "fraud", "note": "test"}).json()
    assert dec["payment_status"] == "confirmed_fraud"
    assert c.post(f"/api/cases/{r['txn_id']}/decision", headers=H, json={"label": "maybe"}).status_code in (400, 422)


@needs_models
def test_llm_used_when_configured_and_falls_back_on_failure(client, monkeypatch):
    import src.llm as llm

    class Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"candidates": [{"content": {"parts": [{"text": "**Summary** mocked analyst summary"}]}}]}

    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: calls.append((url, kw)) or Resp())
    c = client
    login(c, "analyst1")
    r = c.post("/api/check", headers=H, json={"customer_id": "U01586", "payee": "P2P_U06666", "amount": 70000, "device": "this_browser"}).json()
    s = c.post(f"/api/cases/{r['txn_id']}/summary", headers=H).json()
    assert s["source"] == "gemini" and "mocked" in s["summary"]
    assert calls and calls[-1][1]["headers"]["x-goog-api-key"] == "test-key" and "key=" not in calls[-1][0]   # key not in URL

    def boom(url, **kw):
        raise llm.requests.ConnectionError("offline")
    monkeypatch.setattr(llm.requests, "post", boom)
    s2 = c.post(f"/api/cases/{r['txn_id']}/summary", headers=H).json()
    assert s2["source"] == "template" and s2["summary"]


@needs_models
def test_integration_api_requires_token(client, monkeypatch):
    c = client
    assert c.post("/api/v1/score", json={}).status_code == 503
    monkeypatch.setenv("PAYTRACE_API_TOKEN", "tok")
    assert c.post("/api/v1/score", json={}, headers={"authorization": "Bearer nope"}).status_code == 401
    r = c.post("/api/v1/score", headers={"authorization": "Bearer tok"},
               json={"user_id": "U01022", "payee": "M00003", "amount": 250, "device_id": "D01022_0", "category": "grocery"})
    assert r.status_code == 200 and r.json()["decision"] in ("ALLOW", "REVIEW", "BLOCK")


@needs_models
def test_security_headers(client):
    r = client.get("/")
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers.get("x-content-type-options") == "nosniff"


@needs_models
def test_empty_password_rejected_and_blank_env_lines_ignored(client, tmp_path, monkeypatch):
    c = client
    c.cookies.clear()
    assert c.post("/api/login", json={"username": "admin", "password": ""}, headers=H).status_code == 401
    import src.settings as st
    env = tmp_path / ".env"
    env.write_text("FS_TEST_BLANK=\nFS_TEST_SET=x\n")
    monkeypatch.delenv("FS_TEST_BLANK", raising=False)
    monkeypatch.setattr(st, "ROOT", tmp_path)
    st._load_dotenv()
    import os
    assert "FS_TEST_BLANK" not in os.environ and os.environ.get("FS_TEST_SET") == "x"
    monkeypatch.delenv("FS_TEST_SET")


@needs_models
def test_accounts_with_empty_password_are_repaired(client):
    from webapp.server import hash_pw, seed_users
    from src.store import connect
    con = connect()
    salt = "abc"
    con.execute("UPDATE users SET salt=?, password_hash=? WHERE username='analyst2'", (salt, hash_pw("", salt)))
    con.commit(); con.close()
    seed_users()
    client.cookies.clear()
    assert client.post("/api/login", json={"username": "analyst2", "password": "demo1234"}, headers=H).status_code == 200


@needs_models
def test_signup_creates_customer_who_can_pay(client):
    c = client
    c.cookies.clear()
    bad = c.post("/api/signup", headers=H, json={"display_name": "A", "username": "x", "password": "short"})
    assert bad.status_code == 422
    assert c.post("/api/signup", headers=H, json={"display_name": "A", "username": "newcust1", "password": "nodigitshere"}).status_code == 422
    assert c.post("/api/signup", json={"display_name": "A", "username": "newcust1", "password": "Passw0rd!"}).status_code == 403
    r = c.post("/api/signup", headers=H, json={"display_name": "New Customer", "username": "newcust1", "password": "Passw0rd!", "role": "admin"})
    assert r.status_code == 201 and r.json()["user"]["role"] == "customer"        # role in the body is ignored
    cid = r.json()["user"]["customer_id"]
    assert cid.startswith("U9")
    me = c.get("/api/me").json()                                                  # signed in straight away
    assert me["user"]["username"] == "newcust1" and me["profile"]["past_payments"] == 0
    pay = c.post("/api/customer/pay", headers=H, json={"payee": "M00003", "amount": 400, "device": "this_browser"})
    assert pay.status_code == 200 and pay.json()["decision"] in ("ALLOW", "REVIEW", "BLOCK")
    assert c.post("/api/signup", headers=H, json={"display_name": "B", "username": "newcust1", "password": "Passw0rd!"}).status_code == 409
    names = {a["username"] for a in c.get("/api/demo-accounts").json()["accounts"]}
    assert "newcust1" not in names and "demo" in names                            # real users are never advertised


@needs_models
def test_change_password(client):
    c = client
    c.cookies.clear()
    c.post("/api/signup", headers=H, json={"display_name": "Pw User", "username": "pwuser", "password": "Firstpass1"})
    assert c.post("/api/account/password", headers=H, json={"current_password": "wrong", "new_password": "Secondpass2"}).status_code == 403
    assert c.post("/api/account/password", headers=H, json={"current_password": "Firstpass1", "new_password": "weak"}).status_code == 422
    assert c.post("/api/account/password", headers=H, json={"current_password": "Firstpass1", "new_password": "Secondpass2"}).status_code == 200
    c.cookies.clear()
    assert c.post("/api/login", headers=H, json={"username": "pwuser", "password": "Firstpass1"}).status_code == 401
    assert c.post("/api/login", headers=H, json={"username": "pwuser", "password": "Secondpass2"}).status_code == 200


@needs_models
def test_admin_user_management(client):
    c = client
    login(c, "analyst1")
    assert c.post("/api/admin/users/create", headers=H, json={"role": "admin"}).status_code == 403
    login(c, "demo")                                                              # the quick demo login is an admin
    r = c.post("/api/admin/users/create", headers=H, json={"role": "analyst", "display_name": "Priya"}).json()
    assert r["username"].startswith("ana_") and len(r["password"]) >= 12
    c.cookies.clear()
    assert c.post("/api/login", headers=H, json={"username": r["username"], "password": r["password"]}).status_code == 200
    login(c, "demo")
    new = c.post(f"/api/admin/users/{r['username']}/reset", headers=H).json()["password"]
    c.cookies.clear()
    assert c.post("/api/login", headers=H, json={"username": r["username"], "password": r["password"]}).status_code == 401
    assert c.post("/api/login", headers=H, json={"username": r["username"], "password": new}).status_code == 200
    login(c, "demo")
    assert c.post("/api/admin/users/demo/delete", headers=H).status_code == 409   # not yourself
    assert c.post(f"/api/admin/users/{r['username']}/delete", headers=H).status_code == 200
    assert r["username"] not in {u["username"] for u in c.get("/api/admin/users").json()["users"]}
    cust = c.post("/api/admin/users/create", headers=H, json={"role": "customer", "customer_id": "U01022"}).json()
    assert cust["customer_id"] == "U01022"


@needs_models
def test_policy_limits_and_rules(client):
    c = client
    login(c, "analyst1")
    # the exact case from the user: ₹20 crore by net banking to a payee that doesn't exist
    r = c.post("/api/check", headers=H, json={"customer_id": "U12567", "payee": "M0003", "amount": 200_000_000,
                                              "category": "crypto", "channel": "NETBANKING", "device": "this_browser"})
    assert r.status_code == 404 and "No customer U12567" in r.json()["error"]       # that customer doesn't exist
    r = c.post("/api/check", headers=H, json={"customer_id": "U01022", "payee": "M00003", "amount": 200_000_000,
                                              "category": "crypto", "channel": "NETBANKING", "device": "this_browser"})
    assert r.status_code == 422 and "limit" in r.json()["error"]
    # within the limit but 10x+ the largest past payment, new device: blocked by policy whatever the model says
    prof = c.get("/api/customers/U01022").json()["profile"]
    amt = min(950_000, max(60_000, prof["largest_amount"] * 12))
    r = c.post("/api/check", headers=H, json={"customer_id": "U01022", "payee": "M00003", "amount": amt,
                                              "channel": "NETBANKING", "device": "this_browser"}).json()
    assert r["decision"] == "BLOCK" and any(p["rule"] == "extreme_amount" for p in r["policy"])
    # a customer that doesn't exist can't be checked at all
    assert c.post("/api/check", headers=H, json={"customer_id": "U99999999", "payee": "M00003", "amount": 100}).status_code == 404
    assert c.get("/api/customers/U99999999").status_code == 404
    # a large payment to a payee the bank can't verify is held, not just reviewed
    r = c.post("/api/check", headers=H, json={"customer_id": "U01022", "payee": "M00003455", "amount": 200_000,
                                              "channel": "NETBANKING", "device": "usual"}).json()
    assert r["decision"] == "BLOCK" and any(p["rule"] == "unverified_payee_large" for p in r["policy"])
    # the risk breakdown includes the starting point, so the bars add up to the risk
    import math
    total = sum(b["contribution"] for b in r["risk_breakdown"])
    assert abs(1 / (1 + math.exp(-total)) - r["risk"]) < 0.01
    # fake payee: at least REVIEW, with a policy reason
    r = c.post("/api/check", headers=H, json={"customer_id": "U01022", "payee": "M0003", "amount": 300,
                                              "device": "usual"}).json()
    assert r["decision"] in ("REVIEW", "BLOCK") and any(p["rule"] == "unverified_payee" for p in r["policy"])
    assert r["reasons"][0]["reason"].startswith("Policy:")
    # policy never lowers a decision and the model's own score is still reported
    assert "model_decision" in r and 0 <= r["risk"] <= 1


@needs_models
def test_customer_never_sees_policy_internals(client):
    c = client
    login(c, a_customer(c))
    r = c.post("/api/customer/pay", headers=H, json={"payee": "M0003", "amount": 300, "device": "usual"}).json()
    assert r["action"] in ("VERIFY", "HELD") and "policy" not in r and "model_decision" not in r
    assert "couldn't verify the payee" in " ".join(r["plain_reasons"])
    over = c.post("/api/customer/pay", headers=H, json={"payee": "M00003", "amount": 150_000, "channel": "UPI"})
    assert over.status_code == 422 and "UPI limit" in over.json()["error"]


def test_gemini_retired_model_is_replaced_automatically(monkeypatch):
    import src.llm as llm

    class R:
        def __init__(self, code, data): self.status_code, self._d, self.text = code, data, str(data)
        def json(self): return self._d
        def raise_for_status(self):
            if self.status_code >= 400:
                raise llm.requests.HTTPError(response=self)

    models = {"models": [
        {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.9-flash-preview", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]}]}
    used = []

    def post(url, **kw):
        used.append(url)
        if "gemini-2.5-flash" in url:
            return R(404, {"error": {"message": "This model models/gemini-2.5-flash is no longer available"}})
        return R(200, {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]})

    monkeypatch.setattr(llm, "_gemini_model_override", None)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm.requests, "get", lambda url, **kw: R(200, models))
    text, p = llm.complete("s", "u")
    assert text == "OK" and "gemini-3.8-flash:generateContent" in used[-1]          # newest stable non-lite Flash
    assert llm.config()["model"] == "gemini-3.8-flash"                             # remembered for later calls


SPARKOV_READY = (__import__("pathlib").Path(__file__).resolve().parents[1] / "models/real_sparkov/engine_state.pkl").exists()


@pytest.mark.skipif(not SPARKOV_READY, reason="Sparkov model / saved state not present")
def test_sparkov_engine_uses_merchant_category_and_hides_device_from_model():
    from webapp.engine import AppEngine
    eng = AppEngine("sparkov")
    eng.start(background=False)
    assert eng.ready, eng.error
    o = eng.options()
    assert "CARD_ONLINE" in o["channels"] and o["has_devices"] is False and "grocery_pos" in o["categories"]
    uid = o["examples"]["customer"]
    prof = eng.profile(uid)
    m = prof["saved_payees"][0]
    r = eng.submit_payment(user_id=uid, payee=m, amount=prof["usual_amount"], category="travel", channel="UPI",
                           device_id="D_WEB_test", city=prof["usual_city"], source="customer", actor="t", role="customer",
                           explain_with_llm=False)
    assert r["category"] == eng.merchant_category[m]                       # typed category ignored for a known merchant
    assert r["channel"] in ("CARD_POS", "CARD_ONLINE")
    from webapp.policy import PolicyDecline
    with pytest.raises(PolicyDecline):
        eng.submit_payment(user_id=uid, payee=m, amount=900_000, category="", channel="", device_id="D", city="",
                           source="customer", actor="t", role="customer", explain_with_llm=False)


def test_replay_reports_availability():
    from webapp.replay import REPLAY, available
    st = REPLAY.status()
    assert st["available"] == available() and st["stats"]["processed"] == 0


@needs_models
def test_directory_suggests_real_customers_and_merchants(client):
    c = client
    login(c, a_customer(c))
    assert c.get("/api/directory?q=u").status_code == 403                 # staff only
    login(c, "analyst1")
    cust = c.get("/api/directory?kind=customer&q=").json()["results"]
    assert cust and all(c.get(f"/api/customers/{x['value']}").status_code == 200 for x in cust[:3])
    m = c.get("/api/directory?kind=merchant&q=M000").json()["results"]
    assert m and all("M000" in x["value"] for x in m)



def test_captcha_store_rules():
    from webapp.captcha import ALPHABET, CaptchaStore, svg
    st = CaptchaStore()
    t = st.issue("p1")
    assert len(t) == 5 and set(t) <= set(ALPHABET) and not set("0Oo1Iil") & set(ALPHABET)
    assert any(ch.isupper() for ch in t) and any(ch.islower() for ch in t)      # always a mix of capital and small
    assert t not in svg("ABCDE") and "<svg" in svg(t)
    for _ in range(5):
        with pytest.raises(PermissionError):
            st.check("p1", "nope!")
    with pytest.raises(PermissionError, match="too many"):
        st.check("p1", t)                                    # locked after 5 wrong answers, even with the right one
    t2 = st.issue("p1")
    with pytest.raises(PermissionError):
        st.check("p1", t2.swapcase())                        # case matters
    st.check("p1", f" {t2} ")                                # spaces ignored
    with pytest.raises(PermissionError, match="expired"):
        st.check("p1", t2)                                   # one use only


def test_customer_explanation_template_uses_real_comparisons():
    from src.llm import template_customer_message
    m = template_customer_message({"action": "VERIFY", "amount": 23000, "payee": "Kilback LLC", "usual_amount": 6694,
                                   "largest_past_payment": 30000, "past_payments": 2089, "times_paid_this_payee_before": 0,
                                   "payee_known_to_bank": True, "time_of_day": "23:10", "new_device": False,
                                   "plain_reasons": []})
    assert "₹23,000 is about 3.4 times your usual payment of ₹6,694" in m
    assert "first payment to Kilback LLC" in m and "late at night (23:10)" in m and "Yes, send it" in m


@needs_models
def test_payee_confirmed_by_customer_is_trusted_for_similar_amounts(client):
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    login(c, custs[-1])                                             # a customer no other test has used
    first = c.post("/api/customer/pay", headers=H, json={"payee": "kojaj_motar", "amount": 2000, "channel": "UPI",
                                                         "device": "usual"}).json()
    assert first["status"] == "verification_required"              # unknown payee: the customer decides
    from webapp.server import ENGINE
    c.get(f"/api/customer/payments/{first['txn_id']}/captcha")
    assert c.post(f"/api/customer/payments/{first['txn_id']}/respond", headers=H,
                  json={"choice": "proceed", "captcha": ENGINE.captchas.peek(first["txn_id"])}).json()["status"] == "completed"
    small = c.post("/api/customer/pay", headers=H, json={"payee": "Kojaj _Motar", "amount": 23, "channel": "UPI",
                                                         "device": "usual"}).json()
    assert "known merchants" not in small["message"] and small["checked"]["confirmed_before"] == 1   # spaces/case ignored
    big = c.post("/api/customer/pay", headers=H, json={"payee": "kojaj_motar", "amount": 60000, "channel": "UPI",
                                                       "device": "usual"}).json()
    assert big["decision"] == "BLOCK"                               # trust doesn't stretch to a much larger amount


@needs_models
def test_small_payment_to_usual_merchant_is_not_stopped_by_an_anomaly_flag_alone(client):
    from webapp.server import ENGINE
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    login(c, custs[-2])
    prof = c.get("/api/me").json()["profile"]
    payee = prof["saved_payees"][0]
    orig = ENGINE.scorer.score
    def only_tripwire(txn, **k):                       # model score low, but one anomaly tripwire fires
        r = orig(txn, **k)
        r.update(decision="REVIEW", risk=0.0, supervised_prob=0.0, tripwires=["flow"])
        return r
    ENGINE.scorer.score = only_tripwire
    try:
        r = c.post("/api/customer/pay", headers=H, json={"payee": payee, "amount": 23, "device": "usual"}).json()
    finally:
        ENGINE.scorer.score = orig
    assert r["decision"] == "ALLOW" and r["status"] == "completed"


@needs_models
def test_scenario_lab_runs_through_the_real_engine(client):
    c = client
    login(c, a_customer(c))
    assert c.get("/api/admin/scenarios").status_code == 403              # admin only
    login(c, "admin")
    cat = c.get("/api/admin/scenarios").json()["catalog"]
    assert {x["id"] for x in cat} >= {"normal", "account_takeover", "fraud_ring", "unseen_pattern", "qr_fraud"}
    assert c.post("/api/admin/scenarios", headers=H, json={"scenario": "nope"}).status_code == 404
    r = c.post("/api/admin/scenarios", headers=H, json={"scenario": "qr_fraud", "delay": 0.2})
    assert r.status_code == 200
    for _ in range(100):
        d = c.get("/api/admin/scenarios").json()
        if not d["running"]:
            break
        time.sleep(0.2)
    assert not d["error"], d["error"]
    steps = d["steps"]
    assert len(steps) == 7 and all(s["simulated"] for s in steps)
    for s in steps:                                                       # every step carries the engine's own outputs
        assert s["decision"] in ("ALLOW", "REVIEW", "BLOCK") and "supervised_prob" in s["scores"] and "ring_rules" in s["ring"]
    assert steps[-1]["ring"]["payee_senders_24h"] == 7 and "fan_in" in steps[-1]["ring"]["ring_rules"]
    assert steps[-1]["decision"] in ("REVIEW", "BLOCK")                    # a ring signal never leaves it at ALLOW
    from webapp.server import connect
    con = connect()
    try:
        src = {r[0] for r in con.execute("SELECT source FROM payments WHERE txn_id IN (%s)" % ",".join("?" * len(steps)),
                                         [s["txn_id"] for s in steps])}
    finally:
        con.close()
    assert src == {"scenario"}
    # ring analysis and graph explorer see the scenario
    login(c, "analyst1")
    rep = c.get("/api/rings?days=7").json()
    assert rep["stats"]["payments"] > 0 and any(any(n.startswith("P:SIM_QR_") for n in r["nodes"]) for r in rep["rings"])
    qr = next(n for r in rep["rings"] for n in r["nodes"] if n.startswith("P:SIM_QR_"))
    g = c.get(f"/api/graph/node?id={qr}").json()
    assert g["center"] == qr and len(g["nodes"]) >= 7 and g["rings"]
    assert c.get("/api/graph/node?id=P:does_not_exist").status_code == 404
    case = c.get(f"/api/cases/{steps[-1]['txn_id']}").json()
    assert case["rings"] and case["timeline"] and any(t["kind"] == "this case" for t in case["timeline"])
    assert any(t["kind"] == "same payee" for t in case["timeline"])


@needs_models
def test_familiar_payee_does_not_override_a_ring_signal(client):
    """A shared device across accounts is strong evidence: even a payee the customer pays all the time stays checked."""
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    last = None
    for u in custs[3:6]:
        login(c, u)
        payee = c.get("/api/me").json()["profile"]["saved_payees"][0]
        last = c.post("/api/customer/pay", headers=H, json={"payee": payee, "amount": 23, "device": "custom",
                                                            "custom_device": "SHARED-FARM-01"}).json()
    assert last["decision"] in ("REVIEW", "BLOCK")
    assert "scores" not in last and "ring" not in last                    # customers never see internals


@needs_models
def test_copilot_questions_report_and_drift(client):
    c = client
    login(c, a_customer(c))
    tid = c.post("/api/customer/pay", headers=H, json={"payee": "SIM_UNKNOWN_PAYEE", "amount": 30000, "device": "this_browser"}).json()["txn_id"]
    login(c, "analyst1")
    for q in ("why", "behaviour", "beneficiary", "chain", "evidence", "similar", "next", "is the payee linked to anything?"):
        r = c.post(f"/api/cases/{tid}/ask", headers=H, json={"question": q})
        assert r.status_code == 200, r.text
        a = r.json()
        assert a["verified"] and a["source"] == "template" and "Verified from the data" in a["answer"]
    rep = c.get(f"/api/cases/{tid}/report").text
    assert "Evidence timeline" in rep and "Related open cases" in rep
    assert c.get("/api/admin/drift").status_code == 403
    login(c, "admin")
    d = c.get("/api/admin/drift").json()
    assert "psi" in d and "feedback" in d and d["feedback"]["criteria"]
    m = c.get("/api/admin/model").json()
    assert "v21" in m


@needs_models
def test_smaller_payment_to_a_payee_just_verified_on_this_device_goes_through(client):
    from webapp.server import ENGINE
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    login(c, custs[-3])
    first = c.post("/api/customer/pay", headers=H, json={"payee": "sai_nippun", "amount": 23455, "device": "this_browser"}).json()
    assert first["status"] == "verification_required"
    c.get(f"/api/customer/payments/{first['txn_id']}/captcha")
    assert c.post(f"/api/customer/payments/{first['txn_id']}/respond", headers=H,
                  json={"choice": "proceed", "captcha": ENGINE.captchas.peek(first["txn_id"])}).json()["status"] == "completed"
    small = c.post("/api/customer/pay", headers=H, json={"payee": "sai_nippun", "amount": 23, "device": "this_browser"}).json()
    assert small["decision"] == "ALLOW" and small["status"] == "completed", small["message"]
    # a LARGER amount than the one verified is still checked
    big = c.post("/api/customer/pay", headers=H, json={"payee": "sai_nippun", "amount": 49000, "device": "this_browser"}).json()
    assert big["decision"] in ("REVIEW", "BLOCK")


@needs_models
def test_customer_query_on_blocked_payment_and_admin_decision(client):
    c = client
    custs = [a["username"] for a in c.get("/api/demo-accounts").json()["accounts"] if a["role"] == "customer"]
    login(c, custs[-4])
    blocked = c.post("/api/customer/pay", headers=H, json={"payee": "SIM_UNKNOWN_BIG", "amount": 95000, "device": "this_browser"}).json()
    assert blocked["status"] == "held"
    tid = blocked["txn_id"]
    assert c.post("/api/customer/queries/new", headers=H, json={"txn_id": tid, "reason": "short"}).status_code == 400
    r = c.post("/api/customer/queries/new", headers=H, json={"txn_id": tid, "reason": "This is my landlord, I pay rent to him every year."})
    assert r.status_code == 200 and r.json()["status"] == "open"
    qid = r.json()["id"]
    assert c.post("/api/customer/queries/new", headers=H, json={"txn_id": tid, "reason": "Again, it is my landlord, rent."}).status_code == 409
    assert c.get("/api/customer/queries").json()["queries"][0]["status_text"] == "Under review"
    assert c.get("/api/admin/queries").status_code == 403
    login(c, "analyst1")
    assert c.post(f"/api/admin/queries/{qid}", headers=H, json={"decision": "approve", "response": "ok fine"}).status_code == 403
    login(c, "admin")
    assert any(q["id"] == qid for q in c.get("/api/admin/queries").json()["queries"])
    assert c.post(f"/api/admin/queries/{qid}", headers=H, json={"decision": "approve", "response": ""}).status_code == 400
    d = c.post(f"/api/admin/queries/{qid}", headers=H, json={"decision": "approve", "response": "Verified with the customer by phone."}).json()
    assert d["status"] == "approved" and d["payment_status"] == "released"
    assert c.post(f"/api/admin/queries/{qid}", headers=H, json={"decision": "reject", "response": "changed my mind"}).status_code == 409
    login(c, custs[-4])
    q = c.get("/api/customer/queries").json()["queries"][0]
    assert q["status"] == "approved" and q["payment_status"] == "released" and "phone" in q["response"]
    # a query on a payment that isn't blocked is refused
    ok = c.post("/api/customer/pay", headers=H, json={"payee": c.get("/api/me").json()["profile"]["saved_payees"][0], "amount": 50, "device": "this_browser"}).json()
    if ok["status"] != "held":
        assert c.post("/api/customer/queries/new", headers=H, json={"txn_id": ok["txn_id"], "reason": "just testing a query here"}).status_code == 409


def test_former_env_names_still_work(tmp_path, monkeypatch):
    """The project was renamed; a .env written with the former variable prefix must keep working."""
    import os
    import src.settings as st
    old = "FRAUD" + "SHIELD_"
    monkeypatch.setattr(st, "ROOT", tmp_path)
    monkeypatch.delenv("PAYTRACE_ALIAS_TEST", raising=False)
    (tmp_path / ".env").write_text(old + "ALIAS_TEST=42\n")
    st._load_dotenv()
    assert os.environ.get("PAYTRACE_ALIAS_TEST") == "42"
    monkeypatch.delenv("PAYTRACE_ALIAS_TEST")
