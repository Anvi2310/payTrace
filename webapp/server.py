"""payTrace web application: JSON API + single-page web front end.

    uvicorn webapp.server:app --host 127.0.0.1 --port 8000     (or ./run_app.sh)
    open http://localhost:8000

Roles: customer (send money, confirm or deny flagged payments, see own history)
       analyst  (case queue, investigation, copilot, decisions, check any payment)
       admin    (everything + overview, simulated traffic, model evidence, LLM status, users, audit)
Security: PBKDF2 password hashes, signed HttpOnly SameSite=Strict session cookie, custom request header required on
every state-changing call (CSRF), server-side role checks on every route, login throttling, security headers,
input validation in the scorer, no secrets in code (.env).
"""
from __future__ import annotations

import base64
import contextlib
import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import string
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from src import llm
from src.config import REPORTS_DIR, ROOT
from src.scorer_v2 import InvalidTransaction
from src.store import audit, connect
from webapp.engine import FIRST_NAMES, AppEngine, read_dataset
from webapp.replay import REPLAY
from src.config import MODELS_DIR as MODEL_ROOT

STATIC = Path(__file__).parent / "static"
COOKIE = "fs_session"
SESSION_HOURS = 8
ENGINE = AppEngine()          # dataset/model: PAYTRACE_APP_DATASET (default: the Sparkov-trained model when present)
_fails: dict[str, list[float]] = defaultdict(list)


# ---------------- secrets, passwords, sessions ----------------
def _secret() -> bytes:
    env = os.getenv("PAYTRACE_SECRET")
    if env:
        return env.encode()
    f = ROOT / ".secret_key"
    if not f.exists():
        f.write_text(secrets.token_hex(32))
        f.chmod(0o600)
    return f.read_text().strip().encode()


SECRET = _secret()


def hash_pw(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def make_token(user: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"u": user["username"], "r": user["role"],
                                                "exp": time.time() + SESSION_HOURS * 3600}).encode()).decode()
    sig = hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_token(tok: str | None) -> dict | None:
    if not tok or "." not in tok:
        return None
    body, sig = tok.rsplit(".", 1)
    if not hmac.compare_digest(sig, hmac.new(SECRET, body.encode(), hashlib.sha256).hexdigest()):
        return None
    try:
        d = json.loads(base64.urlsafe_b64decode(body))
    except Exception:
        return None
    return d if d.get("exp", 0) > time.time() else None


def demo_password() -> str | None:
    """The configured demo password, or None when unset/blank (then the public default 'demo1234' is used)."""
    return os.getenv("PAYTRACE_DEMO_PASSWORD") or None


def _migrate_users(con):
    cols = {r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()}
    if "demo" not in cols:                      # databases created by an earlier build
        con.execute("ALTER TABLE users ADD COLUMN demo INTEGER DEFAULT 0")
        con.execute("UPDATE users SET demo=1")  # every account that existed then was a seeded demo account
        con.commit()


def insert_user(con, username, role, display_name, customer_id, password, demo=0):
    salt = secrets.token_hex(16)
    con.execute("INSERT INTO users (username, role, display_name, customer_id, salt, password_hash, created, demo) "
                "VALUES (?,?,?,?,?,?,?,?)", (username, role, display_name, customer_id, salt, hash_pw(password, salt),
                                            time.time(), demo))


def set_password(con, username, password):
    salt = secrets.token_hex(16)
    con.execute("UPDATE users SET salt=?, password_hash=? WHERE username=?", (salt, hash_pw(password, salt), username))


def random_password(n: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits + "@#%-_"
    while True:
        p = "".join(secrets.choice(alphabet) for _ in range(n))
        if any(c.islower() for c in p) and any(c.isupper() for c in p) and any(c.isdigit() for c in p):
            return p


def seed_users():
    """Demo accounts (created once, flagged demo=1). Password from PAYTRACE_DEMO_PASSWORD, default 'demo1234'."""
    con = connect()
    try:
        _migrate_users(con)
        pw = demo_password() or "demo1234"
        # repair: no account may ever have an empty password (an earlier build could create them from a blank .env line)
        fixed = 0
        for u in con.execute("SELECT username, salt, password_hash FROM users").fetchall():
            if hmac.compare_digest(u["password_hash"], hash_pw("", u["salt"])):
                set_password(con, u["username"], pw)
                fixed += 1
        if fixed:
            con.commit()
            audit(con, "system", "system", "reset_empty_passwords", "", {"accounts": fixed})
        if not con.execute("SELECT 1 FROM users WHERE username='demo'").fetchone():
            insert_user(con, "demo", "admin", "Demo Admin", None, pw, demo=1)    # one quick login for presentations
            con.commit()
        rows = [("admin", "admin", "Admin", None), ("analyst1", "analyst", "Analyst One", None),
                ("analyst2", "analyst", "Analyst Two", None)]
        for username, role, name, cid in rows:
            if not con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                insert_user(con, username, role, name, cid, pw, demo=1)
        # demo customers are real customers of the dataset the app runs on; when the dataset changes, the old demo
        # customer accounts are replaced (self-registered customers are never touched)
        df = read_dataset(ENGINE.dataset)[["user_id", "is_fraud"]]
        users = set(df.user_id.unique())
        for r in con.execute("SELECT username, customer_id FROM users WHERE demo=1 AND role='customer'").fetchall():
            if r["customer_id"] not in users:
                con.execute("DELETE FROM users WHERE username=?", (r["username"],))
        if not con.execute("SELECT 1 FROM users WHERE demo=1 AND role='customer'").fetchone():
            good = df.groupby("user_id").is_fraud.agg(["size", "sum"])
            good = good[good["sum"] == 0].sort_values("size", ascending=False).head(12)
            for i, uid in enumerate(good.index):
                if not con.execute("SELECT 1 FROM users WHERE username=?", (uid.lower(),)).fetchone():
                    insert_user(con, uid.lower(), "customer", f"{FIRST_NAMES[i % len(FIRST_NAMES)]} ({uid})", uid, pw, demo=1)
        con.commit()
    finally:
        con.close()


def current_user(request: Request) -> dict | None:
    t = read_token(request.cookies.get(COOKIE))
    if not t:
        return None
    con = connect()
    try:
        u = con.execute("SELECT username, role, display_name, customer_id FROM users WHERE username=?", (t["u"],)).fetchone()
        return dict(u) if u else None
    finally:
        con.close()


def err(msg, code=400):
    return JSONResponse({"error": msg}, status_code=code)


def guard(*roles):
    """Decorator: require login (and one of `roles`), plus the CSRF header on state-changing methods."""
    def deco(fn):
        async def wrapper(request: Request):
            if request.method in ("POST", "PUT", "DELETE") and request.headers.get("x-fs-request") != "1":
                return err("missing request header", 403)
            u = current_user(request)
            if u is None:
                return err("not signed in", 401)
            if roles and u["role"] not in roles:
                return err("not allowed for your role", 403)
            request.state.user = u
            return await fn(request)
        return wrapper
    return deco


def need_engine():
    if not ENGINE.ready:
        return err(f"engine {ENGINE.progress}" + (f": {ENGINE.error}" if ENGINE.error else ""), 503)
    return None


async def body(request: Request) -> dict:
    try:
        d = await request.json()
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


# ---------------- public ----------------
async def index(request):
    return FileResponse(STATIC / "index.html")


async def health(request):
    return JSONResponse({"ready": ENGINE.ready, "progress": ENGINE.progress, "error": ENGINE.error,
                         "llm": llm.status(), "dataset": ENGINE.dataset, "bank_time": ENGINE.now().strftime("%Y-%m-%d %H:%M:%S") if ENGINE.ready else None,
                         "demo_accounts": demo_password() is None})


async def login(request):
    if request.headers.get("x-fs-request") != "1":
        return err("missing request header", 403)
    d = await body(request)
    name = str(d.get("username", "")).strip().lower()[:64]
    recent = [t for t in _fails[name] if time.time() - t < 300]
    _fails[name] = recent
    if len(recent) >= 5:
        return err("too many attempts; wait 5 minutes", 429)
    con = connect()
    try:
        u = con.execute("SELECT * FROM users WHERE username=?", (name,)).fetchone()
        ok = u is not None and str(d.get("password", "")) != "" and hmac.compare_digest(u["password_hash"], hash_pw(str(d.get("password", "")), u["salt"]))
        if not ok:
            _fails[name].append(time.time())
            audit(con, name or "?", "-", "login_failed")
            return err("wrong username or password", 401)
        audit(con, u["username"], u["role"], "login")
        user = {"username": u["username"], "role": u["role"], "display_name": u["display_name"], "customer_id": u["customer_id"]}
    finally:
        con.close()
    return _session_response(user)


def _session_response(user: dict, status: int = 200) -> JSONResponse:
    r = JSONResponse({"user": user}, status_code=status)
    r.set_cookie(COOKIE, make_token(user), max_age=SESSION_HOURS * 3600, httponly=True, samesite="strict",
                 secure=os.getenv("PAYTRACE_HTTPS") == "1")
    return r


USERNAME_RE = re.compile(r"^[a-z0-9_.]{3,32}$")
_signups: dict[str, list[float]] = defaultdict(list)


def _check_password(pw: str) -> str | None:
    if len(pw) < 8 or len(pw) > 128:
        return "password must be 8 to 128 characters"
    if not (re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw)):
        return "password must contain letters and numbers"
    return None


def _new_customer_id(con) -> str:
    taken = {r[0] for r in con.execute("SELECT customer_id FROM users WHERE customer_id IS NOT NULL")}
    known = set(ENGINE.profiles.index) if ENGINE.ready else set()
    while True:
        cid = f"U9{secrets.randbelow(10**5):05d}"          # new-customer range, never used by the bank's history
        if cid not in taken and cid not in known:
            return cid


async def signup(request):
    """Self-service registration. Always creates a CUSTOMER account (staff accounts are created by an admin)."""
    if request.headers.get("x-fs-request") != "1":
        return err("missing request header", 403)
    ip = request.client.host if request.client else "?"
    recent = [t for t in _signups[ip] if time.time() - t < 3600]
    _signups[ip] = recent
    if len(recent) >= 10:
        return err("too many accounts created from this address; try again later", 429)
    d = await body(request)
    username = str(d.get("username", "")).strip().lower()
    name = " ".join(str(d.get("display_name", "")).split())[:60]
    pw = str(d.get("password", ""))
    if not USERNAME_RE.match(username):
        return err("username: 3-32 characters, lowercase letters, numbers, dot or underscore", 422)
    if not name:
        return err("please enter your name", 422)
    if (m := _check_password(pw)):
        return err(m, 422)
    con = connect()
    try:
        if con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return err("that username is taken", 409)
        cid = _new_customer_id(con)
        insert_user(con, username, "customer", name, cid, pw)
        con.commit()
        audit(con, username, "customer", "signup", cid)
    finally:
        con.close()
    _signups[ip].append(time.time())
    return _session_response({"username": username, "role": "customer", "display_name": name, "customer_id": cid}, 201)


@guard()
async def change_password(request):
    u = request.state.user
    d = await body(request)
    current, new = str(d.get("current_password", "")), str(d.get("new_password", ""))
    if (m := _check_password(new)):
        return err(m, 422)
    con = connect()
    try:
        row = con.execute("SELECT salt, password_hash FROM users WHERE username=?", (u["username"],)).fetchone()
        if not current or not hmac.compare_digest(row["password_hash"], hash_pw(current, row["salt"])):
            return err("current password is wrong", 403)
        set_password(con, u["username"], new)
        con.execute("UPDATE users SET demo=0 WHERE username=?", (u["username"],))   # no longer on the public demo list
        con.commit()
        audit(con, u["username"], u["role"], "password_changed", u["username"])
    finally:
        con.close()
    return JSONResponse({"ok": True})


async def logout(request):
    r = JSONResponse({"ok": True})
    r.delete_cookie(COOKIE)
    return r


async def demo_accounts(request):
    if demo_password() is not None:
        return JSONResponse({"accounts": []})          # custom password set: don't advertise accounts
    con = connect()
    try:
        rows = con.execute("SELECT username, role, display_name FROM users WHERE demo=1 ORDER BY role, username").fetchall()
    finally:
        con.close()
    return JSONResponse({"password": "demo1234", "accounts": [dict(r) for r in rows]})


@guard()
async def me(request):
    u = request.state.user
    out = {"user": u, "options": ENGINE.options()}
    if u["role"] == "customer" and ENGINE.ready:
        out["profile"] = ENGINE.profile(u["customer_id"])
    return JSONResponse(out)


# ---------------- customer ----------------
def _payment_args(d: dict, uid: str, browser_device: str) -> dict:
    try:
        amount = float(d.get("amount"))
    except (TypeError, ValueError):
        raise InvalidTransaction("amount must be a number")
    payee = str(d.get("payee", "")).strip()[:64]
    if not payee:
        raise InvalidTransaction("payee is required")
    cat = str(d.get("category") or "")                    # replaced by the merchant's own category when known
    ch = str(d.get("channel") or "")
    prof = ENGINE.profile(uid)
    city = " ".join(str(d.get("city") or "").split())[:60] or prof["usual_city"] or "UNKNOWN"
    dev_choice = d.get("device", "this_browser")
    if dev_choice == "usual" and prof["usual_device"]:
        device = prof["usual_device"]
    elif dev_choice == "custom" and str(d.get("custom_device", "")).strip():
        device = str(d["custom_device"]).strip()[:64]
    else:
        # one presenter's laptop is shared by every demo account, which is itself a fraud-ring signal (one device,
        # many accounts); in a real app each customer's phone is their own, so the browser id is scoped per account.
        # Use "custom" with the same device id on several accounts to show a real shared-device attack.
        device = browser_device if browser_device == "UNKNOWN" else f"{browser_device}_{hashlib.sha1(uid.encode()).hexdigest()[:6]}"
    return {"user_id": uid, "payee": payee, "amount": amount, "category": cat, "channel": ch, "device_id": device, "city": city}


def _browser_device(request: Request) -> str:
    raw = request.headers.get("x-fs-device", "")
    ok = raw.replace("-", "").isalnum() and 8 <= len(raw) <= 40
    return f"D_WEB_{raw[:24]}" if ok else "UNKNOWN"


@guard("customer")
async def customer_pay(request):
    if (r := need_engine()):
        return r
    u = request.state.user
    try:
        args = _payment_args(await body(request), u["customer_id"], _browser_device(request))
        res = ENGINE.submit_payment(**args, source="customer", actor=u["username"], role="customer")
    except InvalidTransaction as e:
        return err(str(e), 422)
    for k in ("reasons", "risk_breakdown", "tripwires", "risk", "policy", "model_decision", "scores", "features_view",
              "ring", "familiar_payee_override"):   # customers never see internals
        res.pop(k, None)
    return JSONResponse(res)


@guard("customer")
async def customer_payments(request):
    u = request.state.user
    con = connect()
    try:
        rows = con.execute("SELECT txn_id, ts, amount, payee, category, channel, decision, status, customer_message, "
                           "message_source, device_id, city FROM payments WHERE user_id=? ORDER BY created DESC LIMIT 50",
                           (u["customer_id"],)).fetchall()
    finally:
        con.close()
    hist = ENGINE.history_for(u["customer_id"], 10) if ENGINE.ready else []
    return JSONResponse({"payments": [dict(r) for r in rows], "earlier_activity": hist})


# ---------------- queries on blocked payments ----------------
# A customer who believes a BLOCKED payment was genuine can raise a query with their reasons. It does not send the
# money: an admin reviews the payment's evidence together with the reasons and decides. Approving releases the
# payment and records a 'genuine' label (the same path as an analyst release); rejecting keeps it blocked for good.
QUERY_STATUS = {"open": "Under review", "approved": "Approved: payment released", "rejected": "Rejected: payment stays blocked"}


def _query_row(r) -> dict:
    d = dict(r)
    d["status_text"] = QUERY_STATUS.get(d["status"], d["status"])
    for k in ("created", "updated"):
        d[k] = time.strftime("%Y-%m-%d %H:%M", time.localtime(d[k])) if d.get(k) else None
    return d


@guard("customer")
async def customer_query_create(request):
    u = request.state.user
    d = await body(request)
    tid = str(d.get("txn_id", ""))
    reason = " ".join(str(d.get("reason", "")).split())
    if len(reason) < 10:
        return err("please explain in a sentence or two why this payment is genuine (at least 10 characters)", 400)
    con = connect()
    try:
        p = con.execute("SELECT status FROM payments WHERE txn_id=? AND user_id=?", (tid, u["customer_id"])).fetchone()
        if p is None:
            return err("payment not found", 404)
        if p["status"] != "held":
            return err("a query can only be raised on a blocked payment", 409)
        if con.execute("SELECT 1 FROM queries WHERE txn_id=? AND status='open'", (tid,)).fetchone():
            return err("there is already an open query for this payment", 409)
        if con.execute("SELECT COUNT(*) FROM queries WHERE user_id=? AND created>?", (u["customer_id"], time.time() - 86400)).fetchone()[0] >= 10:
            return err("too many queries today; please contact the bank", 429)
        now = time.time()
        cur = con.execute("INSERT INTO queries (txn_id, user_id, username, reason, status, created, updated) VALUES (?,?,?,?,?,?,?)",
                          (tid, u["customer_id"], u["username"], reason[:1000], "open", now, now))
        con.commit()
        audit(con, u["username"], "customer", "query_raised", tid)
        row = con.execute("SELECT * FROM queries WHERE id=?", (cur.lastrowid,)).fetchone()
        return JSONResponse(_query_row(row))
    finally:
        con.close()


@guard("customer")
async def customer_queries(request):
    u = request.state.user
    con = connect()
    try:
        rows = con.execute("SELECT q.*, p.amount, p.payee, p.ts AS payment_ts, p.status AS payment_status FROM queries q "
                           "JOIN payments p ON p.txn_id=q.txn_id WHERE q.user_id=? ORDER BY q.created DESC", (u["customer_id"],)).fetchall()
        return JSONResponse({"queries": [_query_row(r) for r in rows]})
    finally:
        con.close()


@guard("admin")
async def admin_queries(request):
    con = connect()
    try:
        rows = con.execute("SELECT q.*, p.amount, p.payee, p.ts AS payment_ts, p.status AS payment_status, p.risk, p.decision, "
                           "p.customer_message FROM queries q JOIN payments p ON p.txn_id=q.txn_id "
                           "ORDER BY CASE q.status WHEN 'open' THEN 0 ELSE 1 END, q.created DESC LIMIT 300").fetchall()
        counts = dict(con.execute("SELECT status, COUNT(*) FROM queries GROUP BY status").fetchall())
        return JSONResponse({"queries": [_query_row(r) for r in rows], "counts": counts})
    finally:
        con.close()


@guard("admin")
async def admin_query_decide(request):
    qid = int(request.path_params["qid"])
    d = await body(request)
    decision, response = d.get("decision"), " ".join(str(d.get("response", "")).split())
    if decision not in ("approve", "reject"):
        return err("decision must be approve or reject", 400)
    if len(response) < 5:
        return err("write a short response to the customer explaining the decision", 400)
    admin = request.state.user["username"]
    con = connect()
    try:
        q = con.execute("SELECT * FROM queries WHERE id=?", (qid,)).fetchone()
        if q is None:
            return err("query not found", 404)
        if q["status"] != "open":
            return err("this query has already been decided", 409)
    finally:
        con.close()
    if decision == "approve":
        # same path as an analyst release: genuine label, payment released (sent)
        res = ENGINE.analyst_decide(q["txn_id"], admin, "admin", 0, f"query #{qid} approved: {response}")
        pay_status = res["payment_status"]
    else:
        pay_status = "blocked_final"
    con = connect()
    try:
        if decision == "reject":
            con.execute("UPDATE payments SET status='blocked_final', updated=?, resolved_by=?, resolution_note=? WHERE txn_id=?",
                        (time.time(), admin, f"query #{qid} rejected: {response}"[:300], q["txn_id"]))
        con.execute("UPDATE queries SET status=?, admin=?, response=?, updated=? WHERE id=?",
                    ("approved" if decision == "approve" else "rejected", admin, response[:1000], time.time(), qid))
        con.commit()
        audit(con, admin, "admin", f"query_{decision}", q["txn_id"], {"query": qid})
        row = con.execute("SELECT * FROM queries WHERE id=?", (qid,)).fetchone()
        return JSONResponse({**_query_row(row), "payment_status": pay_status})
    finally:
        con.close()


@guard("customer")
async def customer_respond(request):
    u = request.state.user
    d = await body(request)
    choice = d.get("choice")
    if choice is None and "confirmed" in d:                  # older clients: confirmed true/false
        choice = "proceed" if d.get("confirmed") else "cancel"
    try:
        res = ENGINE.customer_respond(request.path_params["txn_id"], u["customer_id"], str(choice or ""), d.get("captcha"))
    except KeyError as e:
        return err(str(e), 404)
    except PermissionError as e:
        return err(str(e), 403)
    except ValueError as e:
        return err(str(e), 409)
    return JSONResponse(res)


@guard("customer")
async def customer_captcha(request):
    """A fresh security-check image for one of the customer's own flagged payments (each load = a new challenge)."""
    from webapp.captcha import svg
    u, tid = request.state.user, request.path_params["txn_id"]
    con = connect()
    try:
        p = con.execute("SELECT status FROM payments WHERE txn_id=? AND user_id=?", (tid, u["customer_id"])).fetchone()
    finally:
        con.close()
    if p is None:
        return err("payment not found", 404)
    if p["status"] != "verification_required":            # blocked payments have no customer choice
        return err("nothing to confirm for this payment", 409)
    return Response(svg(ENGINE.captchas.issue(tid)), media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store, max-age=0"})


# ---------------- analyst ----------------
@guard("analyst", "admin")
async def cases(request):
    status = request.query_params.get("status", "open")
    source = request.query_params.get("source", "all")
    q = ("SELECT e.txn_id, e.ts, e.user_id, e.amount, e.merchant_id AS payee, e.category, e.decision, e.risk, e.tripwires, "
         "COALESCE(p.status, 'review') AS status, COALESCE(p.source, 'simulator') AS source, "
         "(SELECT f.status FROM feedback f WHERE f.txn_id=e.txn_id ORDER BY f.wall_time DESC LIMIT 1) AS label_status "
         "FROM events e LEFT JOIN payments p ON p.txn_id=e.txn_id WHERE e.decision!='ALLOW'")
    params = []
    if status == "open":
        q += " AND COALESCE(p.status,'review') IN ('release_requested','held','review','verification_required','cancelled')"
    if source != "all":
        q += " AND COALESCE(p.source,'simulator')=?"; params.append(source)
    q += (" ORDER BY (p.status='release_requested') DESC, (COALESCE(p.source,'simulator')!='simulator') DESC, "
          "e.risk DESC LIMIT 300")
    con = connect()
    try:
        rows = [dict(r) for r in con.execute(q, params).fetchall()]
        counts = dict(con.execute("SELECT COALESCE(p.status,'review'), COUNT(*) FROM events e LEFT JOIN payments p "
                                  "ON p.txn_id=e.txn_id WHERE e.decision!='ALLOW' GROUP BY 1").fetchall())
    finally:
        con.close()
    return JSONResponse({"cases": rows, "counts": counts})


def _network(txn_id, ts, seeds):
    from src.graph_intel import build_graph, ego, entity_findings, layout
    con = connect()
    try:
        since = (pd.Timestamp(ts[:19]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        ev = pd.read_sql("SELECT txn_id, ts, user_id, device_id, merchant_id, amount, risk AS probability, decision FROM events "
                         "WHERE ts>=? AND ts<=?", con, params=(since, ts))
    finally:
        con.close()
    G = build_graph(ev)
    con = connect()
    try:
        conf_tx = {r[0] for r in con.execute("SELECT txn_id FROM feedback WHERE label=1 AND status='accepted' UNION "
                                             "SELECT txn_id FROM payments WHERE status='confirmed_fraud'")}
    finally:
        con.close()
    from src.graph_intel import node_of_payee
    conf_nodes = set()
    for r in ev[ev.txn_id.isin(conf_tx)].itertuples():
        conf_nodes |= {f"U:{r.user_id}", node_of_payee(r.merchant_id), f"D:{r.device_id}"}
    H = ego(G, seeds, radius=2, max_nodes=45)
    if H.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}
    pos = layout(H)
    nodes = [{"id": n, "kind": a["kind"], "flagged": a["n_flagged"], "confirmed": int(n in conf_nodes), "risk": round(float(a["max_risk"]), 3),
              "x": float(pos[n][0]), "y": float(pos[n][1]), "seed": n in seeds, "findings": entity_findings(G, n)}
             for n, a in H.nodes(data=True)]
    edges = [{"a": a, "b": b, "rel": d.get("rel"), "flagged": int(d.get("flagged", 0)), "count": int(d.get("count", 0))}
             for a, b, d in H.edges(data=True)]
    return {"nodes": nodes, "edges": edges}


# ---------------- fraud-ring intelligence ----------------
_RING_CACHE = {"t": 0.0, "key": None, "report": None}


def _ring_events(days: float = 7.0, until: str | None = None):
    con = connect()
    try:
        last = until or (con.execute("SELECT MAX(ts) FROM events").fetchone()[0] or "1970-01-01 00:00:00")
        since = (pd.Timestamp(last[:19]) - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        ev = pd.read_sql("SELECT txn_id, ts, user_id, device_id, merchant_id, amount, risk AS probability, decision "
                         "FROM events WHERE ts>=? AND ts<=?", con, params=(since, last))
        conf = {r[0] for r in con.execute("SELECT txn_id FROM feedback WHERE label=1 AND status='accepted'")}
        conf |= {r[0] for r in con.execute("SELECT txn_id FROM payments WHERE status='confirmed_fraud'")}
    finally:
        con.close()
    ev["confirmed"] = ev.txn_id.isin(conf).astype(int)
    return ev


def ring_report(days: float = 7.0, max_age: float = 20.0):
    from src.fraud_rings import analyse
    con = connect()
    try:
        key = (days, con.execute("SELECT COUNT(*), MAX(seq) FROM events").fetchone()[:],
               con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])
    finally:
        con.close()
    if _RING_CACHE["report"] is not None and _RING_CACHE["key"] == key and time.time() - _RING_CACHE["t"] < max_age:
        return _RING_CACHE["report"]
    ev = _ring_events(days)
    rep = analyse(ev.tail(60000))
    _RING_CACHE.update(t=time.time(), key=key, report=rep)
    return rep


@guard("analyst", "admin")
async def rings(request):
    days = min(max(float(request.query_params.get("days", 7)), 0.5), 30)
    rep = await asyncio.to_thread(ring_report, days)
    return JSONResponse(rep)


@guard("analyst", "admin")
async def graph_node(request):
    """Neighbourhood of one entity for the interactive graph explorer (click a node to expand it)."""
    from src.fraud_rings import _simple, build_temporal_graph, node_rings
    from src.graph_intel import ego, entity_findings, layout
    node = request.query_params.get("id", "")
    radius = 1 if request.query_params.get("radius") == "1" else 2
    ev = await asyncio.to_thread(_ring_events, 7.0)
    H = _simple(build_temporal_graph(ev))
    if node not in H:
        return err("entity not found in the last 7 days of payments", 404)
    for n, a in H.nodes(data=True):                       # graph_intel helpers expect these attributes
        a.setdefault("amount", 0.0)
    E = ego(H, [node], radius=radius, max_nodes=40)
    pos = layout(E)
    rep = _RING_CACHE["report"] or {}
    return JSONResponse({
        "center": node,
        "nodes": [{"id": n, "kind": a["kind"], "flagged": int(a["n_flagged"]), "confirmed": int(a["n_confirmed"]),
                   "risk": round(float(a["max_risk"]), 3), "x": float(pos[n][0]), "y": float(pos[n][1]),
                   "seed": n == node, "findings": entity_findings(H, n)} for n, a in E.nodes(data=True)],
        "edges": [{"a": a, "b": b, "rel": d.get("rel"), "flagged": int(d.get("flagged", 0)),
                   "confirmed": int(d.get("confirmed", 0)), "count": int(d.get("count", 0))} for a, b, d in E.edges(data=True)],
        "rings": [{k: r[k] for k in ("id", "score", "size", "status", "evidence", "hubs")} for r in node_rings(rep, node)]})


def _timeline(t: dict, labels: list[dict]) -> list[dict]:
    """Evidence timeline around one case, from stored events only: the customer's payments in the 48 h before it,
    other payments to the same payee or from the same device in that window, the case itself and analyst labels."""
    con = connect()
    try:
        since = (pd.Timestamp(t["ts"][:19]) - pd.Timedelta(hours=48)).strftime("%Y-%m-%d %H:%M:%S")
        rows = con.execute(
            "SELECT txn_id, ts, user_id, merchant_id, device_id, amount, decision, risk FROM events WHERE ts>=? AND ts<=? AND "
            "(user_id=? OR merchant_id=? OR (device_id=? AND device_id NOT IN ('UNKNOWN','','nan'))) ORDER BY ts DESC LIMIT 40",
            (since, t["ts"], t["user_id"], t["merchant_id"], t["device_id"])).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        why = ("this case" if r["txn_id"] == t["txn_id"] else "same customer" if r["user_id"] == t["user_id"]
               else "same payee" if r["merchant_id"] == t["merchant_id"] else "same device")
        out.append({"ts": r["ts"][:19], "kind": why, "txn_id": r["txn_id"], "text": f"{r['user_id']} → {r['merchant_id']}",
                    "amount": float(r["amount"]), "decision": r["decision"], "risk": float(r["risk"] or 0)})
    for l in labels:
        out.append({"ts": str(l.get("time") or "")[:19], "kind": "analyst label", "txn_id": t["txn_id"],
                    "text": f"{l['analyst']} marked it {'fraud' if l['label'] else 'genuine'} ({l['status']})" + (f": {l['note']}" if l.get("note") else ""),
                    "amount": None, "decision": None, "risk": None, "wall_clock": True})
    return out


@guard("analyst", "admin")
async def case_detail(request):
    from src.graph_intel import node_of_payee
    from src.investigation import evidence_pack
    tid = request.path_params["txn_id"]
    con = connect()
    try:
        try:
            pack = evidence_pack(con, tid, history=ENGINE.history if ENGINE.ready else None)
        except KeyError:
            return err("case not found", 404)
        p = con.execute("SELECT status, source, customer_message, resolved_by, resolution_note FROM payments WHERE txn_id=?",
                        (tid,)).fetchone()
        fb = [dict(r) for r in con.execute("SELECT analyst, label, status, note, datetime(wall_time,'unixepoch') AS time "
                                           "FROM feedback WHERE txn_id=?", (tid,)).fetchall()]
    finally:
        con.close()
    t = pack["transaction"]
    seeds = [f"U:{t['user_id']}", f"D:{t['device_id']}", node_of_payee(t["merchant_id"])]
    pack.pop("replay_ground_truth", None)
    try:
        from src.fraud_rings import node_rings
        rep = ring_report()
        case_rings = {r["id"]: {k: r[k] for k in ("id", "score", "size", "status", "evidence", "hubs", "confirmed_entities",
                                                   "flagged_payments", "customers", "devices", "payees")}
                      for sd in seeds for r in node_rings(rep, sd)}
    except Exception:                                      # ring analysis is optional evidence, never blocks a case
        case_rings = {}
    return JSONResponse({"pack": pack, "payment": dict(p) if p else {"status": "review", "source": "simulator"},
                         "rings": list(case_rings.values()), "timeline": _timeline(t, fb),
                         "labels": fb, "network": _network(tid, t["ts"], seeds),
                         "customer_recent": ENGINE.history_for(t["user_id"], 8) if ENGINE.ready else []}, )


@guard("analyst", "admin")
async def case_summary(request):
    from src.investigation import evidence_pack, summarise
    tid = request.path_params["txn_id"]
    con = connect()
    try:
        pack = evidence_pack(con, tid, history=ENGINE.history if ENGINE.ready else None)
        text, source = summarise(pack)
        audit(con, request.state.user["username"], request.state.user["role"], "copilot_summary", tid, {"source": source})
    except KeyError:
        return err("case not found", 404)
    finally:
        con.close()
    return JSONResponse({"summary": text, "source": source})


@guard("analyst", "admin")
async def case_report(request):
    from src.investigation import case_report as build, evidence_pack, summarise
    tid = request.path_params["txn_id"]
    con = connect()
    try:
        pack = evidence_pack(con, tid, history=ENGINE.history if ENGINE.ready else None)
        text, _ = summarise(pack, use_llm=request.query_params.get("llm") == "1")
        md = build(pack, text, request.state.user["username"])
        try:
            from src.copilot_qa import answer, to_markdown
            ctx = _case_context(tid)
            md += "\n\n## Fraud-ring and network evidence\n\n" + to_markdown(answer("beneficiary", ctx))
            md += "\n\n## Evidence timeline (48 h before the case)\n\n" + "\n".join(
                f"- {x['ts']} · {x['kind']} · {x['text']}" + (f" · Rs {x['amount']:,.0f} · {x['decision']}" if x["amount"] is not None else "")
                for x in ctx["timeline"]) + "\n\n## Related open cases\n\n" + to_markdown(answer("next", ctx))
        except Exception as ex:                             # extra sections are best-effort; the core report stands
            md += f"\n\n(Network sections unavailable: {type(ex).__name__})"
        audit(con, request.state.user["username"], request.state.user["role"], "case_report", tid)
    except KeyError:
        return err("case not found", 404)
    finally:
        con.close()
    return PlainTextResponse(md, headers={"Content-Disposition": f'attachment; filename="case_{tid}.md"'},
                             media_type="text/markdown")


def _case_context(tid: str) -> dict:
    """Everything the copilot may use for one case: the evidence pack, rings, timeline and related open cases."""
    from src.fraud_rings import node_rings
    from src.graph_intel import node_of_payee
    from src.investigation import evidence_pack
    con = connect()
    try:
        pack = evidence_pack(con, tid, history=ENGINE.history if ENGINE.ready else None)
        fb = [dict(r) for r in con.execute("SELECT analyst, label, status, note, datetime(wall_time,'unixepoch') AS time "
                                           "FROM feedback WHERE txn_id=?", (tid,)).fetchall()]
        t = pack["transaction"]
        since = (pd.Timestamp(t["ts"][:19]) - pd.Timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        rows = con.execute(
            "SELECT e.txn_id, e.ts, e.user_id, e.merchant_id, e.device_id, e.amount, e.decision, e.risk FROM events e "
            "LEFT JOIN payments p ON p.txn_id=e.txn_id WHERE e.ts>=? AND e.decision!='ALLOW' AND e.txn_id!=? AND "
            "COALESCE(p.status,'review') IN ('review','verification_required','held','release_requested') "
            "AND e.txn_id NOT IN (SELECT txn_id FROM feedback) ORDER BY e.risk DESC LIMIT 3000", (since, tid)).fetchall()
    finally:
        con.close()
    seeds = [f"U:{t['user_id']}", f"D:{t['device_id']}", node_of_payee(t["merchant_id"])]
    rep = ring_report()
    rings = {r["id"]: r for sd in seeds for r in node_rings(rep, sd)}
    ring_nodes = set().union(*[set(r["nodes"]) for r in rings.values()]) if rings else set()
    open_cases = []
    for r in rows:
        nodes = {f"U:{r['user_id']}", node_of_payee(r["merchant_id"]), f"D:{r['device_id']}"}
        why = ("same customer" if r["user_id"] == t["user_id"] else "same payee" if r["merchant_id"] == t["merchant_id"]
               else "same device" if r["device_id"] == t["device_id"] and r["device_id"] not in ("UNKNOWN", "") else
               "same ring" if nodes & ring_nodes else None)
        if why:
            open_cases.append({"txn_id": r["txn_id"], "why": why, "amount": float(r["amount"]), "decision": r["decision"],
                               "risk": float(r["risk"] or 0), "rank": {"same ring": 0, "same payee": 1, "same device": 2, "same customer": 3}[why]})
    open_cases.sort(key=lambda o: (o["rank"], -o["risk"]))
    return {"pack": pack, "rings": list(rings.values()), "timeline": _timeline(t, fb), "open_cases": open_cases}


@guard("analyst", "admin")
async def case_ask(request):
    from src.copilot_qa import QUESTIONS, answer, answer_text, route
    tid = request.path_params["txn_id"]
    d = await body(request)
    intent = route(str(d.get("question", ""))[:300])
    try:
        ctx = await asyncio.to_thread(_case_context, tid)
    except KeyError:
        return err("case not found", 404)
    a = answer(intent, ctx)
    text, src = await asyncio.to_thread(answer_text, a, bool(d.get("llm", True)))
    con = connect(); audit(con, request.state.user["username"], request.state.user["role"], "copilot_question", tid, {"intent": intent}); con.close()
    return JSONResponse({**a, "answer": text, "source": src, "questions": QUESTIONS})


@guard("analyst", "admin")
async def case_decide(request):
    from src.feedback import FeedbackRejected
    d = await body(request)
    label = {"fraud": 1, "genuine": 0}.get(d.get("label"))
    if label is None:
        return err("label must be 'fraud' or 'genuine'")
    u = request.state.user
    try:
        res = ENGINE.analyst_decide(request.path_params["txn_id"], u["username"], u["role"], label, str(d.get("note", "")))
    except FeedbackRejected as e:
        return err(str(e), 409)
    return JSONResponse(res)


@guard("analyst", "admin")
async def check_payment(request):
    """Analyst/admin checks any payment on behalf of a customer (e.g. a call-centre request)."""
    if (r := need_engine()):
        return r
    d = await body(request)
    uid = str(d.get("customer_id", "")).strip().upper()[:64]
    if not uid:
        return err("customer_id is required", 422)
    u = request.state.user
    if uid not in ENGINE.known_customers and uid not in ENGINE._app_customers():
        return err(f"No customer {uid} in the bank's records. Use Customer lookup to find an existing customer id.", 404)
    try:
        args = _payment_args(d, uid, f"D_CHECK_{secrets.token_hex(4)}")
        res = ENGINE.submit_payment(**args, source="analyst_check", actor=u["username"], role=u["role"])
    except InvalidTransaction as e:
        return err(str(e), 422)
    return JSONResponse(res)


@guard("analyst", "admin")
async def customer_lookup(request):
    if (r := need_engine()):
        return r
    uid = request.path_params["uid"].strip().upper()
    if uid not in ENGINE.known_customers and uid not in ENGINE._app_customers():
        return err(f"No customer {uid} in the bank's records.", 404)
    con = connect()
    try:
        pays = [dict(r) for r in con.execute("SELECT txn_id, ts, amount, payee, decision, status, source FROM payments "
                                             "WHERE user_id=? ORDER BY created DESC LIMIT 30", (uid,)).fetchall()]
    finally:
        con.close()
    return JSONResponse({"profile": ENGINE.profile(uid), "history": ENGINE.history_for(uid, 20), "payments": pays})


@guard("analyst", "admin")
async def directory(request):
    """Search-as-you-type for the staff forms: customers (by id or home city) and merchants (by name)."""
    if (r := need_engine()):
        return r
    q = request.query_params.get("q", "").strip().lower()[:40]
    kind = request.query_params.get("kind", "customer")
    out = []
    if kind == "customer":
        prof = ENGINE.profiles
        ids = prof.index if not q else prof.index[prof.index.str.lower().str.contains(q, regex=False)
                                                  | prof.city.str.lower().str.contains(q, regex=False)]
        for uid in prof.loc[ids].sort_values("payments", ascending=False).index[:25]:
            p = prof.loc[uid]
            out.append({"value": uid, "label": f"{p.city} · {int(p.payments)} payments · usually ₹{p.median_amount:,.0f}"})
    else:
        names = sorted(ENGINE.merchant_category)
        hits = [m for m in names if q in m.lower()] if q else names
        out = [{"value": m, "label": ENGINE.merchant_category[m]} for m in hits[:25]]
    return JSONResponse({"results": out})


# ---------------- admin ----------------
@guard("admin")
async def admin_overview(request):
    con = connect()
    try:
        tot = dict(con.execute("SELECT decision, COUNT(*) FROM events GROUP BY decision").fetchall())
        st = dict(con.execute("SELECT status, COUNT(*) FROM payments GROUP BY status").fetchall())
        src = dict(con.execute("SELECT source, COUNT(*) FROM payments GROUP BY source").fetchall())
        held_val = con.execute("SELECT COALESCE(SUM(amount),0) FROM payments WHERE status IN ('held','confirmed_fraud','cancelled')").fetchone()[0]
        lab = con.execute("SELECT SUM(decision!='ALLOW' AND is_fraud=1), SUM(is_fraud=1), SUM(decision!='ALLOW' AND is_fraud=0), "
                          "SUM(is_fraud=0) FROM events WHERE is_fraud>=0").fetchone()
        lat = con.execute("SELECT AVG(latency_ms) FROM (SELECT latency_ms FROM events ORDER BY seq DESC LIMIT 2000)").fetchone()[0]
        fbk = dict(con.execute("SELECT status, COUNT(*) FROM feedback GROUP BY status").fetchall())
    finally:
        con.close()
    tp, pos, fp, neg = [x or 0 for x in lab]
    return JSONResponse({"decisions": tot, "payment_status": st, "sources": src, "protected_value": held_val,
                         "replay_recall": tp / pos if pos else None, "replay_fpr": fp / neg if neg else None,
                         "avg_latency_ms": lat, "feedback": fbk, "simulator": ENGINE.sim_state, "llm": llm.status(),
                         "model_version": ENGINE.scorer.meta.get("trained_at") if ENGINE.ready else None})


_SCEN = {"runner": None}


def scenario_runner():
    from webapp.scenarios import ScenarioRunner
    if _SCEN["runner"] is None:
        _SCEN["runner"] = ScenarioRunner(ENGINE)
    return _SCEN["runner"]


@guard("admin")
async def admin_drift(request):
    """Concept drift (PSI + Page-Hinkley) on scored events, and the state of the feedback-learning loop."""
    from src.drift import drift_report

    def build():
        con = connect()
        try:
            ev = pd.read_sql("SELECT ts, amount, decision, risk, merchant_id, device_id, user_id FROM events "
                             "ORDER BY seq DESC LIMIT 40000", con)
            fb = dict(con.execute("SELECT status, COUNT(*) FROM feedback GROUP BY status").fetchall())
            fb_lab = dict(con.execute("SELECT label, COUNT(*) FROM feedback WHERE status='accepted' GROUP BY label").fetchall())
            reg = [dict(r) for r in con.execute("SELECT version, role, kind, trained_on, datetime(created,'unixepoch') AS created "
                                                "FROM model_registry ORDER BY created DESC LIMIT 6")]
        finally:
            con.close()
        train = ENGINE.history.sample(min(50000, len(ENGINE.history)), random_state=1) if ENGINE.ready else None
        rep = drift_report(ev, train)
        ready = fb_lab.get(1, 0) >= 10 and sum(fb_lab.values()) >= 50
        rep["feedback"] = {"labels_by_status": fb, "accepted_fraud": fb_lab.get(1, 0), "accepted_genuine": fb_lab.get(0, 0),
                           "challenger_ready": ready, "registry": reg,
                           "criteria": ["Only labels with status 'accepted' are used (contradictions need a second analyst).",
                                        "A challenger is trained only on demand, never on every label: at least 50 accepted labels including 10 fraud.",
                                        "It is compared with the champion on payments after the newest label (unseen by both).",
                                        "Promotion only if challenger PR-AUC >= champion and false-positive rate at most 0.2 points worse; the old model is kept."],
                           "command": "python -m src.retrain --promote"}
        return rep
    return JSONResponse(await asyncio.to_thread(build))


@guard("admin")
async def admin_scenarios(request):
    """GET: catalog + live steps of the current run. POST {scenario}: start one (all payments go through the
    real engine; outcomes are never scripted)."""
    if (r := need_engine()):
        return r
    run = scenario_runner()
    if request.method == "POST":
        d = await body(request)
        try:
            run.start(str(d.get("scenario", "")), request.state.user["username"], delay=min(max(float(d.get("delay", 1.0)), 0.2), 5))
        except KeyError:
            return err("unknown scenario", 404)
        except RuntimeError as e:
            return err(str(e), 409)
        con = connect(); audit(con, request.state.user["username"], "admin", "scenario_start", str(d.get("scenario"))); con.close()
    return JSONResponse(run.status())


@guard("admin")
async def admin_simulator(request):
    if (r := need_engine()):
        return r
    d = await body(request)
    if d.get("action") == "start":
        rate = min(max(float(d.get("rate", 200)), 10), 2000)
        ENGINE.start_simulator(rate)
    else:
        ENGINE.stop_simulator()
    con = connect(); audit(con, request.state.user["username"], "admin", f"simulator_{d.get('action')}"); con.close()
    return JSONResponse(ENGINE.sim_state)


@guard("admin")
async def admin_model(request):
    def rd(p):
        f = REPORTS_DIR / p
        return json.loads(f.read_text()) if f.exists() else None
    m, ex, lat = rd("v2/metrics.json"), rd("experiments/results.json"), rd("v2/latency.json")
    out = {"deployed": m and {"test": m["test"], "recall_by_pattern": m["recall_by_pattern"], "fusion_weights": m["fusion_weights"],
                              "thresholds": m["thresholds"]},
           "latency": lat and {"latency_ms": lat["latency_ms"], "throughput": lat["throughput_txn_per_sec_single_core"]}}
    if ex:
        out["comparison"] = {k: {"pr_auc": v["pr_auc"], "recall": v["recall"], "precision": v["precision"], "fpr": v["fpr"],
                                 "loss_inr": v["loss_inr"]} for k, v in ex["compare"]["v2"]["results"].items()}
        out["unseen"] = {k: {"supervised": v["supervised_only"], "hybrid": v["hybrid_system"],
                             "after_5d": v["adaptation"]["after_5d_feedback"]["eval_recall"]} for k, v in ex["lopo"].items()}
        if "confirm" in ex:
            out["confirm"] = {k: v["pr_auc"] for k, v in ex["confirm"]["compare"]["results"].items()}
    out["app_dataset"] = ENGINE.dataset
    out["app_dataset_label"] = ENGINE.options().get("dataset_label")
    out["real"] = rd("real/results.json")
    nat = MODEL_ROOT / "ieee_native" / "meta.json"
    out["ieee_native"] = json.loads(nat.read_text())["metrics"] if nat.exists() else None
    # payTrace 2.1 evaluation outputs (all produced by src/v21/*.py; nothing is typed in by hand)
    ieee = rd("v21/ieee_opt.json")
    out["v21"] = {"ieee": ieee and {"experiments": {k: {kk: v[kk] for kk in ("learner", "n_features", "pr_auc", "roc_auc", "recall_at_1pct_fpr", "review", "best_iteration")}
                                                    for k, v in (ieee.get("experiments") or {}).items()}},
                  "unseen": rd("v21/unseen.json"), "graph": rd("v21/graph_eval.json"),
                  "fusion": {k: v for k in ("v2", "sparkov") if (v := rd(f"v21/fusion_{k}.json"))}}
    return JSONResponse(out)


def audit_simple(request, action):
    con = connect()
    try:
        audit(con, request.state.user["username"], request.state.user["role"], action, "ieee_replay")
    finally:
        con.close()


@guard("admin")
async def admin_replay(request):
    if request.method == "POST":
        d = await body(request)
        action = d.get("action")
        try:
            if action == "start":
                REPLAY.start(max(10.0, min(float(d.get("rate", 300)), 5000.0)))
            elif action == "stop":
                REPLAY.stop()
            elif action == "reset":
                REPLAY.stop()
                if REPLAY.thread:
                    REPLAY.thread.join(timeout=5)
                REPLAY.reset()
            else:
                return err("action must be start, stop or reset", 422)
        except RuntimeError as e:
            return err(str(e), 409)
        audit_simple(request, f"replay_{action}")
    return JSONResponse(REPLAY.status())


@guard("admin")
async def admin_llm_test(request):
    return JSONResponse(llm.test_connection())


@guard("admin")
async def admin_audit(request):
    con = connect()
    try:
        rows = [dict(r) for r in con.execute("SELECT datetime(wall_time,'unixepoch') AS time, actor, role, action, target, detail "
                                             "FROM audit_log ORDER BY wall_time DESC LIMIT 200").fetchall()]
        users = [dict(r) for r in con.execute("SELECT username, role, display_name FROM users ORDER BY role").fetchall()]
    finally:
        con.close()
    return JSONResponse({"audit": rows, "users": users})


@guard("admin")
async def admin_users(request):
    con = connect()
    try:
        rows = [dict(r) for r in con.execute("SELECT username, role, display_name, customer_id, demo, "
                                             "datetime(created,'unixepoch') AS created FROM users ORDER BY role, username")]
    finally:
        con.close()
    return JSONResponse({"users": rows})


@guard("admin")
async def admin_create_user(request):
    """Admin creates any account; the password is generated and shown once."""
    a = request.state.user
    d = await body(request)
    role = d.get("role")
    if role not in ("admin", "analyst", "customer"):
        return err("role must be admin, analyst or customer", 422)
    name = " ".join(str(d.get("display_name", "")).split())[:60]
    con = connect()
    try:
        username = str(d.get("username", "")).strip().lower()
        if not username:
            while True:
                username = f"{role[:3]}_{secrets.token_hex(3)}"
                if not con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                    break
        if not USERNAME_RE.match(username):
            return err("username: 3-32 characters, lowercase letters, numbers, dot or underscore", 422)
        if con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            return err("that username is taken", 409)
        cid = None
        if role == "customer":
            cid = str(d.get("customer_id", "")).strip().upper() or _new_customer_id(con)
            if not re.match(r"^U\d{3,7}$", cid):
                return err("customer id looks like U01022", 422)
        pw = random_password()
        insert_user(con, username, role, name or username, cid, pw)
        con.commit()
        audit(con, a["username"], "admin", "user_created", username, {"role": role})
    finally:
        con.close()
    return JSONResponse({"username": username, "role": role, "customer_id": cid, "password": pw}, status_code=201)


@guard("admin")
async def admin_reset_password(request):
    a, target = request.state.user, request.path_params["username"]
    con = connect()
    try:
        if not con.execute("SELECT 1 FROM users WHERE username=?", (target,)).fetchone():
            return err("no such user", 404)
        pw = random_password()
        set_password(con, target, pw)
        con.execute("UPDATE users SET demo=0 WHERE username=?", (target,))
        con.commit()
        audit(con, a["username"], "admin", "password_reset", target)
    finally:
        con.close()
    return JSONResponse({"username": target, "password": pw})


@guard("admin")
async def admin_delete_user(request):
    a, target = request.state.user, request.path_params["username"]
    if target == a["username"]:
        return err("you cannot delete your own account", 409)
    con = connect()
    try:
        if con.execute("SELECT role FROM users WHERE username=?", (target,)).fetchone() is None:
            return err("no such user", 404)
        others = con.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND username != ?", (target,)).fetchone()[0]
        if not others:
            return err("cannot delete the last admin", 409)
        con.execute("DELETE FROM users WHERE username=?", (target,))
        con.commit()
        audit(con, a["username"], "admin", "user_deleted", target)
    finally:
        con.close()
    return JSONResponse({"ok": True})


# ---------------- integration API (other systems) ----------------
async def api_score(request):
    token = os.getenv("PAYTRACE_API_TOKEN")
    if not token:
        return err("integration API disabled: set PAYTRACE_API_TOKEN", 503)
    if not hmac.compare_digest(request.headers.get("authorization", "").removeprefix("Bearer ").strip(), token):
        return err("unauthorised", 401)
    if (r := need_engine()):
        return r
    d = await body(request)
    try:
        args = _payment_args(d, str(d.get("user_id", "")).strip()[:64] or "UNKNOWN", str(d.get("device_id", "UNKNOWN"))[:64])
        if d.get("device_id"):
            args["device_id"] = str(d["device_id"])[:64]
        res = ENGINE.submit_payment(**args, source="api", actor="api", role="system", explain_with_llm=False)
    except InvalidTransaction as e:
        return err(str(e), 422)
    return JSONResponse({k: res[k] for k in ("txn_id", "decision", "model_decision", "policy", "status", "risk", "reasons",
                                             "tripwires", "latency_ms")})


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp: Response = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                                   "script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp


@contextlib.asynccontextmanager
async def lifespan(app):
    connect().close()
    seed_users()
    ENGINE.start(background=True, history_limit=int(os.getenv("PAYTRACE_HISTORY_LIMIT", "0")))
    yield
    ENGINE.stop_simulator()


routes = [
    Route("/", index), Route("/api/health", health), Route("/api/login", login, methods=["POST"]),
    Route("/api/logout", logout, methods=["POST"]), Route("/api/signup", signup, methods=["POST"]),
    Route("/api/account/password", change_password, methods=["POST"]),
    Route("/api/admin/users", admin_users), Route("/api/admin/users/create", admin_create_user, methods=["POST"]),
    Route("/api/admin/users/{username}/reset", admin_reset_password, methods=["POST"]),
    Route("/api/admin/users/{username}/delete", admin_delete_user, methods=["POST"]), Route("/api/demo-accounts", demo_accounts), Route("/api/me", me),
    Route("/api/customer/payments", customer_payments), Route("/api/customer/pay", customer_pay, methods=["POST"]),
    Route("/api/customer/payments/{txn_id}/captcha", customer_captcha),
    Route("/api/customer/payments/{txn_id}/respond", customer_respond, methods=["POST"]),
    Route("/api/customer/queries", customer_queries), Route("/api/customer/queries/new", customer_query_create, methods=["POST"]),
    Route("/api/admin/queries", admin_queries), Route("/api/admin/queries/{qid}", admin_query_decide, methods=["POST"]),
    Route("/api/cases", cases), Route("/api/cases/{txn_id}", case_detail),
    Route("/api/rings", rings), Route("/api/graph/node", graph_node),
    Route("/api/admin/scenarios", admin_scenarios, methods=["GET", "POST"]), Route("/api/admin/drift", admin_drift),
    Route("/api/cases/{txn_id}/summary", case_summary, methods=["POST"]),
    Route("/api/cases/{txn_id}/ask", case_ask, methods=["POST"]), Route("/api/cases/{txn_id}/report", case_report),
    Route("/api/cases/{txn_id}/decision", case_decide, methods=["POST"]), Route("/api/check", check_payment, methods=["POST"]),
    Route("/api/customers/{uid}", customer_lookup), Route("/api/directory", directory),
    Route("/api/admin/overview", admin_overview), Route("/api/admin/simulator", admin_simulator, methods=["POST"]),
    Route("/api/admin/model", admin_model),
    Route("/api/admin/replay", admin_replay, methods=["GET", "POST"]), Route("/api/admin/llm-test", admin_llm_test, methods=["POST"]),
    Route("/api/admin/audit", admin_audit), Route("/api/v1/score", api_score, methods=["POST"]),
    Mount("/static", StaticFiles(directory=STATIC), name="static"),
]
app = Starlette(routes=routes, middleware=[Middleware(SecurityHeaders)], lifespan=lifespan)
