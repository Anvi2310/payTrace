"""LLM provider layer. The LLM only writes explanations from verified evidence; it never decides anything.

Providers (first one with a key wins, or force with LLM_PROVIDER=gemini|groq):
  gemini  GEMINI_API_KEY  (free key: https://aistudio.google.com/apikey)        model: GEMINI_MODEL
  groq    GROQ_API_KEY    (free key: https://console.groq.com/keys)             model: GROQ_MODEL
Model names change over time. For Gemini, if the configured model has been retired (HTTP 404) the app asks Google
which models this key can use, switches to the newest Flash model automatically and remembers it. For the others,
set the *_MODEL variables if a default stops working (the admin "Test connection" button shows the provider's error).
With no key, callers fall back to deterministic template text built from the same evidence.
"""
from __future__ import annotations

import json
import os
import re
import threading

import requests

from src import settings  # noqa: F401  (loads .env)

TIMEOUT = 25
GEMINI_DEFAULT = "gemini-3.8-flash"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_gemini_model_override: str | None = None    # set when the configured model was retired and we found a replacement
_gemini_lock = threading.Lock()


def config() -> dict:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    keys = {"gemini": os.getenv("GEMINI_API_KEY"), "groq": os.getenv("GROQ_API_KEY")}
    models = {"gemini": _gemini_model_override or os.getenv("GEMINI_MODEL") or GEMINI_DEFAULT, "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")}
    order = [forced] if forced in keys else ["gemini", "groq"]
    for p in order:
        if keys.get(p):
            return {"provider": p, "model": models[p], "key": keys[p]}
    return {"provider": None, "model": None, "key": None}


def status() -> dict:
    c = config()
    return {"provider": c["provider"] or "none (template fallback)", "model": c["model"], "configured": bool(c["key"])}


class LLMError(RuntimeError):
    pass


def _version(name: str) -> tuple:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)


def discover_gemini_model(key: str, timeout: float = 10) -> str | None:
    """Ask Google which models this key can call and pick the newest stable Flash model (Flash-Lite as a fallback)."""
    try:
        r = requests.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": key}, params={"pageSize": 1000}, timeout=timeout)
        r.raise_for_status()
    except Exception:
        return None
    names = []
    for m in r.json().get("models", []):
        name = m.get("name", "").removeprefix("models/")
        if "generateContent" not in m.get("supportedGenerationMethods", []) or "flash" not in name:
            continue
        if any(w in name for w in ("preview", "exp", "image", "tts", "audio", "live", "thinking", "embedding", "8b")):
            continue
        names.append(name)
    if not names:
        return None
    full = [n for n in names if "lite" not in n] or names
    return sorted(full, key=lambda n: (_version(n), "latest" not in n, -len(n)))[-1]


def _gemini_call(model, key, system, user, max_tokens, timeout):
    return requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                         headers={"x-goog-api-key": key}, timeout=timeout,
                         json={"system_instruction": {"parts": [{"text": system}]},
                               "contents": [{"role": "user", "parts": [{"text": user}]}],
                               "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}})


def complete(system: str, user: str, max_tokens: int = 700, timeout: float = TIMEOUT) -> tuple[str, str]:
    """Returns (text, provider). Raises LLMError when no provider is configured or the call fails."""
    c = config()
    p, model, key = c["provider"], c["model"], c["key"]
    if not p:
        raise LLMError("no LLM key configured")
    try:
        if p == "gemini":
            global _gemini_model_override
            r = _gemini_call(model, key, system, user, max_tokens, timeout)
            if r.status_code == 404:                       # model retired or not offered to this key: find a current one
                with _gemini_lock:
                    found = discover_gemini_model(key)
                if found and found != model:
                    _gemini_model_override = found
                    r = _gemini_call(found, key, system, user, max_tokens, timeout)
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"].get("parts", [])
            text = "".join(x.get("text", "") for x in parts if not x.get("thought"))
            if not text.strip():
                raise LLMError("gemini returned an empty reply")
        else:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=timeout,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0.2, "max_tokens": max_tokens,
                                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
    except LLMError:
        raise
    except requests.HTTPError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        raise LLMError(f"{p} returned {e.response.status_code if e.response is not None else '?'}: {detail}")
    except Exception as e:
        raise LLMError(f"{p} call failed: {type(e).__name__}: {e}")
    return text.strip(), p


def test_connection() -> dict:
    try:
        text, p = complete("Reply with exactly: OK", "Connection test", max_tokens=20)
        return {"ok": True, "provider": p, "model": config()["model"], "reply": text[:50]}
    except LLMError as e:
        return {"ok": False, "error": str(e)}


# ---------------- prompts ----------------
ANALYST_SYSTEM = (
    "You are a payments fraud investigation assistant for bank analysts. Use ONLY the JSON evidence provided; do not "
    "invent names, amounts, dates or facts, and quote numbers exactly as given (amounts in rupees). Never state that a "
    "customer is guilty or a fraudster; describe risk indicators. Never approve, reject or block a payment: the "
    "analyst decides. Explain HOW the evidence points to fraud or away from it, as a senior investigator would to a "
    "colleague. Format in Markdown with these sections:\n"
    "**Risk assessment**: 2-3 sentences: the system's decision and risk, the strongest single reason, and whether "
    "the evidence is consistent (several independent signals) or rests on one signal.\n"
    "**Evidence that raises risk**: bullets, each naming the signal, its value and what normal looks like for this "
    "customer (e.g. amount vs their median, times they paid this payee, device and payee history, time of day).\n"
    "**Evidence that lowers risk**: bullets (signals that look normal), or 'none found'.\n"
    "**Bank policy rules**: the rules that fired and what each means, or 'none'.\n"
    "**Network and merchant context**: connected customers, devices and payees with flagged activity, or 'none found'.\n"
    "**Similar past cases**: one or two lines with their outcomes, or 'none with known outcomes'.\n"
    "**What would change this assessment**: 2-3 bullets (facts the analyst could check).\n"
    "**Recommended next steps**: numbered.")

CUSTOMER_SYSTEM = (
    "You explain to a bank customer, calmly and clearly, why a payment they just tried to make was checked. Use ONLY "
    "the facts provided; quote amounts in rupees exactly as given. Plain language: no jargon, model names, scores or "
    "exact thresholds (those would help criminals). Never accuse the customer. Format in Markdown:\n"
    "1. One or two sentences on what happened: SENT = it went through; VERIFY = we need them to choose whether to "
    "send it; HELD = it has been BLOCKED for their protection and will not be sent (the customer cannot override this).\n"
    "2. **What looked different** - 2 to 4 bullets comparing this payment with their usual activity, using the "
    "numbers in the facts (e.g. 'Rs 23,000 is about 3.4 times your usual Rs 6,694'; 'first payment to this merchant'; "
    "'late at night for you'). Use the rupee sign.\n"
    "3. **What you can do** - 1 or 2 bullets: VERIFY = send it if they made it, cancel if not; HELD = nothing is "
    "needed, no money has left the account; if they really made it they can call the bank, and the fraud team "
    "reviews blocked payments. Do NOT offer a button or a way to release it.\n"
    "Keep it under 120 words.")


def customer_message(facts: dict) -> tuple[str, str]:
    """Plain-language customer message; template unless an LLM is configured."""
    try:
        return complete(CUSTOMER_SYSTEM, json.dumps(facts, default=str), max_tokens=350,
                        timeout=float(os.getenv("LLM_CUSTOMER_TIMEOUT", "6")))  # a payment must not wait long on an LLM
    except LLMError:
        return template_customer_message(facts), "template"


def _inr(x) -> str:
    s = f"{int(round(float(x)))}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        s = f"{head},{tail}"
    return "₹" + s


def what_looked_different(f: dict) -> list[str]:
    """Facts-based comparison with the customer's normal activity (used by the template and checked by tests)."""
    out = []
    amt, usual, largest = f["amount"], f.get("usual_amount"), f.get("largest_past_payment")
    if usual:
        ratio = amt / usual
        if ratio >= 1.5:
            out.append(f"{_inr(amt)} is about {ratio:.1f} times your usual payment of {_inr(usual)}.")
        if largest and amt > largest:
            out.append(f"It is more than your largest earlier payment ({_inr(largest)}).")
    elif not f.get("past_payments"):
        out.append("Your account has no payment history yet, so we have nothing to compare it with.")
    conf = f.get("confirmed_by_you_before") or 0
    if conf and not f.get("payee_known_to_bank", True):
        if amt > 2 * (f.get("largest_confirmed_to_payee") or 0):
            out.append(f"You've paid {f['payee']} before, but this is much more than the {_inr(f['largest_confirmed_to_payee'])} "
                       "you confirmed then.")
    elif not f.get("payee_known_to_bank", True):
        out.append(f"We couldn't find {f['payee']} in our list of known merchants.")
    elif not f.get("times_paid_this_payee_before"):
        out.append(f"It is your first payment to {f['payee']}.")
    hour = int(str(f.get("time_of_day", "12:00"))[:2])
    if hour >= 22 or hour < 4:
        out.append(f"It was made late at night ({f['time_of_day']}).")
    if f.get("new_device"):
        out.append("It came from a device we haven't seen you use.")
    for r in f.get("plain_reasons") or []:
        if len(out) >= 4:
            break
        line = r[0].upper() + r[1:] + "."
        if not any(r.split()[0] in o for o in out) and "usual" not in r:
            out.append(line)
    return out[:4]


def template_customer_message(f: dict) -> str:
    amt = _inr(f["amount"])
    if f["action"] == "SENT":
        return f"Your payment of {amt} to {f['payee']} was sent."
    diff = what_looked_different(f)
    bullets = "\n".join(f"- {d}" for d in diff) or "- It is different from how you usually pay."
    if f["action"] == "VERIFY":
        head = (f"We stopped to check your payment of {amt} to {f['payee']} before sending it. You decide whether it "
                "goes ahead.")
        todo = "- If you made this payment, complete the security check and choose **Yes, send it**.\n- If not, choose **Cancel**."
    else:
        head = (f"We've **blocked** your payment of {amt} to {f['payee']} to protect your account. Several strong "
                "warning signs appeared together, so it will not be sent. No money has left your account.")
        todo = ("- You don't need to do anything; your account is safe.\n- If you really made this payment, call your "
                "bank. Our fraud team reviews every blocked payment and will contact you.")
    return f"{head}\n\n**What looked different**\n{bullets}\n\n**What you can do**\n{todo}"
