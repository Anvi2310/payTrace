"""Bank policy layer: hard limits and common-sense rules applied around the ML model.

Why it exists: the model only knows the amounts, payees and customers it was trained on. A ₹20 crore payment is so far
outside the training data that tree models simply treat it like "a large payment" (risk saturates), and the model
never checks whether a payee or customer actually exists. Real payment systems handle these with deterministic rules
BEFORE and AFTER the fraud model, so payTrace does too.

  1. Limits (checked before scoring): a payment above the channel's per-transaction limit is declined outright,
     like a real bank would. Defaults follow common Indian limits and are configurable in .env:
       PAYTRACE_LIMIT_UPI=100000   PAYTRACE_LIMIT_CARD=500000   PAYTRACE_LIMIT_NETBANKING=1000000
       PAYTRACE_LIMIT_CARD_POS=500000   PAYTRACE_LIMIT_CARD_ONLINE=500000
  2. Rules (checked after scoring; three or more warning rules together also mean BLOCK): they can only RAISE the model's decision (ALLOW -> REVIEW -> BLOCK), never lower
     it, and every rule that fires is shown as a reason. The model's own risk score is reported unchanged, so it is
     always clear which part came from the model and which from policy.
"""
from __future__ import annotations

import os
import re

from src.scorer_v2 import InvalidTransaction

DEFAULT_LIMITS = {"UPI": 100_000, "CARD": 500_000, "CARD_POS": 500_000, "CARD_ONLINE": 500_000, "NETBANKING": 1_000_000}
ORDER = {"ALLOW": 0, "REVIEW": 1, "BLOCK": 2}
HIGH_VALUE = 200_000          # any payment this large gets a second look
NEW_ACCOUNT_LIMIT = 50_000    # first payments of a customer with no history above this get a second look
UNVERIFIED_BLOCK = 50_000     # a payment this large to a payee the bank can't verify is held, not just reviewed


class PolicyDecline(InvalidTransaction):
    """The payment is not allowed at all (e.g. above the channel limit). It is never sent to the model."""


def limit_for(channel: str) -> float:
    env = os.getenv(f"PAYTRACE_LIMIT_{channel.upper()}")
    try:
        return float(env) if env else float(DEFAULT_LIMITS.get(channel.upper(), DEFAULT_LIMITS["UPI"]))
    except ValueError:
        return float(DEFAULT_LIMITS.get(channel.upper(), DEFAULT_LIMITS["UPI"]))


def inr(x: float) -> str:
    """Indian digit grouping: 2500000 -> 25,00,000."""
    s = f"{int(round(x))}"
    if len(s) <= 3:
        return "₹" + s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"₹{head},{tail}"


def check_limits(amount: float, channel: str):
    if not amount > 0:
        raise PolicyDecline("amount must be more than zero")
    lim = limit_for(channel)
    if amount > lim:
        name = {"CARD_POS": "in-store card", "CARD_ONLINE": "online card", "NETBANKING": "net banking"}.get(channel.upper(), channel)
        raise PolicyDecline(f"{inr(amount)} is above the {name} limit of {inr(lim)} per payment. "
                            f"Split it or use another channel.")


def apply_rules(*, amount: float, payee: str, customer_known: bool, payee_known: bool, past_payments: int,
                median_amount: float | None, max_amount: float | None, new_device: bool, model_decision: str,
                confirmed_count: int = 0, confirmed_max: float = 0.0, takeover: bool = False) -> dict:
    """Return {"decision": final decision, "rules": [ {rule, effect, reason, plain} ... ]}.

    confirmed_count / confirmed_max: payments this customer already completed to this payee in the app (after passing
    the security check, or released by the fraud team). Such a payee is trusted by THIS customer for amounts up to
    twice the largest one they confirmed (and below the large-payment limit), so a payee the bank doesn't know isn't
    questioned again on every small payment."""
    rules = []
    if not payee_known and confirmed_count and amount <= 2 * confirmed_max and amount < UNVERIFIED_BLOCK:
        payee_known = True

    def fire(rule, effect, reason, plain):
        rules.append({"rule": rule, "effect": effect, "reason": reason, "plain": plain})

    if not customer_known:
        fire("unknown_customer", "REVIEW", "No bank record of this customer",
             "we couldn't match this payment to an existing account")
    if not payee_known:
        if amount >= UNVERIFIED_BLOCK:
            fire("unverified_payee_large", "BLOCK",
                 f"{inr(amount)} to payee {payee}, which is not in the bank's merchant or customer records",
                 "we couldn't verify the payee for a payment this size")
        else:
            fire("unverified_payee", "REVIEW", f"Payee {payee} not found in the bank's merchant or customer records",
                 "we couldn't verify the payee")
    if amount >= HIGH_VALUE:
        fire("high_value", "REVIEW", f"High-value payment ({inr(amount)})", "it is a high-value payment")
    if past_payments and max_amount:
        ratio_max = amount / max_amount
        if ratio_max >= 10 and amount >= 50_000 and (new_device or not payee_known):
            fire("extreme_amount", "BLOCK",
                 f"{ratio_max:,.0f}× the customer's largest past payment ({inr(max_amount)}), from a new device or to an unverified payee",
                 "the amount is far above anything sent from this account before")
        elif median_amount and amount / median_amount >= 5 and amount >= 25_000:
            fire("large_vs_history", "REVIEW", f"{amount / median_amount:,.0f}× the customer's usual payment ({inr(median_amount)})",
                 "the amount is much higher than you usually send")
    elif amount >= NEW_ACCOUNT_LIMIT:
        fire("new_account_large", "REVIEW", f"Large payment ({inr(amount)}) from an account with no payment history",
             "it is a large payment from a new account")

    if takeover:
        # measured on the synthetic UPI data: 0.02% of genuine payments, 21% of account takeovers (reports/v21)
        fire("takeover_pattern", "REVIEW", "Account-takeover pattern: a new device and new city paying a never-paid payee, "
             "or a device first used in the last hour whose first payment was flagged",
             "it came from a new device in a new place, to someone you have never paid")
    if len({r["rule"] for r in rules if r["effect"] == "REVIEW"}) >= 3:
        fire("multiple_warnings", "BLOCK", "Several independent warning signs at once", "several things about it look unusual")
    final = model_decision
    for r in rules:
        if ORDER[r["effect"]] > ORDER[final]:
            final = r["effect"]
    return {"decision": final, "rules": rules}
