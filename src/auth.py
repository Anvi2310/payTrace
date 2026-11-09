"""Role-based access: viewer < analyst < admin.

Users come from the PAYTRACE_USERS environment variable:  "alice:analyst:<sha256 hex>;bob:admin:<hex>"
where hex = sha256(PAYTRACE_SALT + password). Generate one with:  python -m src.auth <password>
If no users are configured the dashboard runs in clearly labelled DEMO MODE, where the role is picked freely;
that mode exists for the hackathon demo only.
"""
from __future__ import annotations

import hashlib
import hmac
import sys

from src import settings

ROLES = {"viewer": 0, "analyst": 1, "admin": 2}
PERMISSIONS = {"view": "viewer", "label": "analyst", "copilot": "analyst", "case_report": "analyst",
               "stream_control": "admin", "retrain": "admin", "promote": "admin"}


def hash_password(pw: str) -> str:
    return hashlib.sha256((settings.SALT + pw).encode()).hexdigest()


def users() -> dict[str, tuple[str, str]]:
    out = {}
    for item in filter(None, settings.USERS.split(";")):
        try:
            name, role, h = item.split(":")
        except ValueError:
            continue
        if role in ROLES:
            out[name.strip()] = (role, h.strip())
    return out


def demo_mode() -> bool:
    return not users()


def login(name: str, password: str) -> str | None:
    u = users().get(name)
    if u and hmac.compare_digest(u[1], hash_password(password)):
        return u[0]
    return None


def can(role: str, action: str) -> bool:
    return ROLES.get(role, -1) >= ROLES[PERMISSIONS[action]]


if __name__ == "__main__":
    print(hash_password(sys.argv[1]))
