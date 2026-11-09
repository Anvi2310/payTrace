"""Environment-based configuration. Secrets are only ever read from the environment (or a local .env file that is
git-ignored); nothing secret is stored in code or in the repository. See .env.example."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_OLD_PREFIX = "FRAUD" + "SHIELD_"   # former name of the PAYTRACE_ variables


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                k = k.strip()
                if k.startswith(_OLD_PREFIX):          # former variable names still work
                    k = "PAYTRACE_" + k[len(_OLD_PREFIX):]
                if v:                                  # an empty "KEY=" line means "not set", not "set to empty"
                    os.environ.setdefault(k, v)


_load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
USERS = os.getenv("PAYTRACE_USERS", "")            # "name:role:sha256(salt+password);..."
SALT = os.getenv("PAYTRACE_SALT", "")
FEEDBACK_RATE_LIMIT_PER_HOUR = int(os.getenv("PAYTRACE_FEEDBACK_LIMIT", "200"))
DATASET = os.getenv("PAYTRACE_DATASET", "v2")
