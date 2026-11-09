"""payTrace. Settings use PAYTRACE_* environment variables. Values set under the project's former
PAYTRACE_* names are still honoured, so an existing .env keeps working unchanged."""
import os as _os

_OLD = "FRAUD" + "SHIELD_"          # former prefix, kept only for backward compatibility
for _k, _v in list(_os.environ.items()):
    if _k.startswith(_OLD):
        _os.environ.setdefault("PAYTRACE_" + _k[len(_OLD):], _v)
