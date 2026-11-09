"""A small self-hosted CAPTCHA for confirming flagged payments.

The app makes its own challenge (no third-party service, nothing leaves the machine): 5 characters mixing capital
letters, small letters and digits, drawn as an SVG with each character slightly rotated, shifted and resized, over light
background noise (a few curves, dots and one wavy strike line), like a usual CAPTCHA.

Case matters (an "a" is not an "A"), so the alphabet leaves out every character that is easy to misread: 0/O/o, 1/I/l/i/j,
and letters whose small form looks like the capital (c, k, p, s, u, v, w, x, z in both cases).
The text is kept only in server memory, expires after 10 minutes, and allows 5 attempts.

It proves a person is answering, which stops scripted payment runs. It does not prove WHO the person is: a one-time
code sent to the customer's registered phone does that, and a real deployment should use both (see README).
"""
from __future__ import annotations

import secrets
import time

UPPER = "ABDEFGHJLMNQRTY"
LOWER = "abdefghmnqrty"
DIGITS = "2345678"
ALPHABET = UPPER + LOWER + DIGITS
LENGTH = 5
TTL_SECONDS = 600
MAX_TRIES = 5


def new_text() -> str:
    """Always at least one capital and one small letter, so the mix is visible."""
    while True:
        t = "".join(secrets.choice(ALPHABET) for _ in range(LENGTH))
        if any(c in UPPER for c in t) and any(c in LOWER for c in t):
            return t


def _r(lo: float, hi: float) -> float:
    return lo + (hi - lo) * secrets.randbelow(10_000) / 10_000


def svg(text: str) -> str:
    w, h = 220, 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'role="img" aria-label="Security check image">',
             f'<rect width="{w}" height="{h}" rx="8" fill="#f4f2ea"/>']
    for _ in range(3):                                   # a few faint background curves
        parts.append(f'<path d="M{_r(0, w):.0f} {_r(0, h):.0f} Q{_r(0, w):.0f} {_r(0, h):.0f} {_r(0, w):.0f} {_r(0, h):.0f}" '
                     f'stroke="#{secrets.choice(["a7a294", "8f9cb8", "b89f8f", "94b09a"])}" stroke-width="1.2" '
                     f'fill="none" opacity="0.6"/>')
    for _ in range(22):                                  # light speckle
        parts.append(f'<circle cx="{_r(0, w):.0f}" cy="{_r(0, h):.0f}" r="{_r(0.6, 1.4):.1f}" fill="#8a8574" opacity="0.45"/>')
    step = (w - 30) / len(text)
    fonts = ["Georgia, 'Times New Roman', serif", "Verdana, Arial, sans-serif", "'Courier New', monospace"]
    for i, ch in enumerate(text):
        x, y = 24 + i * step + _r(-3, 3), 47 + _r(-6, 6)
        parts.append(f'<text x="{x:.0f}" y="{y:.0f}" transform="rotate({_r(-16, 16):.0f} {x:.0f} {y:.0f}) '
                     f'skewX({_r(-10, 10):.0f})" font-family="{secrets.choice(fonts)}" font-size="{_r(30, 36):.0f}" '
                     f'font-weight="{secrets.choice(["600", "700"])}" '
                     f'fill="#{secrets.choice(["1f2a44", "3a1f2a", "1f3a2a", "2a2a2a", "37306b"])}">{ch}</text>')
    y0 = _r(30, 45)                                      # one gentle wavy strike line through the text
    parts.append(f'<path d="M6 {y0:.0f} C{_r(50, 80):.0f} {_r(15, 60):.0f} {_r(130, 170):.0f} {_r(15, 60):.0f} {w - 6} '
                 f'{_r(28, 48):.0f}" stroke="#4a4840" stroke-width="1.4" fill="none" opacity="0.5"/>')
    parts.append("</svg>")
    return "".join(parts)


class CaptchaStore:
    """In-memory challenges keyed by payment id."""

    def __init__(self):
        self._items: dict[str, dict] = {}

    def issue(self, key: str) -> str:
        text = new_text()
        self._items[key] = {"text": text, "tries": 0, "expires": time.time() + TTL_SECONDS}
        return text

    def peek(self, key: str) -> str | None:                # tests only
        item = self._items.get(key)
        return item["text"] if item else None

    def check(self, key: str, answer: str | None) -> None:
        """Raise PermissionError with a user-readable reason unless the answer matches."""
        item = self._items.get(key)
        if item is None or time.time() > item["expires"]:
            raise PermissionError("the security check expired; load a new image")
        if item["tries"] >= MAX_TRIES:
            raise PermissionError("too many wrong attempts; load a new image")
        given = "".join(str(answer or "").split())         # spaces ignored; case matters
        if not secrets.compare_digest(given.encode(), item["text"].encode()):
            item["tries"] += 1
            raise PermissionError(f"the characters don't match ({MAX_TRIES - item['tries']} tries left)")
        del self._items[key]                              # one use only
