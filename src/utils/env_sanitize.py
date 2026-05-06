"""Strip stray whitespace from environment values (e.g. CRLF from Windows-saved .env)."""

from __future__ import annotations

import os


def getenv_strip(key: str, default: str | None = None) -> str | None:
    raw = os.getenv(key, default)
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None
