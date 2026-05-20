"""Tests for JSON encoding helpers."""

from __future__ import annotations

import json
from decimal import Decimal

from src.utils.json_encode import json_default


def test_json_default_decimal() -> None:
    payload = {"z": Decimal("1.2345")}
    encoded = json.dumps(payload, default=json_default)
    assert '"z": 1.2345' in encoded
