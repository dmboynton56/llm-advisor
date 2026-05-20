"""JSON helpers for telemetry and API payloads (BQ Decimal, numpy scalars)."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def json_default(value: Any) -> Any:
    """``json.dumps(..., default=json_default)`` handler for non-stdlib types."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    # numpy / pandas scalars (optional dependency)
    try:
        import numpy as np

        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    return str(value)
