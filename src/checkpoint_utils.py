from __future__ import annotations

import math
import re
from typing import Any


_STEP_RE = re.compile(r"step(-?\d+)$")


def checkpoint_step(value: str | int) -> int:
    """Return the integer training step from values such as 'step16000'."""
    if isinstance(value, int):
        return value
    text = str(value)
    match = _STEP_RE.search(text)
    if match:
        return int(match.group(1))
    return int(text)


def checkpoint_sort_key(value: str | int) -> tuple[int, str]:
    try:
        return (checkpoint_step(value), str(value))
    except Exception:
        return (10**18, str(value))


def finite(value: Any) -> float | None:
    """Convert finite numeric values to float, otherwise None."""
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None
