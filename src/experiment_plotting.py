from __future__ import annotations

from pathlib import Path
from typing import Any


def save_figure(figure: Any, path: str | Path, *, dpi: int = 180, bbox_inches: str = "tight") -> Path:
    """Save a matplotlib figure with consistent parent-directory creation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(p, dpi=dpi, bbox_inches=bbox_inches)
    return p
