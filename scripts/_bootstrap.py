from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def add_repo_root() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def require_packages(packages: list[str]) -> None:
    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if missing:
        names = ", ".join(missing)
        raise SystemExit(f"Missing dependencies: {names}. Install them with: pip install -r requirements.txt")
