from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL rows from a UTF-8 file."""
    p = Path(path)
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, sort_keys: bool = False) -> int:
    """Write JSONL rows non-atomically and return the row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
            count += 1
    return count


def atomic_json(path: str | Path, value: object, *, indent: int = 2, ensure_ascii: bool = False) -> None:
    """Atomically write JSON through a sibling .tmp file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=indent, ensure_ascii=ensure_ascii), encoding="utf-8")
    temporary.replace(p)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, sort_keys: bool = False) -> int:
    """Atomically write JSONL rows through a sibling .tmp file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
            count += 1
    temporary.replace(p)
    return count


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path: str | Path, value: dict[str, Any]) -> None:
    atomic_json(path, value)
