#!/usr/bin/env python3
"""Reliably download an HF blob in verified HTTP range chunks."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests

url, output, expected_sha, expected_size = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if "hf-p-cfw.fyan.top" in url:
    urls = [url.replace("hf-p-cfw.fyan.top", "hf-mirror.com"), url]
elif "hf-mirror.com" in url:
    urls = [url, url.replace("hf-mirror.com", "hf-p-cfw.fyan.top")]
else:
    urls = [url]
chunk_size = 1 * 1024 * 1024
partial = output.with_suffix(output.suffix + ".range.incomplete")
partial.parent.mkdir(parents=True, exist_ok=True)
lock_handle = output.with_suffix(output.suffix + ".range.lock").open("w")
fcntl.flock(lock_handle, fcntl.LOCK_EX)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

if output.exists() and output.stat().st_size == expected_size and sha256(output) == expected_sha:
    print(f"RANGE_ALREADY_COMPLETE {output} {expected_size} {expected_sha}", flush=True)
    raise SystemExit(0)
offset = partial.stat().st_size if partial.exists() else 0
if offset > expected_size:
    partial.unlink(missing_ok=True); offset = 0

thread_state = threading.local()

def fetch_chunk(bounds: tuple[int, int]) -> bytes:
    start, end = bounds
    for attempt in range(1, 21):
        try:
            active_url = urls[(attempt - 1) % len(urls)]
            if not hasattr(thread_state, "session"):
                thread_state.session = requests.Session()
            response = thread_state.session.get(active_url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 30), allow_redirects=True)
            content_range = response.headers.get("content-range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if response.status_code != 206 or not match:
                raise RuntimeError(f"expected 206 Content-Range, got {response.status_code} {content_range!r}")
            got_start, got_end, got_total = map(int, match.groups())
            if (got_start, got_end, got_total) != (start, end, expected_size):
                raise RuntimeError(f"range mismatch {(got_start,got_end,got_total)} != {(start,end,expected_size)}")
            body = response.content
            if len(body) != end - start + 1:
                raise RuntimeError(f"body length {len(body)} != {end-start+1}")
            return body
        except Exception as error:
            print(f"RANGE_RETRY offset={start} attempt={attempt} endpoint={urls[(attempt-1)%len(urls)]} error={type(error).__name__}:{error}", flush=True)
            if attempt == 20: raise
            time.sleep(min(30, 3 * attempt))

pool = ThreadPoolExecutor(max_workers=16)
while offset < expected_size:
    bounds = []
    cursor = offset
    for _ in range(16):
        if cursor >= expected_size: break
        end = min(expected_size - 1, cursor + chunk_size - 1)
        bounds.append((cursor, end))
        cursor = end + 1
    bodies = list(pool.map(fetch_chunk, bounds))
    with partial.open("ab") as handle:
        for body in bodies: handle.write(body)
        handle.flush(); os.fsync(handle.fileno())
    offset = bounds[-1][1] + 1
    print(f"RANGE_PROGRESS {offset}/{expected_size}", flush=True)
pool.shutdown()

actual_sha = sha256(partial)
if actual_sha != expected_sha:
    raise RuntimeError(f"SHA256 mismatch {actual_sha} != {expected_sha}")
partial.replace(output)
print(f"RANGE_COMPLETE {output} {expected_size} {actual_sha}", flush=True)
