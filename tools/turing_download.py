#!/usr/bin/env python3
"""Resume pinned model downloads inside this checkout; pause while benchmarking.

Run through tools/turing-run.sh. The manifest is a JSON list with url, path,
size and sha256 fields. Files become usable only after SHA-256 verification.
Create .turing/downloads/PAUSE to suspend transfer and hashing at a chunk boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PAUSE = ROOT / ".turing/downloads/PAUSE"


def local(value: str) -> Path:
    p = (ROOT / value).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("path must stay inside the checkout")
    return p


def wait_unpaused() -> None:
    while PAUSE.exists():
        time.sleep(1)


def verified(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            wait_unpaused()
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest() == expected


def download(item: dict, rate: float) -> None:
    target = local(item["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    size = int(item["size"])
    if target.exists():
        if target.stat().st_size == size and verified(target, item["sha256"]):
            print(f"verified existing {target.name}", flush=True)
            return
        raise ValueError(f"existing target failed verification: {target}")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"partial file exceeds expected size: {part}")
    if shutil.disk_usage(target.parent).free < size - offset + 8 * 1024**3:
        raise ValueError("insufficient disk space including 8 GiB safety margin")
    attempts = 0
    while offset < size:
        wait_unpaused()
        try:
            req = urllib.request.Request(item["url"], headers={"Range": f"bytes={offset}-"})
            with urllib.request.urlopen(req, timeout=60) as response:
                expected_range = f"bytes {offset}-"
                if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected_range):
                    raise ValueError("server did not honor byte range; partial file preserved")
                with part.open("ab") as output:
                    report = time.monotonic()
                    while offset < size:
                        wait_unpaused()
                        started = time.monotonic()
                        chunk = response.read(min(4 * 1024 * 1024, size - offset))
                        if not chunk:
                            raise OSError("incomplete response")
                        output.write(chunk)
                        offset += len(chunk)
                        output.flush()
                        time.sleep(max(0, len(chunk) / rate - (time.monotonic() - started)))
                        if time.monotonic() - report >= 30:
                            print(f"{target.name}: {offset}/{size} ({100*offset/size:.1f}%)", flush=True)
                            report = time.monotonic()
            attempts = 0
        except (OSError, TimeoutError) as error:
            attempts += 1
            print(f"retry {attempts}: {type(error).__name__}; saved {offset} bytes", flush=True)
            if attempts >= 20:
                raise
            time.sleep(min(60, attempts * 5))
            offset = part.stat().st_size if part.exists() else 0
    print(f"checking SHA-256 {target.name}", flush=True)
    if not verified(part, item["sha256"]):
        raise ValueError(f"SHA-256 mismatch; partial file preserved: {part}")
    part.rename(target)
    print(f"verified {target.name}: {item['sha256']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--mib-per-second", type=float, default=48)
    args = parser.parse_args()
    if os.environ.get("STRATA_TURING_ISOLATED") != "1":
        parser.error("run through tools/turing-run.sh")
    if args.mib_per_second <= 0:
        parser.error("rate must be positive")
    for item in json.loads(local(args.manifest).read_text()):
        download(item, args.mib_per_second * 1024**2)
    print("all downloads verified", flush=True)


if __name__ == "__main__":
    main()
