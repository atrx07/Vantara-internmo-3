#!/usr/bin/env python3
"""Verify immutable Vantara governance/reference files against REFERENCE_LOCK.json."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "governance" / "REFERENCE_LOCK.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    failures = []
    for rel, expected in data["files"].items():
        path = ROOT / rel
        if not path.exists():
            failures.append(f"MISSING {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"CHANGED {rel}: expected {expected}, got {actual}")
    if failures:
        print("REFERENCE LOCK FAILED")
        for item in failures:
            print(f"- {item}")
        return 1
    print(f"REFERENCE LOCK PASS: {len(data['files'])} immutable files verified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
