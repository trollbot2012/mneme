"""Verify that Hermes runs the exact canonical Mneme engine bytes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("active", type=Path, help="active Hermes plugin mneme.py")
    args = parser.parse_args()
    canonical = Path(__file__).resolve().parents[1] / "mneme.py"
    if not args.active.is_file():
        print(f"MISSING: {args.active}")
        return 2
    expected, actual = digest(canonical), digest(args.active)
    print(f"canonical {expected}  {canonical}")
    print(f"active    {actual}  {args.active.resolve()}")
    if expected != actual:
        print("DRIFT: active Hermes engine is not byte-identical")
        return 1
    print("OK: byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

