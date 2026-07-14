#!/usr/bin/env python3
"""Verify that Hermes runs the exact canonical Mneme engine + host adapter."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "active",
        type=Path,
        nargs="?",
        help="active Hermes plugin mneme.py (legacy single-file check)",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Hermes profile home — verifies engine + runtime + provider",
    )
    args = parser.parse_args()

    if args.profile:
        from sync_hermes_bundle import verify as verify_bundle

        problems = verify_bundle(args.profile.expanduser().resolve())
        if problems:
            print("DRIFT:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"OK: full hermes bundle byte-identical ({args.profile})")
        return 0

    if not args.active:
        print(
            "usage: verify_hermes_bundle.py <active-mneme.py>"
            " | --profile <HERMES_HOME>"
        )
        return 2

    canonical = ROOT / "mneme.py"
    active = args.active
    if not active.is_file():
        print(f"MISSING: {active}")
        return 2
    expected, actual = digest(canonical), digest(active)
    print(f"canonical {expected}  {canonical}")
    print(f"active    {actual}  {active.resolve()}")
    if expected != actual:
        print("DRIFT: active Hermes engine is not byte-identical")
        return 1
    print("OK: engine byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
