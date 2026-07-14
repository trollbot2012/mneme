#!/usr/bin/env python3
"""Print conflict candidates for explicit supersession review.

Detection never mutates canon. Resolve by writing a successor note with
supersedes=... (or mneme_add_note with scope + supersedes).

Usage:
  python scripts/list_conflicts.py --db PATH --notes PATH
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "mneme.py",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("mneme_conflicts", args.engine)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mem = mod.Mneme(args.db, args.notes)
    try:
        pairs = mem.find_conflicts(limit=args.limit)
        if args.json:
            print(json.dumps(pairs, indent=2))
            return 0
        print(f"{len(pairs)} conflict candidates (review, then supersede explicitly)\n")
        for c in pairs:
            print(
                f"  sim={c['similarity']:.2f} [{c['left']['kind']}] "
                f"{c['left']['title']!r}\n"
                f"           <-> {c['right']['title']!r}"
            )
        return 0
    finally:
        mem.close()


if __name__ == "__main__":
    raise SystemExit(main())
