#!/usr/bin/env python3
"""Deploy the canonical Mneme engine + Hermes host adapter into a profile.

Source of truth:
  mneme.py                 — host-agnostic engine
  host/hermes/provider.py  — MemoryProvider (installed as plugins/mneme/__init__.py)
  host/hermes/runtime.py   — MnemeRuntime policy module
  host/hermes/plugin.yaml  — plugin metadata

Usage:
  python scripts/sync_hermes_bundle.py --profile %LOCALAPPDATA%/hermes/profiles/aletheon
  python scripts/sync_hermes_bundle.py --profile ... --check   # dry-run / verify only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "mneme.py"
HOST = ROOT / "host" / "hermes"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets(profile: Path) -> dict[str, tuple[Path, Path]]:
    plugin = profile / "plugins" / "mneme"
    return {
        "engine": (ENGINE, plugin / "mneme.py"),
        "provider": (HOST / "provider.py", plugin / "__init__.py"),
        "runtime": (HOST / "runtime.py", plugin / "runtime.py"),
        "plugin_yaml": (HOST / "plugin.yaml", plugin / "plugin.yaml"),
    }


def verify(profile: Path) -> list[str]:
    errors = []
    for name, (src, dst) in targets(profile).items():
        if not src.is_file():
            errors.append(f"missing source {name}: {src}")
            continue
        if not dst.is_file():
            errors.append(f"missing active {name}: {dst}")
            continue
        if digest(src) != digest(dst):
            errors.append(f"drift {name}: {dst}")
    return errors


def sync(profile: Path) -> None:
    plugin = profile / "plugins" / "mneme"
    plugin.mkdir(parents=True, exist_ok=True)
    for name, (src, dst) in targets(profile).items():
        if not src.is_file():
            raise FileNotFoundError(f"source missing for {name}: {src}")
        shutil.copy2(src, dst)
        print(f"  synced {name}: {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Hermes profile home (contains plugins/ and mneme/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify byte-identity; do not write",
    )
    args = parser.parse_args()
    profile = args.profile.expanduser().resolve()
    if not profile.is_dir():
        print(f"profile not found: {profile}", file=sys.stderr)
        return 2

    if args.check:
        problems = verify(profile)
        if problems:
            print("DRIFT:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(f"OK: hermes bundle byte-identical at {profile}")
        return 0

    print(f"Syncing Mneme hermes bundle -> {profile}")
    sync(profile)
    problems = verify(profile)
    if problems:
        print("POST-SYNC DRIFT (unexpected):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("OK: sync complete and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
