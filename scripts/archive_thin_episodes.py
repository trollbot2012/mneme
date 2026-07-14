#!/usr/bin/env python3
"""Archive thin episode stubs into mneme_archive.db immediately.

Historical unclean exits left hundreds of checkpoint-only episodes. They are
safe to archive (not reconstruct). Selected stubs leave the live store now.

Usage:
  python scripts/archive_thin_episodes.py --db PATH --notes PATH --min-age-days 2
  python scripts/archive_thin_episodes.py ... --dry-run
  python scripts/archive_thin_episodes.py ... --include-invalidated
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def load_engine(engine_path: Path):
    spec = importlib.util.spec_from_file_location("mneme_hygiene", engine_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "mneme.py",
    )
    parser.add_argument("--min-age-days", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-invalidated",
        action="store_true",
        help="Also force-archive already-invalidated thin rows",
    )
    args = parser.parse_args()

    mod = load_engine(args.engine)
    mem = mod.Mneme(args.db, args.notes)
    cutoff = time.time() - args.min_age_days * 86400
    try:
        with mem._lock:
            if args.include_invalidated:
                # Age gate uses valid_at (creation). Soft-deleted stubs may have
                # invalid_at=now from a prior pass and must still be shippable.
                rows = mem._conn.execute(
                    "SELECT m.dedupe_key, m.body, m.tags, m.valid_at,"
                    " COALESCE(s.served, 0), m.invalid_at"
                    " FROM mem m LEFT JOIN mem_stats s ON s.note_id = m.note_id"
                    " WHERE m.kind='episode' AND m.valid_at < ?"
                    " ORDER BY m.valid_at ASC",
                    (cutoff,),
                ).fetchall()
            else:
                rows = mem._conn.execute(
                    "SELECT m.dedupe_key, m.body, m.tags, m.valid_at,"
                    " COALESCE(s.served, 0), m.invalid_at"
                    " FROM mem m LEFT JOIN mem_stats s ON s.note_id = m.note_id"
                    " WHERE m.kind='episode' AND m.invalid_at IS NULL"
                    " AND m.valid_at < ?"
                    " ORDER BY m.valid_at ASC",
                    (cutoff,),
                ).fetchall()

        doomed: list[str] = []
        for key, body, tags, _valid_at, _served, invalid_at in rows:
            text = body or ""
            tags_s = tags or ""
            quality = mod.Mneme.episode_quality_weight(text, tags_s)
            is_stub = (
                text.startswith("Checkpointed at first serve")
                or len(text.strip()) < 80
            )
            if invalid_at is not None:
                # Soft-deleted from a prior pass — ship thin stubs only.
                if not is_stub and quality > 0.25:
                    continue
            elif not is_stub or quality > 0.25:
                continue
            doomed.append(key)
            if len(doomed) >= args.limit:
                break

        print(json.dumps({
            "candidates": len(doomed),
            "min_age_days": args.min_age_days,
            "dry_run": args.dry_run,
            "include_invalidated": args.include_invalidated,
            "sample": doomed[:10],
        }, indent=2))
        if args.dry_run or not doomed:
            return 0

        # Stamp invalid_at far in the past so compact()'s age gate admits them,
        # then run compact to ship rows into mneme_archive.db and delete live.
        ancient = time.time() - (int(mem.cfg.get("compact_after_days", 30)) + 7) * 86400
        archive = args.db.with_name("mneme_archive.db")

        def _stamp() -> int:
            with mem._lock:
                for key in doomed:
                    mem._conn.execute(
                        "UPDATE mem SET invalid_at=? WHERE dedupe_key=?",
                        (ancient, key),
                    )
                    mem._conn.execute(
                        "DELETE FROM mem_fts WHERE dedupe_key=?", (key,))
                    mem._conn.execute(
                        "DELETE FROM mem_vec WHERE dedupe_key=?", (key,))
                mem._conn.commit()
            return len(doomed)

        stamped = mem._with_write_retry(_stamp)
        report = mem.compact(archive_path=archive)
        print(json.dumps({
            "stamped": stamped,
            "compact": report,
            "archive": str(archive),
        }, indent=2))
        return 0
    finally:
        mem.close()


if __name__ == "__main__":
    raise SystemExit(main())
